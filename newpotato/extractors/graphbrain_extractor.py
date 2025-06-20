import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import islice, product
from typing import Any, Dict, List, Optional, Set, Tuple

from graphbrain.hyperedge import Hyperedge, hedge
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

from newpotato.constants import NON_ATOM_WORDS, NON_WORD_ATOMS
from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graphbrain_parser import GraphbrainParserClient, GraphParse
from newpotato.modifications.oie_evaluation import (
    apply_curly_brackets,
    apply_variables,
    information_extraction,
)
from newpotato.modifications.pattern_ops import all_variables, contains_variable
from newpotato.modifications.patterns import match_pattern


def edge2toks(edge: Hyperedge, graph: Dict[str, Any]):
    """
    find IDs of tokens covered by an edge of a graph
    If some atom names match more than one token, candidate token sequences are disambiguated
    based on length and the shortest sequence (i.e. the one with the fewest gaps) is returned

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge to be mapped to token IDs
        graph (Dict[str, Any]): the Graphbrain Hypergraph of the full utterance

    Returns:
        Tuple[int, ...]: tuple of token IDs covered by the subedge
    """

    logging.debug(f"edge2toks\n{edge=}\n{graph=}")

    toks = set()
    strs_to_atoms = defaultdict(list)
    for atom, word in graph["atom2word"].items():
        strs_to_atoms[atom.to_str()].append(atom)

    to_disambiguate = []
    for atom in edge.all_atoms():
        atom_str = atom.to_str()
        if atom_str not in strs_to_atoms:
            assert (
                str(atom) in NON_WORD_ATOMS
            ), f"no token corresponding to {atom=} in {strs_to_atoms=}"
        else:
            cands = strs_to_atoms[atom_str]
            if len(cands) == 1:
                toks.add(graph["atom2word"][cands[0]][1])
            else:
                to_disambiguate.append([graph["atom2word"][cand][1] for cand in cands])

    # added solution for CPU memory issue with sorting (too many combinations)
    if len(to_disambiguate) > 0:
        logging.debug(f"edge2toks disambiguation needed: {toks=}, {to_disambiguate=}")
        # check number of combinations
        total_combinations = 1
        for lst in to_disambiguate:
            total_combinations *= len(lst)

        # skip computation if combinations exceed threshold
        COMBINATIONS_THRESHOLD = 10_000_000
        if total_combinations > COMBINATIONS_THRESHOLD:
            logging.error(f"too many combinations: {total_combinations}")
            logging.error("skipping")
            return set()

        hyp_sets = []
        for cand in product(*to_disambiguate):
            hyp_toks = sorted(toks | set(cand))
            hyp_length = hyp_toks[-1] - hyp_toks[0]
            hyp_sets.append((hyp_length, hyp_toks))

        shortest_hyp = sorted(hyp_sets)[0][1]
        logging.debug(f"{shortest_hyp=}")
        return set(shortest_hyp)

    return tuple(sorted(toks))


def matches2triplets(matches: List[Dict], graph: Dict[str, Any]) -> List[Triplet]:
    """
    convert graphbrain matches on a sentence to triplets of the tokens of the sentence

    Args:
        matches (List[Dict]): a list of hypergraphs corresponding to the matches
        graphs (Dict[str, Any]]): The hypergraph of the sentence

    Returns:
        List[Triplet] the list of triplets corresponding to the matches
    """
    triplets = []
    for triple_dict in matches:
        pred = []
        args = []
        for key, edge in triple_dict.items():
            if key == "REL":
                pred = edge2toks(edge, graph)
            else:
                args.append((int(key[-1]), edge2toks(edge, graph)))

        sorted_args = [arg[1] for arg in sorted(args)]
        triplets.append(Triplet(pred, sorted_args))

    return triplets


class UnmappableTripletError(Exception):
    pass


def _toks2subedge(
    edge: Hyperedge,
    toks_to_cover: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Tuple[Hyperedge, Set[str], Set]:
    """
    recursive helper function of toks2subedge

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (tuple): the tokens to be covered by the subedge
        all_toks (tuple): all tokens in the sentence
        words_to_i (dict): words mapped to token indices

    Returns:
        Hyperedge: the best matching subedge
        set: tokens covered by the matching hyperedge
        set: additional tokens in the matching hyperedge
    """
    words_to_cover = [all_toks[i] for i in toks_to_cover]
    logging.debug(f"_toks2subedge got: {edge=}, {words_to_cover=}")
    if edge.is_atom():
        lowered_word = edge.label().lower()
        # handle hyphenated tokens
        if lowered_word not in words_to_i:
            logging.debug(f"{lowered_word=}, {words_to_i=}")
            cands = [key for key in words_to_i if lowered_word in key]
            if len(cands) == 1:
                lowered_word = cands[0]
            else:
                logging.debug(
                    f"no token corresponding to edge label {lowered_word} and it is not listed as a non-word atom"
                )
                logging.debug(f"return empty sets")
                return edge, set(), set()

        toks = words_to_i[lowered_word]
        relevant_toks = toks & toks_to_cover
        if len(relevant_toks) > 0:
            return edge, relevant_toks, set()
        else:
            if lowered_word in NON_ATOM_WORDS:
                return edge, set(), set()
            else:
                return edge, set(), toks

    relevant_toks, irrelevant_toks = set(), set()
    subedges_to_keep = []
    for i, subedge in enumerate(edge):
        s_edge, subedge_relevant_toks, subedge_irrelevant_toks = _toks2subedge(
            subedge, toks_to_cover, all_toks, words_to_i
        )
        if subedge_relevant_toks == toks_to_cover:
            # a subedge covering everything, search can stop
            logging.debug(
                f"_toks2subedge: subedge covers everything, returning {s_edge=}, {subedge_relevant_toks=}, {subedge_irrelevant_toks=}"
            )
            return s_edge, subedge_relevant_toks, subedge_irrelevant_toks

        if len(subedge_relevant_toks) > 0 or (i == 0 and subedge.atom):
            # the first subedge must be kept, to connect the rest (unless it is not atomic)
            subedges_to_keep.append(s_edge)
            relevant_toks |= subedge_relevant_toks
            irrelevant_toks |= subedge_irrelevant_toks

    if len(relevant_toks) == 0:
        # no words covered
        logging.debug(
            f"_toks2subedge: no words covered, returning {edge=}, {relevant_toks=}, {irrelevant_toks=}"
        )
        return edge, relevant_toks, irrelevant_toks
    else:
        pruned_edge = hedge(subedges_to_keep)
        logging.debug(
            f"_toks2subedge: returning {pruned_edge=}, {relevant_toks=}, {irrelevant_toks=}"
        )
        return pruned_edge, relevant_toks, irrelevant_toks


def toks2subedge(
    edge: Hyperedge,
    toks: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Hyperedge:
    """
    find subedge in edge corresponding to the phrase in text.
    Based on graphbrain.learner.text2subedge, but keeps track of the set of words covered
    by partial results.
    If an exact match is not possible, returns the smallest subedge that covers all words.
    Raises ValueError if input edge does not contain all words.

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (set): the words to be covered by the subedge

    Returns:
        Hyperedge: the best matching hyperedge
        bool: whether the matching edge is exact (contains all the words and no other words)
    """
    try:
        toks_to_cover = {
            tok for tok in toks if all_toks[tok].lower() not in NON_ATOM_WORDS
        }
    except IndexError:
        logging.warning(f"some token IDs out of range: {toks=}, {all_toks=}")
        raise UnmappableTripletError()
    subedge, relevant_toks, irrelevant_toks = _toks2subedge(
        edge, toks_to_cover, all_toks, words_to_i
    )
    logging.debug(f"toks2subedge: {subedge=}, {relevant_toks=}, {irrelevant_toks=}")

    if toks_to_cover == relevant_toks:
        if len(irrelevant_toks) == 0:
            return subedge, relevant_toks, True
        logging.warning(f"returning incomplete match: {irrelevant_toks=}")
        return subedge, relevant_toks, False
    else:
        words = [all_toks[t] for t in toks_to_cover]
        logging.warning(
            f"hyperedge {edge} does not contain all words in {words}\n{toks_to_cover=}, {relevant_toks=}"
        )
        raise UnmappableTripletError()


def flatten_and_join_list(value):
    if isinstance(value, list):
        # special case: single element without parentheses
        if len(value) == 1:
            return flatten_and_join_list(value[0])
        return f"({' '.join(flatten_and_join_list(item) for item in value)})"
    return value  # return non-list value as-is


class GraphbrainMappedTriplet(Triplet):
    def __init__(
        self, mapped_pred, mapped_args, toks=None, variables=None, sen_graph=None
    ):
        super(GraphbrainMappedTriplet, self).__init__(mapped_pred, mapped_args)
        self.toks = toks
        self.variables = variables
        self.sen_graph = sen_graph
        self.mapped = True

    @staticmethod
    def from_json(data):
        if data["type"] == "triplet":
            print()
            return GraphbrainMappedTriplet(
                data["pred"], data["args"], variables=data["variables"]
            )
        else:
            raise ValueError(data["type"])

    def to_json(self):
        superclass_dict = super(GraphbrainMappedTriplet, self).to_json()
        superclass_dict["variables"] = self.variables
        return superclass_dict


class GraphbrainExtractor(Extractor):
    """A class to extract triplets from Semantic Hypergraphs.

    Attributes:
        parser_params (Dict): parameters to be used to initialize a TextParser object
    """

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
    ):
        super(GraphbrainExtractor, self).__init__()
        self.text_parser = GraphbrainParserClient(parser_url)
        self.spacy_vocab = self.text_parser.get_vocab()
        self.patterns = None

    @staticmethod
    def from_json(data: Dict[str, Any]):
        extractor = GraphbrainExtractor()
        extractor.text_parser.check_params(data["parser_params"])
        extractor.parsed_graphs = {
            text: GraphParse.from_json(graph_dict, extractor.spacy_vocab)
            for text, graph_dict in data["parsed_graphs"].items()
        }
        return extractor

    def to_json(self) -> Dict[str, Any]:
        data = {
            "extractor_type": "graphbrain",
            "parsed_graphs": {
                text: graph.to_json() for text, graph in self.parsed_graphs.items()
            },
            "parser_params": self.text_parser.get_params(),
        }

        return data

    def save_patterns(self, fn: str):
        assert self.patterns is not None, "no rules available"
        with open(fn, "w") as f:
            for line in self.patterns:
                f.write(f"{line}\n")

    def load_patterns(self, fn: str, N: int = 20):
        with open(fn, "r") as f:
            self.patterns = list(islice(f, N))

        if len(self.patterns) < N:
            raise ValueError(
                f"The file {fn} contains only {len(self.patterns)} patterns, but {N} were requested."
            )

    def _parse_text(self, text: str) -> List[GraphParse]:
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            Generator[Dict[str, Dict[str, Any]]]: The parsed graphs.
        """
        graphs = self.text_parser.parse(text)
        for graph in graphs:
            yield graph["text"], graph

    def is_trained(self):
        return self.patterns is not None

    def get_n_rules(self):
        if self.patterns is None:
            return 0
        return len(self.patterns)

    def get_rules(self, text_to_triplets=None, top_n=20) -> List[Tuple]:
        """
        Get the top N rules.
        """
        pc = self.extract_rules(text_to_triplets)
        self.patterns = [key for key, _ in pc.most_common(top_n)]
        return [(key, cnt) for key, cnt in pc.most_common(top_n)]

    def extract_rules(self, text_to_triplets=None) -> Counter:
        """
        Extract the rules from the annotated graphs.
        """
        assert text_to_triplets is not None, "annotated sentences missing"
        annotated_graphs = self.get_annotated_sentences(text_to_triplets)
        patterns = Counter()

        for hyperedge in annotated_graphs:
            logging.debug(f"{hyperedge=}")
            vars = all_variables(hyperedge)
            all_levels = get_levels(hyperedge, expand=False)
            for level in all_levels:
                logging.debug(f"{level=}")
                pattern = generalize_edge(level)
                if pattern is None:
                    logging.debug("skipping - no pattern")
                    continue
                atoms = pattern.atoms()
                roots = {atom.root() for atom in atoms if atom.root() != "*"}
                # skip patterns with only two variables, e.g. (REL/P.{p} ARG0/C)
                if len(roots) < 3:
                    logging.debug("skipping - not enough annotations")
                    continue
                logging.debug(f"{pattern=}")
                subpatterns = [subp for subp in pattern]
                subpatterns.append(pattern)
                for subp in subpatterns:
                    logging.debug(f"Pattern subedge: {subp}")
                    skip = check_pattern(subp, vars)
                    if not skip:
                        # uncomment the next line for patterns without subtypes, argroles and namespaces
                        # subp = subp.simplify(namespaces=False)
                        logging.debug(f"count: {subp}")
                        patterns[hedge(apply_curly_brackets(subp))] += 1

        return patterns

    def get_annotated_sentences(
        self,
        text_to_triplets: Dict[str, List[Triplet]],
    ):
        """
        Returns a set of annotated hyperedges.

        Args:
            parsed_graphs (Dict[str, Dict[str, Any]]): The parsed graphs.
            text_to_triplets (Dict[str, List[Tuple]]): The texts and corresponding triplets.
        """

        cases = []

        total, skipped = 0, 0
        for text, triplets in text_to_triplets.items():
            graph = self.parsed_graphs[text]
            main_edge = graph["main_edge"]
            for triplet, positive in triplets:
                total += 1
                variables = {
                    key: hedge(flatten_and_join_list(value))
                    for key, value in triplet.variables.items()
                }
                logging.debug(f"{main_edge=}")
                vedge = apply_variables(main_edge, variables)
                logging.debug(f"{vedge=}")
                if vedge is None:
                    logging.debug("failed to add annotations.")
                    logging.debug(f"{variables=}")
                    skipped += 1
                    continue
                else:
                    cases.append(vedge)

        print(f"{skipped=}, {total=}")
        return cases

    def classify(self, graph: Hyperedge) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Classify the graph.

        Args:
            graph (Hyperedge): The graph to classify.

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: The matches and the rules triggered.
        """
        matches = []
        rules_triggered = []

        try:
            for pattern in self.patterns:
                for match in match_pattern(graph, pattern):
                    if match == {}:
                        continue
                    else:
                        matches.append(match)
                        rules_triggered.append(pattern)
                        # TODO: eventually save rule ids triggered
            logging.debug(f"{self.patterns=}")
            logging.debug(f"{rules_triggered=}")
        except AttributeError as err:
            logging.error(f"Graphbrain matcher threw exception:\n{err}")
            matches, rules_triggered = [], []

        return matches, rules_triggered

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return [tok.text for tok in self.parsed_graphs[text]["spacy_sentence"]]

    def add_text_to_graphs(self, text: str) -> None:
        """Add the given text to the graphs.

        Args:
            text (str): The text to add to the graphs.

        Returns:
            None
        """
        self.get_graphs(text)

    def match_rules(self, text: str) -> List[Dict]:
        """
        match rules against sentence by passing the sentence's graph to the extractor

        Args:
            text (str): the sentence to be matched against

        Returns:
            List[Dict] a list of hypergraphs corresponding to the matches
        """
        all_matches = []
        for sen, graph in self.get_graphs(text).items():
            main_graph = graph["main_edge"]
            matches, _ = self.classify(main_graph)
            all_matches += matches
        return all_matches

    def extract_triplets_from_text(
        self, text: str, convert_to_text: bool = False
    ) -> Dict[str, Any]:
        """
        Extract the triplets from the given text with the Extractor.
        First the text is parsed into graphs, then the graphs are classified by the Extractor.

        Args:
            text (str): The text to extract triplets from.

        Returns:
            Dict[str, Any]: The matches and rules triggered. The matches are a list of dicts, where each dict is a triplet. The rules triggered are a list of strings, where each string is a rule.
        """

        graphs = self.get_graphs(text)
        matches_by_text = {
            graph["text"]: {"matches": [], "rules_triggered": [], "triplets": []}
            for graph in graphs
        }

        for graph in graphs:
            matches, rules_triggered = self.classify(graph["main_edge"])
            logging.info(f"matches: {matches}")
            triplets = matches2triplets(matches, graph)
            mapped_triplets = [
                self.map_triplet(triplet, graph["text"]) for triplet in triplets
            ]
            logging.info(f"triplets: {mapped_triplets}")

            if convert_to_text:
                matches = [
                    {k: v.label() for k, v in match.items()} for match in matches
                ]

            matches_by_text[graph["text"]]["matches"] = matches
            matches_by_text[graph["text"]]["rules_triggered"] = rules_triggered
            matches_by_text[graph["text"]]["triplets"] = mapped_triplets

        return matches_by_text

    def infer_triplets(self, sen: str) -> List[Triplet]:
        """
        match rules against sentence and return triplets corresponding to the matches

        Args:
            sen (str): the sentence to perform inference on

        Returns:
            List[Triple]: list of triplets inferred
        """
        assert self.patterns is not None, "no rules available"

        logging.debug(f'inferring triplets for: "{sen}"')
        graph = self.parsed_graphs[sen]
        logging.debug(f'graph: "{graph}"')
        matches = self.match_rules(sen)
        logging.debug(f'matches: "{matches}"')
        triplets = matches2triplets(matches, graph)
        logging.debug(f'triplets: "{triplets}"')

        # Error in matches2triplets > edge2toksstr(atom) in NON_WORD_ATOMS
        # AssertionError: no token corresponding to atom=+/B.aa/. in strs_to_atoms

        return triplets

    def map_to_subgraphs(self, triplet, sen_graph, strict=True):
        """
        helper function for map_triplet
        """
        # tok.text for tok in sen_graph["spacy_sentence"]
        # misses punctuation marks -> shifted indices
        all_toks = tuple(triplet.toks)
        words_to_i = defaultdict(set)
        for i, word in enumerate(all_toks):
            words_to_i[word.lower()].add(i)

        edge = sen_graph["main_edge"]
        variables = {}

        if triplet.pred is not None:
            rel_edge, relevant_toks, exact_match = toks2subedge(
                edge, triplet.pred, all_toks, words_to_i
            )
            logging.debug(f"{rel_edge=}, {relevant_toks=}, {exact_match=}")
            if not exact_match and strict:
                logging.debug(
                    f"cannot map pred {triplet.pred} to subedge of {edge} (closest: {rel_edge}"
                )
                raise UnmappableTripletError
            variables["REL"] = rel_edge
            mapped_pred = tuple(sorted(relevant_toks))
        else:
            mapped_pred = None

        mapped_args = []
        for i in range(len(triplet.args)):
            if triplet.args[i] is not None and len(triplet.args[i]) > 0:
                arg_edge, relevant_toks, exact_match = toks2subedge(
                    edge, triplet.args[i], all_toks, words_to_i
                )
                logging.debug(f"{arg_edge=}, {relevant_toks=}, {exact_match=}")
                if not exact_match and strict:
                    logging.debug(
                        f"cannot map arg {triplet.args[i]} to subedge of {edge} (closest: {arg_edge}"
                    )
                    raise UnmappableTripletError()
                variables[f"ARG{i}"] = arg_edge
                mapped_args.append(tuple(sorted(relevant_toks)))
            else:
                mapped_args.append(None)

        logging.debug(f"{mapped_pred=}, {mapped_args=}, {variables=}")
        return mapped_pred, mapped_args, variables

    def map_triplet(self, triplet, sentence, strict=True):
        """
        map predicate and arguments of a triplet (each a tuple of token indices) to
        corresponding subgraphs (Hyperedges). The mapping may change the indices, since words
        not showing up in the hypergraph (e.g. punctuation) are not to be considered part of the triplet
        """

        sen_graph = self.parsed_graphs[sentence]
        try:
            mapped_pred, mapped_args, variables = self.map_to_subgraphs(
                triplet, sen_graph, strict=strict
            )
        except UnmappableTripletError:
            logging.warning(
                f"could not map triplet ({triplet.pred=}, {triplet.args=} to {sen_graph=}"
            )
            return False
        else:
            toks = self.get_tokens(sentence)
            return GraphbrainMappedTriplet(
                mapped_pred, tuple(mapped_args), toks, variables, sen_graph
            )

    def generate_triplets_and_gold_data(self, input, max_extr, comb=True):
        start_time = time.time()
        goldfile = input.split(".")[0] + "_gold.json"
        extractions = {}
        latencies = []
        # skip gold data creation if gold data already available (and no parsing errors can occur)
        if os.path.exists(goldfile):
            with open(goldfile, "r") as gfile:
                gold = json.load(gfile)

                # loop through sentences
                for key in gold.keys():
                    for entry in gold[key]:
                        sent_start_time = time.time()
                        sentence, sen_idx = entry["sent"], entry["id"]
                        text_to_graph = self.get_graphs(sentence, doc_id=str(sen_idx))
                        graph = text_to_graph[sentence]["main_edge"]
                        atom2word = text_to_graph[sentence]["atom2word"]
                        logging.info("-" * 100)
                        logging.info(f"START of information extraction for {sen_idx=}:")
                        logging.info(f"{sen_idx=}, {sentence=}")
                        logging.info(f"{graph=}")
                        information_extraction(
                            extractions,
                            graph,
                            sen_idx,
                            atom2word,
                            self.patterns,
                            max_extr,
                        )
                        latency = time.time() - sent_start_time
                        latencies.append(latency)

        else:
            # create dictionary with correctly ordered and named keys
            name_mapping = {
                "A0": "arg1",
                "P": "rel",
                "A1": "arg2",
                "A2": "arg3",
                "A3": "arg4",
                "A4": "arg5",
                "A5": "arg6",
                "A6": "arg7",
            }

            with open(input) as stream:
                total, skipped = 0, 0
                last_sent = ""
                sent_cnt = 0
                rm_sen_idx = set()
                gold_list = list()
                no_obj_cnt = 0
                start_time = time.time()
                for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
                    total += 1
                    words = [t[1] for t in sen]
                    sentence = " ".join(words)

                    # prediction: infer triplets for new sentences

                    # case 1: known sentence
                    # save id for gold data creation -> add triplet to previous sentence ID
                    # avoid double parsing/new entry in extractions for same sentence
                    # remark: sen_idx skipped -> from 0 to xy with gaps
                    if sentence == last_sent:
                        sen_idx = last_id
                    # case 2: new sentence
                    else:
                        last_sent = sentence
                        last_id = sen_idx
                        sent_cnt += 1
                        skip = False

                        # Step 1: add gold_dict to gold_list for previous sentence (but skip the very first)
                        if total > 1:
                            gold_dict = combine_args(gold_dict)
                            gold_list.append(gold_dict)

                        # Step 2: create new gold_dict for new sentence
                        gold_dict = {
                            "id": str(sen_idx),
                            "sent": sentence,
                            "tokens": words,
                            "tuples": list(),
                        }

                        # Step 3: text parsing for new sentence (only if needed)
                        text_to_graph = self.get_graphs(sentence, doc_id=str(sen_idx))
                        if len(text_to_graph) > 1:
                            logging.error(f"sentence split into two: {words}")
                            logging.error(f"{sen_idx=}, {text_to_graph=}")
                            logging.error("skipping")
                            skipped += 1
                            skip = True
                            logging.error(f"{skipped=}, {total=}")
                            # remove gold triplets for sentence as well
                            rm_sen_idx.add(str(sen_idx))

                        logging.debug(f"{sentence=}, {text_to_graph=}")

                        # Step 4: generating triplets after conjunction decomposition (information_extraction)
                        if not skip:
                            sent_start_time = time.time()
                            try:
                                # rare problems with sentence modifications need this check
                                graph = text_to_graph[sentence]["main_edge"]
                            except KeyError:
                                logging.error(f"sentence not found after parsing")
                                logging.error(f"{sen_idx=}, {sentence=}")
                                logging.error("skipping")
                                skipped += 1
                                logging.error(f"{skipped=}, {total=}")
                                # remove gold triplets for sentence as well
                                rm_sen_idx.add(str(sen_idx))
                            else:
                                atom2word = text_to_graph[sentence]["atom2word"]
                                logging.info("-" * 100)
                                logging.info(
                                    f"START of information extraction for {sen_idx=}:"
                                )
                                logging.info(f"{sen_idx=}, {sentence=}")
                                logging.info(f"{graph=}")
                                information_extraction(
                                    extractions,
                                    graph,
                                    sen_idx,
                                    atom2word,
                                    self.patterns,
                                    max_extr,
                                )
                            latency = time.time() - sent_start_time
                            latencies.append(latency)

                    # Step 5: gold data creation
                    # add multiple triplets to gold_dict for one sentence
                    temp = {k: defaultdict(list) for k in name_mapping.keys()}
                    for i, tok in enumerate(sen):
                        label = tok[7].split("-")[0]
                        if label == "O":
                            continue
                        # uncomment next elif clause if you want to ignore arg3+ annotations
                        # elif label in ["A2", "A3", "A4", "A5", "A6"]:
                        #     continue
                        temp[label]["words"].append(tok[1])
                        temp[label]["words_indexes"].append(i)

                    # check if annotation misses objects (A1 and higher)
                    if not temp["A1"]:
                        logging.debug(f"skipping gold triplet without objects")
                        logging.debug(f"{sen_idx=}, {temp=}")
                        rm_sen_idx.add(str(sen_idx))
                        no_obj_cnt += 1
                        continue

                    args_dict = {
                        name_mapping.get(key, key): value
                        for key, value in temp.items()
                        if value
                    }
                    logging.debug(f"{sentence=}, {args_dict=}")
                    gold_dict["tuples"].append(args_dict)

                # add gold_dict to gold_list for very last sentence
                gold_dict = combine_args(gold_dict)
                gold_list.append(gold_dict)

                # remove sentences
                gold_list[:] = [d for d in gold_list if d["id"] not in rm_sen_idx]

                # save gold dictionary to json file
                save_dict = {input: gold_list}
                with open(goldfile, "w", encoding="utf-8") as f:
                    json.dump(save_dict, f, ensure_ascii=False, indent=4)

                print(f"Original sentence count: {sent_cnt}")
                print(f"Removed sentence count (due to parsing errors): {skipped}")
                print(f"Removed sentence count (total): {len(rm_sen_idx)}")
                print(f"Triplets without object count: {no_obj_cnt}")

        # first extraction time measurement
        extr_time = time.time() - start_time

        filtered_extr = {}
        filtered_extr_cnt = 0
        extr_cnt = 0
        # comb = False
        if comb:
            # take biggest set of overlapping matches per sentence
            for k, v in extractions.items():
                extr_cnt += len(v)
                newv = combine_triplets(v)
                filtered_extr[k] = newv
                filtered_extr_cnt += len(newv)
        else:
            for k, v in extractions.items():
                extr_cnt += len(v)
                filtered_extr[k] = v
            filtered_extr_cnt = extr_cnt

        filtered_extr_time = time.time() - start_time

        # save predictions to json file
        output = input.split(".")[0] + "_pred.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(filtered_extr, f, ensure_ascii=False, indent=4)

        # efficiency metrics
        raw_triplets_per_second = extr_cnt / extr_time
        evaluated_triplets_per_second = filtered_extr_cnt / filtered_extr_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return raw_triplets_per_second, evaluated_triplets_per_second, avg_latency


# function to combine arg3-arg6 to "arg3+" and print gold tuples in the console
def combine_args(gold_dict):
    keys_to_combine = ["arg3", "arg4", "arg5", "arg6", "arg7"]
    tuples_list = []
    for tup in gold_dict["tuples"]:
        arg3to6 = []
        for key in keys_to_combine:
            if key in tup:
                arg3to6.append(tup[key])
                # remove original key
                del tup[key]
        # add arg2 if missing
        if "arg2" not in tup:
            tup["arg2"] = {"words": [], "words_indexes": []}
        # add combined keys or empty list
        tup["arg3+"] = arg3to6
        tuples_list.append(tup)

    # Add to the list under a new key
    gold_dict["tuples"] = tuples_list
    return gold_dict


# function to check similarity between two strings (based on character similarity)
def is_similar(s1, s2, threshold=0.5):
    return SequenceMatcher(None, s1, s2).ratio() >= threshold


# function to process each key in the dictionary
# arg2 is combined from triplets sharing the same (arg1, rel)
# two options: EITHER combine arg2 as a merged arg2 OR add second arg2 as arg3 (if not existent)
def combine_triplets(entries):
    grouped = {}

    for entry in entries:
        group_key = (entry["arg1"], entry["rel"])

        if group_key not in grouped:
            # initialize grouped entry
            grouped[group_key] = {
                k: v for k, v in entry.items() if k not in ["arg2", "arg3+"]
            }
            grouped[group_key]["arg2"] = entry["arg2"]
            grouped[group_key]["arg3+"] = entry.get("arg3+", [])
        else:
            current_arg2 = grouped[group_key]["arg2"]

            # check similarity for arg2 to existing arg2 and arg3+
            if not is_similar(current_arg2, entry["arg2"]):
                if not any(
                    is_similar(entry["arg2"], existing)
                    for existing in grouped[group_key]["arg3+"]
                ):
                    grouped[group_key]["arg3+"].append(entry["arg2"])

            # check similarity for arg3+ to existing arg2 and arg3+
            if "arg3+" in entry:
                for new_arg in entry["arg3+"]:
                    if not is_similar(new_arg, grouped[group_key]["arg2"]) and not any(
                        is_similar(new_arg, existing)
                        for existing in grouped[group_key]["arg3+"]
                    ):
                        grouped[group_key]["arg3+"].append(new_arg)

    # remove arg3+ if empty
    for _, val in grouped.items():
        if not val["arg3+"]:
            del val["arg3+"]

    return list(grouped.values())


def extract_elements(hyperedge) -> List[str]:
    """
    Extracts all elements from a hyperedge,
    including atoms (non-parenthesized elements)
    """
    hyperedge = str(hyperedge)
    # remove outer parentheses if present
    if hyperedge.startswith("(") and hyperedge.endswith(")"):
        hyperedge = hyperedge[1:-1]

    elements = []
    stack = []
    start_idx = 0
    i = 0
    while i < len(hyperedge):
        char = hyperedge[i]

        if char == "(":
            if not stack:
                start_idx = i  # start of a top-level element
            stack.append(char)
        elif char == ")":
            stack.pop()
            if not stack:  # end of a top-level element
                element = hyperedge[start_idx : i + 1]
                elements.append(element)
        elif not stack and char not in " ()":
            # detecting atoms (words not inside parentheses)
            match = re.match(r"[^\s()]+", hyperedge[i:])
            if match:
                elements.append(match.group(0))
                # move forward to skip the matched atom
                i += len(match.group(0)) - 1

        i += 1
    return elements


def expand_subedges(top_level_elements: List[str]) -> List[List[str]]:
    """
    Extracts the next level of elements
    but keeps 'var' objects as whole
    """
    elements = []
    for element in top_level_elements:
        # keep "var" objects intact
        if element.startswith("(var "):
            elements.append(element)
        else:
            # check for nested components
            nested_elements = extract_elements(element)
            # only unnest multiple components
            if len(nested_elements) > 1:
                elements.append(nested_elements)
            else:
                elements.append(element)

    return elements


def get_levels(hyperedge, expand=False) -> List:
    top_level_elements = extract_elements(hyperedge)
    logging.debug(f"{top_level_elements=}")

    # only consider relations of sizes 3 or 4
    if len(top_level_elements) > 4:
        logging.debug("skipping - too many relations")
        return []
    if len(top_level_elements) < 3:
        logging.debug("skipping - not enough relations")
        return []

    # only top level patterns are returned
    if not expand:
        return [top_level_elements]

    expanded_subedges = expand_subedges(top_level_elements)
    logging.debug(f"{expanded_subedges=}")
    if top_level_elements == expanded_subedges:
        return [top_level_elements]
    else:
        return [top_level_elements, expanded_subedges]


def edge2pattern(edge, root=False, subtype=False):
    edge = hedge(edge)
    if root and edge.atom:
        root_str = edge.root()
    elif contains_variable(edge):
        var_matches = all_variables(edge)
        if len(var_matches) > 1:
            # at least two variables in second level hyperedge - does not meet conditions
            logging.debug("skipping - multiple vars")
            return None
        root_str = next(iter(var_matches))
    else:
        root_str = "*"
    if subtype:
        et = edge.type()
    else:
        et = edge.mtype()
    pattern = "{}/{}".format(root_str, et)
    ar = edge.argroles()
    if ar == "" or et in ["C", "R", "S"]:
        return hedge(pattern)
    else:
        return hedge("{}.{}".format(pattern, ar))


# function to generalise each hyperedge with functional patterns
# generalisation approach: using var as root and adding main type (mtype) of edge after /
# only consider recursive expansion to depth 2 for simplicity
def generalize_edge(hlist):
    final_result = ["("]
    for item in hlist:
        if type(item) == list:
            final_result.append("(")
            for i in item:
                newitem = edge2pattern(i)
                if newitem is None:
                    return None
                final_result.append(newitem.to_str())
            final_result.append(")")
        else:
            newitem = edge2pattern(item)
            if newitem is None:
                return None
            final_result.append(newitem.to_str())

    final_result.append(")")
    final_result = " ".join(final_result)
    return hedge(final_result)


def check_pattern(pattern, vars):
    atoms = pattern.atoms()
    roots = {atom.root() for atom in atoms}
    for var in vars:
        # make sure to include all variables in the final pattern
        if str(var) in roots:
            continue
        else:
            logging.debug("skipping pattern - var missing")
            logging.debug(f"{vars.keys()=}")
            return True

    return False
