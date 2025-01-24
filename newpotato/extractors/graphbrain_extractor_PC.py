import itertools
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from graphbrain.hyperedge import Hyperedge, hedge
from graphbrain.learner.classifier import Classifier, apply_curly_brackets
from graphbrain.learner.classifier import from_json as classifier_from_json
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

from newpotato.constants import NON_ATOM_WORDS, NON_WORD_ATOMS
from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graphbrain_parser import GraphbrainParserClient, GraphParse
from newpotato.modifications.oie_patterns import information_extraction
from newpotato.modifications.pattern_ops import (
    all_variables,
    apply_variable,
    apply_variables,
    contains_variable,
)
from newpotato.modifications.patterns import _matches_atomic_pattern, match_pattern

# def evaluate_combination(args):
#     toks, candidate = args
#     hyp_toks = toks | set(candidate)
#     return max(hyp_toks) - min(hyp_toks), sorted(hyp_toks)


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
            # TODO: count skipped cases and log them for further investigation -> not easy
            return set()

        hyp_sets = []
        for cand in itertools.product(*to_disambiguate):
            hyp_toks = sorted(toks | set(cand))
            hyp_length = hyp_toks[-1] - hyp_toks[0]
            hyp_sets.append((hyp_length, hyp_toks))

        shortest_hyp = sorted(hyp_sets)[0][1]
        logging.debug(f"{shortest_hyp=}")
        return set(shortest_hyp)

    return tuple(sorted(toks))


# # possible ideas for optimization
# import heapq
# # special case: toks is empty set -> problems with min, max
# if not toks:
#     # choose smallest token from each group in `to_disambiguate`
#     smallest_tokens = {min(cand) for cand in to_disambiguate}
#     logging.debug(
#         f"Disambiguation resolved by choosing smallest tokens: {smallest_tokens}"
#     )
#     return tuple(sorted(smallest_tokens))

# # Step 1: Filter to_disambiguate (top-k candidates)
# k = 3
# filtered_disambiguate = []
# for lst in to_disambiguate:
#     filtered_lst = sorted(lst, key=lambda x: min(abs(x - t) for t in toks))[:k]
#     filtered_disambiguate.append(filtered_lst)

# # pre-filter candidates
# toks_min, toks_max = min(toks), max(toks)
# max_distance = 5  # adjust as needed based on your data characteristics
# filtered_disambiguate = [
#     [
#         x
#         for x in sublist
#         if toks_min - max_distance <= x <= toks_max + max_distance
#     ]
#     for sublist in to_disambiguate
# ]

# # Set up a limited heap
# max_heap_size = 5_000  # Limit heap size for performance
# heap = []

# for cand in itertools.product(*filtered_disambiguate):
#     hyp_toks = toks | set(cand)
#     hyp_length = max(hyp_toks) - min(hyp_toks)

#     # Add to the heap and maintain its size
#     if len(heap) < max_heap_size:
#         heapq.heappush(heap, (hyp_length, sorted(hyp_toks)))
#     else:
#         heapq.heappushpop(heap, (hyp_length, sorted(hyp_toks)))
# shortest_hyp = min(heap, key=lambda x: x[0])[1]

# # Step 2: Parallelize combination evaluation -> makes process very slow
# with Pool(processes=8) as pool:  # Adjust processes as needed
#     results = pool.map(
#         evaluate_combination,
#         [(toks, cand) for cand in itertools.product(*filtered_disambiguate)],
#     )
# # Step 3: Find the shortest hypothesis
# shortest_hyp = min(results)[1]

# logging.debug(f"{shortest_hyp=}")
# return tuple(shortest_hyp)  # TODO: tuple or set


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
) -> Tuple[Hyperedge, Set[str]]:
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
        if lowered_word not in words_to_i:
            assert (
                lowered_word in NON_WORD_ATOMS
            ), f"no token corresponding to edge label {lowered_word} and it is not listed as a non-word atom"

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
        classifier (Optional[Classifier]): The classifier to use for extraction.
        parser_params (Dict): parameters to be used to initialize a TextParser object
    """

    def __init__(
        self,
        classifier: Optional[Classifier] = None,
        parser_url: Optional[str] = "http://localhost:7277",
    ):
        super(GraphbrainExtractor, self).__init__()
        self.classifier = classifier
        self.text_parser = GraphbrainParserClient(parser_url)
        self.spacy_vocab = self.text_parser.get_vocab()
        self.patterns = None

    @staticmethod
    def from_json(data: Dict[str, Any]):
        extractor = GraphbrainExtractor()
        extractor.text_parser.check_params(data["parser_params"])
        if data["classifier"] is not None:
            extractor.classifier = classifier_from_json(data["classifier"])

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
            "classifier": None,
        }
        if self.classifier is not None:
            data["classifier"] = self.classifier.to_json()

        return data

    def save_patterns(self, fn: str):
        assert self.patterns is not None, "no rules available"
        with open(fn, "w") as f:
            for line in self.patterns:
                f.write(f"{line}\n")

    def load_patterns(self, fn: str):
        with open(fn, "r") as f:
            # self.patterns = f.readlines()
            self.patterns = f.read().splitlines()
        # with open(fn, "r") as f:
        #     self.patterns = json.load(f)

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
            yield graph["text"], graph  # Generator[Tuple[str, GraphParse]]

    def is_trained(self):
        return self.patterns is not None

    def get_n_rules(self):
        if self.patterns is None:
            return 0
        return len(self.patterns)

    def get_rules(self, text_to_triplets=None, top_n=20) -> List[str]:
        """
        Get the top N rules.
        """
        pc = self.extract_rules(text_to_triplets)
        self.patterns = [key for key, _ in pc.most_common(top_n)]
        return [key for key, _ in pc.most_common(top_n)]

    def extract_rules(self, text_to_triplets=None, check_vars=True) -> Counter:
        """
        Extract the rules from the annotated graphs.
        """
        assert text_to_triplets is not None, "annotated sentences missing"
        annotated_graphs = self.add_cases(text_to_triplets)
        patterns = Counter()

        # TODO: conjunction decomposition, include special builder
        # TODO: for hyperedge, positive in self.get_annotated_graphs(): -> what is more correct?
        for hyperedge in annotated_graphs:
            hyperedge = hedge(hyperedge)
            # print(f"{hyperedge=}")
            vars = all_variables(hyperedge)
            # print(f"{vars.keys()=}")
            if hyperedge.not_atom:
                # edges = conjunctions_decomposition(hyperedge, concepts=True)
                # print(f"{edges=}")
                # for edge in edges:
                # vars2 = all_variables(edge)
                # if len(vars) == len(vars2):
                pattern = generalise_edge(hyperedge)
                # print(f"{pattern=}")
                if pattern is None:
                    # print("skipped - no pattern")
                    continue
                # TODO: eliminate this elif - should always be true
                elif check_vars:
                    skip = False
                    atoms = pattern.atoms()
                    roots = {atom.root() for atom in atoms}
                    for var in vars:
                        # make sure to include all variables in the final pattern
                        if str(var) in roots:
                            continue
                        else:
                            # print("skipped - var missing")
                            skip = True
                            break
                    if not skip:
                        # if not positive:
                        #     print(f"{hyperedge=}")
                        # TODO: patterns without REL are counted - why?
                        print("count: ", pattern)
                        patterns[hedge(apply_curly_brackets(pattern))] += 1
                else:
                    # TODO: do i want this else clause?
                    print("count: ", pattern)
                    patterns[hedge(apply_curly_brackets(pattern))] += 1

        return patterns

    def add_cases(
        self,
        text_to_triplets: Dict[str, List[Triplet]],
    ):
        """
        Add cases to the classifier.

        Args:
            parsed_graphs (Dict[str, Dict[str, Any]]): The parsed graphs.
            text_to_triplets (Dict[str, List[Tuple]]): The texts and corresponding triplets.
        """

        cases = []

        for text, triplets in text_to_triplets.items():
            # print(f"{text=}")
            graph = self.parsed_graphs[text]
            main_edge = graph["main_edge"]
            for triplet, positive in triplets:
                # print(f"{triplet.variables=}")
                variables = {
                    key: hedge(flatten_and_join_list(value))
                    for key, value in triplet.variables.items()
                }
                # print(f"{variables=}")
                vedge = apply_variables(main_edge, variables)
                # print(f"{vedge=}")
                if vedge is None:
                    logging.debug("failed to add case.")
                    continue
                else:
                    cases.append((vedge, positive))

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

        # logging.info(f"classifier matches: {matches}")
        # logging.info(f"classifier rules triggered: {rules_triggered}")

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
        all_toks = tuple(tok.text for tok in sen_graph["spacy_sentence"])
        words_to_i = defaultdict(set)
        for i, word in enumerate(all_toks):
            words_to_i[word.lower()].add(i)

        edge = sen_graph["main_edge"]
        variables = {}

        if triplet.pred is not None:
            rel_edge, relevant_toks, exact_match = toks2subedge(
                edge, triplet.pred, all_toks, words_to_i
            )
            if not exact_match and strict:
                logging.warning(
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
                if not exact_match and strict:
                    logging.warning(
                        f"cannot map arg {triplet.args[i]} to subedge of {edge} (closest: {arg_edge}"
                    )
                    raise UnmappableTripletError()
                variables[f"ARG{i}"] = arg_edge
                mapped_args.append(tuple(sorted(relevant_toks)))
            else:
                mapped_args.append(None)

        return mapped_pred, mapped_args, variables

    def map_triplet(self, triplet, sentence, strict=False):
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

    def temporary_triplets_creation(self, input, max_items=999999999):
        # create dictionary with correctly ordered and named keys
        name_mapping = {
            "A0": "arg1",
            "P": "rel",
            "A1": "arg2",
            "A2": "arg3",
            "A3": "arg4",
            "A4": "arg5",
            "A5": "arg6",
        }

        with open(input) as stream:
            total, skipped = 0, 0
            last_sent = ""
            sent_cnt = 0
            gold_list = list()
            extractions = {}
            for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
                # early break condition
                if total == max_items:
                    break
                total += 1
                words = [t[1] for t in sen]
                sentence = " ".join(words)

                # prediction: infer triplets for new sentences
                if sentence == last_sent:
                    # avoid double parsing/new entry in extractions for same sentence
                    # but save id for gold data creation -> add triplet to previous sentence ID
                    # remark: sen_idx skipped -> from 0 to xy with gaps
                    sen_idx = last_id
                else:
                    last_sent = sentence
                    last_id = sen_idx
                    sent_cnt += 1
                    skip = False
                    # text parsing (only if needed)
                    text_to_graph = self.get_graphs(sentence)
                    # TODO: how to handle skipped cases
                    # quick solution for now: skip gold data creation as well
                    if len(text_to_graph) > 1:
                        logging.error(f"sentence split into two: {words}")
                        logging.error(f"{text_to_graph=}")
                        logging.error("skipping")
                        skipped += 1
                        skip = True
                        logging.error(f"{skipped=}, {total=}")

                    # TODO: investigate why this try/except is needed
                    try:
                        graph = text_to_graph[sentence]["main_edge"]
                    except:
                        logging.error(f"{text_to_graph=}")
                        logging.error("skipping")
                        skipped += 1
                        skip = True
                        logging.error(f"{skipped=}, {total=}")
                    logging.debug(f"{sentence=}, {graph=}")

                    # TODO: check if necessary
                    # generating triplets after conjunction decomposition (inside information_extraction())
                    if graph and not skip:
                        atom2word = text_to_graph[sentence]["atom2word"]
                        # print(f"{sentence=}, {graph=}, {atom2word=}")
                        information_extraction(  # in oie_patterns.py
                            extractions, graph, sen_idx, atom2word, self.patterns
                        )

                    # add gold_dict to gold_list for last sentence (but skip the very first)
                    if total > 1:
                        # combine arg3-arg6 to "arg3+"
                        keys_to_combine = ["arg3", "arg4", "arg5", "arg6"]
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
                        gold_list.append(gold_dict)

                    # create new gold_dict for new sentence
                    gold_dict = {
                        "id": str(sen_idx),
                        "sent": sentence,
                        "tokens": words,
                        "tuples": list(),
                    }

                # gold data creation: add multiple triplets to gold_dict for one sentence
                temp = {k: defaultdict(list) for k in name_mapping.keys()}
                for i, tok in enumerate(sen):
                    # 'text': '(The second)/(The second danger to a year of relatively healthy global economic growth)',
                    # 'words': ['The', 'second'],
                    # 'words_indexes': [0, 1],
                    label = tok[7].split("-")[0]
                    if label == "O":
                        continue
                    temp[label]["words"].append(tok[1])
                    temp[label]["words_indexes"].append(i)

                # TODO: add "text" entry with joining "words" list to string
                # for key in temp.keys():
                #     joined_text = " ".join(temp[key]["words"])
                #     temp[key]["text"] = joined_text

                args_dict = {
                    name_mapping.get(key, key): value
                    for key, value in temp.items()
                    if value
                }
                logging.debug(f"{sentence=}, {args_dict=}")

                gold_dict["tuples"].append(args_dict)

            logging.info(
                f"{skipped/sent_cnt:.2%} of annotated sentences skipped (parsing problems)"
            )
            # save gold dictionary to json file
            save_dict = {input: gold_list}
            output = input.split(".")[0] + "_gold.json"
            with open(output, "w", encoding="utf-8") as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=4)

            # save predictions to json file
            output = input.split(".")[0] + "_pred.json"
            with open(output, "w", encoding="utf-8") as f:
                json.dump(extractions, f, ensure_ascii=False, indent=4)


def extract_top_level_elements(hyperedge):
    elements = []
    stack = []
    start_idx = 0

    str_edge = str(hyperedge)

    # remove outermost parentheses
    if str_edge.startswith("(") and str_edge.endswith(")"):
        str_edge = str_edge[1:-1]

    for idx, char in enumerate(str_edge):
        if char == "(":
            if not stack:
                start_idx = idx  # start of a top-level element
            stack.append(char)
        elif char == ")":
            stack.pop()
            if not stack:  # end of a top-level element
                element = str_edge[start_idx : idx + 1]
                elements.append(element)

    return elements


def extract_second_level(hyperedge):
    top_level_elements = extract_top_level_elements(hyperedge)
    second_level = []
    # only consider relations of sizes 3 or 4
    if len(top_level_elements) > 4:
        # print("too many relations")
        return second_level
    # TODO: what about smaller relations
    # if len(top_level_elements) < 3:
    #     print("not enough relations")
    #     print(top_level_elements)
    #     return second_level

    for element in top_level_elements:
        # keep "var" objects intact
        if element.startswith("(var"):
            second_level.append(element)
        else:
            # check for nested components
            nested_elements = extract_top_level_elements(element)
            # only unnest multiple components
            if len(nested_elements) > 1:
                second_level.append(nested_elements)
            else:
                second_level.append(element)

    return second_level


def edge2pattern(edge, root=False, subtype=False):
    # print("edge2pattern-e: ", edge)
    if root and edge.atom:
        root_str = edge.root()
    elif contains_variable(edge):
        # print("contains var")
        # count the number of unique variables
        var_cnt = len(re.findall(r"\bvar\b", str(edge)))
        if var_cnt > 1:
            # print("multiple vars")
            # print("edge2pattern-e: ", edge)
            # at least two variables in second level hyperedge - does not meet conditions
            return None
        elif edge.contains("REL", deep=True):
            root_str = "REL"
        elif edge.contains("ARG0", deep=True):
            root_str = "ARG0"
        elif edge.contains("ARG1", deep=True):
            root_str = "ARG1"
        elif edge.contains("ARG2", deep=True):
            root_str = "ARG2"
        elif edge.contains("ARG3", deep=True):
            root_str = "ARG3"
        elif edge.contains("ARG4", deep=True):
            root_str = "ARG4"
        elif edge.contains("ARG5", deep=True):
            root_str = "ARG5"
        else:
            # print("other problem")
            root_str = "*"
    else:
        root_str = "*"
    if subtype:
        et = edge.type()
    else:
        et = edge.mtype()
    pattern = "{}/{}".format(root_str, et)
    ar = edge.argroles()
    if ar == "":
        # print("edge2pattern-p: ", hedge(pattern))
        return hedge(pattern)
    else:
        # print("edge2pattern-p: ", hedge("{}.{}".format(pattern, ar)))
        return hedge("{}.{}".format(pattern, ar))


# function to generalise each hyperedge with functional patterns
# generalisation approach: using var as root and adding main type (mtype) of edge after /
# only consider recursive expansion to depth 2 for simplicity
def generalise_edge(hyperedge):
    second_level_result = extract_second_level(hyperedge)
    # print(f"{second_level_result=}")
    if len(second_level_result) == 0:
        return None
    final_result = []
    for item in second_level_result:
        if type(item) == list:
            for i in item:
                # print(f"{i=}")
                newitem = edge2pattern(hedge(i))
                # print(f"{newitem=}")
                if newitem is not None:
                    final_result.append(str(newitem))
        else:
            # print(f"{item=}")
            newitem = edge2pattern(hedge(item))
            # print(f"{newitem=}")
            if newitem is not None:
                final_result.append(str(newitem))

    if None not in final_result:
        return hedge(" ".join(final_result))
    else:
        return None
