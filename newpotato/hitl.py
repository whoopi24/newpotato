import json
import logging
import os
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import stanza
from graphbrain import hgraph
from graphbrain.hyperedge import Atom, Hyperedge
from graphbrain.learner.classifier import from_json as classifier_from_json
from graphbrain.learner.rule import Rule
from graphbrain.parsers import create_parser

# from graphbrain.patterns import PatternCounter
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate
from tuw_nlp.text.utils import gen_tsv_sens

# from graphbrain.hypergraph import Hypergraph
# from graphbrain.learner.classifier import Classifier
from newpotato.classifier import Classifier
from newpotato.datatypes import GraphParse, Triplet
from newpotato.oie_patterns import *

# from newpotato.entrypoints import edge_matches_pattern, match_pattern
# from newpotato.matcher import Matcher
# from newpotato.counter import PatternCounter
from newpotato.parser import TextParserClient

# from graphbrain.learner.pattern_ops import *
from newpotato.pattern_ops import *
from newpotato.patterns import PatternCounter
from newpotato.utils import matches2triplets


class AnnotatedWordsNotFoundError(Exception):
    def __init__(self, words_txt, pattern, sen):
        message = (
            f'Words "{words_txt}" (pattern: "{pattern}") not found in sentence "{sen}"'
        )
        super().__init__(message)

        self.words_txt = words_txt
        self.sen = sen
        self.pattern = pattern


@dataclass
class Extractor:
    """A class to extract triplets from graphs, texts, and annotated graphs.

    Attributes:
        classifier (Optional[Classifier]): The classifier to use for extraction.
    """

    classifier: Optional[Classifier] = field(default=None)

    @staticmethod
    def from_json(classifier_data: Dict[str, Any]):
        extractor = Extractor()
        if classifier_data is not None:
            extractor.classifier = classifier_from_json(classifier_data)
        return extractor

    def to_json(self) -> Dict[str, List]:
        if self.classifier is None:
            return None
        return self.classifier.to_json()

    def get_rules(self) -> List[Rule]:
        """
        Get the rules.
        """
        if self.classifier is None:
            return []
        return [rule.pattern for rule in self.classifier.rules]

    def extract_rules(self, learn: bool = False):
        """
        Extract the rules from the annotated graphs.
        """
        assert self.classifier is not None, "classifier not initialized"
        if learn:
            print("learn is true")
            self.classifier.learn()
        else:
            print("learn is false")
            start = time.time()
            self.classifier.extract_patterns()
            t = time.time() - start
            print("extract patterns done; duration: ", t)

            start = time.time()
            self.classifier._index_rules()
            t = time.time() - start
            print("indexing done; duration: ", t)

    def get_annotated_graphs_from_classifier(self) -> List[str]:
        """
        Get the annotated graphs

        Returns:
            List[str]: The annotated graphs. An annotated graph is a hyperedge that has been annotated with variables. e.g. "REL(ARG1, ARG2)"
        """
        assert self.classifier is not None, "classifier not initialized"
        return [str(rule[0]) for rule in self.classifier.cases]

    def add_cases(
        self,
        parsed_graphs: Dict[str, Dict[str, Any]],
        text_to_triplets: Dict[str, List[Triplet]],
    ):
        """
        Add cases to the classifier.

        Args:
            parsed_graphs (Dict[str, Dict[str, Any]]): The parsed graphs.
            text_to_triplets (Dict[str, List[Tuple]]): The texts and corresponding triplets.
        """
        classifier = Classifier()
        for text, triplets in text_to_triplets.items():
            graph = parsed_graphs[text]
            main_edge = graph["main_edge"]
            for triplet, positive in triplets:
                if not triplet.mapped:
                    logging.warning(
                        f"trying to map unmapped triplet {triplet} to {graph}"
                    )
                    success = triplet.map_to_subgraphs(graph)
                    if not success:
                        logging.warning("failed to map triplet, skipping")
                        continue

                logging.info("adding case:")
                logging.info(f"{text=}, {main_edge=}")
                logging.info(f"{triplet=}, {triplet.variables=}, positive: {positive}")

                # positive means whether we want to treat it as a positive or negative example
                # this helps graphbrain to learn the rules
                classifier.add_case(
                    main_edge, positive=positive, variables=triplet.variables
                )

        self.classifier = classifier

    def classify(self, graph: Hyperedge) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Classify the graph.

        Args:
            graph (Hyperedge): The graph to classify.

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: The matches and the rules triggered.
        """
        assert self.classifier is not None, "classifier not initialized"

        try:
            matches = self.classifier.classify(graph)
            rule_ids_triggered = self.classifier.rules_triggered(graph)
            logging.debug(f"{self.classifier.rules=}")
            logging.debug(f"{rule_ids_triggered=}")
            rules_triggered = [
                str(self.classifier.rules[rule_id - 1].pattern)
                for rule_id in rule_ids_triggered
            ]
        except AttributeError as err:
            logging.error(f"Graphbrain classifier threw exception:\n{err}")
            matches, rules_triggered = [], []

        logging.info(f"classifier matches: {matches}")
        logging.info(f"classifier rules triggered: {rules_triggered}")

        return matches, rules_triggered


@dataclass
class HITLManager:
    """A class to manage the HITL process and store parsed graphs.

    Attributes:
        parsed_graphs (Dict[str, Dict[str, Any]]): A dict mapping
            sentences to parsed graphs.
        annotated_graphs (Dict[str, List[Hyperedge]]): A dict mapping
            sentences to annotated graphs.
        triplets (Dict[str, List[Tuple]]): A dict mapping sentences to
            triplets.
        latest (Optional[str]): The latest sentence.
        extractor (Extractor): The extractor that uses classifiers to extract triplets from graphs.
        parser (TextParser): The text parser that parses text into graphs.
        parser_params (Dict): parameters to be used to initialize a TextParser object
    """

    def __init__(self, parser_url: Optional[str] = "http://localhost:7277"):
        self.text_parser = TextParserClient(parser_url)
        self.spacy_vocab = self.text_parser.get_vocab()
        self.latest = None
        self.parsed_graphs = {}
        self.text_to_triplets = defaultdict(list)
        self.oracle = None
        self.extractor = Extractor()
        logging.info("HITL manager initialized")

    def check_parser(self, parser_params):
        self.text_parser.check_params(parser_params)

    def load_extractor(self, extractor_data):
        self.extractor = Extractor.from_json(extractor_data)

    def load_data(self, graph_data, triplet_data, oracle=False):
        self.parsed_graphs = {
            text: GraphParse.from_json(graph_dict, self.spacy_vocab)
            for text, graph_dict in graph_data.items()
        }

        text_to_triplets = {
            text: [
                (
                    Triplet.from_json_and_graph(triplet[0], self.parsed_graphs[text]),
                    triplet[1],
                )
                for triplet in triplets
            ]
            for text, triplets in triplet_data.items()
        }

        if oracle:
            self.oracle = text_to_triplets
            self.text_to_triplets = defaultdict(list)
        else:
            self.text_to_triplets = defaultdict(list, text_to_triplets)

    @staticmethod
    def load(fn, oracle=False):
        logging.info(f"loading HITL state from {fn=}")
        with open(fn) as f:
            data = json.load(f)
        return HITLManager.from_json(data, oracle=oracle)

    @staticmethod
    def from_json(
        data: Dict[str, Any], parser_url="http://localhost:7277", oracle=False
    ):
        """
        load HITLManager from saved state

        Args:
            data (dict): the saved state, as returned by the to_json function

        Returns:
            HITLManager: a new HITLManager object with the restored state
        """
        hitl = HITLManager(parser_url)
        hitl.check_parser(data["parser_params"])
        hitl.load_data(data["parsed_graphs"], data["triplets"], oracle=oracle)
        hitl.load_extractor(data["extractor_data"])
        return hitl

    def to_json(self) -> Dict[str, Any]:
        """
        get the state of the HITLManager so that it can be saved

        Returns:
            dict: a dict with all the HITLManager object's attributes that are relevant to
                its state
        """

        return {
            "parsed_graphs": {
                text: graph.to_json() for text, graph in self.parsed_graphs.items()
            },
            "triplets": {
                text: [(triplet[0].to_json(), triplet[1]) for triplet in triplets]
                for text, triplets in self.text_to_triplets.items()
            },
            "extractor_data": self.extractor.to_json(),
            "parser_params": self.text_parser.get_params(),
        }

    def save(self, fn: str):
        """
        save HITLManager state to a file

        Args:
            fn (str): path of the file to be written (will be overwritten if it exists)
        """
        with open(fn, "w") as f:
            f.write(json.dumps(self.to_json()))

    def get_status(self) -> Dict[str, Any]:
        """
        return basic stats about the HITL state
        """
        n_rules = 0
        if self.extractor.classifier is not None:
            n_rules = len(self.extractor.classifier.rules)

        return {
            "n_sens": len(self.parsed_graphs),
            "n_annotated": len(self.text_to_triplets),
            "n_rules": n_rules,
        }

    def parse_text(self, text: str) -> List[GraphParse]:
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            List[Dict[str, Any]]: The parsed graphs.
        """
        return self.text_parser.parse(text)

    def get_rules(self, learn: bool = True) -> List[Rule]:
        """
        Get the rules.

        Args:
            learn (bool): whether to run graphbrain classifier's learn function.
                If False (default), only extract_patterns is called

        """

        _ = self.get_annotated_graphs()
        print("Get agraphs done!")
        self.extractor.extract_rules(learn=learn)
        print("Extract rules done!")

        return self.extractor.get_rules()

    def infer_triplets(self, sen: str) -> List[Triplet]:
        """
        match rules against sentence and return triplets corresponding to the matches

        Args:
            sen (str): the sentence to perform inference on

        Returns:
            List[Triple]: list of triplets inferred
        """
        logging.debug(f'inferring triplets for: "{sen}"')
        graph = self.parsed_graphs[sen]
        logging.debug(f'graph: "{graph}"')
        matches = self.match_rules(sen)
        logging.debug(f'matches: "{matches}"')
        triplets = matches2triplets(matches, graph)
        logging.debug(f'triplets: "{triplets}"')

        return triplets

    def triplets_to_str(self, triplets: List[Triplet], sen: str) -> List[str]:
        """
        Returns human-readable versions of triplets for a sentence

        Args:
            triplets (List[Triplet]): the triplets to convert
            sen (str): the sentence that is the source of this triplet

        Returns:
            List[str]: the human-readable form of the triplet
        """
        return [str(triplet) for triplet in triplets]

    def get_annotated_graphs(self) -> List[str]:
        """
        Get the annotated graphs.
        """

        self.extractor.add_cases(self.parsed_graphs, self.text_to_triplets)

        return self.extractor.get_annotated_graphs_from_classifier()

    def add_text_to_graphs(self, text: str) -> None:
        """Add the given text to the graphs.

        Args:
            text (str): The text to add to the graphs.

        Returns:
            None
        """
        self.get_graphs(text)

    def is_parsed(self, text: str) -> bool:
        """
        Check if the given text is parsed.
        """

        return text in self.parsed_graphs

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return [tok for tok in self.parsed_graphs[text]["spacy_sentence"]]

    def get_true_triplets(self) -> Dict[str, List[Triplet]]:
        """
        Get the triplets, return everything except the latest triplets.

        Returns:
            Dict[str, List[Triplet]]: The triplets.
        """

        return {
            sen: [triplet for triplet, positive in triplets if positive is True]
            for sen, triplets in self.text_to_triplets.items()
            if sen != "latest"
        }

    def get_graphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Get graphs for text, parsing it if necessary

        Args:
            text (str): the text to get the graphs for
            graphs (List[Dict[str, Any]]): the graphs corresponding to the text
        """
        if text in self.parsed_graphs:
            return [self.parsed_graphs[text]]

        graphs = self.parse_text(text)
        for graph in graphs:
            self.latest = text
            self.parsed_graphs[graph["text"]] = graph
            self.parsed_graphs["latest"] = graph

        return graphs

    def delete_triplet(self, text: str, triplet: Triplet):
        """
        Delete the triplet.

        Args:
            text (str): the text to delete the triplet for.
            triplet (Triplet): the triplet to delete
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.delete_triplet(self.latest, triplet)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"deleting from triplets: {text=}, {triplet=}")
        self.text_to_triplets[text].remove((triplet, True))

    def store_triplet(
        self,
        text: str,
        triplet: Triplet,
        positive=True,
    ):
        """
        Store the triplet.

        Args:
            text (str): the text to store the triplet for.
            triplet (Triplet): the triplet to store
            positive (bool): whether to store the triplet as a positive (default) or negative
                example
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, triplet, positive)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"appending to triplets: {text=}, {triplet=}")
        self.text_to_triplets[text].append((triplet, positive))

    def get_toks_from_txt(
        self, words_txt: str, sen: str, ignore_brackets: bool = False
    ) -> Tuple[int, ...]:
        """
        Map a substring of a sentence to its tokens. Used to parse annotations of triplets
        provided as plain text strings of the predicate and the arguments

        Args:
            words_txt (str): the substring of the sentence
            sen (str): the sentence
            ignore_brackets (bool): whether to remove brackets from the text before matching (required for ORE annotation)

        Returns:
            Tuple[int, ...] the tokens of the sentence corresponding to the substring
        """
        logging.debug(f"{words_txt=}, {sen=}")
        if ignore_brackets:
            pattern = re.escape(re.sub('["()]', "", words_txt))
        else:
            pattern = re.escape(words_txt)
        logging.debug(f"{pattern=}")
        if pattern[0].isalpha():
            pattern = r"\b" + pattern
        if pattern[-1].isalpha():
            pattern = pattern + r"\b"
        m = re.search(pattern, sen, re.IGNORECASE)

        if m is None:
            raise AnnotatedWordsNotFoundError(words_txt, pattern, sen)

        start, end = m.span()
        logging.debug(f"span: {(start, end)}")

        tok_i, tok_j = None, None
        tokens = self.get_tokens(sen)
        logging.debug(f"tokens: {tokens}")
        logging.debug(f"tok idxs: {[tok.idx for tok in tokens]}")
        for i, token in enumerate(tokens):
            if token.idx == start:
                tok_i = i
            if token.idx >= end:
                tok_j = i
                break
        if tok_i is None:
            logging.error(
                f'left side of annotation "{words_txt}" does not match the left side of any token in sen "{sen}"'
            )
            raise Exception()
        if tok_j is None:
            tok_j = len(tokens)

        return tuple(range(tok_i, tok_j))

    def get_unannotated_sentences(
        self, max_sens: Optional[int] = None, random_order: bool = False
    ) -> Generator[str, None, None]:
        """
        get a list of sentences that have been added and parsed but not yet annotated

        Args:
            max_sens (int): the maximum number of sentences to return. If None (this is the
                default) or larger than the total number of unannotated sentences, all
                unannotated sentences are returned
            random_order (bool): if False (default), sentences are yielded in the order of
                the self.parsed_graphs dict, which is always the same. If True, a random
                sample is generated, with a new random seed on each function call.

        Returns:
            Generator[str] the unannotated sentences
        """
        sens = [
            sen
            for sen in self.parsed_graphs
            if sen != "latest" and sen not in self.text_to_triplets
        ]
        n_graphs = len(sens)
        max_n = min(max_sens, n_graphs) if max_sens is not None else n_graphs

        if random_order:
            random.seed()
            logging.debug(f"sampling {max_n} indices from {n_graphs}")
            indices = set(random.sample(range(n_graphs), max_n))
            logging.debug(f"sample indices: {indices}")
            yield from (
                sen for i, sen in enumerate(sens) if i in indices and sen != "latest"
            )
        else:
            yield from sens[:max_n]

    def match_rules(self, sen: str) -> List[Dict]:
        """
        match rules against sentence by passing the sentence's graph to the extractor

        Args:
            sen (str): the sentence to be matched against

        Returns:
            List[Dict] a list of hypergraphs corresponding to the matches
        """
        graphs = self.get_graphs(sen)
        all_matches = []
        for graph in graphs:
            main_graph = graph["main_edge"]
            matches, _ = self.extractor.classify(main_graph)
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
            matches, rules_triggered = self.extractor.classify(graph["main_edge"])
            logging.info(f"matches: {matches}")
            triplets = matches2triplets(matches, graph)
            logging.info(f"triplets: {triplets}")

            if convert_to_text:
                matches = [
                    {k: v.label() for k, v in match.items()} for match in matches
                ]

            matches_by_text[graph["text"]]["matches"] = matches
            matches_by_text[graph["text"]]["rules_triggered"] = rules_triggered
            matches_by_text[graph["text"]]["triplets"] = triplets

        return matches_by_text

    def replace_atom_with_annotation(self, edge, to_replace, replacement, unique=False):
        atoms = edge.all_atoms()  # 'graphbrain.hyperedge.Atom'
        sum = 0
        for atom in atoms:
            if atom.root() == to_replace.lower():
                sum += 1
                found = atom  # saves first match
                break
        if sum == 0:
            return edge
        elif sum == 1 or unique == False:
            newatom = found.replace_atom_part(0, replacement)
            newedge = edge.replace_atom(
                found, newatom, unique=False
            )  # replaces all occurences with newatom when argroles match
            return newedge
        else:
            # ToDo: find correct atom when there are multiple atoms found
            print("multiple atoms found")

    def parse_sent_with_ann_eval(
        self,
        PATTERNS,
        extractions,
        max_items,
        expect_mappable=True,
        input="lsoie_wiki_dev.conll",
    ):
        stream = open(input, "r", encoding="utf-8")
        iter = 0
        parser = create_parser(lang="en", corefs=False)
        last_sent = ""
        for sen_idx, sen in enumerate(gen_tsv_sens(stream)):
            if iter == max_items:
                break
            print(f"processing sentence {sen_idx}")
            # get sentence
            sent = []
            for _, tok in enumerate(sen):
                sent.append(tok[1])

            # parse sentence
            sent = " ".join(sent)
            if sent == last_sent:
                continue
            last_sent = sent
            print(sent)
            parse_result = parser.parse(sent)
            for parse in parse_result["parses"]:
                main_edge = parse["main_edge"]
                atom2word = parse["atom2word"]
                if main_edge:
                    # print(f"{sent=}, {main_edge=}, {atom2word=}")
                    information_extraction(
                        extractions, main_edge, sen_idx, atom2word, PATTERNS
                    )

            # next round
            iter += 1

        # return(extractions)

    def parse_sent_with_annotations(
        self, max_items, expect_mappable=True, input="lsoie_wiki_train.conll", output=""
    ):
        stream = open(input, "r", encoding="utf-8")
        iter = 0
        for sen_idx, sen in enumerate(gen_tsv_sens(stream)):
            if iter == max_items:
                break
            print(f"processing sentence {sen_idx}")
            sent = []
            argsdict = defaultdict(list)
            predlist = []

            for i, tok in enumerate(sen):
                sent.append(tok[1])
                label = tok[7].split("-")[0]
                if label == "O":
                    continue
                elif label == "P":
                    predlist.append(i)
                    continue
                argsdict[label].append(i)
            args_sort = dict(sorted(argsdict.items()))
            args = []
            for label in args_sort:
                args.append(tuple(argsdict[label]))
            pred = tuple(predlist)

            # print annotations (positions)
            print(pred, args)

            # parse sentence
            sent = " ".join(sent)
            self.get_graphs(sent)
            try:
                sen_graph = self.parsed_graphs[sent]
            except KeyError:
                if not self.is_parsed(sent):
                    continue
                sen_graph = self.parsed_graphs["latest"]

            # create triplet
            triplet = Triplet(pred, args, sen_graph)
            if expect_mappable and triplet.mapped is False:
                print("Could not map annotation to subedges!")
                continue
            elif isinstance(triplet, Triplet):
                self.store_triplet(sent, triplet, True)

            # elif isinstance(triplet, Triplet):
            #     self.store_triplet(sent, triplet, True)
            #     success = triplet.map_to_subgraphs(sen_graph)
            #     if not success:
            #         logging.warning("failed to map triplet, skipping")
            #         continue
            # print(f"{triplet=}, {triplet.variables=}")

            # next round
            iter += 1

        # pred, args: ((7,), [(3, 4), (9, 10)])
        # variables: {'REL': found/Pd.poxx.<pf----/en,
        # 'ARG0': (and/J/en (a/Md/en man/Cc.s/en) (two/M#/en children/Cc.p/en)),
        # 'ARG1': (this/Md/en morning/Cc.s/en),
        # 'ARG2': (at/Br.ma/en (a/Md/en home/Cc.s/en) (+/B.mm/. (+/B.am/. fort/Cp.s/en hood/Cp.s/en) (in/Br.ma/en (a/Md/en (+/B.am/. (+/B.am/. us/Cp.s/en army/Cp.s/en) base/Cc.s/en)) texas/Cp.s/en)))}

    def parse_sent_with_annotations_v2(
        self, max_items, input="sample.pkl", path="ex.db"
    ):
        with open(input, "rb") as f:
            loaded_dict = pickle.load(f)
        if os.path.exists(path):
            os.remove(path)
        hg = hgraph(path)
        iter = 0

        for i in range(0, min(len(loaded_dict), max_items)):
            print(f"processing sentence {i}")
            sen = " ".join(loaded_dict[i]["sent"])
            toks = loaded_dict[i]["sent"]
            # parse sentence
            self.get_graphs(sen)
            edge = self.parsed_graphs["latest"]["main_edge"]
            newedge = edge
            for p in loaded_dict[i]["pred"]:
                to_replace = toks[p]
                # print(to_replace)
                replacement = "REL"
                newedge = self.replace_atom_with_annotation(
                    newedge, to_replace, replacement, unique=False
                )
            args = loaded_dict[i]["args"]
            for replacement in args:
                # print(replacement)
                for a in args[replacement]:
                    to_replace = toks[a]
                    # print(to_replace)
                    newedge = self.replace_atom_with_annotation(
                        newedge, to_replace, replacement, unique=False
                    )

            print(newedge)
            # ToDo: type inference rules (p. 8) for equal annotations
            subedges = newedge.subedges()
            for se in subedges:
                if se.atom:
                    continue
                atoms = se.all_atoms()
                roots = []
                for atom in atoms:
                    roots.append(atom.root())
                if len(set(roots)) == 1:
                    print(se)
                    print(se.argroles())
                    newatom = (
                        "(" + roots[0] + "/" + se.type() + "." + se.argroles() + ")"
                    )
                    newedge = se.replace_atom(se, newatom, unique=False)
                    print(newatom)
                    print(newedge)

                    # how to build hyperedge again?

            break

            # exclusion of conjunctions
            edges = conjunctions_decomposition(newedge, concepts=True)
            for e in edges:
                hg.add(e)
            iter += 1
            break
        return hg

    def generalise_graph_v2(self, hg, top_n=50):

        # transform hyperedges into abstract patterns
        pc = PatternCounter(
            expansions={
                "(* * *)",
                "(* * * *)",
            },
            match_roots={"+/B"},
            count_subedges=False,
        )

        for e in hg.all():
            if hg.is_primary(e):
                try:
                    pc.count(e)
                except:
                    continue

        return pc.patterns.most_common(top_n)

    def generalise_graph(self, top_n=50, path="example.db"):

        # transform hyperedges into abstract patterns
        pc = PatternCounter(count_subedges=False)

        # 1st version
        # for _, graph in self.parsed_graphs.items():
        #     edge = graph["main_edge"]
        #     # exclusion of conjunctions
        #     edges = conjunctions_decomposition(edge, concepts=True)
        #     for e in edges:
        #         pc.count(e)

        # 2nd version (https://graphbrain.net/tutorials/hypergraph-operations.html#parse-sentence-and-add-hyperedge-to-hypergraph)
        # if os.path.exists(path):
        #     os.remove(path)
        # hg = hgraph(path)
        # for _, graph in self.parsed_graphs.items():
        #     edge = graph["main_edge"]
        #     # exclusion of conjunctions
        #     edges = conjunctions_decomposition(edge, concepts=True)
        #     for e in edges:
        #         hg.add(e)

        # for e in hg.all():
        #     if hg.is_primary(e):
        #         pc.count(e)

        # 3rd version
        # iter = 0
        # for edge in self.get_annotated_graphs():
        #     print(edge)
        #     pc.count(edge)
        #     iter += 1
        #     if iter == max_items:
        #         break

        # 4th version
        cases = [edge for edge, positive in self.extractor.classifier.cases if positive]
        for edge in cases:
            pc.count(edge)

        return pc.patterns.most_common(top_n)
