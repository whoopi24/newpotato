import json
import logging
import os
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional  # , Tuple

from graphbrain.learner.classifier import apply_curly_brackets
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens, tuple_if_list

from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graphbrain_extractor_PC import GraphbrainMappedTriplet

# TODO: do not use import * -> check which functions are needed
from newpotato.modifications.oie_patterns import *
from newpotato.modifications.pattern_ops import *  # all_variables
from newpotato.modifications.patterns import PatternCounter


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
    """

    def __init__(self, extractor_type):
        self.latest = None
        self.text_to_triplets = defaultdict(list)
        self.oracle = None
        self.extractor_type = extractor_type
        self.init_extractor()
        logging.info("HITL manager initialized")

    def init_extractor(self):
        if self.extractor_type == "ud":
            from newpotato.extractors.graph_extractor import GraphBasedExtractor

            self.extractor = GraphBasedExtractor()
        elif self.extractor_type == "graphbrain":
            from newpotato.extractors.graphbrain_extractor_PC import GraphbrainExtractor

            self.extractor = GraphbrainExtractor()
        else:
            raise ValueError(f"unsupported extractor type: {self.extractor_type}")

    def load_extractor(self, extractor_data):
        self.extractor = Extractor.from_json(extractor_data)

    def load_triplets(self, triplet_data, oracle=False):
        text_to_triplets = {
            tuple_if_list(item["text"]): [
                (
                    GraphbrainMappedTriplet.from_json(triplet[0]),
                    triplet[1],
                )
                for triplet in item["triplets"]
            ]
            for item in triplet_data
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
    def from_json(data: Dict[str, Any], oracle=False):
        """
        load HITLManager from saved state

        Args:
            data (dict): the saved state, as returned by the to_json function

        Returns:
            HITLManager: a new HITLManager object with the restored state
        """
        hitl = HITLManager(extractor_type=data["extractor_type"])
        hitl.load_triplets(data["triplets"], oracle=oracle)
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
            "triplets": [
                {
                    "text": text,
                    "triplets": [
                        (triplet[0].to_json(), triplet[1])
                        for triplet in triplets
                        if triplet[0] != False
                    ],
                }
                for text, triplets in self.text_to_triplets.items()
            ],
            "extractor_data": self.extractor.to_json(),
            "extractor_type": self.extractor_type,
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

        return {
            "n_sens": len(self.extractor.parsed_graphs),
            "n_annotated": len(self.text_to_triplets),
            "n_rules": self.extractor.get_n_rules(),
        }

    def get_rules(self, *args, **kwargs):
        return self.extractor.get_rules(self.text_to_triplets, *args, **kwargs)

    def print_rules(self, console):
        return self.extractor.print_rules(console)

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        return self.extractor.infer_triplets(sen, **kwargs)

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
        # assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"deleting from triplets: {text=}, {triplet=}")
        logging.info(self.text_to_triplets[text])
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
        logging.info(f"appending to triplets: {text=}, {triplet=}")
        self.text_to_triplets[text].append((triplet, positive))

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
            for sen in self.extractor.parsed_graphs
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

    # my own functions
    def parse_sent_with_ann_eval(
        self,
        patterns,
        extractions,
        max_items,
        input="lsoie_wiki_dev.conll",
    ):

        # use load_and_map_lsoie(input_file, extractor) without mapping?
        # replaced new parser with existing one from HITLManager

        with open(input) as stream:
            total, skipped = 0, 0
            last_sent = ""
            for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
                if total == max_items:
                    break
                total += 1
                words = [t[1] for t in sen]
                sentence = " ".join(words)

                # avoid double parsing - ONLY when Triplet creation not needed!
                # remark: sen_idx skipped -> from 0 to xy with gaps
                if sentence == last_sent:
                    continue
                else:
                    last_sent = sentence

                # text parsing and information extraction (prediction)
                text_to_graph = self.extractor.get_graphs(sentence)
                # TODO: how to handle skipped cases
                if len(text_to_graph) > 1:
                    logging.error(f"sentence split into two: {words}")
                    logging.error(f"{text_to_graph=}")
                    logging.error("skipping")
                    skipped += 1
                    # print(f"{sen_idx=}")
                    # extractions[sen_idx] = []
                    logging.error(f"{skipped=}, {total=}")
                    continue

                try:
                    graph = text_to_graph[sentence]["main_edge"]
                except:
                    logging.error(f"{text_to_graph=}")
                    logging.error("skipping")
                    skipped += 1
                    # print(f"{sen_idx=}")
                    # extractions[sen_idx] = []
                    continue
                logging.debug(f"{sentence=}, {graph=}")

                if graph:  # maybe not necessary
                    atom2word = text_to_graph[sentence]["atom2word"]
                    # print(f"{sentence=}, {graph=}, {atom2word=}")
                    information_extraction(  # in oie_patterns.py
                        extractions, graph, sen_idx, atom2word, patterns
                    )

        # return(extractions)

    # function needed for functional patterns
    def get_annotated_graphs(self) -> List[str]:
        """
        Get the annotated graphs.
        """

        return self.extractor.add_cases(self.text_to_triplets)

        # self.extractor.add_cases(self.text_to_triplets)
        # return self.extractor.get_annotated_graphs_from_classifier()

    def generalise_graph(
        self, top_n=50, method="supervised", check_vars=True, path="hg_test.db"
    ):
        # differentiate between supervised/unsupervised
        if method == "unsupervised":
            # 1st version: simple unsupervised approach
            pc = PatternCounter(
                expansions={
                    "(* * *)",
                    "(* * * *)",
                },
                match_roots={"+/B"},
                count_subedges=False,
            )
            for _, graph in self.extractor.parsed_graphs.items():
                edge = graph["main_edge"]
                # exclusion of conjunctions
                edges = conjunctions_decomposition(edge, concepts=True)
                for e in edges:
                    try:
                        pc.count(e)
                    except:
                        continue

            return pc.patterns

        elif method == "supervised":

            # uses functional patterns from parsed sentences with mapped triplets
            # UPDATE: removed adding the cases to the classifier and double parsing

            patterns = Counter()
            # TODO: conjunction decomposition, include special builder
            # TODO: for hyperedge, positive in self.get_annotated_graphs(): -> what is more correct?
            for hyperedge in self.get_annotated_graphs():
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
                        print("count: ", pattern)
                        patterns[hedge(apply_curly_brackets(pattern))] += 1

            return patterns

            # 4th version - not tested!
            # cases = [edge for edge, positive in self.extractor.classifier.cases if positive]
            # for edge in cases:
            #     pc.count(edge)

            # 2nd version (https://graphbrain.net/tutorials/hypergraph-operations.html#parse-sentence-and-add-hyperedge-to-hypergraph)
            # takes longer than 1st version with the use of an existing hg (with manipulated edges)
            # necessary to set replace_yn=True when executing lsoie_gb.py
            # not done yet !!
            # pc = PatternCounter(
            #     expansions={
            #         "(* * *)",
            #         "(* * * *)",
            #     },
            #     # match_roots={"+/B"},
            #     count_subedges=False,
            # )
            # hg = hgraph(path)
            # for edge in hg.all():
            #     if hg.is_primary(edge):
            #         pc.count(edge)

        else:
            print("Invalid method!")
            return Counter()

    # def generalise_graph_v2(self, hg, top_n=50):

    #     # transform hyperedges into abstract patterns
    #     pc = PatternCounter(
    #         expansions={
    #             "(* * *)",
    #             "(* * * *)",
    #         },
    #         match_roots={"+/B"},
    #         count_subedges=False,
    #     )

    #     for e in hg.all():
    #         if hg.is_primary(e):
    #             try:
    #                 pc.count(e)
    #             except:
    #                 continue

    #     return pc.patterns.most_common(top_n)


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
