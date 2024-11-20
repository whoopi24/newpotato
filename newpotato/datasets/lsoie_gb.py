import argparse
import logging
import os
import re
from collections import defaultdict

import networkx as nx
from graphbrain import hgraph
from graphbrain.utils.conjunctions import conjunctions_decomposition
from rich.console import Console
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

from newpotato.datatypes import Triplet
from newpotato.hitl_marina import HITLManager

console = Console()


def replace_atom_with_annotation(edge, to_replace, replacement, unique=False):
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
        # TODO: find correct atom when there are multiple atoms found
        print("multiple atoms found")


def extract_subedges(hyperedge, max_depth=1):
    hyperedge = str(hyperedge)
    subedges = []
    stack = []
    current = ""
    current_depth = 0  # Track the current depth

    for char in hyperedge:
        if char == "(":
            if current_depth < max_depth:
                if stack:  # If already inside a parenthesis, save to current
                    current += char
                stack.append("(")
                current_depth += 1
            else:
                current += char  # Beyond max_depth, add to the current subedge
        elif char == ")":
            if current_depth <= max_depth:
                if stack:
                    stack.pop()
                if stack:
                    current += char
                else:
                    # End of a subedge within max_depth
                    subedges.append(current.strip())
                    current = ""
            else:
                current += char
            current_depth = len(stack)  # Update depth based on stack
        else:
            if stack:
                current += char

    return subedges


def load_and_map_lsoie(input_file, extractor, replace_yn=False, path="hg_test.db"):
    with open(input_file) as stream:
        total, skipped = 0, 0
        # creates hypergraph for replaced (sub)edges
        if os.path.exists(path):
            os.remove(path)
        hg = hgraph(path)
        for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
            total += 1
            words = [t[1] for t in sen]
            sentence = " ".join(words)
            text_to_graph = extractor.get_graphs(sentence)
            if len(text_to_graph) > 1:
                logging.error(f"sentence split into two: {words}")
                logging.error(f"{text_to_graph=}")
                logging.error("skipping")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue

            graph = text_to_graph[sentence]["main_edge"]
            logging.debug(f"{sentence=}, {graph=}")

            arg_dict = defaultdict(list)
            pred = []
            for i, tok in enumerate(sen):
                label = tok[7].split("-")[0]
                if label == "O":
                    continue
                elif label == "P":
                    pred.append(i)
                    continue
                arg_dict[label].append(i)

            pred = tuple(pred)
            args = [
                tuple(indices)
                for label, indices in sorted(
                    arg_dict.items(), key=lambda i: int(i[0][1:])
                )
            ]
            logging.debug(f"{pred=}, {args=}")

            triplet = Triplet(pred, args, toks=words)
            try:
                mapped_triplet = extractor.map_triplet(triplet, sentence, strict=False)
            except (KeyError, nx.exception.NetworkXPointlessConcept):
                logging.error(f"error mapping triplet: {triplet=}, {words=}")
                logging.error("skipping")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue

            hg.add(graph)
            if replace_yn:
                newedge = graph
                # print(newedge)
                for p in pred:
                    to_replace = words[p]
                    # print(to_replace)
                    replacement = "REL"
                    newedge = replace_atom_with_annotation(
                        newedge, to_replace, replacement, unique=False
                    )
                count = 0
                for arg in args:
                    replacement = "A" + str(count)
                    # print(replacement)
                    for a in arg:
                        to_replace = words[a]
                        # print(to_replace)
                        newedge = replace_atom_with_annotation(
                            newedge, to_replace, replacement, unique=False
                        )
                    count += 1
                # print(newedge)
                arg_cnt = len(args)

                # exclusion of conjunctions
                edges = conjunctions_decomposition(newedge, concepts=True)
                # only add new edge to hg if all annotations are included
                for newedge in edges:
                    # find roots with ARG
                    roots = {atom.root() for atom in newedge.atoms()}
                    pattern = r"^A[0-5]"
                    matches = [a for a in roots if re.match(pattern, a)]
                    cnt = len(matches)
                    if cnt == arg_cnt and "REL" in roots:
                        # TODO: type inference rules (p. 8) for equal annotations
                        newedge = newedge.simplify(
                            subtypes=False, argroles=False, namespaces=False
                        )
                        # subedges = extract_subedges(newedge)
                        # print(subedges)

                        # save new edge with annotations
                        hg.add(newedge)
                    else:
                        # print("skipped")
                        continue

                # TODO: how to re-build entire hyperedge again after replacing several subedges?
                # TODO: unite atoms with equal annotations
                # subedges = newedge.simplify(
                #     subtypes=False, argroles=False, namespaces=False
                # ).subedges()
                # for se in subedges:
                #     if se.atom:
                #         continue
                #     print("se:", se)
                #     atoms = se.all_atoms()
                #     roots = {atom.root() for atom in atoms}
                #     if len(set(roots)) == 1:
                #         print("can be united")
                #         print(se.type()) # or mtype() ??
                #         newatom = roots[0] + "/" + se.type()
                #         newedge = se.replace_atom(se, newatom, unique=False)
                #         print(newatom)
                #         print(newedge)

            yield sentence, mapped_triplet


def load_lsoie_to_hitl(input_file, hitl):
    for sen, triplet in load_and_map_lsoie(input_file, hitl.extractor, True):
        hitl.store_triplet(sen, triplet, True)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-s", "--state_file", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("initializing HITL session")
    hitl = HITLManager(extractor_type="graphbrain")
    console.print(f"loading LSOIE data from {args.input_file}")
    load_lsoie_to_hitl(args.input_file, hitl)
    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)


if __name__ == "__main__":
    main()
