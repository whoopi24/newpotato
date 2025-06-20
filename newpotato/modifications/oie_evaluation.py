import logging
import re

from graphbrain.hyperedge import hedge
from graphbrain.utils.conjunctions import conjunctions_decomposition
from rapidfuzz import process

from newpotato.modifications.pattern_ops import all_variables, apply_variable
from newpotato.modifications.patterns import match_pattern


## function from graphbrain > learner > classifier.py
def apply_curly_brackets(edge):
    if edge.atom:
        if edge.mtype() in {"P", "B"}:
            argroles = edge.argroles()
            if len(argroles) > 0 and argroles[0] != "{":
                return edge.replace_argroles("{" + argroles + "}")
        return edge
    else:
        return hedge([apply_curly_brackets(subedge) for subedge in edge])


## functions for evaluation (from openie.py)
# function to find matching key in dictionary with small changes
def get_closest_key(target, key_dict) -> str:
    closest_match, score, _ = process.extractOne(target, key_dict.keys())
    return closest_match if score > 80 else ""


def edge_text(atom2word, edge):
    atoms = edge.all_atoms()
    atom2word = {k.to_str(): v for k, v in atom2word.items()}
    words = [
        atom2word[atom.to_str()] for atom in atoms if atom.to_str() in atom2word.keys()
    ]
    if len(words) == 0:
        modified_atoms = [get_closest_key(str(atom), atom2word) for atom in atoms]
        words = [atom2word[atom] for atom in modified_atoms if len(atom) > 0]
        if len(words) == 0:
            logging.error("matching issue (edge <> atom2word)")
            logging.error(f"{edge=}")

    words.sort(key=lambda word: word[1])
    text = " ".join([word[0] for word in words])
    # remove spaces around non alpha-numeric characters
    # e.g.: "passive-aggressive" instead of "passive - aggressive"
    text = re.sub(" ([^a-zA-Z\\d\\s]) ", "\\g<1>", text)
    return text


def label(edge, atom2word):
    if type(edge) == list:
        final_result = []
        for e in edge:
            if e.to_str() in ["+/B.mm/.", "+/B.ma/.", "+/B.am/."]:
                continue
            final_result.append(edge_text(atom2word, e))
        return " ".join(final_result)
    if edge.to_str() in ["+/B.mm/.", "+/B.ma/.", "+/B.am/."]:
        return ""
    return edge_text(atom2word, edge)


def main_conjunction(edge):
    if edge.is_atom():
        return edge
    if edge[0] == ":/J/.":
        return edge[1]
    return hedge([main_conjunction(subedge) for subedge in edge])


# function to save extraction in the same format as WiRe57 data
def add_to_extractions(extractions, sent_id, arg1, rel, arg2, arg3, cnt=0):
    data = {"arg1": arg1, "rel": rel, "arg2": arg2, "extractor": "shg"}
    if len(arg3) > 0:
        data["arg3+"] = arg3

    # check several cases
    # case 1: new sentence
    if sent_id not in extractions:
        if len(arg3) > 0:
            data["arg3+"] = arg3
        extractions[sent_id] = [data]
        logging.info(f"{arg1=}, {rel=}, {arg2=}, {arg3=}")
        return cnt + 1
    # case 2: new triplet arg1, rel, arg2 for existing sentence
    elif data not in extractions[sent_id]:
        if len(arg3) > 0:
            data["arg3+"] = arg3
        extractions[sent_id].append(data)
        logging.info(f"{arg1=}, {rel=}, {arg2=}, {arg3=}")
        return cnt + 1
    # case 3: new argument arg3+ for existing triplet arg1, rel, arg2
    elif len(arg3) > 0:
        existing_extr = extractions[sent_id]
        for idx, extr in enumerate(existing_extr):
            # check if arg1, rel, arg2 are the same
            if extr == data:
                if "arg3+" in extr:
                    # arg3 is datatype list!
                    for arg3_item in arg3:
                        if arg3_item not in existing_extr[idx]["arg3+"]:
                            existing_extr[idx]["arg3+"] += arg3_item
                            logging.info(f"{arg1=}, {rel=}, {arg2=}, arg3={arg3_item}")
                            cnt += 1
                    return cnt
                else:
                    existing_extr[idx]["arg3+"] = arg3
                    logging.info(f"{arg1=}, {rel=}, {arg2=}, {arg3=}")
                    return cnt + 1
    # case 4: redundant extraction
    return cnt


def find_tuples(extractions, edge, sent_id, atom2word, patterns, max_extr):
    extr_cnt = 0
    for pattern in patterns:
        special = False
        pattern = hedge(pattern)
        atoms = pattern.atoms()
        roots = {atom.root() for atom in atoms if atom.root() != "*"}
        # TODO: fix issue in match_pattern() with special builder atom +/B
        if "+" in roots:
            special = True
        # skip patterns with only two variables, e.g. (REL/P.{p} ARG0/C)
        elif len(roots) < 3:
            logging.debug("skip pattern")
            continue
        for match in match_pattern(edge, pattern):
            logging.debug(f"{pattern=}")
            logging.debug(f"{match=}")
            # stop extracting triplets after certain number of extractions
            if extr_cnt == max_extr:
                return
            elif special:
                logging.debug("special builder pattern")
            # skip missing/incomplete matches
            elif not match or len(match) != len(roots):
                logging.debug("skip match")
                continue

            # attention: arg1 = ARG0; arg2 = ARG1 for LSOIE patterns
            if "ARG0" in match:
                arg1 = label(match["ARG0"], atom2word)
                arg2 = label(match["ARG1"], atom2word)

                if "ARG2" in match.keys():
                    arg3 = [label(match["ARG2"], atom2word)]
                else:
                    arg3 = []

                # the following case is rare (not in top20 pattern)
                if "ARG3" in match.keys():
                    arg3.append(label(match["ARG3"], atom2word))

            # else for original SH patterns
            else:
                arg1 = label(match["ARG1"], atom2word)
                arg2 = label(match["ARG2"], atom2word)

                if "ARG3..." in match:
                    arg3 = [label(match["ARG3..."], atom2word)]
                else:
                    arg3 = []

            # relation cannot be splitted (REL1/REL2/REL3) with my approach
            if "REL1" in match:
                rel_parts = []
                i = 1
                while "REL{}".format(i) in match:
                    rel_parts.append(label(match["REL{}".format(i)], atom2word))
                    i += 1
                rel = " ".join(rel_parts)
            elif special:
                rel = "is"
            else:
                rel = label(match["REL"], atom2word)

            extr_cnt = add_to_extractions(
                extractions, sent_id, arg1, rel, arg2, arg3, extr_cnt
            )


def information_extraction(
    extractions, main_edge, sent_id, atom2word, patterns, max_extr
):
    if main_edge.is_atom():
        return
    if main_edge.connector_type() == "B":
        logging.debug(f"find_tuples() for {main_conjunction(main_edge)=}")
        find_tuples(
            extractions,
            main_conjunction(main_edge),
            sent_id,
            atom2word,
            patterns,
            max_extr,
        )
    if main_edge.type()[0] == "R":
        try:
            edges = conjunctions_decomposition(main_edge, concepts=True)
            for edge in edges:
                logging.debug(f"find_tuples() for {main_conjunction(edge)=}")
                find_tuples(
                    extractions,
                    main_conjunction(edge),
                    sent_id,
                    atom2word,
                    patterns,
                    max_extr,
                )
        except IndexError:
            # e.g. IndexError: tuple index out of range: if edge[0].type() == 'J' and edge.mtype() != 'C':
            logging.error(
                f"Issue with conjunction decomposition for one of the subedges of {main_edge=}"
            )
    for edge in main_edge:
        information_extraction(
            extractions, edge, sent_id, atom2word, patterns, max_extr
        )


# TODO: remove these functions
def apply_variables(edge, variables):
    new_edge = edge
    for var_name, var_edge in variables.items():
        new_edge, found = apply_variable(new_edge, var_name, var_edge)
        if not found:
            logging.debug(f"{var_name=}, {var_edge=}, {new_edge=}")
            return None
    # added check if number of variables is correct
    var_matches = all_variables(new_edge)
    if len(variables) != len(var_matches):
        return None

    return new_edge
