import inspect
import json

import graphbrain.patterns as pattrn
from graphbrain import *
from graphbrain.hyperedge import UniqueAtom, hedge
from graphbrain.parsers import *
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate

# The 10 patterns that are shown in Table 4 and discussed in section 4.2.
# PATTERNS = [
#     "(REL/P.so,x ARG1 ARG2 ARG3+)",
#     "(REL/P.sc,x ARG1 ARG2 ARG3+)",
#     "(REL-1/P.sx~oc ARG1 (REL-2/T ARG2))",
#     "(REL-1/P.sr ARG1 ARG2)",
#     "(REL-1/P.px~oc ARG1 (REL-2/T ARG2))",
#     "(REL-1/P.pa,x ARG1 (REL-2/T ARG2) ARG3+)",
#     "(and (PRED/P.pa,x ARG2 ARG1 ARG3+) (= REL (inner-atom PRED)))",
#     "(REL-1/P.po,x ARG1 REL-2 (REL-3 ARG2))",
#     "(REL/P.s? ARG1 ARG2)",
#     "(REL/P.sc,x ARG1 (*/M ARG2) ARG3+)",
# ]

PATTERNS = [
    # "(REL/P.{[sp][cora]x} ARG1/C ARG2 ARG3...)",
    "(REL/P.{scx} ARG1/C ARG2 ARG3...)",
    "(REL/P.{sox} ARG1/C ARG2 ARG3...)",
    "(REL/P.{srx} ARG1/C ARG2 ARG3...)",
    "(REL/P.{sax} ARG1/C ARG2 ARG3...)",
    "(REL/P.{pcx} ARG1/C ARG2 ARG3...)",
    "(REL/P.{pox} ARG1/C ARG2 ARG3...)",
    "(REL/P.{prx} ARG1/C ARG2 ARG3...)",
    "(REL/P.{pax} ARG1/C ARG2 ARG3...)",
    # "(+/B.{m[ma]} (ARG1/C...) (ARG2/C...))",
    "(+/B.{ma} (ARG1/C...) (ARG2/C...))",
    "(+/B.{mm} (ARG1/C...) (ARG2/C...))",
    "(REL1/P.{sx}-oc ARG1/C (REL2/T ARG2))",
    "(REL1/P.{px} ARG1/C (REL2/T ARG2))",
    "(REL1/P.{sc} ARG1/C (REL3/B REL2/C ARG2/C))",
]


# Directories & files
DIR = "WiRe57/data"
EXTR_MANUAL = "WiRe57_343-manual-oie.json"
EXTR_BEFORE = (
    "WiRe57_extractions_by_ollie_clausie_openie_stanford_minie_"
    "reverb_props-export.json"
)
EXTR_AFTER = (
    "WiRe57_extractions_by_ollie_clausie_openie_stanford_minie_"
    "reverb_props_shg-export.json"
)


def label(edge, atom2word):
    return edge_text(atom2word, edge)


# Conjunction resolution, as discussed in section 4.1
# def conjunctions_resolution(edge, atom2word):
#     if edge.is_atom():
#         return []

#     if edge[0].type() == "J" and edge.type()[0] == "R":
#         cur_subj = None
#         cur_pred = None
#         cur_role = None
#         edges = []
#         no_obj_edges = []
#         for subedge in edge[1:]:
#             subj = subedge.edges_with_argrole("s")
#             passive = subedge.edges_with_argrole("p")
#             newedge = subedge

#             if (len(subj) > 0 or len(passive) > 0) and len(subedge) == 2:
#                 no_obj_edges.append(subedge)

#             if len(subj) > 0 and subj[0] is not None:
#                 cur_subj = subj[0]
#                 cur_pred = subedge[0]
#                 cur_role = "s"
#             elif len(passive) > 0 and passive[0] is not None:
#                 cur_subj = passive[0]
#                 cur_pred = subedge[0]
#                 cur_role = "p"
#             elif (
#                 cur_subj is not None
#                 and subedge.type()[0] == "R"
#                 and subedge[0].type() != "J"
#             ):
#                 newedge = hedge(
#                     [cur_pred.replace_atom(predicate(cur_pred), predicate(subedge[0]))]
#                 ) + hedge(subedge[1:])
#                 old_pred = predicate(newedge)
#                 newedge = newedge.insert_edge_with_argrole(cur_subj, cur_role, 0)
#                 new_pred = predicate(newedge)
#                 if old_pred and new_pred:
#                     old_pred_u = UniqueAtom(old_pred)
#                     new_pred_u = UniqueAtom(new_pred)
#                     atom2word[new_pred_u] = atom2word[old_pred_u]
#             new_edges = conjunctions_resolution(newedge, atom2word)
#             edges += new_edges
#         return edges

#     for pos, subedge in enumerate(edge):
#         if not subedge.is_atom():
#             if (
#                 subedge[0].type() == "J"
#                 and subedge[0].to_str()[0] != ":"
#                 and subedge.type()[0] == "C"
#             ):
#                 edges = []
#                 for list_item in subedge[1:]:
#                     subedges = conjunctions_resolution(hedge([list_item]), atom2word)
#                     for se in subedges:
#                         newedge = hedge(edge[0:pos]) + se + hedge(edge[pos + 1 :])
#                         edges.append(newedge)
#                 return edges
#             else:
#                 subedges = conjunctions_resolution(subedge, atom2word)
#                 if len(subedges) > 1:
#                     edges = []
#                     for list_item in subedges:
#                         newedge = (
#                             hedge(edge[0:pos])
#                             + hedge([list_item])
#                             + hedge(edge[pos + 1 :])
#                         )
#                         edges.append(newedge)
#                     return edges
#     return [edge]


def main_conjunction(edge):
    if edge.is_atom():
        return edge
    if edge[0] == ":/J/.":
        return edge[1]
    return hedge([main_conjunction(subedge) for subedge in edge])


def add_to_extractions(extractions, edge, sent_id, arg1, rel, arg2, arg3):
    data = {"arg1": arg1, "rel": rel, "arg2": arg2, "extractor": "shg", "score": 1.0}

    if len(arg3) > 0:
        data["arg3+"] = arg3

    extraction = "|".join((sent_id, edge.to_str(), arg1, rel, arg2))

    if extraction not in extractions:
        extractions[extraction] = {"data": data, "sent_id": sent_id}
    elif len(arg3) > 0:
        if "arg3+" in extractions[extraction]["data"]:
            extractions[extraction]["data"]["arg3+"] += arg3
        else:
            extractions[extraction]["data"]["arg3+"] = arg3


def find_tuples(extractions, edge, sent_id, atom2word):
    for pattern in PATTERNS:
        for match in pattrn.match_pattern(edge, pattern):
            arg1 = match["ARG1"]
            arg2 = match["ARG2"]
            if "ARG3..." in match:
                arg3 = [label(match["ARG3..."], atom2word)]
            else:
                arg3 = []

            if "REL1" in match:
                rel_parts = []
                i = 1
                while "REL{}".format(i) in match:
                    rel_parts.append(label(match["REL{}".format(i)], atom2word))
                    i += 1
                rel = " ".join(rel_parts)
            else:
                rel = label(match["REL"], atom2word)

            arg1 = label(arg1, atom2word)
            arg2 = label(arg2, atom2word)

            add_to_extractions(extractions, edge, sent_id, arg1, rel, arg2, arg3)


def information_extraction(extractions, main_edge, sent_id, atom2word):
    if main_edge.is_atom():
        return
    if main_edge.type()[0] == "R":
        # edges = conjunctions_resolution(main_edge, atom2word)
        edges = conjunctions_decomposition(main_edge, concepts=True)
        for edge in edges:
            find_tuples(extractions, main_conjunction(edge), sent_id, atom2word)
    for edge in main_edge:
        information_extraction(extractions, edge, sent_id, atom2word)


def parse_sent(extractions, parser, sent, sent_id):
    parse_result = parser.parse(sent)
    for parse in parse_result["parses"]:
        main_edge = parse["main_edge"]
        atom2word = parse["atom2word"]
        if main_edge:
            information_extraction(extractions, main_edge, sent_id, atom2word)


if __name__ == "__main__":
    parser = create_parser(lang="en", corefs=False)
    extractions = {}

    parse_sent(extractions, parser, "Mary likes astronomy and plays football.", "MY1")

    # manual = json.load(open("{}/{}".format(DIR, EXTR_MANUAL)))
    # extr = json.load(open("{}/{}".format(DIR, EXTR_BEFORE)))

    # for key in manual:
    #     print(key)
    #     for case in manual[key]:
    #         parse_sent(extractions, parser, case["sent"], case["id"])

    # for _, extraction in extractions.items():
    #     extr[extraction["sent_id"]].append(extraction["data"])

    # with open("{}/{}".format(DIR, EXTR_AFTER), "w", encoding="utf-8") as f:
    #     json.dump(extr, f, ensure_ascii=False, indent=4)
