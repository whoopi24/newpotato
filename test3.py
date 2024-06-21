import itertools
import logging
import operator

from graphbrain.hyperedge import Atom, hedge
from graphbrain.learner.pattern_ops import *
from graphbrain.parsers import *
from graphbrain.patterns import PatternCounter
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


# copied from: https://stackoverflow.com/questions/71732405/splitting-words-by-whitespace-without-affecting-brackets-content-using-regex
def split_pattern(s):
    string = str(s)
    result = []
    brace_depth = 0
    temp = ""
    string = string[1:-1]
    for ch in string:
        if ch == " " and brace_depth == 0:
            result.append(temp[:])
            temp = ""
        elif ch == "(" or ch == "[":
            brace_depth += 1
            temp += ch
        elif ch == "]" or ch == ")":
            brace_depth -= 1
            temp += ch
        else:
            temp += ch
    if temp != "":
        result.append(temp[:])
    logger.debug(f"split {s} into {result}")
    return result


def compare_patterns(edge1, edge2):
    e1 = split_pattern(edge1)
    e2 = split_pattern(edge2)
    if len(e1) == len(e2):
        logger.debug(f"patterns have equal length")
        final = []
        for i in range(0, len(e1)):
            s1 = e1[i]
            s2 = e2[i]
            # print(s1, s2)
            if s1 == s2:
                final.append(s1)
            elif s1.count(" ") == s2.count(" ") and s1.count(" ") > 0:
                logger.debug(f"recursion needed")
                s3 = compare_patterns(s1, s2)
                final.append("".join(s3))  # type: ignore
            elif s1[:3] == s2[:3]:
                logger.debug(f"patterns have common characters")
                # compare each character of the string
                s3 = []
                iter = 0
                for k, l in zip(s1, s2):
                    if iter < 4:
                        iter += 1
                        s3.append(k)
                    elif k == l:
                        s3.append(k)
                    else:
                        logger.debug(f"patterns were compressed")
                        s3.append("[" + k + l + "]")
                final.append("".join(s3))
            else:
                logger.debug(f"patterns cannot be compressed")
                return None
        final = "(" + " ".join(final) + ")"
        return hedge(final)
    else:
        logger.debug(f"patterns have unequal length")
        return None


def _simplify_pattern(edge):
    e1 = split_pattern(edge)
    final = []
    for i in range(0, len(e1)):
        s1 = e1[i]
        if s1.count(" ") > 0:
            logger.debug(f"recursion needed")
            s = _simplify_pattern(s1)
            final.append(str(s))
        # ignore argroles for concepts
        elif s1[2] == "C":
            final.append("".join(s1[:3]))
        # add order independent brackets
        elif len(s1) > 3:
            a1 = hedge(s1)
            roles = a1.argroles()
            final.append(s1[:4] + "{" + roles + "}")

        # no simplification possible
        else:
            final.append(s1)
    final = "(" + " ".join(final) + ")"
    return hedge(final)


def simplify_patterns(mylist):
    # initializations
    mydict = {}

    # create dictionary with patterns as keys and counts as values
    for p, cnt in mylist:
        print("p: ", p)
        new_p = _simplify_pattern(p)
        print("new_p: ", new_p)
        mydict[new_p] = mydict.get(new_p, 0) + cnt
    simplifed_patterns = sorted(
        mydict.items(), key=operator.itemgetter(1), reverse=True
    )
    return simplifed_patterns


# parser = create_parser(lang="en")
text = """
In 2007 , member states agreed that , in the future , 20 % of the energy used across the EU must be renewable ,
and carbon dioxide emissions have to be lower in 2020 by at least 20 % compared to 1990 levels.
"""
# text = """
# Between the three of them, during their training with Bruce, they won every karate championship in the United States.
# """
pc = PatternCounter(
    expansions={
        "(*/T */R)",
        "(*/T */C)",
        "(* * *)",
        "(* * * *)",
    },
    match_roots={"+/B"},
)

parse_results = parser.parse(text)
for parse in parse_results["parses"]:
    edges = conjunctions_decomposition(parse["main_edge"], concepts=True)
    for e in edges:
        pc.count(e)

edge1 = "(*/P.sor */C.ma */S */S)"
edge2 = "(*/P.sxr */C.mm */S */S)"
# edge1 = "(*/B.ma */C */C)"
# edge2 = "(*/B.mm */C */C)"
# edge1 = "(*/T */C)"
# edge1 = "(*/T (*/B.ma */C */C))"
# edge2 = "(*/T (*/B.mm */C */C))"

comm = common_pattern(hedge(edge1), hedge(edge2))
# print(comm)
# (*/P.{sxr} */C */S */S), (*/P.{so} */S */S),
# (*/B.{m} */C) -> not correct !!


# result = compare_pattern(edge1, edge2)
# print("result: ", result)

# (won/Pd.xxso.<f-----/en (between/T/en (of/Br.ma/en (the/Md/en three/C#/en) them/Ci/en)) (during/T/en (with/Br.ma/en (their/Mp/en training/Cc.s/en) bruce/Cp.s/en)) they/Ci/en (in/Br.ma/en (every/Md/en
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))

# test with list
mylist = [
    # "(*/B.ma */C */C)",
    # "(*/T */C)",
    "(*/P.{sx} */C (*/T */C))"
    # "(*/T */C.ma)",
    # "(*/B.mm */C */C)",
    # "(*/B.ma */C */C.ma)",
    "(*/P.{so} */C (*/B.{ma} */C */C))",
]

# problem: '(*/P.{s[ox]} */C (*/B[. ][{*][m/][aC][})])'
compressed = []
used_idx = []
for i in range(len(mylist)):
    if i in used_idx:
        continue
    for j in range(i + 1, len(mylist)):
        print(mylist[i], mylist[j])
        res = compare_patterns(mylist[i], mylist[j])
        print(res)
        if mylist[i] != res:
            compressed.append(res)
            used_idx.append(j)
            break
    # no compression found
    compressed.append(mylist[i])
print("result: ", set(compressed))

# test with dict
# mylist = [
#     ("(*/B.ma */C */C)", 309),
#     ("(*/T */C)", 107),
#     ("(*/T */C.ma)", 65),
#     ("(*/B.mm */C */C)", 45),
#     ("(*/B.ma */C */C.ma)", 41),
# ]
# mydict = {}
# for p, cnt in pc.patterns.most_common(5):
#     mydict[p] = cnt

# mydict = {}
# for p, cnt in mylist:
#     mydict[p] = cnt
# print(mydict)

# text 1
# {(*/B.ma */C */C): 5, (*/T */C.ma): 2, (*/T (*/B.ma */C */C)): 2}
# text 2
# {(*/T */C): 5, (*/B.ma */C */C): 3, (*/P.sxr */C.ma */S */S): 1,
# (*/P.sxr (*/B.ma */C */C) */S */S): 1, (*/P.sxr */C.ma (*/T */C) */S): 1}

# total
# {'(*/B.ma */C */C)': 309, '(*/T */C)': 107, '(*/T */C.ma)': 65, '(*/B.mm */C */C)': 45, '(*/B.ma */C */C.ma)': 41}
# mylist = pc.patterns.most_common(5)
# sim = simplify_patterns(mylist)
# print(sim)

# compressed = {}
# used_keys = []
# for key in mydict:
#     comp_tf = False
#     if key in used_keys:
#         continue
#     for key2 in mydict:
#         if key == key2 or key2 in used_keys:
#             continue
#         logger.debug(f"Compare {key} against {key2}")
#         res = compare_pattern(key, key2)
#         if key != res:
#             logger.debug(f"Compression found: {res}")
#             compressed[res] = mydict[key] + mydict[key2]
#             used_keys.append(key)
#             used_keys.append(key2)
#             comp_tf = True
#             break
#     # no compression found
#     if comp_tf == False:
#         compressed[key] = mydict[key]
# print("result: ", compressed)


# test common_pattern() -> macht schon fast zu viel: (*/B.ma */C */C) -> */C
# for p1, p2 in combs:
#     print(p1, p2)
#     res = common_pattern(hedge(p1), hedge(p2))
#     res = merge_patterns(hedge(p1), hedge(p2))
#     print(res)
