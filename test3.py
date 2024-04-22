from graphbrain import hgraph
from graphbrain.learner.pattern_ops import *
from graphbrain.parsers import *
from graphbrain.patterns import PatternCounter
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate


# copied from: https://www.geeksforgeeks.org/python-count-the-number-of-matching-characters-in-a-pair-of-string/
def common_char(str1, str2):
    return len((set(str1)).intersection(set(str2)))


# copied from: https://stackoverflow.com/questions/71732405/splitting-words-by-whitespace-without-affecting-brackets-content-using-regex
def split_pattern(s):
    s = str(s)
    print("split: " + s)
    result = []
    brace_depth = 0
    temp = ""
    s = s[1:-1]
    for ch in s:
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
    print("into: ", result)
    return result


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
        "(+/B.am/. * *)",
        "(+/B.ma/. * *)",
        "(+/B.mm/. * *)",
        "(+/B/. * *)",
        "(* * * *)",
    }
)

# # +/B.am/.  bzw. +/B/. bzw. +/B.mm/. bzw. +/B.ma/.
# parse_results = parser.parse(text)
# for parse in parse_results["parses"]:
#     edges = conjunctions_decomposition(parse["main_edge"], concepts=True)
#     for e in edges:
#         pc.count(e)

# print(pc.patterns.most_common(10))

# for p, cnt in pc.patterns.most_common(10):
#     print(p)

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


def compare_pattern(edge1, edge2):
    e1 = split_pattern(edge1)
    e2 = split_pattern(edge2)
    print(e1, e2)
    if len(e1) == len(e2):
        print("equal length")
        final = []
        for i in range(0, len(e1)):
            s1 = e1[i]
            s2 = e2[i]
            # print(i)
            print(s1, s2)
            print(s1[:3], s2[:3])
            if s1 == s2:
                print("equal")
                final.append(s1)
            elif s1.count(" ") == s2.count(" ") and s1.count(" ") > 0:
                print("recursion")
                s3 = compare_pattern(s1, s2)
                final.append("".join(s3))  # type: ignore
            elif s1[:3] == s2[:3]:  # common_char(s1, s2) > 2 and len(s1) == len(s2):
                print("common characters")
                s3 = []
                iter = 0
                # compare each character of the string
                for k, l in zip(s1, s2):
                    if iter < 4:
                        iter += 1
                        s3.append(k)
                    elif k == l:
                        s3.append(k)
                    else:
                        print("compressed")
                        s3.append("[" + k + l + "]")
                final.append("".join(s3))
            else:
                print("cannot compress pattern")
                return edge1
        print(final, type(final))
        final = "(" + " ".join(final) + ")"
        return final
    else:
        print("patterns have unequal length")
        return edge1


# result = compare_pattern(edge1, edge2)
# print("result: ", result)

# (won/Pd.xxso.<f-----/en (between/T/en (of/Br.ma/en (the/Md/en three/C#/en) them/Ci/en)) (during/T/en (with/Br.ma/en (their/Mp/en training/Cc.s/en) bruce/Cp.s/en)) they/Ci/en (in/Br.ma/en (every/Md/en
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))

mylist = []
compressed = []
# for p, _ in pc.patterns.most_common(10):
#     mylist.append(p)
mylist = [
    "(*/B.ma */C */C)",
    "(*/T */C)",
    "(*/T */C.ma)",
    "(*/B.mm */C */C)",
    "(*/B.ma */C */C.ma)",
    # "(*/T (*/B.ma */C */C))",
    # "(*/B.ma */C.ma */C)",
    # "(*/B.ma (*/B.ma */C */C) */C)",
    # "(*/B.ma */C */C.mm)",
    # "(*/P.so */C */C.ma)",
    # "(*/B.mm */C */C.ma)",
]
for i in range(len(mylist)):
    for j in range(i + 1, len(mylist)):
        print(mylist[i], mylist[j])
        res = compare_pattern(mylist[i], mylist[j])
        compressed.append(res)
print("result: ", set(compressed))
