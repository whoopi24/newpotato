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
    return result


parser = create_parser(lang="en")
# text = """
# In 2007 , member states agreed that , in the future , 20 % of the energy used across the EU must be renewable ,
# and carbon dioxide emissions have to be lower in 2020 by at least 20 % compared to 1990 levels.
# """
text = """
Between the three of them, during their training with Bruce, they won every karate championship in the United States. 
"""
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

# +/B.am/.  bzw. +/B/. bzw. +/B.mm/. bzw. +/B.ma/.
parse_results = parser.parse(text)
for parse in parse_results["parses"]:
    edges = conjunctions_decomposition(parse["main_edge"], concepts=True)
    for e in edges:
        pc.count(e)

# print(pc.patterns.most_common(10))

for p, cnt in pc.patterns.most_common(10):
    print(p)

# edge1 = "(*/P.sxr */C.ma */S */S)"
# edge2 = "(*/P.sxr */C.mm */S */S)"

edge1 = "(*/B.ma */C */C)"
edge2 = "(*/B.mm */C */C)"
# edge1 = "(*/T */C)"
# edge2 = "(*/T (*/B.ma */C */C))"

comm = common_pattern(hedge(edge1), hedge(edge2))
print(comm)
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
            print(i)
            print(s1, s2)
            if s1 == s2:
                final.append(s1)
            elif common_char(s1, s2) > 2:
                print("common characters")
                if s1.count(" ") > 0:
                    s1 = split_pattern(s1)
                    print(s1)
                else:
                    s1 = [s1]
                if s2.count(" ") > 0:
                    s2 = split_pattern(s2)
                    print(s2)
                else:
                    s2 = [s2]
                if len(s1) != len(s2):
                    print("cannot compress pattern")
                    return None
                else:
                    print(s1)
                    if len(s1) == 1:

                    # ToDo: compare each character (not nested for loop)
                s3 = "tbd"
                final.append(s3)
            else:
                print("cannot compress pattern")
                return None

        return final


result = compare_pattern(edge1, edge2)
print("result: ", result)

# (won/Pd.xxso.<f-----/en (between/T/en (of/Br.ma/en (the/Md/en three/C#/en) them/Ci/en)) (during/T/en (with/Br.ma/en (their/Mp/en training/Cc.s/en) bruce/Cp.s/en)) they/Ci/en (in/Br.ma/en (every/Md/en
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))
