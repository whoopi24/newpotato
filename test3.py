from graphbrain import hgraph
from graphbrain.learner.pattern_ops import *
from graphbrain.parsers import *
from graphbrain.patterns import PatternCounter
from graphbrain.utils.conjunctions import conjunctions_decomposition, predicate

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

comm = common_pattern(hedge(edge1), hedge(edge2))
print(comm)
# (*/P.{sxr} */C */S */S), (*/P.{so} */S */S),
# (*/B.{m} */C) -> not correct !!


# (won/Pd.xxso.<f-----/en (between/T/en (of/Br.ma/en (the/Md/en three/C#/en) them/Ci/en)) (during/T/en (with/Br.ma/en (their/Mp/en training/Cc.s/en) bruce/Cp.s/en)) they/Ci/en (in/Br.ma/en (every/Md/en
# (+/B.am/. karate/Cc.s/en championship/Cc.s/en)) (the/Md/en (+/B.am/. united/Cp.s/en states/Cp.s/en))))
