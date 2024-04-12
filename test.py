from graphbrain import *
from graphbrain import hgraph
from graphbrain.notebook import *
from graphbrain.parsers import *
from graphbrain.patterns import PatternCounter

hg = hgraph("example.hg")
parser = create_parser(lang="en")
text = """
Mary is playing a very old violin.
"""
parse_results = parser.parse(text)
for parse in parse_results["parses"]:
    edge = parse["main_edge"]
    print(edge)
    hg.add(edge)

for edge in hg.all():
    show(edge, style="oneline")

hg = hgraph("ai.db")  # graphbrain.memory.sqlite.SQLite object
pc = PatternCounter(expansions={"(*/P * *)", "(*/P * * *)"})
iter = 0
for edge in hg.all():  # graphbrain.hyperedge.Hyperedge or graphbrain.hyperedge.Atom
    print(edge)
    break
    if hg.is_primary(edge):
        iter += 1
        pc.count(edge)
print(iter)
print(pc.patterns.most_common(10))
