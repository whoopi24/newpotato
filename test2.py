import pkg_resources
from graphbrain.parsers.alpha import Alpha

#cases_str = pkg_resources.resource_string("graphbrain", "data/atoms-en.csv").decode("utf-8")
#alpha = Alpha(cases_str)

with open("atoms-en.csv", 'rt') as f:
    for line in f.readlines():
        row = line.strip().split('\t')
        print(row)
        break