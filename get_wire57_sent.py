# Python program to read
# json file

import json

# Opening JSON file
DIR = "WiRe57/data"
EXTR_MANUAL = "WiRe57_343-manual-oie.json"
EXTR_AFTER = "WiRe57_test.txt"
manual = json.load(open("{}/{}".format(DIR, EXTR_MANUAL)))
sentences = []

# Iterating through the json
with open("{}/{}".format(DIR, EXTR_AFTER), "w", encoding="utf-8") as f:
    for key in manual:
        for case in manual[key]:
            print(case["id"])
            f.write("{}\n".format(case["sent"]))
