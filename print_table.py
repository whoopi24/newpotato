import json
import os

import pandas as pd

# choose between "shg", "shg-org" or "clausie"
system = "clausie"

# load data
json_file = "evaluation_results_clausie.json"
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        try:
            results_dict = json.load(f)
        except json.JSONDecodeError:
            print("An error occured!")

table_file = json_file.split(".")[0] + "_table.txt"
with open(table_file, "w") as f:
    # prep list of rows with extracted information
    for key in results_dict:
        rows = []
        for entry in results_dict[key]:
            results = entry["results"][system]
            total = results["total"]
            matches = results["matches_only"]
            exact = results["exact_match"]

            if system == "shg":
                row = {
                    "pattern_cnt": entry["nr_of_patterns"],
                    "max_extr_cnt": entry["max_nr_of_extractions"],
                    "remark": entry["remark"],
                    "extr_cnt": exact["predicted"],
                    "gold_cnt": exact["reference"],
                    "matches_cnt": matches["nr_of_matches"],
                    "prec": total["precision"],
                    "rec": total["recall"],
                    "f1": total["f1_score"],
                    "exact_matches_cnt": exact["correct"],
                    "exact_prec": exact["precision"],
                    "exact_rec": exact["recall"],
                    "exact_f1": exact["f1_score"],
                    #'Raw Rate': entry['raw_rate'],
                    #'Evaluated Rate': entry['evaluated_rate'],
                    #'Avg Latency': entry['avg_latency']
                }
            else:
                row = {
                    "extr_cnt": exact["predicted"],
                    "gold_cnt": exact["reference"],
                    "matches_cnt": matches["nr_of_matches"],
                    "prec": total["precision"],
                    "rec": total["recall"],
                    "f1": total["f1_score"],
                    "exact_matches_cnt": exact["correct"],
                    "exact_prec": exact["precision"],
                    "exact_rec": exact["recall"],
                    "exact_f1": exact["f1_score"],
                }

            rows.append(row)

        df = pd.DataFrame(rows)

        # mark max f1 for other systems
        df["max_f1"] = df["f1"] == df["f1"].max()

        if system == "shg":
            # exclude "bi" scenario for max f1
            max_f1 = df.loc[df["remark"] != "bi", "f1"].max()
            df["max_f1"] = df["f1"] == max_f1
            # sort table
            df = df.sort_values(
                by=["pattern_cnt", "max_extr_cnt", "remark"],
                ascending=[True, True, True],
            )

        # round numeric columns for readability
        df.loc[:, df.dtypes == "float64"] = df.loc[:, df.dtypes == "float64"].round(3)

        f.write("-" * 35)
        f.write(f"\n Dataset {key}:\n")
        f.write("-" * 35)
        f.write("\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

print(f"Tables successfully saved to {table_file}!")
