import json
from collections import Counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# load JSON file
json_file = "evaluation_results.json"
with open(json_file, "r") as f:
    data_dict = json.load(f)

# choose which precision-recall values to plot ("total", "matches_only", "exact_match")
category = "total"

# color palette for number of extractions (black, dark blue, dark gray, dark green, red)
extraction_colors = ["#000000", "#0b3d91", "#808080", "#1a4d2e", "#e41a1c"]

# color palette for number of patterns
pattern_colors = plt.get_cmap("Set2")(np.linspace(0, 1, 8))

# initialize min/max values
min_precision, max_precision = float("inf"), float("-inf")
min_recall, max_recall = float("inf"), float("-inf")

# find global min/max across all evaluations
for evaluations in data_dict.values():
    for eval_data in evaluations:
        results = eval_data["results"]["shg"].get(category, {})
        if "precision" in results and "recall" in results:
            min_precision = min(min_precision, results["precision"])
            max_precision = max(max_precision, results["precision"])
            min_recall = min(min_recall, results["recall"])
            max_recall = max(max_recall, results["recall"])

# expand range slightly for better visualization
padding = 0.05
precision_range = max_precision - min_precision
recall_range = max_recall - min_recall

min_precision -= precision_range * padding
max_precision += precision_range * padding
min_recall -= recall_range * padding
max_recall += recall_range * padding

# collect unique pattern and extraction values
unique_patterns = sorted(
    set(
        eval_data["nr_of_patterns"]
        for evaluations in data_dict.values()
        for eval_data in evaluations
    )
)
unique_extr = sorted(
    set(
        eval_data["max_nr_of_extractions"]
        for evaluations in data_dict.values()
        for eval_data in evaluations
    )
)

# create color maps
n_patterns = len(unique_patterns)
pattern_cmap = mcolors.ListedColormap(pattern_colors[:n_patterns])
n_extr = len(unique_extr)
extr_cmap = mcolors.ListedColormap(extraction_colors[:n_extr])

# explicitly set boundaries
pattern_norm = mcolors.BoundaryNorm([i for i in range(n_patterns + 1)], n_patterns)
extr_norm = mcolors.BoundaryNorm([i for i in range(n_extr + 1)], n_extr)

# create mapping
pattern_to_color = {
    unique_patterns[i]: pattern_cmap(i / n_patterns) for i in range(n_patterns)
}
extr_to_color = {unique_extr[i]: extr_cmap(i / n_extr) for i in range(n_extr)}

for input_file, evaluations in data_dict.items():
    if not evaluations:
        print(f"Skipping {input_file}: No data available.")
        continue

    # sort configurations by patterns (asc) and extractions (asc)
    evaluations.sort(key=lambda x: (x["nr_of_patterns"], x["max_nr_of_extractions"]))

    plt.figure(figsize=(8, 6))
    precision_values = []
    recall_values = []
    remarks = []
    f1_scores = []
    best_f1 = 0
    best_f1_point = None
    jitter_strength = 0.001
    redundant = False

    # count occurrences of (precision, recall) pairs for skipping/jittering
    value_counts = Counter(
        (res["precision"], res["recall"])
        for eval_data in evaluations
        if "precision" in (res := eval_data["results"]["shg"].get(category, {}))
        and "recall" in res
    )

    for eval_data in evaluations:
        results = eval_data["results"]["shg"].get(category, {})
        if "precision" not in results or "recall" not in results:
            continue
        precision, recall, f1_score = (
            results["precision"],
            results["recall"],
            results["f1_score"],
        )

        # get key configurations
        num_patterns = eval_data["nr_of_patterns"]
        num_extractions = eval_data["max_nr_of_extractions"]

        # skip equal results if plotted already
        if value_counts[(precision, recall)] > 1 and redundant == True:
            print(
                f"skip redundant point {(round(recall, 3), round(precision, 3))} for {num_patterns=}, {num_extractions=}"
            )
            continue

        # plot first point of redundant results
        elif value_counts[(precision, recall)] > 1:
            redundant = True

        # save best f1 score
        if f1_score > best_f1:
            best_f1 = f1_score
            best_f1_point = (recall, precision)

        precision_values.append(precision)
        recall_values.append(recall)

        # scatter plot with face and edge color
        plt.scatter(
            recall,
            precision,
            s=80,
            c=[pattern_to_color[num_patterns]],
            edgecolors=[extr_to_color[num_extractions]],
            linewidths=1.5,
            alpha=0.8,
        )

        # store remark if available
        if "remark" in eval_data:
            remarks.append((precision, recall, eval_data["remark"]))

    if not precision_values:
        print(f"Skipping {input_file}: No valid precision-recall data.")
        continue

    arrow_length = (max(precision_values) - min(precision_values)) * 0.05

    # highlight best f1 point
    plt.annotate(
        "max f1",
        best_f1_point,
        xytext=(best_f1_point[0], best_f1_point[1] - arrow_length),
        arrowprops=dict(arrowstyle="-", linewidth=0.5),
        fontsize=9,
        color="black",
        alpha=0.75,
    )

    print(
        f"{input_file}: The maximum f1 score is {round(best_f1, 3)} for point ({round(best_f1_point[0], 3)}, {round(best_f1_point[1], 3)})!"
    )

    for py, rx, remark in remarks:
        if len(remark) > 1:
            plt.annotate(
                remark,
                (rx, py),
                xytext=(rx, py + arrow_length),
                arrowprops=dict(arrowstyle="-", linewidth=0.5),
                fontsize=9,
                color="black",
                alpha=0.75,
            )

    # axes and title
    left, right = plt.xlim()
    plt.xlim(left, right * 1.1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"Precision-Recall Plot for {input_file}")

    # colorbar for patterns
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=pattern_norm, cmap=pattern_cmap),
        ax=plt.gca(),
        ticks=np.arange(n_patterns) + 0.5,
    )
    cb1.set_ticklabels(unique_patterns)
    cb1.set_label("number of patterns (inner circle)")

    # colorbar for extractions
    cb2 = plt.colorbar(
        plt.cm.ScalarMappable(norm=extr_norm, cmap=extr_cmap),
        ax=plt.gca(),
        ticks=np.arange(n_extr) + 0.5,
    )
    cb2.set_ticklabels(unique_extr)
    cb2.set_label("max. number of extractions (outer circle)")

    filename = f"prec_rec_{input_file.split('.')[0]}.png"
    plt.savefig(filename, dpi=600, bbox_inches="tight")  # , transparent=True for slides
    plt.close()

print("Plots saved successfully!")
