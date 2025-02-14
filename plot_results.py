import json
import random
from collections import Counter

import matplotlib.pyplot as plt

# Load JSON file
json_file = "evaluation_results.json"
with open(json_file, "r") as f:
    data_dict = json.load(f)

category = "total"  # Choose which precision-recall values to plot ("total", "matches_only", "exact_match")

# Define a fixed set of distinguishable colors (extendable if needed)
distinct_colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Cyan
]

jitter_strength = 0.001

for input_file, evaluations in data_dict.items():
    if not evaluations:
        print(f"Skipping {input_file}: No data available.")
        continue

    # **Sort configurations**: first by nr_of_patterns ↓, then by max_nr_of_extractions ↓
    evaluations.sort(key=lambda x: (-x["nr_of_patterns"], -x["max_nr_of_extractions"]))

    # Create figure for each input file
    plt.figure(figsize=(8, 6))

    precision_values = []
    recall_values = []
    labels = []
    remarks = []
    color_map = (
        {}
    )  # Mapping each (nr_of_patterns, max_nr_of_extractions) to a unique color
    handles = []  # For the legend

    # Count occurrences of (precision, recall) pairs
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

        config = (eval_data["nr_of_patterns"], eval_data["max_nr_of_extractions"])

        # Assign a unique color per configuration
        if config not in color_map:
            color_map[config] = distinct_colors[len(color_map) % len(distinct_colors)]

        precision, recall = results["precision"], results["recall"]

        # Apply jitter only if the same (precision, recall) pair appears more than once
        if value_counts[(precision, recall)] > 1:
            precision += random.uniform(-jitter_strength, jitter_strength)
            recall += random.uniform(-jitter_strength, jitter_strength)

        precision_values.append(precision)
        recall_values.append(recall)
        labels.append(f"{config[0]} patterns, {config[1]} extractions")

        # Store remark if available
        if "remark" in eval_data:
            remarks.append(
                (results["precision"], results["recall"], eval_data["remark"])
            )

    if not precision_values:
        print(f"Skipping {input_file}: No valid precision-recall data.")
        continue

    # Scatter plot with distinct colors per configuration
    for i in range(len(precision_values)):
        plt.scatter(
            recall_values[i],
            precision_values[i],
            color=color_map[
                (
                    evaluations[i]["nr_of_patterns"],
                    evaluations[i]["max_nr_of_extractions"],
                )
            ],
            label=labels[i] if labels[i] not in labels[:i] else "",
            alpha=0.7,
            # edgecolors="k",
        )

    # Add remarks as text labels
    for py, rx, remark in remarks:
        plt.text(rx, py, remark, fontsize=9, color="black", alpha=0.75, ha="right")

    # Labels and title
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Plot: {input_file}")

    # **Sorted Legend**
    sorted_configs = sorted(
        color_map.keys(), key=lambda x: (-x[0], -x[1])
    )  # Sort: patterns ↓, extractions ↓
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[config],
            markersize=8,
        )
        for config in sorted_configs
    ]
    legend_labels = [
        f"{config[0]} patterns, max. {config[1]} extraction(s)"
        for config in sorted_configs
    ]
    plt.legend(handles, legend_labels, title="Configurations", loc="lower right")

    # Save plot
    filename = f"prec_rec_{input_file.replace('.','_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Plots saved successfully!")
