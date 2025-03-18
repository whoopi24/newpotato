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
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]

jitter_strength = 0.001

# Initialize min/max values
min_precision, max_precision = float("inf"), float("-inf")
min_recall, max_recall = float("inf"), float("-inf")

# Find global min/max across all evaluations
for evaluations in data_dict.values():
    for eval_data in evaluations:
        results = eval_data["results"]["shg"].get(category, {})
        if "precision" in results and "recall" in results:
            min_precision = min(min_precision, results["precision"])
            max_precision = max(max_precision, results["precision"])
            min_recall = min(min_recall, results["recall"])
            max_recall = max(max_recall, results["recall"])

# Optional: Expand range slightly for better visualization
padding = 0.05  # 5% padding
precision_range = max_precision - min_precision
recall_range = max_recall - min_recall

min_precision -= precision_range * padding
max_precision += precision_range * padding
min_recall -= recall_range * padding
max_recall += recall_range * padding


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
    f1_scores = []
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

    best_f1 = 0
    best_f1_point = None

    for eval_data in evaluations:
        results = eval_data["results"]["shg"].get(category, {})
        if "precision" not in results or "recall" not in results:
            continue

        config = (eval_data["nr_of_patterns"], eval_data["max_nr_of_extractions"])

        # Assign a unique color per configuration
        if config not in color_map:
            color_map[config] = distinct_colors[len(color_map) % len(distinct_colors)]

        precision, recall = results["precision"], results["recall"]

        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1_score)

            # Track best F1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                best_f1_point = (recall, precision)

        # Apply jitter only if the same (precision, recall) pair appears more than once
        if value_counts[(precision, recall)] > 1:
            precision += random.uniform(-jitter_strength, jitter_strength)
            recall += random.uniform(-jitter_strength, jitter_strength)
            best_f1_point = (recall, precision)

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
        )

    # highlight best f1 point
    if best_f1_point:
        plt.text(
            best_f1_point[0],
            best_f1_point[1],
            f".f1: {round(best_f1, 3)}",
            fontsize=10,
            ha="left",
            color="black",
        )

    print(
        f"{input_file}: The maximum f1 score is {round(best_f1, 3)} for point ({round(best_f1_point[0], 3)}, {round(best_f1_point[1], 3)})!"
    )

    # Add remarks as text labels
    for py, rx, remark in remarks:
        plt.text(rx, py, remark, fontsize=9, color="black", alpha=0.75, ha="right")

    # Labels and title
    # plt.xlim(min_recall, max_recall)
    # plt.ylim(min_precision, max_precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Plot: {input_file}")

    # sorted legend
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
    filename = f"prec_rec_{input_file.split('.')[0]}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

print("Plots saved successfully!")
