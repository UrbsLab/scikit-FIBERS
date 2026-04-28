import argparse
import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_category_logic import categorize_three_mode_features


MODE_COLORS = {
    "default": "#5B7DB1",
    "permissive": "#6E88B7",
    "protective": "#4D9960",
    "high_risk": "#C6544D",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FIBERS mode outputs with a compact dashboard.")
    parser.add_argument("--output-root", default="output", help="Root output directory containing per-mode subfolders.")
    parser.add_argument(
        "--modes",
        default="default,high_risk,protective",
        help="Comma-separated list of modes to summarize.",
    )
    parser.add_argument("--top-bin-index", type=int, default=0, help="0-based top-bin index to summarize from each fold population.")
    parser.add_argument(
        "--top-feature-count",
        type=int,
        default=0,
        help="Optional cap on the number of heatmap features. Use 0 to keep all features that pass --min-feature-count.",
    )
    parser.add_argument(
        "--min-feature-count",
        type=int,
        default=3,
        help="Minimum recurrence in at least one mode for a feature to appear in the heatmap.",
    )
    parser.add_argument(
        "--core-shared-min",
        type=int,
        default=4,
        help="Minimum count in all 3 modes for a feature to be labeled core shared in the heatmap.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Directory for outputs. Defaults to <output-root>/comparison_figures.",
    )
    parser.add_argument("--prefix", default="mode_summary_dashboard", help="Output filename prefix.")
    return parser.parse_args()


def normalize_feature_name(feature_name):
    if feature_name.startswith("MM_"):
        return feature_name.replace("MM_", "", 1)
    return feature_name


def load_mode_data(output_root, modes, top_bin_index):
    summary_rows = []
    feature_counter_by_mode = {}
    feature_presence_by_mode = {}

    for mode in modes:
        mode_path = output_root / mode
        feature_counter = Counter()
        feature_presence = {}
        fold_dirs = sorted([path for path in mode_path.iterdir() if path.is_dir()], key=lambda path: int(path.name))

        for fold_dir in fold_dirs:
            fold = int(fold_dir.name)
            pop_path = fold_dir / f"{fold}_pop.csv"
            cox_path = fold_dir / f"{fold}_coxph_unadj_bin_test_{top_bin_index}.csv"

            pop_df = pd.read_csv(pop_path)
            top_bin = pop_df.iloc[top_bin_index]
            feature_list = [normalize_feature_name(feature_name) for feature_name in ast.literal_eval(top_bin["feature_list"])]
            feature_counter.update(feature_list)
            feature_presence[fold] = set(feature_list)

            cox_df = pd.read_csv(cox_path)
            cox_row = cox_df.iloc[0]

            summary_rows.append(
                {
                    "mode": mode,
                    "fold": fold,
                    "feature_list": feature_list,
                    "threshold": int(top_bin["group_threshold"]),
                    "bin_size": int(top_bin["bin_size"]),
                    "pre_fitness": float(top_bin["pre_fitness"]),
                    "log_rank_score": float(top_bin["log_rank_score"]),
                    "group_strata_prop": float(top_bin["group_strata_prop"]),
                    "count_bt": int(top_bin["count_bt"]),
                    "count_at": int(top_bin["count_at"]),
                    "test_coef": float(cox_row["coef"]),
                    "test_hr": float(cox_row["exp(coef)"]),
                    "test_p": float(cox_row["p"]),
                }
            )

        feature_counter_by_mode[mode] = feature_counter
        feature_presence_by_mode[mode] = feature_presence

    return pd.DataFrame(summary_rows), feature_counter_by_mode, feature_presence_by_mode


def build_feature_count_table(feature_counter_by_mode, modes, top_feature_count, min_feature_count, core_shared_min):
    all_features = sorted(set().union(*[set(counter.keys()) for counter in feature_counter_by_mode.values()]))

    if len(modes) == 3:
        section_rows, feature_rows = categorize_three_mode_features(
            feature_counter_by_mode,
            modes,
            core_shared_min,
            min_feature_count,
        )
    else:
        filtered_features = [
            feature_name
            for feature_name in all_features
            if max(feature_counter_by_mode[mode].get(feature_name, 0) for mode in modes) >= min_feature_count
        ]
        filtered_features = sorted(
            filtered_features,
            key=lambda feature_name: (
                max(feature_counter_by_mode[mode].get(feature_name, 0) for mode in modes),
                sum(feature_counter_by_mode[mode].get(feature_name, 0) for mode in modes),
                feature_name,
            ),
            reverse=True,
        )
        section_rows = [("Recurrent features", filtered_features)]

        feature_rows = []
        for feature_name in filtered_features:
            row = {
                "section": "Recurrent features",
                "feature": feature_name,
                "category_rule": f"At least one mode recurs in at least {min_feature_count} folds.",
                "category_reason": (
                    f"{feature_name} reaches "
                    + ", ".join(
                        [
                            f"{feature_counter_by_mode[mode].get(feature_name, 0)} {mode} folds"
                            for mode in modes
                        ]
                    )
                    + "."
                ),
            }
            for mode in modes:
                row[mode] = feature_counter_by_mode[mode].get(feature_name, 0)
            feature_rows.append(row)

    section_order = {section_name: index for index, (section_name, _) in enumerate(section_rows)}
    feature_rank = {}
    for section_name, feature_names in section_rows:
        for rank, feature_name in enumerate(feature_names):
            feature_rank[(section_name, feature_name)] = rank

    feature_count_df = pd.DataFrame(feature_rows)
    if len(feature_count_df) > 0:
        feature_count_df["section_order"] = feature_count_df["section"].map(section_order)
        feature_count_df["feature_rank"] = feature_count_df.apply(
            lambda row: feature_rank[(row["section"], row["feature"])],
            axis=1,
        )
        feature_count_df = feature_count_df.sort_values(["section_order", "feature_rank"]).drop(
            columns=["section_order", "feature_rank"]
        )
    if top_feature_count > 0:
        feature_count_df = feature_count_df.head(top_feature_count).copy()
    return feature_count_df


def build_jaccard_matrix(feature_counter_by_mode):
    modes = list(feature_counter_by_mode.keys())
    unique_feature_sets = {mode: set(counter.keys()) for mode, counter in feature_counter_by_mode.items()}
    matrix = np.zeros((len(modes), len(modes)))

    for i, mode_a in enumerate(modes):
        for j, mode_b in enumerate(modes):
            union = unique_feature_sets[mode_a] | unique_feature_sets[mode_b]
            intersection = unique_feature_sets[mode_a] & unique_feature_sets[mode_b]
            matrix[i, j] = len(intersection) / len(union) if union else 1.0

    return modes, matrix


def write_summary_outputs(summary_df, feature_count_df, jaccard_modes, jaccard_matrix, figure_dir, prefix):
    summary_csv_path = figure_dir / f"{prefix}.csv"
    summary_df.sort_values(["mode", "fold"]).to_csv(summary_csv_path, index=False)

    feature_csv_path = figure_dir / f"{prefix}_feature_counts.csv"
    feature_count_df.assign(section=feature_count_df["section"].str.replace("\n", " ", regex=False)).to_csv(feature_csv_path, index=False)

    jaccard_csv_path = figure_dir / f"{prefix}_jaccard.csv"
    pd.DataFrame(jaccard_matrix, index=jaccard_modes, columns=jaccard_modes).to_csv(jaccard_csv_path)

    mode_stats = summary_df.groupby("mode").agg(
        top_bin_pre_fitness_mean=("pre_fitness", "mean"),
        top_bin_pre_fitness_median=("pre_fitness", "median"),
        top_bin_log_rank_mean=("log_rank_score", "mean"),
        top_bin_group_strata_mean=("group_strata_prop", "mean"),
        top_bin_test_hr_mean=("test_hr", "mean"),
        top_bin_threshold_median=("threshold", "median"),
        top_bin_size_mean=("bin_size", "mean"),
    )

    report_lines = [
        "FIBERS mode summary dashboard",
        "",
        "Heatmap category rules:",
    ]

    if "category_rule" in feature_count_df.columns and len(feature_count_df) > 0:
        for section_name, group_df in feature_count_df.groupby("section", sort=False):
            report_lines.append(
                f"- {section_name.replace(chr(10), ' ')}: {group_df['category_rule'].iloc[0]}"
            )

    report_lines.extend(
        [
            "",
        "Mode-level observations:",
        ]
    )

    if "default" in jaccard_modes and "permissive" in jaccard_modes:
        i = jaccard_modes.index("default")
        j = jaccard_modes.index("permissive")
        report_lines.append(
            f"- Default vs permissive top-feature Jaccard overlap: {jaccard_matrix[i, j]:.3f}."
        )
    if "default" in jaccard_modes and "high_risk" in jaccard_modes:
        i = jaccard_modes.index("default")
        j = jaccard_modes.index("high_risk")
        report_lines.append(
            f"- Default vs high_risk top-feature Jaccard overlap: {jaccard_matrix[i, j]:.3f}."
        )
    if "default" in jaccard_modes and "protective" in jaccard_modes:
        i = jaccard_modes.index("default")
        j = jaccard_modes.index("protective")
        report_lines.append(
            f"- Default vs protective top-feature Jaccard overlap: {jaccard_matrix[i, j]:.3f}."
        )

    for mode in mode_stats.index:
        stats = mode_stats.loc[mode]
        report_lines.append(
            "- "
            + f"{mode}: mean test HR={stats['top_bin_test_hr_mean']:.3f}, "
            + f"median threshold={stats['top_bin_threshold_median']:.1f}, "
            + f"mean bin size={stats['top_bin_size_mean']:.1f}, "
            + f"mean group strata={stats['top_bin_group_strata_mean']:.3f}, "
            + f"mean pre-fitness={stats['top_bin_pre_fitness_mean']:.2f}."
        )

    report_path = figure_dir / f"{prefix}_inferences.txt"
    report_path.write_text("\n".join(report_lines))


def add_value_labels(ax, x_positions, values):
    for x_position, value in zip(x_positions, values):
        ax.text(x_position, value + 0.2, str(int(value)), ha="center", va="bottom", fontsize=9)


def plot_dashboard(summary_df, feature_count_df, jaccard_modes, jaccard_matrix, figure_dir, prefix):
    modes = list(summary_df["mode"].drop_duplicates())
    mode_positions = np.arange(len(modes))
    mode_colors = [MODE_COLORS.get(mode, "#666666") for mode in modes]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.05, 1.15, 1.0], height_ratios=[1.0, 1.0, 1.0])

    ax_jaccard = fig.add_subplot(grid[0, 0])
    ax_strata = fig.add_subplot(grid[1, 0])
    ax_threshold = fig.add_subplot(grid[2, 0])
    ax_features = fig.add_subplot(grid[:, 1])
    ax_size = fig.add_subplot(grid[0, 2])
    ax_hr = fig.add_subplot(grid[1, 2])
    ax_fitness = fig.add_subplot(grid[2, 2])

    heatmap = ax_jaccard.imshow(jaccard_matrix, cmap="Blues", vmin=0, vmax=1)
    ax_jaccard.set_xticks(np.arange(len(jaccard_modes)))
    ax_jaccard.set_xticklabels([mode.replace("_", "\n") for mode in jaccard_modes], fontsize=10)
    ax_jaccard.set_yticks(np.arange(len(jaccard_modes)))
    ax_jaccard.set_yticklabels([mode.replace("_", " ") for mode in jaccard_modes], fontsize=10)
    ax_jaccard.set_title("Top-Feature Jaccard", fontsize=14, weight="bold")
    for i in range(len(jaccard_modes)):
        for j in range(len(jaccard_modes)):
            ax_jaccard.text(j, i, f"{jaccard_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(heatmap, ax=ax_jaccard, fraction=0.046, pad=0.04)

    feature_counts = feature_count_df[modes].to_numpy()
    feature_heatmap = ax_features.imshow(feature_counts, cmap="YlGnBu", vmin=0, vmax=10, aspect="auto")
    ax_features.set_xticks(np.arange(len(modes)))
    ax_features.set_xticklabels([mode.replace("_", "\n") for mode in modes], fontsize=10)
    feature_font_size = 9 if len(feature_count_df) <= 26 else 8 if len(feature_count_df) <= 40 else 7
    ax_features.set_yticks(np.arange(len(feature_count_df)))
    ax_features.set_yticklabels(feature_count_df["feature"].tolist(), fontsize=feature_font_size)
    ax_features.set_title("Recurrent Top-Bin Features by Shared/Mode Category", fontsize=14, weight="bold")
    for row_index in range(feature_counts.shape[0]):
        for col_index in range(feature_counts.shape[1]):
            ax_features.text(col_index, row_index, int(feature_counts[row_index, col_index]), ha="center", va="center", fontsize=8)
    fig.colorbar(feature_heatmap, ax=ax_features, fraction=0.035, pad=0.02)

    section_start = 0
    section_font_size = 11
    for section_name, group_df in feature_count_df.groupby("section", sort=False):
        section_end = section_start + len(group_df)
        section_center = (section_start + section_end - 1) / 2.0
        ax_features.text(
            -1.35,
            section_center,
            section_name,
            ha="right",
            va="center",
            fontsize=section_font_size,
            color="#4B5563",
            fontweight="bold",
            clip_on=False,
        )
        if section_end < len(feature_count_df):
            ax_features.hlines(section_end - 0.5, -0.5, len(modes) - 0.5, color="#CFC6B8", linewidth=1.5)
        section_start = section_end

    for x_position, mode in enumerate(modes):
        size_values = summary_df.loc[summary_df["mode"] == mode, "bin_size"].to_numpy()
        size_jitter = np.linspace(-0.12, 0.12, len(size_values))
        ax_size.scatter(
            np.full(len(size_values), x_position) + size_jitter,
            size_values,
            color=MODE_COLORS.get(mode, "#666666"),
            edgecolor="white",
            linewidth=0.4,
            s=34,
            alpha=0.9,
        )
        ax_size.hlines(np.median(size_values), x_position - 0.22, x_position + 0.22, color="black", linewidth=1.2)
    ax_size.set_xticks(mode_positions)
    ax_size.set_xticklabels([mode.replace("_", "\n") for mode in modes], fontsize=10)
    ax_size.set_ylabel("Features", fontsize=10)
    ax_size.set_title("Top-Bin Size", fontsize=14, weight="bold")
    ax_size.grid(axis="y", alpha=0.2)

    for ax, column_name, title, y_label, log_scale in [
        (ax_hr, "test_hr", "Top-Bin Test HR", "HR", False),
        (ax_fitness, "pre_fitness", "Top-Bin Pre-Fitness", "Pre-fitness", True),
    ]:
        for x_position, mode in enumerate(modes):
            values = summary_df.loc[summary_df["mode"] == mode, column_name].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(
                np.full(len(values), x_position) + jitter,
                values,
                color=MODE_COLORS.get(mode, "#666666"),
                edgecolor="white",
                linewidth=0.4,
                s=34,
                alpha=0.9,
            )
            ax.hlines(np.median(values), x_position - 0.22, x_position + 0.22, color="black", linewidth=1.2)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(mode_positions)
        ax.set_xticklabels([mode.replace("_", "\n") for mode in modes], fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.2)

    for x_position, mode in enumerate(modes):
        strata_values = summary_df.loc[summary_df["mode"] == mode, "group_strata_prop"].to_numpy()
        strata_jitter = np.linspace(-0.12, 0.12, len(strata_values))
        ax_strata.scatter(
            np.full(len(strata_values), x_position) + strata_jitter,
            strata_values,
            color=MODE_COLORS.get(mode, "#666666"),
            edgecolor="white",
            linewidth=0.4,
            s=34,
            alpha=0.9,
        )
        ax_strata.hlines(np.median(strata_values), x_position - 0.22, x_position + 0.22, color="black", linewidth=1.2)
    ax_strata.set_xticks(mode_positions)
    ax_strata.set_xticklabels([mode.replace("_", "\n") for mode in modes], fontsize=10)
    ax_strata.set_ylabel("Group strata", fontsize=10)
    ax_strata.set_ylim(0, 0.55)
    ax_strata.set_title("Top-Bin Group Strata", fontsize=12, weight="bold")
    ax_strata.grid(axis="y", alpha=0.2)

    for x_position, mode in enumerate(modes):
        threshold_values = summary_df.loc[summary_df["mode"] == mode, "threshold"].to_numpy()
        threshold_jitter = np.linspace(-0.12, 0.12, len(threshold_values))
        ax_threshold.scatter(
            np.full(len(threshold_values), x_position) + threshold_jitter,
            threshold_values,
            color=MODE_COLORS.get(mode, "#666666"),
            edgecolor="white",
            linewidth=0.4,
            s=34,
            alpha=0.9,
        )
        ax_threshold.hlines(np.median(threshold_values), x_position - 0.22, x_position + 0.22, color="black", linewidth=1.2)
    ax_threshold.set_xticks(mode_positions)
    ax_threshold.set_xticklabels([mode.replace("_", "\n") for mode in modes], fontsize=10)
    ax_threshold.set_ylabel("Threshold", fontsize=10)
    ax_threshold.set_title("Top-Bin Threshold", fontsize=12, weight="bold")
    ax_threshold.grid(axis="y", alpha=0.2)

    fig.suptitle("FIBERS Mode Summary from Top Bin per CV Fold", fontsize=17, weight="bold")
    figure_path = figure_dir / f"{prefix}.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    figure_dir = Path(args.figure_dir).resolve() if args.figure_dir else output_root / "comparison_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_df, feature_counter_by_mode, _ = load_mode_data(output_root, modes, args.top_bin_index)
    feature_count_df = build_feature_count_table(
        feature_counter_by_mode,
        modes,
        args.top_feature_count,
        args.min_feature_count,
        args.core_shared_min,
    )
    jaccard_modes, jaccard_matrix = build_jaccard_matrix(feature_counter_by_mode)

    write_summary_outputs(summary_df, feature_count_df, jaccard_modes, jaccard_matrix, figure_dir, args.prefix)
    plot_dashboard(summary_df, feature_count_df, jaccard_modes, jaccard_matrix, figure_dir, args.prefix)


if __name__ == "__main__":
    main()
