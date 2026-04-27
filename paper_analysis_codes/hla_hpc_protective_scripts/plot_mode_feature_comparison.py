import argparse
import ast
import os
import tempfile
from collections import Counter

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


LOCUS_COLORS = {
    "A": "#6AA84F",
    "B": "#CC4125",
    "C": "#674EA7",
    "DQA1": "#B47E24",
    "DQB1": "#2F5597",
    "DRB1": "#3D7F74",
    "DRB345": "#8D5A3B",
}

MODE_HEADER_COLORS = {
    "default": "#5477A8",
    "protective": "#4C8F5A",
    "permissive": "#8A6BBE",
}

COUNT_LABELS = {
    "default": "D",
    "protective": "P",
    "permissive": "Pm",
}

BACKGROUND_COLOR = "#F5F1E8"
EMPTY_CELL_COLOR = "#ECE7DD"
GRID_EDGE_COLOR = "#D8D0C4"
TEXT_COLOR = "#1F2430"
SECTION_LINE_COLOR = "#CFC6B8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot screenshot-style feature presence comparisons across FIBERS modes."
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        type=str,
        required=True,
        help="Folder containing mode subfolders such as default/, protective/, permissive/.",
    )
    parser.add_argument(
        "--compare-modes",
        dest="compare_modes",
        type=str,
        default="protective,permissive",
        help="Comma-separated modes to compare against default.",
    )
    parser.add_argument(
        "--reference-mode",
        dest="reference_mode",
        type=str,
        default="default",
        help="Reference mode. Default is 'default'.",
    )
    parser.add_argument(
        "--top-bin-count",
        dest="top_bin_count",
        type=int,
        default=1,
        help="Union features from the top N bins per fold. Default is 1.",
    )
    parser.add_argument(
        "--core-shared-min",
        dest="core_shared_min",
        type=int,
        default=3,
        help="Minimum count in both modes for a feature to be labeled core shared.",
    )
    parser.add_argument(
        "--cv-order",
        dest="cv_order",
        type=str,
        default=None,
        help="Optional comma-separated CV labels to force column order. Example: 1,2,3,4,5,6,7,8,9,10",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default=None,
        help="Where to save figures. Defaults to <output-root>/comparison_figures",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        default="",
        help="Optional filename prefix for saved outputs.",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def mode_title(mode_name):
    return mode_name.replace("_", " ").title()


def short_mode_label(mode_name):
    return COUNT_LABELS.get(mode_name, mode_name[:2].title())


def normalize_feature_name(feature_name):
    if feature_name.startswith("MM_"):
        return feature_name[3:]
    return feature_name


def detect_locus(feature_name):
    short_name = normalize_feature_name(feature_name)
    for locus in sorted(LOCUS_COLORS.keys(), key=len, reverse=True):
        if short_name.startswith(locus + "_") or short_name == locus:
            return locus
    return "Other"


def feature_color(feature_name):
    return LOCUS_COLORS.get(detect_locus(feature_name), "#6D6D6D")


def parse_feature_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, list):
        return parsed
    raise ValueError("feature_list column must evaluate to a list")


def get_mode_fold_pop_files(output_root, mode_name):
    mode_path = os.path.join(output_root, mode_name)
    if not os.path.isdir(mode_path):
        raise FileNotFoundError(f"Mode folder not found: {mode_path}")

    fold_files = {}
    for fold_name in os.listdir(mode_path):
        fold_path = os.path.join(mode_path, fold_name)
        if not os.path.isdir(fold_path):
            continue
        pop_files = [
            filename
            for filename in os.listdir(fold_path)
            if filename.endswith("_pop.csv")
        ]
        if len(pop_files) == 0:
            continue
        pop_files.sort()
        fold_files[fold_name] = os.path.join(fold_path, pop_files[0])

    if len(fold_files) == 0:
        raise FileNotFoundError(f"No *_pop.csv files found under: {mode_path}")

    return fold_files


def sort_cv_labels(cv_labels):
    def sort_key(value):
        if str(value).isdigit():
            return (0, int(value))
        return (1, str(value))

    return sorted(cv_labels, key=sort_key)


def load_fold_features(pop_file, top_bin_count):
    pop_df = pd.read_csv(pop_file, low_memory=False)
    if "feature_list" not in pop_df.columns:
        raise KeyError(f"'feature_list' column missing in {pop_file}")

    selected_rows = pop_df.head(max(top_bin_count, 1))
    selected_features = set()
    for raw_feature_list in selected_rows["feature_list"]:
        for feature_name in parse_feature_list(raw_feature_list):
            selected_features.add(normalize_feature_name(feature_name))
    return selected_features


def build_mode_feature_summary(output_root, reference_mode, compare_mode, top_bin_count, cv_order):
    reference_fold_files = get_mode_fold_pop_files(output_root, reference_mode)
    compare_fold_files = get_mode_fold_pop_files(output_root, compare_mode)

    if cv_order is None:
        cv_labels = sort_cv_labels(set(reference_fold_files.keys()) & set(compare_fold_files.keys()))
    else:
        cv_labels = [label.strip() for label in cv_order.split(",") if label.strip() != ""]

    if len(cv_labels) == 0:
        raise ValueError("No overlapping CV folders found between the requested modes.")

    reference_presence = {}
    compare_presence = {}
    for cv_label in cv_labels:
        if cv_label not in reference_fold_files:
            raise FileNotFoundError(f"Missing reference fold {cv_label} in mode {reference_mode}")
        if cv_label not in compare_fold_files:
            raise FileNotFoundError(f"Missing compare fold {cv_label} in mode {compare_mode}")

        reference_presence[cv_label] = load_fold_features(reference_fold_files[cv_label], top_bin_count)
        compare_presence[cv_label] = load_fold_features(compare_fold_files[cv_label], top_bin_count)

    reference_counts = Counter()
    compare_counts = Counter()
    for cv_label in cv_labels:
        reference_counts.update(reference_presence[cv_label])
        compare_counts.update(compare_presence[cv_label])

    return cv_labels, reference_presence, compare_presence, reference_counts, compare_counts


def categorize_features(reference_counts, compare_counts, compare_mode, core_shared_min):
    all_features = sorted(set(reference_counts.keys()) | set(compare_counts.keys()))

    core_shared = []
    compare_skewed_shared = []
    compare_only = []

    for feature_name in all_features:
        reference_count = reference_counts.get(feature_name, 0)
        compare_count = compare_counts.get(feature_name, 0)

        if compare_count == 0:
            continue

        if reference_count >= core_shared_min and compare_count >= core_shared_min:
            core_shared.append(feature_name)
        elif reference_count == 0:
            compare_only.append(feature_name)
        elif compare_count > reference_count:
            compare_skewed_shared.append(feature_name)

    core_shared = sort_group(core_shared, reference_counts, compare_counts)
    compare_skewed_shared = sort_group(compare_skewed_shared, reference_counts, compare_counts)
    compare_only = sort_group(compare_only, reference_counts, compare_counts)

    section_rows = []
    if len(core_shared) > 0:
        section_rows.append(("Core shared", core_shared))
    if len(compare_skewed_shared) > 0:
        section_rows.append((f"{mode_title(compare_mode)}-skewed\nshared", compare_skewed_shared))
    if len(compare_only) > 0:
        section_rows.append((f"{mode_title(compare_mode)} only", compare_only))

    return section_rows


def sort_group(feature_names, reference_counts, compare_counts):
    return sorted(
        feature_names,
        key=lambda feature_name: (
            max(reference_counts.get(feature_name, 0), compare_counts.get(feature_name, 0)),
            compare_counts.get(feature_name, 0),
            reference_counts.get(feature_name, 0),
            feature_name,
        ),
        reverse=True,
    )


def build_feature_summary_dataframe(section_rows, cv_labels, reference_presence, compare_presence, reference_counts, compare_counts):
    rows = []
    for section_name, feature_names in section_rows:
        for feature_name in feature_names:
            row = {
                "section": section_name.replace("\n", " "),
                "feature": feature_name,
                "reference_count": reference_counts.get(feature_name, 0),
                "compare_count": compare_counts.get(feature_name, 0),
            }
            for cv_label in cv_labels:
                row[f"reference_cv_{cv_label}"] = int(feature_name in reference_presence[cv_label])
                row[f"compare_cv_{cv_label}"] = int(feature_name in compare_presence[cv_label])
            rows.append(row)
    return pd.DataFrame(rows)


def draw_count_badge(ax, x_center, y_center, value, facecolor):
    ax.text(
        x_center,
        y_center,
        str(value),
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=facecolor, edgecolor="none"),
    )


def draw_feature_comparison_figure(
    section_rows,
    cv_labels,
    reference_mode,
    compare_mode,
    reference_presence,
    compare_presence,
    reference_counts,
    compare_counts,
    save_path,
    show,
):
    flattened_features = [feature_name for _, feature_names in section_rows for feature_name in feature_names]
    if len(flattened_features) == 0:
        raise ValueError(f"No features to plot for {reference_mode} vs {compare_mode}.")

    nrows = len(flattened_features)
    cv_count = len(cv_labels)

    left_label_width = 6.5
    feature_label_width = 3.7
    cell_size = 1.0
    panel_gap = 1.0
    count_gap = 1.8
    count_width = 1.2
    right_margin = 0.8
    header_height = 2.4
    legend_height = 2.2

    ref_panel_x = left_label_width + feature_label_width
    compare_panel_x = ref_panel_x + cv_count * cell_size + panel_gap
    ref_count_x = compare_panel_x + cv_count * cell_size + count_gap
    compare_count_x = ref_count_x + count_width + 0.9

    total_width = compare_count_x + count_width + right_margin
    total_height = header_height + nrows + legend_height

    fig_width = max(12, total_width * 0.52)
    fig_height = max(7.5, total_height * 0.42)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    reference_header_color = MODE_HEADER_COLORS.get(reference_mode, "#5477A8")
    compare_header_color = MODE_HEADER_COLORS.get(compare_mode, "#4C8F5A")

    header_y = 0.35
    header_box_height = 0.85
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (ref_panel_x, header_y),
            cv_count * cell_size,
            header_box_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=reference_header_color,
            edgecolor="none",
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (compare_panel_x, header_y),
            cv_count * cell_size,
            header_box_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=compare_header_color,
            edgecolor="none",
        )
    )

    ax.text(
        ref_panel_x + (cv_count * cell_size) / 2.0,
        header_y + header_box_height / 2.0,
        mode_title(reference_mode),
        ha="center",
        va="center",
        color="white",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        compare_panel_x + (cv_count * cell_size) / 2.0,
        header_y + header_box_height / 2.0,
        mode_title(compare_mode),
        ha="center",
        va="center",
        color="white",
        fontsize=20,
        fontweight="bold",
    )

    cv_label_y = 1.6
    for cv_index, cv_label in enumerate(cv_labels):
        ax.text(
            ref_panel_x + (cv_index + 0.5) * cell_size,
            cv_label_y,
            f"CV{cv_label}",
            ha="center",
            va="center",
            fontsize=11,
            color="#6C7280",
        )
        ax.text(
            compare_panel_x + (cv_index + 0.5) * cell_size,
            cv_label_y,
            f"CV{cv_label}",
            ha="center",
            va="center",
            fontsize=11,
            color="#6C7280",
        )

    ax.text(
        ref_count_x + count_width / 2.0,
        header_y + header_box_height / 2.0,
        short_mode_label(reference_mode),
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        compare_count_x + count_width / 2.0,
        header_y + header_box_height / 2.0,
        short_mode_label(compare_mode),
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
    )

    row_y_start = header_height
    row_index = 0
    section_boundaries = []

    for section_name, feature_names in section_rows:
        section_start = row_index
        for feature_name in feature_names:
            locus_color = feature_color(feature_name)
            y = row_y_start + row_index

            ax.text(
                left_label_width + feature_label_width - 0.25,
                y + 0.5,
                feature_name,
                ha="right",
                va="center",
                fontsize=20,
                color=locus_color,
            )

            for cv_index, cv_label in enumerate(cv_labels):
                ref_present = feature_name in reference_presence[cv_label]
                compare_present = feature_name in compare_presence[cv_label]

                ax.add_patch(
                    mpatches.Rectangle(
                        (ref_panel_x + cv_index * cell_size, y),
                        0.9,
                        0.9,
                        facecolor=locus_color if ref_present else EMPTY_CELL_COLOR,
                        edgecolor=GRID_EDGE_COLOR,
                        linewidth=1.0,
                    )
                )
                ax.add_patch(
                    mpatches.Rectangle(
                        (compare_panel_x + cv_index * cell_size, y),
                        0.9,
                        0.9,
                        facecolor=locus_color if compare_present else EMPTY_CELL_COLOR,
                        edgecolor=GRID_EDGE_COLOR,
                        linewidth=1.0,
                    )
                )

            draw_count_badge(
                ax,
                ref_count_x + count_width / 2.0,
                y + 0.45,
                reference_counts.get(feature_name, 0),
                reference_header_color,
            )
            draw_count_badge(
                ax,
                compare_count_x + count_width / 2.0,
                y + 0.45,
                compare_counts.get(feature_name, 0),
                compare_header_color,
            )

            row_index += 1

        section_end = row_index
        section_boundaries.append((section_name, section_start, section_end))

    for section_name, section_start, section_end in section_boundaries:
        section_center_y = row_y_start + (section_start + section_end) / 2.0
        ax.text(
            0.55,
            section_center_y,
            section_name,
            ha="left",
            va="center",
            fontsize=21,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    for _, _, section_end in section_boundaries[:-1]:
        divider_y = row_y_start + section_end - 0.1
        ax.plot(
            [0.35, compare_panel_x + cv_count * cell_size],
            [divider_y, divider_y],
            color=SECTION_LINE_COLOR,
            linewidth=2.0,
        )

    legend_y = row_y_start + nrows + 1.2
    legend_x = 0.7
    for locus_name, color in LOCUS_COLORS.items():
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (legend_x, legend_y - 0.23),
                0.48,
                0.48,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            legend_x + 0.75,
            legend_y,
            locus_name,
            ha="left",
            va="center",
            fontsize=16,
            color="#4B5563",
        )
        legend_x += 2.2

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    args = parse_args()

    if args.top_bin_count < 1:
        raise ValueError("--top-bin-count must be >= 1")

    compare_modes = [mode.strip() for mode in args.compare_modes.split(",") if mode.strip() != ""]
    if len(compare_modes) == 0:
        raise ValueError("At least one compare mode must be specified.")

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(args.output_root, "comparison_figures")
    ensure_directory(save_dir)

    for compare_mode in compare_modes:
        cv_labels, reference_presence, compare_presence, reference_counts, compare_counts = build_mode_feature_summary(
            args.output_root,
            args.reference_mode,
            compare_mode,
            args.top_bin_count,
            args.cv_order,
        )

        section_rows = categorize_features(
            reference_counts,
            compare_counts,
            compare_mode,
            args.core_shared_min,
        )

        if len(section_rows) == 0:
            print(f"Skipping {args.reference_mode} vs {compare_mode}: no features matched the display rules.")
            continue

        filename_stub = f"{args.prefix}{args.reference_mode}_vs_{compare_mode}_top{args.top_bin_count}"
        figure_path = os.path.join(save_dir, filename_stub + "_feature_comparison.png")
        summary_csv_path = os.path.join(save_dir, filename_stub + "_feature_comparison.csv")

        summary_df = build_feature_summary_dataframe(
            section_rows,
            cv_labels,
            reference_presence,
            compare_presence,
            reference_counts,
            compare_counts,
        )
        summary_df.to_csv(summary_csv_path, index=False)

        draw_feature_comparison_figure(
            section_rows,
            cv_labels,
            args.reference_mode,
            compare_mode,
            reference_presence,
            compare_presence,
            reference_counts,
            compare_counts,
            figure_path,
            args.show,
        )

        print(f"Saved figure: {figure_path}")
        print(f"Saved summary: {summary_csv_path}")


if __name__ == "__main__":
    main()
