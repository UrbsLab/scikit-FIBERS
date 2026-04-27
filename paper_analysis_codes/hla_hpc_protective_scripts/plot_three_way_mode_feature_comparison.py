import argparse
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

from plot_mode_feature_comparison import (
    BACKGROUND_COLOR,
    EMPTY_CELL_COLOR,
    GRID_EDGE_COLOR,
    LOCUS_COLORS,
    MODE_HEADER_COLORS,
    SECTION_LINE_COLOR,
    TEXT_COLOR,
    draw_count_badge,
    ensure_directory,
    feature_color,
    get_mode_fold_pop_files,
    load_fold_features,
    mode_title,
    short_mode_label,
    sort_cv_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot screenshot-style feature presence comparisons across three FIBERS modes."
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        type=str,
        required=True,
        help="Folder containing mode subfolders such as default/, high_risk/, protective/.",
    )
    parser.add_argument(
        "--mode-order",
        dest="mode_order",
        type=str,
        default="default,high_risk,protective",
        help="Comma-separated 3-mode order to display. Default is default,high_risk,protective.",
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
        default=4,
        help="Minimum count in all three modes for a feature to be labeled core shared.",
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


def build_three_mode_feature_summary(output_root, mode_order, top_bin_count, cv_order):
    fold_files_by_mode = {
        mode_name: get_mode_fold_pop_files(output_root, mode_name)
        for mode_name in mode_order
    }

    if cv_order is None:
        shared_cv_labels = set(fold_files_by_mode[mode_order[0]].keys())
        for mode_name in mode_order[1:]:
            shared_cv_labels &= set(fold_files_by_mode[mode_name].keys())
        cv_labels = sort_cv_labels(shared_cv_labels)
    else:
        cv_labels = [label.strip() for label in cv_order.split(",") if label.strip() != ""]

    if len(cv_labels) == 0:
        raise ValueError("No overlapping CV folders found across the requested modes.")

    presence_by_mode = {mode_name: {} for mode_name in mode_order}
    counts_by_mode = {mode_name: Counter() for mode_name in mode_order}

    for cv_label in cv_labels:
        for mode_name in mode_order:
            if cv_label not in fold_files_by_mode[mode_name]:
                raise FileNotFoundError(f"Missing fold {cv_label} in mode {mode_name}")
            fold_features = load_fold_features(fold_files_by_mode[mode_name][cv_label], top_bin_count)
            presence_by_mode[mode_name][cv_label] = fold_features
            counts_by_mode[mode_name].update(fold_features)

    return cv_labels, presence_by_mode, counts_by_mode


def sort_group(feature_names, counts_by_mode, primary_mode, secondary_mode=None):
    return sorted(
        feature_names,
        key=lambda feature_name: (
            counts_by_mode[primary_mode].get(feature_name, 0),
            counts_by_mode[secondary_mode].get(feature_name, 0) if secondary_mode is not None else 0,
            feature_name,
        ),
        reverse=True,
    )


def categorize_features_three_way(counts_by_mode, mode_order, core_shared_min):
    reference_mode, compare_mode_a, compare_mode_b = mode_order
    all_features = sorted(set().union(*[set(counter.keys()) for counter in counts_by_mode.values()]))

    core_shared = []
    compare_a_skewed_shared = []
    compare_b_skewed_shared = []
    compare_only_shared = []
    compare_a_only = []
    compare_b_only = []

    for feature_name in all_features:
        reference_count = counts_by_mode[reference_mode].get(feature_name, 0)
        compare_a_count = counts_by_mode[compare_mode_a].get(feature_name, 0)
        compare_b_count = counts_by_mode[compare_mode_b].get(feature_name, 0)

        if (
            reference_count >= core_shared_min
            and compare_a_count >= core_shared_min
            and compare_b_count >= core_shared_min
        ):
            core_shared.append(feature_name)
        elif reference_count > 0 and (compare_a_count > 0 or compare_b_count > 0):
            if compare_a_count >= compare_b_count:
                compare_a_skewed_shared.append(feature_name)
            else:
                compare_b_skewed_shared.append(feature_name)
        elif compare_a_count > 0 and compare_b_count > 0:
            compare_only_shared.append(feature_name)
        elif compare_a_count > 0:
            compare_a_only.append(feature_name)
        elif compare_b_count > 0:
            compare_b_only.append(feature_name)

    section_rows = []
    if len(core_shared) > 0:
        section_rows.append(
            (
                "Core shared",
                sort_group(core_shared, counts_by_mode, compare_mode_b, compare_mode_a),
            )
        )
    if len(compare_a_skewed_shared) > 0:
        section_rows.append(
            (
                f"{mode_title(compare_mode_a)}-skewed\nshared",
                sort_group(compare_a_skewed_shared, counts_by_mode, compare_mode_a, reference_mode),
            )
        )
    if len(compare_b_skewed_shared) > 0:
        section_rows.append(
            (
                f"{mode_title(compare_mode_b)}-skewed\nshared",
                sort_group(compare_b_skewed_shared, counts_by_mode, compare_mode_b, reference_mode),
            )
        )
    if len(compare_only_shared) > 0:
        section_rows.append(
            (
                f"{mode_title(compare_mode_a)} / {mode_title(compare_mode_b)}\nshared",
                sort_group(compare_only_shared, counts_by_mode, compare_mode_b, compare_mode_a),
            )
        )
    if len(compare_a_only) > 0:
        section_rows.append(
            (
                f"{mode_title(compare_mode_a)} only",
                sort_group(compare_a_only, counts_by_mode, compare_mode_a),
            )
        )
    if len(compare_b_only) > 0:
        section_rows.append(
            (
                f"{mode_title(compare_mode_b)} only",
                sort_group(compare_b_only, counts_by_mode, compare_mode_b),
            )
        )

    return section_rows


def build_feature_summary_dataframe(section_rows, cv_labels, mode_order, presence_by_mode, counts_by_mode):
    rows = []
    for section_name, feature_names in section_rows:
        for feature_name in feature_names:
            row = {
                "section": section_name.replace("\n", " "),
                "feature": feature_name,
            }
            for mode_name in mode_order:
                row[f"{mode_name}_count"] = counts_by_mode[mode_name].get(feature_name, 0)
                for cv_label in cv_labels:
                    row[f"{mode_name}_cv_{cv_label}"] = int(feature_name in presence_by_mode[mode_name][cv_label])
            rows.append(row)
    return pd.DataFrame(rows)


def draw_three_way_feature_comparison_figure(
    section_rows,
    cv_labels,
    mode_order,
    presence_by_mode,
    counts_by_mode,
    save_path,
    show,
):
    flattened_features = [feature_name for _, feature_names in section_rows for feature_name in feature_names]
    if len(flattened_features) == 0:
        raise ValueError("No features to plot for the requested three-mode comparison.")

    nrows = len(flattened_features)
    cv_count = len(cv_labels)

    left_label_width = 6.5
    feature_label_width = 3.7
    cell_size = 1.0
    panel_gap = 0.9
    count_gap = 1.8
    count_width = 1.2
    right_margin = 0.8
    header_height = 2.4
    legend_height = 2.2

    panel_start_x = left_label_width + feature_label_width
    panel_x_positions = []
    current_x = panel_start_x
    for _ in mode_order:
        panel_x_positions.append(current_x)
        current_x += cv_count * cell_size + panel_gap

    count_x_positions = []
    current_x += count_gap - panel_gap
    for _ in mode_order:
        count_x_positions.append(current_x)
        current_x += count_width + 0.9

    total_width = current_x + right_margin - 0.9
    total_height = header_height + nrows + legend_height

    fig_width = max(15, total_width * 0.47)
    fig_height = max(8, total_height * 0.42)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    header_y = 0.35
    header_box_height = 0.85
    for mode_name, panel_x in zip(mode_order, panel_x_positions):
        header_color = MODE_HEADER_COLORS.get(mode_name, "#5477A8")
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (panel_x, header_y),
                cv_count * cell_size,
                header_box_height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=header_color,
                edgecolor="none",
            )
        )
        ax.text(
            panel_x + (cv_count * cell_size) / 2.0,
            header_y + header_box_height / 2.0,
            mode_title(mode_name),
            ha="center",
            va="center",
            color="white",
            fontsize=18,
            fontweight="bold",
        )

    cv_label_y = 1.6
    for panel_x in panel_x_positions:
        for cv_index, cv_label in enumerate(cv_labels):
            ax.text(
                panel_x + (cv_index + 0.5) * cell_size,
                cv_label_y,
                f"CV{cv_label}",
                ha="center",
                va="center",
                fontsize=11,
                color="#6C7280",
            )

    for mode_name, count_x in zip(mode_order, count_x_positions):
        ax.text(
            count_x + count_width / 2.0,
            header_y + header_box_height / 2.0,
            short_mode_label(mode_name),
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=15,
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

            for mode_name, panel_x in zip(mode_order, panel_x_positions):
                for cv_index, cv_label in enumerate(cv_labels):
                    mode_present = feature_name in presence_by_mode[mode_name][cv_label]
                    ax.add_patch(
                        mpatches.Rectangle(
                            (panel_x + cv_index * cell_size, y),
                            0.9,
                            0.9,
                            facecolor=locus_color if mode_present else EMPTY_CELL_COLOR,
                            edgecolor=GRID_EDGE_COLOR,
                            linewidth=1.0,
                        )
                    )

            for mode_name, count_x in zip(mode_order, count_x_positions):
                draw_count_badge(
                    ax,
                    count_x + count_width / 2.0,
                    y + 0.45,
                    counts_by_mode[mode_name].get(feature_name, 0),
                    MODE_HEADER_COLORS.get(mode_name, "#5477A8"),
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
            fontsize=20,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    final_panel_end = panel_x_positions[-1] + cv_count * cell_size
    for _, _, section_end in section_boundaries[:-1]:
        divider_y = row_y_start + section_end - 0.1
        ax.plot(
            [0.35, final_panel_end],
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


def main():
    args = parse_args()

    if args.top_bin_count < 1:
        raise ValueError("--top-bin-count must be >= 1")

    mode_order = [mode.strip() for mode in args.mode_order.split(",") if mode.strip() != ""]
    if len(mode_order) != 3:
        raise ValueError("--mode-order must include exactly 3 comma-separated modes.")

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(args.output_root, "comparison_figures")
    ensure_directory(save_dir)

    cv_labels, presence_by_mode, counts_by_mode = build_three_mode_feature_summary(
        args.output_root,
        mode_order,
        args.top_bin_count,
        args.cv_order,
    )

    section_rows = categorize_features_three_way(
        counts_by_mode,
        mode_order,
        args.core_shared_min,
    )

    if len(section_rows) == 0:
        raise ValueError("No features matched the display rules for the requested modes.")

    filename_stub = f"{args.prefix}{'_vs_'.join(mode_order)}_top{args.top_bin_count}"
    figure_path = os.path.join(save_dir, filename_stub + "_feature_comparison.png")
    summary_csv_path = os.path.join(save_dir, filename_stub + "_feature_comparison.csv")

    summary_df = build_feature_summary_dataframe(
        section_rows,
        cv_labels,
        mode_order,
        presence_by_mode,
        counts_by_mode,
    )
    summary_df.to_csv(summary_csv_path, index=False)

    draw_three_way_feature_comparison_figure(
        section_rows,
        cv_labels,
        mode_order,
        presence_by_mode,
        counts_by_mode,
        figure_path,
        args.show,
    )

    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_csv_path}")


if __name__ == "__main__":
    main()
