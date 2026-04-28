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

from feature_category_logic import categorize_three_mode_features, mode_title
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
    short_mode_label,
    sort_cv_labels,
)

FIGURE_BACKGROUND_COLOR = "#FFFFFF"

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
        "--min-feature-count",
        dest="min_feature_count",
        type=int,
        default=1,
        help="Minimum recurrence in at least one mode for a feature to be shown. Use 3 for a compact abstract-style figure.",
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
    parser.add_argument(
        "--focus-mode",
        dest="focus_mode",
        type=str,
        default=None,
        help="Optional mode name to move mode-containing sections to the top for storytelling.",
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


def build_feature_summary_dataframe(categorized_rows, cv_labels, mode_order, presence_by_mode):
    rows = []
    for categorized_row in categorized_rows:
        row = {
            "section": categorized_row["section"].replace("\n", " "),
            "feature": categorized_row["feature"],
            "category_rule": categorized_row["category_rule"],
            "category_reason": categorized_row["category_reason"],
        }
        for mode_name in mode_order:
            row[f"{mode_name}_count"] = categorized_row[mode_name]
            for cv_label in cv_labels:
                row[f"{mode_name}_cv_{cv_label}"] = int(
                    categorized_row["feature"] in presence_by_mode[mode_name][cv_label]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def reorder_sections_for_focus(section_rows, categorized_rows, focus_mode):
    if focus_mode is None:
        return section_rows, categorized_rows

    focus_title = mode_title(focus_mode)
    section_rank = {}
    for index, (section_name, _) in enumerate(section_rows):
        has_focus = focus_title in section_name
        is_core = section_name == "Core shared"
        section_rank[section_name] = (
            0 if has_focus else 1,
            0 if is_core else 1,
            index,
        )

    section_rows = sorted(section_rows, key=lambda row: section_rank[row[0]])
    feature_rank = {}
    for section_name, feature_names in section_rows:
        for index, feature_name in enumerate(feature_names):
            feature_rank[(section_name, feature_name)] = index
    categorized_rows = sorted(
        categorized_rows,
        key=lambda row: (
            section_rank[row["section"]],
            feature_rank[(row["section"], row["feature"])],
        ),
    )
    return section_rows, categorized_rows


def draw_three_way_feature_comparison_figure(
    section_rows,
    cv_labels,
    mode_order,
    presence_by_mode,
    counts_by_mode,
    min_feature_count,
    save_path,
    show,
):
    flattened_features = [feature_name for _, feature_names in section_rows for feature_name in feature_names]
    if len(flattened_features) == 0:
        raise ValueError("No features to plot for the requested three-mode comparison.")

    nrows = len(flattened_features)
    cv_count = len(cv_labels)

    left_label_width = 8.8
    feature_label_width = 4.4
    cell_size = 1.0
    panel_gap = 0.9
    count_gap = 1.8
    count_width = 1.2
    right_margin = 0.8
    header_height = 1.85
    legend_height = 1.5
    row_height = 1.18 if nrows <= 45 else 1.08
    section_gap = 0.75
    feature_font_size = 15 if nrows <= 28 else 13
    section_font_size = 16 if nrows <= 28 else 14
    header_font_size = 16
    cv_font_size = 9

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
    total_height = header_height + (nrows * row_height) + (max(len(section_rows) - 1, 0) * section_gap) + legend_height

    fig_width = max(15, total_width * 0.47)
    fig_height = max(9, total_height * 0.38)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(FIGURE_BACKGROUND_COLOR)
    ax.set_facecolor(FIGURE_BACKGROUND_COLOR)
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    header_y = 0.08
    header_box_height = 0.72
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
            fontsize=header_font_size,
            fontweight="bold",
        )

    cv_label_y = 1.18
    for panel_x in panel_x_positions:
        for cv_index, cv_label in enumerate(cv_labels):
            ax.text(
                panel_x + (cv_index + 0.5) * cell_size,
                cv_label_y,
                f"CV{cv_label}",
                ha="center",
                va="center",
                fontsize=cv_font_size,
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
    row_cursor = 0.0
    section_boundaries = []

    for section_index, (section_name, feature_names) in enumerate(section_rows):
        section_start = row_cursor
        for feature_name in feature_names:
            locus_color = feature_color(feature_name)
            y = row_y_start + row_cursor

            ax.text(
                left_label_width + feature_label_width - 0.3,
                y + (row_height * 0.46),
                feature_name,
                ha="right",
                va="center",
                fontsize=feature_font_size,
                color=locus_color,
            )

            for mode_name, panel_x in zip(mode_order, panel_x_positions):
                for cv_index, cv_label in enumerate(cv_labels):
                    mode_present = feature_name in presence_by_mode[mode_name][cv_label]
                    ax.add_patch(
                        mpatches.Rectangle(
                            (panel_x + cv_index * cell_size, y),
                            0.9,
                            min(0.9, row_height * 0.82),
                            facecolor=locus_color if mode_present else EMPTY_CELL_COLOR,
                            edgecolor=GRID_EDGE_COLOR,
                            linewidth=1.0,
                        )
                    )

            for mode_name, count_x in zip(mode_order, count_x_positions):
                draw_count_badge(
                    ax,
                    count_x + count_width / 2.0,
                    y + (row_height * 0.42),
                    counts_by_mode[mode_name].get(feature_name, 0),
                    MODE_HEADER_COLORS.get(mode_name, "#5477A8"),
                )

            row_cursor += row_height

        section_end = row_cursor
        section_boundaries.append((section_name, section_start, section_end))
        if section_index < len(section_rows) - 1:
            row_cursor += section_gap

    for section_name, section_start, section_end in section_boundaries:
        section_center_y = row_y_start + (section_start + section_end) / 2.0
        ax.text(
            0.45,
            section_center_y,
            section_name,
            ha="left",
            va="center",
            fontsize=section_font_size,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    final_panel_end = panel_x_positions[-1] + cv_count * cell_size
    for (_, _, section_end), (_, next_section_start, _) in zip(section_boundaries[:-1], section_boundaries[1:]):
        divider_y = row_y_start + (section_end + next_section_start) / 2.0
        ax.plot(
            [0.35, final_panel_end],
            [divider_y, divider_y],
            color=SECTION_LINE_COLOR,
            linewidth=2.0,
        )

    legend_y = row_y_start + row_cursor + 0.72
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
        legend_x += max(2.2, 1.15 + (0.42 * len(locus_name)))

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

    section_rows, categorized_rows = categorize_three_mode_features(
        counts_by_mode,
        mode_order,
        args.core_shared_min,
        args.min_feature_count,
    )
    section_rows, categorized_rows = reorder_sections_for_focus(
        section_rows,
        categorized_rows,
        args.focus_mode,
    )

    if len(section_rows) == 0:
        raise ValueError("No features matched the display rules for the requested modes.")

    filename_stub = f"{args.prefix}{'_vs_'.join(mode_order)}_top{args.top_bin_count}"
    figure_path = os.path.join(save_dir, filename_stub + "_feature_comparison.png")
    summary_csv_path = os.path.join(save_dir, filename_stub + "_feature_comparison.csv")

    summary_df = build_feature_summary_dataframe(
        categorized_rows,
        cv_labels,
        mode_order,
        presence_by_mode,
    )
    summary_df.to_csv(summary_csv_path, index=False)

    draw_three_way_feature_comparison_figure(
        section_rows,
        cv_labels,
        mode_order,
        presence_by_mode,
        counts_by_mode,
        args.min_feature_count,
        figure_path,
        args.show,
    )

    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_csv_path}")


if __name__ == "__main__":
    main()
