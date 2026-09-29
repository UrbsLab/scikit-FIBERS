"""Reconstruct mixed intra/interlocus blocks from completed correlation tables.

Creates feature-inclusion plots only. Does not refit models or estimate survival.
python plot_mixed_blocks.py --input simple.zip --output mixed_blocks --threshold .90
"""
import argparse
import colorsys
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from review_outputs import Results, features, jaccard


def feature_key(name):
    order = {l: i for i, l in enumerate(["A", "B", "C", "DRB1", "DRB345", "DQA1", "DQB1"])}
    _, locus, position = name.split("_")
    return order[locus], int(position)


def make_blocks(table, names, threshold):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    index = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(names), len(names)), np.nan)
    np.fill_diagonal(matrix, 1)
    if len(table) != len(names) * (len(names) - 1) // 2:
        raise ValueError("The stored correlation table is not a complete pairwise matrix")
    left = table.feature1.map(index).to_numpy()
    right = table.feature2.map(index).to_numpy()
    if pd.isna(left).any() or pd.isna(right).any():
        raise ValueError("Correlation features differ from the retained feature list")
    matrix[left, right] = table.pearson_r
    matrix[right, left] = table.pearson_r
    if not np.isfinite(matrix).all() or (np.abs(matrix) > 1 + 1e-12).any():
        raise ValueError("Incomplete or invalid stored correlations")
    distances = np.clip(1 - matrix, 0, 2)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances), method="complete")
    labels = fcluster(tree, np.nextafter(1 - threshold, -np.inf), criterion="distance")
    groups = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        members = sorted([names[i] for i in ids], key=feature_key)
        minimum = float(matrix[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)].min()) if len(ids) > 1 else None
        if minimum is not None and minimum <= threshold:
            raise AssertionError("A reconstructed block fails the all-pairs cutoff")
        loci = sorted({name.split("_")[1] for name in members})
        groups.append({"block": "M_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:12],
                       "features": members, "loci": loci, "cross_locus": len(loci) > 1,
                       "minimum_r": minimum})
    return sorted(groups, key=lambda g: feature_key(g["features"][0]))


def expand(original, blocks):
    selected = set(original)
    groups = [g for g in blocks if selected.intersection(g["features"])]
    assert selected <= {f for g in groups for f in g["features"]}
    return groups


def plot_style(blocks_by_fold):
    names = sorted({f for groups in blocks_by_fold.values() for g in groups for f in g["features"]}, key=feature_key)
    maps = [{f: g["block"] for g in groups for f in g["features"]} for groups in blocks_by_fold.values()]
    clusters = {}
    for name in names:
        signature = tuple(mapping.get(name, "missing:" + name) for mapping in maps)
        clusters.setdefault(signature, []).append(name)
    groups = sorted(clusters.values(), key=lambda group: feature_key(group[0]))
    color, group_id, ordered = {}, {}, []
    colored = 0
    for i, group in enumerate(groups):
        rgb = np.array([.15, .15, .15])
        if len(group) > 1:
            rgb = np.array(colorsys.hsv_to_rgb((.53 + .61803398875 * colored) % 1, .70, .58))
            colored += 1
        for name in group:
            color[name], group_id[name] = rgb, i
            ordered.append(name)
    return ordered, color, group_id


def figure(records, names, labels, colors, group_ids, panels, column_label):
    fig, axes = plt.subplots(1, len(panels), figsize=(16 if len(panels) == 2 else 17, max(5.5, .34 * len(names) + 2)), sharey=True)
    for ax, (field, label) in zip(axes, panels):
        cells = np.ones((len(names), len(records), 3))
        for j, record in enumerate(records):
            original = set(record["original"])
            selected = set(record[field])
            for i, name in enumerate(names):
                if name in selected:
                    cells[i, j] = colors[name] if name in original else .28 * colors[name] + .72
        ax.imshow(cells, interpolation="nearest", aspect="auto")
        ticks = range(len(labels)) if len(labels) <= 12 else sorted(set([0, *range(4, len(labels), 5)]))
        ax.set_xticks(list(ticks), [labels[i] for i in ticks])
        ax.set_xlabel(label + "\n" + column_label)
        ax.set_yticks(range(len(names)), [n.removeprefix("MM_").replace("DRB345", "DRB3/4/5").replace("_", " ") for n in names])
        for i in range(1, len(names)):
            if group_ids[names[i]] != group_ids[names[i - 1]]:
                ax.axhline(i - .5, color="#bbbbbb", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Amino acid mismatch position")
    handles = [Patch(facecolor="#276b81", label="Original position (dark)"),
               Patch(facecolor="#bdd5de", label="Added alternative (light)"),
               Patch(facecolor="white", edgecolor="#999999", label="Absent")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(left=.13, right=.995, top=.90, bottom=.16, wspace=.08)
    return fig


def save_pages(records, labels, style, output, stem, rows_per_page, column_label):
    ordered, colors, group_ids = style
    present = {f for record in records for f in record["mixed"]}
    groups = []
    for _, grouped in itertools.groupby([f for f in ordered if f in present], key=group_ids.get):
        members = list(grouped)
        groups.extend(members[i:i + rows_per_page] for i in range(0, len(members), rows_per_page))
    pages, page = [], []
    for group in groups:
        if page and len(page) + len(group) > rows_per_page:
            pages.append(page)
            page = []
        page += group
    if page:
        pages.append(page)
    with PdfPages(output / (stem + ".pdf")) as pdf:
        for n, names in enumerate(pages, 1):
            fig = figure(records, names, labels, colors, group_ids,
                         [("original", "Original bins"), ("mixed", "Mixed-block processing")], column_label)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(output / f"{stem}_page{n:02d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    return len(pages)


def save_single_top_figure(records, style, output, colored_only=False):
    ordered, colors, group_ids = style
    present = {name for record in records for name in record["mixed"]}
    names = [name for name in ordered if name in present]
    assert len(names) == len(present)
    if colored_only:
        sizes = {}
        for group in group_ids.values():
            sizes[group] = sizes.get(group, 0) + 1
        names = [name for name in names if sizes[group_ids[name]] > 1]
        if not names:
            raise ValueError("No colored multi-position groups remain for this figure")
    fig = figure(records, names, [str(r["fold"]) for r in records], colors, group_ids,
                 [("original", "Original top bins"), ("mixed", "Mixed-block processing")],
                 "Cross-validation fold")
    height = max(6, .225 * len(names) + 1.8)
    fig.set_size_inches(13, height)
    fig.subplots_adjust(left=.15, right=.995, top=1 - .65 / height,
                        bottom=.85 / height, wspace=.08)
    stem = "top_bins_mixed_single_figure" + ("_colored_only" if colored_only else "")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if colored_only:
        (output / f"{stem}_caption.txt").write_text(
            f"Top bins across {len(records)} cross-validation folds, original (left) versus mixed intra/interlocus block processing (right). "
            f"Shows {len(names)} colored-group positions out of {len(present)} positions in the complete comparison. "
            f"The {len(present) - len(names)} black/gray-coded rows are omitted from both panels, not removed from the underlying bins. "
            "Hues denote positions sharing mixed-block membership across the training folds; dark cells are original positions, "
            "light cells are added alternatives, and white means absent. No shared multi-position color group across folds does not "
            "necessarily mean no correlation within an individual fold. Complete-bin similarity statistics are unchanged; "
            "no hazard ratios were recalculated.\n")
    return len(names)


def run(args):
    source = Results(args.input)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = source.json("summary/completed.json")["settings"]
    if args.imputation not in settings["imputations"] or args.seed not in settings["seeds"]:
        raise ValueError("The requested imputation/seed is not present in these outputs")
    if args.population_fold not in settings["folds"] or args.rows_per_page < 1:
        raise ValueError("Invalid population fold or rows-per-page")
    scheme = "r" + str(float(args.threshold)).replace(".", "p")
    blocks_by_fold, records, audits = {}, [], []
    for fold in settings["folds"]:
        root = f"imp_{args.imputation:02d}/cv_{fold:02d}"
        filters = source.csv(root + "/correlation/feature_filter.csv")
        names = filters.loc[filters.retained, "feature"].tolist()
        blocks = make_blocks(source.csv(root + "/correlation/correlations.csv.gz"), names, args.threshold)
        blocks_by_fold[fold] = blocks
        saved = {r["rank"]: r for r in source.json(root + "/correlation/processed_bins.json")
                 if r["seed"] == args.seed and r["scheme"] == scheme}
        if not saved:
            raise ValueError("A same-threshold within-locus comparison is required in the input outputs")
        population = source.csv(root + f"/seed_{args.seed:03d}/population.csv")
        for row in population.itertuples(index=False):
            original = json.loads(row.features)
            assert set(saved[row.rank]["original"]) == set(original)
            groups = expand(original, blocks)
            records.append({"imputation": args.imputation, "fold": fold, "seed": args.seed,
                            "rank": int(row.rank), "original_threshold": float(row.threshold),
                            "original": original, "within": sorted(features(saved[row.rank]), key=feature_key),
                            "mixed": sorted({f for g in groups for f in g["features"]}, key=feature_key),
                            "groups": groups})
        multi = [g for g in blocks if len(g["features"]) > 1]
        audits.append({"fold": fold, "features": len(names), "multi_position_blocks": len(multi),
                       "cross_locus_blocks": sum(g["cross_locus"] for g in multi),
                       "minimum_block_r": min(g["minimum_r"] for g in multi)})
    top = sorted([r for r in records if r["rank"] == 1], key=lambda r: r["fold"])
    pop = sorted([r for r in records if r["fold"] == args.population_fold], key=lambda r: r["rank"])
    rows = []
    scopes = [("top_bins_across_cv", top)] + [(f"population_cv{fold}", [r for r in records if r["fold"] == fold]) for fold in settings["folds"]]
    for scope, subset in scopes:
        for field in ("original", "within", "mixed"):
            sets = [set(r[field]) for r in subset]
            scores = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
            rows.append({"scope": scope, "scheme": field, "bins": len(subset), "pairs": len(scores),
                         "mean_jaccard": np.mean(scores), "mean_positions": np.mean([len(s) for s in sets])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "consistency.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "block_audit.csv", index=False)
    (output / "mixed_blocks_by_fold.json").write_text(json.dumps(blocks_by_fold, indent=2) + "\n")
    (output / "mixed_bin_memberships.json").write_text(json.dumps(records, indent=2) + "\n")
    style = plot_style(blocks_by_fold)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelsize": 16, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    top_position_count = save_single_top_figure(top, style, output, args.colored_only)
    single_name = "top_bins_mixed_single_figure" + ("_colored_only" if args.colored_only else "")
    single_scope = "Black/gray-coded rows are omitted only from this display." if args.colored_only else "No row filtering."
    pages = {"top_bins": save_pages(top, [str(r["fold"]) for r in top], style, output, "top_bins_mixed", args.rows_per_page, "Cross-validation fold"),
             "population": save_pages(pop, [str(r["rank"]) for r in pop], style, output, "population_mixed", args.rows_per_page, "Training-ranked bin")}
    ordered, colors, group_ids = style
    for subset, stem, column, label in [(top, "cross_locus_focus", "fold", "Cross-validation fold"),
                                       (pop, "population_cross_locus_focus", "rank", "Training-ranked bin")]:
        cross_features = {f for r in subset for g in r["groups"] if g["cross_locus"] for f in g["features"]}
        focused = [name for name in ordered if name in cross_features]
        if not focused:
            continue
        fig = figure(subset, focused, [str(r[column]) for r in subset], colors, group_ids,
                     [("original", "Original bins"), ("within", "Within-locus processing"), ("mixed", "Mixed-block processing")], label)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
        plt.close(fig)
    cross = {}
    for fold, blocks in blocks_by_fold.items():
        for block in blocks:
            if block["cross_locus"]:
                entry = cross.setdefault(block["block"], {"features": block["features"], "loci": block["loci"], "folds": [], "minimum_r": []})
                entry["folds"].append(fold)
                entry["minimum_r"].append(block["minimum_r"])
    (output / "cross_locus_blocks.json").write_text(json.dumps(cross, indent=2) + "\n")
    comparisons = metrics.loc[metrics.scope == "top_bins_across_cv"].set_index("scheme")
    caption = (f"Imputation {args.imputation}, seed {args.seed}, {len(top)} cross-validation folds; strict signed Pearson r > {args.threshold:g}. "
               "Blocks are constructed independently in each training fold using complete linkage across all retained loci. Every pair in a block exceeds the cutoff; no connected-component chaining is used. "
               "A bin includes all members of each block touched by an original feature. Original training rank and bin threshold are retained. "
               "Dark cells are original features; light cells are added alternatives; white is absent. Each hue denotes a group of positions sharing block membership across all available training folds. "
               "Black/gray indicates no shared multi-position color group across those folds, not necessarily absence of a correlation in an individual fold. "
               "Colors are assigned from the mixed analysis and held fixed in every panel: the same hue in the original or within-locus panel does not mean those positions were previously grouped across loci. "
               "Colors and row order are identical across panels. These figures show memberships, not patient mismatch scores or risk effects.\n\n")
    caption += ("cross_locus_focus: Original, within-locus, and mixed intra/interlocus processing at the same cutoff. "
                "Rows include all members of cross-locus blocks touched by any top bin; other bin positions are omitted only from this focused figure.\n\n"
                f"population_cross_locus_focus: The same three-way comparison for the {len(pop)}-bin population in fold {args.population_fold}, restricted to all cross-locus blocks touched by that population.\n\n"
                f"{single_name}: {top_position_count} positions in one original-versus-mixed figure across the top bins. {single_scope} No pagination; a tall layout preserves readable labels when enlarged.\n\n"
                f"top_bins_mixed: Complete original versus mixed-processed top bins across validation folds, {pages['top_bins']} readable pages.\n\n"
                f"population_mixed: Complete original versus mixed-processed population for fold {args.population_fold}, {len(pop)} bins in original training order, {pages['population']} readable pages.\n\n")
    caption += ("Full-bin mean Jaccard similarity across the top bins: " + ", ".join(f"{name} {comparisons.loc[name, 'mean_jaccard']:.6f}" for name in ["original", "within", "mixed"]) + ". "
                "These are descriptive comparisons of overlapping training analyses, not independent replications. "
                "No new hazard ratios have been calculated. Reuse the trained models and stored blocks on HPC to compute one maximum mismatch value per touched block and refit held-out survival comparisons; do not transfer hazard ratios from the within-locus results to these mixed blocks.\n")
    (output / "captions_and_methods.txt").write_text(caption)
    print(comparisons.to_string())
    print("Cross-locus blocks:", json.dumps(cross, indent=2))
    print("Pages:", pages)
    print("Written to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.90)
    parser.add_argument("--imputation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population-fold", type=int, default=1)
    parser.add_argument("--rows-per-page", type=int, default=24)
    parser.add_argument("--colored-only", action="store_true", help="Omit black/gray-coded rows from the single top-bin figure only")
    run(parser.parse_args())
