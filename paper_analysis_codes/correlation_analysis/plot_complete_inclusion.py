"""Plot every represented position from saved top bins without rerunning models."""
import argparse
import colorsys
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from review_outputs import Results, features, jaccard


LOCI = ["A", "B", "C", "DRB1", "DRB345", "DQA1", "DQB1"]
SINGLE = np.array([0.58, 0.62, 0.65])


def key(name):
    _, locus, position = name.split("_")
    return LOCI.index(locus), int(position)


def members(group):
    return tuple(sorted(group["features"], key=key))


def load_top(source, imputation, seed, threshold):
    settings = source.json("summary/completed.json")["settings"]
    if imputation not in settings["imputations"] or seed not in settings["seeds"]:
        raise ValueError("Requested imputation/seed not present")
    scheme = "r" + str(float(threshold)).replace(".", "p")
    records, maps = [], []
    for fold in sorted(settings["folds"]):
        root = f"imp_{imputation:02d}/cv_{fold:02d}/correlation/"
        records_in_fold = [r for r in source.json(root + "processed_bins.json")
                           if r["seed"] == seed and r["rank"] == 1 and r["scheme"] == scheme]
        if len(records_in_fold) != 1:
            raise ValueError(f"Expected exactly one top bin in fold {fold}")
        record = dict(records_in_fold[0], fold=fold)
        if not set(record["original"]) <= features(record):
            raise ValueError("Processing dropped an original position")
        definitions = source.json(root + "blocks.json")[scheme]["blocks"]
        mapping = {}
        for group in definitions:
            names = members(group)
            if len({f.split("_")[1] for f in names}) != 1:
                raise ValueError("This figure requires within-locus blocks")
            if len(names) > 1 and group["min_r"] <= threshold:
                raise ValueError("Stored block fails the strict correlation cutoff")
            for name in names:
                if name in mapping:
                    raise ValueError("Overlapping stored blocks")
                mapping[name] = names
        for group in record["groups"]:
            names = members(group)
            for name in names:
                if mapping.get(name, (name,)) != names:
                    raise ValueError("Processed-bin group differs from the stored training block")
        records.append(record)
        maps.append(mapping)
    return records, maps


def load_any_top(directory, imputation, seed, threshold):
    directory = Path(directory)
    metadata = json.loads((directory / "completed.json").read_text())
    if (metadata["imputation"] != imputation or metadata["seed"] != seed
            or metadata["threshold"] != threshold or not metadata["all_estimable"]):
        raise ValueError("Requested scope differs from the completed any-locus analysis")
    saved = json.loads((directory / "mixed_bin_memberships.json").read_text())
    definitions = json.loads((directory / "mixed_blocks_by_fold.json").read_text())
    records, maps = [], []
    for fold in sorted(metadata["folds"]):
        matches = [r for r in saved if r["fold"] == fold and r["imputation"] == imputation
                   and r["seed"] == seed and r["rank"] == 1]
        if len(matches) != 1:
            raise ValueError(f"Expected one any-locus top bin in fold {fold}")
        record = dict(matches[0], groups=matches[0]["mixed_groups"])
        if features(record) != set(record["mixed"]) or not set(record["original"]) <= features(record):
            raise ValueError("Any-locus memberships are inconsistent")
        mapping = {}
        for group in definitions[str(fold)]:
            names = members(group)
            if len(names) > 1 and group["minimum_r"] <= threshold:
                raise ValueError("Any-locus block fails the strict correlation cutoff")
            for name in names:
                if name in mapping:
                    raise ValueError("Overlapping any-locus blocks")
                mapping[name] = names
        for group in record["groups"]:
            if any(mapping[name] != members(group) for name in group["features"]):
                raise ValueError("Selected group differs from the stored training block")
        records.append(record)
        maps.append(mapping)
    return records, maps


def order_rows(names, maps):
    # Shared membership patterns define row order only. Each cell still uses
    # its own fold's exact block, so changing block membership is not hidden.
    groups = defaultdict(list)
    for name in names:
        signature = tuple(mapping.get(name, (name,)) for mapping in maps)
        groups[(key(name)[0], signature)].append(name)
    groups = [sorted(group, key=key) for group in groups.values()]
    groups.sort(key=lambda group: key(group[0]))
    ordered = [name for group in groups for name in group]
    group_ids = {name: i for i, group in enumerate(groups) for name in group}
    return ordered, group_ids


def run(args):
    source = None
    if args.scope == "any":
        records, maps = load_any_top(args.input, args.imputation, args.seed, args.threshold)
    else:
        source = Results(args.input)
        records, maps = load_top(source, args.imputation, args.seed, args.threshold)
    originals = [set(r["original"]) for r in records]
    processed = [features(r) for r in records]
    names, row_groups = order_rows(set.union(*processed), maps)
    block_keys = sorted({members(g) for r in records for g in r["groups"]
                         if len(g["features"]) > 1}, key=lambda group: key(group[0]))
    colors = {group: np.array(colorsys.hsv_to_rgb((.52 + i * .61803398875) % 1, .72, .64))
              for i, group in enumerate(block_keys)}
    # Keep the same DQA1 example color as the presentation.
    example = tuple("MM_DQA1_" + str(n) for n in [21, 88, 89, 90, 91, 92, 93, 94])
    if example in colors:
        colors[example] = np.array([7, 149, 164]) / 255
    block_ids = {group: f"Block {i + 1:02d}" for i, group in enumerate(block_keys)}
    matrices = [np.ones((len(names), len(records), 3)) for _ in range(2)]
    rows = []
    for j, (record, mapping) in enumerate(zip(records, maps)):
        for i, name in enumerate(names):
            group = mapping.get(name, (name,))
            color = colors.get(group, SINGLE)
            original, present = name in originals[j], name in processed[j]
            if original:
                matrices[0][i, j] = color
                matrices[1][i, j] = color
            elif present:
                if len(group) < 2:
                    raise ValueError("Added position does not belong to a block")
                matrices[1][i, j] = .3 * color + .7
            rows.append({"row": i + 1, "feature": name, "fold": record["fold"],
                         "original": original, "processed": present,
                         "state": "original" if original else "added" if present else "absent",
                         "block": block_ids.get(group, "ungrouped"),
                         "block_features": "|".join(group),
                         "dark_color": to_hex(color)})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_any_locus" if args.scope == "any" else ""
    stem = f"top_bins_imp{args.imputation}_r{str(args.threshold).replace('.', 'p')}{suffix}_complete"
    plt.rcParams.update({"font.family": "Arial", "font.size": 14, "pdf.fonttype": 42,
                         "text.color": "#303E46", "xtick.color": "#303E46",
                         "ytick.color": "#303E46", "axes.labelcolor": "#303E46"})
    height = .24 * len(names) + 2.2
    fig, axes = plt.subplots(1, 2, figsize=(14, height), sharey=True)
    processed_label = "Any-locus processed top bins" if args.scope == "any" else "Processed top bins"
    for ax, matrix, panel in zip(axes, matrices, ["Original top bins", processed_label]):
        ax.imshow(matrix, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(records)), [str(r["fold"]) for r in records])
        ax.set_yticks(range(len(names)), [name.removeprefix("MM_").replace("DRB345", "DRB3/4/5").replace("_", " ") for name in names])
        ax.set_xticks(np.arange(-.5, len(records), 1), minor=True)
        ax.grid(which="minor", axis="x", color="#E1E7EA", linewidth=.3)
        ax.tick_params(which="both", length=0, pad=5)
        ax.set_xlabel(panel + "\nCross-validation fold", labelpad=9, fontsize=16)
        ax.secondary_xaxis("top").set_xticks(range(len(records)), [str(r["fold"]) for r in records])
        for i in range(1, len(names)):
            if names[i].split("_")[1] != names[i - 1].split("_")[1]:
                ax.axhline(i - .5, color="#718792", lw=1.1)
            elif row_groups[names[i]] != row_groups[names[i - 1]]:
                ax.axhline(i - .5, color="#CEDADD", lw=.45)
        for spine in ax.spines.values():
            spine.set_color("#ACBDC4")
            spine.set_linewidth(.7)
    axes[0].set_ylabel("Amino acid mismatch position", labelpad=13, fontsize=16)
    handles = [Patch(facecolor="#0795A4", label="Original block member (dark)"),
               Patch(facecolor="#BCE6E9", label="Added block member (light)"),
               Patch(facecolor=SINGLE, label="Ungrouped original (gray)"),
               Patch(facecolor="white", edgecolor="#ACBDC4", label="Absent")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1 - .08 / height),
               ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(left=.175, right=.995, top=1 - 1.40 / height,
                        bottom=1.05 / height, wspace=.07)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"{stem}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / f"{stem}_cells.csv", index=False)
    values = [[jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
              for sets in [originals, processed]]
    summary = {"source": str(args.input.resolve()), "imputation": args.imputation, "seed": args.seed,
               "folds": [r["fold"] for r in records], "correlation_scope": args.scope,
               "pearson_r_threshold": args.threshold,
               "positions_shown": len(names), "original_distinct_positions": len(set.union(*originals)),
               "omitted_positions": 0, "colored_block_definitions": len(block_keys),
               "mean_positions_original": float(np.mean([len(x) for x in originals])),
               "mean_positions_processed": float(np.mean([len(x) for x in processed])),
               "mean_jaccard_original": float(np.mean(values[0])),
               "mean_jaccard_processed": float(np.mean(values[1]))}
    if args.scope == "any":
        reference = pd.read_csv(args.input / "summary.csv").set_index("scheme").loc["mixed"]
        for name in ["mean_jaccard", "mean_positions"]:
            if not np.isclose(summary[name + "_processed"], reference[name], rtol=0, atol=1e-12):
                raise ValueError("Figure memberships disagree with the completed metrics")
    (output / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    processing = "locus-agnostic" if args.scope == "any" else "within-locus"
    cross_note = ("One cutoff applies whether positions are in the same or different loci. Members of an "
                  "interlocus block share the same hue across their locus sections. " if args.scope == "any" else "")
    caption = (f"Complete feature inclusion for the top training-ranked FIBERS bin from each of "
               f"{len(records)} cross-validation folds in imputation {args.imputation}, seed {args.seed}, "
               f"before (left) and after (right) {processing} processing at Pearson r > {args.threshold:g}. "
               + cross_note +
               f"All {len(names)} positions appearing in either version of any top bin are shown, with "
               "identical row and column order. Different hues identify distinct exact training-block "
               "memberships; a shared hue does not denote an HLA locus or clinical risk. Dark cells mark "
               "originally selected block members and matching light cells mark added alternatives. Gray "
               "cells retain ungrouped original positions; white cells indicate absence. When a block's "
               "membership differs across folds its hue also differs. Rows are grouped by locus and "
               "shared block-membership patterns for display only. Each stored multi-position block "
               "satisfies the strict all-pairs cutoff in its training fold. No positions were filtered "
               "out and no FIBERS, correlation or survival models were rerun.\n")
    (output / f"{stem}_caption.txt").write_text(caption)
    if source is not None and source.archive:
        source.archive.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imputation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=.95)
    parser.add_argument("--scope", choices=["within", "any"], default="within",
                        help="For any, --input is the completed run_mixed_risk.py output directory")
    run(parser.parse_args())
