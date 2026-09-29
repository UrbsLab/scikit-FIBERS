"""Explain feature-list consistency with output-only controls and compact plots.

Random alternatives preserve every original position and each fold's exact
number of added positions per locus. They are not new models or risk estimates.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from review_outputs import Results, features, jaccard, save


LOCI = ["A", "B", "C", "DRB1", "DRB345", "DQA1", "DQB1"]
BLUE = "#17657a"
ORANGE = "#b16a32"


def locus(name):
    return name.split("_")[1]


def mean_overlap(sets):
    return float(np.mean([jaccard(a, b) for a, b in itertools.combinations(sets, 2)]))


def matrix_overlap(matrix):
    matrix = np.asarray(matrix, dtype=np.int32)
    shared = matrix @ matrix.T
    totals = matrix.sum(axis=1)
    union = totals[:, None] + totals[None, :] - shared
    pair_ids = np.triu_indices(len(matrix), 1)
    return float(np.mean(shared[pair_ids] / union[pair_ids]))


def random_control(originals, expanded, eligible, repeats, seed):
    names = sorted(set.union(*eligible))
    index = {name: i for i, name in enumerate(names)}
    original_matrix = np.zeros((len(originals), len(names)), dtype=np.int32)
    expected_counts = np.zeros((len(originals), len(LOCI)), dtype=int)
    masks = {loc: np.array([locus(name) == loc for name in names]) for loc in LOCI}
    instructions = []
    for i, (original, processed, pool) in enumerate(zip(originals, expanded, eligible)):
        assert original <= processed <= pool
        original_matrix[i, [index[name] for name in original]] = 1
        for j, loc in enumerate(LOCI):
            expected_counts[i, j] = sum(locus(name) == loc for name in processed)
            n = sum(locus(name) == loc for name in processed - original)
            candidates = np.array([index[name] for name in sorted(pool - original) if locus(name) == loc])
            if n:
                assert len(candidates) >= n
                instructions.append((i, n, candidates))
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for repeat in range(repeats):
        matrix = original_matrix.copy()
        for i, n, candidates in instructions:
            matrix[i, rng.choice(candidates, size=n, replace=False)] = 1
        assert np.all(matrix >= original_matrix)
        for j, loc in enumerate(LOCI):
            assert np.array_equal(matrix[:, masks[loc]].sum(axis=1), expected_counts[:, j])
        values[repeat] = matrix_overlap(matrix)
    assert np.isclose(matrix_overlap(original_matrix), mean_overlap(originals))
    return values


def controlled_result(original, processed, randomized, risk, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    values = [original, float(randomized.mean()), processed]
    limits = np.quantile(randomized, [.025, .975])
    ax.barh(range(3), values, color=["#9faeb4", "#c7ced1", BLUE], height=.52)
    ax.errorbar(values[1], 1, xerr=[[values[1] - limits[0]], [limits[1] - values[1]]],
                fmt="none", color="#555555", capsize=6, lw=1.8)
    for i, value in enumerate(values):
        x = limits[1] + .009 if i == 1 else value + .009
        ax.text(x, i, f"{value:.3f}", va="center", fontsize=18, fontweight="bold")
    ax.set(yticks=range(3), yticklabels=["Original bins", "Random\nalternatives", "Correlated\nalternatives"],
           xlim=(0, .34), xlabel="Mean overlap between top bins\n(Jaccard index)")
    ax.invert_yaxis()
    ax.set_xticks([0, .1, .2, .3])
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    for row in risk.itertuples():
        ax.plot([0, 1], [row.original_HR, row.processed_HR], color="#adb8bc", lw=1.5)
    means = risk[["original_HR", "processed_HR"]].mean().to_numpy()
    ax.plot([0, 1], means, color=BLUE, lw=4)
    for i, value in enumerate(means):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 16), textcoords="offset points",
                    ha="center", fontsize=20, fontweight="bold", color=BLUE,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 2})
    ax.set(xticks=[0, 1], xticklabels=["Original", "Processed"], xlim=(-.3, 1.3), ylim=(1, 1.32),
           ylabel="Held-out hazard ratio\n(unadjusted)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.legend(handles=[Line2D([], [], color="#adb8bc", lw=1.5, label="One fold"),
                       Line2D([], [], color=BLUE, lw=4, label="Mean")],
              loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False, fontsize=15)
    for letter, ax in zip("AB", axes):
        ax.text(-.08, 1.04, letter, transform=ax.transAxes, fontsize=21, fontweight="bold")
    fig.subplots_adjust(left=.17, right=.98, top=.91, bottom=.23, wspace=.52)
    save(fig, output, "05_controlled_primary_result")


def locus_figure(originals, expanded, output):
    baseline = mean_overlap(originals)
    rows = []
    for loc in LOCI:
        modified = [original | {name for name in processed if locus(name) == loc}
                    for original, processed in zip(originals, expanded)]
        overlap = mean_overlap(modified)
        rows.append({"locus": loc, "mean_jaccard": overlap, "delta_from_original": overlap - baseline})
    rows.append({"locus": "All", "mean_jaccard": mean_overlap(expanded),
                 "delta_from_original": mean_overlap(expanded) - baseline})
    table = pd.DataFrame(rows)
    table.to_csv(output / "locus_processing_effect.csv", index=False)
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    values = table.delta_from_original.to_numpy()
    colors = [BLUE if v >= 0 else ORANGE for v in values]
    colors[-1] = "#80969d"
    ax.barh(range(len(rows)), values, height=.58, color=colors)
    for i, value in enumerate(values):
        label = f"{value:+.4f}" if 0 < abs(value) < .0005 else f"{value:+.3f}"
        ax.text(max(value, 0) + .0025, i, label, ha="left", va="center", fontsize=16)
    ax.axvline(0, color="#777777", lw=1)
    ax.axhline(6.5, color="#bbbbbb", lw=1)
    labels = [loc.replace("DRB345", "DRB3/4/5") for loc in LOCI] + ["All loci together"]
    ax.set(yticks=range(len(rows)), yticklabels=labels, xlim=(-.015, .122),
           xticks=[0, .025, .050, .075, .100],
           xlabel="Change in complete-bin Jaccard after processing the named locus")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.21, right=.98, top=.97, bottom=.18)
    save(fig, output, "06_locus_processing_effect")
    return rows


def block_coverage(records, blocks_by_fold, minimum_folds, output):
    stable = {tuple(b["features"]): b for b in blocks_by_fold[0]}
    for fold_blocks in blocks_by_fold[1:]:
        current = {tuple(b["features"]) for b in fold_blocks}
        stable = {k: v for k, v in stable.items() if k in current}
    candidates = []
    originals = [set(r["original"]) for r in records]
    for members, block in stable.items():
        counts = [len(set(members) & original) for original in originals]
        touched = sum(n > 0 for n in counts)
        if touched >= minimum_folds:
            candidates.append({"block": block["block"], "locus": block["locus"],
                               "features": list(members), "original_counts": counts, "folds_present": touched})
    candidates.sort(key=lambda r: (-r["folds_present"], LOCI.index(r["locus"]), r["features"]))
    if not candidates:
        raise ValueError("No stable blocks satisfy the displayed recurrence rule")
    counters = {loc: 0 for loc in LOCI}
    for row in candidates:
        loc = row["locus"]
        counters[loc] += 1
        row["label"] = f"{loc.replace('DRB345', 'DRB3/4/5')} group {counters[loc]} ({len(row['features'])} positions)"
    counts = np.array([r["original_counts"] for r in candidates])
    fig, ax = plt.subplots(figsize=(12.5, max(5.5, len(candidates) * .48 + 1.7)))
    ax.imshow(counts > 0, cmap=ListedColormap(["white", BLUE]), vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    for i, row in enumerate(candidates):
        for j, count in enumerate(row["original_counts"]):
            if count:
                ax.text(j, i, f"{count}/{len(row['features'])}", color="white", ha="center", va="center", fontsize=15)
        ax.text(10.05, i, f"{row['folds_present']}/10", va="center", ha="left", fontsize=15)
    ax.set(xticks=range(10), xticklabels=[str(r["fold"]) for r in records],
           yticks=range(len(candidates)), yticklabels=[r["label"] for r in candidates],
           xlabel="Cross-validation fold (original top bins)", xlim=(-.5, 11))
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(candidates), 1), minor=True)
    ax.grid(which="minor", color="#e3e8ea", lw=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(10.35, -.83, "Folds", ha="center", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(facecolor=BLUE, label="At least one member originally selected"),
                        Patch(facecolor="white", edgecolor="#aaaaaa", label="No member selected")],
               loc="upper center", ncol=2, frameon=False, fontsize=15)
    fig.subplots_adjust(left=.32, right=.97, top=.85, bottom=.15)
    save(fig, output, "07_block_coverage")
    (output / "displayed_blocks.json").write_text(json.dumps(candidates, indent=2) + "\n")
    return candidates


def run(args):
    if args.repeats < 100 or not 1 <= args.minimum_folds <= 10:
        raise ValueError("Use at least 100 randomized repeats and a recurrence cutoff from 1 to 10")
    result = Results(args.input)
    settings = result.json("summary/completed.json")["settings"]
    if settings["imputations"] != [1] or settings["seeds"] != [1] or sorted(settings["folds"]) != list(range(1, 11)):
        raise ValueError("These talk figures explicitly describe imputation 1, seed 1, ten folds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, blocks_by_fold, eligible = [], [], []
    all_coverage_unchanged = True
    for fold in range(1, 11):
        root = f"imp_01/cv_{fold:02d}"
        record = next(r for r in result.json(root + "/correlation/processed_bins.json")
                      if r["rank"] == 1 and r["seed"] == 1 and r["scheme"] == "r0p95")
        blocks = result.json(root + "/correlation/blocks.json")["r0p95"]["blocks"]
        mapping = {feature: b["block"] for b in blocks for feature in b["features"]}
        original_ids = {mapping.get(f, "single:" + f) for f in record["original"]}
        expanded_ids = {mapping.get(f, "single:" + f) for f in features(record)}
        assert original_ids == expanded_ids
        records.append(record)
        blocks_by_fold.append(blocks)
        eligible.append(set(result.json(root + "/seed_001/completed.json")["features"]))
    originals = [set(r["original"]) for r in records]
    expanded = [features(r) for r in records]
    original, processed = mean_overlap(originals), mean_overlap(expanded)
    verified = json.loads((args.review / "verified_summary.json").read_text())
    assert np.isclose(original, verified["mean_jaccard_original"])
    assert np.isclose(processed, verified["mean_jaccard_processed"])
    randomized = random_control(originals, expanded, eligible, args.repeats, args.seed)
    pd.DataFrame({"repeat": np.arange(1, args.repeats + 1), "mean_jaccard": randomized}).to_csv(
        output / "random_alternatives.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "pdf.fonttype": 42})
    risk = pd.read_csv(args.review / "top_bin_fold_comparison.csv")
    assert len(risk) == 10 and np.isfinite(risk[["original_HR", "processed_HR"]]).all().all()
    controlled_result(original, processed, randomized, risk, output)
    locus_rows = locus_figure(originals, expanded, output)
    blocks = block_coverage(records, blocks_by_fold, args.minimum_folds, output)
    summary = {"input": str(args.input.resolve()), "imputation": 1, "seed": 1, "folds": 10,
               "threshold": .95, "block_scope": "within_locus", "original_mean_jaccard": original,
               "processed_mean_jaccard": processed, "randomized_mean_jaccard": float(randomized.mean()),
               "randomized_central_95_percent_range": np.quantile(randomized, [.025, .975]).tolist(),
               "randomized_repeats": args.repeats, "randomized_seed": args.seed,
               "randomized_max_jaccard": float(randomized.max()),
               "null_preserves_original_features_and_per_locus_sizes": True,
               "block_coverage_unchanged_in_all_folds": all_coverage_unchanged,
               "locus_processing_effects": locus_rows, "displayed_blocks": len(blocks),
               "minimum_folds_for_display": args.minimum_folds,
               "interpretation": "Descriptive conditional randomization benchmark, not a patient confidence interval or a formal clinical test"}
    (output / "new_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if result.archive:
        result.archive.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--minimum-folds", type=int, default=3)
    run(parser.parse_args())
