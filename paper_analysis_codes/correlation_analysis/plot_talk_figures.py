"""Make compact ASHI figures from audited outputs, without refitting models."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from review_outputs import Results, features, jaccard, compact_result_figure, save


DARK = "#17657a"
LIGHT = "#bedde4"


def inclusion_panel(ax, records, names, field, label):
    cells = np.ones((len(names), len(records), 3))
    dark, light = matplotlib.colors.to_rgb(DARK), matplotlib.colors.to_rgb(LIGHT)
    for j, record in enumerate(records):
        original, selected = set(record["original"]), set(record[field])
        for i, name in enumerate(names):
            if name in selected:
                cells[i, j] = dark if name in original else light
    ax.imshow(cells, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(records)), [str(r["fold"]) for r in records])
    ax.set_yticks(range(len(names)), [n.removeprefix("MM_").replace("_", " ") for n in names])
    ax.set_xlabel(label + "\nCross-validation fold")
    ax.spines[["top", "right"]].set_visible(False)


def inclusion_legend(fig):
    fig.legend(handles=[Patch(facecolor=DARK, label="Originally selected"),
                        Patch(facecolor=LIGHT, label="Added correlated alternative"),
                        Patch(facecolor="white", edgecolor="#aaaaaa", label="Absent")],
               loc="upper center", ncol=3, frameon=False, fontsize=14)


def threshold_figure(table, summary, output):
    # These are categorical settings, not equally spaced numerical cutoffs.
    thresholds = [.99, .95, .90, .40, .10]
    selected = table.set_index("scheme").loc[
        ["r" + str(r).replace(".", "p") for r in thresholds]].copy()
    selected["changed_percent"] = 100 * selected.changed_patients / selected.total_patients
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for ax, values, ylabel, ylim, baseline in [
        (axes[0], selected.mean_jaccard, "Mean overlap between top bins\n(Jaccard index)", (0, .42),
         summary["mean_jaccard_original"]),
        (axes[1], selected.changed_percent, "Patients changing risk group (%)", (-.7, 19), 0),
    ]:
        ax.axvspan(.65, 1.35, color="#e9f1f4", zorder=0)
        ax.axhline(baseline, color="#888888", ls="--", lw=1.4)
        ax.plot(range(len(thresholds)), values, color=DARK, lw=3)
        for i, value in enumerate(values):
            label = f"{value:.3f}" if ax is axes[0] or value < 1 else f"{value:.1f}"
            ax.annotate(label, (i, value), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=15)
        ax.set(xticks=range(len(thresholds)), xticklabels=[f"{r:.2f}" for r in thresholds],
               xlim=(-.4, 4.4), ylim=ylim, ylabel=ylabel,
               xlabel="Within-locus correlation cutoff (r >)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
    axes[0].text(3.6, summary["mean_jaccard_original"] + .009, "Original bins",
                 ha="right", fontsize=14, color="#666666")
    fig.subplots_adjust(left=.085, right=.99, bottom=.19, top=.94, wspace=.36)
    save(fig, output, "03_cutoff_sensitivity")
    selected.to_csv(output / "03_cutoff_values.csv", index=True)


def interlocus_figure(records, comparison, output):
    fig = plt.figure(figsize=(12.5, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=.85, wspace=.35)
    names = ["MM_DRB1_14", "MM_DRB1_25", "MM_DQA1_54"]
    for ax, field, label in [(fig.add_subplot(grid[0, 0]), "original", "Original top bins"),
                             (fig.add_subplot(grid[0, 1]), "mixed", "Mixed-block top bins")]:
        inclusion_panel(ax, records, names, field, label)
    ax = fig.add_subplot(grid[1, :])
    schemes = ["original", "within", "mixed"]
    values = comparison.set_index("scheme").loc[schemes, "mean_jaccard"]
    ax.barh(range(3), values, color=["#9aadb3", "#659caa", DARK], height=.52)
    for i, value in enumerate(values):
        ax.text(value + .006, i, f"{value:.3f}", va="center", fontsize=17)
    ax.set(yticks=range(3), yticklabels=["Original", "Within-locus, r > 0.90", "Mixed loci, r > 0.90"],
           xlim=(0, .40), xlabel="Mean overlap between complete top bins (Jaccard index)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    inclusion_legend(fig)
    fig.subplots_adjust(left=.21, right=.99, top=.90, bottom=.12)
    save(fig, output, "04_interlocus_followup")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.review / "verified_summary.json").read_text())
    assert (summary["imputation"], summary["seed"], summary["folds"], summary["primary_threshold"]) == (1, 1, 10, .95)
    result = Results(args.input)
    records = []
    for fold in range(1, 11):
        candidates = result.json(f"imp_01/cv_{fold:02d}/correlation/processed_bins.json")
        record = next(r for r in candidates if r["rank"] == 1 and r["seed"] == 1 and r["scheme"] == "r0p95")
        records.append({"fold": fold, "original": record["original"], "processed": sorted(features(record))})
    for field, key in [("original", "mean_jaccard_original"), ("processed", "mean_jaccard_processed")]:
        actual = np.mean([jaccard(set(a[field]), set(b[field])) for a, b in itertools.combinations(records, 2)])
        assert np.isclose(actual, summary[key])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 15, "ytick.labelsize": 15, "pdf.fonttype": 42})
    block = next(b for b in summary["illustrative_blocks"] if all(f.startswith("MM_DQA1_") for f in b["features"]))
    names = block["features"]
    for fold in range(1, 11):
        blocks = result.json(f"imp_01/cv_{fold:02d}/correlation/blocks.json")["r0p95"]["blocks"]
        assert any(set(b["features"]) == set(names) for b in blocks)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for ax, field, label in zip(axes, ["original", "processed"], ["Original top bins", "Processed top bins"]):
        inclusion_panel(ax, records, names, field, label)
    inclusion_legend(fig)
    fig.subplots_adjust(left=.12, right=.99, bottom=.18, top=.85, wspace=.35)
    save(fig, output, "01_feature_inclusion_example")
    pairs = pd.read_csv(args.review / "top_bin_overlap_pairs.csv")
    risk = result.csv("summary/risk_comparison.csv.gz")
    risk = risk.loc[(risk["rank"] == 1) & (risk.dataset == "test") &
                    ((risk.variant == "original") | ((risk.scheme == "r0p95") & (risk.variant == "processed_fixed")))]
    assert len(risk) == 20 and risk["HR status"].eq("ok").all()
    compact_result_figure(pairs, risk, output)
    for extension in ["png", "pdf"]:
        (output / f"main_result.{extension}").replace(output / f"02_primary_result.{extension}")
    threshold_figure(pd.read_csv(args.review / "cutoff_sensitivity.csv"), summary, output)
    mixed = json.loads((args.mixed / "mixed_bin_memberships.json").read_text())
    mixed = sorted([r for r in mixed if r["rank"] == 1 and r["imputation"] == 1 and r["seed"] == 1], key=lambda r: r["fold"])
    assert len(mixed) == 10
    comparison = pd.read_csv(args.mixed / "consistency.csv")
    comparison = comparison.loc[comparison.scope == "top_bins_across_cv"]
    for scheme in ["original", "within", "mixed"]:
        actual = np.mean([jaccard(set(a[scheme]), set(b[scheme])) for a, b in itertools.combinations(mixed, 2)])
        assert np.isclose(actual, comparison.set_index("scheme").loc[scheme, "mean_jaccard"])
    interlocus_figure(mixed, comparison, output)
    print(f"Four verified compact figures (PNG/PDF) written to {output}. No models refitted.")
    if result.archive:
        result.archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed run ZIP or directory")
    parser.add_argument("--review", type=Path, required=True, help="Audited within-locus review directory")
    parser.add_argument("--mixed", type=Path, required=True, help="Mixed-block r > 0.90 review directory")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
