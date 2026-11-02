"""Plot top-bin recurrence and descriptive CV statistics from FIBERS exports.

Requires numpy and matplotlib. Uses CSV/TXT files only; never loads pickle files.
Run with --results /path/to/results --o /path/to/report --pdf /path/to/report.pdf.
Reads the saved first population row (Bin_0) in every fold. No pickle is loaded.
"""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True, help='Folder containing numbered CV exports')
parser.add_argument('--o', type=Path, required=True, help='New report output directory')
parser.add_argument('--pdf', type=Path, help='PDF path (default: output directory/report.pdf)')
args = parser.parse_args()
SOURCE = args.results.resolve()
ROOT = args.o.resolve()
RUN_LABEL = SOURCE.name.split('_product_')[-1].replace('_adaptive_', ' / adaptive ')
if SOURCE == ROOT or SOURCE in ROOT.parents:
    raise ValueError('Use a report directory outside the input results directory')
DATA, PNG, SVG = ROOT / 'tables', ROOT / 'png', ROOT / 'svg'
PDF = args.pdf.resolve() if args.pdf else ROOT / 'report.pdf'
for directory in (DATA, PNG, SVG, PDF.parent):
    directory.mkdir(parents=True, exist_ok=True)
provenance = {'source_directory': str(SOURCE), 'files': {}}
for path in sorted(SOURCE.glob('*/*')):
    if path.suffix in ('.csv', '.txt'):
        provenance['files'][str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT / 'input_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

INK, MUTED, TEAL = "#192E43", "#526577", "#087F8C"
GRID, EMPTY, FILTERED, ORANGE = "#DFE6EB", "#F5F7F9", "#B6C0CA", "#B65A28"
LOCI = ["A", "B", "C", "DRB1", "DRB345", "DQA1", "DQB1"]
COLORS = dict(zip(LOCI, ["#237A8C", "#348FA0", "#68AEB8", "#675A94",
                        "#8872A9", "#A16C82", "#BE8394"]))
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID, "xtick.color": MUTED,
    "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, records):
    with (DATA / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def feature_parts(name):
    match = re.fullmatch(r"MM_(.+)_(\d+)_r([A-Z])_d([A-Z])", name)
    if not match:
        raise ValueError(f"Unexpected feature name: {name}")
    locus, position, recipient, donor = match.groups()
    return locus, int(position), recipient, donor


def feature_key(name):
    locus, position, recipient, donor = feature_parts(name)
    return LOCI.index(locus), position, donor, recipient


def feature_label(name, include_locus=True):
    locus, position, recipient, donor = feature_parts(name)
    return f"{locus + '  ' if include_locus else ''}{position}  |  {donor} → {recipient}"


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "n_available": len(values), "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


folds = sorted(int(p.name) for p in SOURCE.iterdir() if p.is_dir() and p.name.isdigit())
assert folds == list(range(1, 11)), f"Expected CV1-CV10; found {folds}"
fold_rows, cox_rows, parameters, selected_sets, retained_sets = [], [], [], [], []
pkpra_rows = []
model_files = {
    "Unadjusted": "coxph_unadj_bin_{split}_0",
    "Base adjusted": "coxph_adj_bin_{split}_0_NoAg",
    "Fully adjusted": "coxph_adj_bin_{split}_0",
}
for fold in folds:
    folder = SOURCE / str(fold)
    population = read_csv(folder / f"{fold}_pop.csv")
    top = population[0]
    features = ast.literal_eval(top["feature_list"])
    assert len(set(features)) == len(features) == int(top["bin_size"])
    selected_sets.append(set(features))
    audit = read_csv(folder / f"{fold}_rare_filter_feature_frequencies.csv")
    retained = {r["feature"] for r in audit if r["kept_for_training"].lower() == "true"}
    assert set(features) <= retained
    assert all((float(r["nonzero_fraction"]) >= 0.1) == (r["feature"] in retained) for r in audit)
    retained_sets.append(retained)
    params = dict(line.split(": ", 1) for line in (folder / f"{fold}_run_parameters.txt").read_text().splitlines()
                  if ": " in line)
    parameters.append(params)
    count_low, count_mid, count_high = (int(top[k]) for k in ("count_bt", "count_mt", "count_at"))
    assert count_mid == 0 and len(ast.literal_eval(top["group_threshold_list"])) == 1
    record = {
        "cv": fold, "random_seed": int(params["random_seed"]),
        "bin_size": len(features), "threshold": float(top["group_threshold"]),
        "train_n": count_low + count_high, "train_at_or_below_threshold": count_low,
        "train_above_threshold": count_high,
        "train_above_threshold_pct": 100 * count_high / (count_low + count_high),
        "birth_iteration": int(top["birth_iteration"]),
        "train_log_rank_score": float(top["log_rank_score"]),
        "train_log_rank_p": float(top["log_rank_p_value"]),
        "train_residual_score": float(top["residuals_score"]),
        "train_pre_fitness": float(top["pre_fitness"]),
        "retained_features": len(retained), "population_size": len(population),
        "feature_list": json.dumps(features),
    }
    for locus in LOCI:
        record[f"bin_features_{locus}"] = sum(feature_parts(f)[0] == locus for f in features)
    for split in ("train", "test"):
        for model, template in model_files.items():
            filename = f"{fold}_" + template.format(split=split)
            csv_file, failure_file = folder / f"{filename}.csv", folder / f"{filename}_failed.txt"
            result = {"cv": fold, "split": split, "model": model, "status": "ok",
                      "hr": None, "ci_lower": None, "ci_upper": None, "p": None,
                      "adjustment_covariates": None, "failure_reason": "", "source_file": ""}
            if csv_file.exists():
                table = read_csv(csv_file)
                for coefficient in table:
                    if coefficient['covariate'] == 'PKPRA_MS':
                        pkpra_rows.append({
                            'cv': fold, 'split': split, 'model': model,
                            'coef': float(coefficient['coef']),
                            'se': float(coefficient['se(coef)']),
                            'source_file': str(csv_file.relative_to(SOURCE)),
                        })
                bin_row = next(r for r in table if r["covariate"] == "Bin_0")
                result.update(
                    hr=float(bin_row["exp(coef)"]), ci_lower=float(bin_row["exp(coef) lower 95%"]),
                    ci_upper=float(bin_row["exp(coef) upper 95%"]), p=float(bin_row["p"]),
                    adjustment_covariates=len(table) - 1,
                    source_file=str(csv_file.relative_to(SOURCE)))
                assert 0 < result["ci_lower"] <= result["hr"] <= result["ci_upper"]
                assert 0 <= result["p"] <= 1
                covariates = {r["covariate"] for r in table} - {"Bin_0"}
                if model == "Base adjusted":
                    assert len(covariates) == 43 and "Agmm0" in covariates
                    base_covariates = covariates
                if model == "Fully adjusted":
                    assert covariates - base_covariates == {
                        "AgmmA0", "AgmmA1", "AgmmB0", "AgmmB1", "AgmmDR0", "AgmmDR1"}
            elif failure_file.exists():
                result.update(status="failed", failure_reason=failure_file.read_text().strip(),
                              source_file=str(failure_file.relative_to(SOURCE)))
            else:
                raise FileNotFoundError(f"No estimate or failure report for {filename}")
            cox_rows.append(result)
            prefix = f"{split}_{model.lower().replace(' ', '_')}"
            for key in ("status", "hr", "ci_lower", "ci_upper", "p"):
                record[f"{prefix}_{key}"] = result[key]
    fold_rows.append(record)
write_csv("fold_summary.csv", fold_rows)
write_csv("cox_estimates_long.csv", cox_rows)
write_csv("failed_models.csv", [r for r in cox_rows if r["status"] != "ok"])
write_csv("pkpra_ms_estimates.csv", pkpra_rows)

for key in ("random_seed", "iterations", "pop_size", "fitness_metric", "diversity_pressure",
            "min_thresh", "max_thresh", "thresh_evolve_prob", "group_thresh", "covariates"):
    assert len({p[key] for p in parameters}) == 1
feature_counts = Counter(feature for selected in selected_sets for feature in selected)
all_features = sorted(feature_counts, key=feature_key)
recurrent = sorted((f for f in all_features if feature_counts[f] >= 4),
                   key=lambda f: (-feature_counts[f], feature_key(f)))
selection = np.array([[int(f in selected) for f in all_features] for selected in selected_sets])
eligibility = np.array([[int(f in retained) for f in all_features] for retained in retained_sets])
assert np.all(selection <= eligibility)
assert selection.sum(axis=1).tolist() == [r["bin_size"] for r in fold_rows]
assert selection.sum(axis=0).tolist() == [feature_counts[f] for f in all_features]
for filename, matrix in [("selection_matrix.csv", selection), ("eligibility_matrix.csv", eligibility)]:
    write_csv(filename, [{"feature": f, "locus": feature_parts(f)[0],
                          **{f"CV{fold}": int(matrix[i, j]) for i, fold in enumerate(folds)}}
                         for j, f in enumerate(all_features)])
recurrence_rows = []
for f in sorted(all_features, key=lambda f: (-feature_counts[f], feature_key(f))):
    locus, position, recipient, donor = feature_parts(f)
    recurrence_rows.append({
        "feature": f, "locus": locus, "position": position, "donor": donor, "recipient": recipient,
        "selected_folds": feature_counts[f], "selected_pct": 100 * feature_counts[f] / len(folds),
        "eligible_folds": sum(f in r for r in retained_sets),
        "cv_list": ",".join(str(fold) for fold, selected in zip(folds, selected_sets) if f in selected),
    })
write_csv("feature_recurrence.csv", recurrence_rows)
jaccard = np.array([[len(a & b) / len(a | b) for b in selected_sets] for a in selected_sets])
pairwise = [float(jaccard[i, j]) for i in range(10) for j in range(i+1, 10)]
write_csv("pairwise_jaccard.csv", [
    {"cv": fold, **{f"CV{other}": float(jaccard[i, j]) for j, other in enumerate(folds)}}
    for i, fold in enumerate(folds)])
aggregate = []
for metric in ("bin_size", "threshold", "train_n", "train_above_threshold_pct",
               "birth_iteration", "retained_features", "train_log_rank_score", "train_residual_score", "train_pre_fitness"):
    aggregate.append({"metric": metric, **stats([r[metric] for r in fold_rows])})
aggregate.append({"metric": "pairwise_jaccard_45_pairs", **stats(pairwise)})
write_csv("descriptive_statistics.csv", aggregate)
hr_summary = []
for split in ("train", "test"):
    for model in model_files:
        rows = [r for r in cox_rows if r["split"] == split and r["model"] == model]
        valid = [r for r in rows if r["status"] == "ok"]
        hr_summary.append({
            "split": split, "model": model, **stats([r["hr"] for r in valid]),
            "n_failed": len(rows) - len(valid),
            "nominal_p_below_0_05": sum(r["p"] < 0.05 for r in valid),
            "ci_entirely_above_1": sum(r["ci_lower"] > 1 for r in valid),
            "failed_folds": ",".join(str(r["cv"]) for r in rows if r["status"] != "ok"),
        })
write_csv("hazard_ratio_summary.csv", hr_summary)
summary = {
    "run": SOURCE.name, "parameters": parameters[0],
    "folds": folds, "seed": int(parameters[0]["random_seed"]),
    "union_features": len(all_features), "features_in_all_folds": [f for f in all_features if feature_counts[f] == 10],
    "features_in_at_least_4_folds": len(recurrent),
    "selection_frequency_distribution": dict(sorted(Counter(feature_counts.values()).items())),
    "bin_size": stats([r["bin_size"] for r in fold_rows]),
    "threshold": stats([r["threshold"] for r in fold_rows]),
    "jaccard": stats(pairwise),
    "descriptive_hr_summaries": hr_summary,
    "base_model": "43 covariates including Agmm0",
    "full_model": "Base plus 6 indicators: AgmmA0/A1, AgmmB0/B1, AgmmDR0/DR1",
    "estimand_note": "Each fold has its own selected bin and threshold; HR means are descriptive, not pooled estimates.",
    "test_sample_size_note": "Test sample sizes are not supplied; no counts were inferred or summed.",
    "cox_status_note": "ok means a coefficient CSV was exported, not independently verified convergence. PKPRA_MS estimates are audited separately.",
}
(DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
full_test = next(r for r in hr_summary if r['split'] == 'test' and r['model'] == 'Fully adjusted')
failed_label = ', '.join('CV' + value for value in full_test['failed_folds'].split(',') if value) or 'none'



def canvas(title, subtitle, page, appendix=False):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.055, .955, f"FIBERS  /  {RUN_LABEL}  /  10 CV FOLDS  /  SEED {summary['seed']}" + ("  /  FULL FEATURE DETAIL" if appendix else ""),
             fontsize=11, color=TEAL, weight="bold", va="top")
    fig.text(.055, .905, title, fontsize=27, weight="bold", va="top")
    fig.text(.055, .846, subtitle, fontsize=14, color=MUTED, va="top")
    fig.add_artist(Line2D([.055, .945], [.07, .07], transform=fig.transFigure, color=GRID))
    fig.text(.055, .043, "One selected bin per fold. Training sets overlap; summaries are descriptive, not pooled estimates.",
             fontsize=10.5, color=MUTED)
    fig.text(.945, .043, str(page).zfill(2), ha="right", fontsize=11, color=MUTED)
    return fig


def export(fig, name, pdf):
    fig.savefig(PNG / f"{name}.png", dpi=300)
    fig.savefig(SVG / f"{name}.svg")
    pdf.savefig(fig)
    plt.close(fig)


def rgb_matrix(features):
    colors = np.full((len(features), len(folds), 4), to_rgba(EMPTY))
    for row, feature in enumerate(features):
        for col, (selected, retained) in enumerate(zip(selected_sets, retained_sets)):
            colors[row, col] = to_rgba(COLORS[feature_parts(feature)[0]] if feature in selected
                                      else EMPTY if feature in retained else FILTERED)
    return colors


def membership_legend(fig, y=.115):
    fig.legend(handles=[Patch(facecolor=TEAL, label="Selected (color indicates locus)"),
                        Patch(facecolor=EMPTY, edgecolor=GRID, label="Eligible, not selected"),
                        Patch(facecolor=FILTERED, label="Not eligible after filtering")],
               bbox_to_anchor=(.50, y), loc="center", ncol=3, frameon=False, fontsize=11)


def feature_heatmap(fig, features, rect, font_size=13, locus_labels=True):
    ax = fig.add_axes(rect)
    ax.imshow(rgb_matrix(features), interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"CV{f}" for f in folds], fontsize=13)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([feature_label(f, locus_labels) for f in features], fontsize=font_size)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, feature in enumerate(features):
        ax.text(10.05, i, f"{feature_counts[feature]}/10", fontsize=font_size,
                va="center", clip_on=False, color=MUTED)
    return ax


with PdfPages(PDF) as pdf:
    pdf.infodict().update(Title=f"FIBERS {RUN_LABEL} - all CV heatmaps and descriptive summary",
                          Author="Harsh Bandhey", Subject="Top-bin feature recurrence across ten folds")
    # Full union overview: keep every feature; individual labels appear in appendix.
    fig = canvas(f"All ten selected bins, across {len(all_features)} distinct permutations",
                 "Each column is one permutation; selected cells are colored by locus. Columns are ordered by locus and position.", 1)
    ax = fig.add_axes([.095, .31, .75, .41])
    ax.imshow(rgb_matrix(all_features).transpose(1, 0, 2), interpolation="nearest", aspect="auto")
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels([f"CV{f}" for f in folds])
    ax.set_xticks([])
    ax.tick_params(length=0)
    start = 0
    for locus in LOCI:
        count = sum(feature_parts(f)[0] == locus for f in all_features)
        ax.text(start + (count-1)/2, -1.05, f"{locus}\n{count} features", ha="center",
                va="bottom", color=COLORS[locus], fontsize=12, weight="bold", clip_on=False)
        if start:
            ax.axvline(start-.5, color="white", linewidth=2)
        start += count
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linewidth=1.7)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.87, .752, "SIZE  /  CUTOFF", fontsize=10.5, color=MUTED, weight="bold")
    for i, record in enumerate(fold_rows):
        fig.text(.885, .31 + .41 * (1-(i+.5)/10),
                 f"{record['bin_size']:2}  /  {int(record['threshold'])}",
                 fontsize=14, va="center")
    fig.text(.055, .225, f"{len(summary['features_in_all_folds'])} permutations are selected in all 10 folds.", fontsize=21, weight="bold", color=TEAL)
    fig.text(.055, .172, f"Exact labels for all {len(all_features)} permutations are provided in the locus-specific heatmaps and CSV matrix.",
             fontsize=13, color=MUTED)
    membership_legend(fig, .113)
    export(fig, "01_all_features_overview", pdf)

    # Readable recurring-feature figure for the main presentation.
    fig = canvas("Recurring permutations across the ten folds",
                 f"{len(recurrent)} features selected in at least 4 folds; labels show locus, position and donor → recipient.", 2)
    feature_heatmap(fig, recurrent, [.28, .19, .54, .59],
                    font_size=11 if len(recurrent) > 25 else 14)
    fig.text(.84, .799, "FOLDS", fontsize=11, color=MUTED, weight="bold")
    membership_legend(fig)
    export(fig, "02_recurring_features_heatmap", pdf)

    # Composition counts retain all features, not only the recurrent subset.
    fig = canvas("Bin size and locus composition vary across folds",
                 "Cell values count selected permutations at each locus; the right columns give total bin size and cutoff.", 3)
    counts = np.array([[r[f"bin_features_{locus}"] for locus in LOCI] for r in fold_rows])
    ax = fig.add_axes([.16, .24, .57, .53])
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=int(counts.max()), aspect="auto")
    ax.set_xticks(np.arange(7)); ax.set_xticklabels(LOCI, fontsize=14); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(10)); ax.set_yticklabels([f"CV{f}" for f in folds], fontsize=14)
    ax.tick_params(length=0, pad=10)
    for i in range(10):
        for j in range(7):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if counts[i, j] > counts.max()*.60 else INK)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    fig.text(.78, .8, "BIN SIZE", weight="bold", color=MUTED, fontsize=12)
    fig.text(.88, .8, "CUTOFF", weight="bold", color=MUTED, fontsize=12)
    for i, record in enumerate(fold_rows):
        y = .24 + .53 * (1-(i+.5)/10)
        fig.text(.80, y, str(record["bin_size"]), ha="center", va="center", fontsize=16)
        fig.text(.90, y, str(int(record["threshold"])), ha="center", va="center", fontsize=16)
    fig.text(.055, .15, f"Mean bin size: {summary['bin_size']['mean']:.1f} (range {summary['bin_size']['min']:.0f}-{summary['bin_size']['max']:.0f}). Median cutoff: {summary['threshold']['median']:.1f} (range {summary['threshold']['min']:.0f}-{summary['threshold']['max']:.0f}).",
             fontsize=19, weight="bold")
    fig.text(.055, .104, "Each fold compares bin sum above its own cutoff against bin sum at or below that cutoff.",
             fontsize=12, color=MUTED)
    export(fig, "03_locus_composition", pdf)

    # Fold-wise test estimates; failed estimates remain visible.
    fig = canvas("Test-set associations attenuate after adjustment",
                 f"Fold-specific bin hazard ratios with 95% CIs. Adjusted test fits failed in {failed_label}.", 4)
    for col, model in enumerate(model_files):
        left = .105 + col*.30
        ax = fig.add_axes([left, .29, .24, .43])
        rows = [r for r in cox_rows if r["split"] == "test" and r["model"] == model]
        for i, result in enumerate(rows):
            if result["status"] != "ok":
                ax.text(1.11, i, "Fit failed", va="center", ha="center", fontsize=12, color=ORANGE)
                continue
            ax.errorbar(result["hr"], i,
                        xerr=[[result["hr"]-result["ci_lower"]], [result["ci_upper"]-result["hr"]]],
                        fmt="o", color=TEAL, capsize=3, markersize=6, linewidth=1.6)
        ax.axvline(1, color=MUTED, linestyle="--", linewidth=1)
        test_valid = [r for r in cox_rows if r['split'] == 'test' and r['status'] == 'ok']
        lo = min(.90, min(r['ci_lower'] for r in test_valid) * .98)
        hi = max(1.35, max(r['ci_upper'] for r in test_valid) * 1.02)
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(9.7, -.7)
        ax.xaxis.set_major_locator(FixedLocator([.9, 1.0, 1.1, 1.2, 1.3]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.9", "1.0", "1.1", "1.2", "1.3"]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([f"CV{f}" for f in folds] if col==0 else [])
        ax.set_xlabel("Hazard ratio (log scale)", fontsize=12)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        fig.text(left+.12, .779, model, ha="center", fontsize=18, weight="bold")
        desc = {"Unadjusted": "Bin only", "Base adjusted": "43 covariates, including Agmm0",
                "Fully adjusted": "Base + 6 locus Agmm indicators"}[model]
        fig.text(left+.12, .746, desc, ha="center", fontsize=10.5, color=MUTED)
        entry = next(r for r in hr_summary if r["split"]=="test" and r["model"]==model)
        fig.text(left+.12, .185, f"Mean HR {entry['mean']:.3f}  (n={entry['n_available']} folds)",
                 fontsize=13, ha="center", weight="bold")
        fig.text(left+.12, .144, f"95% CI above 1: {entry['ci_entirely_above_1']}/{entry['n_available']}",
                 fontsize=12, ha="center", color=MUTED)
    fig.text(.055, .10, "Available exports only; convergence is not verified. PKPRA_MS instability remains. Means are not pooled HRs.",
             fontsize=11.5, color=MUTED)
    export(fig, "04_test_hazard_ratios", pdf)

    # Descriptive summary: present denominators and variability explicitly.
    fig = canvas("Ten-fold descriptive summary",
                 f"One run per fold: seed {summary['seed']}, {parameters[0]['iterations']} iterations, population {parameters[0]['pop_size']}, log-rank × residuals fitness.", 5)
    cards = [
        ("DISTINCT FEATURES", str(len(all_features)), "Across the 10 selected bins"),
        ("SHARED BY ALL FOLDS", str(len(summary["features_in_all_folds"])), "Exact directional permutations"),
        ("MEAN BIN SIZE", f"{summary['bin_size']['mean']:.1f}", f"{summary['bin_size']['min']:.0f}-{summary['bin_size']['max']:.0f} selected features"),
        ("ADJUSTED TEST EXPORTS", f"{full_test['n_available']}/10", f"{full_test['n_failed']} failed folds"),
    ]
    for i, (label, value, note) in enumerate(cards):
        x = .055 + i*.23
        fig.text(x, .75, label, fontsize=11.5, color=TEAL, weight="bold")
        fig.text(x, .66, value, fontsize=32, weight="bold")
        fig.text(x, .61, note, fontsize=12, color=MUTED)
    fig.text(.055, .515, "TEST HAZARD RATIOS", fontsize=13, weight="bold", color=TEAL)
    headers = ["Model", "Folds", "Mean ± SD", "Median", "Range", "Nominal P<0.05"]
    rows = []
    for r in hr_summary:
        if r["split"] == "test":
            rows.append([r["model"], f"{r['n_available']}/10", f"{r['mean']:.3f} ± {r['sd']:.3f}",
                         f"{r['median']:.3f}", f"{r['min']:.3f}-{r['max']:.3f}",
                         f"{r['nominal_p_below_0_05']}/{r['n_available']}"])
    ax = fig.add_axes([.055, .27, .89, .215]); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", colLoc="left",
                     colWidths=[.19, .09, .20, .11, .18, .23], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(14)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(GRID if r==0 else EMPTY if r%2 else "white")
        cell.get_text().set_color(INK)
        if r==0: cell.get_text().set_weight("bold")
    fig.text(.055, .205, f"Mean pairwise feature overlap: {100*summary['jaccard']['mean']:.1f}% Jaccard across 45 fold pairs.",
             fontsize=18, weight="bold")
    fig.text(.055, .155, f"{sum(n==1 for n in feature_counts.values())} of {len(all_features)} distinct features occur in only one selected bin.",
             fontsize=14, color=MUTED)
    fig.text(.055, .105, "SD is between-fold sample SD; P values are nominal. Base adjustment includes Agmm0. Failed fits are excluded from HR means.",
             fontsize=11, color=MUTED)
    export(fig, "05_summary_statistics", pdf)

    # Pairwise set similarity is based on exact directional features.
    fig = canvas("Selected feature sets have limited overlap across folds",
                 "Jaccard overlap = shared selected features / features selected in either fold. Diagonal cells are omitted.", 6)
    ax = fig.add_axes([.12, .20, .45, .57])
    masked = np.ma.array(jaccard, mask=np.eye(10, dtype=bool))
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad(EMPTY)
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=max(pairwise), aspect="auto")
    ax.set_xticks(np.arange(10)); ax.set_xticklabels([f"CV{f}" for f in folds], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(10)); ax.set_yticklabels([f"CV{f}" for f in folds], fontsize=12)
    ax.tick_params(length=0, pad=7)
    for i in range(10):
        for j in range(10):
            if i != j:
                ax.text(j, i, f"{100*jaccard[i,j]:.0f}%", ha="center", va="center", fontsize=10.5,
                        color="white" if jaccard[i,j] > max(pairwise)*.6 else INK)
    fig.text(.65, .68, f"{100*summary['jaccard']['mean']:.1f}%", fontsize=42, color=TEAL, weight="bold")
    fig.text(.65, .61, "Mean pairwise overlap", fontsize=19, weight="bold")
    fig.text(.65, .52, f"Median: {100*summary['jaccard']['median']:.1f}%", fontsize=17)
    fig.text(.65, .46, f"Range: {100*min(pairwise):.1f}-{100*max(pairwise):.1f}%", fontsize=17)
    fig.text(.65, .37, "45 unique fold pairs", fontsize=15, color=MUTED)
    fig.text(.65, .29, "Exact locus, position and\nresidue direction must match.", fontsize=14, color=MUTED)
    fig.text(.055, .112, "Feature-set overlap describes selection recurrence; it does not measure prediction agreement or independent replication.",
             fontsize=12, color=MUTED)
    export(fig, "06_fold_overlap", pdf)

    # Full feature identities, not truncated: one readable page per locus.
    covered = []
    page = 7
    for locus in LOCI:
        features = [f for f in all_features if feature_parts(f)[0] == locus]
        covered.extend(features)
        # Balance continuation pages so a 31-feature locus is split 16/15.
        chunk_size = math.ceil(len(features) / math.ceil(len(features) / 30))
        chunks = [features[i:i+chunk_size] for i in range(0, len(features), chunk_size)]
        for part, subset in enumerate(chunks, 1):
            detail = f" (part {part}/{len(chunks)})" if len(chunks) > 1 else ''
            fig = canvas(f"{locus}: {len(features)} selected permutations{detail}",
                         "Position | donor → recipient. Every permutation selected in at least one fold is included.", page, appendix=True)
            feature_heatmap(fig, subset, [.26, .20, .57, .57],
                            font_size=12 if len(subset)>27 else 13, locus_labels=False)
            fig.text(.85, .80, "FOLDS", fontsize=11, color=MUTED, weight="bold")
            membership_legend(fig)
            export(fig, f"{page:02}_all_features_{locus}_{part}", pdf)
            page += 1
    assert set(covered) == set(all_features) and len(covered) == len(all_features)

    fig = canvas(f"Adjusted test Cox models failed in {full_test['n_failed']} folds",
                 'A saved coefficient table does not establish numerical stability; fit warnings are not retained.', page)
    ax = fig.add_axes([.14, .29, .57, .46])
    labels = [(split, model) for split in ('train', 'test') for model in model_files]
    status_matrix = np.array([[int(next(r for r in cox_rows if r['cv'] == f and r['split'] == s and r['model'] == m)['status'] == 'ok') for s,m in labels] for f in folds])
    ax.imshow(status_matrix, cmap=ListedColormap([ORANGE, '#D2E9E8']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(6)); ax.set_xticklabels([s.title() + '\n' + m.replace(' adjusted', '') for s,m in labels], fontsize=11)
    ax.xaxis.tick_top(); ax.set_yticks(range(10)); ax.set_yticklabels([f'CV{f}' for f in folds], fontsize=12)
    ax.tick_params(length=0, pad=8)
    for i in range(10):
        for j in range(6):
            ax.text(j, i, 'Exported' if status_matrix[i,j] else 'Failed', ha='center', va='center', fontsize=10, color=INK if status_matrix[i,j] else 'white')
    ax.set_xticks(np.arange(-.5,6,1), minor=True); ax.set_yticks(np.arange(-.5,10,1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)
    fig.text(.75, .68, f'{full_test["n_available"]}/10', fontsize=40, color=TEAL, weight='bold')
    fig.text(.75, .61, 'Adjusted test exports', fontsize=15, weight='bold')
    fig.text(.75, .52, 'Failures in', fontsize=14, color=MUTED)
    fig.text(.75, .46, failed_label, fontsize=13, weight='bold')
    pkpra_test = [r for r in pkpra_rows if r['split'] == 'test' and r['model'] == 'Fully adjusted']
    extreme = [r for r in pkpra_test if r['se'] > 100]
    extreme_label = ', '.join(f"CV{r['cv']}" for r in extreme)
    fig.text(.055, .21, f'PKPRA_MS has standard errors above 100 in {extreme_label}.', fontsize=19, weight='bold')
    fig.text(.055, .155, 'Those exported fits also need review. Raw fold data and omission refits are needed to diagnose this run.', fontsize=13, color=MUTED)
    fig.text(.055, .105, 'NoAg includes Agmm0. Generic failure files alone do not identify the responsible covariate.', fontsize=12, color=MUTED)
    export(fig, f'{page:02}_cox_fit_status', pdf)
    page += 1

    fig = canvas('Run settings and interpretation',
                 'Parameters below are read from all ten saved run-parameter files and checked for agreement.', page)
    config = [
        ('Diversity pressure', parameters[0]['diversity_pressure']),
        ('Threshold search range', f"{parameters[0]['min_thresh']}-{parameters[0]['max_thresh']} (integer)"),
        ('Threshold evolution probability', parameters[0]['thresh_evolve_prob']),
        ('Fixed threshold', parameters[0]['group_thresh']),
        ('Iterations / population / seed', f"{parameters[0]['iterations']} / {parameters[0]['pop_size']} / {summary['seed']}"),
        ('Adjustment', 'Base: 43 covariates; full: 49; both include PKPRA_MS'),
    ]
    for i, (label, value) in enumerate(config):
        y = .74 - .073*i
        fig.text(.055, y, label, fontsize=16, weight='bold')
        fig.text(.48, y, value, fontsize=15, color=MUTED)
    fig.text(.055, .26, 'Thresholds are learned from training data; each fold defines a different bin and group contrast.', fontsize=15)
    fig.text(.055, .205, 'HR size and nominal significance do not establish predictive superiority.', fontsize=15)
    fig.text(.055, .15, 'These exports contain no C-index, time-dependent AUC, calibration or Brier-score comparison.', fontsize=14, color=MUTED)
    fig.text(.055, .105, 'Source: ' + SOURCE.name, fontsize=11, color=MUTED)
    export(fig, f'{page:02}_run_settings', pdf)

summary_md = f"""# All-CV heatmaps and summary

Source: {SOURCE.name}. CV1-CV10, seed {summary['seed']}, {parameters[0]['iterations']} iterations,
population {parameters[0]['pop_size']}, DP{parameters[0]['diversity_pressure']}, log-rank x residuals fitness.
Threshold search {parameters[0]['min_thresh']}-{parameters[0]['max_thresh']},
threshold evolution probability {parameters[0]['thresh_evolve_prob']}. First population row (Bin_0) per fold.

## Findings

- {len(all_features)} distinct selected permutations; {len(summary['features_in_all_folds'])} occur in every fold.
- {len(recurrent)} appear in at least four folds; {sum(v == 1 for v in feature_counts.values())} appear in only one.
- Mean bin size {summary['bin_size']['mean']:.1f}, sample SD {summary['bin_size']['sd']:.2f}, range {summary['bin_size']['min']:.0f}-{summary['bin_size']['max']:.0f}.
- Median cutoff {summary['threshold']['median']:.1f}, range {summary['threshold']['min']:.0f}-{summary['threshold']['max']:.0f}.
- Mean pairwise Jaccard overlap {100 * summary['jaccard']['mean']:.1f}% (45 pairs).

## Test hazard ratios

| Adjustment | Available folds | Mean +/- SD | Median | Range | Nominal P < 0.05 |
| --- | --- | --- | --- | --- | --- |
"""
for r in hr_summary:
    if r['split'] == 'test':
        summary_md += f"| {r['model']} | {r['n_available']}/10 | {r['mean']:.3f} +/- {r['sd']:.3f} | {r['median']:.3f} | {r['min']:.3f}-{r['max']:.3f} | {r['nominal_p_below_0_05']}/{r['n_available']} |\n"
summary_md += f"""
Each fold has its own selected bin and cutoff. These are unweighted descriptive
HR averages, not pooled estimates. Training sets overlap. No pooled CI or combined
P value is calculated. Failed estimates remain missing, never zero or one.

## Cox failures

Adjusted test fits failed in {failed_label}. See tables/failed_models.csv for
model-by-model status. A status of `ok` means a coefficient CSV is available,
not independently verified convergence. PKPRA_MS standard errors exceed 100 in
the fully adjusted test exports for {extreme_label}; see tables/pkpra_ms_estimates.csv.
The recorded failure is only `CoxPHFitter could not estimate this model.`
The FIBERS methods catch the underlying exception, so these exports cannot tell
which covariate(s) caused non-convergence. Failure is a model-level event.

The NoAg model still includes Agmm0 among its 43 covariates. The fully adjusted
model adds six indicators: AgmmA0/A1, AgmmB0/B1, AgmmDR0/DR1 (49 covariates total).
Failure without those six indicators means they are not required for the failure;
it does not identify the responsible base covariate. A shared issue in base
covariates, the bin indicator, or their joint design still needs examination.

Use job_diagnose_cox.py with the matching raw fold CSVs to capture the real errors,
check constants, near-zero variance, sparse categories, event-specific variance,
and linear dependencies. Optional univariate and leave-one-covariate-out fits
help localize suspects. A successful refit after omission is diagnostic evidence,
not a reason to choose adjustment variables using test outcomes.

## Figures and interpretation

Figures 1-6 reproduce the earlier overview, recurrent-feature heatmap, locus
composition, test HR forest plots, descriptive statistics, and pairwise overlap.
The appendix shows every selected feature, with at most 30 rows per heatmap.
The final two pages summarize Cox export status, coefficient instability and run settings.

Direction is donor to recipient: source r = recipient, d = donor. Colored means
selected; near-white means eligible but not selected; gray means filtered out.
The 10% nonzero-frequency rule is checked against each fold's saved audit.
Test sample sizes and event counts are not in these exports; none are inferred.
The folder does not include partition reports, so the splitting method cannot
be independently verified from these model exports alone.

Files in tables/ contain all underlying numbers. input_provenance.json records
SHA-256 hashes of source CSV/TXT files. Pickles are never loaded.
"""
(ROOT / 'README.md').write_text(summary_md)
print(json.dumps(summary, indent=2))
print(f'PDF: {PDF}')
