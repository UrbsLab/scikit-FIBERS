"""Audit completed outputs and make compact talk figures without rerunning models.

Reads JSON/CSV results only, never model pickles or patient identifiers. Example:
python review_outputs.py --input /path/simple.zip --output /path/talk_review
"""
import argparse
import io
import itertools
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


class Results:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.duplicate_files = 0
        self.prefix = ""
        if self.archive:
            names = set(self.archive.namelist())
            if "summary/completed.json" not in names:
                candidates = [n.removesuffix("summary/completed.json") for n in names
                              if n.endswith("summary/completed.json")]
                if len(candidates) != 1:
                    raise ValueError("Cannot identify one completed result directory")
                self.prefix = candidates[0]
            else:
                for name in names:
                    if name.endswith("/") or "simple/" + name not in names:
                        continue
                    a, b = self.archive.getinfo(name), self.archive.getinfo("simple/" + name)
                    if (a.CRC, a.file_size) != (b.CRC, b.file_size):
                        raise ValueError(f"Conflicting duplicate output: {name}")
                    self.duplicate_files += 1

    def read(self, name):
        return self.archive.read(self.prefix + name) if self.archive else (self.path / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)), compression="gzip" if name.endswith(".gz") else None)


def jaccard(a, b):
    return len(a & b) / len(a | b)


def features(record):
    return {f for group in record["groups"] for f in group["features"]}


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def locus_threshold_review(records, settings, output):
    """Change one locus's feature lists at a time; this does not recalculate HR."""
    loci = [name for name in ["A", "B", "C", "DRB1", "DRB345", "DQA1", "DQB1"]
            if name in settings["columns"]["ranges"]]
    primary = "r" + str(float(settings["correlation"]["primary_threshold"])).replace(".", "p")
    by_fold = {}
    for record in records:
        by_fold.setdefault(record["fold"], {})[record["scheme"]] = record
    folds = sorted(by_fold)
    baseline = {fold: features(by_fold[fold][primary]) for fold in folds}
    baseline_score = np.mean([jaccard(a, b) for a, b in itertools.combinations(baseline.values(), 2)])

    def at_locus(names, locus):
        return {name for name in names if name.split("_")[1] == locus}

    rows = []
    thresholds = sorted(settings["correlation"]["thresholds"], reverse=True)
    for locus in loci:
        originals = [at_locus(by_fold[fold][primary]["original"], locus) for fold in folds]
        valid_pairs = [(i, j) for i, j in itertools.combinations(range(len(folds)), 2)
                       if originals[i] | originals[j]]
        for threshold in [None] + thresholds:
            scheme = "original" if threshold is None else "r" + str(float(threshold)).replace(".", "p")
            sets = originals if threshold is None else [at_locus(features(by_fold[f][scheme]), locus) for f in folds]
            hybrid = [(baseline[fold] - at_locus(baseline[fold], locus)) | selected
                      for fold, selected in zip(folds, sets)]
            locus_score = np.mean([jaccard(sets[i], sets[j]) for i, j in valid_pairs]) if valid_pairs else np.nan
            hybrid_score = np.mean([jaccard(a, b) for a, b in itertools.combinations(hybrid, 2)])
            rows.append({"locus": locus, "scheme": scheme, "threshold": threshold,
                         "locus_jaccard": locus_score, "locus_pairs": len(valid_pairs),
                         "folds_with_locus": sum(bool(s) for s in originals),
                         "mean_positions": np.mean([len(s) for s in sets]),
                         "mean_added_positions": np.mean([len(s - o) for s, o in zip(sets, originals)]),
                         "hybrid_full_bin_jaccard": hybrid_score,
                         "hybrid_delta_vs_primary": hybrid_score - baseline_score,
                         "hybrid_hazard_ratio_evaluated": False})
    table = pd.DataFrame(rows)
    table.to_csv(output / "locus_cutoff_comparison.csv", index=False)
    chosen = [r for r in [.9, .8, .6, .4, .2, .1] if r in thresholds]
    data = table.pivot(index="locus", columns="threshold", values="hybrid_delta_vs_primary").reindex(index=loci, columns=chosen)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    limit = max(.12, np.nanmax(np.abs(data.to_numpy())))
    plot = ax.imshow(data, cmap="BrBG", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for i, j in itertools.product(range(len(loci)), range(len(chosen))):
        value = data.iloc[i, j]
        ax.text(j, i, f"{value:+.3f}" if abs(value) >= .0005 else "0.000", ha="center", va="center",
                fontsize=16, color="white" if abs(value) > limit * .70 else "#222222")
    ax.set_xticks(range(len(chosen)), [f"{r:.2f}" for r in chosen])
    ax.set_yticks(range(len(loci)), [l.replace("DRB345", "DRB3/4/5") for l in loci])
    ax.set_xlabel(f"Pearson cutoff at the changed locus\n(other loci kept at r > {settings['correlation']['primary_threshold']:g})")
    ax.set_ylabel("Locus / locus group")
    fig.colorbar(plot, ax=ax, label="Change in full-bin Jaccard similarity")
    fig.tight_layout()
    save(fig, output, "locus_cutoff_effect")
    return table


def compact_result_figure(pairs, top, output):
    originals = top.loc[top.variant == "original"].set_index("fold")
    processed = top.loc[top.variant == "processed_fixed"].set_index("fold")
    fold_rows = []
    for fold in sorted(originals.index):
        values = pairs.loc[(pairs.fold_left == fold) | (pairs.fold_right == fold)]
        fold_rows.append({"fold": fold, "original": values.original.mean(),
                          "processed": values.processed.mean()})
    overlap = pd.DataFrame(fold_rows).set_index("fold")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, table, ylabel, limits in (
        (axes[0], overlap, "Overlap between top-bin feature lists\n(Jaccard index)", (0, .42)),
        (axes[1], pd.DataFrame({"original": originals.HR, "processed": processed.HR}),
         "Held-out hazard ratio\n(unadjusted)", (1, 1.32)),
    ):
        for row in table.itertuples():
            ax.plot([0, 1], [row.original, row.processed], color="#acb5ba", lw=1.5, zorder=1)
        mean = table[["original", "processed"]].mean()
        ax.plot([0, 1], mean, color="#14677b", lw=4, zorder=2)
        for x, value in enumerate(mean):
            ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 15), textcoords="offset points",
                        ha="center", fontsize=20, fontweight="bold", color="#115565",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 2})
        ax.set(xticks=[0, 1], xticklabels=["Original", "Processed"], xlim=(-.25, 1.25),
               ylim=limits, ylabel=ylabel)
        ax.grid(axis="y", color="#e9edef", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(-.17, 1.04, "A", transform=axes[0].transAxes, fontsize=22, fontweight="bold")
    axes[1].text(-.17, 1.04, "B", transform=axes[1].transAxes, fontsize=22, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color="#acb5ba", lw=2, label="One cross-validation fold"),
                        Line2D([], [], color="#14677b", lw=4, label="Mean")],
               loc="lower center", ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.92, wspace=.42)
    save(fig, output, "main_result")
    overlap.to_csv(output / "fold_overlap_for_figure.csv")


def inclusion_example(records, stable_blocks, output):
    candidates = []
    for bid, members in stable_blocks.items():
        touched = [r for r in records if any(g["block"] == bid for g in r["groups"])]
        added = sum(len(set(members) - set(r["original"])) for r in touched)
        if added:
            candidates.append((len(touched), added, bid, members))
    # Pick recurrent blocks, not favorable hazard ratios or selected validation folds.
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2]))
    chosen = candidates[:2]
    names = [name for _, _, _, members in chosen for name in members]
    matrices = [np.zeros((len(names), len(records))) for _ in range(2)]
    for j, record in enumerate(records):
        original, expanded = set(record["original"]), features(record)
        for i, name in enumerate(names):
            matrices[0][i, j] = 2 if name in original else 0
            matrices[1][i, j] = 2 if name in original else 1 if name in expanded else 0
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.2, len(names) * .28 + 1.6)))
    cmap = ListedColormap(["white", "#a6cdd6", "#173e4d"])
    for ax, values, label in zip(axes, matrices, ["Original top bins", "Processed top bins"]):
        ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(records)), [str(r["fold"]) for r in records])
        ax.set_xlabel(label + "\nCross-validation fold")
        ax.set_yticks(range(len(names)), [n.removeprefix("MM_").replace("DRB345", "DRB3/4/5").replace("_", " ") for n in names])
        ax.axhline(len(chosen[0][3]) - .5, color="#666666", lw=1.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mismatch position (two example blocks)")
    fig.legend(handles=[Patch(facecolor="#173e4d", label="Originally selected"),
                        Patch(facecolor="#a6cdd6", label="Added correlated alternative"),
                        Patch(facecolor="white", edgecolor="#aaa", label="Absent")],
               loc="upper center", ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(left=.12, right=.99, top=.90, bottom=.15, wspace=.40)
    save(fig, output, "feature_inclusion_example")
    return [{"block": bid, "features": members, "top_bins_touched": count, "positions_added_total": added}
            for count, added, bid, members in chosen]


def run(source, output):
    result = Results(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = result.json("summary/completed.json")["settings"]
    if len(settings["imputations"]) != 1 or len(settings["seeds"]) != 1:
        raise ValueError("This compact review expects one imputation and one seed; do not silently pool them")
    imp, seed = settings["imputations"][0], settings["seeds"][0]
    cutoff = settings["correlation"]["primary_threshold"]
    primary = "r" + str(float(cutoff)).replace(".", "p")
    records, all_records, audits, stable_blocks = [], [], [], None
    for fold in settings["folds"]:
        root = f"imp_{imp:02d}/cv_{fold:02d}"
        fit = result.json(root + f"/seed_{seed:03d}/completed.json")
        corr = result.json(root + "/correlation/completed.json")
        assert fit["split_audit"] == corr["split_audit"], "Mismatched split metadata"
        population = result.csv(root + f"/seed_{seed:03d}/population.csv")
        top_population = population.loc[population["rank"] == 1].iloc[0]
        subset = [r for r in result.json(root + "/correlation/processed_bins.json") if r["rank"] == 1 and r["seed"] == seed]
        record = next(r for r in subset if r["scheme"] == primary)
        assert set(record["original"]) == set(json.loads(top_population.features))
        assert set(record["original"]) <= features(record)
        assert record["original_threshold"] == top_population.threshold
        all_records += subset
        records.append(record)
        schemes = result.json(root + "/correlation/blocks.json")
        pairs = result.csv(root + "/correlation/correlations.csv.gz")
        lookup = {frozenset((r.feature1, r.feature2)): r.pearson_r for r in pairs.itertuples()}
        for block in schemes[primary]["blocks"]:
            actual = min(lookup[frozenset(pair)] for pair in itertools.combinations(block["features"], 2))
            assert actual > cutoff
            assert np.isclose(actual, block["min_r"])
        blocks = {b["block"]: b["features"] for b in schemes[primary]["blocks"]}
        stable_blocks = blocks if stable_blocks is None else {k: v for k, v in stable_blocks.items() if blocks.get(k) == v}
        audits.append({"fold": fold, "candidate_features": len(result.csv(root + "/correlation/feature_filter.csv")),
                       "retained_features": len(fit["features"]), "train_n": fit["split_audit"]["train_n"],
                       "test_n": fit["split_audit"]["test_n"], "primary_blocks": len(blocks),
                       "block_pairs_checked": True})
    records.sort(key=lambda r: r["fold"])
    paired = pd.DataFrame([{"fold_left": a["fold"], "fold_right": b["fold"],
                           "original": jaccard(set(a["original"]), set(b["original"])),
                           "processed": jaccard(features(a), features(b))}
                          for a, b in itertools.combinations(records, 2)])
    paired["delta"] = paired.processed - paired.original
    recorded = result.csv("summary/consistency_summary.csv")
    for scheme, field in [("original", "original"), (primary, "processed")]:
        value = recorded.loc[(recorded.comparison == "across_cv") & (recorded.scheme == scheme), "jaccard"].item()
        assert np.isclose(value, paired[field].mean())
    risk = result.csv("summary/risk_comparison.csv.gz")
    top_all = risk.loc[(risk["rank"] == 1) & (risk.dataset == "test")].copy()
    top = top_all.loc[(top_all.variant == "original") | ((top_all.scheme == primary) & (top_all.variant == "processed_fixed"))].copy()
    assert len(top) == 2 * len(settings["folds"])
    assert top["HR status"].eq("ok").all()
    base = top.loc[top.variant == "original"].set_index("fold")
    processed = top.loc[top.variant == "processed_fixed"].set_index("fold")
    assert base.index.equals(processed.index)
    comparison = pd.DataFrame({"original_HR": base.HR, "processed_HR": processed.HR,
                               "HR_delta": processed.HR - base.HR, "n_test": base.n,
                               "changed_count": np.rint(processed.changed_group_fraction * processed.n).astype(int),
                               "original_logrank": base.logrank, "processed_logrank": processed.logrank})
    for metric in ["HR", "Adj HR", "Adj NoAg HR"]:
        for suffix in [" lower", " upper", " p"]:
            comparison[metric + suffix + " original"] = base[metric + suffix]
            comparison[metric + suffix + " processed"] = processed[metric + suffix]
    unchanged = comparison.changed_count.eq(0)
    assert np.allclose(comparison.loc[unchanged, "HR_delta"], 0)
    for metric in ["Adj HR", "Adj NoAg HR"]:
        comparison[metric + " original"] = base[metric]
        comparison[metric + " processed"] = processed[metric]
        comparison[metric + " original status"] = base[metric + " status"]
        comparison[metric + " processed status"] = processed[metric + " status"]
    sensitivity = []
    for scheme in dict.fromkeys(r["scheme"] for r in all_records):
        rs = [r for r in all_records if r["scheme"] == scheme]
        r = top_all.loc[(top_all.scheme == scheme) & (top_all.variant == "processed_fixed")]
        sensitivity.append({"scheme": scheme, "mean_jaccard": np.mean([jaccard(features(a), features(b)) for a, b in itertools.combinations(rs, 2)]),
                            "mean_unadjusted_HR": r.HR.mean(), "mean_logrank": r.logrank.mean(),
                            "changed_patients": int(np.rint(r.changed_group_fraction * r.n).sum()),
                            "total_patients": int(r.n.sum()), "mean_positions": r.positions.mean(),
                            "smallest_group_fraction": np.minimum(r.n_above / r.n, 1 - r.n_above / r.n).min()})
    within = recorded.loc[(recorded.comparison == "within_population") & recorded.scheme.isin(["original", primary])]
    within = within.pivot(index="scope", columns="scheme", values="jaccard")
    cohort = result.csv("summary/cv_validation.csv")
    assert cohort.partition_valid.all()
    assert int(base.n.sum()) == int(cohort.unique_transplants.sum())
    summary = {"source": str(result.path), "duplicate_archive_files_ignored": result.duplicate_files,
               "imputation": imp, "seed": seed, "folds": len(settings["folds"]),
               "primary_threshold": cutoff, "fixed_bin_thresholds": sorted(base.threshold.unique().tolist()),
               "n_transplants": int(base.n.sum()), "pair_count": len(paired),
               "mean_jaccard_original": paired.original.mean(), "mean_jaccard_processed": paired.processed.mean(),
               "mean_jaccard_delta": paired.delta.mean(), "pairs_increased": int((paired.delta > 0).sum()),
               "pairs_decreased": int((paired.delta < 0).sum()), "pairs_unchanged": int((paired.delta == 0).sum()),
               "mean_unadjusted_HR_original": base.HR.mean(), "mean_unadjusted_HR_processed": processed.HR.mean(),
               "mean_unadjusted_HR_delta": comparison.HR_delta.mean(),
               "mean_logrank_original": base.logrank.mean(), "mean_logrank_processed": processed.logrank.mean(),
               "changed_patients": int(comparison.changed_count.sum()),
               "unchanged_folds": int(comparison.changed_count.eq(0).sum()),
               "adjusted_failed_folds": base.index[base["Adj HR"].isna()].tolist(),
               "adjusted_warning_folds": base.index[base["Adj HR status"].str.startswith("warning:")].tolist(),
               "adjusted_original_mean_available_only": base["Adj HR"].mean(),
               "adjusted_processed_mean_available_only": processed["Adj HR"].mean(),
               "adjusted_pairs_available": int((base["Adj HR"].notna() & processed["Adj HR"].notna()).sum()),
               "within_population_mean_original": within.original.mean(),
               "within_population_mean_processed": within[primary].mean(),
               "within_population_improved_folds": int((within[primary] > within.original).sum())}
    paired.to_csv(output / "top_bin_overlap_pairs.csv", index=False)
    comparison.to_csv(output / "top_bin_fold_comparison.csv")
    pd.DataFrame(sensitivity).to_csv(output / "cutoff_sensitivity.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "audit_by_fold.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 15, "ytick.labelsize": 14, "pdf.fonttype": 42})
    compact_result_figure(paired, top, output)
    summary["illustrative_blocks"] = inclusion_example(records, stable_blocks, output)
    locus_threshold_review(all_records, settings, output)
    (output / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed output ZIP or directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
