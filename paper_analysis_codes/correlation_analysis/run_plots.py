"""Combine completed fold results, check the CV partition and create figures."""
from common import configure_worker
configure_worker()

from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (begin_result, finish_result, fold_dir, identity, require_result, worker_args)
from data import id_digest
from methods import jaccard, scheme_name
from plotting import inclusion_plot, interlocus_figures, threshold_plot
from run_correlation import fit_results


def check_partition(config, directories):
    reference = None
    records = []
    identifier = config["input"]["id_column"]
    for imputation in config["imputations"]:
        expected = None
        counts = Counter()
        for fold in config["folds"]:
            ids = pd.read_csv(directories[imputation, fold] / "split_ids.csv.gz", dtype={identifier: str})
            train_ids = set(ids.loc[ids.split == "train", identifier])
            test_ids = set(ids.loc[ids.split == "test", identifier])
            cohort = train_ids | test_ids
            if train_ids & test_ids or len(ids) != len(cohort):
                raise ValueError(f"Nonunique or overlapping split identifiers: imp={imputation}, fold={fold}")
            if expected is not None and cohort != expected:
                raise ValueError(f"Cohort membership differs across folds: imp={imputation}")
            expected = cohort
            counts.update(test_ids)
        if set(counts) != expected or any(value != 1 for value in counts.values()):
            raise ValueError(f"Test sets do not form a one-time partition: imp={imputation}. Include every CV fold in config.")
        if reference is not None and expected != reference:
            raise ValueError("Cohort membership differs across imputations")
        reference = expected
        records.append({"imputation": imputation, "folds": len(config["folds"]),
                        "unique_transplants": len(expected), "cohort_sha256": id_digest(expected), "partition_valid": True})
    return pd.DataFrame(records)


def consistency_rows(records, config):
    rows = []
    names = list(dict.fromkeys(record["scheme"] for record in records))
    scopes = []
    for imp, seed in itertools.product(config["imputations"], config["seeds"]):
        scopes.append(("across_cv", f"imp{imp}_seed{seed}", [r for r in records if r["imputation"] == imp and r["seed"] == seed and r["rank"] == 1]))
    for fold, seed in itertools.product(config["folds"], config["seeds"]):
        scopes.append(("across_imputations", f"cv{fold}_seed{seed}", [r for r in records if r["fold"] == fold and r["seed"] == seed and r["rank"] == 1]))
    for imp, fold, seed in itertools.product(config["imputations"], config["folds"], config["seeds"]):
        scopes.append(("within_population", f"imp{imp}_cv{fold}_seed{seed}", [r for r in records if (r["imputation"], r["fold"], r["seed"]) == (imp, fold, seed)]))
    for comparison, scope, items in scopes:
        if not items:
            continue
        for name in ["original"] + names:
            selected = [r for r in items if r["scheme"] == (names[0] if name == "original" else name)]
            sets = [set(r["original"]) if name == "original" else {f for group in r["groups"] for f in group["features"]} for r in selected]
            for (i, left), (j, right) in itertools.combinations(enumerate(sets), 2):
                task = lambda r: f"imp{r['imputation']}_cv{r['fold']}_seed{r['seed']}_bin{r['rank']}"
                rows.append({"comparison": comparison, "scope": scope, "scheme": name,
                             "left": task(selected[i]), "right": task(selected[j]),
                             "jaccard": jaccard(left, right), "positions": (len(left) + len(right)) / 2})
    return pd.DataFrame(rows, columns=["comparison", "scope", "scheme", "left", "right", "jaccard", "positions"])


def aggregate_interlocus(config, directories):
    summaries, sensitivities = [], []
    cutoffs = sorted(set(config["plots"]["interlocus_reporting_thresholds"] + [config["plots"]["interlocus_threshold"]]))
    for imp in config["imputations"]:
        tables = []
        for fold in config["folds"]:
            frame = pd.read_csv(directories[imp, fold] / "correlations.csv.gz")
            frame = frame.loc[frame.locus1 != frame.locus2].copy()
            frame["fold"] = fold
            for i, cutoff in enumerate(cutoffs):
                frame[f"above_{i}"] = (frame.pearson_r > cutoff).astype(int)
            tables.append(frame)
        pairs = pd.concat(tables, ignore_index=True)
        keys = ["locus1", "locus2", "feature1", "feature2"]
        aggregation = {"mean_r": ("pearson_r", "mean"), "min_r": ("pearson_r", "min"),
                       "max_r": ("pearson_r", "max"), "folds_available": ("fold", "nunique")}
        aggregation.update({f"above_{i}": (f"above_{i}", "sum") for i in range(len(cutoffs))})
        combined = pairs.groupby(keys, sort=False).agg(**aggregation).reset_index()
        for i, cutoff in enumerate(cutoffs):
            for (left, right), group in combined.groupby(["locus1", "locus2"], sort=False):
                sensitivities.append({"imputation": imp, "locus1": left, "locus2": right, "cutoff": cutoff,
                                      "pairs_at_least_1_fold": int((group[f"above_{i}"] >= 1).sum()),
                                      "pairs_at_least_3_folds": int((group[f"above_{i}"] >= 3).sum()),
                                      "pairs_all_folds": int((group[f"above_{i}"] == len(config["folds"])).sum())})
        for i, cutoff in enumerate(cutoffs):
            combined.rename(columns={f"above_{i}": f"folds_r_gt_{cutoff:g}"}, inplace=True)
        combined["folds_above_plot_cutoff"] = combined[f"folds_r_gt_{config['plots']['interlocus_threshold']:g}"]
        combined["imputation"] = imp
        summaries.append(combined)
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(sensitivities)


def run(config, force=False):
    directories, dependencies = {}, []
    for imp, fold in itertools.product(config["imputations"], config["folds"]):
        _, fits = fit_results(config, imp, fold)
        directory = fold_dir(config, imp, fold) / "correlation"
        require_result(directory, identity(config, "correlation", imp, fold, dependencies=fits))
        directories[imp, fold] = directory
        dependencies.append(directory / "completed.json")
    destination = Path(config["output_root"]) / "summary"
    expected = identity(config, "plots", dependencies=dependencies)
    if not begin_result(destination, expected, force):
        return
    check_partition(config, directories).to_csv(destination / "cv_validation.csv", index=False)
    metrics = pd.concat([pd.read_csv(path / "metrics.csv.gz") for path in directories.values()], ignore_index=True)
    records = [r for path in directories.values() for r in json.loads((path / "processed_bins.json").read_text())]
    keys = ["imputation", "fold", "seed", "rank", "dataset"]
    comparison_metrics = ["logrank", "HR", "Adj HR", "Adj NoAg HR"]
    baseline = metrics.loc[metrics.variant == "original", keys + comparison_metrics]
    baseline = baseline.rename(columns={col: col + " original" for col in comparison_metrics})
    risk = metrics.merge(baseline, on=keys, validate="many_to_one")
    for col in comparison_metrics:
        risk[col + " delta"] = risk[col] - risk[col + " original"]
    risk.to_csv(destination / "risk_comparison.csv.gz", index=False)
    consistency = consistency_rows(records, config)
    consistency.to_csv(destination / "consistency_pairs.csv.gz", index=False)
    consistency.groupby(["comparison", "scope", "scheme"])[["jaccard", "positions"]].mean().to_csv(destination / "consistency_summary.csv")
    interlocus, sensitivity = aggregate_interlocus(config, directories)
    interlocus.to_csv(destination / "interlocus_summary.csv.gz", index=False)
    sensitivity.to_csv(destination / "interlocus_cutoff_counts.csv", index=False)
    threshold_rows = []
    for (imp, fold), directory in directories.items():
        for scheme, blocks in json.loads((directory / "blocks.json").read_text()).items():
            for locus, threshold in blocks["thresholds"].items():
                threshold_rows.append({"imputation": imp, "fold": fold, "scheme": scheme, "locus": locus,
                                       "threshold": threshold, "multi_position_blocks": sum(b["locus"] == locus for b in blocks["blocks"])})
    pd.DataFrame(threshold_rows).to_csv(destination / "locus_thresholds.csv", index=False)
    captions = []
    plot_config = config["plots"]
    schemes = [scheme_name(r) for r in plot_config["thresholds"]]
    schemes += [name for name in ("locus_specific", "locus_adaptive") if any(r["scheme"] == name for r in records)]
    figures = destination / "figures"
    for name in schemes:
        selection = [r for r in records if r["scheme"] == name]
        population = [r for r in selection if (r["imputation"], r["fold"], r["seed"]) ==
                      (plot_config["population_imputation"], plot_config["population_fold"], plot_config["population_seed"])]
        population.sort(key=lambda r: r["rank"])
        inclusion_plot(population, [str(r["rank"]) for r in population], config, figures,
                       f"figure1_population_{name}", captions)
        for imp, seed in itertools.product(config["imputations"], config["seeds"]):
            selected = sorted([r for r in selection if r["imputation"] == imp and r["seed"] == seed and r["rank"] == 1], key=lambda r: r["fold"])
            inclusion_plot(selected, [f"CV{r['fold']}" for r in selected], config, figures,
                           f"figure2_top_cv_imp{imp}_seed{seed}_{name}", captions)
        if len(config["imputations"]) > 1:
            for seed in config["seeds"]:
                selected = sorted([r for r in selection if r["fold"] == plot_config["population_fold"] and r["seed"] == seed and r["rank"] == 1], key=lambda r: r["imputation"])
                inclusion_plot(selected, [f"Imp{r['imputation']}" for r in selected], config, figures,
                               f"figure3_top_imputations_cv{plot_config['population_fold']}_seed{seed}_{name}", captions)
    threshold_plot(consistency, risk, config, figures, captions)
    interlocus_figures(interlocus, config, figures, captions)
    (destination / "figure_captions.txt").write_text("\n\n".join(captions) + "\n")
    (destination / "interpretation.txt").write_text(
        "Compare held-out HR changes and across-fold feature consistency. A larger Jaccard value alone is not evidence of better risk prediction.\n"
        "Pearson mismatch correlations are not haplotype LD estimates. Adaptive thresholds are an exploratory structural rule, not a validated optimum.\n"
        "Processed scores sum one maximum per touched block; for counts 0/1/2 this maximum can be 2. Threshold-retuned results are separate from fixed-threshold results.\n"
        "Adj HR includes clinical and antigen covariates. Adj NoAg HR excludes all antigen covariates including Agmm0. Residuals are used for FIBERS training fitness, not inserted as an outcome-derived test-set Cox predictor.\n"
        "Hazard ratios compare above-threshold versus at/below-threshold; above-threshold is not assumed to mean higher risk.\n"
        "Across-imputation figures require multiple complete imputations and compare the same held-out fold and seed. No external validation is inferred.\n"
        "Review Cox status columns: warnings/failures are retained; unrequested or failed estimates are missing.\n")
    names = [str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file() and path.name != "completed.json"]
    finish_result(destination, expected, names, figure_count=len(captions))
    print(f"Completed {len(captions)} figures and summary tables: {destination}", flush=True)


if __name__ == "__main__":
    args, config = worker_args("plots")
    run(config, args.force)
