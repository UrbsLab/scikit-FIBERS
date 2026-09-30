"""Fit complete imputations and compare their bins, without constructing CV folds."""
from common import configure_worker
configure_worker()

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from skfibers.fibers import FIBERS

from common import (HERE, REPO, begin_result, digest, file_state, finish_result, input_paths,
                    load_config, require_result, save_json)
from data import active_covariates, id_digest, read_data, retained_features
from methods import (build_schemes, correlation_table, cox_designs, evaluate, expand_bin,
                     jaccard, optimize_threshold, pearson_matrix, scheme_name, score_groups)
from plotting import inclusion_plot, interlocus_figures, threshold_plot


def root_dir(config):
    return Path(config["output_root"]) / "imp_summary"


def full_path(config, imputation):
    # Only resolve full_template; never read or create a CV training/test split.
    whole = {**config, "input": {**config["input"], "mode": "full_dataset"}}
    return input_paths(whole, imputation, None)[0]


def result_identity(config, stage, imputation=None, seed=None, dependencies=()):
    keys = ["columns", "rare_filter", "fibers"]
    if stage != "fit":
        keys += ["correlation", "evaluation", "seeds"]
    if stage == "summary":
        keys += ["plots", "imputations"]
    payload = {key: config[key] for key in keys}
    payload["input"] = {key: config["input"][key] for key in ("full_template", "id_column", "chunksize")}
    payload["analysis"] = "whole_imputation_apparent"
    payload["task"] = [stage, imputation, seed]
    code = ["common.py", "data.py", "methods.py", "run_imp_summary.py"]
    if stage == "summary":
        code.append("plotting.py")
    payload["code"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in code}
    payload["fibers_source"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((REPO / "src/skfibers").rglob("*.py"))}
    payload["inputs"] = [file_state(full_path(config, imputation))] if imputation is not None else []
    payload["dependencies"] = [file_state(path) for path in dependencies]
    return {"fingerprint": digest(payload), "settings": payload}


def fit_population(frame, features, clinical, antigen, config, imputation, seed, force):
    destination = root_dir(config) / f"imp_{imputation:02d}" / f"seed_{seed:03d}"
    expected = result_identity(config, "fit", imputation, seed)
    if begin_result(destination, expected, force):
        parameters = dict(config["fibers"])
        if len(features) < parameters["min_bin_size"]:
            raise ValueError("Filtering left fewer features than min_bin_size")
        parameters["max_bin_init_size"] = min(parameters["max_bin_init_size"], len(features))
        if parameters["max_bin_size"] is not None:
            parameters["max_bin_size"] = min(parameters["max_bin_size"], len(features))
        covs = clinical + antigen if parameters["fitness_metric"] in ("residuals", "log_rank_residuals") else []
        if parameters["fitness_metric"] in ("residuals", "log_rank_residuals") and not covs:
            raise ValueError("Product/residual fitness requires nonconstant training covariates")
        outcome, event = config["columns"]["outcome"], config["columns"]["event"]
        parameters.update(outcome_label=outcome, censor_label=event, outcome_type="survival",
                          random_seed=seed, covariates=covs or None, verbose=False)
        print(f"Fitting WHOLE imp={imputation}, seed={seed}, n={len(frame)}, features={len(features)}", flush=True)
        model = FIBERS(**parameters).fit(frame[features + covs + [outcome, event]])
        population = pd.DataFrame([{"rank": rank, "features": json.dumps(list(bin_object.feature_list)),
                                    "threshold": float(bin_object.group_threshold), "fitness": float(bin_object.fitness),
                                    "logrank_training": float(bin_object.log_rank_score)}
                                   for rank, bin_object in enumerate(model.set.bin_pop, 1)])
        if population.empty:
            raise RuntimeError("FIBERS returned no bins")
        with (destination / "fibers.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        population.to_csv(destination / "population.csv", index=False)
        finish_result(destination, expected, ["fibers.pkl", "population.csv"], parameters=parameters,
                      features=features, n_training=len(frame), evaluation="full_cohort_apparent")
    return pd.read_csv(destination / "population.csv")


def run(config, imputation, force=False):
    destination = root_dir(config) / f"imp_{imputation:02d}"
    expected = result_identity(config, "analysis", imputation)
    if not begin_result(destination, expected, force):
        return
    started = time.monotonic()
    frame, candidates = read_data(full_path(config, imputation), config)
    features, stats = retained_features(frame, candidates, config)
    clinical, antigen = active_covariates(frame, config)
    stats.to_csv(destination / "feature_filter.csv", index=False)
    identifier = config["input"]["id_column"]
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    # Retain only a hash of IDs/outcomes, sufficient to audit the same cohort across imputations.
    cohort = frame[[identifier, outcome, event]].sort_values(identifier)
    outcome_hash = hashlib.sha256(cohort.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    audit = {"n": len(frame), "cohort_ids_sha256": id_digest(set(frame[identifier])),
             "outcomes_sha256": outcome_hash, "evaluation": "full_cohort_apparent"}
    matrix = pearson_matrix(frame, features, config["input"]["chunksize"])
    correlation_table(matrix, features, len(frame)).to_csv(destination / "correlations.csv.gz", index=False)
    schemes = build_schemes(matrix, features, config)
    save_json(destination / "blocks.json", schemes)
    del matrix
    designs = cox_designs(frame, config, clinical, antigen)
    settings = config["evaluation"]
    primary = scheme_name(config["correlation"]["primary_threshold"])
    scores_cache, metric_cache = {}, {}
    rows, membership, processed = [], [], []

    def metrics(score, threshold, rank, name):
        unadjusted = rank <= settings["cox_top_bins"] or (settings["primary_hr_all_bins"] and name in ("original", primary))
        adjusted = rank <= settings["adjusted_top_bins"]
        key = (hashlib.sha256(np.packbits(score > threshold).tobytes()).hexdigest(), unadjusted, adjusted)
        if key not in metric_cache:
            metric_cache[key] = evaluate(frame, score, threshold, config, clinical, antigen,
                                         unadjusted, adjusted, designs)
        return {**metric_cache[key], "threshold": float(threshold)}

    for seed in config["seeds"]:
        population = fit_population(frame, features, clinical, antigen, config, imputation, seed, force)
        for bin_row in population.head(settings["bins"]).itertuples(index=False):
            rank, original = int(bin_row.rank), json.loads(bin_row.features)
            base = {"imputation": imputation, "seed": seed, "rank": rank, "original_count": len(original)}
            original_score = frame[original].sum(axis=1).to_numpy()
            rows.append({**base, "dataset": "full_cohort", "scheme": "original", "variant": "original",
                         "positions": len(original), "blocks": len(original), "changed_group_fraction": 0.0,
                         **metrics(original_score, bin_row.threshold, rank, "original")})
            for name, scheme in schemes.items():
                groups = expand_bin(original, scheme)
                score = score_groups(frame, groups, scores_cache)
                expanded = {f for group in groups for f in group["features"]}
                variants = [("processed_fixed", float(bin_row.threshold))]
                optimized = None
                if rank <= settings["reoptimize_top_bins"]:
                    optimized = optimize_threshold(frame, score, bin_row.threshold, config)
                    if optimized is not None:
                        variants.append(("processed_reoptimized", optimized))
                processed.append({**base, "scheme": name, "original": original, "groups": groups,
                                  "original_threshold": float(bin_row.threshold), "reoptimized_threshold": optimized})
                for group in groups:
                    for feature in group["features"]:
                        membership.append({**base, "scheme": name, "feature": feature, "block": group["block"],
                                           "status": "original" if feature in original else "added"})
                for variant, threshold in variants:
                    rows.append({**base, "dataset": "full_cohort", "scheme": name, "variant": variant,
                                 "positions": len(expanded), "blocks": len(groups),
                                 "changed_group_fraction": float(np.mean((score > threshold) != (original_score > bin_row.threshold))),
                                 **metrics(score, threshold, rank, name)})
            print(f"Processed whole imp={imputation}, seed={seed}, bin={rank}", flush=True)
    pd.DataFrame(rows).to_csv(destination / "metrics.csv.gz", index=False)
    pd.DataFrame(membership).to_csv(destination / "membership.csv.gz", index=False)
    save_json(destination / "processed_bins.json", processed)
    names = ["feature_filter.csv", "correlations.csv.gz", "blocks.json", "metrics.csv.gz",
             "membership.csv.gz", "processed_bins.json"]
    names += [f"seed_{seed:03d}/{name}" for seed in config["seeds"]
              for name in ("fibers.pkl", "population.csv", "completed.json")]
    finish_result(destination, expected, names, cohort_audit=audit,
                  clinical_covariates=clinical, antigen_covariates=antigen,
                  cox_adjustment={label: info for label, (_, info) in designs.items()},
                  cox_excluded_covariates=settings.get("exclude_covariates", []),
                  unique_survival_evaluations=len(metric_cache), elapsed_seconds=time.monotonic() - started)
    print(f"Completed whole-imputation analysis: {destination}", flush=True)


def consistency_rows(records, config):
    rows = []
    schemes = list(dict.fromkeys(r["scheme"] for r in records))
    for seed in config["seeds"]:
        scopes = [("across_imputations", f"seed{seed}", [r for r in records if r["seed"] == seed and r["rank"] == 1])]
        scopes += [("within_population", f"imp{imp}_seed{seed}", [r for r in records if r["seed"] == seed and r["imputation"] == imp])
                   for imp in config["imputations"]]
        for comparison, scope, items in scopes:
            for name in ["original"] + schemes:
                selected = [r for r in items if r["scheme"] == (schemes[0] if name == "original" else name)]
                sets = [set(r["original"]) if name == "original" else {f for g in r["groups"] for f in g["features"]} for r in selected]
                for i, j in itertools.combinations(range(len(selected)), 2):
                    task = lambda r: f"imp{r['imputation']}_seed{r['seed']}_bin{r['rank']}"
                    rows.append({"comparison": comparison, "scope": scope, "seed": seed, "scheme": name,
                                 "left": task(selected[i]), "right": task(selected[j]), "jaccard": jaccard(sets[i], sets[j]),
                                 "positions": (len(sets[i]) + len(sets[j])) / 2})
    return pd.DataFrame(rows)


def aggregate_correlations(config, directories):
    cutoffs = sorted(set(config["plots"]["interlocus_reporting_thresholds"] + [config["plots"]["interlocus_threshold"]]))
    tables = []
    for imp, directory in directories.items():
        table = pd.read_csv(directory / "correlations.csv.gz")
        table["imputation"] = imp
        for i, cutoff in enumerate(cutoffs):
            table[f"above_{i}"] = (table.pearson_r > cutoff).astype(int)
        tables.append(table)
    aggregation = {"mean_r": ("pearson_r", "mean"), "min_r": ("pearson_r", "min"), "max_r": ("pearson_r", "max"),
                   "imputations_available": ("imputation", "nunique")}
    aggregation.update({f"imputations_r_gt_{cutoff:g}": (f"above_{i}", "sum") for i, cutoff in enumerate(cutoffs)})
    pairs = pd.concat(tables, ignore_index=True).groupby(["locus1", "locus2", "feature1", "feature2"], sort=False).agg(**aggregation).reset_index()
    pairs["imputations_above_plot_cutoff"] = pairs[f"imputations_r_gt_{config['plots']['interlocus_threshold']:g}"]
    return pairs


def summarize(config, force=False):
    if len(config["imputations"]) < 2:
        raise ValueError("Across-imputation consistency requires at least two configured imputations")
    directories, dependencies, audits = {}, [], []
    for imp in config["imputations"]:
        directory = root_dir(config) / f"imp_{imp:02d}"
        result = require_result(directory, result_identity(config, "analysis", imp))
        audits.append({"imputation": imp, **result["cohort_audit"]})
        directories[imp] = directory
        dependencies.append(directory / "completed.json")
    for key in ("n", "cohort_ids_sha256", "outcomes_sha256"):
        if len({a[key] for a in audits}) != 1:
            raise ValueError(f"Whole-imputation cohorts or survival outcomes differ: {key}")
    destination = root_dir(config)
    expected = result_identity(config, "summary", dependencies=dependencies)
    if not begin_result(destination, expected, force):
        return
    pd.DataFrame(audits).to_csv(destination / "cohort_validation.csv", index=False)
    metrics = pd.concat([pd.read_csv(d / "metrics.csv.gz") for d in directories.values()], ignore_index=True)
    records = [r for d in directories.values() for r in json.loads((d / "processed_bins.json").read_text())]
    measures = ["logrank", "HR", "Adj NoAg HR", "Adj HR"]
    keys = ["imputation", "seed", "rank", "dataset"]
    baseline = metrics.loc[metrics.variant == "original", keys + measures + [m + " status" for m in measures[1:]]]
    baseline = baseline.rename(columns={c: c + " original" for c in baseline if c not in keys})
    risk = metrics.merge(baseline, on=keys, validate="many_to_one")
    for measure in measures:
        risk[measure + " delta"] = risk[measure] - risk[measure + " original"]
    risk.to_csv(destination / "risk_comparison.csv.gz", index=False)
    consistency = consistency_rows(records, config)
    consistency.to_csv(destination / "consistency_pairs.csv.gz", index=False)
    consistency.groupby(["comparison", "scope", "scheme"])[["jaccard", "positions"]].mean().to_csv(destination / "consistency_summary.csv")
    top = risk.loc[(risk["rank"] == 1) & risk.variant.isin(["original", "processed_fixed"])]
    top.to_csv(destination / "top_bin_metrics.csv", index=False)
    summary = []
    for (seed, scheme), group in top.groupby(["seed", "scheme"], sort=False):
        agreement = consistency.loc[(consistency.comparison == "across_imputations") & (consistency.seed == seed) & (consistency.scheme == scheme)]
        original_agreement = consistency.loc[(consistency.comparison == "across_imputations") & (consistency.seed == seed) & (consistency.scheme == "original")]
        row = {"seed": seed, "scheme": scheme, "evaluation": "full_cohort_apparent", "n_imputations": len(group),
               "n_pairs": len(agreement), "mean_jaccard": agreement.jaccard.mean(),
               "jaccard_delta": agreement.jaccard.mean() - original_agreement.jaccard.mean(),
               "mean_positions": group.positions.mean(), "mean_logrank": group.logrank.mean(),
               "mean_logrank_delta": group["logrank delta"].mean(),
               "mean_changed_group_fraction": group.changed_group_fraction.mean()}
        for measure in measures[1:]:
            good = group[measure + " status"].eq("ok") & np.isfinite(group[measure])
            paired = good & group[measure + " status original"].eq("ok") & np.isfinite(group[measure + " original"])
            row["n_" + measure], row["n_paired_" + measure] = int(good.sum()), int(paired.sum())
            row["mean_" + measure] = group[measure].mean() if good.all() else float("nan")
            row["mean_" + measure + "_delta"] = group[measure + " delta"].mean() if paired.all() else float("nan")
        summary.append(row)
    pd.DataFrame(summary).to_csv(destination / "top_bin_summary.csv", index=False)
    correlations = aggregate_correlations(config, directories)
    correlations.to_csv(destination / "correlation_summary.csv.gz", index=False)
    figures, captions = destination / "figures", []
    schemes = [scheme_name(r, scope) for scope in config["correlation"].get("scopes", ["within"])
               for r in config["plots"]["thresholds"]]
    schemes += [name for name in ("locus_specific", "locus_adaptive") if any(r["scheme"] == name for r in records)]
    for name, seed in itertools.product(schemes, config["seeds"]):
        selection = [r for r in records if r["scheme"] == name and r["seed"] == seed]
        for imp in config["imputations"]:
            population = sorted([r for r in selection if r["imputation"] == imp], key=lambda r: r["rank"])
            inclusion_plot(population, [str(r["rank"]) for r in population], config, figures,
                           f"figure1_population_imp{imp}_seed{seed}_{name}", captions, column_axis="Bin rank")
        selected = sorted([r for r in selection if r["rank"] == 1], key=lambda r: r["imputation"])
        inclusion_plot(selected, [f"Imp{r['imputation']}" for r in selected], config, figures,
                       f"figure2_top_imputations_seed{seed}_{name}", captions, column_axis="Imputation")
    for seed in config["seeds"]:
        threshold_plot(consistency.loc[consistency.seed == seed], risk.loc[risk.seed == seed], config, figures, captions,
                       comparison="across_imputations", dataset="full_cohort", name=f"figure4_imputation_comparison_seed{seed}")
    interlocus_figures(correlations.loc[correlations.locus1 != correlations.locus2], config, figures, captions,
                      across_imputations=True)
    warning = ("Each model was newly fitted to its entire imputed cohort. There are no CV folds or held-out patients in this analysis. "
               "All HRs, confidence intervals and log-rank statistics here are apparent, training-cohort estimates and do not account for feature selection. "
               "Use the separate CV analysis to assess held-out risk separation. Imputations represent the same transplants, not independent cohorts. "
               "Mean HRs are descriptive arithmetic averages, not pooled multiple-imputation estimates. Jaccard measures feature-set overlap, not predictive accuracy.\n")
    (destination / "figure_captions.txt").write_text(warning + "\n" + "\n\n".join(captions) + "\n")
    (destination / "interpretation.txt").write_text(warning +
        "Top bins retain their original rank and threshold; no reranking by whole-cohort HR. Seeds are summarized separately. "
        "Within-locus and any-locus blocks use the configured strict positive Pearson cutoff. Each block contributes its maximum mismatch count once (0, 1 or 2). "
        "Adj NoAg HR uses clinical covariates; Adj HR also includes antigen mismatch covariates. Product-fitness residuals are not inserted into the evaluation Cox model. "
        "Failed or unrequested estimates remain missing; full-summary means are suppressed if any required fit failed. "
        "Correlations unavailable after feature filtering are missing, not zero; correlation_summary.csv.gz reports the denominator.\n")
    names = [p.name for p in destination.iterdir() if p.is_file() and p.name != "completed.json"]
    names += [str(p.relative_to(destination)) for p in figures.rglob("*") if p.is_file()]
    finish_result(destination, expected, names, figure_count=len(captions), evaluation="full_cohort_apparent")
    print(f"Completed whole-imputation tables and {len(captions)} figures: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--imputation", type=int)
    choice.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.summary_only:
        summarize(config, args.force)
    else:
        if args.imputation not in config["imputations"]:
            parser.error("--imputation must occur in config.imputations")
        run(config, args.imputation, args.force)
