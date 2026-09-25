"""Compute one fold's correlations once and process all its fitted populations."""
from common import configure_worker
configure_worker()

import hashlib
import json

import numpy as np
import pandas as pd

from common import (begin_result, finish_result, fit_dir, fold_dir, identity,
                    require_result, save_json, worker_args)
from data import active_covariates, load_fold, retained_features
from methods import (build_schemes, correlation_table, evaluate, expand_bin,
                     optimize_threshold, pearson_matrix, scheme_name, score_groups)


def fit_results(config, imputation, fold):
    results, dependencies = {}, []
    for seed in config["seeds"]:
        directory = fit_dir(config, imputation, fold, seed)
        results[seed] = require_result(directory, identity(config, "fibers", imputation, fold, seed))
        dependencies += [directory / "completed.json", directory / "population.csv"]
    return results, dependencies


def run(config, imputation, fold, force=False):
    fits, dependencies = fit_results(config, imputation, fold)
    destination = fold_dir(config, imputation, fold) / "correlation"
    expected = identity(config, "correlation", imputation, fold, dependencies=dependencies)
    if not begin_result(destination, expected, force):
        return
    train, test, candidates, audit = load_fold(config, imputation, fold)
    features, stats = retained_features(train, candidates, config)
    for fit in fits.values():
        if fit["split_audit"] != audit or fit["features"] != features:
            raise ValueError("FIBERS and correlation data or feature filtering do not match")
    clinical, antigen = active_covariates(train, config)
    matrix = pearson_matrix(train, features, config["input"]["chunksize"])
    correlation_table(matrix, features, len(train)).to_csv(destination / "correlations.csv.gz", index=False)
    schemes = build_schemes(matrix, features, config)
    save_json(destination / "blocks.json", schemes)
    stats.to_csv(destination / "feature_filter.csv", index=False)
    identifier = config["input"]["id_column"]
    pd.concat([train[[identifier]].assign(split="train"), test[[identifier]].assign(split="test")]).to_csv(
        destination / "split_ids.csv.gz", index=False)
    primary = scheme_name(config["correlation"]["primary_threshold"])
    settings = config["evaluation"]
    rows, membership, processed = [], [], []
    score_caches = {"train": {}, "test": {}}
    metric_cache = {}

    def metrics(split, frame, score, threshold, unadjusted, adjusted):
        # Identical group assignments have identical survival results, across bins/cutoffs.
        group_hash = hashlib.sha256(np.packbits(score > threshold).tobytes()).hexdigest()
        key = (split, group_hash, unadjusted, adjusted)
        if key not in metric_cache:
            metric_cache[key] = evaluate(frame, score, threshold, config, clinical, antigen, unadjusted, adjusted)
        return {**metric_cache[key], "threshold": float(threshold)}

    for seed in config["seeds"]:
        population = pd.read_csv(fit_dir(config, imputation, fold, seed) / "population.csv")
        population = population.head(settings["bins"])
        for bin_row in population.itertuples(index=False):
            rank, original = int(bin_row.rank), json.loads(bin_row.features)
            base = {"imputation": imputation, "fold": fold, "seed": seed, "rank": rank,
                    "original_count": len(original)}
            original_scores = {split: frame[original].sum(axis=1).to_numpy()
                               for split, frame in (("train", train), ("test", test))}
            for split, frame in (("train", train), ("test", test)):
                rows.append({**base, "dataset": split, "scheme": "original", "variant": "original",
                             "positions": len(original), "blocks": len(original), "changed_group_fraction": 0.0,
                             **metrics(split, frame, original_scores[split], bin_row.threshold,
                                       split == "test" and (settings["primary_hr_all_bins"] or rank <= settings["cox_top_bins"]),
                                       split == "test" and rank <= settings["adjusted_top_bins"])})
            for name, scheme in schemes.items():
                groups = expand_bin(original, scheme)
                expanded = {f for group in groups for f in group["features"]}
                scores = {split: score_groups(frame, groups, score_caches[split])
                          for split, frame in (("train", train), ("test", test))}
                variants = [("processed_fixed", float(bin_row.threshold))]
                optimized = None
                if rank <= settings["reoptimize_top_bins"]:
                    optimized = optimize_threshold(train, scores["train"], bin_row.threshold, config)
                    if optimized is not None:
                        variants.append(("processed_reoptimized", optimized))
                processed.append({**base, "scheme": name, "original": original, "groups": groups,
                                  "original_threshold": float(bin_row.threshold), "reoptimized_threshold": optimized})
                for group in groups:
                    for feature in group["features"]:
                        membership.append({**base, "scheme": name, "feature": feature,
                                           "block": group["block"], "status": "original" if feature in original else "added"})
                for variant, threshold in variants:
                    for split, frame in (("train", train), ("test", test)):
                        changed = np.mean((scores[split] > threshold) != (original_scores[split] > bin_row.threshold))
                        rows.append({**base, "dataset": split, "scheme": name, "variant": variant,
                                     "positions": len(expanded), "blocks": len(groups),
                                     "changed_group_fraction": float(changed),
                                     **metrics(split, frame, scores[split], threshold,
                                               split == "test" and (rank <= settings["cox_top_bins"] or
                                                   (name == primary and settings["primary_hr_all_bins"])),
                                               split == "test" and rank <= settings["adjusted_top_bins"])})
            print(f"Processed imp={imputation}, CV={fold}, seed={seed}, bin={rank}", flush=True)
    pd.DataFrame(rows).to_csv(destination / "metrics.csv.gz", index=False)
    pd.DataFrame(membership).to_csv(destination / "membership.csv.gz", index=False)
    save_json(destination / "processed_bins.json", processed)
    finish_result(destination, expected,
                  ["correlations.csv.gz", "blocks.json", "feature_filter.csv", "split_ids.csv.gz",
                   "metrics.csv.gz", "membership.csv.gz", "processed_bins.json"],
                  split_audit=audit, clinical_covariates=clinical, antigen_covariates=antigen,
                  unique_survival_evaluations=len(metric_cache))
    print(f"Completed correlation analysis: {destination}", flush=True)


if __name__ == "__main__":
    args, config = worker_args("correlation")
    run(config, args.imputation, args.fold, args.force)
