"""Fit and save one FIBERS population; invoked directly by bsub."""
from common import configure_worker
configure_worker()

import json
import pickle
import time

import pandas as pd
from skfibers.fibers import FIBERS

from common import begin_result, finish_result, fit_dir, identity, worker_args
from data import active_covariates, load_fold, retained_features
from methods import evaluate


def run(config, imputation, fold, seed, force=False):
    destination = fit_dir(config, imputation, fold, seed)
    expected = identity(config, "fibers", imputation, fold, seed)
    if not begin_result(destination, expected, force):
        return
    train, test, candidates, audit = load_fold(config, imputation, fold)
    features, filter_table = retained_features(train, candidates, config)
    clinical, antigen = active_covariates(train, config)
    parameters = dict(config["fibers"])
    training_covariates = clinical + antigen if parameters["fitness_metric"] in ("residuals", "log_rank_residuals") else []
    if parameters["fitness_metric"] in ("residuals", "log_rank_residuals") and not training_covariates:
        raise ValueError("Product/residual fitness requires nonconstant training covariates")
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    parameters.update(outcome_label=outcome, censor_label=event, outcome_type="survival",
                      random_seed=seed, covariates=training_covariates or None, verbose=False)
    started = time.monotonic()
    print(f"Fitting imp={imputation}, CV={fold}, seed={seed}, features={len(features)}", flush=True)
    model = FIBERS(**parameters).fit(train[features + training_covariates + [outcome, event]])
    rows = []
    for rank, bin_object in enumerate(model.set.bin_pop, 1):
        rows.append({"rank": rank, "features": json.dumps(list(bin_object.feature_list)),
                     "threshold": float(bin_object.group_threshold), "fitness": float(bin_object.fitness),
                     "logrank_training": float(bin_object.log_rank_score)})
    if not rows:
        raise RuntimeError("FIBERS returned no bins")
    population = pd.DataFrame(rows)
    with (destination / "fibers.pkl").open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    population.to_csv(destination / "population.csv", index=False)
    filter_table.to_csv(destination / "feature_filter.csv", index=False)
    original = json.loads(population.iloc[0]["features"])
    threshold = population.iloc[0]["threshold"]
    metrics = []
    for split, frame in (("train", train), ("test", test)):
        metrics.append({"dataset": split, **evaluate(frame, frame[original].sum(axis=1).to_numpy(),
                       threshold, config, clinical, antigen, split == "test", split == "test")})
    pd.DataFrame(metrics).to_csv(destination / "top_bin_metrics.csv", index=False)
    finish_result(destination, expected, ["fibers.pkl", "population.csv", "feature_filter.csv", "top_bin_metrics.csv"],
                  split_audit=audit, features=features, clinical_covariates=clinical,
                  antigen_covariates=antigen, elapsed_seconds=time.monotonic() - started,
                  parameters=parameters)
    print(f"Completed FIBERS: {destination}", flush=True)


if __name__ == "__main__":
    args, config = worker_args("fibers")
    run(config, args.imputation, args.fold, args.seed, args.force)
