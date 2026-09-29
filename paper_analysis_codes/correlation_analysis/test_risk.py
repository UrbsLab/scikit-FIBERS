"""Synthetic regression tests for the held-out risk-only repair."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from common import HERE, file_state, load_config, save_json
from data import id_digest
from run_risk import covariate_basis, fit_noag, run, source_inputs


def sample(n=800):
    rng = np.random.default_rng(729)
    age = rng.normal(0, 1, n)
    high = rng.binomial(1, .45, n)
    death = rng.exponential(4, n) * np.exp(-.3 * age - .4 * high)
    censor = rng.exponential(8, n)
    return pd.DataFrame({"age": age, "age_copy": age * 3 + 10, "constant": np.zeros(n),
                         "graftyrs": np.minimum(death, censor), "grf_fail": (death <= censor).astype(int),
                         "high": high})


def test_constant_and_alias_repair_preserves_bin_estimate():
    frame = sample()
    basis, design = covariate_basis(frame, ["age", "age_copy", "constant"])
    assert design["constants"] == ["constant"]
    assert len(design["aliases"]) == 1
    result, attempts, _ = fit_noag(frame, frame.high, "graftyrs", "grf_fail", basis)
    direct = CoxPHFitter().fit(frame[["age", "high", "graftyrs", "grf_fail"]], "graftyrs", "grf_fail")
    assert result["status"] == "ok"
    assert result["Adj NoAg HR"] == pytest.approx(direct.summary.loc["high", "exp(coef)"], rel=1e-5)
    assert attempts[-1]["status"] == "ok"


def test_near_collinearity_is_not_silently_discarded():
    frame = sample()
    frame["near_age"] = frame.age + np.random.default_rng(10).normal(0, 1e-5, len(frame))
    basis, design = covariate_basis(frame, ["age", "near_age"])
    assert len(basis.columns) == 2
    assert design["aliases"] == []


def test_bin_alias_and_zero_events_are_not_forced_to_succeed():
    frame = sample()
    basis, _ = covariate_basis(frame, ["high", "age"])
    result, _, _ = fit_noag(frame, frame.high, "graftyrs", "grf_fail", basis, ridge=.01)
    assert result["status"] == "bin_not_identifiable_given_covariates"
    basis, _ = covariate_basis(frame, ["age"])
    frame.loc[frame.high == 0, "grf_fail"] = 0
    result, _, _ = fit_noag(frame, frame.high, "graftyrs", "grf_fail", basis)
    assert result["status"] == "no_events_in_one_bin_group"


def test_fixed_ridge_is_explicit_and_bin_remains_unpenalized():
    frame = sample()
    basis, _ = covariate_basis(frame, ["age"])
    result, _, _ = fit_noag(frame, frame.high, "graftyrs", "grf_fail", basis, ridge=.01)
    assert result["ridge"] == .01
    direct = pd.concat([frame[["graftyrs", "grf_fail", "high"]], basis], axis=1)
    model = CoxPHFitter(penalizer=np.array([0, .01])).fit(direct, "graftyrs", "grf_fail")
    assert result["Adj NoAg HR"] == pytest.approx(model.summary.loc["high", "exp(coef)"], rel=1e-5)


def test_bsub_submitter_is_standard_library_only():
    completed = subprocess.run([sys.executable, "-S", str(HERE / "main_correlation.py"), "--dry-run"],
                               text=True, capture_output=True, check=True)
    commands = [line for line in completed.stdout.splitlines() if line.startswith("bsub ")]
    assert len(commands) == 10
    assert all("run_correlation.py" in line for line in commands)
    assert not any("main_fibers" in line or "-w " in line for line in commands)


def test_saved_bin_reevaluation_and_complete_summary(tmp_path):
    config = json.loads((HERE / "config.json").read_text())
    config.update(imputations=[1], folds=[1, 2], seeds=[1], output_root=str(tmp_path / "old"))
    config["columns"].update(clinical_covariates=["age", "age_copy", "constant"], antigen_covariates=[])
    config["input"].update(train_template=str(tmp_path / "train{fold}.csv"),
                           test_template=str(tmp_path / "test{fold}.csv"))
    frame = sample(1200)
    frame["TX_ID"] = [f"T{i:05d}" for i in range(len(frame))]
    frame["MM_A_10"] = frame.high
    frame["MM_A_20"] = frame.high
    frame["MM_B_30"] = 1 - frame.high
    audits = []
    for fold, indexes in enumerate(np.array_split(np.arange(len(frame)), 2), 1):
        test, train = frame.iloc[indexes], frame.drop(indexes)
        train_path, test_path = tmp_path / f"train{fold}.csv", tmp_path / f"test{fold}.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        train_ids, test_ids = set(train.TX_ID), set(test.TX_ID)
        audit = {"train_n": len(train), "test_n": len(test),
                 "train_ids_sha256": id_digest(train_ids), "test_ids_sha256": id_digest(test_ids),
                 "cohort_ids_sha256": id_digest(train_ids | test_ids)}
        audits.append(audit)
        directory = tmp_path / "old" / "imp_01" / f"cv_{fold:02d}" / "correlation"
        save_json(directory / "completed.json", {"clinical_covariates": config["columns"]["clinical_covariates"],
                  "settings": {"columns": config["columns"], "inputs": [file_state(train_path), file_state(test_path)]},
                  "split_audit": audit})
        save_json(directory / "processed_bins.json", [{"rank": 1, "seed": 1, "scheme": "r0p95",
                  "original": ["MM_A_10"], "original_threshold": 0,
                  "groups": [{"features": ["MM_A_10", "MM_A_20"]}]}])
    config_path = tmp_path / "config.json"
    save_json(config_path, config)
    config = load_config(config_path)
    loaded, _, _, _, _ = source_inputs(config, 1, 1, extra_features=["MM_B_30"])
    assert "MM_B_30" in loaded
    assert loaded["MM_B_30"].equals(1 - loaded["MM_A_10"])
    output = tmp_path / "risk"
    run(config, 1, 1, output)
    assert not (output / "summary.csv").exists()
    run(config, 1, 2, output)
    summary = pd.read_csv(output / "summary.csv")
    assert summary.n_folds.eq(2).all()
    assert summary.mean_paired_delta.eq(0).all()
    assert set(summary.model) == {"HR", "Adj NoAg HR", "Adj HR"}
    metrics = pd.read_csv(output / "all_fold_metrics.csv")
    assert len(metrics) == 12 and metrics.status.eq("ok").all()
    with pytest.raises(FileExistsError):
        run(config, 1, 2, output)
    assert json.loads((output / "imp_01/cv_02/completed.json").read_text())["split_audit"] == audits[1]
    excluded_output = tmp_path / "excluded_risk"
    run(config, 1, 1, excluded_output, exclude_covariates=["age_copy", "constant"])
    diagnostics = json.loads((excluded_output / "imp_01/cv_01/diagnostics.json").read_text())
    assert diagnostics["design"]["Adj NoAg HR"]["retained"] == ["age"]
    assert diagnostics["design"]["Adj HR"]["explicit_exclusions"] == ["age_copy", "constant"]
