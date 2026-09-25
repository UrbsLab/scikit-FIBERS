"""Run with python -m pytest test_analysis.py; creates only temporary synthetic data."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

from common import configure_worker
configure_worker()

import numpy as np
import pandas as pd
import pytest

from common import HERE, identity, load_config, require_result
from data import load_fold
from methods import build_schemes, expand_bin, pearson_matrix, score_groups


@pytest.fixture
def example(tmp_path):
    config = json.loads((HERE / "config.json").read_text())
    config.update(imputations=[1, 2], folds=[1, 2], seeds=[1], output_root=str(tmp_path / "results"))
    config["input"].update(train_template=str(tmp_path / "imp{imputation}_cv{fold}_Train.csv"),
                           test_template=str(tmp_path / "imp{imputation}_cv{fold}_Test.csv"),
                           full_template=str(tmp_path / "imp{imputation}.csv"), chunksize=31)
    config["columns"].update(ranges={"A": [1, 182], "DQA1": [6, 94], "DQB1": [6, 95]},
                             clinical_covariates=["age"], antigen_covariates=["Agmm0"])
    config["fibers"].update(iterations=1, pop_size=10, max_bin_init_size=3)
    config["correlation"].update(thresholds=[0.1, 0.4, 0.95])
    config["evaluation"].update(bins=10, primary_hr_all_bins=False)
    config["plots"].update(thresholds=[0.1, 0.95], rows_per_page=28)
    rng = np.random.default_rng(704)
    n = 180
    base = pd.DataFrame({"TX_ID": [f"T{i:03d}" for i in range(n)], "age": rng.normal(50, 9, n),
                         "Agmm0": rng.integers(0, 2, n)})
    for locus, positions in {"A": [10, 20, 30], "DQA1": [11, 18, 45], "DQB1": [84, 85]}.items():
        for pos in positions:
            base[f"MM_{locus}_{pos}"] = rng.binomial(2, .22, n)
    base["MM_DQA1_18"] = base.MM_DQA1_11
    base["MM_DQB1_84"] = base.MM_DQA1_11
    base["MM_DQB1_85"] = base.MM_DQB1_84
    base["graftyrs"] = rng.exponential(4, n) * np.exp(-.5 * base.MM_DQA1_11)
    base["grf_fail"] = rng.binomial(1, .8, n)
    for imp in config["imputations"]:
        frame = base.copy()
        if imp == 2:
            frame["MM_A_20"] = rng.binomial(2, .22, n)
        frame.to_csv(tmp_path / f"imp{imp}.csv", index=False)
        for fold, indexes in enumerate(np.array_split(np.arange(n), 2), 1):
            frame.iloc[indexes].to_csv(tmp_path / f"imp{imp}_cv{fold}_Test.csv", index=False)
            frame.drop(index=indexes).to_csv(tmp_path / f"imp{imp}_cv{fold}_Train.csv", index=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return load_config(path)


def test_matrix_and_block_math(example):
    frame, _, features, _ = load_fold(example, 1, 1)
    matrix = pearson_matrix(frame, features, 13)
    np.testing.assert_allclose(matrix, frame[features].corr(), atol=1e-12)
    schemes = build_schemes(matrix, features, example)
    for name, scheme in schemes.items():
        for block in scheme["blocks"]:
            indexes = [features.index(f) for f in block["features"]]
            values = matrix[np.ix_(indexes, indexes)][np.triu_indices(len(indexes), 1)]
            assert np.all(values > scheme["thresholds"][block["locus"]])
    score_data = pd.DataFrame({"MM_DQA1_11": [1, 0, 2], "MM_DQA1_18": [1, 1, 2]})
    groups = expand_bin(list(score_data), schemes["r0p95"])
    assert score_groups(score_data, groups).tolist() == [1, 1, 2]


def test_complete_linkage_does_not_chain(example):
    names = ["MM_A_10", "MM_A_20", "MM_A_30"]
    matrix = np.array([[1, .97, .90], [.97, 1, .96], [.90, .96, 1]])
    scheme = build_schemes(matrix, names, example)["r0p95"]
    assert all(len(block["features"]) < 3 for block in scheme["blocks"])


def test_overlap_is_rejected(example):
    path = Path(example["input"]["test_template"].format(imputation=1, fold=1))
    frame = pd.read_csv(path)
    frame.loc[0, "TX_ID"] = "T100"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_fold(example, 1, 1)


def test_full_dataset_fold_membership_is_shared_across_imputations(example):
    example["input"]["mode"] = "full_dataset"
    audits = [load_fold(example, imp, 1)[3] for imp in (1, 2)]
    assert audits[0] == audits[1]


def test_submitters_are_stdlib_only_and_all_jobs_use_bsub(example):
    expected = {"fibers": 4, "correlation": 4, "plots": 1}
    for stage, count in expected.items():
        result = subprocess.run([sys.executable, "-S", str(HERE / f"main_{stage}.py"),
                                 "--config", example["_config_path"], "--dry-run"],
                                capture_output=True, text=True, check=True)
        commands = [line for line in result.stdout.splitlines() if line.startswith("bsub ")]
        assert len(commands) == count
        assert all(f"run_{stage}.py" in line for line in commands)
        assert all("-w " not in line and "-K " not in line for line in commands)


def test_complete_three_stage_pipeline(example):
    from run_fibers import run as fit
    from run_correlation import run as correlate
    from run_plots import run as plot

    for imp, fold in itertools.product(example["imputations"], example["folds"]):
        fit(example, imp, fold, 1)
        correlate(example, imp, fold)
    plot(example)
    root = Path(example["output_root"])
    metrics = pd.read_csv(root / "summary/risk_comparison.csv.gz")
    assert {"HR", "Adj HR", "Adj NoAg HR", "changed_group_fraction"}.issubset(metrics)
    top = metrics.loc[(metrics["rank"] == 1) & (metrics.dataset == "test")]
    assert top["Adj HR"].notna().any()
    assert top["Adj NoAg HR"].notna().any()
    assert (root / "summary/figures/figure4_threshold_comparison.png").is_file()
    assert list((root / "summary/figures").glob("figure3*.png"))
    assert list((root / "summary/figures").glob("figure5*.png"))
    partition = pd.read_csv(root / "summary/cv_validation.csv")
    assert partition.partition_valid.all()
    summary = json.loads((root / "summary/completed.json").read_text())
    assert summary["figure_count"] > 0
    fit(example, 1, 1, 1)  # Compatible completed work is skipped.
    changed = copy.deepcopy(example)
    changed["fibers"]["iterations"] = 3
    with pytest.raises(RuntimeError, match="do not match"):
        require_result(root / "imp_01/cv_01/seed_001", identity(changed, "fibers", 1, 1, 1))
