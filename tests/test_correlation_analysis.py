"""Regression tests for the three-stage ASHI pipeline, not part of the HPC bundle."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_analysis_codes" / "correlation_analysis"))

from common import configure_worker
configure_worker()

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from common import HERE, identity, load_config, require_result
from data import load_fold, retained_features
from methods import build_schemes, covariate_basis, expand_bin, fit_noag, pearson_matrix, score_groups


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


def test_matrix_and_block_math(example):
    frame, _, features, _ = load_fold(example, 1, 1)
    matrix = pearson_matrix(frame, features, 13)
    np.testing.assert_allclose(matrix, frame[features].corr(), atol=1e-12)
    schemes = build_schemes(matrix, features, example)
    for name, scheme in schemes.items():
        for block in scheme["blocks"]:
            indexes = [features.index(f) for f in block["features"]]
            values = matrix[np.ix_(indexes, indexes)][np.triu_indices(len(indexes), 1)]
            assert np.all(values > scheme["thresholds"][block["features"][0].split("_")[1]])
    score_data = pd.DataFrame({"MM_DQA1_11": [1, 0, 2], "MM_DQA1_18": [1, 1, 2]})
    groups = expand_bin(list(score_data), schemes["r0p95"])
    assert score_groups(score_data, groups).tolist() == [1, 1, 2]


def test_complete_linkage_does_not_chain(example):
    names = ["MM_A_10", "MM_A_20", "MM_A_30"]
    matrix = np.array([[1, .97, .90], [.97, 1, .96], [.90, .96, 1]])
    scheme = build_schemes(matrix, names, example)["r0p95"]
    assert all(len(block["features"]) < 3 for block in scheme["blocks"])


def test_rare_filter_matches_training_nonzero_frequency_and_boundary(example):
    frame = pd.DataFrame({"MM_A_1": [0] * 2000, "MM_A_2": [1] * 2000,
                          "MM_A_3": [2] * 2000, "MM_A_4": [1] + [0] * 1999,
                          "MM_A_5": [1, 2] + [0] * 1998, "MM_A_6": [0, 2] * 1000})
    example["rare_filter"] = .001
    kept, stats = retained_features(frame, list(frame), example)
    assert kept == ["MM_A_5", "MM_A_6"]
    assert stats.set_index("feature").loc["MM_A_4", "nonzero_frequency"] == .0005
    assert stats.invariant.tolist() == [True, True, True, False, False, False]
    example["rare_filter"] = 0
    assert retained_features(frame, list(frame), example)[0] == ["MM_A_4", "MM_A_5", "MM_A_6"]


def test_filter_is_identical_for_fibers_and_correlations_not_refit_on_test(example):
    train, test, features, _ = load_fold(example, 1, 1)
    feature = features[0]
    train[feature] = 0
    test[feature] = 1
    kept, _ = retained_features(train, features, example)
    assert feature not in kept
    assert list(train[kept].columns) == list(test[kept].columns)


def test_any_locus_blocks_use_one_strict_cutoff(example):
    names = ["MM_A_10", "MM_DQA1_11", "MM_DQB1_84"]
    matrix = np.array([[1, .96, .95], [.96, 1, .99], [.95, .99, 1]])
    schemes = build_schemes(matrix, names, example)
    assert schemes["r0p95"]["blocks"] == []
    any_blocks = schemes["any_r0p95"]["blocks"]
    assert len(any_blocks) == 1
    assert set(any_blocks[0]["features"]) == {"MM_DQA1_11", "MM_DQB1_84"}


@pytest.mark.parametrize("r", [.95, -.99])
def test_any_locus_rejects_equality_and_negative_correlations(example, r):
    names = ["MM_A_10", "MM_DQA1_11"]
    matrix = np.array([[1., r], [r, 1.]])
    assert build_schemes(matrix, names, example)["any_r0p95"]["blocks"] == []


def test_mixed_block_is_counted_once_but_preserves_nonbinary_mismatches():
    frame = pd.DataFrame({"MM_A_10": [1, 0, 2, 0], "MM_DQA1_11": [1, 1, 1, 0],
                          "MM_DQB1_84": [0, 0, 1, 0]})
    groups = [{"features": ["MM_A_10", "MM_DQA1_11"]}, {"features": ["MM_DQB1_84"]}]
    np.testing.assert_array_equal(score_groups(frame, groups), [1, 1, 3, 0])


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
    expected = {"fibers": 4, "correlation": 4, "plots": 1, "imp_summary": 2}
    for stage, count in expected.items():
        result = subprocess.run([sys.executable, "-S", str(HERE / f"main_{stage}.py"),
                                 "--config", example["_config_path"], "--dry-run"],
                                capture_output=True, text=True, check=True)
        commands = [line for line in result.stdout.splitlines() if line.startswith("bsub ")]
        assert len(commands) == count
        assert all(f"run_{stage}.py" in line for line in commands)
        assert all("-w " not in line and "-K " not in line for line in commands)
    summary = subprocess.run([sys.executable, "-S", str(HERE / "main_imp_summary.py"),
                              "--config", example["_config_path"], "--summary-only", "--dry-run"],
                             capture_output=True, text=True, check=True)
    commands = [line for line in summary.stdout.splitlines() if line.startswith("bsub ")]
    assert len(commands) == 1 and "--summary-only" in commands[0]
    assert "--imputation " not in commands[0]


def test_complete_three_stage_pipeline(example):
    from run_fibers import run as fit
    from run_correlation import run as correlate
    from run_plots import run as plot

    example["correlation"]["thresholds"] = [.95]
    example["plots"]["thresholds"] = [.95]

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
    assert "any_r0p95" in set(top.scheme)
    assert (root / "summary/top_bin_summary.csv").is_file()
    assert list((root / "summary/figures").glob("figure2*any_r0p95*.png"))
    for scope in ("r0p95", "any_r0p95"):
        for number in (1, 2, 3):
            assert list((root / "summary/figures").glob(f"figure{number}*_{scope}*.png"))
            assert list((root / "summary/figures").glob(f"figure{number}*_{scope}*.pdf"))
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


def test_whole_imputation_identity_does_not_depend_on_cv_splits(example):
    from run_imp_summary import result_identity
    expected = result_identity(example, "analysis", 1)
    changed = copy.deepcopy(example)
    changed["folds"] = list(range(1, 11))
    changed["input"].update(split_seed=123, train_template="missing.csv", test_template="missing.csv", mode="full_dataset")
    assert expected == result_identity(changed, "analysis", 1)


def test_whole_imputation_consistency_keeps_seeds_separate(example):
    from run_imp_summary import consistency_rows
    records = []
    config = {**example, "seeds": [1, 2]}
    for seed, imp in itertools.product(config["seeds"], config["imputations"]):
        for scheme in ("r0p95", "any_r0p95"):
            records.append({"imputation": imp, "seed": seed, "rank": 1, "scheme": scheme,
                            "original": [f"MM_A_{10 * imp}"],
                            "groups": [{"features": ["MM_A_10", "MM_A_20"]}]})
    table = consistency_rows(records, config)
    assert len(table) == 6
    assert table.loc[table.scheme == "original", "jaccard"].eq(0).all()
    assert table.loc[table.scheme != "original", "jaccard"].eq(1).all()
    assert set(table.scope) == {"seed1", "seed2"}


def test_whole_imputation_summary_requires_multiple_completed_imputations(example):
    from run_imp_summary import summarize
    with pytest.raises(ValueError, match="at least two"):
        summarize({**example, "imputations": [1]})
    with pytest.raises(RuntimeError, match="Required job has not completed"):
        summarize(example)
    assert not (Path(example["output_root"]) / "imp_summary/completed.json").exists()


def test_complete_whole_imputation_pipeline(example, monkeypatch):
    import data
    from methods import evaluate, jaccard
    from run_imp_summary import run, summarize

    example["correlation"]["thresholds"] = [.95]
    example["plots"]["thresholds"] = [.95]
    # Whole-cohort jobs must neither reuse nor construct folds.
    def reject_fold(*args, **kwargs):
        raise AssertionError("Whole-imputation analysis tried to load a CV fold")
    monkeypatch.setattr(data, "load_fold", reject_fold)
    example["input"].update(train_template="does_not_exist.csv", test_template="does_not_exist.csv")
    for imp in example["imputations"]:
        run(example, imp)
    summarize(example)
    root = Path(example["output_root"]) / "imp_summary"
    assert not list(Path(example["output_root"]).glob("imp_*/cv_*"))
    metrics = pd.read_csv(root / "risk_comparison.csv.gz")
    assert set(metrics.dataset) == {"full_cohort"}
    assert "fold" not in metrics and metrics.n.eq(180).all()
    assert {"original", "r0p95", "any_r0p95"} == set(metrics.scheme)
    top = pd.read_csv(root / "top_bin_metrics.csv")
    assert len(top) == 6
    for measure in ("HR", "Adj NoAg HR", "Adj HR"):
        assert top[measure].notna().all()
        assert {measure + " lower", measure + " upper", measure + " delta"}.issubset(top)
    summary = pd.read_csv(root / "top_bin_summary.csv")
    assert summary.n_imputations.eq(2).all() and summary.n_pairs.eq(1).all()
    assert summary.evaluation.eq("full_cohort_apparent").all()
    original = []
    for imp in (1, 2):
        path = root / f"imp_{imp:02d}"
        pop = pd.read_csv(path / "seed_001/population.csv")
        fit = json.loads((path / "seed_001/completed.json").read_text())
        assert fit["n_training"] == 180
        features = json.loads(pop.iloc[0].features)
        original.append(set(features))
        frame = pd.read_csv(example["input"]["full_template"].format(imputation=imp))
        expected = evaluate(frame, frame[features].sum(axis=1).to_numpy(), pop.iloc[0].threshold,
                            example, ["age"], ["Agmm0"], True, True)
        row = top.loc[(top.imputation == imp) & (top.scheme == "original")].iloc[0]
        for measure in ("HR", "Adj NoAg HR", "Adj HR"):
            assert row[measure] == pytest.approx(expected[measure])
        pairs = pd.read_csv(path / "correlations.csv.gz")
        pair = pairs.loc[(pairs.feature1 == "MM_A_10") & (pairs.feature2 == "MM_A_20")].iloc[0]
        assert pair.pearson_r == pytest.approx(frame[["MM_A_10", "MM_A_20"]].corr().iloc[0, 1])
    assert summary.loc[summary.scheme == "original", "mean_jaccard"].iloc[0] == pytest.approx(jaccard(*original))
    assert pd.read_csv(root / "correlation_summary.csv.gz").imputations_available.eq(2).all()
    captions = (root / "figure_captions.txt").read_text()
    assert "whole-imputation fits" in captions and "training-cohort estimates" in captions
    for scope in ("r0p95", "any_r0p95"):
        for extension in ("pdf", "png"):
            assert (root / f"figures/figure2_top_imputations_seed1_{scope}.{extension}").is_file()
            assert list((root / "figures").glob(f"figure1*_{scope}*.{extension}"))
    assert (root / "figures/figure4_imputation_comparison_seed1.png").is_file()
    assert list((root / "figures").glob("figure5_whole_imputations*.png"))
    mtime = (root / "imp_01/seed_001/fibers.pkl").stat().st_mtime_ns
    run(example, 1)
    summarize(example)
    assert mtime == (root / "imp_01/seed_001/fibers.pkl").stat().st_mtime_ns
    # A nominally completed result is still rejected if it used different outcomes.
    marker = root / "imp_02/completed.json"
    result = json.loads(marker.read_text())
    result["cohort_audit"]["outcomes_sha256"] = "different"
    marker.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="outcomes differ"):
        summarize(example, force=True)
