"""Regression checks for the standalone permutation jobs and their HPC launchers."""

import contextlib
import csv
import importlib
import io
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from paper_analysis_codes.hla_perm_hpc_scripts import job_prepare_fibers_data as prepare
from paper_analysis_codes.hla_perm_hpc_scripts import run_prepare_fibers_data as run_prepare
from paper_analysis_codes.hla_perm_hpc_scripts.job_HLA_CV_Partitioner import load_and_split_dataset

PACKAGE = "paper_analysis_codes.hla_perm_hpc_scripts"
SCRIPTS = Path(prepare.__file__).parent


def model_frame():
    frame = pd.DataFrame({column: [0, 1, 0, 1] for column in prepare.BASE_COVARIATES})
    frame["graftyrs"] = [1, 2, 3, 4]
    frame["grf_fail"] = [1, 0, 1, 1]
    for feature in ("MM_A_9_rF_dY", "MM_A_9_rY_dF", "MM_A_182_rF_dY",
                    "MM_A_183_rF_dY", "MM_DRB1_5_rF_dY", "MM_DRB1_6_rF_dY",
                    "MM_DRB1_94_rF_dY", "MM_DRB1_95_rF_dY", "MM_DQB1_95_rF_dY",
                    "MM_DQB1_96_rF_dY", "MM_DPB1_8_rF_dY"):
        frame[feature] = [0, 0, 2, 2]
    frame["MM_A_10_rF_dY"] = [0, 0, 0, 1]
    frame["MM_A_11_rF_dY"] = 0
    for name in ("MM_A_9", "MM_A_9_Y", "MM_A_COUNT", "PX_ID", "MM_A_9_rF_dY_extra"):
        frame[name] = 1
    for name in prepare.METADATA_COLUMNS:
        frame[name] = ["0001", "NA", "", "2020-01-01"]
    return frame


def run_worker(tmp_path, monkeypatch, frame, cv=False, test=None, extra=()):
    module = importlib.import_module(PACKAGE + ".job_fibers_hpc" + ("_cv" if cv else ""))
    records = {}

    class FakeFIBERS:
        def __init__(self, **kwargs):
            records["parameters"] = kwargs
            self.set = SimpleNamespace(bin_pop=[object()])

        def fit(self, data):
            records["train"] = data.copy()
            return self

    def evaluate(model, data, output, prefix, split, covariates, base_covariates):
        records[split or "evaluation"] = data.copy()

    monkeypatch.setattr(module, "FIBERS", FakeFIBERS)
    monkeypatch.setattr(module, "save_model", lambda *args: None)
    monkeypatch.setattr(module, "evaluate_model", evaluate)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if cv:
        source = tmp_path / "folds"
        source.mkdir()
        frame.to_csv(source / "long_dataset_name_CV_1_Train.csv", index=False)
        (test if test is not None else frame).to_csv(source / "long_dataset_name_CV_1_Test.csv", index=False)
    else:
        source = tmp_path / "BestTyping_1.csv"
        frame.to_csv(source, index=False)
    args = ["job", "--d", str(source), "--o", str(tmp_path / "output"),
            "--cov-list", "None", "--loci-list", "A,DRB1,DQB1", "--r", "2"]
    if cv:
        args += ["--cv", "1"]
    with contextlib.redirect_stdout(io.StringIO()):
        module.main(args + list(extra))
    return records


@pytest.mark.parametrize("cv", [False, True])
@pytest.mark.parametrize("all_positions", [False, True])
def test_worker_preserves_direction_counts_and_all_positions(tmp_path, monkeypatch, cv, all_positions):
    frame = model_frame()
    result = run_worker(tmp_path, monkeypatch, frame, cv=cv,
                        extra=["--all-positions"] if all_positions else [])
    expected = ["MM_A_9_rF_dY", "MM_A_9_rY_dF", "MM_A_182_rF_dY",
                "MM_DRB1_6_rF_dY", "MM_DRB1_94_rF_dY", "MM_DQB1_95_rF_dY", "MM_A_10_rF_dY",
                "MM_A_183_rF_dY", "MM_DRB1_5_rF_dY", "MM_DRB1_95_rF_dY", "MM_DQB1_96_rF_dY"]
    features = [name for name in result["train"] if name.startswith("MM_")]
    assert set(features) == set(expected)
    assert result["train"]["MM_A_9_rF_dY"].tolist() == [0, 0, 2, 2]
    assert result["parameters"]["covariates"] == list(prepare.BASE_COVARIATES)
    assert not set(prepare.METADATA_COLUMNS) & set(result["train"].columns)
    if cv:
        assert result["test"].columns.tolist() == result["train"].columns.tolist()


def test_cv_rare_filter_uses_training_only(tmp_path, monkeypatch):
    train = model_frame()
    test = train.copy()
    test["MM_A_9_rF_dY"] = 0
    test["MM_A_10_rF_dY"] = 2
    result = run_worker(tmp_path, monkeypatch, train, cv=True, test=test, extra=["--ra", "0.5"])
    assert "MM_A_10_rF_dY" not in result["train"]
    assert result["train"].columns.tolist() == result["test"].columns.tolist()
    assert (result["test"]["MM_A_9_rF_dY"] == 0).all()
    audit = pd.read_csv(tmp_path / "output/1_rare_filter_feature_frequencies.csv").set_index("feature")
    assert not audit.loc["MM_A_10_rF_dY", "kept_for_training"]


@pytest.mark.parametrize("column,value,error", [
    ("MM_A_9_rF_dY", float("nan"), "finite numeric"),
    ("MM_A_9_rF_dY", -1, "nonnegative integer"),
    ("MM_A_9_rF_dY", 1.5, "nonnegative integer"),
    ("grf_fail", 2, "event indicators"),
])
def test_invalid_training_data_rejected(tmp_path, monkeypatch, column, value, error):
    frame = model_frame()
    frame[column] = value
    with pytest.raises(ValueError, match=error):
        run_worker(tmp_path, monkeypatch, frame)


def test_cv_missing_required_test_feature_rejected(tmp_path, monkeypatch):
    frame = model_frame()
    with pytest.raises(ValueError, match="Testing fold is missing"):
        run_worker(tmp_path, monkeypatch, frame, cv=True, test=frame.drop(columns="MM_A_9_rF_dY"))


def test_antigen_covariates_deduplicated(tmp_path, monkeypatch):
    frame = model_frame()
    frame["AgmmDR0"] = [0, 1, 0, 1]
    frame["AgmmDR1"] = [1, 0, 1, 0]
    result = run_worker(tmp_path, monkeypatch, frame, extra=["--cov-list", "DRB1,DRB345"])
    assert result["parameters"]["covariates"] == list(prepare.BASE_COVARIATES) + ["AgmmDR0", "AgmmDR1"]
    assert len(prepare.get_covariates()) == 55


@pytest.mark.parametrize("cluster", ["LSF", "SLURM"])
@pytest.mark.parametrize("name", [
    "run_fibers_hpc", "run_fibers_hpc_cv", "run_sum_fibers_hpc", "run_sum_fibers_hpc_cv",
    "run_master_sum_fibers_hpc", "run_master_sum_fibers_hpc_cv",
    "run_table_maker_hpc", "run_table_maker_hpc_cv", "run_HLA_CV_Partitioner",
])
def test_standalone_scheduler_routes_and_options(tmp_path, monkeypatch, cluster, name):
    module = importlib.import_module(PACKAGE + "." + name)
    data = tmp_path / "input with spaces"
    data.mkdir()
    (data / "best_typing_1.csv").write_text("feature\n1\n")
    for fold in (1, 2):
        for split in ("Train", "Test"):
            (data / f"long_name_CV_{fold}_{split}.csv").write_text("feature\n1\n")
    work = tmp_path / "results with spaces"
    args = [name, "--w", str(work), "--rc", cluster, "--dry-run", "--python-executable", sys.executable]
    if name.endswith("_cv"):
        args += ["--cv", "2"]
        if name == "run_fibers_hpc_cv":
            args += ["--d", str(data), "--o", "perm", "--r", "7", "--all-positions", "--de", "protective"]
    elif name == "run_HLA_CV_Partitioner":
        args += ["--d", str(data / "best_typing_1.csv"), "--o", str(tmp_path / "folds"), "--cv", "2"]
    else:
        args += ["--o", "perm", "--rs", "2", "--re", "1"]
        if name != "run_table_maker_hpc":
            args += ["--d", str(data)]
        if name == "run_fibers_hpc":
            args += ["--all-positions", "--de", "high_risk"]
    with monkeypatch.context() as patcher:
        patcher.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Dry run submitted a job"))
        with contextlib.redirect_stdout(io.StringIO()):
            module.main(args)
    scripts = sorted((work / "scratch").glob("*.sh"))
    assert len(scripts) == (2 if name in ("run_fibers_hpc", "run_fibers_hpc_cv") else 1)
    seeds = set()
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
        command = shlex.split(script.read_text().splitlines()[-1])
        if cluster == "SLURM":
            assert command.pop(0) == "srun"
        assert command[:2] == [sys.executable, str(SCRIPTS / (name.replace("run_", "job_", 1) + ".py"))]
        if name.startswith("run_fibers_hpc"):
            assert "--all-positions" in command
            assert command.count("--r") == 1
            seeds.add(command[command.index("--r") + 1])
            assert command[command.index("--de") + 1] == ("protective" if name.endswith("_cv") else "high_risk")
        if name == "run_master_sum_fibers_hpc":
            assert command.count("--rs") == 1
            assert command[command.index("--rs") + 1] == "2"
            assert command[command.index("--re") + 1] == "1"
        if name == "run_HLA_CV_Partitioner":
            assert command[command.index("--cv") + 1] == "2"
    if seeds:
        assert seeds == ({"7"} if name.endswith("_cv") else {"0", "1"})


def test_cv_launcher_preflights_all_pairs(tmp_path, monkeypatch):
    module = importlib.import_module(PACKAGE + ".run_fibers_hpc_cv")
    (tmp_path / "example_1_Train.csv").touch()
    (tmp_path / "example_1_Test.csv").touch()
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Submitted before validation"))
    with pytest.raises(ValueError, match="found 0"):
        module.main(["run", "--d", str(tmp_path), "--w", str(tmp_path / "work"), "--cv", "2"])


def test_partitioner_disjoint_complete_folds(tmp_path):
    source = tmp_path / "best_typing_1.csv"
    original = pd.DataFrame({"row": range(12)})
    for name in prepare.METADATA_COLUMNS:
        original[name] = ["0001", "NA", "", "2020-01-01"] * 3
    original.to_csv(source, index=False)
    output = tmp_path / "folds"
    load_and_split_dataset(str(source), str(output), 3)
    seen = []
    for fold in (1, 2, 3):
        train = pd.read_csv(output / f"best_typing_1_CV_{fold}_Train.csv", keep_default_na=False)
        test = pd.read_csv(output / f"best_typing_1_CV_{fold}_Test.csv", keep_default_na=False)
        assert not set(train.row) & set(test.row)
        assert set(train.row) | set(test.row) == set(range(12))
        pd.testing.assert_frame_equal(
            pd.concat([train, test]).sort_values("row").reset_index(drop=True), original)
        seen.extend(test.row)
    assert sorted(seen) == list(range(12))


@pytest.mark.parametrize("cluster", ["LSF", "SLURM"])
def test_prepare_generated_job_keeps_all_mismatch_types(tmp_path, monkeypatch, cluster):
    source = tmp_path / "SampleImp7.txt"
    row = {column: "0" for column in prepare.BASE_COVARIATES}
    metadata = dict(zip(prepare.METADATA_COLUMNS, [
        "000123", "9007199254740993", "NA", "D-042", "2019-01-01", "", ".", "2022-02-03", "L",
    ]))
    row.update(metadata)
    row.update({"graftyrs": "2.5", "grf_fail": "1", "MM_A_9": "2",
                "MM_A_9_Y": "1", "MM_A_9_rF_dY": "2", "UNUSED_FIELD": "discard"})
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "prepared files"
    work = tmp_path / "jobs"
    with monkeypatch.context() as patcher:
        patcher.setattr(run_prepare.subprocess, "run", lambda *a, **k: pytest.fail("Dry run submitted"))
        run_prepare.main(["--d", str(source), "--w", str(work), "--o", str(output),
                          "--rc", cluster, "--cov-list", "None", "--python-executable", sys.executable, "--dry-run"])
    script = next((work / "scratch").glob("*.sh"))
    subprocess.run(["bash", "-n", str(script)], check=True)
    command = shlex.split(script.read_text().splitlines()[-1])
    if cluster == "SLURM":
        assert command.pop(0) == "srun"
    assert Path(command[1]) == SCRIPTS / "job_prepare_fibers_data.py"
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    result = pd.read_csv(output / "BestTyping_7.csv", dtype={column: str for column in metadata},
                         keep_default_na=False)
    assert result.columns.tolist() == list(prepare.METADATA_COLUMNS) + list(prepare.BASE_COVARIATES) + ["graftyrs", "grf_fail", "MM_A_9", "MM_A_9_Y", "MM_A_9_rF_dY"]
    assert result["MM_A_9_rF_dY"].tolist() == [2]
    assert result.loc[0, list(metadata)].to_dict() == metadata


@pytest.mark.parametrize("cov_list", ["None", ",".join(prepare.DEFAULT_COV_LOCI)])
def test_preparation_accepts_missing_metadata_without_adding_columns(tmp_path, capsys, cov_list):
    source = tmp_path / "SampleImp1.csv"
    frame = model_frame().drop(columns=list(prepare.METADATA_COLUMNS))
    frame.to_csv(source, index=False)
    output = tmp_path / "BestTyping_1.csv"
    prepare.main(["--d", str(source), "--o", str(output), "--cov-list", cov_list])
    assert not set(prepare.METADATA_COLUMNS) & set(pd.read_csv(output).columns)
    messages = capsys.readouterr().out
    assert not any(column in messages for column in prepare.METADATA_COLUMNS)


@pytest.mark.parametrize("cov_loci", [[], ["A"], prepare.DEFAULT_COV_LOCI])
def test_preparation_keeps_all_antigen_fields_without_encoding(tmp_path, cov_loci):
    rows = []
    for count in (0, 1, 2):
        row = {column: "0" for column in prepare.BASE_COVARIATES}
        row.update({"graftyrs": "2.5", "grf_fail": "1", "MM_A_9_rF_dY": str(count),
                    "DON_TY": "L" if count == 0 else "C"})
        for source, _, _ in prepare.ANTIGEN_COVARIATES.values():
            row[source] = str(count)
        for locus in ("A", "B", "DR", "DPB1"):
            for category in (0, 1, 2):
                row[f"Agmm{locus}{category}"] = str(int(count == category))
        rows.append(row)
    source = tmp_path / "SampleImp1.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "BestTyping_1.csv"
    prepare.prepare_dataset(source, output, cov_loci=cov_loci)
    with output.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    assert result == rows
    assert "Agmmc0" not in result[0]
    assert "Agmmc1" not in result[0]
    assert [row["REC_C_MM_EQUIV_CUR"] for row in result] == ["0", "1", "2"]
    assert [row["AgmmA2"] for row in result] == ["0", "0", "1"]


@pytest.mark.parametrize("cv", [False, True])
def test_workers_ignore_raw_antigen_counts_and_use_existing_covariates(tmp_path, monkeypatch, cv):
    train = model_frame()
    test = train.copy()
    for source, _, _ in prepare.ANTIGEN_COVARIATES.values():
        train[source] = ["unused", "raw", "antigen", "counts"]
        test[source] = ["unused", "test", "antigen", "counts"]
    train["AgmmA0"] = [1, 0, 0, 0]
    train["AgmmA1"] = [0, 1, 0, 0]
    test["AgmmA0"] = [0, 0, 1, 0]
    test["AgmmA1"] = [0, 1, 0, 1]
    test["Agmmc0"] = 1  # A test-only covariate must not change the training schema.
    train["AgmmA2"] = [0, 0, 1, 1]
    test["AgmmA2"] = [1, 0, 0, 0]
    result = run_worker(tmp_path, monkeypatch, train, cv=cv, test=test,
                        extra=["--cov-list", ",".join(prepare.DEFAULT_COV_LOCI)])
    assert result["parameters"]["covariates"] == list(prepare.BASE_COVARIATES) + ["AgmmA0", "AgmmA1"]
    for locus in prepare.DEFAULT_COV_LOCI:
        source, zero, one = prepare.ANTIGEN_COVARIATES[locus]
        assert source not in result["train"]
        if locus != "A":
            assert zero not in result["train"]
            assert one not in result["train"]
    assert result["train"]["AgmmA0"].tolist() == train["AgmmA0"].tolist()
    assert result["train"]["AgmmA1"].tolist() == train["AgmmA1"].tolist()
    if cv:
        assert result["test"]["AgmmA0"].tolist() == test["AgmmA0"].tolist()
        assert result["test"]["AgmmA1"].tolist() == test["AgmmA1"].tolist()
        assert "Agmmc0" not in result["test"]
    assert "AgmmA2" not in result["train"]
    source = tmp_path / ("folds/long_dataset_name_CV_1_Train.csv" if cv else "BestTyping_1.csv")
    pd.testing.assert_frame_equal(pd.read_csv(source, keep_default_na=False), train)


@pytest.mark.parametrize("cv", [False, True])
def test_workers_support_dp_loci_outside_old_ranges(tmp_path, monkeypatch, cv):
    frame = model_frame()
    frame["MM_DPA1_228_rF_dY"] = [0, 1, 2, 0]
    frame["MM_DPB1_215_rF_dY"] = [1, 0, 2, 0]
    result = run_worker(tmp_path, monkeypatch, frame, cv=cv, extra=["--loci-list", "DPA1,DPB1"])
    assert {column for column in result["train"] if column.startswith("MM_")} == {
        "MM_DPA1_228_rF_dY", "MM_DPB1_215_rF_dY", "MM_DPB1_8_rF_dY",
    }


def test_cv_requires_selected_source_covariates_in_test_fold(tmp_path, monkeypatch):
    train = model_frame()
    train["AgmmA0"] = [0, 1, 0, 0]
    test = train.drop(columns="AgmmA0")
    with pytest.raises(ValueError, match="Testing fold is missing required columns: AgmmA0"):
        run_worker(tmp_path, monkeypatch, train, cv=True, test=test, extra=["--cov-list", "A"])


def test_preparation_numbering_and_duplicate_rejection(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for name in ("SampleImp10.txt.gz", "SampleImp2.txt", "SampleImp1.tsv", "ignore.md"):
        (source / name).touch()
    plan = run_prepare.preparation_plan(source, tmp_path / "output")
    assert [output.name for _, output in plan] == ["BestTyping_1.csv", "BestTyping_2.csv", "BestTyping_10.csv"]
    (source / "Combined1.csv").touch()
    with pytest.raises(ValueError, match="Multiple inputs"):
        run_prepare.preparation_plan(source, tmp_path / "output")


def test_cv_reporting_preserves_full_adjusted_model(tmp_path):
    module = importlib.import_module(PACKAGE + ".job_fibers_hpc_cv")

    class Model:
        feature_names = ["MM_A_9_rF_dY"]
        adjusted = None

        def get_cox_prop_hazard_adjusted(self, data, new_covariates=None):
            self.adjusted = "base" if new_covariates is not None else "full"
            self.feature_names = ["changed_during_report"]
            return pd.DataFrame({"exp(coef)": [1.2]})

        def get_cox_prop_hazard_unadjust(self, data):
            return pd.DataFrame({"exp(coef)": [1.3]})

        def get_kaplan_meir(self, *args, **kwargs):
            pass

    model = Model()
    frame = pd.DataFrame({"MM_A_9_rF_dY": [1], "base": [1], "AgmmA0": [0]})
    module.evaluate_model(model, frame, tmp_path, "1", "train", ["base", "AgmmA0"], ["base"])
    assert model.feature_names == ["MM_A_9_rF_dY"]
    assert model.adjusted == "full"
    assert (tmp_path / "1_coxph_adj_bin_train_0_NoAg.csv").exists()


@pytest.mark.parametrize("suffix", ["", "_cv"])
def test_table_statistics_handle_identical_and_missing_metrics(suffix):
    module = importlib.import_module(PACKAGE + ".job_table_maker_hpc" + suffix)
    assert not module.wilcoxon_sig(pd.Series([1, 1]), pd.Series([1, 1]), 0.05)
    assert not module.wilcoxon_sig(pd.Series([float("nan"), 1]), pd.Series([2, 3]), 0.05)
    assert module.wilcoxon_sig(pd.Series(range(10)), pd.Series(range(1, 11)), 0.05)
