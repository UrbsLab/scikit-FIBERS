"""Regression checks for the standalone permutation jobs and their HPC launchers."""

import contextlib
import csv
import gzip
import importlib
import io
import json
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
    assert result["train"].select_dtypes(exclude="number").empty
    if cv:
        assert result["test"].columns.tolist() == result["train"].columns.tolist()
        assert result["test"].select_dtypes(exclude="number").empty


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
    ("grf_fail", 998, "event indicators"),
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
    source = tmp_path / "Best-Typing_AA_MM_9loc_matrix_7_aligned.txt.gz"
    row = {column: "0" for column in prepare.BASE_COVARIATES}
    metadata = dict(zip(prepare.METADATA_COLUMNS, [
        "000123", "9007199254740993", "NA", "D-042", "2019-01-01", "", ".", "2022-02-03", "L",
    ]))
    row.update(metadata)
    row.update({"graftyrs": "2.5", "grf_fail": "1", "MM_A_9": "2",
                "MM_A_9_Y": "1", "MM_A_9_rF_dY": "2", "UNUSED_FIELD": "discard"})
    row["shared"] = ""
    with gzip.open(source, "wt", newline="") as handle:
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
    assert result.loc[0, "shared"] == ""
    reports = output / "missing_data_reports"
    details = pd.read_csv(reports / "BestTyping_7_missing_cells.csv", keep_default_na=False)
    assert details.loc[details.column == "shared", ["row_number", "TX_ID"]].values.tolist() == [[1, "NA"]]
    assert (reports / "BestTyping_7_missing_columns.csv").is_file()
    assert (reports / "BestTyping_7_missing_summary.json").is_file()


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
def test_preparation_keeps_processed_antigen_covariates_without_encoding(tmp_path, cov_loci):
    rows = []
    for count in (0, 1, 2):
        row = {column: "0" for column in prepare.BASE_COVARIATES}
        row.update({"graftyrs": "2.5", "grf_fail": "1", "MM_A_9_rF_dY": str(count),
                    "DON_TY": "L" if count == 0 else "C"})
        for source, _, _ in prepare.ANTIGEN_COVARIATES.values():
            row[source] = ["raw text", "NA", "998"][count]
        row["Agmm_notes"] = "unprocessed label"
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
    prepare.prepare_dataset(source, output, cov_loci=cov_loci, drop_missing=True)
    with output.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    excluded = {fields[0] for fields in prepare.ANTIGEN_COVARIATES.values()} | {"Agmm_notes"}
    assert result == [{column: value for column, value in row.items() if column not in excluded}
                      for row in rows]
    assert "Agmmc0" not in result[0]
    assert "Agmmc1" not in result[0]
    assert not excluded.intersection(result[0])
    assert [row["AgmmA2"] for row in result] == ["0", "0", "1"]


@pytest.mark.parametrize("compressed", [False, True])
def test_preparation_preserves_missing_cells_and_reports_all_source_columns(tmp_path, compressed):
    rows = []
    for tx_id in ("000123", "", "000123", "9007199254740993"):
        row = {column: "0" for column in prepare.BASE_COVARIATES}
        row.update({"TX_ID": tx_id, "DON_TY": "L", "graftyrs": "2.5", "grf_fail": "1",
                    "AgmmA2": "0", "REC_A_MM_EQUIV_CUR": "0", "MM_A_9": "0",
                    "MM_A_9_Y": "0", "MM_A_9_rF_dY": "0", "UNUSED": "present"})
        rows.append(row)
    rows[0].update(shared="", UNUSED="two\nlines")
    rows[1].update(shared=" NA ", graftyrs="NaN")
    rows[2].update(DON_TY="", AgmmA2="null", REC_A_MM_EQUIV_CUR=".",
                   MM_A_9="N/A", MM_A_9_Y=r"\N", MM_A_9_rF_dY="<NA>")
    rows[3].update(UNUSED="   ", grf_fail="None")
    source = tmp_path / ("SampleImp1.tsv.gz" if compressed else "SampleImp1.tsv")
    opener = gzip.open if compressed else open
    with opener(source, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerow(rows[0])
        handle.write("\n")  # Blank lines and a multiline field must not shift record numbers.
        writer.writerows(rows[1:])
    output = tmp_path / ("BestTyping_1.csv.gz" if compressed else "BestTyping_1.csv")
    summary = prepare.prepare_dataset(source, output)
    with opener(output, "rt", newline="") as handle:
        result = list(csv.DictReader(handle))
    expected = [{column: value for column, value in row.items()
                 if column not in ("UNUSED", "REC_A_MM_EQUIV_CUR")} for row in rows]
    expected[1].update(shared="", graftyrs="")
    expected[2].update(AgmmA2="", MM_A_9="", MM_A_9_Y="", MM_A_9_rF_dY="")
    expected[3].update(grf_fail="")
    assert result == expected
    numeric = pd.read_csv(output).drop(columns=["TX_ID", "DON_TY"])
    assert numeric.select_dtypes(exclude="number").empty
    assert numeric["shared"].isna().sum() == 2
    assert "shared" in summary["covariates"]
    assert summary["rows"] == 4
    report_dir = tmp_path / "missing_data_reports"
    with (report_dir / "BestTyping_1_missing_cells.csv").open(newline="") as handle:
        details = list(csv.DictReader(handle))
    expected_cells = [
        (1, 2, "shared"), (2, 5, "shared"), (2, 5, "TX_ID"), (2, 5, "graftyrs"),
        (3, 6, "DON_TY"), (3, 6, "AgmmA2"), (3, 6, "REC_A_MM_EQUIV_CUR"),
        (3, 6, "MM_A_9"), (3, 6, "MM_A_9_Y"), (3, 6, "MM_A_9_rF_dY"),
        (4, 7, "grf_fail"), (4, 7, "UNUSED"),
    ]
    assert details == [
        {"row_number": str(number), "source_line_number": str(line), "TX_ID": rows[number - 1]["TX_ID"],
         "column": column, "missing_value": rows[number - 1][column],
         "retained_in_output": str(column not in ("UNUSED", "REC_A_MM_EQUIV_CUR")), "row_dropped": "False"}
        for number, line, column in expected_cells
    ]
    with (report_dir / "BestTyping_1_missing_columns.csv").open(newline="") as handle:
        column_rows = list(csv.DictReader(handle))
    assert [entry["column"] for entry in column_rows] == list(rows[0])
    columns = {entry["column"]: entry for entry in column_rows}
    assert columns["shared"] == {"column": "shared", "retained_in_output": "True",
                                  "total_rows": "4", "missing_count": "2", "missing_percent": "50.000000"}
    assert columns["DCD"]["missing_count"] == "0"
    assert columns["UNUSED"]["missing_percent"] == "25.000000"
    assert columns["UNUSED"]["retained_in_output"] == "False"
    overall = json.loads((report_dir / "BestTyping_1_missing_summary.json").read_text())
    assert overall == summary["missing_summary"]
    assert overall["TX_ID_present"] is True
    assert overall["source"] == {"rows": 4, "columns": len(rows[0]), "missing_cells": 12,
                                 "rows_with_missing": 4, "columns_with_missing": 11}
    assert overall["prepared"] == {"rows": 4, "columns": len(rows[0]) - 2, "missing_cells": 10,
                                   "rows_with_missing": 4, "columns_with_missing": 9}
    assert overall["numeric_validation"]["passed"] is True
    assert overall["numeric_validation"]["rows_checked"] == 4
    assert overall["numeric_validation"]["columns_checked"] == len(numeric.columns)


def test_preparation_reports_missing_without_tx_id_and_refreshes_reports(tmp_path):
    frame = model_frame().drop(columns=list(prepare.METADATA_COLUMNS))
    frame["shared"] = ["", "1", "1", "0"]
    source = tmp_path / "SampleImp1.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "BestTyping_1.csv"
    result = prepare.prepare_dataset(source, output)
    with open(result["missing_reports"]["cells"], newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"row_number": "1", "source_line_number": "2",
            "TX_ID": "", "column": "shared", "missing_value": "", "retained_in_output": "True",
            "row_dropped": "False"}]
    assert result["missing_summary"]["TX_ID_present"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare.prepare_dataset(source, output)
    frame["shared"] = [0, 1, 1, 0]
    frame.to_csv(source, index=False)
    result = prepare.prepare_dataset(source, output, overwrite=True)
    with open(result["missing_reports"]["cells"], newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert result["missing_summary"]["source"]["missing_cells"] == 0
    with open(result["missing_reports"]["columns"], newline="") as handle:
        assert all(row["missing_count"] == "0" for row in csv.DictReader(handle))


@pytest.mark.parametrize("options,drop", [
    ([], False), (["--drop-missing", "false"], False),
    (["--drop-missing"], True), (["--drop-missing", "True"], True),
])
def test_drop_missing_is_opt_in_and_reports_dropped_rows(tmp_path, options, drop):
    rows = []
    for number in range(10):
        row = {column: "0" for column in prepare.BASE_COVARIATES}
        row.update({"TX_ID": f"{number:06}", "TFL_DEATH_DT": "2020-01-01",
                    "graftyrs": "1", "grf_fail": "0", "AgmmA2": "0", "REC_A_MM_EQUIV_CUR": "0",
                    "MM_A_9": "0", "MM_A_9_Y": "0", "MM_A_9_rF_dY": "0", "UNUSED": "present"})
        rows.append(row)
    for row, (column, marker) in zip(rows, [
        ("shared", "NA"), ("grf_fail", "null"), ("graftyrs", "NaN"), ("MM_A_9", " N/A "),
        ("MM_A_9_Y", "."), ("MM_A_9_rF_dY", ""), ("AgmmA2", "None"),
        ("DCD", "<NA>"),
    ]):
        row[column] = marker
    rows[8].update(TX_ID="", TFL_DEATH_DT="", UNUSED="NA")
    source = tmp_path / "SampleImp1.txt.gz"
    with gzip.open(source, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "BestTyping_1.csv.gz"
    prepare.main(["--d", str(source), "--o", str(output)] + options)
    with gzip.open(output, "rt", newline="") as handle:
        result = list(csv.DictReader(handle))
    expected = rows[8:] if drop else rows
    expected = [{column: value for column, value in row.items()
                 if column not in ("UNUSED", "REC_A_MM_EQUIV_CUR")} for row in expected]
    if not drop:
        for row, column in zip(expected, ["shared", "grf_fail", "graftyrs", "MM_A_9", "MM_A_9_Y",
                                         "MM_A_9_rF_dY", "AgmmA2", "DCD"]):
            row[column] = ""
    assert result == expected
    reports = tmp_path / "missing_data_reports"
    summary = json.loads((reports / "BestTyping_1_missing_summary.json").read_text())
    assert summary["drop_missing"] is drop
    assert summary["rows_dropped"] == (8 if drop else 0)
    assert summary["source"]["rows"] == 10
    assert summary["source"]["missing_cells"] == 11
    assert summary["prepared"]["rows"] == len(expected)
    assert summary["prepared"]["missing_cells"] == (2 if drop else 10)
    with (reports / "BestTyping_1_missing_cells.csv").open(newline="") as handle:
        details = list(csv.DictReader(handle))
    assert len(details) == 11
    for entry in details:
        number = int(entry["row_number"])
        assert entry["TX_ID"] == rows[number - 1]["TX_ID"]
        assert entry["missing_value"] == rows[number - 1][entry["column"]]
        assert entry["row_dropped"] == str(drop and number <= 8)


def test_drop_missing_all_rows_still_writes_reports_and_output_header(tmp_path):
    frame = model_frame()
    frame["shared"] = "NA"
    source = tmp_path / "SampleImp1.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "BestTyping_1.csv"
    result = prepare.prepare_dataset(source, output, drop_missing=True)
    assert result["rows"] == 0
    assert result["rows_dropped"] == 4
    assert result["missing_summary"]["prepared"]["missing_cells"] == 0
    assert result["missing_summary"]["source"]["rows"] == 4
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert "shared" in reader.fieldnames
        assert list(reader) == []


@pytest.mark.parametrize("cluster", ["LSF", "SLURM"])
@pytest.mark.parametrize("options,drop", [([], False), (["--drop-missing", "false"], False),
                                        (["--drop-missing"], True)])
def test_prepare_launcher_forwards_drop_missing(tmp_path, monkeypatch, cluster, options, drop):
    source = tmp_path / "Best-Typing_AA_MM_9loc_matrix_1_aligned.tsv"
    frame = model_frame()
    frame["shared"] = ["NA", "0", "1", "0"]
    frame.to_csv(source, sep="\t", index=False)
    output = tmp_path / "prepared"
    work = tmp_path / "work"
    with monkeypatch.context() as patcher:
        patcher.setattr(run_prepare.subprocess, "run", lambda *a, **k: pytest.fail("Dry run submitted"))
        run_prepare.main(["--d", str(source), "--o", str(output), "--w", str(work), "--rc", cluster,
                          "--python-executable", sys.executable, "--dry-run"] + options)
    script = next((work / "scratch").glob("*.sh"))
    command = shlex.split(script.read_text().splitlines()[-1])
    if cluster == "SLURM":
        assert command.pop(0) == "srun"
    assert ("--drop-missing" in command) is drop
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((output / "missing_data_reports/BestTyping_1_missing_summary.json").read_text())
    assert summary["drop_missing"] is drop
    assert summary["prepared"]["rows"] == (3 if drop else 4)


@pytest.mark.parametrize("value", ["bad", "inf", "-inf", "-1", "1.5"])
def test_invalid_nonmissing_values_leave_prepared_data_and_reports_unchanged(tmp_path, value):
    frame = model_frame()
    source = tmp_path / "SampleImp1.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "BestTyping_1.csv"
    result = prepare.prepare_dataset(source, output)
    files = [output, *(Path(path) for path in result["missing_reports"].values())]
    before = {path: path.read_bytes() for path in files}
    frame["MM_A_9_rF_dY"] = value
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Data row 1, column MM_A_9_rF_dY|nonnegative integer"):
        prepare.prepare_dataset(source, output, overwrite=True)
    assert {path: path.read_bytes() for path in files} == before
    assert not list(tmp_path.glob(".fibers-*"))


@pytest.mark.parametrize("cv", [False, True])
def test_workers_require_complete_shared_for_training(tmp_path, monkeypatch, cv):
    frame = model_frame()
    frame["shared"] = ["", "NA", "NaN", "1"]
    with pytest.raises(ValueError, match="finite numeric"):
        run_worker(tmp_path, monkeypatch, frame, cv=cv)


@pytest.mark.parametrize("column", ["shared", "AgmmA2"])
def test_preparation_rejects_text_in_processed_covariates(tmp_path, column):
    frame = model_frame()
    frame[column] = "unprocessed category"
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match=f"column {column}: invalid or non-finite numeric"):
        prepare.prepare_dataset(source, output)
    assert not output.exists()


def test_streaming_numeric_check_covers_later_rows_before_replacing_outputs(tmp_path):
    frame = model_frame()
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    frame.to_csv(source, index=False)
    result = prepare.prepare_dataset(source, output)
    paths = [output, *(Path(path) for path in result["missing_reports"].values())]
    before = {path: path.read_bytes() for path in paths}
    frame = pd.concat([frame.iloc[:1]] * 1001, ignore_index=True)
    frame["shared"] = "1"
    # Python float accepts underscores, but pandas CSV inference leaves them as text.
    frame.loc[1000, "shared"] = "1_000"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Data row 1001, column shared: invalid or non-finite numeric"):
        prepare.prepare_dataset(source, output, overwrite=True)
    assert {path: path.read_bytes() for path in paths} == before
    frame.loc[1000, "shared"] = "1000"
    frame.to_csv(source, index=False)
    result = prepare.prepare_dataset(source, output, overwrite=True)
    assert result["missing_summary"]["numeric_validation"]["rows_checked"] == 1001


@pytest.mark.parametrize("compressed", [False, True])
def test_preparation_runs_without_site_packages(tmp_path, compressed):
    source = tmp_path / ("input.tsv.gz" if compressed else "input.tsv")
    output = tmp_path / ("prepared.csv.gz" if compressed else "prepared.csv")
    frame = model_frame()
    frame["shared"] = ["998", "NA", "1.25e-2", "0"]
    frame.to_csv(source, sep="\t", index=False)
    # -S disables site-packages, including pandas/numpy, in the worker process.
    completed = subprocess.run([sys.executable, "-S", str(SCRIPTS / "job_prepare_fibers_data.py"),
                                "--d", str(source), "--o", str(output)],
                               check=True, capture_output=True, text=True)
    assert "Streaming numeric check passed" in completed.stdout
    data = pd.read_csv(output).drop(columns=list(prepare.METADATA_COLUMNS))
    assert data.select_dtypes(exclude="number").empty
    assert data["shared"].iloc[0] == 998
    assert pd.isna(data["shared"].iloc[1])
    assert data["shared"].iloc[2] == pytest.approx(0.0125)
    summary = json.loads((tmp_path / "missing_data_reports/prepared_missing_summary.json").read_text())
    assert summary["numeric_validation"]["method"] == "streaming numeric validation during conversion"
    assert summary["numeric_validation"]["rows_checked"] == len(frame)


@pytest.mark.parametrize("value", ["1_000", "１２", "\u00a01", "True", "1,000"])
def test_preparation_rejects_numeric_lookalikes_that_load_as_text(tmp_path, value):
    frame = model_frame()
    frame["shared"] = value
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="column shared: invalid or non-finite numeric"):
        prepare.prepare_dataset(source, output)
    assert not output.exists()


@pytest.mark.parametrize("cv", [False, True])
@pytest.mark.parametrize("value", ["998", "998.0", "9.98e2"])
def test_workers_accept_numeric_998(tmp_path, monkeypatch, cv, value):
    frame = model_frame()
    frame["shared"] = value
    result = run_worker(tmp_path, monkeypatch, frame, cv=cv)
    assert result["train"]["shared"].tolist() == [998] * 4
    if cv:
        assert result["test"]["shared"].tolist() == [998] * 4


def test_cv_accepts_numeric_998_in_test_data(tmp_path, monkeypatch):
    train = model_frame()
    test = train.copy()
    test["shared"] = "998.0"
    result = run_worker(tmp_path, monkeypatch, train, cv=True, test=test)
    assert result["test"]["shared"].tolist() == [998] * 4


@pytest.mark.parametrize("drop_missing", [False, True])
def test_preparation_preserves_998_without_reporting_or_dropping_it(tmp_path, drop_missing):
    values = ["998", "998.0", " 998 ", "9.98e2"]
    frame = model_frame().drop(columns=list(prepare.METADATA_COLUMNS))
    for column in ("shared", "DON_AGE", "TX_ID", "UNUSED"):
        frame[column] = values
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    frame.to_csv(source, index=False)
    result = prepare.prepare_dataset(source, output, drop_missing=drop_missing)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("shared", "DON_AGE", "TX_ID"):
        assert [row[column] for row in rows] == values
    assert result["rows_dropped"] == 0
    assert result["missing_summary"]["source"]["missing_cells"] == 0
    assert result["missing_summary"]["prepared"]["missing_cells"] == 0
    assert "998" not in result["missing_summary"]["missing_markers"]
    with open(result["missing_reports"]["cells"], newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with open(result["missing_reports"]["columns"], newline="") as handle:
        assert all(row["missing_count"] == "0" for row in csv.DictReader(handle))


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


@pytest.mark.parametrize("suffix,compressed", [
    (".tsv", False), (".txt", False), (".csv", False),
    (".tsv.gz", True), (".txt.gz", True), (".csv.gz", True),
    (".gz", True), ("", False), ("", True), (".TSV.GZ", True),
])
@pytest.mark.parametrize("delimiter", ["\t", ","])
def test_aligned_inputs_detect_format_and_preserve_values(tmp_path, suffix, compressed, delimiter):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / ("Best-Typing_AA_MM_9loc_matrix_7_aligned" + suffix)
    row = {column: "0" for column in prepare.BASE_COVARIATES}
    row.update({"graftyrs": "2.5", "grf_fail": "1", "PX_ID": "000123", "DON_TY": "L",
                "AgmmA2": "1", "REC_A_MM_EQUIV_CUR": "2", "MM_A_9": "2",
                "MM_A_9_Y": "1", "MM_A_9_rF_dY": "2"})
    opener = gzip.open if compressed else open
    with opener(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)
    plan = run_prepare.preparation_plan(inputs, tmp_path / "prepared")
    assert len(plan) == 1
    input_path, output_path = plan[0]
    assert input_path == source.resolve()
    assert output_path.name == "BestTyping_7.csv"
    prepare.main(["--d", str(input_path), "--o", str(output_path)])
    with output_path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{key: value for key, value in row.items()
                                                if key != "REC_A_MM_EQUIV_CUR"}]


def test_aligned_batch_numbering_natural_order_and_duplicate_formats(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("Best-Typing_AA_MM_9loc_matrix_10_aligned.gz",
                 "Best-Typing_AA_MM_9loc_matrix_2_aligned.tsv",
                 "Best-Typing_AA_MM_9loc_matrix_1_aligned", "README", "notes.md"):
        (inputs / name).touch()
    plan = run_prepare.preparation_plan(inputs, tmp_path / "output")
    assert [output.name for _, output in plan] == ["BestTyping_1.csv", "BestTyping_2.csv", "BestTyping_10.csv"]
    (inputs / "Best-Typing_AA_MM_9loc_matrix_1_aligned.txt.gz").touch()
    with pytest.raises(ValueError, match="Multiple inputs"):
        run_prepare.preparation_plan(inputs, tmp_path / "output")


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
