"""Cohort boundaries, outcome censoring, and scheduler propagation."""

import csv
import json
import shlex
import subprocess
import sys
from datetime import date

import pytest

from paper_analysis_codes.hla_perm_hpc_scripts import job_prepare_fibers_data as prepare
from paper_analysis_codes.hla_perm_hpc_scripts import run_prepare_fibers_data as launcher


FILTERS = dict(tx_year_min=2010, tx_year_max=2025, kidney_only=True,
               deceased_donor_only=True, recipient_age_min=18, donor_age_min=9,
               censor_date="2025-12-31", drop_missing=True)
FLAGS = ["--tx-year-min", "2010", "--tx-year-max", "2025", "--kidney-only",
         "--deceased-donor-only", "--recipient-age-min", "18", "--donor-age-min", "9",
         "--censor-date", "2025-12-31", "--days-per-year", "365.24", "--drop-missing"]


def source_row(**updates):
    row = {column: "0" for column in prepare.BASE_COVARIATES}
    row.update(PX_ID="000001", TX_ID="000101", REC_TX_DT="01/01/15", ORG_TY="KI",
               REC_TX_ORG_TY="KI", DON_TY="C", REC_AGE_AT_TX="40", DON_AGE="30",
               graftyrs="1", grf_fail="1", MM_A_9="2", MM_A_9_Y="1",
               MM_A_9_rF_dY="2", AgmmA2="1", TFL_GRAFT_DT="", TFL_DEATH_DT="",
               strobe_year="0", strobe_age="0", strobe_type="0", strobe_ki_only="0")
    row.update(updates)
    return row


def write_source(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_requested_cohort_boundaries_missingness_and_exclusion_audit(tmp_path):
    cases = [
        ({"REC_TX_DT": "01/01/10", "REC_AGE_AT_TX": "18", "DON_AGE": "9"}, None),
        ({"REC_TX_DT": "2025-12-31", "graftyrs": "0"}, None),
        ({"REC_TX_DT": "12/31/09"}, "transplant_before_min_year"),
        ({"REC_TX_DT": "01/01/26"}, "transplant_after_max_year"),
        ({"REC_TX_DT": ""}, "missing_REC_TX_DT"),
        ({"REC_TX_DT": "NA"}, "missing_REC_TX_DT"),
        ({"REC_TX_ORG_TY": "KP"}, "not_kidney_only"),
        ({"REC_TX_ORG_TY": ""}, "missing_REC_TX_ORG_TY"),
        ({"DON_TY": "L"}, "not_deceased_donor"),
        ({"DON_TY": ""}, "missing_DON_TY"),
        ({"REC_AGE_AT_TX": "17.99"}, "recipient_age_below_min"),
        ({"DON_AGE": "8.99"}, "donor_age_below_min"),
        ({"DON_AGE": "NA"}, "missing_DON_AGE"),
        ({"shared": "NA"}, "missing_retained_numeric"),
        ({"REC_TX_ORG_TY": "KI: Kidney", "DON_TY": "C: Deceased", "TX_ID": ""}, None),
    ]
    rows = [source_row(PX_ID=f"{i:06}", **change) for i, (change, _) in enumerate(cases, 1)]
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, rows)
    result = prepare.prepare_dataset(source, output, **FILTERS)
    written = read_rows(output)
    expected = [row for row, (_, reason) in zip(rows, cases) if reason is None]
    assert [row["PX_ID"] for row in written] == [row["PX_ID"] for row in expected]
    assert len(written) == 3
    assert all(row["AgmmA2"] == "1" and row["MM_A_9_rF_dY"] == "2" for row in written)
    exclusions = read_rows(result["missing_reports"]["exclusions"])
    by_row = {int(r["row_number"]): r for r in exclusions}
    for i, ((_, reason), row) in enumerate(zip(cases, rows), 1):
        if reason is None:
            assert i not in by_row
        else:
            assert reason in by_row[i]["reasons"].split(";")
            assert by_row[i]["source_line_number"] == str(i + 1)
            assert by_row[i]["PX_ID"] == row["PX_ID"]
    summary = result["missing_summary"]
    assert summary["rows_dropped_for_cohort"] == 11
    assert summary["rows_dropped_for_missing_after_cohort"] == 1
    assert summary["rows_dropped"] == 12
    assert summary["prepared"]["rows"] == 3
    assert summary["cohort_filters"]["censor_date"] == "2025-12-31"
    # Audit keeps ignored date missingness, but it never removes eligible rows.
    assert all(entry["row_dropped"] == str(int(entry["row_number"]) in by_row)
               for entry in read_rows(result["missing_reports"]["cells"]))


def test_censoring_shortens_only_and_preserves_boundary_events(tmp_path):
    days = (date(2025, 12, 31) - date(2020, 1, 1)).days
    rows = [source_row(REC_TX_DT="2020-01-01", graftyrs=str(duration), grf_fail=event)
            for duration, event in [(1, "1"), (1, "0"),
                                    (round(days / 365.24, 9), "1"),
                                    (10, "1"), (10, "0")]]
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, rows)
    result = prepare.prepare_dataset(source, output, **FILTERS)
    actual = read_rows(output)
    assert [r["grf_fail"] for r in actual] == ["1", "0", "1", "0", "0"]
    assert [r["graftyrs"] for r in actual[:3]] == [r["graftyrs"] for r in rows[:3]]
    assert float(actual[3]["graftyrs"]) == pytest.approx(days / 365.24)
    assert actual[3]["graftyrs"] == actual[4]["graftyrs"]
    assert result["missing_summary"]["censoring"]["rows_shortened"] == 2
    assert result["missing_summary"]["censoring"]["events_changed_to_censored"] == 1
    assert read_rows(source) == rows  # Never mutate original data.


def test_censoring_preserves_missing_outcomes_unless_drop_requested(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, [source_row(graftyrs="NA", grf_fail="")])
    result = prepare.prepare_dataset(source, output, censor_date="2025-12-31")
    assert read_rows(output)[0]["graftyrs"] == ""
    assert result["missing_summary"]["censoring"]["rows_not_censored_missing_outcomes"] == 1
    result = prepare.prepare_dataset(source, output, censor_date="2025-12-31",
                                     drop_missing=True, overwrite=True)
    assert result["rows"] == 0
    assert read_rows(output) == []


@pytest.mark.parametrize("tx_date", ["2010-01-01", "01/01/10", "01/01/2010", "2010-01-01 00:00:00"])
def test_source_date_formats(tmp_path, tx_date):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, [source_row(REC_TX_DT=tx_date)])
    assert prepare.prepare_dataset(source, output, **FILTERS)["rows"] == 1
    assert read_rows(output)[0]["REC_TX_DT"] == tx_date


def test_censor_divisor_can_match_other_upstream_units(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, [source_row(REC_TX_DT="2025-01-01", graftyrs="3")])
    prepare.prepare_dataset(source, output, censor_date="2025-12-31", days_per_year=365)
    assert float(read_rows(output)[0]["graftyrs"]) == pytest.approx(364 / 365)


@pytest.mark.parametrize("column", ["REC_TX_DT", "REC_TX_ORG_TY", "DON_TY"])
def test_enabled_filters_require_source_columns(tmp_path, column):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    row = source_row()
    row.pop(column)
    write_source(source, [row])
    with pytest.raises(ValueError, match=column):
        prepare.prepare_dataset(source, output, **FILTERS)
    assert not output.exists()


def test_invalid_date_does_not_replace_existing_outputs(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, [source_row()])
    result = prepare.prepare_dataset(source, output, **FILTERS)
    paths = [output, *map(type(output), result["missing_reports"].values())]
    original = {p: p.read_bytes() for p in paths}
    write_source(source, [source_row(REC_TX_DT="02/30/2020")])
    with pytest.raises(ValueError, match="invalid date"):
        prepare.prepare_dataset(source, output, **FILTERS, overwrite=True)
    assert {p: p.read_bytes() for p in paths} == original


def test_outside_cohort_outcomes_are_not_used(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    write_source(source, [source_row(REC_TX_DT="2008-01-01", graftyrs="-5")])
    result = prepare.prepare_dataset(source, output, **FILTERS)
    assert result["rows"] == 0
    assert result["missing_summary"]["rows_dropped_for_cohort"] == 1


@pytest.mark.parametrize("cluster", ["LSF", "SLURM"])
def test_launcher_passes_cohort_parameters_to_standalone_worker(tmp_path, monkeypatch, cluster):
    source = tmp_path / "Best-Typing_AA_MM_9loc_matrix_1_aligned.csv"
    write_source(source, [source_row(), source_row(REC_TX_DT="2008-01-01"), source_row(shared="NA")])
    work, output = tmp_path / "jobs", tmp_path / "prepared"
    with monkeypatch.context() as patch:
        patch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected submission"))
        launcher.main(["--d", str(source), "--w", str(work), "--o", str(output), "--rc", cluster,
                       "--python-executable", sys.executable, "--dry-run", *FLAGS])
    script = next((work / "scratch").glob("*.sh"))
    subprocess.run(["bash", "-n", str(script)], check=True)
    command = shlex.split(script.read_text().splitlines()[-1])
    if cluster == "SLURM":
        assert command.pop(0) == "srun"
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(read_rows(output / "BestTyping_1.csv")) == 1
    report = json.loads((output / "missing_data_reports/BestTyping_1_missing_summary.json").read_text())
    assert report["cohort_filters"]["donor_age_min"] == 9
    assert report["cohort_filters"]["tx_year_max"] == 2025
    assert report["rows_dropped"] == 2


@pytest.mark.parametrize("kwargs", [dict(tx_year_min=2025, tx_year_max=2010),
    dict(recipient_age_min=float("nan")), dict(donor_age_min=-1), dict(days_per_year=0),
    dict(censor_date="bad"), dict(tx_year_min=2010, censor_date="2009-12-31")])
def test_invalid_filter_configuration_fails_before_reading_source(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare.prepare_dataset(tmp_path / "absent.csv", tmp_path / "output.csv", **kwargs)


def test_launcher_rejects_reversed_window_before_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected submission"))
    with pytest.raises(SystemExit):
        launcher.main(["--d", str(tmp_path), "--w", str(tmp_path / "jobs"),
                       "--o", str(tmp_path / "output"), "--tx-year-min", "2025", "--tx-year-max", "2010"])
