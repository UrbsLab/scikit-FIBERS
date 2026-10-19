#!/usr/bin/env python3
"""Keep record metadata, FIBERS covariates, survival outcomes, and AA mismatches.

Accepts combined best-typing TXT/TSV/CSV or gzip input and writes CSV without
an index. By default, keeps position, residue, AND donor-to-recipient mismatch
columns. Numeric missing markers are written as empty cells; other values and
feature names are preserved unless filtering or censoring is requested.
--drop-missing optionally removes rows with missing retained numeric fields;
optional cohort parameters filter source dates, organ/donor type, and ages.
--censor-date caps existing survival follow-up without extending it.
No shuffling, aggregation, imputation, or rare filtering occurs.

Uses only the Python standard library. Numeric validation and reports stream
one row at a time without loading the input or output as a DataFrame.
"""

import argparse
import csv
import gzip
import json
import math
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

# HLA base adjustment set, including shared.
BASE_COVARIATES = (
    "shared", "DCD", "DON_AGE", "donage_slope_ge18", "dcadcodanox",
    "dcadcodcva", "dcadcodcnst", "dcadcodoth", "don_cmv_negative",
    "don_htn_0c", "ln_don_wgt_kg_0c", "ln_don_wgt_kg_0c_s55", "don_ecd",
    "age_ecd", "yearslice", "REC_AGE_AT_TX", "rec_age_spline_35",
    "rec_age_spline_50", "rec_age_spline_65", "diab_noted", "age_diab",
    "dm_can_age_spline_50", "can_dgn_htn_ndm", "can_dgn_pk_ndm",
    "can_dgn_gd_ndm", "rec_prev_ki_tx", "rec_prev_ki_tx_dm", "rbmi_0c",
    "rbmi_miss", "rbmi_gt_20", "rbmi_DM", "rbmi_gt_20_DM", "ln_c_hd_m",
    "ln_c_hd_0c", "ln_c_hd_m_ptx", "PKPRA_MS", "PKPRA_1080", "PKPRA_GE80",
    "hispanic", "CAN_RACE_BLACK", "CAN_RACE_asian", "CAN_RACE_WHITE", "Agmm0",
)
DEFAULT_COV_LOCI = ("A", "B", "C", "DRB1", "DQA1", "DQB1")
OUTCOMES = ("graftyrs", "grf_fail")
# Optional record-linkage and outcome dates, preserved as source text.
METADATA_COLUMNS = (
    "PX_ID", "PERS_ID", "TX_ID", "DONOR_ID", "REC_TX_DT",
    "TFL_GRAFT_DT", "TFL_DEATH_DT", "TFL_LAFUDATE", "DON_TY",
)

# (source antigen mismatch count, indicator for zero, indicator for one).
# Raw source fields are excluded; preparation keeps existing processed indicators.
ANTIGEN_COVARIATES = {
    "A": ("REC_A_MM_EQUIV_CUR", "AgmmA0", "AgmmA1"),
    "B": ("REC_B_MM_EQUIV_CUR", "AgmmB0", "AgmmB1"),
    "C": ("REC_C_MM_EQUIV_CUR", "Agmmc0", "Agmmc1"),
    "DRB1": ("REC_DR_MM_EQUIV_CUR", "AgmmDR0", "AgmmDR1"),
    "DRB345": ("REC_DR_MM_EQUIV_CUR", "AgmmDR0", "AgmmDR1"),
    "DQA1": ("REC_DQA1_MM_EQUIV_CUR", "Agmmdqa10", "Agmmdqa11"),
    "DQB1": ("REC_DQ_MM_EQUIV_CUR", "Agmmdqb10", "Agmmdqb11"),
    "DPA1": ("REC_DPA1_MM_EQUIV_CUR", "Agmmdpa10", "Agmmdpa11"),
    "DPB1": ("REC_DPB1_MM_EQUIV_CUR", "Agmmdpb10", "Agmmdpb11"),
}
PROCESSED_ANTIGEN_PATTERN = re.compile(
    r"Agmm(?:A|B|C|DR|DRB1|DRB345|DQA1|DQ|DQB1|DPA1|DPB1)?[012]", re.IGNORECASE
)
_MM_PREFIX = r"MM_(?:A|B|C|DRB1|DRB345|DQA1|DQB1|DPA1|DPB1)_[1-9][0-9]*"
MISMATCH_PATTERNS = {
    "position": re.compile(_MM_PREFIX),
    "residue": re.compile(_MM_PREFIX + r"_[A-Z]"),
    "permutation": re.compile(_MM_PREFIX + r"_r[A-Z]_d[A-Z]"),
}
# Recognize common CSV/pandas and SAS missing markers; retain originals in reports.
# Matching ignores surrounding whitespace and letter case; zero is not missing.
MISSING_VALUES = frozenset({
    "", "na", "n/a", "nan", "-nan", "null", "none", "<na>", "nat", ".", r"\n",
    "#na", "#n/a", "#n/a n/a", "-1.#ind", "-1.#qnan", "1.#ind", "1.#qnan",
})
# Accept decimal/scientific notation that numeric CSV readers recognize. Python
# float alone also accepts underscores and Unicode digits, which can load as text.
NUMERIC_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def is_missing(value):
    """Recognize text markers, ignoring surrounding whitespace and letter case."""
    return value.strip().lower() in MISSING_VALUES


def get_covariates(cov_loci=DEFAULT_COV_LOCI):
    """Return candidate model covariates; workers use available antigen indicators."""
    covariates = list(BASE_COVARIATES)
    for locus in cov_loci:
        if locus not in ANTIGEN_COVARIATES:
            raise ValueError(f"Unknown covariate locus: {locus}")
        covariates.extend(ANTIGEN_COVARIATES[locus][1:])
    return list(dict.fromkeys(covariates))


def mismatch_type(column):
    """Recognize only AA mismatch features, excluding MM_*_COUNT totals."""
    for level, pattern in MISMATCH_PATTERNS.items():
        if pattern.fullmatch(column):
            return level
    return None


def _open_text(path, mode):
    if mode == "r":
        # Detect gzip by its contents, including extensionless aligned files.
        with path.open("rb") as probe:
            compressed = probe.read(2) == b"\x1f\x8b"
    else:
        compressed = path.suffix.lower() == ".gz"
    opener = gzip.open if compressed else open
    return opener(path, mode + "t", encoding="utf-8-sig" if mode == "r" else "utf-8",
                  newline="")


def _number(value, column, row_number):
    try:
        # The common mismatch values need no regex; keep this per-cell check cheap.
        if value not in ("0", "1", "2", "0.0", "1.0", "2.0") and not NUMERIC_PATTERN.fullmatch(
                value.strip(" \t\r\n\v\f")):
            raise ValueError("Not decimal or scientific notation")
        number = float(value)
    except (ValueError, TypeError):
        number = float("nan")
    if not math.isfinite(number):
        raise ValueError(
            f"Data row {row_number}, column {column}: invalid or non-finite numeric value."
        )
    return number


def _date(value, column, row_number):
    """Parse ISO dates/timestamps and US dates, including the supplied MM/DD/YY."""
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for pattern in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"Data row {row_number}, column {column}: invalid date; "
                     "expected ISO YYYY-MM-DD or US MM/DD/YYYY (MM/DD/YY also accepted).")


def _filter_settings(tx_year_min, tx_year_max, kidney_only, deceased_donor_only,
                     recipient_age_min, donor_age_min, censor_date, days_per_year):
    for name, value in (("tx_year_min", tx_year_min), ("tx_year_max", tx_year_max)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or not 1 <= value <= 9999):
            raise ValueError(f"{name} must be an integer year from 1 to 9999.")
    if tx_year_min is not None and tx_year_max is not None and tx_year_min > tx_year_max:
        raise ValueError("tx_year_min must not exceed tx_year_max.")
    for name, value in (("recipient_age_min", recipient_age_min), ("donor_age_min", donor_age_min)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not math.isfinite(days_per_year) or days_per_year <= 0:
        raise ValueError("days_per_year must be finite and positive.")
    cutoff = _date(str(censor_date), "censor_date", 0) if censor_date is not None else None
    if cutoff is not None and tx_year_min is not None and cutoff.year < tx_year_min:
        raise ValueError("censor_date precedes the entire requested transplant-year window.")
    return {
        "tx_year_min": tx_year_min, "tx_year_max": tx_year_max,
        "kidney_only": kidney_only, "deceased_donor_only": deceased_donor_only,
        "recipient_age_min": recipient_age_min, "donor_age_min": donor_age_min,
        "censor_date": cutoff.isoformat() if cutoff else None, "days_per_year": days_per_year,
    }, cutoff


def _cohort_reasons(row, indices, row_number, settings, cutoff):
    """Evaluate source variables before selecting columns; never use STROBE flags."""
    reasons = []
    transplant_date = None
    if any(settings[key] is not None for key in ("tx_year_min", "tx_year_max", "censor_date")):
        value = row[indices["REC_TX_DT"]]
        if is_missing(value):
            reasons.append("missing_REC_TX_DT")
        else:
            transplant_date = _date(value, "REC_TX_DT", row_number)
            if settings["tx_year_min"] is not None and transplant_date.year < settings["tx_year_min"]:
                reasons.append("transplant_before_min_year")
            if settings["tx_year_max"] is not None and transplant_date.year > settings["tx_year_max"]:
                reasons.append("transplant_after_max_year")
            if cutoff is not None and transplant_date > cutoff:
                reasons.append("transplant_after_censor_date")
    for enabled, column, accepted, reason in (
        (settings["kidney_only"], "REC_TX_ORG_TY", {"KI", "KIDNEY", "KIDNEY ONLY", "KIDNEY-ONLY"}, "not_kidney_only"),
        (settings["deceased_donor_only"], "DON_TY", {"C", "CADAVERIC", "DECEASED", "DECEASED DONOR"}, "not_deceased_donor"),
    ):
        if enabled:
            value = row[indices[column]]
            if is_missing(value):
                reasons.append("missing_" + column)
            elif value.split(":", 1)[0].strip().upper() not in accepted:
                reasons.append(reason)
    for key, column, reason in (
        ("recipient_age_min", "REC_AGE_AT_TX", "recipient_age_below_min"),
        ("donor_age_min", "DON_AGE", "donor_age_below_min"),
    ):
        if settings[key] is not None:
            value = row[indices[column]]
            if is_missing(value):
                reasons.append("missing_" + column)
            elif _number(value, column, row_number) < settings[key]:
                reasons.append(reason)
    return reasons, transplant_date


def prepare_dataset(input_path, output_path, *, mismatch_level="all",
                    cov_loci=DEFAULT_COV_LOCI, delimiter=None, overwrite=False,
                    drop_missing=False, tx_year_min=None, tx_year_max=None,
                    kidney_only=False, deceased_donor_only=False, recipient_age_min=None,
                    donor_age_min=None, censor_date=None, days_per_year=365.24):
    """Write the selected dataset; return counts and the FIBERS covariate list.

    Keep processed numeric base/Agmm covariates; exclude raw antigen fields.
    Numeric missing markers become empty cells and are reported in their original form.
    With drop_missing, rows missing retained numeric fields are reported but omitted.
    Enabled eligibility filters exclude unknown/missing criteria as well as failures.
    Censoring only shortens existing follow-up; the original event definition is retained.
    Absent required columns, malformed rows, or invalid numeric data in eligible rows cause an error.
    An unsuccessful conversion leaves any existing output untouched.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    settings, cutoff = _filter_settings(tx_year_min, tx_year_max, kidney_only,
                                       deceased_donor_only, recipient_age_min, donor_age_min,
                                       censor_date, days_per_year)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    report_stem = output_path.name[:-3] if output_path.suffix.lower() == ".gz" else output_path.name
    if report_stem.lower().endswith(".csv"):
        report_stem = report_stem[:-4]
    report_dir = output_path.parent / "missing_data_reports"
    report_paths = {
        "cells": report_dir / f"{report_stem}_missing_cells.csv",
        "columns": report_dir / f"{report_stem}_missing_columns.csv",
        "summary": report_dir / f"{report_stem}_missing_summary.json",
        "exclusions": report_dir / f"{report_stem}_row_exclusions.csv",
    }
    for destination in (output_path, *report_paths.values()):
        if destination.resolve() == input_path.resolve():
            raise ValueError("Outputs and reports must not replace the input file.")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}; use --overwrite to replace it.")
    if mismatch_level not in ("all", *MISMATCH_PATTERNS):
        raise ValueError(f"Unknown mismatch level: {mismatch_level}")
    cov_loci = tuple(cov_loci)
    requested_covariates = get_covariates(cov_loci)

    with _open_text(input_path, "r") as source:
        first_line = source.readline()
        if not first_line:
            raise ValueError("Input is empty.")
        if delimiter is None:
            delimiter = "\t" if "\t" in first_line else ","
        source.seek(0)
        reader = csv.reader(source, delimiter=delimiter)
        header = next(reader)
        duplicates = [name for name, count in Counter(header).items() if count > 1]
        if duplicates or any(not name.strip() for name in header):
            raise ValueError(f"Input has empty or duplicate column names: {duplicates}")
        indices = {column: i for i, column in enumerate(header)}

        required = set(BASE_COVARIATES) | set(OUTCOMES)
        if tx_year_min is not None or tx_year_max is not None or cutoff is not None:
            required.add("REC_TX_DT")
        if kidney_only:
            required.add("REC_TX_ORG_TY")
        if deceased_donor_only:
            required.add("DON_TY")
        missing = sorted(required.difference(indices))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        covariates = [column for column in requested_covariates if column in indices]
        antigen_columns = [column for column in header
                           if PROCESSED_ANTIGEN_PATTERN.fullmatch(column)]

        feature_types = {}
        for column in header:
            level = mismatch_type(column)
            if level and mismatch_level in ("all", level):
                feature_types[column] = level
        if not feature_types:
            raise ValueError(f"No AA mismatch columns found for level '{mismatch_level}'.")
        metadata_columns = [column for column in METADATA_COLUMNS if column in indices]
        output_columns = list(dict.fromkeys(
            metadata_columns + covariates + antigen_columns + list(OUTCOMES) + list(feature_types)
        ))
        # Precompute selectors once; retain mismatch columns in their input order.
        selectors = [(column, indices[column]) for column in output_columns]
        numeric_columns = set(output_columns) - set(metadata_columns)
        retained = [column in numeric_columns or column in metadata_columns for column in header]
        tx_id_index = indices.get("TX_ID")
        px_id_index = indices.get("PX_ID")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        missing_counts = Counter()
        prepared_missing_counts = Counter()
        rows_with_missing = 0
        prepared_rows_with_missing = 0
        written_rows = 0
        dropped_rows = 0
        cohort_dropped_rows = 0
        missing_dropped_rows = 0
        reason_counts = Counter()
        censor_counts = {"rows_shortened": 0, "events_changed_to_censored": 0,
                         "rows_not_censored_missing_outcomes": 0}
        with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".fibers-") as staging:
            temporary_path = Path(staging) / output_path.name
            temporary_reports = {key: Path(staging) / path.name for key, path in report_paths.items()}
            with _open_text(temporary_path, "w") as destination, temporary_reports["cells"].open(
                    "w", encoding="utf-8", newline="") as details, temporary_reports["exclusions"].open(
                    "w", encoding="utf-8", newline="") as exclusions:
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(output_columns)
                detail_writer = csv.writer(details, lineterminator="\n")
                detail_writer.writerow(["row_number", "source_line_number", "TX_ID", "column",
                                        "missing_value", "retained_in_output", "row_dropped"])
                exclusion_writer = csv.writer(exclusions, lineterminator="\n")
                exclusion_writer.writerow(["row_number", "source_line_number", "PX_ID", "TX_ID", "reasons"])
                previous_line_number = reader.line_num
                for row in reader:
                    source_line_number = previous_line_number + 1
                    previous_line_number = reader.line_num
                    if not row:
                        continue
                    row_count += 1
                    if len(row) != len(header):
                        raise ValueError(
                            f"Data row {row_count}: expected {len(header)} fields, got {len(row)}."
                        )
                    missing_indices = {index for index, value in enumerate(row) if is_missing(value)}
                    reasons, transplant_date = _cohort_reasons(row, indices, row_count, settings, cutoff)
                    cohort_failed = bool(reasons)
                    numeric_missing = any(header[index] in numeric_columns for index in missing_indices)
                    if drop_missing and numeric_missing:
                        reasons.append("missing_retained_numeric")
                    row_dropped = bool(reasons)
                    tx_id = row[tx_id_index] if tx_id_index is not None else ""
                    for index in sorted(missing_indices):
                        missing_counts[header[index]] += 1
                        detail_writer.writerow([row_count, source_line_number, tx_id,
                                                header[index], row[index], retained[index], row_dropped])
                    rows_with_missing += bool(missing_indices)
                    if cohort_failed:
                        cohort_dropped_rows += 1
                    elif row_dropped:
                        missing_dropped_rows += 1
                    if row_dropped:
                        reason_counts.update(reasons)
                        exclusion_writer.writerow([row_count, source_line_number,
                                                   row[px_id_index] if px_id_index is not None else "",
                                                   tx_id, ";".join(reasons)])
                    # Rows outside the cohort cannot enter fitting. Do not interpret their outcomes.
                    if cohort_failed:
                        dropped_rows += 1
                        continue
                    for column, index in selectors:
                        if column not in numeric_columns or index in missing_indices:
                            continue
                        value = row[index]
                        number = _number(value, column, row_count)
                        if column == "grf_fail":
                            if number not in (0, 1):
                                raise ValueError(f"Data row {row_count}: {column} must be 0 or 1.")
                        if column == "graftyrs" and number < 0:
                            raise ValueError(f"Data row {row_count}: graftyrs must be nonnegative.")
                        if column in feature_types and (number < 0 or not number.is_integer()):
                            raise ValueError(
                                f"Data row {row_count}: {column} must be a nonnegative integer count."
                            )
                    if row_dropped:
                        dropped_rows += 1
                        continue
                    if cutoff is not None:
                        duration_index, event_index = indices["graftyrs"], indices["grf_fail"]
                        if duration_index in missing_indices or event_index in missing_indices:
                            censor_counts["rows_not_censored_missing_outcomes"] += 1
                        else:
                            observed_days = float(row[duration_index]) * days_per_year
                            available_days = (cutoff - transplant_date).days
                            # Source graftyrs is rounded. Preserve events exactly on the cutoff.
                            if observed_days > available_days and not math.isclose(
                                    observed_days, available_days, rel_tol=0, abs_tol=1e-5):
                                censor_counts["rows_shortened"] += 1
                                censor_counts["events_changed_to_censored"] += int(float(row[event_index]) == 1)
                                row[duration_index] = format(available_days / days_per_year, ".15g")
                                row[event_index] = "0"
                    prepared_rows_with_missing += any(retained[index] for index in missing_indices)
                    prepared_missing_counts.update(header[index] for index in missing_indices if retained[index])
                    # Blank numeric missing cells let pandas infer numeric dtypes even
                    # when the source used SAS/text markers such as '.' or '\\N'.
                    writer.writerow(["" if column in numeric_columns and index in missing_indices
                                     else row[index] for column, index in selectors])
                    written_rows += 1
            if row_count == 0:
                raise ValueError("Input contains a header but no data rows.")
            # Every written numeric cell was validated above or normalized to a
            # blank missing value. Do not reread this wide matrix into a DataFrame.
            numeric_validation = {"passed": True, "rows_checked": written_rows,
                                  "columns_checked": len(numeric_columns),
                                  "method": "streaming numeric validation during conversion"}
            with temporary_reports["columns"].open("w", encoding="utf-8", newline="") as handle:
                summary_writer = csv.writer(handle, lineterminator="\n")
                summary_writer.writerow(["column", "retained_in_output", "total_rows",
                                         "missing_count", "missing_percent"])
                for index, column in enumerate(header):
                    count = missing_counts[column]
                    summary_writer.writerow([column, retained[index], row_count, count,
                                             f"{100 * count / row_count:.6f}"])
            missing_summary = {
                "input_file": str(input_path.resolve()),
                "output_file": str(output_path.resolve()),
                "row_number_definition": "1-based data record, excluding header and blank lines",
                "source_line_number_definition": "1-based starting line in the uncompressed source file",
                "TX_ID_present": tx_id_index is not None,
                "missing_markers": sorted(MISSING_VALUES),
                "marker_matching": "case-insensitive after stripping whitespace; numeric missing cells written blank; originals preserved in reports",
                "numeric_validation": numeric_validation,
                "drop_missing": drop_missing,
                "drop_missing_scope": "retained numeric columns; metadata and excluded columns do not cause drops",
                "rows_dropped": dropped_rows,
                "rows_dropped_for_cohort": cohort_dropped_rows,
                "rows_dropped_for_missing_after_cohort": missing_dropped_rows,
                "cohort_filters": settings,
                "exclusion_reason_counts": dict(reason_counts),
                "exclusion_count_definition": "reason counts may overlap; cohort and subsequent missing-row totals are disjoint",
                "censoring": {"method": "cap existing graftyrs; never extend follow-up or redefine earlier events",
                              **censor_counts},
                "source": {
                    "rows": row_count, "columns": len(header),
                    "missing_cells": sum(missing_counts.values()),
                    "rows_with_missing": rows_with_missing,
                    "columns_with_missing": len(missing_counts),
                },
                "prepared": {
                    "rows": written_rows, "columns": len(output_columns),
                    "missing_cells": sum(prepared_missing_counts.values()),
                    "rows_with_missing": prepared_rows_with_missing,
                    "columns_with_missing": len(prepared_missing_counts),
                },
            }
            with temporary_reports["summary"].open("w", encoding="utf-8") as handle:
                json.dump(missing_summary, handle, indent=2)
                handle.write("\n")
            # Finish validation and report generation before replacing existing outputs.
            report_dir.mkdir(parents=True, exist_ok=True)
            for key, path in report_paths.items():
                os.replace(temporary_reports[key], path)
            os.replace(temporary_path, output_path)

    return {
        "rows": written_rows,
        "rows_dropped": dropped_rows,
        "columns": len(output_columns),
        "metadata_columns": metadata_columns,
        "covariates": covariates,
        "antigen_columns": antigen_columns,
        "mismatch_counts": dict(Counter(feature_types.values())),
        "missing_summary": missing_summary,
        "missing_reports": {key: str(path) for key, path in report_paths.items()},
    }


def add_preparation_arguments(parser):
    """Keep preparation options consistent between the run and job scripts."""
    parser.add_argument("--mismatch-level", choices=("all", *MISMATCH_PATTERNS), default="all",
                        help="Mismatch features to keep (default: all three types)")
    parser.add_argument("--cov-list", "--cov-loci", dest="cov_loci", default=",".join(DEFAULT_COV_LOCI),
                        help="Antigen loci for reporting available model covariates (default: A,B,C,DRB1,DQA1,DQB1); "
                             "'none' reports base covariates only. Existing Agmm 0/1/2 indicators are retained; raw antigen fields are excluded.")
    parser.add_argument("--delimiter", choices=("tab", "comma"),
                        help="Input delimiter (default: detect tab or comma from header)")
    parser.add_argument("--overwrite", action="store_true", help="Replace the dataset and its missing-data reports")
    parser.add_argument("--drop-missing", nargs="?", const="true", default="false", type=str.lower,
                        choices=("true", "false"),
                        help="Drop rows missing retained numeric fields (default: false); cohort eligibility is checked separately")
    parser.add_argument("--tx-year-min", type=int, help="Inclusive minimum transplant year from REC_TX_DT")
    parser.add_argument("--tx-year-max", type=int, help="Inclusive maximum transplant year from REC_TX_DT")
    parser.add_argument("--kidney-only", nargs="?", const="true", default="false", type=str.lower,
                        choices=("true", "false"), help="Require REC_TX_ORG_TY=KI (default: false)")
    parser.add_argument("--deceased-donor-only", nargs="?", const="true", default="false", type=str.lower,
                        choices=("true", "false"), help="Require DON_TY=C (default: false)")
    parser.add_argument("--recipient-age-min", type=float, help="Inclusive minimum REC_AGE_AT_TX in years")
    parser.add_argument("--donor-age-min", type=float, help="Inclusive minimum DON_AGE in years")
    parser.add_argument("--censor-date", help="Administrative cutoff, e.g. 2025-12-31; caps existing outcomes")
    parser.add_argument("--days-per-year", type=float, default=365.24,
                        help="Divisor used by source graftyrs (default: 365.24, verified in SampleImp1)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", "--input", dest="input", type=Path, required=True,
                        help="Combined best-typing TXT/TSV/CSV or gzip input; tab/comma auto-detected")
    parser.add_argument("--o", "-o", "--output", dest="output", type=Path, required=True,
                        help="Output CSV or CSV.gz, e.g. BestTyping_1.csv")
    add_preparation_arguments(parser)
    args = parser.parse_args(argv)
    cov_loci = [] if args.cov_loci.strip().lower() == "none" else [
        locus.strip() for locus in args.cov_loci.split(",") if locus.strip()
    ]
    try:
        result = prepare_dataset(
            args.input, args.output, mismatch_level=args.mismatch_level, cov_loci=cov_loci,
            delimiter={"tab": "\t", "comma": ","}.get(args.delimiter), overwrite=args.overwrite,
            drop_missing=args.drop_missing == "true",
            tx_year_min=args.tx_year_min, tx_year_max=args.tx_year_max,
            kidney_only=args.kidney_only == "true", deceased_donor_only=args.deceased_donor_only == "true",
            recipient_age_min=args.recipient_age_min, donor_age_min=args.donor_age_min,
            censor_date=args.censor_date, days_per_year=args.days_per_year,
        )
    except (ValueError, OSError, csv.Error, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {result['rows']:,} rows x {result['columns']:,} columns to {args.output}")
    summary = result["missing_summary"]
    print(f"Rows dropped: {result['rows_dropped']:,} "
          f"(cohort: {summary['rows_dropped_for_cohort']:,}; "
          f"missing numeric fields after cohort: {summary['rows_dropped_for_missing_after_cohort']:,})")
    if args.censor_date:
        print(f"Follow-up shortened at censor date: {summary['censoring']['rows_shortened']:,}")
    print(f"Covariates: {len(result['covariates'])}; outcomes: {', '.join(OUTCOMES)}")
    validation = summary["numeric_validation"]
    print(f"Streaming numeric check passed: {validation['rows_checked']:,} rows x "
          f"{validation['columns_checked']:,} model columns")
    print("AA mismatches: " + "; ".join(
        f"{level}={result['mismatch_counts'].get(level, 0):,}" for level in MISMATCH_PATTERNS
    ))
    for scope in ("source", "prepared"):
        summary = result["missing_summary"][scope]
        print(f"Missing data ({scope}): {summary['missing_cells']:,} cells in "
              f"{summary['rows_with_missing']:,} rows and {summary['columns_with_missing']:,} columns")
    for kind, path in result["missing_reports"].items():
        print(f"Missing-data {kind} report: {path}")


if __name__ == "__main__":
    main()
