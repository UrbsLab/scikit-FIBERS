#!/usr/bin/env python3
"""Keep record metadata, FIBERS covariates, survival outcomes, and AA mismatches.

Accepts combined best-typing TXT/TSV/CSV or gzip input and writes CSV without
an index. By default, keeps position, residue, AND donor-to-recipient mismatch
columns. By default, rows, feature names, and existing values are preserved.
--drop-missing optionally removes rows with missing retained numeric fields;
no shuffling, aggregation, imputation, or rare filtering occurs.

Uses only the Python standard library and streams one row at a time.
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
# These identify source antigen fields; preparation never encodes them.
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
_MM_PREFIX = r"MM_(?:A|B|C|DRB1|DRB345|DQA1|DQB1|DPA1|DPB1)_[1-9][0-9]*"
MISMATCH_PATTERNS = {
    "position": re.compile(_MM_PREFIX),
    "residue": re.compile(_MM_PREFIX + r"_[A-Z]"),
    "permutation": re.compile(_MM_PREFIX + r"_r[A-Z]_d[A-Z]"),
}
# Recognize common CSV/pandas and SAS missing markers without rewriting them.
# Matching ignores surrounding whitespace and letter case; zero is not missing.
MISSING_VALUES = frozenset({
    "", "na", "n/a", "nan", "-nan", "null", "none", "<na>", "nat", ".", r"\n",
    "#na", "#n/a", "#n/a n/a", "-1.#ind", "-1.#qnan", "1.#ind", "1.#qnan", "998",
})


def is_missing(value):
    """Recognize text markers and numeric representations of the 998 sentinel."""
    normalized = value.strip().lower()
    if normalized in MISSING_VALUES:
        return True
    try:
        return float(normalized) == 998
    except ValueError:
        return False


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
        number = float(value)
    except (ValueError, TypeError):
        number = float("nan")
    if not math.isfinite(number):
        raise ValueError(
            f"Data row {row_number}, column {column}: invalid or non-finite numeric value."
        )
    return number


def prepare_dataset(input_path, output_path, *, mismatch_level="all",
                    cov_loci=DEFAULT_COV_LOCI, delimiter=None, overwrite=False,
                    drop_missing=False):
    """Write the selected dataset; return counts and the FIBERS covariate list.

    Existing Agmm columns and raw antigen mismatch counts are copied unchanged.
    Missing cells are preserved and reported for every input column.
    With drop_missing, rows missing retained numeric fields are reported but omitted.
    Absent required columns, invalid nonmissing numeric data, or malformed rows cause an error.
    An unsuccessful conversion leaves any existing output untouched.
    """
    input_path, output_path = Path(input_path), Path(output_path)
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
        missing = sorted(required.difference(indices))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        covariates = [column for column in requested_covariates if column in indices]
        antigen_counts = {columns[0] for columns in ANTIGEN_COVARIATES.values()}
        antigen_columns = [column for column in header
                           if column.lower().startswith('agmm') or column in antigen_counts]

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        missing_counts = Counter()
        prepared_missing_counts = Counter()
        rows_with_missing = 0
        prepared_rows_with_missing = 0
        written_rows = 0
        dropped_rows = 0
        with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".fibers-") as staging:
            temporary_path = Path(staging) / output_path.name
            temporary_reports = {key: Path(staging) / path.name for key, path in report_paths.items()}
            with _open_text(temporary_path, "w") as destination, temporary_reports["cells"].open(
                    "w", encoding="utf-8", newline="") as details:
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(output_columns)
                detail_writer = csv.writer(details, lineterminator="\n")
                detail_writer.writerow(["row_number", "source_line_number", "TX_ID", "column",
                                        "missing_value", "retained_in_output", "row_dropped"])
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
                    row_dropped = drop_missing and any(header[index] in numeric_columns
                                                       for index in missing_indices)
                    tx_id = row[tx_id_index] if tx_id_index is not None else ""
                    for index in sorted(missing_indices):
                        missing_counts[header[index]] += 1
                        detail_writer.writerow([row_count, source_line_number, tx_id,
                                                header[index], row[index], retained[index], row_dropped])
                    rows_with_missing += bool(missing_indices)
                    for column, index in selectors:
                        if column not in numeric_columns or index in missing_indices:
                            continue
                        value = row[index]
                        number = _number(value, column, row_count)
                        if column == "grf_fail":
                            if number not in (0, 1):
                                raise ValueError(f"Data row {row_count}: {column} must be 0 or 1.")
                        if column in antigen_counts and number not in (0, 1, 2):
                            raise ValueError(f"Data row {row_count}: {column} must be 0, 1, or 2.")
                        if column == "graftyrs" and number < 0:
                            raise ValueError(f"Data row {row_count}: graftyrs must be nonnegative.")
                        if column in feature_types and (number < 0 or not number.is_integer()):
                            raise ValueError(
                                f"Data row {row_count}: {column} must be a nonnegative integer count."
                            )
                    if row_dropped:
                        dropped_rows += 1
                        continue
                    prepared_rows_with_missing += any(retained[index] for index in missing_indices)
                    prepared_missing_counts.update(header[index] for index in missing_indices if retained[index])
                    writer.writerow([row[index] for _, index in selectors])
                    written_rows += 1
            if row_count == 0:
                raise ValueError("Input contains a header but no data rows.")
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
                "marker_matching": "case-insensitive after stripping whitespace; numeric 998 equivalents included; values preserved",
                "drop_missing": drop_missing,
                "drop_missing_scope": "retained numeric columns; metadata and excluded columns do not cause drops",
                "rows_dropped": dropped_rows,
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
                             "'none' reports base covariates only. All existing Agmm fields are retained.")
    parser.add_argument("--delimiter", choices=("tab", "comma"),
                        help="Input delimiter (default: detect tab or comma from header)")
    parser.add_argument("--overwrite", action="store_true", help="Replace the dataset and its missing-data reports")
    parser.add_argument("--drop-missing", nargs="?", const="true", default="false", type=str.lower,
                        choices=("true", "false"),
                        help="Drop rows missing retained numeric fields, including 998; metadata does not cause drops (default: false)")


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
        )
    except (ValueError, OSError, csv.Error, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {result['rows']:,} rows x {result['columns']:,} columns to {args.output}")
    print(f"Rows dropped for missing numeric fields: {result['rows_dropped']:,}")
    print(f"Covariates: {len(result['covariates'])}; outcomes: {', '.join(OUTCOMES)}")
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
