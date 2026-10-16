#!/usr/bin/env python3
"""Keep record metadata, FIBERS covariates, survival outcomes, and AA mismatches.

Accepts a combined best-typing TSV/CSV (optionally .gz) and writes CSV without
an index. By default, keeps position, residue, AND donor-to-recipient mismatch
columns. This is a column projection: rows, feature names, and existing values
are preserved; no shuffling, aggregation, imputation, or rare filtering occurs.

Uses only the Python standard library and streams one row at a time.
"""

import argparse
import csv
import gzip
import math
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import re
import tempfile
from collections import Counter
from pathlib import Path


# Matches the base adjustment set in hla_hpc_protective_scripts/job_fibers_hpc_cv.py.
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
    "TFL_GRAFT_DT", "TFL_DEATH_DT", "TFL_LAFUDATE",
)

# (source antigen mismatch count, indicator for zero, indicator for one).
# Two mismatches is the reference category. DRB345 shares DRB1 adjustment.
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


def get_covariates(cov_loci=DEFAULT_COV_LOCI):
    """Return the exact list to pass to FIBERS(covariates=...)."""
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
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    return opener(path, mode + "t", encoding="utf-8-sig" if mode == "r" else "utf-8",
                  newline="")


def _number(value, column, row_number):
    try:
        number = float(value)
    except (ValueError, TypeError):
        number = float("nan")
    if not math.isfinite(number):
        raise ValueError(
            f"Data row {row_number}, column {column}: missing or non-finite numeric value. "
            "Resolve missing values before preparing the dataset."
        )
    return number


def prepare_dataset(input_path, output_path, *, mismatch_level="all",
                    cov_loci=DEFAULT_COV_LOCI, delimiter=None, overwrite=False):
    """Write the selected dataset; return counts and the FIBERS covariate list.

    Missing antigen indicators are derived only from their corresponding
    REC_*_MM_EQUIV_CUR count. Existing indicators are retained. Missing base
    covariates, outcomes, invalid numeric data, or malformed rows cause an error.
    An unsuccessful conversion leaves any existing output untouched.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace it.")
    if mismatch_level not in ("all", *MISMATCH_PATTERNS):
        raise ValueError(f"Unknown mismatch level: {mismatch_level}")
    cov_loci = tuple(cov_loci)
    covariates = get_covariates(cov_loci)

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

        derivations = {}
        antigen_indicators = set()
        for locus in cov_loci:
            count_column, zero_column, one_column = ANTIGEN_COVARIATES[locus]
            for count, column in enumerate((zero_column, one_column)):
                antigen_indicators.add(column)
                if column not in indices:
                    derivations[column] = (count_column, count)
        required = set(covariates).difference(derivations) | set(OUTCOMES)
        required.update(source_column for source_column, _ in derivations.values())
        missing = sorted(required.difference(indices))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        feature_types = {}
        for column in header:
            level = mismatch_type(column)
            if level and mismatch_level in ("all", level):
                feature_types[column] = level
        if not feature_types:
            raise ValueError(f"No AA mismatch columns found for level '{mismatch_level}'.")
        metadata_columns = [column for column in METADATA_COLUMNS if column in indices]
        output_columns = metadata_columns + covariates + list(OUTCOMES) + list(feature_types)
        # Precompute selectors once; retain mismatch columns in their input order.
        selectors = [(column, indices[column]) for column in output_columns
                     if column not in derivations and column not in metadata_columns]
        metadata_selectors = [(column, indices[column]) for column in metadata_columns]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=".fibers-", suffix=output_path.suffix, delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        row_count = 0
        try:
            with _open_text(temporary_path, "w") as destination:
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(output_columns)
                for row in reader:
                    if not row:
                        continue
                    row_count += 1
                    if len(row) != len(header):
                        raise ValueError(
                            f"Data row {row_count}: expected {len(header)} fields, got {len(row)}."
                        )
                    values = {column: row[index] for column, index in metadata_selectors}
                    for column, index in selectors:
                        value = row[index]
                        number = _number(value, column, row_count)
                        if column in antigen_indicators or column == "grf_fail":
                            if number not in (0, 1):
                                raise ValueError(f"Data row {row_count}: {column} must be 0 or 1.")
                        if column == "graftyrs" and number < 0:
                            raise ValueError(f"Data row {row_count}: graftyrs must be nonnegative.")
                        if column in feature_types and (number < 0 or not number.is_integer()):
                            raise ValueError(
                                f"Data row {row_count}: {column} must be a nonnegative integer count."
                            )
                        values[column] = value
                    for column, (source_column, count) in derivations.items():
                        number = _number(row[indices[source_column]], source_column, row_count)
                        if number not in (0, 1, 2):
                            raise ValueError(
                                f"Data row {row_count}: {source_column} must be 0, 1, or 2 "
                                f"to derive {column}."
                            )
                        values[column] = int(number == count)
                    writer.writerow([values[column] for column in output_columns])
            if row_count == 0:
                raise ValueError("Input contains a header but no data rows.")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    return {
        "rows": row_count,
        "columns": len(output_columns),
        "metadata_columns": metadata_columns,
        "covariates": covariates,
        "derived_covariates": list(derivations),
        "mismatch_counts": dict(Counter(feature_types.values())),
    }


def add_preparation_arguments(parser):
    """Keep preparation options consistent between the run and job scripts."""
    parser.add_argument("--mismatch-level", choices=("all", *MISMATCH_PATTERNS), default="all",
                        help="Mismatch features to keep (default: all three types)")
    parser.add_argument("--cov-list", "--cov-loci", dest="cov_loci", default=",".join(DEFAULT_COV_LOCI),
                        help="Antigen adjustment loci (default: A,B,C,DRB1,DQA1,DQB1); "
                             "use 'none' for the 43 base covariates only")
    parser.add_argument("--delimiter", choices=("tab", "comma"),
                        help="Input delimiter (default: detect tab or comma from header)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", "--input", dest="input", type=Path, required=True,
                        help="Combined best-typing TSV/CSV, optionally .gz")
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
        )
    except (ValueError, OSError, csv.Error, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {result['rows']:,} rows x {result['columns']:,} columns to {args.output}")
    print(f"Covariates: {len(result['covariates'])}; outcomes: {', '.join(OUTCOMES)}")
    print("AA mismatches: " + "; ".join(
        f"{level}={result['mismatch_counts'].get(level, 0):,}" for level in MISMATCH_PATTERNS
    ))
    if result["derived_covariates"]:
        print("Derived antigen indicators: " + ", ".join(result["derived_covariates"]))


if __name__ == "__main__":
    main()
