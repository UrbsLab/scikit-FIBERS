"""Submit combined-typing preparation jobs and name outputs <prefix>_<imputation>.csv."""

import argparse
import math
import shlex
import subprocess
import re
import sys
import time
from pathlib import Path
from datetime import datetime

DEFAULT_COV_LOCI = ('A', 'B', 'C', 'DRB1', 'DQA1', 'DQB1')
MISMATCH_PATTERNS = ('position', 'residue', 'permutation')
IMPUTATION_PATTERN = re.compile(r"([0-9]+)(?:_aligned)?$", re.IGNORECASE)


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


def write_job(command, job_name, writepath, cluster, memory, queue, dry_run=False):
    """Write an inspectable shell script, then submit unless --dry-run was given."""
    if any("\n" in value or "\r" in value for value in (job_name, queue, str(writepath))):
        raise ValueError("Job names, paths, and queues cannot contain newlines.")
    for folder in ("scratch", "logs"):
        (writepath / folder).mkdir(parents=True, exist_ok=True)
    job_path = writepath / "scratch" / f"{job_name}_run.sh"
    log_path = writepath / "logs" / job_name
    lines = ["#!/bin/bash"]
    if cluster == "LSF":
        lines.extend([
            f"#BSUB -q {shlex.quote(queue)}", f"#BSUB -J {shlex.quote(job_name)}",
            f'#BSUB -R "rusage[mem={memory}G]"', f"#BSUB -M {memory}GB",
            f"#BSUB -o {shlex.quote(str(log_path) + '.o')}",
            f"#BSUB -e {shlex.quote(str(log_path) + '.e')}",
        ])
    else:
        lines.extend([
            f"#SBATCH -p {shlex.quote(queue)}", f"#SBATCH --job-name={shlex.quote(job_name)}",
            f"#SBATCH --mem={memory}G",
            f"#SBATCH -o {shlex.quote(str(log_path) + '.o')}",
            f"#SBATCH -e {shlex.quote(str(log_path) + '.e')}",
        ])
        command = ["srun"] + command
    lines.extend(["set -euo pipefail", "export MPLBACKEND=Agg", shlex.join(command), ""])
    job_path.write_text("\n".join(lines))
    if dry_run:
        print(f"Prepared: {job_path}")
    elif cluster == "LSF":
        with job_path.open() as handle:
            subprocess.run(["bsub"], stdin=handle, check=True)
    else:
        subprocess.run(["sbatch", str(job_path)], check=True)
    return job_path


WORKER_SCRIPT = Path(__file__).with_name("job_prepare_fibers_data.py").resolve()


def input_stem(path):
    name = path.name[:-3] if path.name.lower().endswith(".gz") else path.name
    for suffix in (".txt", ".tsv", ".csv"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def preparation_plan(source, output, prefix="BestTyping", pattern="*", replicates=None,
                     start_index=None, overwrite=False):
    """Resolve all input/output pairs before creating or submitting any jobs."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("--prefix must contain only letters, digits, underscores, dots, or hyphens.")
    if start_index is not None and start_index < 1:
        raise ValueError("--start-index must be at least 1.")
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = [path.resolve() for path in source.glob(pattern) if path.is_file()
                 and (path.name.lower().endswith((".txt", ".tsv", ".csv", ".gz"))
                      or (not path.suffix and IMPUTATION_PATTERN.search(path.name)))]
    else:
        raise FileNotFoundError(source)
    paths.sort(key=lambda path: [int(part) if part.isdigit() else part.lower()
                                for part in re.split(r"(\d+)", path.name)])
    if not paths:
        raise ValueError("No TXT, TSV, CSV, gzip, or numbered extensionless input files matched.")
    if replicates is not None:
        if not 1 <= replicates <= len(paths):
            raise ValueError(f"--re must be between 1 and {len(paths)}.")
        paths = paths[:replicates]
    plan = []
    for offset, path in enumerate(paths):
        match = IMPUTATION_PATTERN.search(input_stem(path))
        if start_index is not None:
            index = start_index + offset
        elif match:
            index = int(match[1])
        else:
            raise ValueError(
                f"No imputation number at the end or before '_aligned' in {path.name}; set --start-index."
            )
        if index < 1:
            raise ValueError(f"Imputation numbers must be positive: {path.name}")
        destination = output / f"{prefix}_{index}.csv"
        if destination in paths:
            raise ValueError(f"Output would replace an input file: {destination}")
        if any(existing == destination for _, existing in plan):
            raise ValueError(f"Multiple inputs map to {destination.name}; choose a narrower --pattern.")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}; use --overwrite to replace it.")
        plan.append((path, destination))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=Path, required=True,
                        help="One combined typing TXT/TSV/CSV/gzip file or a folder")
    parser.add_argument("--w", type=Path, required=True, help="Parent folder for scratch scripts and logs")
    parser.add_argument("--o", type=Path, required=True, help="Output folder for prepared CSV datasets")
    parser.add_argument("--prefix", default="BestTyping", help="Output filename prefix (default: BestTyping)")
    parser.add_argument("--pattern", default="*", help="Input filename glob when --d is a folder")
    parser.add_argument("--re", type=int, help="Number of input files; default: all matching files")
    parser.add_argument("--start-index", type=int, help="Assign sequential output numbers instead of inferring them")
    parser.add_argument("--rc", choices=("LSF", "SLURM"), type=str.upper, default="LSF")
    parser.add_argument("--rm", type=int, default=4)
    parser.add_argument("--q", default="i2c2_normal")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    add_preparation_arguments(parser)
    args = parser.parse_args(argv)
    if args.rm < 1:
        parser.error("--rm must be positive.")
    # Validate parameter combinations before submitting any jobs. The worker is standalone.
    for name in ("tx_year_min", "tx_year_max"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 9999:
            parser.error(f"{name} must be between 1 and 9999.")
    if args.tx_year_min is not None and args.tx_year_max is not None and args.tx_year_min > args.tx_year_max:
        parser.error("tx_year_min must not exceed tx_year_max.")
    for name in ("recipient_age_min", "donor_age_min"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative.")
    if not math.isfinite(args.days_per_year) or args.days_per_year <= 0:
        parser.error("days_per_year must be finite and positive.")
    if args.censor_date is not None:
        try:
            cutoff = datetime.fromisoformat(args.censor_date.strip().replace("Z", "+00:00")).date()
        except ValueError:
            cutoff = None
            for pattern in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y"):
                try:
                    cutoff = datetime.strptime(args.censor_date.strip(), pattern).date()
                    break
                except ValueError:
                    continue
            if cutoff is None:
                parser.error("Invalid censor date; use YYYY-MM-DD.")
        if args.tx_year_min is not None and cutoff.year < args.tx_year_min:
            parser.error("censor_date precedes the entire requested transplant-year window.")
    covariates = [] if args.cov_loci.strip().lower() == "none" else [
        locus.strip() for locus in args.cov_loci.split(",") if locus.strip()
    ]
    if set(covariates) - {'A', 'B', 'C', 'DRB1', 'DRB345', 'DQA1', 'DQB1', 'DPA1', 'DPB1'}:
        raise ValueError('Unknown covariate locus')
    plan = preparation_plan(args.d.expanduser(), args.o.expanduser(), args.prefix,
                            args.pattern, args.re, args.start_index, args.overwrite)
    options = ["--mismatch-level", args.mismatch_level, "--cov-list", args.cov_loci]
    if args.delimiter:
        options.extend(["--delimiter", args.delimiter])
    if args.overwrite:
        options.append("--overwrite")
    if args.drop_missing == "true":
        options.append("--drop-missing")
    for name in ("tx_year_min", "tx_year_max", "recipient_age_min", "donor_age_min",
                 "censor_date", "days_per_year"):
        value = getattr(args, name)
        if value is not None:
            options.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("kidney_only", "deceased_donor_only"):
        if getattr(args, name) == "true":
            options.append("--" + name.replace("_", "-"))
    for source, destination in plan:
        command = [args.python_executable, str(WORKER_SCRIPT), "--d", str(source),
                   "--o", str(destination)] + options
        write_job(command, f"HLA_prepare_{destination.stem}_{time.time_ns()}",
                  args.w.expanduser().resolve(), args.rc, args.rm, args.q, args.dry_run)
        print(f"{source.name} -> {destination}")
    print(f"{len(plan)} preparation jobs {'prepared' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
