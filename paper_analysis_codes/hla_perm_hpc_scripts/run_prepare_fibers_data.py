"""Submit combined-typing preparation jobs and name outputs <prefix>_<imputation>.csv."""

import argparse
import shlex
import subprocess
import re
import sys
import time
from pathlib import Path

DEFAULT_COV_LOCI = ('A', 'B', 'C', 'DRB1', 'DQA1', 'DQB1')
MISMATCH_PATTERNS = ('position', 'residue', 'permutation')


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
    return Path(name).stem


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
                 and path.name.lower().endswith((".txt", ".tsv", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz"))]
    else:
        raise FileNotFoundError(source)
    paths.sort(key=lambda path: [int(part) if part.isdigit() else part.lower()
                                for part in re.split(r"(\d+)", path.name)])
    if not paths:
        raise ValueError("No input TXT, TSV, or CSV files matched.")
    if replicates is not None:
        if not 1 <= replicates <= len(paths):
            raise ValueError(f"--re must be between 1 and {len(paths)}.")
        paths = paths[:replicates]
    plan = []
    for offset, path in enumerate(paths):
        match = re.search(r"([0-9]+)$", input_stem(path))
        if start_index is not None:
            index = start_index + offset
        elif match:
            index = int(match[1])
        else:
            raise ValueError(f"No trailing imputation number in {path.name}; set --start-index.")
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
    parser.add_argument("--d", type=Path, required=True, help="One combined typing file or a folder")
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
    for source, destination in plan:
        command = [args.python_executable, str(WORKER_SCRIPT), "--d", str(source),
                   "--o", str(destination)] + options
        write_job(command, f"HLA_prepare_{destination.stem}_{time.time_ns()}",
                  args.w.expanduser().resolve(), args.rc, args.rm, args.q, args.dry_run)
        print(f"{source.name} -> {destination}")
    print(f"{len(plan)} preparation jobs {'prepared' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
