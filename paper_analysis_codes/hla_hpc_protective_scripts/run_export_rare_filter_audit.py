"""Create and submit an HPC job that exports rare-filter audit tables."""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "job_export_rare_filter_audit.py"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-datafolder", type=Path, default=None)
    parser.add_argument("--noncv-datafile", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--ra", dest="rare_filter", type=float, default=0.1)
    parser.add_argument("--loci-list", default="A,B,C,DRB1,DRB345,DQA1,DQB1")
    parser.add_argument(
        "--rc",
        dest="run_cluster",
        type=str.upper,
        choices=("LSF", "SLURM"),
        default="LSF",
    )
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=64)
    parser.add_argument("--q", dest="queue", default="i2c2_normal")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the job file without submitting it.",
    )
    return parser.parse_args(argv[1:])


def validate_args(args):
    if args.cv_datafolder is None and args.noncv_datafile is None:
        raise ValueError("Provide at least one of --cv-datafolder or --noncv-datafile.")
    if not 0.0 <= args.rare_filter <= 1.0:
        raise ValueError("--ra must be between 0 and 1")
    if args.reserved_memory < 1:
        raise ValueError("--rm must be at least 1 GB")

    if args.cv_datafolder is not None:
        args.cv_datafolder = args.cv_datafolder.expanduser().resolve()
        if not args.cv_datafolder.is_dir():
            raise NotADirectoryError(args.cv_datafolder)
    if args.noncv_datafile is not None:
        args.noncv_datafile = args.noncv_datafile.expanduser().resolve()
        if not args.noncv_datafile.is_file():
            raise FileNotFoundError(args.noncv_datafile)
    args.save_dir = args.save_dir.expanduser().resolve()
    return args


def build_worker_command(args):
    command = [
        args.python_executable,
        str(WORKER_SCRIPT),
        "--save-dir", str(args.save_dir),
        "--ra", str(args.rare_filter),
        "--loci-list", args.loci_list,
    ]
    if args.cv_datafolder is not None:
        command.extend(["--cv-datafolder", str(args.cv_datafolder)])
    if args.noncv_datafile is not None:
        command.extend(["--noncv-datafile", str(args.noncv_datafile)])
    return command


def write_job_file(args, scratch_path, log_path):
    job_name = f"RARE_AUDIT_{time.time_ns()}"
    job_path = scratch_path / f"{job_name}_run.sh"
    worker_command = shlex.join(build_worker_command(args))

    if args.run_cluster == "SLURM":
        directives = [
            f"#SBATCH -p {args.queue}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --mem={args.reserved_memory}G",
            f"#SBATCH -o {log_path / (job_name + '.o')}",
            f"#SBATCH -e {log_path / (job_name + '.e')}",
        ]
        launch_command = f"srun {worker_command}"
    else:
        directives = [
            f"#BSUB -q {args.queue}",
            f"#BSUB -J {job_name}",
            f'#BSUB -R "rusage[mem={args.reserved_memory}G]"',
            f"#BSUB -M {args.reserved_memory}GB",
            f"#BSUB -o {log_path / (job_name + '.o')}",
            f"#BSUB -e {log_path / (job_name + '.e')}",
        ]
        launch_command = worker_command

    payload = "\n".join(
        ["#!/bin/bash", *directives, "set -euo pipefail", launch_command, ""]
    )
    job_path.write_text(payload, encoding="utf-8")
    return job_path


def submit_job(job_path, run_cluster):
    if run_cluster == "SLURM":
        subprocess.run(["sbatch", str(job_path)], check=True)
    else:
        with job_path.open("r", encoding="utf-8") as job_file:
            subprocess.run(["bsub"], stdin=job_file, check=True)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    args = validate_args(parse_args(argv))
    write_root = args.save_dir.parent
    scratch_path = write_root / "scratch"
    log_path = write_root / "logs"
    for directory in (args.save_dir, scratch_path, log_path):
        directory.mkdir(parents=True, exist_ok=True)

    job_path = write_job_file(args, scratch_path, log_path)
    if args.dry_run:
        print(f"Audit job created without submission: {job_path}")
    else:
        submit_job(job_path, args.run_cluster)
        print("1 audit job submitted successfully")


if __name__ == "__main__":
    main()
