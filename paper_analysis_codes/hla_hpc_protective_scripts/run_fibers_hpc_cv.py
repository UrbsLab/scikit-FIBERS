"""Create and submit per-fold HPC jobs for directional FIBERS paper runs."""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "job_fibers_hpc_cv.py"
VALID_EFFECTS = ("default", "protective", "high_risk")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", dest="datafolder", type=Path, required=True)
    parser.add_argument("--w", dest="writepath", type=Path, required=True)
    parser.add_argument("--o", dest="outputfolder", default="myOutput")
    parser.add_argument("--pi", dest="manual_bin_init", default="None")
    parser.add_argument(
        "--rc",
        dest="run_cluster",
        type=str.upper,
        choices=("LSF", "SLURM"),
        default="LSF",
    )
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=4)
    parser.add_argument("--q", dest="queue", default="i2c2_normal")
    parser.add_argument("--cv", dest="cv", type=int, default=10)
    parser.add_argument("--loci-list", dest="loci_list", default="A,B,C,DRB1,DRB345,DQA1,DQB1")
    parser.add_argument("--cov-list", dest="cov_list", default="None")
    parser.add_argument("--ra", dest="rare_filter", type=float, default=0.1)

    parser.add_argument("--ol", dest="outcome_label", default="Duration")
    parser.add_argument("--ot", dest="outcome_type", default="survival")
    parser.add_argument("--i", dest="iterations", type=int, default=100)
    parser.add_argument("--ps", dest="pop_size", type=int, default=50)
    parser.add_argument("--tp", dest="tournament_prop", type=float, default=0.2)
    parser.add_argument("--cp", dest="crossover_prob", type=float, default=0.5)
    parser.add_argument("--mi", dest="min_mutation_prob", type=float, default=0.1)
    parser.add_argument("--ma", dest="max_mutation_prob", type=float, default=0.5)
    parser.add_argument("--mp", dest="merge_prob", type=float, default=0.1)
    parser.add_argument("--ng", dest="new_gen", type=float, default=1.0)
    parser.add_argument("--e", dest="elitism", type=float, default=0.1)
    parser.add_argument("--dp", dest="diversity_pressure", type=int, default=0)
    parser.add_argument("--bi", dest="min_bin_size", type=int, default=1)
    parser.add_argument("--ba", dest="max_bin_size", default="None")
    parser.add_argument("--ib", dest="max_bin_init_size", type=int, default=10)
    parser.add_argument("--f", dest="fitness_metric", default="log_rank")
    parser.add_argument("--we", dest="log_rank_weighting", default="None")
    parser.add_argument("--c", dest="censor_label", default="Censoring")
    parser.add_argument("--g", dest="group_strata_min", type=float, default=0.1)
    parser.add_argument("--p", dest="penalty", type=float, default=0.5)
    parser.add_argument("--t", dest="group_thresh", default="0")
    parser.add_argument("--it", dest="min_thresh", type=int, default=0)
    parser.add_argument("--at", dest="max_thresh", type=int, default=5)
    parser.add_argument("--te", dest="thresh_evolve_prob", type=float, default=0.5)
    parser.add_argument(
        "--de-list",
        dest="desired_bin_effects",
        default=",".join(VALID_EFFECTS),
        help="Comma-separated directional modes.",
    )
    parser.add_argument("--cl", dest="pop_clean", default="None")
    parser.add_argument("--r", dest="random_seed", type=int, default=None)
    parser.add_argument(
        "--python-executable",
        default="python",
        help="Python executable available on the compute node.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write job files without submitting them.",
    )
    return parser.parse_args(argv[1:])


def normalize_effects(value):
    effects = [item.strip() for item in value.split(",") if item.strip()]
    effects = ["high_risk" if item == "highrisk" else item for item in effects]
    invalid = sorted(set(effects) - set(VALID_EFFECTS))
    if invalid:
        raise ValueError(
            f"Unsupported desired bin effect(s): {', '.join(invalid)}. "
            f"Choose from {', '.join(VALID_EFFECTS)}."
        )
    if not effects:
        raise ValueError("At least one desired bin effect is required.")
    return list(dict.fromkeys(effects))


def validate_args(args):
    args.datafolder = args.datafolder.expanduser().resolve()
    args.writepath = args.writepath.expanduser().resolve()
    if not args.datafolder.is_dir():
        raise NotADirectoryError(args.datafolder)
    if args.cv < 1:
        raise ValueError("--cv must be at least 1")
    if args.reserved_memory < 1:
        raise ValueError("--rm must be at least 1 GB")
    if not 0.0 <= args.rare_filter <= 1.0:
        raise ValueError("--ra must be between 0 and 1")
    if args.manual_bin_init != "None":
        manual_path = Path(args.manual_bin_init).expanduser().resolve()
        if not manual_path.is_file():
            raise FileNotFoundError(manual_path)
        args.manual_bin_init = str(manual_path)
    return args


def build_worker_command(args, output_path, part, desired_bin_effect):
    command = [
        args.python_executable,
        str(WORKER_SCRIPT),
        "--d", str(args.datafolder),
        "--o", str(output_path),
        "--pi", str(args.manual_bin_init),
        "--ol", args.outcome_label,
        "--ot", args.outcome_type,
        "--cv", str(part),
        "--i", str(args.iterations),
        "--ps", str(args.pop_size),
        "--tp", str(args.tournament_prop),
        "--cp", str(args.crossover_prob),
        "--mi", str(args.min_mutation_prob),
        "--ma", str(args.max_mutation_prob),
        "--mp", str(args.merge_prob),
        "--ng", str(args.new_gen),
        "--e", str(args.elitism),
        "--dp", str(args.diversity_pressure),
        "--bi", str(args.min_bin_size),
        "--ba", str(args.max_bin_size),
        "--ib", str(args.max_bin_init_size),
        "--f", args.fitness_metric,
        "--we", str(args.log_rank_weighting),
        "--c", args.censor_label,
        "--g", str(args.group_strata_min),
        "--p", str(args.penalty),
        "--t", str(args.group_thresh),
        "--it", str(args.min_thresh),
        "--at", str(args.max_thresh),
        "--te", str(args.thresh_evolve_prob),
        "--de", desired_bin_effect,
        "--cl", str(args.pop_clean),
        "--loci-list", args.loci_list,
        "--cov-list", args.cov_list,
        "--ra", str(args.rare_filter),
    ]
    if args.random_seed is not None:
        command.extend(["--r", str(args.random_seed)])
    return command


def write_job_file(args, scratch_path, log_path, output_path, part, desired_bin_effect):
    job_ref = time.time_ns()
    job_name = f"FIBERS_{desired_bin_effect}_{part}_{job_ref}"
    job_path = scratch_path / f"{job_name}_run.sh"
    worker_command = shlex.join(
        build_worker_command(args, output_path, part, desired_bin_effect)
    )

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
    desired_bin_effects = normalize_effects(args.desired_bin_effects)

    scratch_path = args.writepath / "scratch"
    log_path = args.writepath / "logs"
    output_root = args.writepath / "output" / f"Fibers2.0_org_{args.outputfolder}"
    for directory in (scratch_path, log_path, output_root):
        directory.mkdir(parents=True, exist_ok=True)

    job_paths = []
    for desired_bin_effect in desired_bin_effects:
        for part in range(1, args.cv + 1):
            output_path = output_root / desired_bin_effect / str(part)
            output_path.mkdir(parents=True, exist_ok=True)
            job_path = write_job_file(
                args,
                scratch_path,
                log_path,
                output_path,
                part,
                desired_bin_effect,
            )
            job_paths.append(job_path)
            if not args.dry_run:
                submit_job(job_path, args.run_cluster)

    action = "created" if args.dry_run else "submitted"
    print(
        f"{len(job_paths)} jobs {action} successfully across modes: "
        + ", ".join(desired_bin_effects)
    )


if __name__ == "__main__":
    main()
