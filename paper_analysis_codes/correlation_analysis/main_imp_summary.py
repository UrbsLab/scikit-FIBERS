"""Submit whole-imputation fits, or their final summary, directly with bsub."""
import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from common import HERE, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--imputations", type=int, nargs="+")
    parser.add_argument("--summary-only", action="store_true",
                        help="After every whole-imputation job succeeds, submit tables and figures only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--queue")
    parser.add_argument("--memory-gb", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--hours", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["input"].get("full_template"):
        parser.error("input.full_template must point to the complete imputed datasets")
    imputations = args.imputations or config["imputations"]
    if len(set(imputations)) != len(imputations) or not set(imputations).issubset(config["imputations"]):
        parser.error("--imputations must be a unique subset of config.imputations")
    if args.summary_only and (args.imputations or len(config["imputations"]) < 2):
        parser.error("--summary-only requires at least two configured imputations and summarizes all of them")
    resource = config["hpc"]["plots" if args.summary_only else "fibers"]
    cpus, memory, hours = (args.cpus or resource["cpus"], args.memory_gb or resource["memory_gb"],
                           args.hours or resource["hours"])
    if min(cpus, memory, hours) < 1:
        parser.error("cpus, memory-gb and hours must be positive")
    logs = Path(config["output_root"]) / "imp_summary" / "logs"
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    for imp in ([None] if args.summary_only else imputations):
        name = "ashi_imp_summary" if imp is None else f"ashi_whole_imp{imp}"
        command = ["bsub", "-q", args.queue or config["hpc"]["queue"], "-J", name,
                   "-oo", str(logs / f"{name}_%J.out"), "-eo", str(logs / f"{name}_%J.err"),
                   "-n", str(cpus), "-W", f"{hours}:00", "-R",
                   f"span[hosts=1] rusage[mem={math.ceil(memory * 1024 / cpus)}M]",
                   "-M", f"{memory}GB", sys.executable, "-u", str(HERE / "run_imp_summary.py"),
                   "--config", config["_config_path"]]
        command += ["--summary-only"] if imp is None else ["--imputation", str(imp)]
        if args.force:
            command.append("--force")
        if args.dry_run:
            print(shlex.join(command))
        else:
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"bsub failed: {result.stderr or result.stdout}")
            print(result.stdout.strip(), flush=True)
    print("Whole-imputation HRs are apparent (training-cohort) estimates, not held-out estimates.")
    if not args.summary_only:
        print("After all these jobs succeed, rerun this main with --summary-only. No dependencies are submitted.")


if __name__ == "__main__":
    main()
