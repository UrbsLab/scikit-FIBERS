"""Shared paths, configuration and direct LSF submission (standard library only)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def configure_worker():
    sys.path.insert(0, str(REPO / "src"))
    cpus = os.environ.get("LSB_DJOB_NUMPROC", "1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[name] = cpus
    os.environ["MPLBACKEND"] = "Agg"


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    for key in ("imputations", "folds", "seeds"):
        values = config[key]
        minimum = 0 if key == "seeds" else 1
        if (not values or len(set(values)) != len(values)
                or any(type(v) is not int or v < minimum for v in values)):
            raise ValueError(f"{key} must contain unique integers >= {minimum}")
    if config["input"]["mode"] not in ("pre_split", "full_dataset"):
        raise ValueError("input.mode must be pre_split or full_dataset")
    if not config["input"]["id_column"]:
        raise ValueError("A transplant identifier is required; these files use TX_ID")
    if config["input"]["mode"] == "full_dataset":
        if config["folds"] != list(range(1, len(config["folds"]) + 1)) or len(config["folds"]) < 2:
            raise ValueError("full_dataset requires consecutive folds 1..K, K >= 2")
    if config["input"]["chunksize"] < 1:
        raise ValueError("input.chunksize must be positive")
    if not 0 <= config["minimum_nonzero_frequency"] < 1:
        raise ValueError("minimum_nonzero_frequency must be in [0, 1)")
    thresholds = config["correlation"]["thresholds"]
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 < r < 1 for r in thresholds):
        raise ValueError("Correlation thresholds must be unique and between 0 and 1")
    if config["correlation"]["primary_threshold"] not in thresholds:
        raise ValueError("primary_threshold must occur in thresholds")
    if not set(config["plots"]["thresholds"]).issubset(thresholds):
        raise ValueError("Plot thresholds must occur in correlation.thresholds")
    if not 0 < config["plots"]["interlocus_threshold"] < 1:
        raise ValueError("interlocus_threshold must be between 0 and 1")
    for locus, value in config["correlation"].get("locus_thresholds", {}).items():
        if locus not in config["columns"]["ranges"] or value not in thresholds:
            raise ValueError("locus_thresholds must use configured loci and thresholds")
    if config["fibers"].get("n_groups", 2) != 2:
        raise ValueError("This analysis evaluates two-group bins")
    if config["fibers"].get("desired_bin_effect", "default") != "default":
        raise ValueError("This analysis currently supports desired_bin_effect=default")
    if config["fibers"].get("log_rank_weighting") is not None:
        raise ValueError("This analysis evaluates unweighted log-rank separation")
    base, antigen = covariates(config)
    if set(base) & set(antigen):
        raise ValueError("Clinical and antigen covariates must not overlap")
    config["_config_path"] = str(path)
    for key in ("output_root",):
        p = Path(config[key]).expanduser()
        config[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    return config


def covariates(config):
    columns = config["columns"]
    return (list(dict.fromkeys(columns["clinical_covariates"])),
            list(dict.fromkeys(columns["antigen_covariates"])))


def input_paths(config, imputation, fold):
    from glob import glob

    names = ("train_template", "test_template") if config["input"]["mode"] == "pre_split" else ("full_template",)
    paths = []
    for name in names:
        pattern = config["input"][name].format(imputation=imputation, fold=fold)
        p = Path(pattern).expanduser()
        if not p.is_absolute():
            p = Path(config["_config_path"]).parent / p
        matches = sorted(glob(str(p)))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one CSV for {p}; found {len(matches)}")
        paths.append(Path(matches[0]).resolve())
    if len(paths) == 2 and paths[0] == paths[1]:
        raise ValueError("Train and test resolve to the same file")
    return paths


def fold_dir(config, imputation, fold):
    return Path(config["output_root"]) / f"imp_{imputation:02d}" / f"cv_{fold:02d}"


def fit_dir(config, imputation, fold, seed):
    return fold_dir(config, imputation, fold) / f"seed_{seed:03d}"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_state(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def identity(config, stage, imputation=None, fold=None, seed=None, dependencies=()):
    keys = ["input", "columns", "minimum_nonzero_frequency", "fibers"]
    code = ["common.py", "data.py", "methods.py", "run_fibers.py"]
    if stage in ("correlation", "plots"):
        keys += ["correlation", "evaluation", "seeds"]
        code += ["run_correlation.py"]
    if stage == "plots":
        keys += ["plots", "imputations", "folds"]
        code += ["run_plots.py", "plotting.py"]
    payload = {key: config[key] for key in keys}
    if config["input"]["mode"] == "full_dataset":
        payload["folds"] = config["folds"]
    payload["task"] = [imputation, fold, seed]
    payload["code"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in code}
    payload["fibers_source"] = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "src" / "skfibers").rglob("*.py"))
    }
    payload["inputs"] = [file_state(p) for p in input_paths(config, imputation, fold)] if imputation is not None else []
    payload["dependencies"] = [file_state(p) for p in dependencies]
    return {"fingerprint": digest(payload), "settings": payload}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def require_result(directory, expected):
    path = Path(directory) / "completed.json"
    if not path.is_file():
        raise RuntimeError(f"Required job has not completed: {directory}")
    result = json.loads(path.read_text())
    if result["fingerprint"] != expected["fingerprint"]:
        raise RuntimeError(f"Results do not match current inputs/settings/code: {directory}. Rerun with --force or use a new output_root.")
    for name, state in result["outputs"].items():
        if not (Path(directory) / name).is_file() or file_state(Path(directory) / name) != state:
            raise RuntimeError(f"Missing or changed result: {Path(directory) / name}")
    return result


def begin_result(directory, expected, force):
    directory = Path(directory)
    if (directory / "completed.json").exists() and not force:
        require_result(directory, expected)
        print(f"Already complete: {directory}", flush=True)
        return False
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "completed.json").unlink(missing_ok=True)
    return True


def finish_result(directory, expected, names, **details):
    expected = dict(expected)
    expected.update(details)
    expected["outputs"] = {name: file_state(Path(directory) / name) for name in names}
    expected["python"] = sys.executable
    expected["fibers_commit"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() if (REPO / ".git").exists() else "see UPSTREAM_COMMIT.txt"
    save_json(Path(directory) / "completed.json", expected)


def worker_args(stage):
    parser = argparse.ArgumentParser(description=f"Run one {stage} job on a compute node")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    if stage != "plots":
        parser.add_argument("--imputation", type=int, required=True)
        parser.add_argument("--fold", type=int, required=True)
    if stage == "fibers":
        parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("imputation", "fold", "seed"):
        if hasattr(args, name) and getattr(args, name) not in config[name + "s"]:
            raise ValueError(f"--{name} must be listed in config.{name}s")
    return args, config


def submit_jobs(stage):
    parser = argparse.ArgumentParser(description=f"Submit {stage} jobs directly with bsub")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--queue")
    parser.add_argument("--memory-gb", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--hours", type=int)
    if stage != "plots":
        parser.add_argument("--imputations", type=int, nargs="+")
        parser.add_argument("--folds", type=int, nargs="+")
    if stage == "fibers":
        parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    selected = {}
    for name in ("imputations", "folds", "seeds"):
        selected[name] = getattr(args, name, None) or config[name]
        if not set(selected[name]).issubset(config[name]):
            parser.error(f"--{name} must be a subset of config.{name}")
    tasks = [()]
    if stage != "plots":
        tasks = list(itertools.product(selected["imputations"], selected["folds"]))
        if stage == "fibers":
            tasks = [(*task, seed) for task in tasks for seed in selected["seeds"]]
    resources = config["hpc"][stage]
    cpus = args.cpus or resources["cpus"]
    memory = args.memory_gb or resources["memory_gb"]
    hours = args.hours or resources["hours"]
    if min(cpus, memory, hours) < 1:
        parser.error("cpus, memory-gb and hours must be positive")
    logs = Path(config["output_root"]) / "logs"
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    count = 0
    for task in tasks:
        name = f"ashi_{stage}" + ("_" + "_".join(map(str, task)) if task else "")
        command = ["bsub", "-q", args.queue or config["hpc"]["queue"], "-J", name,
                   "-oo", str(logs / f"{name}_%J.out"), "-eo", str(logs / f"{name}_%J.err"),
                   "-n", str(cpus), "-W", f"{hours}:00", "-R",
                   f"span[hosts=1] rusage[mem={math.ceil(memory * 1024 / cpus)}M]",
                   "-M", f"{memory}GB", sys.executable, "-u", str(HERE / f"run_{stage}.py"),
                   "--config", config["_config_path"]]
        for key, value in zip(("imputation", "fold", "seed"), task):
            command += [f"--{key}", str(value)]
        if args.force:
            command.append("--force")
        if args.dry_run:
            print(shlex.join(command))
        else:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(f"bsub failed after {count} successful submissions.\n{shlex.join(command)}\n{result.stdout}\n{result.stderr}")
            print(result.stdout.strip(), flush=True)
        count += 1
    print(f"{count} {stage} jobs {'planned' if args.dry_run else 'submitted'}. Wait for all to finish before the next main file.")
