"""Reevaluate three held-out HR models from saved bins, without refitting FIBERS."""
from common import configure_worker
configure_worker()

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import warnings
import zipfile
import platform
import lifelines

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy.linalg import qr

from common import HERE, fold_dir, input_paths, load_config, save_json
from data import id_digest
from methods import score_groups


def covariate_basis(frame, covariates):
    """Retain the nuisance column space, removing constants/exact aliases only."""
    covariates = list(dict.fromkeys(covariates))
    values = frame[covariates].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Clinical covariates contain non-finite values")
    constants = [c for c in covariates if values[c].nunique() <= 1]
    active = [c for c in covariates if c not in constants]
    if not active:
        return pd.DataFrame(index=frame.index), {"constants": constants, "aliases": [], "retained": []}
    centered = values[active] - values[active].mean()
    scaled = centered / centered.std(ddof=0)
    a = scaled.to_numpy()
    _, r, pivots = qr(a, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    tolerance = np.finfo(float).eps * max(a.shape) * diagonal.max()
    rank = int(np.count_nonzero(diagonal > tolerance))
    keep_indexes = set(pivots[:rank])
    retained = [c for i, c in enumerate(active) if i in keep_indexes]
    aliases = [c for c in active if c not in retained]
    basis = scaled[retained]
    alias_errors = {}
    for name in aliases:
        fitted = basis.to_numpy() @ np.linalg.lstsq(basis, scaled[name], rcond=None)[0]
        relative_error = np.linalg.norm(scaled[name] - fitted) / np.linalg.norm(scaled[name])
        if relative_error > 1e-10:
            raise ValueError(f"Refusing to discard nonredundant covariate {name}")
        alias_errors[name] = float(relative_error)
    return basis, {"constants": constants, "aliases": aliases, "retained": retained,
                   "alias_relative_errors": alias_errors, "rank": rank,
                   "condition_number": float(np.linalg.cond(basis.to_numpy()))}


def fit_noag(frame, high, outcome, event, basis, ridge=0.0, label="Adj NoAg HR"):
    """The bin coefficient is never penalized; ridge, if requested, is fixed."""
    high = np.asarray(high, dtype=bool)
    result = {label: None, label + " lower": None,
              label + " upper": None, label + " p": None,
              "n": len(frame), "n_above": int(high.sum()),
              "events": int(frame[event].sum()), "ridge": float(ridge), "status": "pending"}
    attempts = []
    if not high.any() or high.all():
        result["status"] = "single_group"
        return result, attempts, None
    if frame[event].sum() == 0:
        result["status"] = "no_events"
        return result, attempts, None
    if any(frame.loc[high == group, event].sum() == 0 for group in (False, True)):
        result["status"] = "no_events_in_one_bin_group"
        return result, attempts, None
    centered_high = high.astype(float) - high.mean()
    if len(basis.columns):
        fitted = basis.to_numpy() @ np.linalg.lstsq(basis, centered_high, rcond=None)[0]
        if np.linalg.norm(centered_high - fitted) / np.linalg.norm(centered_high) < 1e-10:
            result["status"] = "bin_not_identifiable_given_covariates"
            return result, attempts, None
    model_frame = frame[[outcome, event]].reset_index(drop=True).copy()
    for name in basis:
        model_frame[name] = basis[name].to_numpy()
    model_frame["_above"] = high.astype(float)
    penalty = np.r_[np.full(len(basis.columns), ridge), 0.0]
    for step in (0.5, 0.1):
        messages = []
        attempt = {"step_size": step}
        try:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                model = CoxPHFitter(penalizer=penalty, l1_ratio=0.0).fit(
                    model_frame, duration_col=outcome, event_col=event,
                    fit_options={"step_size": step, "max_steps": 1000, "precision": 1e-7})
            row = model.summary.loc["_above"]
            fields = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
            finite = (np.isfinite(row[fields].to_numpy(dtype=float)).all()
                      and np.isfinite(model.params_.to_numpy()).all()
                      and np.isfinite(model.standard_errors_.to_numpy()).all()
                      and (row[fields[:3]] > 0).all())
            convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in messages)
            if not finite or convergence_warning:
                attempt["status"] = "convergence_warning" if convergence_warning else "nonfinite_estimate"
            else:
                for suffix, field in zip(("", " lower", " upper", " p"), fields):
                    result[label + suffix] = float(row[field])
                result["status"] = "ok"
                result["step_size"] = step
                result["n_covariates"] = len(basis.columns)
                attempt["status"] = "ok"
                attempt["warnings"] = [str(w.message) for w in messages]
                attempts.append(attempt)
                return result, attempts, model.summary.reset_index()
        except Exception as error:
            attempt["status"] = f"{type(error).__name__}: {error}"
        attempt["warnings"] = [str(w.message) for w in messages]
        attempts.append(attempt)
    result["status"] = "not_estimable_after_numerical_retries"
    return result, attempts, None


def read_ids(path, identifier, chunksize):
    ids = set()
    n = 0
    for chunk in pd.read_csv(path, usecols=[identifier], dtype={identifier: "string"}, chunksize=chunksize):
        values = chunk[identifier].str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Missing identifiers in {path}")
        n += len(values)
        ids.update(values)
    if len(ids) != n:
        raise ValueError(f"Duplicate identifiers in {path}")
    return ids


def source_inputs(config, imputation, fold, archive=None, input_directory=None, extra_features=()):
    if config["input"]["mode"] != "pre_split":
        raise ValueError("This risk-only repair currently requires the existing pre_split files")
    directory = fold_dir(config, imputation, fold) / "correlation"
    if archive:
        with zipfile.ZipFile(archive) as zipped:
            prefix = "" if "summary/completed.json" in zipped.namelist() else "simple/"
            base = f"{prefix}imp_{imputation:02d}/cv_{fold:02d}/correlation/"
            record = json.loads(zipped.read(base + "completed.json"))
            memberships = zipped.read(base + "processed_bins.json")
    else:
        record = json.loads((directory / "completed.json").read_text())
        memberships = (directory / "processed_bins.json").read_bytes()
    if record["settings"]["columns"] != config["columns"]:
        raise ValueError("Current column configuration differs from the completed correlation analysis")
    paths = ([Path(input_directory) / Path(s["path"]).name for s in record["settings"]["inputs"]]
             if input_directory else input_paths(config, imputation, fold))
    for path, state in zip(paths, record["settings"]["inputs"]):
        if path.stat().st_size != state["size"] or (not input_directory and path.stat().st_mtime_ns != state["mtime_ns"]):
            raise ValueError(f"Input changed since original analysis: {path}")
    processed = json.loads(memberships)
    rows = [r for r in processed if r["rank"] == 1 and r["seed"] in config["seeds"]]
    if not rows:
        raise ValueError("No saved top-bin memberships")
    required = set(record["clinical_covariates"] + record.get("antigen_covariates", []))
    required.update(extra_features)
    for row in rows:
        required.update(row["original"])
        required.update(f for group in row["groups"] for f in group["features"])
    identifier = config["input"]["id_column"]
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    required.update([outcome, event])
    frames = []
    for chunk in pd.read_csv(paths[1], usecols=[identifier] + sorted(required),
                             dtype={identifier: "string"}, chunksize=config["input"]["chunksize"]):
        chunk[identifier] = chunk[identifier].str.strip()
        chunk[list(required)] = chunk[list(required)].astype(float)
        if not np.isfinite(chunk[list(required)].to_numpy()).all():
            raise ValueError("Non-finite held-out values")
        frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True)
    if (frame[outcome] < 0).any() or not frame[event].isin([0, 1]).all():
        raise ValueError("Invalid survival outcomes")
    train_ids = read_ids(paths[0], identifier, config["input"]["chunksize"])
    test_ids = read_ids(paths[1], identifier, config["input"]["chunksize"])
    if train_ids & test_ids:
        raise ValueError("Training and test identifiers overlap")
    audit = {"train_n": len(train_ids), "test_n": len(test_ids),
             "train_ids_sha256": id_digest(train_ids), "test_ids_sha256": id_digest(test_ids),
             "cohort_ids_sha256": id_digest(train_ids | test_ids)}
    if audit != record["split_audit"]:
        raise ValueError("Patient partition differs from the saved FIBERS/correlation analysis")
    provenance = {"memberships_sha256": hashlib.sha256(memberships).hexdigest(),
                  "input_paths": [str(p) for p in paths],
                  "relocated_copy": bool(input_directory),
                  "verification": "file sizes and all train/test/cohort identifier hashes"}
    return frame, rows, record, audit, provenance


def summarize_when_complete(config, output, signature):
    """The last finished bsub job writes a complete, paired summary under a lock."""
    with (output / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        all_rows = []
        for imp in config["imputations"]:
            for fold in config["folds"]:
                directory = output / f"imp_{imp:02d}" / f"cv_{fold:02d}"
                marker = directory / "completed.json"
                if not marker.exists():
                    return
                record = json.loads(marker.read_text())
                if record["signature"] != signature or not record["all_estimable"]:
                    return
                all_rows.append(pd.read_csv(directory / "metrics.csv"))
        metrics = pd.concat(all_rows, ignore_index=True)
        summaries = []
        for (imp, seed, model), frame in metrics.groupby(["imputation", "seed", "model"]):
            original = frame.loc[frame.scheme == "original"].set_index("fold")
            for scheme, part in frame.groupby("scheme"):
                part = part.set_index("fold").sort_index()
                if set(part.index) != set(config["folds"]) or part.index.duplicated().any():
                    raise ValueError("Cannot summarize an incomplete or duplicated fold set")
                before = original.loc[part.index, "HR"]
                summaries.append({"imputation": imp, "seed": seed, "scheme": scheme,
                                  "model": model, "n_folds": len(part), "mean_original_HR": before.mean(),
                                  "mean_processed_HR": part["HR"].mean(),
                                  "mean_paired_delta": (part["HR"] - before).mean(),
                                  "ridge": part.ridge.iloc[0]})
        metrics.to_csv(output / "all_fold_metrics.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        print(f"All folds complete. Paired summaries: {output / 'summary.csv'}", flush=True)


def run(config, imputation, fold, output, ridge=0.0, force=False, archive=None, input_directory=None,
        exclude_covariates=()):
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    output = Path(output)
    directory = output / f"imp_{imputation:02d}" / f"cv_{fold:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({"config": {k:v for k,v in config.items() if not k.startswith("_")},
                                          "ridge": ridge, "code": Path(__file__).read_text(),
                                          "archive": str(archive), "input_directory": str(input_directory),
                                          "exclude_covariates": sorted(exclude_covariates),
                                          "python": platform.python_version(), "lifelines": lifelines.__version__},
                                         sort_keys=True).encode()).hexdigest()
    marker = directory / "completed.json"
    if marker.exists() and not force:
        raise FileExistsError(f"Already evaluated: {directory}. Use a new output root or --force.")
    marker.unlink(missing_ok=True)
    with (output / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ("summary.csv", "all_fold_metrics.csv"):
            (output / name).unlink(missing_ok=True)
    frame, bins, source, audit, provenance = source_inputs(config, imputation, fold, archive, input_directory)
    unknown = set(exclude_covariates) - set(source["clinical_covariates"] + source.get("antigen_covariates", []))
    if unknown:
        raise ValueError(f"Unknown excluded covariates: {sorted(unknown)}")
    clinical = [c for c in source["clinical_covariates"] if c not in exclude_covariates]
    models = {"HR": [], "Adj NoAg HR": clinical,
              "Adj HR": clinical + [c for c in source.get("antigen_covariates", []) if c not in exclude_covariates]}
    designs = {name: covariate_basis(frame, covs) for name, covs in models.items()}
    for _, design in designs.values():
        design["explicit_exclusions"] = list(exclude_covariates)
    print(f"Fold {fold}: " + json.dumps({name: {k:v for k,v in info.items() if k in ('constants','aliases')}
                                       for name, (_, info) in designs.items()}), flush=True)
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    rows, fit_cache, coefficient_tables = [], {}, []
    for seed in config["seeds"]:
        saved = [b for b in bins if b["seed"] == seed]
        if not saved:
            raise ValueError(f"Missing seed {seed}")
        first = saved[0]
        original = first["original"]
        threshold = first["original_threshold"]
        if any(b["original"] != original or b["original_threshold"] != threshold for b in saved):
            raise ValueError("Saved schemes disagree about the original top bin")
        scores = {"original": frame[original].sum(axis=1).to_numpy()}
        cache = {}
        for b in saved:
            scores[b["scheme"]] = score_groups(frame, b["groups"], cache)
        original_high = scores["original"] > threshold
        for scheme, score in scores.items():
            high = score > threshold
            group_hash = hashlib.sha256(np.packbits(high).tobytes()).hexdigest()
            for model_name, (basis, _) in designs.items():
                key = model_name + ':' + group_hash
                if key not in fit_cache:
                    result, attempts, coefficients = fit_noag(frame, high, outcome, event, basis, ridge, label="HR")
                    fit_cache[key] = {"result": result, "attempts": attempts}
                    print(f"  {scheme}: {model_name} {result['status']}", flush=True)
                    if coefficients is not None:
                        coefficient_tables.append(coefficients.assign(group_hash=group_hash, model=model_name))
                rows.append({"imputation": imputation, "fold": fold, "seed": seed, "rank": 1,
                             "dataset": "test", "scheme": scheme, "threshold": threshold,
                             "model": model_name, "changed_count": int(np.count_nonzero(high != original_high)),
                             "group_hash": group_hash, **fit_cache[key]["result"]})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    if coefficient_tables:
        pd.concat(coefficient_tables, ignore_index=True).to_csv(directory / "coefficients.csv", index=False)
    save_json(directory / "diagnostics.json", {"design": {name: info for name, (_, info) in designs.items()}, "fits": fit_cache})
    all_estimable = all(row["status"] == "ok" for row in rows)
    save_json(marker, {"signature": signature, "all_estimable": all_estimable,
                       "split_audit": audit, "n_models": len(rows), "unique_fits": len(fit_cache),
                       "source": provenance,
                       "models": models, "python": platform.python_version(), "lifelines": lifelines.__version__,
                       "excluded_covariates": list(exclude_covariates),
                       "ridge": ridge, "bin_penalty": 0.0})
    if not all_estimable:
        raise RuntimeError(f"Some estimates need further review. Details: {directory / 'diagnostics.json'}")
    summarize_when_complete(config, output, signature)
    print(f"Completed all three held-out HR models: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--imputation", type=int, required=True)
    parser.add_argument("--fold", type=int, help="Omit to evaluate all configured folds sequentially")
    parser.add_argument("--archive", type=Path, help="Existing results ZIP instead of output_root")
    parser.add_argument("--input-directory", type=Path, help="Relocated CSV copies, checked against saved split hashes and file sizes")
    parser.add_argument("--exclude-covariates", nargs="*", default=[],
                        help="Explicit, documented exclusions applied to ALL original/processed models and folds")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ridge", type=float, default=0.0, help="Fixed nuisance-only ridge for ALL models; 0 = unpenalized")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.imputation not in config["imputations"] or (args.fold is not None and args.fold not in config["folds"]):
        parser.error("Imputation and fold must occur in the configuration")
    output = args.output_root or Path(config["output_root"]) / "risk_reevaluation"
    for fold in ([args.fold] if args.fold is not None else config["folds"]):
        run(config, args.imputation, fold, output, args.ridge, args.force, args.archive, args.input_directory,
            args.exclude_covariates)


if __name__ == "__main__":
    main()
