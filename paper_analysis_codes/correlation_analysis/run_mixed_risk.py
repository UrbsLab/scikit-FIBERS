"""Compare saved top bins with within-locus and mixed-locus block processing.

Uses existing training correlation tables; only the held-out Cox models are fit.
All folds run sequentially. Existing result directories are never overwritten.
"""
from common import configure_worker
configure_worker()

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform

import lifelines
import numpy as np
import pandas as pd

from common import HERE, load_config, save_json
from data import id_digest
from methods import logrank, score_groups
from plot_mixed_blocks import make_blocks, expand, feature_key
from review_outputs import Results, features, jaccard
from run_risk import covariate_basis, fit_noag, source_inputs


def group_hash(high):
    return hashlib.sha256(np.packbits(high).tobytes()).hexdigest()


def reconstruct(source, config, imputation, seed, threshold):
    scheme = "r" + str(float(threshold)).replace(".", "p")
    records, block_audit, definitions = [], [], {}
    for fold in config["folds"]:
        root = f"imp_{imputation:02d}/cv_{fold:02d}/correlation/"
        selected = [r for r in source.json(root + "processed_bins.json")
                    if r["rank"] == 1 and r["seed"] == seed and r["scheme"] == scheme]
        if len(selected) != 1:
            raise ValueError(f"Expected one saved top bin in fold {fold}")
        saved = selected[0]
        retained = source.csv(root + "feature_filter.csv")
        names = retained.loc[retained.retained, "feature"].tolist()
        corr = source.csv(root + "correlations.csv.gz")
        blocks = make_blocks(corr, names, threshold)
        groups = expand(saved["original"], blocks)
        mixed = sorted({f for g in groups for f in g["features"]}, key=feature_key)
        within = sorted(features(saved), key=feature_key)
        record = {"fold": fold, "imputation": imputation, "seed": seed, "rank": 1,
                  "original": saved["original"], "original_threshold": saved["original_threshold"],
                  "within": within, "mixed": mixed,
                  "within_groups": saved["groups"], "mixed_groups": groups}
        definitions[str(fold)] = blocks
        records.append(record)
        cross = [b for b in blocks if b["cross_locus"]]
        block_audit.append({"fold": fold, "retained_features": len(names),
                            "multi_position_blocks": sum(len(b["features"]) > 1 for b in blocks),
                            "cross_locus_blocks": len(cross),
                            "cross_locus_blocks_touched": sum(g["cross_locus"] for g in groups),
                            "positions_original": len(saved["original"]),
                            "positions_within": len(within), "positions_mixed": len(mixed),
                            "added_vs_within": len(set(mixed) - set(within)),
                            "removed_vs_within": len(set(within) - set(mixed)),
                            "correlations_sha256": hashlib.sha256(source.read(root + "correlations.csv.gz")).hexdigest()})
        print(f"Reconstructed fold {fold}: {len(cross)} cross-locus blocks; "
              f"top-bin positions {len(within)} within / {len(mixed)} mixed", flush=True)
    return records, definitions, block_audit


def consistency(records):
    rows, pairs = [], []
    for scheme in ["original", "within", "mixed"]:
        scores = []
        for a, b in itertools.combinations(records, 2):
            value = jaccard(set(a[scheme]), set(b[scheme]))
            pairs.append({"scheme": scheme, "fold_left": a["fold"], "fold_right": b["fold"],
                          "jaccard": value})
            scores.append(value)
        rows.append({"scheme": scheme, "n_bins": len(records), "n_pairs": len(scores),
                     "mean_jaccard": float(np.mean(scores)),
                     "mean_positions": float(np.mean([len(r[scheme]) for r in records]))})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def evaluate_fold(config, record, archive, input_directory, output, exclusions):
    imp, fold, seed = (record[k] for k in ("imputation", "fold", "seed"))
    directory = output / f"imp_{imp:02d}" / f"cv_{fold:02d}"
    directory.mkdir(parents=True)
    frame, saved, source, audit, provenance = source_inputs(
        config, imp, fold, archive, input_directory, extra_features=record["mixed"])
    if any(r["original"] != record["original"] or r["original_threshold"] != record["original_threshold"]
           for r in saved if r["seed"] == seed):
        raise ValueError("Reconstructed bins differ from the original selected bin")
    known = set(source["clinical_covariates"] + source.get("antigen_covariates", []))
    if set(exclusions) - known:
        raise ValueError("Unknown excluded covariate")
    clinical = [c for c in source["clinical_covariates"] if c not in exclusions]
    covariates = {"HR": [], "Adj NoAg HR": clinical,
                  "Adj HR": clinical + [c for c in source.get("antigen_covariates", []) if c not in exclusions]}
    designs = {label: covariate_basis(frame, covs) for label, covs in covariates.items()}
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    threshold = record["original_threshold"]
    cache = {}
    scores = {"original": frame[record["original"]].sum(axis=1).to_numpy(),
              "within": score_groups(frame, record["within_groups"], cache),
              "mixed": score_groups(frame, record["mixed_groups"], cache)}
    high_groups = {name: values > threshold for name, values in scores.items()}
    results, fit_cache, coefficients = [], {}, []
    print(f"Evaluating fold {fold}: {len(frame)} held-out patients", flush=True)
    for scheme, high in high_groups.items():
        digest = group_hash(high)
        lr, lr_p = logrank(frame, high, outcome, event)
        for label, (basis, _) in designs.items():
            key = label + ":" + digest
            if key not in fit_cache:
                result, attempts, coef = fit_noag(frame, high, outcome, event, basis, 0., label="HR")
                fit_cache[key] = {"result": result, "attempts": attempts}
                if coef is not None:
                    coefficients.append(coef.assign(model=label, group_hash=digest))
                print(f"  {scheme}: {label} {result['status']}", flush=True)
            results.append({"imputation": imp, "fold": fold, "seed": seed, "rank": 1,
                            "dataset": "test", "scheme": scheme, "model": label,
                            "threshold": threshold, "group_hash": digest,
                            "changed_count": int(np.count_nonzero(high != high_groups["original"])),
                            "changed_vs_within": int(np.count_nonzero(high != high_groups["within"])),
                            "logrank": lr, "logrank_p": lr_p, **fit_cache[key]["result"]})
    table = pd.DataFrame(results)
    table.to_csv(directory / "metrics.csv", index=False)
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(directory / "coefficients.csv", index=False)
    save_json(directory / "diagnostics.json", {
        "design": {label: dict(info, explicit_exclusions=list(exclusions)) for label, (_, info) in designs.items()},
        "fits": fit_cache})
    complete = table.status.eq("ok").all()
    save_json(directory / "completed.json", {"all_estimable": bool(complete), "split_audit": audit,
        "source": provenance, "models": covariates, "excluded_covariates": list(exclusions),
        "python": platform.python_version(), "lifelines": lifelines.__version__,
        "unique_fits": len(fit_cache), "ridge": 0.0, "bin_penalty": 0.0})
    if not complete:
        raise RuntimeError(f"A fit needs review; no complete summary will be produced: {directory}")
    identifier = config["input"]["id_column"]
    return table, set(frame[identifier])


def validate_reference(metrics, reference_path, threshold, exclusions, output):
    reference = pd.read_csv(reference_path / "all_fold_metrics.csv")
    scheme = "r" + str(float(threshold)).replace(".", "p")
    comparisons = []
    for (imp, fold), part in metrics.groupby(["imputation", "fold"]):
        relative = Path(f"imp_{imp:02d}/cv_{fold:02d}/completed.json")
        before, after = [json.loads((root / relative).read_text()) for root in [reference_path, output]]
        for field in ["split_audit", "models", "excluded_covariates", "ridge", "bin_penalty"]:
            if before[field] != after[field]:
                raise ValueError(f"Reference design mismatch in {field}")
        for r in part.loc[part.scheme.isin(["original", "within"])].itertuples():
            ref_scheme = "original" if r.scheme == "original" else scheme
            match = reference.loc[(reference.imputation == imp) & (reference.fold == fold)
                                  & (reference.seed == r.seed) & (reference.model == r.model)
                                  & (reference.scheme == ref_scheme)]
            if len(match) != 1:
                raise ValueError("Expected one matching prior HR estimate")
            ref = match.iloc[0]
            if ref.group_hash != r.group_hash or ref.n != r.n or ref.n_above != r.n_above:
                raise ValueError("Reference patient grouping mismatch")
            if not np.isclose(ref.HR, r.HR, rtol=1e-5, atol=1e-8):
                raise ValueError("Reevaluated HR differs unexpectedly from prior result")
            comparisons.append({"imputation": imp, "fold": fold, "scheme": r.scheme,
                                "model": r.model, "previous_HR": ref.HR, "current_HR": r.HR,
                                "absolute_difference": abs(ref.HR - r.HR)})
    pd.DataFrame(comparisons).to_csv(output / "reference_comparison.csv", index=False)


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; prior results are never overwritten")
    config = load_config(args.config)
    if args.imputation not in config["imputations"] or args.seed not in config["seeds"]:
        raise ValueError("Imputation and seed must be configured")
    source = Results(args.archive)
    records, blocks, audit = reconstruct(source, config, args.imputation, args.seed, args.threshold)
    output.mkdir(parents=True)
    save_json(output / "mixed_bin_memberships.json", records)
    save_json(output / "mixed_blocks_by_fold.json", blocks)
    pd.DataFrame(audit).to_csv(output / "block_audit.csv", index=False)
    agreement, pairs = consistency(records)
    agreement.to_csv(output / "consistency.csv", index=False)
    pairs.to_csv(output / "pairwise_jaccard.csv", index=False)
    print(agreement.to_string(index=False), flush=True)
    tables, seen = [], set()
    for record in records:
        table, test_ids = evaluate_fold(config, record, args.archive, args.input_directory, output,
                                       args.exclude_covariates)
        if seen & test_ids:
            raise ValueError("Held-out patients are repeated across folds")
        seen.update(test_ids)
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    if args.reference:
        validate_reference(metrics, args.reference, args.threshold, args.exclude_covariates, output)
    summaries = []
    for scheme, part in metrics.groupby("scheme", sort=False):
        row = agreement.set_index("scheme").loc[scheme].to_dict()
        row.update(scheme=scheme, imputation=args.imputation, seed=args.seed,
                   correlation_threshold=args.threshold)
        unadjusted = part.loc[part.model == "HR"]
        row.update(total_patients=int(unadjusted.n.sum()),
                   changed_patients=int(unadjusted.changed_count.sum()),
                   changed_vs_within=int(unadjusted.changed_vs_within.sum()),
                   mean_logrank=float(unadjusted.logrank.mean()))
        for label, model in part.groupby("model"):
            if len(model) != len(config["folds"]) or not model.status.eq("ok").all():
                raise ValueError("Incomplete model comparison")
            row["mean_" + label] = float(model.HR.mean())
        summaries.append(row)
    metrics.to_csv(output / "all_fold_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    save_json(output / "completed.json", {"all_estimable": True, "imputation": args.imputation,
        "seed": args.seed, "folds": config["folds"], "threshold": args.threshold,
        "block_rule": "complete-linkage over all retained loci; every pair strictly exceeds threshold",
        "scoring": "sum of maximum mismatch per block; unchanged original bin threshold",
        "patient_count": len(seen), "cohort_ids_sha256": id_digest(seen),
        "archive": str(args.archive.resolve()), "reference": str(args.reference) if args.reference else None,
        "excluded_covariates": args.exclude_covariates, "ridge": 0.0,
        "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in ["run_mixed_risk.py", "run_risk.py", "plot_mixed_blocks.py", "methods.py"]}})
    print(summary.to_string(index=False), flush=True)
    if source.archive:
        source.archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Prior successful HR reevaluation for baseline verification")
    parser.add_argument("--imputation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=.95)
    parser.add_argument("--exclude-covariates", nargs="*", default=[])
    run(parser.parse_args())
