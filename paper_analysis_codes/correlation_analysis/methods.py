"""Pearson blocks, bin scoring and survival metrics used by the three workers."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import logrank_test
from scipy.linalg import qr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data import feature_key


def pearson_matrix(frame, features, chunksize):
    """Accumulate all pairs with matrix multiplication, without row/pair loops."""
    total = np.zeros(len(features), dtype=np.float64)
    cross = np.zeros((len(features), len(features)), dtype=np.float64)
    for start in range(0, len(frame), chunksize):
        values = frame.iloc[start:start + chunksize][features].to_numpy(dtype=np.float64)
        total += values.sum(axis=0)
        cross += values.T @ values
    centered = cross - np.outer(total, total) / len(frame)
    variance = np.diag(centered).copy()
    if (variance <= 0).any():
        raise ValueError("Remove constant features before computing correlations")
    matrix = np.clip(centered / np.sqrt(np.outer(variance, variance)), -1, 1)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def correlation_table(matrix, features, n):
    left, right = np.triu_indices(len(features), 1)
    names = np.array(features)
    loci = np.array([feature_key(name)[0] for name in features])
    return pd.DataFrame({"feature1": names[left], "feature2": names[right],
                         "locus1": loci[left], "locus2": loci[right],
                         "pearson_r": matrix[left, right], "n_train": n})


def scheme_name(threshold, scope="within"):
    return ("any_" if scope == "any" else "") + "r" + str(float(threshold)).replace(".", "p")


def block_id(features):
    return "B_" + hashlib.sha256("|".join(sorted(features)).encode()).hexdigest()[:12]


def build_schemes(matrix, features, config):
    settings = config["correlation"]
    thresholds = sorted(settings["thresholds"])
    loci = list(config["columns"]["ranges"])
    blocks = {}
    for locus in loci:
        indexes = [i for i, name in enumerate(features) if feature_key(name)[0] == locus]
        names = [features[i] for i in indexes]
        tree = None
        if len(indexes) > 1:
            distances = np.clip(1 - matrix[np.ix_(indexes, indexes)], 0, 2)
            np.fill_diagonal(distances, 0)
            tree = linkage(squareform(distances, checks=False), method="complete")
        for threshold in thresholds:
            groups = []
            if tree is not None:
                labels = fcluster(tree, np.nextafter(1 - threshold, -np.inf), criterion="distance")
                for label in np.unique(labels):
                    members = [name for name, group in zip(names, labels) if group == label]
                    if len(members) > 1:
                        positions = [features.index(name) for name in members]
                        minimum = float(matrix[np.ix_(positions, positions)][np.triu_indices(len(members), 1)].min())
                        if minimum <= threshold:
                            raise AssertionError("A block violates its pairwise correlation cutoff")
                        groups.append({"block": block_id(members), "locus": locus,
                                       "features": members, "min_r": minimum})
            blocks[locus, threshold] = groups
    schemes = {}

    def add(name, threshold_map):
        schemes[name] = {"thresholds": threshold_map,
                         "blocks": [block for locus in loci for block in blocks[locus, threshold_map[locus]]]}

    for threshold in thresholds:
        add(scheme_name(threshold), {locus: threshold for locus in loci})
    scopes = settings.get("scopes", ["within"])
    if "any" in scopes:
        # Reuse the same training matrix; cluster once across all loci.
        distances = np.clip(1 - matrix, 0, 2)
        np.fill_diagonal(distances, 0)
        tree = linkage(squareform(distances, checks=False), method="complete") if len(features) > 1 else None
        for threshold in thresholds:
            groups = []
            labels = (fcluster(tree, np.nextafter(1 - threshold, -np.inf), criterion="distance")
                      if tree is not None else np.ones(len(features), dtype=int))
            for label in np.unique(labels):
                indexes = np.flatnonzero(labels == label)
                if len(indexes) < 2:
                    continue
                members = [features[i] for i in indexes]
                minimum = float(matrix[np.ix_(indexes, indexes)][np.triu_indices(len(indexes), 1)].min())
                if minimum <= threshold:
                    raise AssertionError("An any-locus block violates its all-pairs cutoff")
                member_loci = list(dict.fromkeys(feature_key(name)[0] for name in members))
                groups.append({"block": block_id(members), "locus": "+".join(member_loci),
                               "loci": member_loci, "features": members, "min_r": minimum})
            schemes[scheme_name(threshold, "any")] = {
                "scope": "any", "thresholds": {locus: threshold for locus in loci}, "blocks": groups}
    if "within" not in scopes:
        schemes = {name: scheme for name, scheme in schemes.items() if name.startswith("any_")}
    overrides = settings.get("locus_thresholds", {})
    if overrides and "within" in scopes:
        add("locus_specific", {locus: overrides.get(locus, settings["primary_threshold"]) for locus in loci})
    adaptive = settings["adaptive"]
    if adaptive["enabled"] and "within" in scopes:
        chosen = {}
        for locus in loci:
            chosen[locus] = settings["primary_threshold"]
            for threshold in reversed(thresholds):
                groups = blocks[locus, threshold]
                if len(groups) >= adaptive["min_blocks"] and sum(len(b["features"]) for b in groups) >= adaptive["min_features"]:
                    chosen[locus] = threshold
                    break
        add("locus_adaptive", chosen)
    return schemes


def expand_bin(original, scheme):
    mapping = {feature: block for block in scheme["blocks"] for feature in block["features"]}
    groups, seen = [], set()
    for feature in original:
        group = mapping.get(feature, {"block": f"single:{feature}", "locus": feature_key(feature)[0], "features": [feature]})
        if group["block"] not in seen:
            groups.append(group)
            seen.add(group["block"])
    return groups


def score_groups(frame, groups, cache=None):
    cache = {} if cache is None else cache
    result = np.zeros(len(frame))
    for group in groups:
        key = tuple(group["features"])
        if key not in cache:
            cache[key] = frame[list(key)].max(axis=1).to_numpy(dtype=float)
        result += cache[key]
    return result


def logrank(frame, high, outcome, event):
    if not np.any(high) or np.all(high):
        return float("nan"), float("nan")
    result = logrank_test(frame.loc[~high, outcome], frame.loc[high, outcome],
                          event_observed_A=frame.loc[~high, event], event_observed_B=frame.loc[high, event])
    return float(result.test_statistic), float(result.p_value)


def optimize_threshold(frame, score, original_threshold, config):
    best = None
    for threshold in np.unique(score)[:-1]:
        if not config["fibers"]["min_thresh"] <= threshold <= config["fibers"]["max_thresh"]:
            continue
        high = score > threshold
        if min(high.mean(), 1 - high.mean()) < config["evaluation"]["minimum_group_fraction"]:
            continue
        statistic, _ = logrank(frame, high, config["columns"]["outcome"], config["columns"]["event"])
        if np.isfinite(statistic):
            candidate = (statistic, -abs(threshold - original_threshold), float(threshold))
            if best is None or candidate > best:
                best = candidate
    return best[2] if best else None


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


def cox_designs(frame, config, clinical, antigen):
    excluded = set(config["evaluation"].get("exclude_covariates", []))
    return {label: covariate_basis(frame, [c for c in covs if c not in excluded])
            for label, covs in (("HR", []), ("Adj HR", clinical + antigen), ("Adj NoAg HR", clinical))}


def evaluate(frame, score, threshold, config, clinical, antigen, unadjusted, adjusted, designs=None):
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    high = np.asarray(score) > threshold
    statistic, p = logrank(frame, high, outcome, event)
    result = {"threshold": float(threshold), "n": len(frame), "n_above": int(high.sum()),
              "logrank": statistic, "logrank_p": p}
    if (unadjusted or adjusted) and designs is None:
        designs = cox_designs(frame, config, clinical, antigen)
    for label, requested in (("HR", unadjusted), ("Adj HR", adjusted), ("Adj NoAg HR", adjusted)):
        for suffix in ("", " lower", " upper", " p"):
            result[label + suffix] = float("nan")
        result[label + " status"] = "not_requested"
        if not requested:
            continue
        if not high.any() or high.all():
            result[label + " status"] = "single_group"
            continue
        fitted, _, _ = fit_noag(frame, high, outcome, event, designs[label][0], label=label)
        for suffix in ("", " lower", " upper", " p"):
            result[label + suffix] = fitted[label + suffix] if fitted[label + suffix] is not None else float("nan")
        result[label + " status"] = fitted["status"]
    return result


def jaccard(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else float("nan")
