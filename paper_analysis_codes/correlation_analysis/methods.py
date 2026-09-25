"""Pearson blocks, bin scoring and survival metrics used by the three workers."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
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


def scheme_name(threshold):
    return "r" + str(float(threshold)).replace(".", "p")


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
    overrides = settings.get("locus_thresholds", {})
    if overrides:
        add("locus_specific", {locus: overrides.get(locus, settings["primary_threshold"]) for locus in loci})
    adaptive = settings["adaptive"]
    if adaptive["enabled"]:
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


def evaluate(frame, score, threshold, config, clinical, antigen, unadjusted, adjusted):
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    high = np.asarray(score) > threshold
    statistic, p = logrank(frame, high, outcome, event)
    result = {"threshold": float(threshold), "n": len(frame), "n_above": int(high.sum()),
              "logrank": statistic, "logrank_p": p}
    for label, covs, requested in (("HR", [], unadjusted), ("Adj HR", clinical + antigen, adjusted),
                                   ("Adj NoAg HR", clinical, adjusted)):
        for suffix in ("", " lower", " upper", " p"):
            result[label + suffix] = float("nan")
        result[label + " status"] = "not_requested"
        if not requested:
            continue
        if not high.any() or high.all():
            result[label + " status"] = "single_group"
            continue
        model_frame = frame[[outcome, event] + covs].copy()
        model_frame["_above"] = high.astype(int)
        try:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                model = CoxPHFitter().fit(model_frame, duration_col=outcome, event_col=event)
            row = model.summary.loc["_above"]
            for suffix, field in (("", "exp(coef)"), (" lower", "exp(coef) lower 95%"),
                                  (" upper", "exp(coef) upper 95%"), (" p", "p")):
                result[label + suffix] = float(row[field])
            result[label + " status"] = "ok" if not messages else "warning:" + " | ".join(str(w.message) for w in messages)
        except Exception as error:
            result[label + " status"] = f"error:{type(error).__name__}:{error}"
    return result


def jaccard(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else float("nan")
