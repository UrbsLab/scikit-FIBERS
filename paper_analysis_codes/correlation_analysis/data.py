"""Read only the configured columns, with checks performed on the compute node."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from common import covariates, input_paths


def feature_key(name):
    match = re.fullmatch(r"MM_([A-Za-z0-9]+)_(\d+)", name)
    return (match[1], int(match[2])) if match else ("", -1)


def selected_features(header, config):
    ranges = config["columns"]["ranges"]
    features = []
    for name in header:
        locus, position = feature_key(name)
        if locus in ranges and ranges[locus][0] <= position <= ranges[locus][1]:
            features.append(name)
    order = {locus: i for i, locus in enumerate(ranges)}
    return sorted(features, key=lambda name: (order[feature_key(name)[0]], feature_key(name)[1]))


def id_digest(values):
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def read_data(path, config):
    header = list(pd.read_csv(path, nrows=0).columns)
    features = selected_features(header, config)
    if not features:
        raise ValueError(f"No configured mismatch features in {path}")
    clinical, antigen = covariates(config)
    identifier = config["input"]["id_column"]
    outcome, event = config["columns"]["outcome"], config["columns"]["event"]
    numeric = list(dict.fromkeys(features + clinical + antigen + [outcome, event]))
    required = [identifier] + numeric
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frames = []
    reader = pd.read_csv(path, usecols=required, dtype={identifier: "string"},
                         chunksize=config["input"]["chunksize"], low_memory=True)
    with reader:
        for chunk in reader:
            values = chunk[numeric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Missing or infinite numeric values in {path}")
            # Numeric strings must not survive into the model as categorical data.
            chunk[numeric] = values
            if (chunk[outcome] < 0).any() or not chunk[event].isin([0, 1]).all():
                raise ValueError(f"Invalid survival duration or 0/1 event indicator in {path}")
            mismatch = chunk[features].to_numpy()
            if not np.isin(mismatch, [0, 1, 2]).all():
                raise ValueError(f"Expected mismatch counts 0, 1 or 2 in {path}")
            chunk[identifier] = chunk[identifier].str.strip()
            if chunk[identifier].isna().any() or chunk[identifier].eq("").any():
                raise ValueError(f"Missing {identifier} values in {path}")
            frames.append(chunk)
    if not frames:
        raise ValueError(f"Empty input: {path}")
    frame = pd.concat(frames, ignore_index=True)
    if frame[identifier].duplicated().any():
        raise ValueError(f"Duplicate {identifier} values in {path}")
    print(f"Read {len(frame):,} rows, {len(features)} candidate features: {path}", flush=True)
    return frame, features


def load_fold(config, imputation, fold):
    paths = input_paths(config, imputation, fold)
    identifier = config["input"]["id_column"]
    train, features = read_data(paths[0], config)
    if config["input"]["mode"] == "pre_split":
        test, test_features = read_data(paths[1], config)
        if test_features != features:
            raise ValueError("Train/test mismatch feature schemas differ")
    else:
        ids = np.sort(train[identifier].to_numpy(dtype=str))
        rng = np.random.default_rng(config["input"]["split_seed"])
        held_out = np.array_split(rng.permutation(ids), len(config["folds"]))[fold - 1]
        mask = train[identifier].isin(held_out)
        test = train.loc[mask].reset_index(drop=True)
        train = train.loc[~mask].reset_index(drop=True)
    train_ids, test_ids = set(train[identifier]), set(test[identifier])
    if train_ids & test_ids:
        raise ValueError(f"Train/test {identifier} overlap in imputation {imputation}, fold {fold}")
    if train.empty or test.empty:
        raise ValueError("Both train and held-out test sets must contain rows")
    audit = {"train_n": len(train), "test_n": len(test),
             "train_ids_sha256": id_digest(train_ids), "test_ids_sha256": id_digest(test_ids),
             "cohort_ids_sha256": id_digest(train_ids | test_ids)}
    return train, test, features, audit


def retained_features(train, features, config):
    variance = train[features].var(ddof=0)
    frequency = (train[features] > 0).mean()
    keep = (variance > 0) & (frequency >= config["minimum_nonzero_frequency"])
    kept = [name for name in features if keep[name]]
    if not kept:
        raise ValueError("No features remain after training-only filtering")
    table = pd.DataFrame({"feature": features, "variance": variance.to_numpy(),
                          "nonzero_frequency": frequency.to_numpy(), "retained": keep.to_numpy()})
    return kept, table


def active_covariates(train, config):
    clinical, antigen = covariates(config)
    keep = [name for name in clinical + antigen if train[name].nunique() > 1]
    return [name for name in clinical if name in keep], [name for name in antigen if name in keep]
