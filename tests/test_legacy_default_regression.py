import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.skfibers.fibers import FIBERS


def make_legacy_regression_data():
    rng = np.random.default_rng(20260923)
    features = rng.binomial(2, 0.28, size=(120, 12))
    signal = features[:, 0] + features[:, 1] + features[:, 2]
    duration = 4.0 + (5 - signal) * 1.7 + rng.uniform(0.1, 1.5, size=120)
    censoring = (rng.random(120) > 0.18).astype(int)
    data = pd.DataFrame(features, columns=[f"F{i}" for i in range(features.shape[1])])
    data["Duration"] = duration
    data["Censoring"] = censoring
    return data


def test_non_multi_default_matches_5ff67da_golden_run():
    """Lock the legacy default search to commit 5ff67da's seeded behavior."""
    data = make_legacy_regression_data()
    model = FIBERS(
        iterations=4,
        pop_size=10,
        tournament_prop=0.4,
        crossover_prob=0.5,
        min_mutation_prob=0.1,
        max_mutation_prob=0.3,
        merge_prob=0.1,
        new_gen=1.0,
        elitism=0.1,
        diversity_pressure=0,
        min_bin_size=1,
        max_bin_size=6,
        max_bin_init_size=4,
        fitness_metric="log_rank",
        group_strata_min=0.2,
        penalty=0.5,
        group_thresh=None,
        min_thresh=0,
        max_thresh=4,
        thresh_evolve_prob=0.5,
        random_seed=17,
        verbose=False,
        desired_bin_effect="default",
        multi_thresholding=False,
    ).fit(data)

    structure = [
        [
            sorted(bin_obj.feature_list),
            int(bin_obj.group_threshold),
            int(bin_obj.birth_iteration),
            int(bin_obj.count_bt),
            int(bin_obj.count_at),
        ]
        for bin_obj in model.set.bin_pop
    ]
    structure_hash = hashlib.sha256(
        json.dumps(structure, separators=(",", ":")).encode()
    ).hexdigest()
    assert structure_hash == "7831b787e467b15488c54bd45a851b0148a3bd2c12d21f80e1045fb154ffd6ca"

    assert [bin_obj.pre_fitness for bin_obj in model.set.bin_pop] == pytest.approx([
        151.315381380608,
        107.934247905643,
        84.508852464526,
        70.197368012714,
        69.279948959104,
        59.706100809778,
        53.534119706592,
        49.146806558383,
        38.478449340531,
        38.067221864982,
    ])

    predictions = model.predict(data, bin_number=0).astype(np.uint8)
    transformed = model.transform(data).iloc[:, 0].to_numpy(dtype=np.uint8)
    expected_hash = "7f21db864a91f25f4e40c52c979b0d6a50c0febfaeed4b8bf99c99bba62c0c57"
    assert hashlib.sha256(predictions.tobytes()).hexdigest() == expected_hash
    assert hashlib.sha256(transformed.tobytes()).hexdigest() == expected_hash
