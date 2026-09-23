import random

import numpy as np
import pandas as pd
import pytest
from lifelines.statistics import logrank_test

from src.skfibers.fibers import FIBERS
from src.skfibers.methods.bin import BIN


MANUAL_BIN_COLUMNS = [
    "feature_list",
    "group_threshold",
    "fitness",
    "pre_fitness",
    "log_rank_score",
    "log_rank_p_value",
    "bin_size",
    "group_strata_prop",
    "count_bt",
    "count_at",
    "birth_iteration",
]


def make_three_group_dataset(desired_effect="high_risk"):
    rows = []
    duration_starts = [14.0, 8.0, 2.0] if desired_effect == "high_risk" else [2.0, 8.0, 14.0]
    for group, duration_start in enumerate(duration_starts):
        for index in range(30):
            row = {
                "Duration": duration_start + index * 0.03,
                "Censoring": 0 if index % 7 == 0 else 1,
            }
            for feature_index in range(10):
                row[f"F{feature_index}"] = group
            rows.append(row)
    return pd.DataFrame(rows)


def make_manual_population(thresholds):
    rows = []
    for feature_index in range(10):
        rows.append([
            str([f"F{feature_index}"]), list(thresholds), None, None, None,
            None, None, None, None, None, 0,
        ])
    return pd.DataFrame(rows, columns=MANUAL_BIN_COLUMNS)


def make_multi_model(**overrides):
    parameters = {
        "iterations": 0,
        "pop_size": 10,
        "diversity_pressure": 0,
        "min_bin_size": 1,
        "max_bin_size": 1,
        "max_bin_init_size": 1,
        "group_thresh": None,
        "min_thresh": 0,
        "max_thresh": 2,
        "multi_thresholding": True,
        "group_thresh_list": [0, 1],
        "manual_bin_init": make_manual_population([0, 1]),
        "random_seed": 11,
    }
    parameters.update(overrides)
    return FIBERS(**parameters)


def test_multi_group_fixed_thresholds_score_and_encode_three_groups():
    data = make_three_group_dataset()
    model = make_multi_model().fit(data)
    top_bin = model.set.bin_pop[0]

    assert top_bin.group_threshold_list == [0, 1]
    assert (top_bin.count_bt, top_bin.count_mt, top_bin.count_at) == (30, 30, 30)
    assert top_bin.group_prop_list == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    low = data[data["F0"] == 0]
    middle = data[data["F0"] == 1]
    high = data[data["F0"] == 2]
    expected_scores = [
        logrank_test(low["Duration"], high["Duration"], low["Censoring"], high["Censoring"]).test_statistic,
        logrank_test(low["Duration"], middle["Duration"], low["Censoring"], middle["Censoring"]).test_statistic,
        logrank_test(middle["Duration"], high["Duration"], middle["Censoring"], high["Censoring"]).test_statistic,
    ]
    assert top_bin.log_rank_score == pytest.approx(np.mean(expected_scores))
    assert top_bin.pairwise_scores == pytest.approx([round(score, 3) for score in expected_scores])

    np.testing.assert_array_equal(model.predict(data, bin_number=0), np.repeat([0, 1, 2], 30))
    np.testing.assert_array_equal(model.transform(data)["Bin_0"], np.repeat([0, 1, 2], 30))


def test_multi_group_helpers_return_all_three_strata_without_changing_legacy_helper_shape():
    data = make_three_group_dataset()
    model = make_multi_model().fit(data)

    groups = model.get_multi_bin_groups(data, bin_index=0)
    assert [len(group) for group in groups] == [30, 30, 30, 30, 30, 30]
    with pytest.raises(ValueError, match="get_multi_bin_groups"):
        model.get_bin_groups(data, bin_index=0)


def test_multi_group_threshold_operators_keep_one_or_two_unique_ordered_thresholds():
    bin_obj = BIN()
    bin_obj.feature_list = ["F0"]
    bin_obj.set_thresholds([0])
    rng = random.Random(4)

    for _ in range(20):
        bin_obj.mutation(
            1.0, ["F0", "F1"], 1, 2, 1, True, 0, 3, rng,
            multi_thresholding=True,
        )
        assert 1 <= len(bin_obj.group_threshold_list) <= 2
        assert bin_obj.group_threshold_list == sorted(set(bin_obj.group_threshold_list))
        assert all(0 <= threshold <= 3 for threshold in bin_obj.group_threshold_list)


def test_adaptive_multi_group_search_evaluates_single_thresholds_and_pairs():
    bin_obj = BIN()
    seen = []

    def fake_evaluation(thresholds, *args):
        seen.append(list(thresholds))
        score = 10.0 if len(thresholds) == 2 else 1.0
        middle_count = 1 if len(thresholds) == 2 else 0
        proportions = [1 / 3, 1 / 3, 1 / 3] if len(thresholds) == 2 else [0.5, 0.5]
        pairwise = [10.0, 10.0, 10.0] if len(thresholds) == 2 else []
        return score, 0.01, None, None, 1, middle_count, 1, pairwise, proportions, True

    bin_obj.evaluate_for_thresholds = fake_evaluation
    bin_obj.evaluate_multi_thresholds(
        pd.DataFrame(), "Duration", "Censoring", "survival", "log_rank",
        None, 0, 2, None, False, 5, 0, None, pd.DataFrame(), "default", 0.2,
    )

    assert seen == [[0, 1], [0, 2], [1, 2], [0], [1], [2]]
    assert bin_obj.group_threshold_list == [0, 1]


@pytest.mark.parametrize("desired_effect", ["protective", "high_risk"])
def test_adaptive_multi_group_search_prefers_directionally_valid_configuration(desired_effect):
    bin_obj = BIN()

    def fake_evaluation(thresholds, *args):
        evaluated_effect = args[-1]
        raw_score = 12.0 if thresholds == [2] else 8.0
        directionally_valid = thresholds == [0, 1]
        score = raw_score if evaluated_effect == "default" or directionally_valid else 0.0
        middle_count = 20 if len(thresholds) == 2 else 0
        proportions = [1 / 3, 1 / 3, 1 / 3] if len(thresholds) == 2 else [0.5, 0.5]
        pairwise = [score, score, score] if len(thresholds) == 2 else []
        return score, 0.01, None, None, 20, middle_count, 20, pairwise, proportions, directionally_valid

    bin_obj.evaluate_for_thresholds = fake_evaluation
    bin_obj.evaluate_multi_thresholds(
        pd.DataFrame(), "Duration", "Censoring", "survival", "log_rank",
        None, 0, 2, None, False, 5, 0, None, pd.DataFrame(), desired_effect, 0.2,
    )

    assert bin_obj.group_threshold_list == [0, 1]
    assert bin_obj.log_rank_score == 8.0
    assert bin_obj.used_group_strata_fallback is False


@pytest.mark.parametrize("desired_effect", ["protective", "high_risk"])
def test_multi_group_applies_directional_order_across_three_strata(desired_effect):
    matching_data = make_three_group_dataset(desired_effect)
    matching_model = make_multi_model(desired_bin_effect=desired_effect).fit(matching_data)
    assert matching_model.set.bin_pop[0].log_rank_score > 0

    opposite_effect = "protective" if desired_effect == "high_risk" else "high_risk"
    nonmatching_model = make_multi_model(desired_bin_effect=opposite_effect).fit(matching_data)
    assert nonmatching_model.set.bin_pop[0].log_rank_score == 0
    assert nonmatching_model.set.bin_pop[0].pre_fitness == 0


def test_multi_group_is_opt_in():
    default_model = FIBERS()
    explicit_legacy_model = FIBERS(multi_thresholding=False)
    assert default_model.get_params()["multi_thresholding"] is False
    assert explicit_legacy_model.get_params()["multi_thresholding"] is False

    with pytest.raises(Exception, match="group_thresh_list"):
        FIBERS(group_thresh_list=[0, 1])
