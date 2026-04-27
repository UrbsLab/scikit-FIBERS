import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time as lifelines_restricted_mean_survival_time

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


def restricted_mean_survival_time(outcome, censor, time_point):
    kmf = KaplanMeierFitter()
    kmf.fit(outcome, event_observed=censor)
    return float(lifelines_restricted_mean_survival_time(kmf, t=time_point))


def make_manual_population():
    feature_order = ["F_protect", "F_wrong"] + [f"Noise_{i}" for i in range(8)]
    rows = []
    for feature_name in feature_order:
        rows.append([str([feature_name]), 0, None, None, None, None, None, None, None, None, 0])
    return pd.DataFrame(rows, columns=MANUAL_BIN_COLUMNS)


def make_directional_dataset():
    rng = np.random.default_rng(42)
    rows = []

    for i in range(40):
        row = {
            "F_protect": 1,
            "F_wrong": 0,
            "Duration": 12.0 + (i * 0.05),
            "Censoring": 0 if i % 6 == 0 else 1,
        }
        for j in range(8):
            row[f"Noise_{j}"] = int(rng.integers(0, 2))
        rows.append(row)

    for i in range(40):
        row = {
            "F_protect": 0,
            "F_wrong": 1,
            "Duration": 3.0 + (i * 0.05),
            "Censoring": 0 if i % 5 == 0 else 1,
        }
        for j in range(8):
            row[f"Noise_{j}"] = int(rng.integers(0, 2))
        rows.append(row)

    return pd.DataFrame(rows)


def make_base_fibers_kwargs(manual_bin_init):
    return {
        "outcome_label": "Duration",
        "outcome_type": "survival",
        "iterations": 0,
        "pop_size": 10,
        "tournament_prop": 0.5,
        "crossover_prob": 0.5,
        "min_mutation_prob": 0.1,
        "max_mutation_prob": 0.1,
        "merge_prob": 0.0,
        "new_gen": 1.0,
        "elitism": 0.1,
        "diversity_pressure": 0,
        "min_bin_size": 1,
        "max_bin_size": 1,
        "max_bin_init_size": 1,
        "fitness_metric": "log_rank",
        "log_rank_weighting": None,
        "censor_label": "Censoring",
        "group_strata_min": 0.2,
        "penalty": 0.5,
        "group_thresh": 0,
        "min_thresh": 0,
        "max_thresh": 1,
        "int_thresh": True,
        "thresh_evolve_prob": 0.5,
        "manual_bin_init": manual_bin_init,
        "covariates": None,
        "pop_clean": None,
        "report": None,
        "random_seed": 7,
        "verbose": False,
    }


def get_pop_row_for_feature(model, feature_name):
    pop_df = model.get_pop()
    return pop_df[pop_df["feature_list"].apply(lambda value: value == [feature_name])].iloc[0]


def test_omitted_default_matches_explicit_default():
    data = make_directional_dataset()
    manual_bin_init = make_manual_population()

    omitted_default = FIBERS(**make_base_fibers_kwargs(manual_bin_init)).fit(data)
    explicit_default = FIBERS(
        **make_base_fibers_kwargs(manual_bin_init),
        desired_bin_effect="default",
    ).fit(data)

    compared_columns = [
        "feature_list",
        "group_threshold",
        "pre_fitness",
        "fitness",
        "log_rank_score",
        "count_bt",
        "count_at",
    ]
    pd.testing.assert_frame_equal(
        omitted_default.get_pop()[compared_columns].reset_index(drop=True),
        explicit_default.get_pop()[compared_columns].reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        omitted_default.predict(data, bin_number=0),
        explicit_default.predict(data, bin_number=0),
    )
    pd.testing.assert_series_equal(
        omitted_default.transform(data, full_sums=False)["Bin_0"],
        explicit_default.transform(data, full_sums=False)["Bin_0"],
        check_names=False,
    )


def test_protective_mode_filters_wrong_direction_bins_and_encodes_presence():
    data = make_directional_dataset()
    manual_bin_init = make_manual_population()
    model = FIBERS(
        **make_base_fibers_kwargs(manual_bin_init),
        desired_bin_effect="protective",
    ).fit(data)

    top_bin = model.set.bin_pop[0]
    assert top_bin.feature_list == ["F_protect"]

    low_outcome, high_outcome, low_censor, high_censor = model.get_bin_groups(data, 0)
    time_point = min(max(low_outcome), max(high_outcome))
    assert restricted_mean_survival_time(high_outcome, high_censor, time_point) > restricted_mean_survival_time(low_outcome, low_censor, time_point)

    wrong_direction_bin = get_pop_row_for_feature(model, "F_wrong")
    assert wrong_direction_bin["pre_fitness"] == 0.0

    expected_protective = (data["F_protect"] > top_bin.group_threshold).astype(int)
    np.testing.assert_array_equal(model.predict(data, bin_number=0), expected_protective.to_numpy())
    pd.testing.assert_series_equal(
        model.transform(data, full_sums=False)["Bin_0"],
        expected_protective,
        check_names=False,
    )


def test_high_risk_mode_filters_wrong_direction_bins_and_preserves_default_encoding():
    data = make_directional_dataset()
    manual_bin_init = make_manual_population()
    model = FIBERS(
        **make_base_fibers_kwargs(manual_bin_init),
        desired_bin_effect="high_risk",
    ).fit(data)

    top_bin = model.set.bin_pop[0]
    assert top_bin.feature_list == ["F_wrong"]

    low_outcome, high_outcome, low_censor, high_censor = model.get_bin_groups(data, 0)
    time_point = min(max(low_outcome), max(high_outcome))
    assert restricted_mean_survival_time(high_outcome, high_censor, time_point) < restricted_mean_survival_time(low_outcome, low_censor, time_point)

    wrong_direction_bin = get_pop_row_for_feature(model, "F_protect")
    assert wrong_direction_bin["pre_fitness"] == 0.0

    expected_high_risk = (data["F_wrong"] > top_bin.group_threshold).astype(int)
    np.testing.assert_array_equal(model.predict(data, bin_number=0), expected_high_risk.to_numpy())
    pd.testing.assert_series_equal(
        model.transform(data, full_sums=False)["Bin_0"],
        expected_high_risk,
        check_names=False,
    )


def test_permissive_mode_preserves_default_thresholding_flips_binary_encoding_and_keeps_cox_coding():
    data = make_directional_dataset()
    manual_bin_init = make_manual_population()

    default_model = FIBERS(
        **make_base_fibers_kwargs(manual_bin_init),
        desired_bin_effect="default",
    ).fit(data)
    permissive_model = FIBERS(
        **make_base_fibers_kwargs(manual_bin_init),
        desired_bin_effect="permissive",
    ).fit(data)

    compared_columns = [
        "feature_list",
        "group_threshold",
        "pre_fitness",
        "fitness",
        "log_rank_score",
        "count_bt",
        "count_at",
    ]
    pd.testing.assert_frame_equal(
        default_model.get_pop()[compared_columns].reset_index(drop=True),
        permissive_model.get_pop()[compared_columns].reset_index(drop=True),
    )

    top_bin = permissive_model.set.bin_pop[0]
    feature_sums = data[top_bin.feature_list].sum(axis=1)
    expected_default = (feature_sums > top_bin.group_threshold).astype(int)
    expected_permissive = (feature_sums <= top_bin.group_threshold).astype(int)

    np.testing.assert_array_equal(default_model.predict(data, bin_number=0), expected_default.to_numpy())
    np.testing.assert_array_equal(permissive_model.predict(data, bin_number=0), expected_permissive.to_numpy())
    pd.testing.assert_series_equal(
        default_model.transform(data, full_sums=False)["Bin_0"],
        expected_default,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        permissive_model.transform(data, full_sums=False)["Bin_0"],
        expected_permissive,
        check_names=False,
    )
    assert not np.array_equal(
        default_model.predict(data, bin_number=0),
        permissive_model.predict(data, bin_number=0),
    )

    default_hr_summary = default_model.get_cox_prop_hazard_unadjust(data, bin_index=0)
    permissive_hr_summary = permissive_model.get_cox_prop_hazard_unadjust(data, bin_index=0)
    assert np.isclose(
        default_hr_summary["exp(coef)"].iloc[0],
        permissive_hr_summary["exp(coef)"].iloc[0],
    )


def test_adaptive_thresholding_respects_default_protective_high_risk_and_permissive_modes():
    feature_df = pd.DataFrame(
        {
            "F": ([0] * 200) + ([1] * 300) + ([2] * 20),
        }
    )
    outcome_df = pd.DataFrame(
        {
            "Duration": (
                [1.0 + (i * 0.01) for i in range(200)]
                + [5.0 + (i * 0.01) for i in range(300)]
                + [0.5 + (i * 0.01) for i in range(20)]
            ),
        }
    )
    censor_df = pd.DataFrame({"Censoring": [1] * len(feature_df)})
    covariate_df = pd.DataFrame(index=feature_df.index)

    bin_df = pd.concat(
        [pd.DataFrame({"feature_sum": feature_df["F"]}), outcome_df, censor_df],
        axis=1,
    )

    default_score_threshold_0 = BIN().evaluate_for_threshold(
        0,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "default",
    )[0]
    default_score_threshold_1 = BIN().evaluate_for_threshold(
        1,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "default",
    )[0]
    protective_score_threshold_0 = BIN().evaluate_for_threshold(
        0,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "protective",
    )[0]
    protective_score_threshold_1 = BIN().evaluate_for_threshold(
        1,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "protective",
    )[0]
    high_risk_score_threshold_0 = BIN().evaluate_for_threshold(
        0,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "high_risk",
    )[0]
    high_risk_score_threshold_1 = BIN().evaluate_for_threshold(
        1,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "high_risk",
    )[0]
    permissive_score_threshold_0 = BIN().evaluate_for_threshold(
        0,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "permissive",
    )[0]
    permissive_score_threshold_1 = BIN().evaluate_for_threshold(
        1,
        bin_df,
        "Duration",
        "Censoring",
        "survival",
        "log_rank",
        None,
        None,
        covariate_df,
        "permissive",
    )[0]

    assert default_score_threshold_1 > default_score_threshold_0
    assert permissive_score_threshold_1 == default_score_threshold_1
    assert permissive_score_threshold_0 == default_score_threshold_0
    assert protective_score_threshold_0 > 0
    assert protective_score_threshold_1 == 0
    assert high_risk_score_threshold_0 == 0
    assert high_risk_score_threshold_1 > 0

    default_bin = BIN()
    default_bin.feature_list = ["F"]
    default_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "default",
        0.2,
    )
    assert default_bin.group_threshold == 1

    permissive_bin = BIN()
    permissive_bin.feature_list = ["F"]
    permissive_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "permissive",
        0.2,
    )
    assert permissive_bin.group_threshold == 1

    protective_bin = BIN()
    protective_bin.feature_list = ["F"]
    protective_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "protective",
        0.2,
    )
    assert protective_bin.group_threshold == 0

    high_risk_bin = BIN()
    high_risk_bin.feature_list = ["F"]
    high_risk_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "high_risk",
        0.2,
    )
    assert high_risk_bin.group_threshold == 0
    assert high_risk_bin.log_rank_score == 0


def test_protective_adaptive_threshold_skips_directionally_valid_thresholds_that_fail_group_strata_min():
    feature_df = pd.DataFrame(
        {
            "F": ([0] * 200) + ([1] * 300) + ([2] * 20),
        }
    )
    outcome_df = pd.DataFrame(
        {
            "Duration": (
                [5.0 for _ in range(200)]
                + [5.0 for _ in range(300)]
                + [100.0 for _ in range(20)]
            ),
        }
    )
    censor_df = pd.DataFrame({"Censoring": [1] * len(feature_df)})
    covariate_df = pd.DataFrame(index=feature_df.index)

    default_bin = BIN()
    default_bin.feature_list = ["F"]
    default_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "default",
        0.2,
    )

    protective_bin = BIN()
    protective_bin.feature_list = ["F"]
    protective_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "protective",
        0.2,
    )

    assert default_bin.group_threshold == 1
    assert protective_bin.group_threshold == 0


def test_protective_all_wrong_direction_thresholds_keep_raw_best_threshold_with_zero_score():
    feature_df = pd.DataFrame(
        {
            "F": ([0] * 120) + ([1] * 120) + ([2] * 120),
        }
    )
    outcome_df = pd.DataFrame(
        {
            "Duration": (
                [10.0 + (i * 0.01) for i in range(120)]
                + [8.0 + (i * 0.01) for i in range(120)]
                + [1.0 + (i * 0.01) for i in range(120)]
            ),
        }
    )
    censor_df = pd.DataFrame({"Censoring": [1] * len(feature_df)})
    covariate_df = pd.DataFrame(index=feature_df.index)

    default_bin = BIN()
    default_bin.feature_list = ["F"]
    default_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "default",
        0.2,
    )

    protective_bin = BIN()
    protective_bin.feature_list = ["F"]
    protective_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "protective",
        0.2,
    )
    protective_bin.calculate_pre_fitness(0.2, 0.5, "log_rank", ["F"])

    assert default_bin.group_threshold == 1
    assert protective_bin.group_threshold == default_bin.group_threshold
    assert protective_bin.log_rank_score == 0
    assert protective_bin.pre_fitness == 0
    assert protective_bin.count_bt == default_bin.count_bt
    assert protective_bin.count_at == default_bin.count_at


def test_high_risk_all_wrong_direction_thresholds_keep_raw_best_threshold_with_zero_score():
    feature_df = pd.DataFrame(
        {
            "F": ([0] * 120) + ([1] * 120) + ([2] * 120),
        }
    )
    outcome_df = pd.DataFrame(
        {
            "Duration": (
                [1.0 + (i * 0.01) for i in range(120)]
                + [8.0 + (i * 0.01) for i in range(120)]
                + [10.0 + (i * 0.01) for i in range(120)]
            ),
        }
    )
    censor_df = pd.DataFrame({"Censoring": [1] * len(feature_df)})
    covariate_df = pd.DataFrame(index=feature_df.index)

    default_bin = BIN()
    default_bin.feature_list = ["F"]
    default_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "default",
        0.2,
    )

    high_risk_bin = BIN()
    high_risk_bin.feature_list = ["F"]
    high_risk_bin.evaluate(
        feature_df,
        outcome_df,
        censor_df,
        "survival",
        "log_rank",
        None,
        "Duration",
        "Censoring",
        0,
        1,
        True,
        None,
        False,
        1,
        0,
        None,
        covariate_df,
        "high_risk",
        0.2,
    )
    high_risk_bin.calculate_pre_fitness(0.2, 0.5, "log_rank", ["F"])

    assert high_risk_bin.group_threshold == default_bin.group_threshold
    assert high_risk_bin.log_rank_score == 0
    assert high_risk_bin.pre_fitness == 0
    assert high_risk_bin.count_bt == default_bin.count_bt
    assert high_risk_bin.count_at == default_bin.count_at
