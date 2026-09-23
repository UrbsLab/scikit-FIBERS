# Protective and high-risk modes

The *desired_bin_effect* parameter lets FIBERS restrict its evolutionary search to bins whose above-threshold group has a requested survival direction. The direction check is applied while candidate bins are evaluated; it is not merely a label assigned after training.

With multi-group thresholding, the same rule is extended to a strict monotonic ordering across the low-, middle-, and high-burden groups. Protective requires increasing RMST as burden increases; high-risk requires decreasing RMST. Each adjacent pair uses its own shared follow-up horizon. See [Two- and three-group thresholding](multi_group.md) for the three-group boundaries and adaptive-selection behavior.

The supported values are:

| Mode | Required survival direction for the above-threshold group | Meaning of binary output `1` |
| ---- | --------------------------------------------------------- | ---------------------------- |
| `default` | No direction is required | Above-threshold membership only |
| `protective` | Better survival than the below-threshold group | Protective-burden membership |
| `high_risk` | Worse survival than the below-threshold group | High-risk-burden membership |

## Forming the two threshold groups

For a candidate bin, FIBERS first sums the values of the features included in that bin for each sample. Given a candidate threshold, the samples are separated as follows:

* **Below-threshold group:** bin sum <= threshold.
* **Above-threshold group:** bin sum > threshold.

The terms *above threshold* and *below threshold* describe burden membership only. They should not automatically be interpreted as high and low risk because the above-threshold group may be protective.

## Censoring-aware direction test

For `protective` and `high_risk`, FIBERS fits a Kaplan-Meier survival estimate for each threshold group. It then chooses a shared evaluation horizon:

```
shared_time = min(maximum below-threshold duration,
                  maximum above-threshold duration)
```

The restricted mean survival time (RMST) for each group is calculated from time zero through `shared_time`. Restricting the comparison to this shared horizon avoids comparing one group beyond the follow-up supported by the other group and allows censored samples to contribute through the Kaplan-Meier estimate.

The direction rules use strict inequalities:

* `protective` is valid when `above_RMST > below_RMST`.
* `high_risk` is valid when `above_RMST < below_RMST`.

Equal RMST values do not satisfy either directional mode. A threshold is also treated as directionally invalid if a group is empty or its RMST cannot be calculated.

## Direction is a gate, not the fitness score

RMST determines whether a candidate has the requested direction, but the size of the RMST difference is not itself the optimization score. Candidates that pass the direction gate continue to be ranked using the configured *fitness_metric*:

| Fitness metric | Score for a directionally valid candidate | Score for a wrong-direction candidate |
| -------------- | ----------------------------------------- | --------------------------------------- |
| `log_rank` | Log-rank test statistic | `0` |
| `residuals` | Absolute rank-sum statistic for deviance residuals | `0` |
| `log_rank_residuals` | Log-rank score multiplied by the residual score | `0` |

The associated p-value is unavailable for a score that was set to zero by the directional gate. A candidate can satisfy the RMST direction with a small difference, so directionally valid does not by itself mean statistically significant. Statistical separation is still represented by the selected fitness statistic and its associated output.

The RMST direction check uses the observed survival and censoring columns and is therefore an unadjusted direction check. If covariates are supplied, `residuals` or `log_rank_residuals` can incorporate covariate-adjusted deviance residuals into candidate ranking, but the protective/high-risk gate itself remains based on the two Kaplan-Meier RMST estimates.

## Fixed and adaptive thresholds

When *group_thresh* is a fixed integer, FIBERS evaluates that threshold directly. If the resulting groups do not have the requested direction, the applicable fitness score is zero.

When `group_thresh=None`, FIBERS evaluates the candidate thresholds from *min_thresh* through *max_thresh*. Directional modes use the following selection order:

1. Select the highest-scoring threshold that satisfies both the requested RMST direction and *group_strata_min*.
2. If none satisfies both requirements, prefer the highest-scoring directionally valid threshold even when its groups are too imbalanced. The regular group-balance penalty is applied, followed by an additional fallback penalty.
3. If no threshold has the requested direction, retain the best available fallback for algorithm continuity. Its directional fitness remains zero.

With the default penalty calculation, a directionally valid fallback that misses *group_strata_min* can receive two multiplicative `(1 - penalty)` reductions: one for group imbalance and one for requiring the fallback.

This threshold logic keeps the requested biological direction as the priority while still allowing the evolutionary process to continue when a candidate cannot simultaneously satisfy the direction and balance constraints.

## Where the direction constraint is applied

The selected mode is used throughout training:

* When random bins are initialized.
* When a manually supplied starting population is evaluated.
* When crossover, mutation, or merge operations create offspring.
* When adaptive thresholds are re-evaluated, including the final iteration.

As a result, `protective` and `high_risk` influence which candidates receive useful fitness throughout evolution rather than filtering only the final population.

## Prediction and transformation semantics

FIBERS does not reverse its binary encoding in protective mode:

```
0 = bin sum <= threshold
1 = bin sum > threshold
```

Therefore:

* In `protective` mode, a prediction of `1` means membership in the group with better censoring-aware survival during training.
* In `high_risk` mode, a prediction of `1` means membership in the group with worse censoring-aware survival during training.
* In `default` mode, a prediction of `1` means above-threshold membership without a guaranteed survival direction.

The same convention is used by `transform(..., full_sums=False)`. When `full_sums=True`, the transformed feature contains the raw bin sum instead of a binary group assignment.

## Worked example

Suppose a candidate bin contains three features and uses a threshold of `1`. Samples with zero or one included feature value are below threshold, while samples with a bin sum of two or more are above threshold.

If the Kaplan-Meier estimates produce the following RMST values over their shared follow-up horizon:

| Group | RMST |
| ----- | ---- |
| Below threshold | 5.7 years |
| Above threshold | 8.4 years |

the candidate passes `protective` because `8.4 > 5.7`. The same candidate fails `high_risk` and receives zero directional fitness in that mode. If the RMST values were reversed, it would pass `high_risk` and fail `protective`.

After passing the direction gate, the candidate still competes using its configured log-rank, residual, or product fitness score. RMST establishes the direction; the fitness metric determines its evolutionary ranking.

## Configuration example

```python
common_options = {
    "outcome_label": "Duration",
    "censor_label": "Censoring",
    "fitness_metric": "log_rank",
    "group_thresh": None,
    "min_thresh": 0,
    "max_thresh": 5,
    "group_strata_min": 0.2,
}

protective_model = FIBERS(
    **common_options,
    desired_bin_effect="protective",
).fit(train_data)

high_risk_model = FIBERS(
    **common_options,
    desired_bin_effect="high_risk",
).fit(train_data)
```

Use separate models when both directions are scientifically relevant. Comparing the discovered populations can reveal whether different feature combinations characterize protective and high-risk burdens.

## Interpretation cautions

* A directional result is an association in the training data and does not establish a causal protective or harmful effect.
* The direction gate does not replace statistical assessment, validation on held-out data, or replication in an independent cohort.
* The RMST gate is unadjusted even when residual-based fitness incorporates covariates.
* `default` remains the appropriate choice when either direction is meaningful or backward-compatible behavior is required.
