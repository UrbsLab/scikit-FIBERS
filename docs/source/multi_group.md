# Two- and three-group thresholding

FIBERS uses an explicit group count. `n_groups=2` is the default and retains the original one-threshold behavior. `n_groups=3` uses two thresholds and produces low-, middle-, and high-burden groups. A single population does not mix the two group counts.

Select the three-group method with `n_groups=3`. Leave *group_thresh_list* as `None` to search threshold pairs adaptively:

```python
model = FIBERS(
    outcome_label="Duration",
    censor_label="Censoring",
    fitness_metric="log_rank",
    n_groups=3,
    group_thresh_list=None,
    min_thresh=0,
    max_thresh=5,
)
model.fit(train_data)
```

The default is `n_groups=2`, so existing runs retain their original behavior. Two-group runs use the scalar *group_thresh*. Three-group runs use *group_thresh_list*, which must contain exactly two increasing thresholds when fixed:

```python
two_group_model = FIBERS(n_groups=2, group_thresh=2)
three_group_model = FIBERS(n_groups=3, group_thresh_list=[1, 3])
```

## Group definitions

For thresholds `[low_threshold, high_threshold]`, a bin assigns samples by their summed feature burden:

| Encoded group | Burden rule |
| --- | --- |
| `0` | sum <= *low_threshold* |
| `1` | *low_threshold* < sum <= *high_threshold* |
| `2` | sum > *high_threshold* |

A two-group model continues to encode its groups as `0` and `1`.

When adaptive three-group thresholding is active, each candidate feature set is evaluated across every increasing threshold pair between *min_thresh* and *max_thresh*. When threshold evolution is selected, crossover, mutation, and merge exchange or replace thresholds while always retaining exactly two. The final training iteration again performs exhaustive threshold-pair evaluation.

## Fitness calculation

With `n_groups=2`, bins use the existing two-sample log-rank test. With `n_groups=3`, bins run three pairwise log-rank tests in this order:

1. Low versus high.
2. Low versus middle.
3. Middle versus high.

The three test statistics are averaged to obtain the bin's log-rank score. The reported p-value is the smallest of the three pairwise p-values, and the individual statistics are available as `bin.pairwise_scores`. If any group is empty, the log-rank score is zero.

For residual fitness, two-group models retain the Wilcoxon rank-sum calculation and three-group models use the Kruskal-Wallis statistic. Product fitness multiplies the applicable log-rank and residual statistics. The *group_strata_min* penalty uses the smallest proportion across all configured groups.

FIBERS does not automatically compare two and three groups because their raw fitness statistics are not guaranteed to be directly comparable. Fit the two group counts separately and use held-out survival performance when a comparison is needed. The experimental AUC/Pareto-front work from separate research branches is not part of this method.

## Results and helper methods

For a selected three-group bin, `predict(data, bin_number=...)` and `transform(data, full_sums=False)` return `0`, `1`, or `2`. `get_multi_bin_groups()` returns:

```text
low_outcome, middle_outcome, high_outcome,
low_censor, middle_censor, high_censor
```

`get_kaplan_meir()` automatically plots all three curves for a three-group bin. `get_multi_kaplan_meir()` is also available explicitly. The existing `get_bin_groups()` keeps its four-value two-group return contract and directs three-group callers to `get_multi_bin_groups()`.

Population-wide prediction combines weighted votes from bins that all share the model's configured group count.

## Protective and high-risk ordering

Both group counts support all three *desired_bin_effect* values. The two-group method uses the existing RMST direction check. For a three-group bin, FIBERS compares each adjacent pair (low versus middle, then middle versus high) using censoring-aware RMST at the latest follow-up time shared by that pair. Pair-specific horizons avoid treating two later-surviving strata as tied merely because an earlier stratum has shorter follow-up.

* `protective` requires `low RMST < middle RMST < high RMST`.
* `high_risk` requires `low RMST > middle RMST > high RMST`.
* `default` accepts either ordering.

The inequalities are strict, so tied or non-monotonic groups do not satisfy a requested direction. A wrong-direction configuration receives zero applicable log-rank and/or residual fitness. During adaptive search, FIBERS first prefers configurations that satisfy both the requested direction and *group_strata_min*, then uses the same direction-valid and group-balance fallback rules as the standard two-group path.
