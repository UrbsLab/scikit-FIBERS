# Two- and three-group thresholding

FIBERS normally learns one burden threshold and divides samples into two groups. Multi-group thresholding is an opt-in mode that lets the same population contain both one-threshold (2-group) and two-threshold (3-group) bins.

Enable the mode with `multi_thresholding=True`. Leave *group_thresh_list* as `None` to search both forms adaptively:

```python
model = FIBERS(
    outcome_label="Duration",
    censor_label="Censoring",
    fitness_metric="log_rank",
    multi_thresholding=True,
    group_thresh_list=None,
    min_thresh=0,
    max_thresh=5,
)
model.fit(train_data)
```

The default is `multi_thresholding=False`, so existing two-group runs retain their original behavior. In multi-group mode, use *group_thresh_list* instead of the scalar *group_thresh*. A fixed list with one value forces two groups, while a fixed list with two increasing values forces three groups:

```python
two_group_model = FIBERS(multi_thresholding=True, group_thresh_list=[2])
three_group_model = FIBERS(multi_thresholding=True, group_thresh_list=[1, 3])
```

## Group definitions

For thresholds `[low_threshold, high_threshold]`, a bin assigns samples by their summed feature burden:

| Encoded group | Burden rule |
| --- | --- |
| `0` | sum <= *low_threshold* |
| `1` | *low_threshold* < sum <= *high_threshold* |
| `2` | sum > *high_threshold* |

A one-threshold bin continues to encode its groups as `0` and `1`.

When adaptive multi-group thresholding is active, each candidate feature set is evaluated across every allowed single threshold and every increasing pair between *min_thresh* and *max_thresh*. When threshold evolution is selected, crossover, mutation, and merge can exchange, add, remove, or replace thresholds. The final training iteration again performs exhaustive threshold evaluation.

## Fitness calculation

Two-group bins use the existing two-sample log-rank test. Three-group bins run three pairwise log-rank tests in this order:

1. Low versus high.
2. Low versus middle.
3. Middle versus high.

The three test statistics are averaged to obtain the bin's log-rank score. The reported p-value is the smallest of the three pairwise p-values, and the individual statistics are available as `bin.pairwise_scores`. If any group is empty, the log-rank score is zero.

For residual fitness, two-group bins retain the Wilcoxon rank-sum calculation and three-group bins use the Kruskal-Wallis statistic. Product fitness multiplies the applicable log-rank and residual statistics. The *group_strata_min* penalty uses the smallest proportion across all groups, including the middle group for a three-group bin.

Pairwise averaging makes the two- and three-group scores comparable within the current multi-group implementation, but it does not guarantee an unbiased preference between them. The experimental AUC/Pareto-front work from separate research branches is not part of this mode.

## Results and helper methods

For a selected three-group bin, `predict(data, bin_number=...)` and `transform(data, full_sums=False)` return `0`, `1`, or `2`. `get_multi_bin_groups()` returns:

```text
low_outcome, middle_outcome, high_outcome,
low_censor, middle_censor, high_censor
```

`get_kaplan_meir()` automatically plots all three curves for a three-group bin. `get_multi_kaplan_meir()` is also available explicitly. The existing `get_bin_groups()` keeps its four-value two-group return contract and directs three-group callers to `get_multi_bin_groups()`.

Population-wide prediction combines weighted votes from bins with different group counts. In multi-group mode, the high group of a two-group bin votes for class `2`, preserving the low/middle/high ordering when mixed with three-group bins.

## Protective and high-risk ordering

Multi-group thresholding supports all three *desired_bin_effect* values. One-threshold bins use the existing two-group RMST direction check. For a two-threshold bin, FIBERS compares each adjacent pair (low versus middle, then middle versus high) using censoring-aware RMST at the latest follow-up time shared by that pair. Pair-specific horizons avoid treating two later-surviving strata as tied merely because an earlier stratum has shorter follow-up.

* `protective` requires `low RMST < middle RMST < high RMST`.
* `high_risk` requires `low RMST > middle RMST > high RMST`.
* `default` accepts either ordering.

The inequalities are strict, so tied or non-monotonic groups do not satisfy a requested direction. A wrong-direction configuration receives zero applicable log-rank and/or residual fitness. During adaptive search, FIBERS first prefers configurations that satisfy both the requested direction and *group_strata_min*, then uses the same direction-valid and group-balance fallback rules as the standard two-group path.
