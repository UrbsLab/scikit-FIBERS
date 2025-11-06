# FIBERS Multi-Thresholding
## Praneel Varshney - 05/2024 -- 09/2025

## Project Overview

This project extends the FIBERS algorithm to support multi-thresholding. This allows bins to define multiple risk groups (3) instead of only 2, as before. The original implementation of this extension used the multivariate log-rank test on all bins. This worked sucessfully when the algorithm 'multi_thresholding' parameter was treated as a toggle, i.e. we know the class of bins we wish to learn (only 3 group or only 2 group). However, a limitation was that the multivariate log-rank test naturally favored 3-group bins over 2-group bins, even when the ground truth suggested binary stratification was optimal. This was problematic when 'multi_thresholding' was 'True,' yet we wished to learn a 2-group bin. 

## Problem

The fitness functions used for evaluating bins have opposite biases:

1. **Multivariate log-rank test**: Favors 3-group bins over 2-group bins, even when 2-group stratification is optimal. It gives slightly higher scores to redundant 3-group bins where the medium risk curve essentially coincides with either the low or high risk curve (differences sometimes as small as a few thousandths). This happens because incorporating pairwise differences in the covariance matrix calculation adds weight even when the middle curve provides no new information.

2. **Pairwise averaging**: This has the opposite problem. It favors 2-group bins when 3-group is optimal. Averaging the three pairwise comparisons (Low-Medium, Low-High, Medium-High) results in lower fitness scores for 3-group bins compared to the best 2-group bin, even when the 3-group solution is actually better. By printing areas, we also notice that the Low-High pairwise comparison results in a lower log-rank score than the log-rank score of the 2-group bin despite there being greater area between the Low-High curves. This is because there are less instances in these 2 curves (some are in the Medium curve) and the 3-group curves are often more imbalanced, which also lowers the score.

## Implementation

### Multi-Thresholding Architecture

Extended the `BIN` class to support `group_threshold_list` instead of a single `group_threshold`, allowing bins to have 1 threshold (2 groups) or 2 thresholds (3 groups). This implementation maintains compatibility with the existing single-threshold functionality.

**BIN Class Extensions** (`src/skfibers/methods/bin.py`):
```python
# New attributes for multi-thresholding
self.group_threshold_list = []  # List of thresholds
self.pairwise_scores = []       # Pairwise log-rank scores for 3-group bins
self.group_prop_list = []       # Group proportions
self.count_mt = None            # Medium group count
```

**Population Management** (`src/skfibers/methods/population.py`):
- Modified crossover, mutation, and merge operations to handle threshold lists
- Enhanced genetic operators to add/remove/modify thresholds
- Updated equivalence checking for multi-threshold bins

### Fitness Functions

Two approaches were implemented for evaluating 3-group bins:

1. **Multivariate log-rank testing**: Initial approach using multivariate log-rank tests on all bins (currently commented out but can be easily re-enabled). This approach favors 3-group bins over 2-group bins, even when 2-group is optimal.

2. **Pairwise log-rank testing** (current implementation): Calculates three pairwise comparisons (Low-Medium, Low-High, Medium-High) and averages the scores:
```python
# For 3-group bins
lh_results = logrank_test(low_outcome, high_outcome, ...)
lm_results = logrank_test(low_outcome, mid_outcome, ...)
mh_results = logrank_test(mid_outcome, high_outcome, ...)

avg_stat = (lh_stat + lm_stat + mh_stat) / 3
self.pairwise_scores = [lh_stat, lm_stat, mh_stat]
```
This approach has the opposite bias. It favors 2-group bins when 3-group is optimal because averaging reduces the fitness score below what a good 2-group bin achieves.

**Evaluation Logic** (`evaluate_for_thresholds()`):
- **2-group bins**: Standard log-rank test between low and high risk groups
- **3-group bins**: Currently implements pairwise log-rank tests with averaged scores

Both approaches have opposite biases that prevent proper handling of 2-group vs 3-group selection.

### Bin Injection and Testing Framework

Created `evaluate_fixed_bin()` method for testing specific bins with predetermined thresholds. This helps with ground truth bin testing, performance comparison, and algorithm validation.

**Multi_Demo.ipynb**: Demo notebook showing multi-thresholding functionality with simulation support for both 2-group and 3-group datasets. Also includes analysis and visualizations like Kaplan-Meier curve plotting, area analysis, and comparisons.

## Current Status

The core multi-thresholding architecture is fully functional. The `BIN` class now supports variable-length threshold lists, and all genetic operators (crossover, mutation, merge) correctly handle both single-threshold and multi-threshold bins. The implementation maintains compatibility with the original single-threshold code.

Two fitness function approaches were tested extensively. The multivariate log-rank test implementation is complete and remains in the codebase (commented out in `evaluate_for_thresholds()`), so it can be re-enabled by uncommenting a few lines. The pairwise averaging approach is currently active. Both implementations have been validated on simulated datasets with known ground truth.

The bin injection mechanism allows testing arbitrary bins with fixed thresholds, which proved essential for validating algorithm behavior against ground truth. Multi_Demo.ipynb demonstrates the full pipeline with multiple simulation scenarios, Kaplan-Meier visualizations, area calculations between curves, and direct comparisons of fitness scores.

## Current Limitations

**Fitness Function Bias**: This is the main unsolved problem. The multivariate log-rank test systematically favors 3-group bins even when the medium risk curve is redundant (essentially overlapping with low or high). The score differences are small (sometimes a few thousandths), but enough to consistently rank the inferior 3-group bin above the correct 2-group bin. Conversely, pairwise averaging systematically penalizes legitimate 3-group bins by averaging three pairwise scores, resulting in fitness values below what a good 2-group bin achieves. This happens even when the 3-group solution shows clear separation between all three curves.

**Computational Complexity**: Multi-thresholding increases runtime, particularly with large populations. The algorithm now searches a more complex solution space (both 2-group and 3-group bins simultaneously), and evaluating 3-group bins requires additional log-rank calculations. While this overhead is expected given the expanded search space, extremely computationally intensive approaches (like AIC-based methods) remain impractical for the genetic algorithm framework.

## Future Directions

The fitness function bias needs to be addressed before the algorithm can reliably learn 2-group bins when multi-thresholding is enabled. Several directions could be explored:

**Area-Based Metrics**: Instead of relying solely on p-values, we can measure the actual area between survival curves and incoporate this into the fitness function. This directly quantifies separation and might naturally favor the solution with greater total separation. The challenge is developing a metric that's robust to censoring and sample size differences.

**Redundancy Detection**: Develop criteria to detect when a middle group provides no additional information (e.g., when its curve stays within some threshold distance of another curve). This could be used as a penalty term or as a filter to convert redundant 3-group bins back to 2-group bins before fitness evaluation. This could be used in conjunction with the multivariate log-rank test.

**Weighted Combinations**: Combine multiple metrics (log-rank scores, area measures, group balance) with learned or heuristic weights. This might capture different aspects of bin quality, but it risks introducing arbitrary parameters.

**Alternative Statistical Tests**: Investigate other survival analysis methods that might have different bias properties. The key requirement is computational efficiency suitable for genetic algorithm evaluation.

The core question is whether a principled, unbiased fitness function exists that can fairly compare 2-group and 3-group stratifications without artificially favoring either, or whether the comparison inherently requires domain knowledge about the expected number of risk groups.

## Contact

**Praneel Varshney**  
Email: [pvarsh@seas.upenn.edu]  
Institution: University of Pennsylvania  
Project: FIBERS Multi-Thresholding