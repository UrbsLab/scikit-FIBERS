"""One correlation rule regardless of locus; strict blocks and original scoring."""
import itertools

import numpy as np
import pandas as pd

from methods import score_groups
from plot_mixed_blocks import make_blocks, expand
from run_mixed_risk import consistency


def table(names, correlations):
    return pd.DataFrame([{"feature1": a, "feature2": b, "pearson_r": r}
                         for (a, b), r in zip(itertools.combinations(names, 2), correlations)])


def test_either_within_or_between_locus_pair_qualifies():
    names = ["MM_A_10", "MM_A_20", "MM_B_30", "MM_C_40"]
    correlations = [.99, .1, .1, .1, .1, .98]
    blocks = make_blocks(table(names, correlations), names, .95)
    assert {tuple(g["features"]) for g in blocks} == {
        ("MM_A_10", "MM_A_20"), ("MM_B_30", "MM_C_40")}
    assert sum(g["cross_locus"] for g in blocks) == 1
    assert set(expand(["MM_B_30"], blocks)[0]["features"]) == {"MM_B_30", "MM_C_40"}


def test_strict_cutoff_does_not_accept_equal_or_negative_correlation():
    names = ["MM_A_10", "MM_B_30"]
    for r in [.95, -.99]:
        blocks = make_blocks(table(names, [r]), names, .95)
        assert all(len(g["features"]) == 1 for g in blocks)


def test_chain_does_not_imply_every_pair_exceeds_cutoff():
    names = ["MM_A_10", "MM_A_20", "MM_B_30"]
    blocks = make_blocks(table(names, [.99, .8, .97]), names, .95)
    assert max(len(g["features"]) for g in blocks) == 2


def test_cross_locus_scoring_is_max_not_sum_or_binary():
    frame = pd.DataFrame({"MM_A_10": [1, 0, 2, 0], "MM_B_30": [1, 1, 1, 0],
                          "MM_C_40": [0, 0, 1, 0]})
    groups = [{"features": ["MM_A_10", "MM_B_30"]}, {"features": ["MM_C_40"]}]
    np.testing.assert_array_equal(score_groups(frame, groups), [1, 1, 3, 0])


def test_consistency_includes_all_pairs_and_positions():
    records = [{"fold": 1, "original": ["A"], "within": ["A"], "mixed": ["A", "B"]},
               {"fold": 2, "original": ["B"], "within": ["B"], "mixed": ["A", "B"]},
               {"fold": 3, "original": ["C"], "within": ["C"], "mixed": ["C"]}]
    summary, pairs = consistency(records)
    assert len(pairs) == 9
    assert summary.n_pairs.eq(3).all()
    assert summary.set_index("scheme").loc["mixed", "mean_jaccard"] == 1 / 3
