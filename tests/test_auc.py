import math

import numpy as np

from assay.eval.auc import roc_auc


def test_known_value():
    # 和 sklearn.metrics.roc_auc_score 的文档示例一致
    assert math.isclose(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)


def test_perfect_and_inverse():
    assert roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0


def test_ties_count_half():
    assert roc_auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_one_class_is_nan():
    assert math.isnan(roc_auc([1, 2], [0, 0]))


def test_large_random_is_about_half():
    rng = np.random.default_rng(0)
    assert abs(roc_auc(rng.random(20000), rng.integers(0, 2, 20000)) - 0.5) < 0.02
