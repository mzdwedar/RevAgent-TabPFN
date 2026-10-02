import math

import numpy as np
import pytest

from revbench.metrics import ece, evaluate, log_loss, pr_auc, roc_auc, timed


def test_log_loss_hand_computed():
    assert log_loss([1, 0], [0.8, 0.2]) == pytest.approx(-math.log(0.8))


def test_log_loss_survives_hard_zero_and_one():
    assert math.isfinite(log_loss([1, 0], [0.0, 1.0]))


def test_pr_auc_and_roc_auc_hand_computed():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(y, p) == pytest.approx(0.75)
    # ranked 0.8(+) 0.4(-) 0.35(+) 0.1(-): AP = 0.5 * 1 + 0.5 * 2/3
    assert pr_auc(y, p) == pytest.approx(5 / 6)


def test_ece_hand_computed():
    # bin 0.1: observed 0 vs 0.1 (weight .5); bin 0.9: observed .5 vs .9 (weight .5)
    assert ece([0, 0, 1, 0], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(0.5 * 0.1 + 0.5 * 0.4)


def test_ece_is_zero_when_perfectly_calibrated_and_handles_p_equal_one():
    assert ece([1, 1], [1.0, 1.0]) == 0.0
    assert ece([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_evaluate_returns_all_metrics():
    out = evaluate(np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.9]))
    assert set(out) == {"pr_auc", "roc_auc", "log_loss", "ece"}


def test_timed_returns_result_and_seconds():
    result, seconds = timed(lambda a, b: a + b, 1, 2)
    assert result == 3
    assert seconds >= 0
