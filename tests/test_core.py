import numpy as np
from fdm.core import bayes_decision, decision_maturity, expected_act_cost

def test_bayes_decision_threshold_is_cost_sensitive():
    assert bayes_decision(0.05, 1.0, 5.0) == 0
    assert bayes_decision(0.50, 1.0, 5.0) == 1

def test_maturity_is_probability():
    futures = np.array([0.9, 0.8, 0.7, 0.1])
    m = decision_maturity(0.8, futures)
    assert 0.0 <= m <= 1.0

def test_expected_act_cost_nonnegative():
    for p in [0.0, 0.1, 0.5, 1.0]:
        assert expected_act_cost(p) >= 0.0
