import numpy as np
from fdm.core import bayes_decision, decision_maturity

def test_same_current_evidence_can_hide_opposite_terminal_actions():
    current_p = 0.15
    world_p = np.array([0.95])
    world_q = np.array([0.01])
    current_action = bayes_decision(current_p)
    assert bayes_decision(world_p[0]) != bayes_decision(world_q[0])
    assert current_action in (0, 1)

def test_maturity_detects_unstable_future_action():
    current_p = 0.15
    futures = np.array([0.01, 0.02, 0.95, 0.98])
    m = decision_maturity(current_p, futures)
    assert 0.0 < m < 1.0
