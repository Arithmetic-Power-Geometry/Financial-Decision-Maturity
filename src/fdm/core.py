from __future__ import annotations
import math
import numpy as np

EPS = 1e-12

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))

def bayes_decision(p_risk: float, false_positive_cost: float = 1.0,
                   false_negative_cost: float = 5.0) -> int:
    """Return 1=intervene/block, 0=allow."""
    p = min(max(float(p_risk), 0.0), 1.0)
    cost_block = (1.0 - p) * false_positive_cost
    cost_allow = p * false_negative_cost
    return int(cost_block <= cost_allow)

def expected_act_cost(p_risk: float, false_positive_cost: float = 1.0,
                      false_negative_cost: float = 5.0) -> float:
    d = bayes_decision(p_risk, false_positive_cost, false_negative_cost)
    p = float(p_risk)
    if d == 1:
        return (1.0 - p) * false_positive_cost
    return p * false_negative_cost

def decision_maturity(current_p: float, terminal_ps: np.ndarray,
                      false_positive_cost: float = 1.0,
                      false_negative_cost: float = 5.0) -> float:
    """Probability current Bayes decision matches a simulated terminal decision."""
    cur = bayes_decision(current_p, false_positive_cost, false_negative_cost)
    if len(terminal_ps) == 0:
        return 1.0
    finals = np.array([
        bayes_decision(float(p), false_positive_cost, false_negative_cost)
        for p in terminal_ps
    ])
    return float(np.mean(finals == cur))

def expected_wait_cost(terminal_ps: np.ndarray, delay_cost: float,
                       false_positive_cost: float = 1.0,
                       false_negative_cost: float = 5.0) -> float:
    if len(terminal_ps) == 0:
        return float(delay_cost)
    costs = [
        expected_act_cost(float(p), false_positive_cost, false_negative_cost)
        for p in terminal_ps
    ]
    return float(np.mean(costs) + delay_cost)

def expected_escalation_cost(p_risk: float, review_cost: float = 0.35,
                             review_accuracy: float = 0.97,
                             false_positive_cost: float = 1.0,
                             false_negative_cost: float = 5.0) -> float:
    """Expected cost of a high-quality review with symmetric review error."""
    p = min(max(float(p_risk), 0.0), 1.0)
    err = 1.0 - review_accuracy
    residual = p * err * false_negative_cost + (1 - p) * err * false_positive_cost
    return float(review_cost + residual)
