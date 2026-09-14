from __future__ import annotations
import numpy as np
from .core import (
    bayes_decision, decision_maturity, expected_act_cost,
    expected_wait_cost, expected_escalation_cost
)
from .simulator import sample_terminal_posteriors

def _realized_cost(decision: int, y: int, fp_cost: float, fn_cost: float) -> float:
    if decision == 1 and y == 0:
        return fp_cost
    if decision == 0 and y == 1:
        return fn_cost
    return 0.0

def run_policy(episode: dict, policy: str, rng: np.random.Generator,
               maturity_threshold: float = 0.93,
               delay_cost_per_step: float = 0.035,
               review_cost: float = 0.35,
               review_accuracy: float = 0.97,
               fp_cost: float = 1.0, fn_cost: float = 5.0) -> dict:
    ps = episode["probabilities"]
    y = int(episode["y"])
    T = len(ps)

    if policy == "immediate":
        t = 0
        d = bayes_decision(ps[t], fp_cost, fn_cost)
        return {"policy": policy, "t": t, "decision": d,
                "maturity": np.nan,
                "cost": _realized_cost(d, y, fp_cost, fn_cost)}

    if policy == "fixed_delay":
        t = min(2, T-1)
        d = bayes_decision(ps[t], fp_cost, fn_cost)
        return {"policy": policy, "t": t, "decision": d,
                "maturity": np.nan,
                "cost": _realized_cost(d, y, fp_cost, fn_cost) + delay_cost_per_step*t}

    if policy == "always_escalate":
        correct = rng.random() < review_accuracy
        d = y if correct else 1-y
        return {"policy": policy, "t": 0, "decision": d,
                "maturity": np.nan,
                "cost": _realized_cost(d, y, fp_cost, fn_cost) + review_cost}

    if policy == "oracle":
        t = T-1
        d = bayes_decision(ps[t], fp_cost, fn_cost)
        return {"policy": policy, "t": t, "decision": d,
                "maturity": 1.0,
                "cost": _realized_cost(d, y, fp_cost, fn_cost) + delay_cost_per_step*t}

    if policy != "fdm":
        raise ValueError(f"Unknown policy: {policy}")

    for t, p in enumerate(ps):
        futures = sample_terminal_posteriors(float(p), t, T, rng)
        maturity = decision_maturity(float(p), futures, fp_cost, fn_cost)
        act_cost = expected_act_cost(float(p), fp_cost, fn_cost)
        wait_cost = expected_wait_cost(
            futures, delay_cost=(delay_cost_per_step if t < T-1 else 1e9),
            false_positive_cost=fp_cost, false_negative_cost=fn_cost
        )
        esc_cost = expected_escalation_cost(
            float(p), review_cost, review_accuracy, fp_cost, fn_cost
        )

        candidates = {"ESCALATE": esc_cost}
        if t < T-1:
            candidates["WAIT"] = wait_cost
        if maturity >= maturity_threshold or t == T-1:
            candidates["ACT"] = act_cost

        action = min(candidates, key=candidates.get)

        if action == "WAIT":
            continue
        if action == "ESCALATE":
            correct = rng.random() < review_accuracy
            d = y if correct else 1-y
            cost = _realized_cost(d, y, fp_cost, fn_cost) + review_cost + delay_cost_per_step*t
            return {"policy": policy, "t": t, "decision": d,
                    "maturity": maturity, "cost": cost}
        d = bayes_decision(float(p), fp_cost, fn_cost)
        cost = _realized_cost(d, y, fp_cost, fn_cost) + delay_cost_per_step*t
        return {"policy": policy, "t": t, "decision": d,
                "maturity": maturity, "cost": cost}

    raise RuntimeError("FDM policy failed to terminate")
