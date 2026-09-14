from __future__ import annotations
import math
import numpy as np

from .core import (
    bayes_decision,
    decision_maturity,
    expected_act_cost,
    expected_wait_cost,
    expected_escalation_cost,
)
from .simulator import sample_terminal_posteriors


def _realized_cost(decision: int, y: int, fp_cost: float, fn_cost: float) -> float:
    if decision == 1 and y == 0:
        return fp_cost
    if decision == 0 and y == 1:
        return fn_cost
    return 0.0


def _action_certainty(p: float, fp_cost: float, fn_cost: float) -> float:
    """Cost-aware distance from the Bayes action boundary, mapped to [0,1].

    This avoids the misleading property of max(p,1-p) in rare-event settings,
    where a very small fraud posterior can appear highly "confident" even when
    it lies near a cost-sensitive blocking threshold.
    """
    threshold = fp_cost / (fp_cost + fn_cost)
    p = float(np.clip(p, 0.0, 1.0))
    if p >= threshold:
        denom = max(1.0 - threshold, 1e-12)
        return float((p - threshold) / denom)
    denom = max(threshold, 1e-12)
    return float((threshold - p) / denom)


def _binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return float(h)


def run_strong_policy(
    episode: dict,
    policy: str,
    rng: np.random.Generator,
    maturity_threshold: float = 0.93,
    certainty_threshold: float = 0.80,
    entropy_threshold: float = 0.35,
    delay_cost_per_step: float = 0.035,
    review_cost: float = 0.35,
    review_accuracy: float = 0.97,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> dict:
    """Run strong timing baselines using the same information/cost model as FDM.

    Policies:
      fdm               : maturity-gated ACT + WAIT/ESCALATE cost comparison.
      voi               : pure expected-value-of-information timing; no maturity gate.
      confidence_gate   : cost-aware certainty gate.
      entropy_gate      : posterior entropy gate.
      immediate         : act at t=0.
      fixed_delay       : act at t=2.
      always_escalate   : review immediately.
      terminal          : act at final evidence step (reference, not deployable oracle).
    """
    ps = np.asarray(episode["probabilities"], dtype=float)
    y = int(episode["y"])
    T = len(ps)

    def finish(policy_name, t, d, maturity=np.nan, action="ACT", extra=0.0):
        return {
            "policy": policy_name,
            "t": int(t),
            "decision": int(d),
            "maturity": float(maturity) if np.isfinite(maturity) else np.nan,
            "action": action,
            "cost": float(
                _realized_cost(int(d), y, fp_cost, fn_cost)
                + delay_cost_per_step * int(t)
                + extra
            ),
        }

    if policy == "immediate":
        return finish(policy, 0, bayes_decision(ps[0], fp_cost, fn_cost))

    if policy == "fixed_delay":
        t = min(2, T - 1)
        return finish(policy, t, bayes_decision(ps[t], fp_cost, fn_cost))

    if policy == "always_escalate":
        correct = rng.random() < review_accuracy
        d = y if correct else 1 - y
        return finish(policy, 0, d, action="ESCALATE", extra=review_cost)

    if policy == "terminal":
        t = T - 1
        return finish(policy, t, bayes_decision(ps[t], fp_cost, fn_cost), maturity=1.0)

    if policy not in {"fdm", "voi", "confidence_gate", "entropy_gate"}:
        raise ValueError(f"Unknown policy: {policy}")

    for t, p in enumerate(ps):
        futures = sample_terminal_posteriors(float(p), t, T, rng)
        maturity = decision_maturity(float(p), futures, fp_cost, fn_cost)
        act_cost = expected_act_cost(float(p), fp_cost, fn_cost)
        esc_cost = expected_escalation_cost(
            float(p), review_cost, review_accuracy, fp_cost, fn_cost
        )
        if t < T - 1:
            wait_cost = expected_wait_cost(
                futures,
                delay_cost=delay_cost_per_step,
                false_positive_cost=fp_cost,
                false_negative_cost=fn_cost,
            )
        else:
            wait_cost = float("inf")

        if policy == "fdm":
            allow_act = (maturity >= maturity_threshold) or (t == T - 1)
        elif policy == "voi":
            allow_act = True
        elif policy == "confidence_gate":
            allow_act = (_action_certainty(float(p), fp_cost, fn_cost) >= certainty_threshold) or (t == T - 1)
        else:
            allow_act = (_binary_entropy(float(p)) <= entropy_threshold) or (t == T - 1)

        candidates = {"ESCALATE": esc_cost}
        if t < T - 1:
            candidates["WAIT"] = wait_cost
        if allow_act:
            candidates["ACT"] = act_cost

        action = min(candidates, key=candidates.get)
        if action == "WAIT":
            continue
        if action == "ESCALATE":
            correct = rng.random() < review_accuracy
            d = y if correct else 1 - y
            return finish(policy, t, d, maturity=maturity, action=action, extra=review_cost)

        d = bayes_decision(float(p), fp_cost, fn_cost)
        return finish(policy, t, d, maturity=maturity, action=action)

    raise RuntimeError(f"{policy} failed to terminate")
