from __future__ import annotations
import numpy as np

from .core import bayes_decision, expected_act_cost, expected_escalation_cost
from .simulator import sample_terminal_posteriors


def _realized_cost(decision: int, y: int, fp_cost: float, fn_cost: float) -> float:
    if decision == 1 and y == 0:
        return fp_cost
    if decision == 0 and y == 1:
        return fn_cost
    return 0.0


def reversal_harm(current_p: float, terminal_ps: np.ndarray, *, fp_cost: float = 1.0, fn_cost: float = 5.0, rollback_block_cost: float = 0.20, rollback_allow_cost: float = 0.80) -> float:
    cur = bayes_decision(current_p, fp_cost, fn_cost)
    if len(terminal_ps) == 0:
        return 0.0
    harms = []
    for q in terminal_ps:
        final = bayes_decision(float(q), fp_cost, fn_cost)
        if final == cur:
            harms.append(0.0)
        elif cur == 1 and final == 0:
            harms.append(float(rollback_block_cost))
        else:
            harms.append(float(rollback_allow_cost))
    return float(np.mean(harms))


def expected_maturity_regret(current_p: float, terminal_ps: np.ndarray, *, fp_cost: float = 1.0, fn_cost: float = 5.0) -> float:
    cur_d = bayes_decision(current_p, fp_cost, fn_cost)
    vals = []
    for q in terminal_ps:
        q = float(q)
        final_d = bayes_decision(q, fp_cost, fn_cost)
        cur_loss = (1-q)*fp_cost if cur_d == 1 else q*fn_cost
        fin_loss = (1-q)*fp_cost if final_d == 1 else q*fn_cost
        vals.append(cur_loss-fin_loss)
    return float(np.mean(vals)) if len(vals) else 0.0


def run_fdm3_policy(episode: dict, rng: np.random.Generator, *, delay_cost_per_step: float = 0.035, review_cost: float = 0.35, review_accuracy: float = 0.97, fp_cost: float = 1.0, fn_cost: float = 5.0, regret_weight: float = 0.5, commitment_weight: float = 1.0, rollback_block_cost: float = 0.20, rollback_allow_cost: float = 0.80, n_mc: int = 200) -> dict:
    ps = episode["probabilities"]
    y = int(episode["y"])
    T = len(ps)
    for t, p in enumerate(ps):
        p = float(p)
        futures = sample_terminal_posteriors(p, t, T, rng, n=n_mc)
        emr = expected_maturity_regret(p, futures, fp_cost=fp_cost, fn_cost=fn_cost)
        rh = reversal_harm(p, futures, fp_cost=fp_cost, fn_cost=fn_cost, rollback_block_cost=rollback_block_cost, rollback_allow_cost=rollback_allow_cost)
        act = expected_act_cost(p, fp_cost, fn_cost)
        act_score = act + regret_weight*emr + commitment_weight*rh
        esc = expected_escalation_cost(p, review_cost, review_accuracy, fp_cost, fn_cost)
        if t < T-1:
            future_risk = np.mean([expected_act_cost(float(q), fp_cost, fn_cost) for q in futures])
            wait = delay_cost_per_step + float(future_risk)
            scores = {"ACT":act_score,"WAIT":wait,"ESCALATE":esc}
        else:
            scores = {"ACT":act_score,"ESCALATE":esc}
        choice = min(scores, key=scores.get)
        if choice == "WAIT":
            continue
        if choice == "ESCALATE":
            correct = rng.random() < review_accuracy
            d = y if correct else 1-y
            return {"policy":"fdm3","t":t,"decision":d,"emr":emr,"reversal_harm":rh,"cost":_realized_cost(d,y,fp_cost,fn_cost)+review_cost+delay_cost_per_step*t}
        d = bayes_decision(p, fp_cost, fn_cost)
        return {"policy":"fdm3","t":t,"decision":d,"emr":emr,"reversal_harm":rh,"cost":_realized_cost(d,y,fp_cost,fn_cost)+delay_cost_per_step*t}
    raise RuntimeError("FDM-III failed to terminate")
