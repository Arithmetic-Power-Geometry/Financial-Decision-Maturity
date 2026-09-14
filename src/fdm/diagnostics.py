from __future__ import annotations
import math
import numpy as np

from .core import bayes_decision, decision_maturity
from .fdm2 import expected_maturity_regret
from .fdm3 import reversal_harm
from .simulator import sample_terminal_posteriors


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    h = -(p*math.log(p) + (1-p)*math.log(1-p))
    return h / math.log(2.0)


def confidence_reversal_score(p: float) -> float:
    return 1.0 - max(float(p), 1.0-float(p))


def margin_reversal_score(p: float) -> float:
    return 1.0 - abs(2.0*float(p) - 1.0)


def diagnostic_scores(current_p: float, t: int, T: int, rng: np.random.Generator, *, fp_cost: float = 1.0, fn_cost: float = 5.0, rollback_block_cost: float = 0.2, rollback_allow_cost: float = 0.8, n_mc: int = 250) -> dict:
    futures = sample_terminal_posteriors(current_p, t, T, rng, n=n_mc)
    maturity = decision_maturity(current_p, futures, fp_cost, fn_cost)
    emr = expected_maturity_regret(current_p, futures, fp_cost=fp_cost, fn_cost=fn_cost)
    harm = reversal_harm(current_p, futures, fp_cost=fp_cost, fn_cost=fn_cost, rollback_block_cost=rollback_block_cost, rollback_allow_cost=rollback_allow_cost)
    return {"confidence": max(float(current_p), 1.0-float(current_p)), "confidence_reversal_score": confidence_reversal_score(current_p), "entropy": binary_entropy(current_p), "margin_reversal_score": margin_reversal_score(current_p), "maturity": maturity, "maturity_reversal_score": 1.0 - maturity, "emr": emr, "reversal_harm": harm}


def premature_financial_decision(current_p: float, terminal_p: float, fp_cost: float = 1.0, fn_cost: float = 5.0) -> int:
    return int(bayes_decision(current_p, fp_cost, fn_cost) != bayes_decision(terminal_p, fp_cost, fn_cost))
