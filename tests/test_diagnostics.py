import numpy as np
from fdm.diagnostics import binary_entropy, confidence_reversal_score, margin_reversal_score, premature_financial_decision
from fdm.diagnostic_metrics import roc_auc, average_precision

def test_entropy_bounds():
    assert 0 <= binary_entropy(0.1) <= 1
    assert abs(binary_entropy(0.5)-1.0) < 1e-9

def test_uncertainty_scores_peak_near_half():
    assert confidence_reversal_score(0.5) > confidence_reversal_score(0.95)
    assert margin_reversal_score(0.5) > margin_reversal_score(0.95)

def test_pfd_detects_action_reversal():
    assert premature_financial_decision(0.01,0.95) == 1

def test_metric_sanity():
    y=np.array([0,0,1,1]); s=np.array([0.1,0.2,0.8,0.9])
    assert abs(roc_auc(y,s)-1.0) < 1e-12
    assert abs(average_precision(y,s)-1.0) < 1e-12
