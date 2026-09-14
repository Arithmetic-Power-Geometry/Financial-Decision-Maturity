# Premature Financial Decision diagnostics

## Target

For each nonterminal evidence stage, define

\[
PFD_t = 1\{d_t \ne d_T^*\}.
\]

`PFD_t=1` means the action that is optimal under current evidence is reversed by the terminal-evidence action.

This experiment tests whether Decision Maturity diagnostics identify later action reversal better than ordinary uncertainty signals.

## Compared predictors

- posterior confidence;
- binary entropy;
- posterior margin;
- Decision Maturity \(1-M_t\);
- Expected Maturity Regret (EMR);
- asymmetric Reversal Harm.

Higher values are oriented to mean greater reversal risk.

## Evaluation protocol

Training seeds and held-out test seeds are disjoint.

Ranking:
- AUROC;
- AUPRC;
- precision among the top 10% highest-risk stages.

Calibration:
- an empirical score-to-probability map is fitted only on training seeds;
- held-out Brier score and expected calibration error (ECE) are reported.

## Critical subgroup

The central falsification test is the set

\[
\{E_t: \max(p_t,1-p_t)\ge 0.90\}.
\]

These are **high-confidence decisions**. The key question is whether the maturity variables can rank which of these decisions will nevertheless reverse later.

If maturity/EMR/reversal-harm do not improve substantially over confidence or entropy inside this subgroup, the diagnostic contribution is weak.
