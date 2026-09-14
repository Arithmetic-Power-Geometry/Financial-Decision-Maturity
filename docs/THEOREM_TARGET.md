# Decision-Maturity Boundary: theorem target and falsification plan

## 1. Premature decision impossibility

Let \(E_t\) be the evidence available at time \(t<T\). Consider two admissible financial environments \(P\) and \(Q\) satisfying

\[
P(E_t)=Q(E_t)
\]

but inducing different full-information Bayes-optimal actions,

\[
a_P^*(E_T)\neq a_Q^*(E_T).
\]

Any deterministic decision rule \(\pi(E_t)\) must output the same action in both environments because the observable evidence is identical at time \(t\). Therefore \(\pi\) cannot equal both full-information optimal actions.

### Proposition
For any environment class containing such a pair \(P,Q\), no rule measurable with respect to \(E_t\) is universally full-information-optimal at time \(t\).

This is an information limitation, not a limitation of classifier capacity.

## 2. Decision-maturity boundary

The proposition motivates a boundary: irreversible action should not be treated as mature merely because a classifier is confident. Maturity concerns the stability of the *action* under plausible future evidence.

The implemented quantity is

\[
M_t=\Pr(d_T=d_t\mid E_t).
\]

A candidate safe-action region is

\[
\mathcal S_\tau=\{E_t: M_t\ge \tau\}.
\]

The empirical question is whether cost-aware gating by \(\mathcal S_\tau\) reduces realized loss relative to acting immediately, fixed delay, and systematic escalation.

## 3. Falsification criteria

The current FDM formulation should be considered unsupported if, across well-calibrated sequential settings:

1. maturity fails to predict future action reversal;
2. maturity gating does not improve cost relative to simpler timing rules;
3. results vanish under moderate prevalence/calibration/cost perturbations;
4. equivalent performance is obtained by a simpler confidence threshold; or
5. the proposed quantities reduce exactly to a standard optimal-stopping or value-of-information construction without an additional testable result.

These criteria are intentionally strict.
