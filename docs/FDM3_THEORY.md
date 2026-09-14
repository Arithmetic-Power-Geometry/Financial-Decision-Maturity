# FDM-III: Commitment-Aware Decision Maturity

FDM-III adds **path dependence**. A premature financial action may create harm even if it is corrected later.

Examples:
- a false block may cause customer friction or lost commerce;
- an unsafe allow may create unrecoverable fraud exposure before reversal;
- delayed correction can therefore have direction-dependent rollback cost.

Define expected reversal harm

\[
H_t = E[c(d_t\rightarrow d_T^*)\,1\{d_t\ne d_T^*\}\mid E_t].
\]

Unlike plain reversal probability, \(H_t\) distinguishes the economic severity and direction of a future reversal.

FDM-III uses

\[
S_A(t)=R_A(t)+\lambda EMR_t+\gamma H_t,
\]

and compares this with WAIT and ESCALATE.

## Important novelty boundary

This path-dependent quantity may still be representable inside a sufficiently general optimal-stopping / sequential-decision cost function. Therefore the software result alone cannot establish theoretical novelty.

The framework is publishable only if at least one of these survives:
1. a distinct theorem about maturity under path-dependent irreversible costs;
2. a diagnostic result not reducible to posterior confidence;
3. robust empirical advantage in clearly defined financial regimes;
4. a real-data finding that changes when a financial decision should be made.
