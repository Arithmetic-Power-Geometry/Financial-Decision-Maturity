# Financial Decision Maturity: computational formulation

## 1. Sequential evidence

Let \(E_t\) denote the information available at time \(t\), with
\(E_1 \subseteq \cdots \subseteq E_T\).

Let \(p_t=P(Y=1\mid E_t)\), where \(Y=1\) denotes the costly adverse state
(e.g. a fraudulent transaction).

Under false-positive cost \(c_{FP}\) and false-negative cost \(c_{FN}\),
the Bayes action is

\[
d_t =
\begin{cases}
1,&(1-p_t)c_{FP}\le p_t c_{FN},\\
0,&\text{otherwise.}
\end{cases}
\]

## 2. Decision maturity

For a modeled distribution of terminal posteriors conditional on current
evidence, define

\[
M_t=P(d_T=d_t\mid E_t).
\]

This quantity measures **decision stability**, not classifier calibration.
A highly confident posterior can still be immature if plausible future
evidence crosses the cost-sensitive decision boundary.

## 3. Action set

FDM uses the action set

\[
\mathcal A_t=\{\mathrm{ACT},\mathrm{WAIT},\mathrm{ESCALATE}\}.
\]

ACT incurs the current expected decision loss. WAIT incurs expected future
decision loss plus delay cost. ESCALATE incurs review cost plus residual
review error.

## 4. Research boundary

The implementation is deliberately conservative about novelty. Sequential
decision theory, optimal stopping, Bayesian value of information,
selective prediction, abstention, conformal risk control, and active
feature acquisition are established areas. The research question is
whether **decision-maturity as future action invariance under asymmetric
financial cost**, combined with an explicit act/wait/escalate controller,
yields useful theoretical or empirical results beyond those literatures.

A novelty claim should only be made after a dedicated prior-art audit.
