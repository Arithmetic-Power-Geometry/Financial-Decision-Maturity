# Asymmetric High-Confidence Reversal Bound

Let

\[
p_t=P(Y=1\mid\mathcal F_t)
\]

and let the cost-sensitive Bayes threshold be

\[
\tau=\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

The current action is \(d_t=1\{p_t\ge\tau\}\). Let \(p_T\) be the posterior after future evidence, satisfying Bayesian coherence

\[
E[p_T\mid\mathcal F_t]=p_t.
\]

## Theorem: sharp one-step reversal bound

If \(p_t<\tau\),

\[
P(d_T=1\mid\mathcal F_t)=P(p_T\ge\tau\mid\mathcal F_t)\le \frac{p_t}{\tau}.
\]

If \(p_t>\tau\),

\[
P(d_T=0\mid\mathcal F_t)=P(p_T<\tau\mid\mathcal F_t)\le \frac{1-p_t}{1-\tau}.
\]

The bounds are sharp in the limiting sense.

### Proof

For \(p_t<\tau\), let \(r=P(p_T\ge\tau\mid\mathcal F_t)\). Because \(p_T\ge\tau\) on the reversal event and \(p_T\ge0\) otherwise,

\[
p_t=E[p_T\mid\mathcal F_t]\ge r\tau,
\]

so \(r\le p_t/\tau\).

For \(p_t>\tau\), let \(r=P(p_T<\tau\mid\mathcal F_t)\). Since \(p_T<\tau\) on the reversal event and \(p_T\le1\) otherwise,

\[
p_t=E[p_T\mid\mathcal F_t]\le r\tau+(1-r),
\]

hence \(r\le(1-p_t)/(1-\tau)\).

Two-point terminal-posterior laws approach equality by placing reversal-event mass arbitrarily close to the threshold and the remaining mass at the appropriate boundary.

## Corollary: high class confidence need not imply high decision maturity

Suppose the current posterior is on the low-risk side and ordinary class confidence is \(c=1-p_t\). Then

\[
r_{\max}=\frac{1-c}{\tau}.
\]

For \(C_{FP}=1,C_{FN}=5\), \(\tau=1/6\). A nominally 95%-confident legitimate posterior has \(p_t=.05\), yet the coherent reversal bound is

\[
r_{\max}=.05/(1/6)=.30.
\]

Thus **95% current class confidence is compatible with as much as 30% future Bayes-action reversal probability** under this asymmetric threshold. At 99% confidence the corresponding bound is 6%.

This does not say those reversal rates occur empirically. It says confidence alone cannot rule them out without assumptions on the future evidence process.

## Interpretation

Confidence is a property of the current posterior. Decision maturity is a property of the current posterior plus the future information process. The theorem is an exact mathematical separation and a motivation for the empirical diagnostic, not by itself a claim of a new branch of decision theory.
