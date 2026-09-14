# Strong-baseline kill-test result

## Verdict

**The current FDM controller does not survive the strong-baseline test.**

On the held-out 250-episode pilot, a tuned confidence-threshold policy has the lowest mean realized cost (0.12950), followed by the Monte Carlo optimal-stopping approximation (0.30802) and myopic VOI (0.32720). FDM has mean cost 0.39262.

This is a useful falsification result. It means the present implementation must not be advertised as superior to standard decision-timing baselines.

## Held-out results

| Policy | Mean cost | Decision accuracy | Mean decision time | Risky recall |
|---|---:|---:|---:|---:|
| Confidence threshold | 0.12950 | 0.964 | 1.300 | 0.8696 |
| Oracle terminal-evidence reference | 0.29100 | 0.884 | 5.000 | 1.0000 |
| Optimal stopping approximation | 0.30802 | 0.964 | 1.252 | 1.0000 |
| Myopic VOI | 0.32720 | 0.952 | 0.000 | 0.9565 |
| Fixed delay | 0.35800 | 0.728 | 2.000 | 0.9565 |
| Always escalate | 0.38200 | 0.968 | 0.000 | 1.0000 |
| FDM | 0.39262 | 0.936 | 0.052 | 0.9565 |
| Immediate | 0.46000 | 0.556 | 0.000 | 0.9565 |

## Paired evidence

Relative to the confidence-threshold policy, FDM is lower cost on only 5.2% of paired episodes and has a mean excess cost of 0.26312.

Relative to myopic VOI, FDM has a mean excess cost of 0.06542.

Relative to the optimal-stopping approximation, FDM has a mean excess cost of 0.08460. FDM is lower cost on 48.4% of paired episodes, indicating that the difference is heterogeneous rather than a universal domination.

## What failed

The present FDM maturity gate adds a constraint before ACT, but its future stability estimate is not sufficiently aligned with realized economic loss. A simple tuned confidence threshold is much more effective in the current synthetic environment.

## Research consequence

The correct next move is **redesign, not manuscript inflation**.

A stronger FDM-II should make maturity an economically calibrated quantity: the expected cost of future action reversal, not merely the probability of future action agreement. The next candidate quantity is

`EMR_t = E[C(a_t,Y)-C(a_T*,Y) | E_t]`,

an **Expected Maturity Regret**. ACT would be permitted only when the expected regret of acting now is below both the value of waiting and the cost of escalation.

This creates a harder target: FDM-II must beat or match the tuned confidence, VOI, and optimal-stopping baselines on held-out data before any strong novelty claim is made.
