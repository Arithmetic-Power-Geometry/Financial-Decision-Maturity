# Decisive Strong-Baseline Test

## Question

Does the current FDM controller add decision value beyond strong timing baselines once hyperparameters are tuned on validation data and evaluated on independent held-out episodes?

## Design

- validation seeds: 0--5;
- held-out test seeds: 20--31;
- 350 episodes per validation seed;
- 500 episodes per test seed;
- all policies use the same loss model and evidence stream;
- FDM maturity threshold, confidence-gate threshold, and entropy-gate threshold are selected only on validation data;
- final comparisons use paired episode-level bootstrap intervals.

## Validation-selected thresholds

- FDM maturity threshold: 0.75
- cost-aware confidence threshold: 0.55
- entropy threshold: 0.70

## Held-out result

| Policy | Mean cost | Accuracy | Risky recall | Mean time | Escalation |
|---|---:|---:|---:|---:|---:|
| Terminal reference | 0.281833 | 0.897833 | 0.985447 | 5.0000 | 0.0000 |
| Entropy gate | 0.327446 | 0.971000 | 0.933472 | 0.0425 | 0.7875 |
| Confidence gate | 0.332612 | 0.958833 | 0.941788 | 0.0137 | 0.7780 |
| VOI | 0.333922 | 0.946000 | 0.943867 | 0.0002 | 0.7483 |
| FDM | 0.350134 | 0.935500 | 0.935551 | 0.0005 | 0.7570 |
| Fixed delay | 0.370167 | 0.719167 | 0.939709 | 2.0000 | 0.0000 |
| Always escalate | 0.392667 | 0.970000 | 0.960499 | 0.0000 | 1.0000 |
| Immediate | 0.510833 | 0.539833 | 0.841996 | 0.0000 | 0.0000 |

## Paired bootstrap conclusion

FDM minus VOI mean cost = +0.016212, 95% CI [0.006983, 0.025898].

FDM minus confidence-gate mean cost = +0.017522, 95% CI [0.007668, 0.027329].

FDM minus entropy-gate mean cost = +0.022688, 95% CI [0.012210, 0.032380].

Therefore the present FDM **controller** does not outperform the strong VOI/confidence/entropy baselines in this synthetic environment. The result is not a failure of the maturity *diagnostic*: earlier reversal-prediction tests showed that maturity-related scores contain signal beyond confidence among high-confidence decisions.

## Research consequence

The paper should not be framed as "FDM is the best control policy." The scientifically stronger direction is now:

> Financial Decision Maturity as a diagnostic of premature high-confidence actions.

The next decisive test is real-data evidence-enrichment validation on IEEE-CIS. If the high-confidence reversal signal survives chronological, leakage-controlled real data, manuscript drafting should begin immediately.
