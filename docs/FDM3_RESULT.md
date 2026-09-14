# FDM-III fair irreversibility benchmark

Local validation: **14 tests passed**.

A fair benchmark was run with the same rollback/irreversibility penalty applied to all policies, and the confidence threshold was re-tuned per scenario on training seeds only.

## Key held-out mean total costs

| Scenario | Confidence | VOI | Optimal | Commitment-optimal | FDM-II | FDM-III |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **0.1914** | 0.4231 | 0.3891 | 0.4288 | 0.4447 | 0.4288 |
| High irreversibility | **0.2533** | 0.5511 | 0.5166 | 0.5813 | 0.5741 | 0.5584 |
| High fraud exposure | **0.2534** | 0.6362 | 0.6091 | 0.6767 | 0.6510 | 0.6373 |
| High false-negative cost | **0.4737** | 0.9687 | 0.9865 | 1.0200 | 0.9498 | 0.9458 |
| Expensive review + irreversibility | **0.2533** | 0.3780 | 0.4253 | 0.4217 | 0.3890 | 0.3827 |

## Verdict

FDM-III improves over FDM-II in several path-dependent regimes, but it still does **not** beat the tuned confidence threshold. It also remains close to standard sequential decision baselines, which confirms that path-dependent cost alone is not yet a sufficiently distinct algorithmic contribution.

## Consequence

Do not write the full manuscript yet. The next research step should shift from trying to beat optimal stopping with another controller toward validating a distinct **decision-maturity diagnostic** on real financial data: whether maturity predicts future action reversal and costly premature commitments better than posterior confidence alone.
