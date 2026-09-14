# Premature Financial Decision diagnostic result

## Verdict

The first held-out diagnostic experiment produces a **promising but narrow** result.

Across all nonterminal decision stages, ordinary posterior uncertainty (confidence / entropy / margin) is stronger than the current maturity quantities for predicting later action reversal.

However, inside the critical **high-confidence subgroup** (posterior confidence >= 0.90), the ordering changes: Decision Maturity, EMR and Reversal Harm rank later reversals better than confidence/entropy/margin.

This is the first result in the project that supports the intended distinction between **confidence** and **decision maturity**.

## Data split

- Training rows: 10,000 sequential decision states
- Held-out test rows: 15,000
- Test premature-decision prevalence: 26.8%
- High-confidence held-out rows: 6,221
- High-confidence later-reversal prevalence: 3.2953%

## Overall held-out prediction

| Predictor | AUROC | AUPRC | Brier | ECE | Top-10% precision |
|---|---:|---:|---:|---:|---:|
| Confidence | 0.8306 | 0.5104 | 0.1176 | 0.0092 | 0.4353 |
| Entropy | 0.8306 | 0.5104 | 0.1176 | 0.0092 | 0.4353 |
| Margin | 0.8306 | 0.5104 | 0.1176 | 0.0092 | 0.4353 |
| Decision Maturity | 0.7416 | 0.4458 | 0.1690 | 0.0113 | **0.4967** |
| EMR | 0.6786 | 0.3380 | 0.1666 | 0.0097 | 0.1640 |
| Reversal Harm | 0.5535 | 0.2632 | 0.1523 | 0.0099 | 0.1400 |

The global result does **not** support replacing uncertainty with maturity.

## High-confidence decisions only

| Predictor | AUROC | AUPRC | Top-10% precision |
|---|---:|---:|---:|
| Reversal Harm | **0.7084** | **0.06668** | 0.07878 |
| Decision Maturity | 0.7084 | 0.06661 | 0.07878 |
| EMR | 0.7078 | 0.06658 | **0.09003** |
| Confidence | 0.6608 | 0.05738 | 0.05466 |
| Entropy | 0.6608 | 0.05738 | 0.05466 |
| Margin | 0.6608 | 0.05738 | 0.05466 |

Relative to confidence in the high-confidence subgroup:

- Decision Maturity AUROC: approximately +7.2% relative
- Decision Maturity AUPRC: approximately +16.1% relative
- Decision Maturity top-10% precision: approximately +44.1% relative
- EMR top-10% precision: approximately +64.7% relative

## Interpretation

The current evidence supports a narrower hypothesis:

> Standard confidence is strong for reversal prediction in the population as a whole, but maturity-oriented quantities may add information precisely among decisions that already look highly confident.

This is the scientifically interesting regime because a high-confidence decision is exactly where a conventional system is least likely to request more evidence.

## What this does NOT establish

These results are generated from the controlled synthetic sequential benchmark. They do not establish real-bank performance, external validity, or theoretical novelty.

## Next gate

Before writing the manuscript, reproduce the high-confidence reversal result on a real temporally ordered financial dataset and compare against stronger uncertainty/calibration baselines.
