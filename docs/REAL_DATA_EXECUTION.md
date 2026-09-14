# Real-data execution gate

## Primary experiment — IEEE-CIS/Vesta

Required local files:

- `train_transaction.csv`
- `train_identity.csv`

Run:

```bash
python experiments/run_ieee_cis_real_validation.py --data-dir /path/to/ieee_cis
```

This is the primary test because it permits a defensible nested evidence-enrichment analysis.

## Secondary experiment — ULB/Worldline

Required local file:

- `creditcard.csv`

Run:

```bash
python experiments/run_ulb_temporal_robustness.py --csv /path/to/creditcard.csv
```

This experiment uses chronological model updating and delayed labels. It is a secondary robustness test only because V1..V28 are anonymized PCA components, not documented stages of evidence arrival.

## Why raw datasets are not committed

The repository intentionally does not redistribute third-party financial data. Users should obtain the source data under the source provider's terms.

## Paper-writing gate

Begin manuscript drafting when the primary IEEE-CIS high-confidence reversal experiment has run and either:

1. maturity/evidence-instability beats ordinary confidence/entropy within the high-confidence subgroup with a positive bootstrap interval; or
2. the result is negative but yields a clear, theoretically supported boundary on when confidence and maturity diverge.

A second real dataset is then used as robustness evidence during manuscript finalization.
