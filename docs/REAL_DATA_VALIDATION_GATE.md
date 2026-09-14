# Real-data validation gate

The next publication gate is external temporal validation of the Premature Financial Decision (PFD) diagnostic on transaction data that preserve chronological order.

## Dataset priority

1. IEEE-CIS Fraud Detection (Vesta real-world e-commerce transactions): preferred because it contains TransactionDT plus card, address, email, device and identity features. Data access is governed by the Kaggle competition/dataset terms and is not redistributed by this repository.
2. ULB/Worldline European credit-card fraud dataset: useful as a real chronological benchmark, but its PCA-anonymized row-level features do not provide a defensible natural evidence-arrival sequence. It is therefore a secondary robustness dataset rather than the primary PFD validation source.
3. Fraud Detection Handbook dataset: excellent reproducible temporal development benchmark, but it is simulated and is not counted as real-data validation.

## Leakage rule

At evidence stage s, every predictor must be computed only from information available at s. Later-stage information may be used only to define the terminal reference action d_T* for evaluation. No future label, future transaction, or later evidence field may enter the current-stage predictor.

## Chronological split

All model fitting, calibration and threshold selection use an earlier time block. Validation uses the next block. Final metrics are reported on a strictly later test block. Random train/test splitting is prohibited for the main result.

## Publication gate

Do not write the final manuscript until:

- the high-confidence reversal result is reproduced on at least one real financial dataset;
- bootstrap confidence intervals or paired resampling support the maturity-vs-confidence difference;
- a second dataset or independent temporal stress test confirms the direction;
- a focused prior-art audit shows the claimed contribution is not merely selective prediction, abstention, optimal stopping, value of information, conformal risk control, or standard temporal uncertainty.
