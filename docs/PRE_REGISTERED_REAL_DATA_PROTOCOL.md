# Frozen Real-Data Analysis Protocol

**Frozen:** 14 September 2026  
**Project:** Financial Decision Maturity (FDM)

This document fixes the primary IEEE-CIS/Vesta analysis before the genuine
full-dataset result is inspected. It is intended to prevent post-hoc threshold
changes, baseline substitution, or test-set tuning.

## Primary question

Among financial decisions that are already highly confident, can a
current-stage decision-maturity diagnostic predict whether the cost-sensitive
action will later reverse after richer evidence better than:

1. ordinary posterior confidence; and
2. a supervised posterior-trajectory reversal model?

The empirical paper claim is allowed only if maturity survives both baselines.

## Fixed data split

Transactions are ordered chronologically.

- first 60%: base fraud-model training;
- next 20%: reversal/maturity-model fitting and any permitted model selection;
- final 20%: sealed test.

There is no random row split and no fitting on the final test block.

## Fixed evidence stages

The ordered evidence sets are:

\[
E_1=\text{core transaction/card},
\]

\[
E_2=E_1+\text{context/address/email/behavioral variables},
\]

\[
E_3=E_2+\text{identity/device variables},
\]

\[
E_4=E_3+\text{point-in-time historical variables}.
\]

\(E_4\) is the terminal enriched evidence set. The public dataset does not prove
that these groups literally arrived in this order in production; they are
therefore described as **evidence-enrichment stages**, not observed wall-clock
arrival stages.

## Fixed action and reversal definition

With false-positive cost \(C_{FP}=1\) and false-negative cost \(C_{FN}=5\),

\[
\tau=\frac{C_{FP}}{C_{FP}+C_{FN}}=\frac16.
\]

For stage \(s\),

\[
d_s=\mathbf 1\{p_s\ge\tau\}.
\]

The target is

\[
R_s=\mathbf 1\{d_s\ne d_4\}.
\]

## Fixed high-confidence subgroup

The primary subgroup is

\[
\max(p_s,1-p_s)\ge0.90.
\]

This threshold is fixed before the genuine final test is inspected.

## Fixed predictors

The maturity model is trained to estimate reversal risk from information
available at the current stage only.

The mandatory baselines are:

- ordinary confidence uncertainty;
- entropy and margin as descriptive uncertainty controls;
- a supervised trajectory model using only posterior history observed up to
  the current stage.

The trajectory baseline uses current posterior, distance to the Bayes
threshold, slope, volatility, largest stage-to-stage jump, mean absolute jump,
and threshold-side changes.

## Fixed primary endpoints

Primary:

\[
\text{AUROC}, \qquad \text{Top-10\% Precision}.
\]

Secondary:

\[
\text{AUPRC}, \qquad \text{Brier score}, \qquad \text{ECE}.
\]

AUPRC is secondary because the high-confidence reversal event is expected to
be rare.

## Fixed uncertainty analysis

Use 1,000 paired bootstrap replicates clustered by `card1`, so repeated
transactions from the same card are resampled together. Transaction-level
fallback is permitted only if a subgroup has fewer than 20 usable card
clusters, and the fallback must be reported.

## Pre-specified success rule

The empirical maturity-superiority claim passes only if, for the **same
pre-terminal stage** and the **same primary endpoint**,

\[
CI_{95\%}
(\Delta_{\text{maturity-confidence}})>0
\]

and

\[
CI_{95\%}
(\Delta_{\text{maturity-trajectory}})>0.
\]

Both lower confidence bounds must be strictly positive.

## Failure rule

If the sealed test does not satisfy the rule above:

- do not change the 0.90 subgroup threshold on the test set;
- do not replace the trajectory baseline after seeing its result;
- do not invent FDM-IV/FDM-V to rescue the claim;
- do not tune on the final test block;
- report the negative result or narrow the manuscript claim.

## Leakage controls

Published full-dataset historical aggregates are removed and rebuilt
point-in-time. Terminal-stage actions are outcome labels only and are never
used as test-time maturity inputs.

## Manuscript trigger

The complete Results and Conclusion sections are written only after this frozen
protocol is run on the genuine full dataset.

This document freezes the primary scientific decision rule; later exploratory
analyses must be labeled exploratory.
