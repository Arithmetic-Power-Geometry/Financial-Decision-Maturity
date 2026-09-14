# Strong baseline comparison

This experiment is a decisive utility/novelty check for Financial Decision Maturity (FDM).

## Baselines

### 1. Confidence threshold
Acts when posterior confidence exceeds a tuned threshold. The threshold is selected on training seeds only and evaluated on disjoint held-out seeds.

### 2. Myopic value of information
At each step compares ACT, WAIT, and ESCALATE using expected next-stage Bayes-risk reduction and explicit delay/review costs.

### 3. Monte Carlo optimal stopping
Uses a finite-horizon recursive continuation-value approximation. It is a stronger timing baseline than fixed delay or myopic one-step VOI.

## Evaluation design

This first decisive pilot uses:
- confidence threshold tuned on seeds 0--2;
- 200 training episodes per seed;
- final comparison on held-out seeds 10--14;
- 50 episodes per held-out seed (250 paired episodes);
- identical generated episodes replayed across policies;
- asymmetric false-negative and false-positive costs;
- paired per-episode cost comparison against FDM.

The pilot size is intentionally modest so the recursive stopping baseline can run in CI. A larger confirmatory run should follow any redesign.

## Interpretation rule

FDM should not be called superior merely because it beats Immediate, Fixed-delay, or Always-escalate. The main question is whether FDM remains competitive with or adds a distinct benefit relative to confidence gating, VOI, and optimal stopping.

If FDM loses consistently to these stronger baselines, the current controller must be redesigned or the contribution narrowed to a diagnostic/theoretical result.
