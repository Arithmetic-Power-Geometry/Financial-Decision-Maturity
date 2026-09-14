# Financial Decision Maturity (FDM)

**Copyright (C) 2026 Mohammad Amir Khusru Akhtar**  
Licensed under the **Apache License 2.0**.

Financial Decision Maturity (FDM) is a reproducible research software framework for studying **when evolving financial evidence is sufficient to justify action**. It is designed for streaming finance problems such as transaction fraud screening, credit-risk intervention, and other sequential decisions with asymmetric costs.

## Research question

Conventional financial machine learning asks whether an event is risky. FDM asks a second question:

> **Is the currently available evidence mature enough to act now, or should the system wait or escalate?**

The software models an evidence sequence

\[
E_1 \subset E_2 \subset \cdots \subset E_T
\]

and supports three actions:

- **ACT** — take the operational decision now;
- **WAIT** — collect another evidence block;
- **ESCALATE** — route to costly secondary or human review.

For a binary risky/not-risky decision, FDM estimates a current posterior risk, a **decision-maturity score** based on future decision stability, and a cost-aware action.

## Important scope

The current repository is a **controlled computational research testbed**. Its synthetic generator creates sequential finance-like evidence with delayed revelation. It is not a claim of deployment validity, regulatory compliance, or proven performance on any bank's production data.

## Core quantities

For current information \(E_t\), let the current risk-optimal binary decision be \(d_t\). Define

\[
M_t = \Pr(d_T=d_t\mid E_t),
\]

the probability that the current decision agrees with the terminal full-evidence decision. Larger \(M_t\) means the current decision is less likely to reverse.

A cost-aware controller compares:

\[
C_{\text{act}}(t),\quad
C_{\text{wait}}(t),\quad
C_{\text{escalate}}(t)
\]

and chooses the lowest expected cost subject to a configurable maturity gate for irreversible action.

## Repository layout

```text
src/fdm/
  core.py           # posterior, decision maturity, action costs
  simulator.py      # synthetic sequential finance generator
  policies.py       # FDM and baseline policies
  evaluation.py     # experiment metrics
experiments/
  run_benchmark.py  # one-command benchmark
tests/              # unit and consistency tests
results/            # generated CSV/JSON outputs
figures/            # generated plots
.github/workflows/
  ci.yml            # tests + reproducibility artifact
```

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
python experiments/run_benchmark.py
```

The benchmark writes:

- `results/session_results.csv`
- `results/summary.csv`
- `results/summary.json`
- `figures/cost_by_policy.png`
- `figures/decision_accuracy_by_policy.png`

## Policies

The benchmark compares:

1. **Immediate** — act at the first observation.
2. **Fixed-delay** — wait to a preset evidence stage, then act.
3. **Always-escalate** — use costly review at the first stage.
4. **Full-information oracle** — wait until the terminal evidence stage.
5. **FDM** — use maturity plus expected cost to choose ACT / WAIT / ESCALATE.

The oracle is an upper-information reference, not a deployable baseline.

## Reproducibility

The GitHub Actions workflow:

1. installs the package;
2. runs the unit tests;
3. runs the benchmark;
4. verifies generated result files;
5. uploads `results/` and `figures/` as a workflow artifact.

## Research claims this software can test

The software is intended to test, not presuppose, the following hypotheses:

- acting on immature evidence can increase expected financial loss;
- waiting can be beneficial when expected information value exceeds delay cost;
- expensive escalation can dominate both premature action and excessive waiting in an intermediate region;
- prediction confidence and decision maturity need not be identical;
- the cost-minimizing action can change before the terminal label is known.

## Citation

If this artifact is used in research, cite the repository and the associated manuscript once available.

## License

Apache License 2.0. See `LICENSE`.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar.
