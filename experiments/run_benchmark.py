from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fdm.simulator import generate_episode
from fdm.policies import run_policy
from fdm.evaluation import summarize

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

def main():
    policies = ["immediate", "fixed_delay", "always_escalate", "oracle", "fdm"]
    rows = []
    n_seeds = 20
    episodes_per_seed = 1000

    for seed in range(n_seeds):
        base_rng = np.random.default_rng(seed)
        episodes = [generate_episode(base_rng) for _ in range(episodes_per_seed)]
        for i, ep in enumerate(episodes):
            for j, pol in enumerate(policies):
                rng = np.random.default_rng(seed * 100000 + i * 17 + j)
                r = run_policy(ep, pol, rng)
                y = int(ep["y"])
                d = int(r["decision"])
                r.update({
                    "seed": seed,
                    "episode": i,
                    "y": y,
                    "correct": int(d == y),
                    "caught_risky": int(d == 1) if y == 1 else np.nan,
                })
                rows.append(r)

    df = pd.DataFrame(rows)
    summary = summarize(df)

    df.to_csv(RESULTS / "session_results.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["policy"], summary["mean_cost"])
    ax.set_ylabel("Mean realized cost")
    ax.set_title("Financial Decision Maturity benchmark: cost")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_by_policy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["policy"], summary["decision_accuracy"])
    ax.set_ylabel("Decision accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Financial Decision Maturity benchmark: decision accuracy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "decision_accuracy_by_policy.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
