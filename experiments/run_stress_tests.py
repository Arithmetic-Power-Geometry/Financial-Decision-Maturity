from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm.simulator import generate_episode
from fdm.policies import run_policy

POLICIES = ["immediate", "fixed_delay", "always_escalate", "oracle", "fdm"]

def evaluate_scenario(name, n_seeds=10, episodes_per_seed=500,
                      prevalence_shift=1.0, score_shift=0.0,
                      label_noise=0.0, delay_cost=0.035,
                      review_cost=0.35, review_accuracy=0.97,
                      fp_cost=1.0, fn_cost=5.0,
                      maturity_threshold=0.93):
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)
        for i in range(episodes_per_seed):
            ep = generate_episode(rng)
            if prevalence_shift != 1.0:
                base = min(max(0.08 * prevalence_shift, 0.001), 0.95)
                ep["y"] = int(rng.random() < base)
            if score_shift != 0.0:
                logits = np.log(np.clip(ep["probabilities"], 1e-6, 1-1e-6) /
                                np.clip(1-ep["probabilities"], 1e-6, 1-1e-6))
                ep["probabilities"] = 1/(1+np.exp(-(logits + score_shift)))
            if label_noise > 0 and rng.random() < label_noise:
                ep["y"] = 1 - int(ep["y"])

            for j, pol in enumerate(POLICIES):
                prng = np.random.default_rng(seed*100000 + i*31 + j)
                out = run_policy(
                    ep, pol, prng,
                    maturity_threshold=maturity_threshold,
                    delay_cost_per_step=delay_cost,
                    review_cost=review_cost,
                    review_accuracy=review_accuracy,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                )
                d, y = int(out["decision"]), int(ep["y"])
                rows.append({
                    "scenario": name,
                    "policy": pol,
                    "cost": out["cost"],
                    "time": out["t"],
                    "correct": int(d == y),
                    "caught_risky": int(d == 1) if y == 1 else np.nan,
                })
    return pd.DataFrame(rows)

def main():
    scenarios = [
        ("baseline", dict()),
        ("prevalence_x2", dict(prevalence_shift=2.0)),
        ("prior_shift_low", dict(prevalence_shift=0.5)),
        ("calibration_up", dict(score_shift=0.75)),
        ("calibration_down", dict(score_shift=-0.75)),
        ("label_noise_10pct", dict(label_noise=0.10)),
        ("high_delay_cost", dict(delay_cost=0.10)),
        ("high_review_cost", dict(review_cost=0.70)),
        ("lower_review_quality", dict(review_accuracy=0.90)),
        ("fn_cost_x2", dict(fn_cost=10.0)),
        ("fp_cost_x2", dict(fp_cost=2.0)),
        ("strict_maturity", dict(maturity_threshold=0.98)),
        ("loose_maturity", dict(maturity_threshold=0.85)),
    ]

    all_df = pd.concat(
        [evaluate_scenario(name, **kwargs) for name, kwargs in scenarios],
        ignore_index=True
    )
    summary = (
        all_df.groupby(["scenario","policy"], as_index=False)
        .agg(
            mean_cost=("cost","mean"),
            decision_accuracy=("correct","mean"),
            mean_decision_time=("time","mean"),
            risky_recall=("caught_risky","mean"),
        )
    )

    out_dir = ROOT/"results"
    out_dir.mkdir(exist_ok=True)
    all_df.to_csv(out_dir/"stress_session_results.csv", index=False)
    summary.to_csv(out_dir/"stress_summary.csv", index=False)
    with open(out_dir/"stress_summary.json","w",encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
