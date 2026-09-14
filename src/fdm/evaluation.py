from __future__ import annotations
import pandas as pd

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("policy", as_index=False)
        .agg(
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_decision_time=("t", "mean"),
            decision_accuracy=("correct", "mean"),
            risky_recall=("caught_risky", "mean"),
        )
    )
