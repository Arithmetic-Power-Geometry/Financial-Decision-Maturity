from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)

def reversal_bound(p, tau):
    if p < tau:
        return min(1.0, p/tau)
    if p > tau:
        return min(1.0, (1-p)/(1-tau))
    return 1.0

rows=[]
confidences=[0.90,0.95,0.975,0.99,0.995]
costs=[(1,1),(1,2),(1,5),(1,10),(2,5),(5,1)]
for fp,fn in costs:
    tau=fp/(fp+fn)
    for c in confidences:
        p=1-c
        rows.append({
            "fp_cost":fp,"fn_cost":fn,"threshold_tau":tau,
            "class_confidence":c,"fraud_posterior":p,
            "current_action":"BLOCK" if p>=tau else "ALLOW",
            "max_coherent_reversal_probability":reversal_bound(p,tau),
        })

df=pd.DataFrame(rows)
df.to_csv(OUT/"confidence_maturity_bound_table.csv",index=False)
print(df.to_string(index=False))
