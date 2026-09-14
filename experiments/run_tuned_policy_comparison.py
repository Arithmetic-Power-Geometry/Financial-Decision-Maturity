from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from fdm.simulator import generate_episode
from fdm.strong_policies import run_strong_policy

RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)

def make_episodes(seeds, n_per_seed=350):
    data=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        for i in range(n_per_seed):
            data.append((seed,i,generate_episode(rng)))
    return data

def evaluate(data, policy, **kwargs):
    vals=[]
    for seed,i,ep in data:
        rng=np.random.default_rng(seed*1000003+i*101+hash(policy)%9973)
        r=run_strong_policy(ep,policy,rng,**kwargs)
        vals.append(r["cost"])
    return float(np.mean(vals))

def tune(data):
    rows=[]
    for x in np.arange(0.50,1.00,0.05):
        rows.append({"policy":"fdm","parameter":"maturity_threshold","value":float(x),"validation_cost":evaluate(data,"fdm",maturity_threshold=float(x))})
    for x in np.arange(0.30,0.96,0.05):
        rows.append({"policy":"confidence_gate","parameter":"certainty_threshold","value":float(x),"validation_cost":evaluate(data,"confidence_gate",certainty_threshold=float(x))})
    for x in np.arange(0.10,0.91,0.05):
        rows.append({"policy":"entropy_gate","parameter":"entropy_threshold","value":float(x),"validation_cost":evaluate(data,"entropy_gate",entropy_threshold=float(x))})
    tab=pd.DataFrame(rows)
    best=tab.sort_values("validation_cost").groupby("policy",as_index=False).first()
    return tab,best

def full_test(data, best):
    cfg={
        "fdm":{"maturity_threshold":float(best.loc[best.policy=="fdm","value"].iloc[0])},
        "confidence_gate":{"certainty_threshold":float(best.loc[best.policy=="confidence_gate","value"].iloc[0])},
        "entropy_gate":{"entropy_threshold":float(best.loc[best.policy=="entropy_gate","value"].iloc[0])},
    }
    policies=["immediate","fixed_delay","always_escalate","voi","fdm","confidence_gate","entropy_gate","terminal"]
    rows=[]
    for seed,i,ep in data:
        y=int(ep["y"])
        for j,pol in enumerate(policies):
            rng=np.random.default_rng(seed*1000003+i*101+j)
            r=run_strong_policy(ep,pol,rng,**cfg.get(pol,{}))
            d=int(r["decision"])
            rows.append({**r,"seed":seed,"episode":i,"y":y,"correct":int(d==y),"caught_risky":int(d==1) if y==1 else np.nan,"escalated":int(r.get("action")=="ESCALATE")})
    df=pd.DataFrame(rows)
    summ=(df.groupby("policy",as_index=False).agg(mean_cost=("cost","mean"),mean_time=("t","mean"),accuracy=("correct","mean"),risky_recall=("caught_risky","mean"),escalation_rate=("escalated","mean")).sort_values("mean_cost"))
    return df,summ,cfg

def bootstrap(df, B=500):
    piv=df.pivot_table(index=["seed","episode"],columns="policy",values="cost")
    rng=np.random.default_rng(20260914)
    ref=piv["fdm"].to_numpy(); rows=[]
    for pol in ["voi","confidence_gate","entropy_gate","fixed_delay","always_escalate"]:
        d=ref-piv[pol].to_numpy(); boots=[]
        for _ in range(B):
            idx=rng.integers(0,len(d),len(d)); boots.append(float(d[idx].mean()))
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append({"comparison":f"fdm_minus_{pol}","mean_delta":float(d.mean()),"ci95_low":float(lo),"ci95_high":float(hi),"prob_fdm_cheaper":float(np.mean(np.asarray(boots)<0))})
    return pd.DataFrame(rows)

def main():
    validation=make_episodes(range(0,6),350)
    test=make_episodes(range(20,32),500)
    grid,best=tune(validation)
    test_rows,summary,cfg=full_test(test,best)
    boot=bootstrap(test_rows)
    grid.to_csv(RESULTS/"policy_tuning_grid.csv",index=False)
    best.to_csv(RESULTS/"policy_tuning_best.csv",index=False)
    test_rows.to_csv(RESULTS/"tuned_policy_test_rows.csv",index=False)
    summary.to_csv(RESULTS/"tuned_policy_test_summary.csv",index=False)
    boot.to_csv(RESULTS/"tuned_policy_bootstrap.csv",index=False)
    (RESULTS/"tuned_policy_result.json").write_text(json.dumps({"selected_parameters":cfg,"summary":summary.to_dict(orient="records"),"bootstrap":boot.to_dict(orient="records"),"design":{"validation_seeds":"0-5","test_seeds":"20-31","episodes_per_validation_seed":350,"episodes_per_test_seed":500}},indent=2),encoding="utf-8")
    print("=== SELECTED PARAMETERS ==="); print(json.dumps(cfg,indent=2))
    print("\n=== HELD-OUT TEST ==="); print(summary.to_string(index=False))
    print("\n=== FDM PAIRED BOOTSTRAP ==="); print(boot.to_string(index=False))

if __name__=="__main__":
    main()
