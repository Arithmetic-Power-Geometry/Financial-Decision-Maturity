from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from fdm.simulator import generate_episode
from fdm.core import bayes_decision
from fdm.fdm3 import run_fdm3_policy
from fdm.fdm2 import run_fdm2_policy
from fdm.baselines import confidence_threshold_policy,myopic_voi_policy,optimal_stopping_policy
from fdm.commitment_baseline import commitment_aware_optimal_stopping

def terminal_action(ep,fp,fn):
    return bayes_decision(float(ep["probabilities"][-1]),fp,fn)

def rollback_penalty(d,final,rb_block,rb_allow):
    if d==final: return 0.0
    return rb_block if d==1 and final==0 else rb_allow

def tune_conf_for_scenario(pars):
    vals=[]
    for c in [0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.99]:
        costs=[]
        for seed in range(0,3):
            rng=np.random.default_rng(seed)
            for i in range(80):
                ep=generate_episode(rng)
                o=confidence_threshold_policy(ep,confidence=c,delay_cost_per_step=pars.get("delay_cost",0.035),fp_cost=pars.get("fp_cost",1.0),fn_cost=pars.get("fn_cost",5.0))
                final=terminal_action(ep,pars.get("fp_cost",1.0),pars.get("fn_cost",5.0))
                costs.append(o["cost"]+rollback_penalty(int(o["decision"]),final,pars.get("rb_block",0.2),pars.get("rb_allow",0.8)))
        vals.append((float(np.mean(costs)),c))
    return min(vals)[1]

def run_case(name,pars):
    fp=pars.get("fp_cost",1.0); fn=pars.get("fn_cost",5.0)
    delay=pars.get("delay_cost",0.035); revc=pars.get("review_cost",0.35)
    reva=pars.get("review_accuracy",0.97)
    rbb=pars.get("rb_block",0.2); rba=pars.get("rb_allow",0.8)
    conf=tune_conf_for_scenario(pars)
    rows=[]
    policies=["fdm3","fdm2","confidence","voi","optimal","commitment_optimal"]
    for seed in range(10,18):
        rng=np.random.default_rng(seed)
        for i in range(80):
            ep=generate_episode(rng)
            final=terminal_action(ep,fp,fn)
            for j,pol in enumerate(policies):
                prng=np.random.default_rng(seed*100000+i*41+j)
                common=dict(delay_cost_per_step=delay,review_cost=revc,review_accuracy=reva,fp_cost=fp,fn_cost=fn)
                if pol=="fdm3":
                    o=run_fdm3_policy(ep,prng,regret_weight=0.25,commitment_weight=0.25,rollback_block_cost=rbb,rollback_allow_cost=rba,n_mc=80,**common)
                elif pol=="fdm2":
                    o=run_fdm2_policy(ep,prng,regret_weight=0.5,n_mc=80,**common)
                elif pol=="confidence":
                    o=confidence_threshold_policy(ep,confidence=conf,delay_cost_per_step=delay,fp_cost=fp,fn_cost=fn)
                elif pol=="voi":
                    o=myopic_voi_policy(ep,prng,n_mc=80,**common)
                elif pol=="optimal":
                    o=optimal_stopping_policy(ep,prng,n_mc=30,**common)
                else:
                    o=commitment_aware_optimal_stopping(ep,prng,rollback_block_cost=rbb,rollback_allow_cost=rba,n_mc=50,**common)
                d=int(o["decision"])
                total=float(o["cost"])+rollback_penalty(d,final,rbb,rba)
                rows.append({"scenario":name,"policy":pol,"seed":seed,"episode":i,"base_cost":float(o["cost"]),"rollback_cost":rollback_penalty(d,final,rbb,rba),"total_cost":total,"accuracy":int(d==int(ep["y"])),"time":int(o["t"]),"confidence_threshold":conf})
    return pd.DataFrame(rows)

def main():
    scenarios=[("baseline",{}),("irreversibility_high",{"rb_block":0.5,"rb_allow":2.0}),("fraud_exposure_high",{"rb_block":0.2,"rb_allow":3.0}),("false_negative_high",{"fn_cost":10.0,"rb_allow":2.0}),("review_expensive_irreversible",{"review_cost":0.70,"rb_block":0.5,"rb_allow":2.0})]
    df=pd.concat([run_case(n,p) for n,p in scenarios],ignore_index=True)
    s=(df.groupby(["scenario","policy"],as_index=False).agg(mean_total_cost=("total_cost","mean"),mean_base_cost=("base_cost","mean"),mean_rollback_cost=("rollback_cost","mean"),accuracy=("accuracy","mean"),mean_time=("time","mean"),confidence_threshold=("confidence_threshold","first")))
    out=ROOT/"results";out.mkdir(exist_ok=True)
    df.to_csv(out/"fdm3_fair_sessions.csv",index=False)
    s.to_csv(out/"fdm3_fair_summary.csv",index=False)
    print(s.to_string(index=False))

if __name__=="__main__":
    main()
