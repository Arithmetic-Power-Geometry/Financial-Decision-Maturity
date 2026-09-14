from __future__ import annotations
import numpy as np
from .core import bayes_decision, expected_act_cost, expected_escalation_cost
from .simulator import sample_terminal_posteriors
from .fdm3 import reversal_harm

def _realized_cost(decision:int,y:int,fp_cost:float,fn_cost:float)->float:
    if decision==1 and y==0: return fp_cost
    if decision==0 and y==1: return fn_cost
    return 0.0

def commitment_aware_optimal_stopping(episode:dict,rng:np.random.Generator,*,delay_cost_per_step:float=0.035,review_cost:float=0.35,review_accuracy:float=0.97,fp_cost:float=1.0,fn_cost:float=5.0,rollback_block_cost:float=0.20,rollback_allow_cost:float=0.80,n_mc:int=60)->dict:
    ps=episode["probabilities"]; y=int(episode["y"]); T=len(ps)
    for t,p in enumerate(ps):
        p=float(p)
        futures=sample_terminal_posteriors(p,t,T,rng,n=n_mc)
        act=expected_act_cost(p,fp_cost,fn_cost)
        harm=reversal_harm(p,futures,fp_cost=fp_cost,fn_cost=fn_cost,rollback_block_cost=rollback_block_cost,rollback_allow_cost=rollback_allow_cost)
        act_score=act+harm
        esc=expected_escalation_cost(p,review_cost,review_accuracy,fp_cost,fn_cost)
        if t<T-1:
            future_vals=[]
            idx=np.linspace(0,len(futures)-1,min(12,len(futures))).astype(int)
            for i in idx:
                q=float(futures[i])
                q_futures=sample_terminal_posteriors(q,t+1,T,rng,n=max(20,n_mc//2))
                q_act=expected_act_cost(q,fp_cost,fn_cost)+reversal_harm(q,q_futures,fp_cost=fp_cost,fn_cost=fn_cost,rollback_block_cost=rollback_block_cost,rollback_allow_cost=rollback_allow_cost)
                q_esc=expected_escalation_cost(q,review_cost,review_accuracy,fp_cost,fn_cost)
                future_vals.append(min(q_act,q_esc))
            wait=delay_cost_per_step+float(np.mean(future_vals))
            scores={"ACT":act_score,"WAIT":wait,"ESCALATE":esc}
        else:
            scores={"ACT":act_score,"ESCALATE":esc}
        choice=min(scores,key=scores.get)
        if choice=="WAIT": continue
        if choice=="ESCALATE":
            correct=rng.random()<review_accuracy
            d=y if correct else 1-y
            return {"policy":"commitment_optimal_stopping","t":t,"decision":d,"cost":_realized_cost(d,y,fp_cost,fn_cost)+review_cost+delay_cost_per_step*t}
        d=bayes_decision(p,fp_cost,fn_cost)
        return {"policy":"commitment_optimal_stopping","t":t,"decision":d,"cost":_realized_cost(d,y,fp_cost,fn_cost)+delay_cost_per_step*t}
    raise RuntimeError("commitment-aware optimal stopping failed")
