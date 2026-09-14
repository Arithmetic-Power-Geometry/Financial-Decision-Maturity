from __future__ import annotations
import numpy as np
from .core import bayes_decision, expected_act_cost, expected_escalation_cost
from .simulator import sample_terminal_posteriors


def realized(decision, y, fp=1.0, fn=5.0):
    if decision == 1 and y == 0:
        return fp
    if decision == 0 and y == 1:
        return fn
    return 0.0


def confidence_threshold_policy(episode, confidence=0.90, delay_cost_per_step=0.035,
                                fp_cost=1.0, fn_cost=5.0):
    ps, y = episode['probabilities'], int(episode['y'])
    for t, p in enumerate(ps):
        if max(float(p), 1.0-float(p)) >= confidence or t == len(ps)-1:
            d = bayes_decision(float(p), fp_cost, fn_cost)
            return {'policy':'confidence_threshold','t':t,'decision':d,
                    'cost':realized(d,y,fp_cost,fn_cost)+delay_cost_per_step*t}
    raise RuntimeError('confidence policy failed')


def myopic_voi_policy(episode, rng, delay_cost_per_step=0.035, review_cost=0.35,
                      review_accuracy=0.97, fp_cost=1.0, fn_cost=5.0, n_mc=500):
    ps, y, T = episode['probabilities'], int(episode['y']), len(episode['probabilities'])
    for t, p0 in enumerate(ps):
        p = float(p0)
        act = expected_act_cost(p, fp_cost, fn_cost)
        esc = expected_escalation_cost(p, review_cost, review_accuracy, fp_cost, fn_cost)
        if t == T-1:
            choice = 'ACT' if act <= esc else 'ESCALATE'
        else:
            fut = sample_terminal_posteriors(p, t, T, rng, n=n_mc)
            wait = delay_cost_per_step + float(np.mean([expected_act_cost(float(q),fp_cost,fn_cost) for q in fut]))
            choice = min({'ACT':act,'WAIT':wait,'ESCALATE':esc}, key={'ACT':act,'WAIT':wait,'ESCALATE':esc}.get)
        if choice == 'WAIT':
            continue
        if choice == 'ESCALATE':
            d = y if rng.random() < review_accuracy else 1-y
            return {'policy':'myopic_voi','t':t,'decision':d,
                    'cost':realized(d,y,fp_cost,fn_cost)+review_cost+delay_cost_per_step*t}
        d = bayes_decision(p, fp_cost, fn_cost)
        return {'policy':'myopic_voi','t':t,'decision':d,
                'cost':realized(d,y,fp_cost,fn_cost)+delay_cost_per_step*t}
    raise RuntimeError('VOI policy failed')


def optimal_stopping_policy(episode, rng, delay_cost_per_step=0.035, review_cost=0.35,
                            review_accuracy=0.97, fp_cost=1.0, fn_cost=5.0, n_mc=40):
    ps, y, T = episode['probabilities'], int(episode['y']), len(episode['probabilities'])
    def value(p, t, depth=0):
        act = expected_act_cost(p, fp_cost, fn_cost)
        esc = expected_escalation_cost(p, review_cost, review_accuracy, fp_cost, fn_cost)
        if t >= T-1:
            return min(act, esc)
        fut = sample_terminal_posteriors(p, t, T, rng, n=max(10,n_mc//(depth+1)))
        if t >= T-2:
            vals = [min(expected_act_cost(float(q),fp_cost,fn_cost),
                        expected_escalation_cost(float(q),review_cost,review_accuracy,fp_cost,fn_cost)) for q in fut]
        else:
            idx = np.linspace(0,len(fut)-1,min(4,len(fut))).astype(int)
            vals = [value(float(fut[i]),t+1,depth+1) for i in idx]
        return min(act, esc, delay_cost_per_step + float(np.mean(vals)))
    for t, p0 in enumerate(ps):
        p = float(p0)
        act = expected_act_cost(p, fp_cost, fn_cost)
        esc = expected_escalation_cost(p, review_cost, review_accuracy, fp_cost, fn_cost)
        costs = {'ACT':act,'ESCALATE':esc}
        if t < T-1:
            fut = sample_terminal_posteriors(p,t,T,rng,n=max(16,n_mc//3))
            idx = np.linspace(0,len(fut)-1,min(4,len(fut))).astype(int)
            costs['WAIT'] = delay_cost_per_step + float(np.mean([value(float(fut[i]),t+1,1) for i in idx]))
        choice = min(costs,key=costs.get)
        if choice == 'WAIT':
            continue
        if choice == 'ESCALATE':
            d = y if rng.random() < review_accuracy else 1-y
            return {'policy':'optimal_stopping','t':t,'decision':d,
                    'cost':realized(d,y,fp_cost,fn_cost)+review_cost+delay_cost_per_step*t}
        d = bayes_decision(p,fp_cost,fn_cost)
        return {'policy':'optimal_stopping','t':t,'decision':d,
                'cost':realized(d,y,fp_cost,fn_cost)+delay_cost_per_step*t}
    raise RuntimeError('optimal stopping failed')
