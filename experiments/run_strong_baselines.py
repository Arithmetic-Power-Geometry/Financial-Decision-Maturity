from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fdm.simulator import generate_episode
from fdm.policies import run_policy
from fdm.baselines import confidence_threshold_policy, myopic_voi_policy, optimal_stopping_policy


def evaluate(ep, name, rng, confidence):
    if name == 'fdm': return run_policy(ep, 'fdm', rng)
    if name == 'confidence_threshold': return confidence_threshold_policy(ep, confidence=confidence)
    if name == 'myopic_voi': return myopic_voi_policy(ep, rng)
    if name == 'optimal_stopping': return optimal_stopping_policy(ep, rng)
    return run_policy(ep, name, rng)


def tune_confidence():
    candidates = [0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.99]
    scores = []
    for c in candidates:
        vals = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            for _ in range(200):
                vals.append(confidence_threshold_policy(generate_episode(rng), confidence=c)['cost'])
        scores.append((c,float(np.mean(vals))))
    return min(scores,key=lambda x:x[1]), scores


def main():
    best, tuning = tune_confidence()
    confidence = best[0]
    print('Best training confidence:', best)
    policies = ['fdm','confidence_threshold','myopic_voi','optimal_stopping',
                'immediate','fixed_delay','always_escalate','oracle']
    rows = []
    for seed in range(10,15):
        base_rng = np.random.default_rng(seed)
        episodes = [generate_episode(base_rng) for _ in range(50)]
        for i, ep in enumerate(episodes):
            for j, pol in enumerate(policies):
                rng = np.random.default_rng(seed*100000+i*31+j)
                out = evaluate(ep,pol,rng,confidence)
                y,d = int(ep['y']),int(out['decision'])
                rows.append({'seed':seed,'episode':i,'policy':pol,'cost':float(out['cost']),
                             'time':int(out['t']),'correct':int(d==y),
                             'caught_risky':int(d==1) if y==1 else np.nan})
    df = pd.DataFrame(rows)
    summary = df.groupby('policy',as_index=False).agg(
        mean_cost=('cost','mean'),std_cost=('cost','std'),decision_accuracy=('correct','mean'),
        mean_decision_time=('time','mean'),risky_recall=('caught_risky','mean')).sort_values('mean_cost')
    piv = df.pivot_table(index=['seed','episode'],columns='policy',values='cost')
    paired=[]
    for pol in [p for p in policies if p!='fdm']:
        diff=piv['fdm']-piv[pol]
        paired.append({'comparison':f'fdm_minus_{pol}','mean_difference':float(diff.mean()),
                       'median_difference':float(diff.median()),
                       'fdm_lower_cost_fraction':float((diff<0).mean()),
                       'equal_fraction':float((diff==0).mean())})
    paired=pd.DataFrame(paired)
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    df.to_csv(out/'strong_baseline_sessions.csv',index=False)
    summary.to_csv(out/'strong_baseline_summary.csv',index=False)
    paired.to_csv(out/'strong_baseline_paired.csv',index=False)
    with open(out/'strong_baseline_tuning.json','w',encoding='utf-8') as f:
        json.dump({'best_confidence':confidence,
                   'training_grid':[{'confidence':c,'mean_cost':v} for c,v in tuning]},f,indent=2)
    print('\nHeld-out summary')
    print(summary.to_string(index=False))
    print('\nPaired comparisons')
    print(paired.to_string(index=False))

if __name__ == '__main__':
    main()
