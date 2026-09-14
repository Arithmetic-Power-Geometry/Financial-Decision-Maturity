from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from fdm.simulator import generate_episode
from fdm.diagnostics import diagnostic_scores, premature_financial_decision
from fdm.diagnostic_metrics import roc_auc, average_precision, brier, ece, top_fraction_precision, empirical_calibrate

FEATURES = {
    'confidence': 'confidence_reversal_score',
    'entropy': 'entropy',
    'margin': 'margin_reversal_score',
    'decision_maturity': 'maturity_reversal_score',
    'emr': 'emr',
    'reversal_harm': 'reversal_harm',
}

def make_rows(seeds, episodes_per_seed=400):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for episode_id in range(episodes_per_seed):
            ep = generate_episode(rng, steps=6)
            ps = ep['probabilities']; terminal_p = float(ps[-1])
            for t in range(len(ps)-1):
                p = float(ps[t])
                drng = np.random.default_rng(seed*1000000 + episode_id*37 + t)
                scores = diagnostic_scores(p, t, len(ps), drng, n_mc=180)
                rows.append({'seed': seed, 'episode': episode_id, 't': t, 'current_p': p, 'terminal_p': terminal_p, 'pfd': premature_financial_decision(p, terminal_p), **scores})
    return pd.DataFrame(rows)

def main():
    train = make_rows(range(0,8), episodes_per_seed=250)
    test = make_rows(range(20,32), episodes_per_seed=250)
    out_dir = ROOT/'results'; out_dir.mkdir(exist_ok=True)
    records = []
    for name,col in FEATURES.items():
        test_prob = empirical_calibrate(train[col].to_numpy(), train['pfd'].to_numpy(), test[col].to_numpy(), bins=20)
        records.append({'predictor': name, 'auroc': roc_auc(test['pfd'], test[col]), 'auprc': average_precision(test['pfd'], test[col]), 'brier': brier(test['pfd'], test_prob), 'ece': ece(test['pfd'], test_prob, bins=10), 'top10_precision': top_fraction_precision(test['pfd'], test[col], 0.10)})
    metrics = pd.DataFrame(records).sort_values(['auprc','auroc'], ascending=False)
    high_conf = test[test['confidence'] >= 0.90].copy(); high_conf_rate = float(high_conf['pfd'].mean()) if len(high_conf) else float('nan')
    subgroup_records=[]
    for name,col in FEATURES.items():
        if len(high_conf):
            subgroup_records.append({'predictor':name,'n_high_confidence':len(high_conf),'high_confidence_reversal_rate':high_conf_rate,'auroc_within_high_confidence':roc_auc(high_conf['pfd'],high_conf[col]),'auprc_within_high_confidence':average_precision(high_conf['pfd'],high_conf[col]),'top10_precision_within_high_confidence':top_fraction_precision(high_conf['pfd'],high_conf[col],0.10)})
    subgroup = pd.DataFrame(subgroup_records).sort_values(['auprc_within_high_confidence','auroc_within_high_confidence'],ascending=False)
    test.to_csv(out_dir/'diagnostic_test_rows.csv',index=False); train.to_csv(out_dir/'diagnostic_train_rows.csv',index=False); metrics.to_csv(out_dir/'diagnostic_metrics.csv',index=False); subgroup.to_csv(out_dir/'high_confidence_reversal_metrics.csv',index=False)
    print('Train rows:',len(train),'Test rows:',len(test)); print('Test PFD prevalence:',round(float(test['pfd'].mean()),6)); print('High-confidence rows:',len(high_conf)); print('High-confidence later-reversal rate:',round(high_conf_rate,6)); print('\nOverall diagnostic comparison'); print(metrics.to_string(index=False)); print('\nHigh-confidence subgroup'); print(subgroup.to_string(index=False))

if __name__ == '__main__':
    main()
