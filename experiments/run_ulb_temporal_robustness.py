"""ULB/Worldline chronological robustness experiment.

Purpose
-------
This is a *secondary real-data robustness test*, not the primary evidence-
enrichment experiment. The ULB dataset contains real transactions and a genuine
relative time variable, but its V1..V28 features are PCA-anonymized and do not
form a semantically defensible evidence-arrival sequence.

The script therefore tests whether high-confidence decisions are temporally
fragile under model updating / delayed-label conditions.

Expected input
--------------
creditcard.csv with columns:
Time, V1..V28, Amount, Class
"""

from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def entropy(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def rank_metrics(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    if len(np.unique(y))<2:
        return {"auroc":float("nan"),"auprc":float("nan")}
    return {"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--delay", type=int, default=2000,
                    help="transactions before a label becomes available")
    ap.add_argument("--warmup", type=int, default=30000)
    ap.add_argument("--snapshot-gap", type=int, default=5000,
                    help="future model-update horizon used to define reversal")
    ap.add_argument("--fp-cost", type=float, default=1.0)
    ap.add_argument("--fn-cost", type=float, default=5.0)
    args=ap.parse_args()

    path=Path(args.csv)
    df=pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    feats=[c for c in df.columns if c not in {"Class"}]
    X=df[feats].astype(float).to_numpy()
    y=df["Class"].astype(int).to_numpy()
    threshold=args.fp_cost/(args.fp_cost+args.fn_cost)

    scaler=StandardScaler()
    scaler.fit(X[:args.warmup])
    Xs=scaler.transform(X)

    model=SGDClassifier(loss="log_loss",class_weight={0:1.0,1:20.0},random_state=0,learning_rate="optimal")
    init_end=max(1000,args.warmup-args.delay)
    model.partial_fit(Xs[:init_end],y[:init_end],classes=np.array([0,1]))

    records=[]; pending=[]; snapshots={}
    for i in range(args.warmup,len(df)):
        release_until=i-args.delay
        while pending and pending[0][0] <= release_until:
            j=pending.pop(0)[1]
            model.partial_fit(Xs[j:j+1],y[j:j+1])

        p=float(model.predict_proba(Xs[i:i+1])[0,1])
        a=int(p>=threshold); conf=max(p,1-p)
        rec={"i":i,"time":float(df.loc[i,"Time"]),"y":int(y[i]),"p_now":p,"a_now":a,
             "confidence":conf,"uncertainty":1-conf,"entropy":float(entropy([p])[0]),
             "margin_uncertainty":1-abs(2*p-1)}
        records.append(rec); snapshots[i]=Xs[i].copy(); pending.append((i,i))

        j=i-args.snapshot_gap
        if j in snapshots:
            p_future=float(model.predict_proba(snapshots[j].reshape(1,-1))[0,1])
            a_future=int(p_future>=threshold)
            records[j-args.warmup]["p_future"]=p_future
            records[j-args.warmup]["a_future"]=a_future
            records[j-args.warmup]["reversed_after_updates"]=int(records[j-args.warmup]["a_now"] != a_future)
            del snapshots[j]

    out=pd.DataFrame(records)
    out=out[out["reversed_after_updates"].notna()].copy()
    out["reversed_after_updates"]=out["reversed_after_updates"].astype(int)

    rows=[]
    for subgroup,mask in [("all",np.ones(len(out),dtype=bool)),("high_confidence",out["confidence"].to_numpy()>=0.90)]:
        for pred in ["uncertainty","entropy","margin_uncertainty"]:
            m=rank_metrics(out.loc[mask,"reversed_after_updates"],out.loc[mask,pred])
            rows.append({"subgroup":subgroup,"predictor":pred,"n":int(mask.sum()),
                         "reversal_rate":float(out.loc[mask,"reversed_after_updates"].mean()),**m})

    results=Path("results"); results.mkdir(exist_ok=True)
    out.to_csv(results/"ulb_temporal_reversal_rows.csv",index=False)
    pd.DataFrame(rows).to_csv(results/"ulb_temporal_reversal_metrics.csv",index=False)
    meta={"source_file":str(path),"sha256":sha256(path),"rows":len(df),"frauds":int(y.sum()),
          "fraud_rate":float(y.mean()),"delay":args.delay,"warmup":args.warmup,
          "snapshot_gap":args.snapshot_gap,"threshold":threshold,
          "interpretation":"Secondary temporal robustness only; not an evidence-arrival sequence."}
    (results/"ulb_temporal_reversal_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
