"""Where does v7 lose F0.5 on train?  2-fold OOF, then per-entity loss attribution."""
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from common import gt_pairs, fbeta_from_assign
from pipeline import PARAMS
from v6 import load, decide
from v7 import prune
t0=time.time(); log=lambda *a: print(f"{time.time()-t0:6.0f}s",*a,flush=True)
W="../work_v3"; meta=json.load(open(f"{W}/model_meta_v7.json")); feats=meta["feats"]; rule=meta["rule"]
s1,q,cand,F=load(W,"train",44); truth=gt_pairs(W,s1,q); cand,F=prune(cand,F,0.5)
y=(truth[cand.q_idx.values]==cand.s1_idx.values).astype(np.int8)
X=F[feats].to_numpy(np.float32); isemb=F.is_emb.values; del F
fold=(cand.s1_idx.values*2654435761%2**32)%2; oof=np.zeros(len(cand),np.float32)
for k in (0,1):
    tr,va=fold!=k,fold==k
    m=lgb.train(dict(PARAMS,num_threads=44),lgb.Dataset(X[tr],y[tr],feature_name=feats),400); oof[va]=m.predict(X[va],num_threads=44)
np.save(f"{W}/train_oof_v7.npy",oof); cand.to_parquet(f"{W}/train_cand_v7.parquet",index=False)
asg=decide(cand,oof,len(q),len(s1),rule); n=len(s1)
log("OOF F", fbeta_from_assign(asg,truth,n))
npred=np.bincount(asg[asg>=0],minlength=n); ntrue=np.bincount(truth[truth>=0],minlength=n)
ok=(asg>=0)&(asg==truth); tp=np.bincount(asg[ok],minlength=n); fp=npred-tp; fn=ntrue-tp
with np.errstate(all="ignore"):
    pr=np.where(npred>0,tp/npred,0); rc=np.where(ntrue>0,tp/ntrue,0); f=np.where(tp>0,1.25*pr*rc/(.25*pr+rc),0)
f=np.where((npred==0)&(ntrue==0),1,f); loss=1-f; L=loss.sum()
log(f"total loss {L:.0f} (= {L/n:.4f} F)")
# what the loss would be if we fixed each error type (oracle)
def F_with(a): return fbeta_from_assign(a,truth,n)
base=F_with(asg)
a=asg.copy(); a[(a>=0)&(a!=truth)]=-1; log(f"remove all false merges      -> F {F_with(a):.4f} (+{F_with(a)-base:.4f})")
inC=np.zeros(len(q),bool); c=cand[y==1]; inC[c.q_idx.values]=True
a=asg.copy(); fix=(truth>=0)&inC&(asg!=truth); a[fix]=truth[fix]; log(f"accept all in-cand true pairs -> F {F_with(a):.4f} (+{F_with(a)-base:.4f})")
a=asg.copy(); fix=(truth>=0)&(asg!=truth); a[fix]=truth[fix]; a[(a>=0)&(a!=truth)]=-1; log(f"perfect everything           -> F {F_with(a):.4f}")
a=asg.copy(); fix=(truth>=0)&inC&(asg!=truth); a[fix]=truth[fix]; a[(a>=0)&(a!=truth)]=-1; log(f"perfect matcher (given cands)-> F {F_with(a):.4f}")
# breakdown of errors
fq=q.a_nums.str.split(" ",n=1).str[0].values; fs=s1.a_nums.str.split(" ",n=1).str[0].values
nat=q.raw_name.str.contains("[ऀ-ൿ]").values; empty=q.a_norm.values==""
def kind(i,j):
    a,b=fq[i],fs[j]
    if not a or not b: return "missing-num"
    if a==b: return "equal"
    if len(a)==len(b): d=abs(int(a[:15])-int(b[:15])); return "samelen<=2" if d<=2 else ("samelen3-50" if d<=50 else "samelen>50")
    if a.startswith(b) or b.startswith(a): return "prefix"
    if a.endswith(b) or b.endswith(a): return "suffix"
    return "other"
rej=np.flatnonzero((truth>=0)&inC&(asg<0)); fpd=np.flatnonzero((truth<0)&(asg>=0)); wr=np.flatnonzero((truth>=0)&(asg>=0)&(asg!=truth)); blk=np.flatnonzero((truth>=0)&~inC)
log("counts: blocked-out",len(blk),"| rejected in-cand",len(rej),"| distractor merged",len(fpd),"| wrong S1",len(wr))
log("blocked-out: empty-addr", empty[blk].mean().round(3), "native", nat[blk].mean().round(3))
log("rejected in-cand by kind:", pd.Series([kind(i,truth[i]) for i in rej]).value_counts(normalize=True).round(3).to_dict(), "empty-addr", empty[rej].mean().round(3))
log("distractor merged by kind:", pd.Series([kind(i,asg[i]) for i in fpd]).value_counts(normalize=True).round(3).to_dict())
rng=np.random.default_rng(1)
print("\n--- rejected true pairs (sample) ---")
pmap=pd.Series(oof,index=cand.q_idx.values*10_000_000+cand.s1_idx.values)
for i in rng.choice(rej,15,replace=False):
    t=truth[i]; p=pmap.get(i*10_000_000+t,np.nan)
    print(f"p={p:.2f} Q: {q.raw_name.values[i]} | {q.raw_addr.values[i]}\n       S1: {s1.raw_name.values[t]} | {s1.raw_addr.values[t]}")
print("\n--- merged distractors (sample) ---")
for i in rng.choice(fpd,10,replace=False):
    j=asg[i]; print(f"Q: {q.raw_name.values[i]} | {q.raw_addr.values[i]}\n   S1: {s1.raw_name.values[j]} | {s1.raw_addr.values[j]}")
