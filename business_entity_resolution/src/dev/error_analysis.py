import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
from common import load_split, gt_pairs, fbeta_from_assign
from pipeline import assign_from_prob
s1,q=load_split("../work","train"); truth=gt_pairs("../work",s1,q)
cand=pd.read_parquet("../work/train_cand_K5_cap2000.parquet"); cand=cand[cand["rank"]<3].reset_index(drop=True)
oof=np.load("../work/train_oof.npy")
asg=assign_from_prob(cand,oof,len(q),0.625)
n=len(s1)
npred=np.bincount(asg[asg>=0],minlength=n); ntrue=np.bincount(truth[truth>=0],minlength=n)
ok=(asg>=0)&(asg==truth); tp=np.bincount(asg[ok],minlength=n)
fp=npred-tp; fn=ntrue-tp
b2=.25
with np.errstate(all="ignore"):
    pr=np.where(npred>0,tp/npred,0); rc=np.where(ntrue>0,tp/ntrue,0); f=np.where(tp>0,1.25*pr*rc/(b2*pr+rc),0)
f=np.where((npred==0)&(ntrue==0),1,f)
loss=1-f
print("mean F",f.mean())
sing=ntrue==0
print("singletons: n",sing.sum(),"loss share",loss[sing].sum()/loss.sum(),"singleton F",f[sing].mean())
print("loss from S1 with FP>0:",loss[fp>0].sum()/loss.sum()," with FN>0 only:",loss[(fp==0)&(fn>0)].sum()/loss.sum())
# FN decomposition: blocked-out vs rejected by model
inC=np.zeros(len(q),bool); c=cand[truth[cand.q_idx.values]==cand.s1_idx.values]; inC[c.q_idx.values]=True
m=truth>=0
print("missed records:",(m&~ok).sum()," not in candidates:",(m&~inC).sum()," in cands but unassigned:",(m&inC&(asg<0)).sum()," assigned to wrong S1:",(m&inC&(asg>=0)&(asg!=truth)).sum())
print("FP records: distractor assigned:",((truth<0)&(asg>=0)).sum()," true-record wrong S1:",((truth>=0)&(asg>=0)&(asg!=truth)).sum())
# what does F look like if blocking were perfect & assignment perfect for in-cand
rng=np.random.default_rng(0)
print("\n=== sample FP (distractor assigned) ===")
for i in rng.choice(np.flatnonzero((truth<0)&(asg>=0)),8,replace=False):
    j=asg[i]; print("Q:",q.raw_name.values[i],"|",q.raw_addr.values[i]); print("S:",s1.raw_name.values[j],"|",s1.raw_addr.values[j]); print()
print("=== sample wrong-S1 ===")
for i in rng.choice(np.flatnonzero((truth>=0)&(asg>=0)&(asg!=truth)),8,replace=False):
    j=asg[i]; t=truth[i]; print("Q:",q.raw_name.values[i],"|",q.raw_addr.values[i]); print("P:",s1.raw_name.values[j],"|",s1.raw_addr.values[j]); print("T:",s1.raw_name.values[t],"|",s1.raw_addr.values[t]); print()
print("=== sample in-cand unassigned ===")
for i in rng.choice(np.flatnonzero(m&inC&(asg<0)),8,replace=False):
    t=truth[i]; pp=oof[(cand.q_idx.values==i)]; print("Q:",q.raw_name.values[i],"|",q.raw_addr.values[i], "probs",np.round(pp,2)); print("T:",s1.raw_name.values[t],"|",s1.raw_addr.values[t]); print()
