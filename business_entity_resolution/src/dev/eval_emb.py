import time, numpy as np, pandas as pd
from common import load_split, gt_pairs
from emb_candidates import add_emb_candidates
t=time.time(); log=lambda *a: print(f"{time.time()-t:6.0f}s",*a,flush=True)
W="../work_v3"
s1,q=load_split(W,"train"); truth=gt_pairs(W,s1,q)
cand=pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet"); cand=cand[cand["rank"]<3].reset_index(drop=True)
knn=pd.read_parquet("../emb/train_labse_knn.parquet")
nat=q.raw_name.str.contains("[ऀ-ൿ]").values; m=truth>=0
def rec(c):
    h=np.zeros(len(q),bool); hit=truth[c.q_idx.values]==c.s1_idx.values; h[c.q_idx.values[hit]]=True
    return h[m&nat].mean(), h[m].mean(), len(c)/len(s1)
log("blocking top3: native %.4f overall %.4f cands/S1 %.2f"%rec(cand))
for extra in (1,2,3):
    c2,cos,ise=add_emb_candidates(cand,knn,s1,q,extra=extra,procs=40)
    log(f"+LaBSE extra={extra}: native %.4f overall %.4f cands/S1 %.2f"%rec(c2))
