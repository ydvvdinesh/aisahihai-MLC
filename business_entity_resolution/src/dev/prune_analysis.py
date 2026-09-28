import numpy as np, pandas as pd
from common import load_split, gt_pairs
W="../work_v3"
s1,q=load_split(W,"train"); truth=gt_pairs(W,s1,q)
d=pd.read_parquet(f"{W}/train_v5_cand_feat.parquet",columns=["q_idx","s1_idx","score","rank","is_emb"])
hit=truth[d.q_idx.values]==d.s1_idx.values
top=d[d["rank"]==0].set_index("q_idx").score
ratio=d.score.values/np.maximum(top.reindex(d.q_idx.values).values,1e-9)
m=(truth>=0).sum()
print(f"full: recall {hit.sum()/m:.4f} cands/S1 {len(d)/len(s1):.2f}")
for a in [0.5,0.6,0.7,0.8,0.85,0.9]:
    keep=(d["rank"].values==0)|(d.is_emb.values==1)|(ratio>=a)
    print(f"alpha={a}: recall {hit[keep].sum()/m:.4f}  cands/S1 {keep.sum()/len(s1):.2f}")
