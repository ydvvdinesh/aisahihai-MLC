"""Do true-match house-number noise and sibling look-alikes differ in edit type?"""
import numpy as np, pandas as pd
from common import load_split, gt_pairs
W="../work_v3"
s1,q=load_split(W,"train"); truth=gt_pairs(W,s1,q)
c=pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet"); c=c[c["rank"]==0]
fq=q.a_nums.str.split(" ",n=1).str[0].values[c.q_idx.values]; fs=s1.a_nums.str.split(" ",n=1).str[0].values[c.s1_idx.values]
ok=(fq!="")&(fs!="")&(fq!=fs); fq,fs=fq[ok],fs[ok]
pos=(truth[c.q_idx.values]==c.s1_idx.values)[ok]
def kind(a,b):
    if len(a)==len(b): return "same-len"
    if a.startswith(b) or b.startswith(a): return "prefix"
    if a.endswith(b) or b.endswith(a): return "suffix"
    if b in a or a in b: return "substring"
    return "other"
k=np.array([kind(a,b) for a,b in zip(fq,fs)])
df=pd.DataFrame({"kind":k,"pos":pos})
sl=k=="same-len"; ai=np.array([int(x) for x in fq[sl]]); bi=np.array([int(x) for x in fs[sl]])
df.loc[sl,"diff"]=np.abs(ai-bi)
print(df.groupby("kind").pos.agg(["size","mean"]).round(3))
d=df[sl].copy(); d["bin"]=pd.cut(d["diff"],[0,2,5,10,20,50,100,1e9])
print(d.groupby("bin",observed=True).pos.agg(["size","mean"]).round(3))
