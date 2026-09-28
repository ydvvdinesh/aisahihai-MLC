import sys, numpy as np, pandas as pd
from common import load_split
W="../work_v3"; s1,q=load_split(W,"test")
qf=pd.Series(q.a_nums.str.split(" ",n=1).str[0].values,index=q.entity_id.values)
sf=pd.Series(s1.a_nums.str.split(" ",n=1).str[0].values,index=s1.entity_id.values)
qn=pd.Series(q.n_core.values,index=q.entity_id.values); sn=pd.Series(s1.n_core.values,index=s1.entity_id.values)
def pairs(path):
    m=pd.read_csv(path,sep="\t",dtype=str,keep_default_na=False); m=m[m.matched_entity_ids!=""]
    e=m.assign(q=m.matched_entity_ids.str.split(",")).explode("q")[["source1_entity_id","q"]]
    return set(zip(e.source1_entity_id,e.q))
def stats(ps,label):
    d=pd.DataFrame(list(ps),columns=["s","q"])
    a,b=qf.reindex(d.q).values,sf.reindex(d.s).values
    both=(a!="")&(b!="")
    mism=(a!=b)&both
    same_name=qn.reindex(d.q).values==sn.reindex(d.s).values
    print(f"{label:28s} n={len(d):8d}  num-mismatch={mism.sum()/both.sum():.3f}  same-core-name={same_name.mean():.3f}  sibling-like(mismatch&same name)={(mism&same_name).sum()/len(d):.3f}")
base=sys.argv[1]; others=sys.argv[2:]
P={p:pairs(f"../{p}/matching_results.tsv") for p in [base]+others}
stats(P[base],base+" (all)")
for o in others:
    stats(P[o]-P[base],f"{o} minus {base}"); stats(P[base]-P[o],f"{base} minus {o}")
