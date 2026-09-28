import time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
t=time.time(); log=lambda *a: print(f"{time.time()-t:6.0f}s",*a,flush=True)
W="../work_v3"
gt=pd.read_parquet(f"{W}/train_gt.parquet"); gt=gt[gt.matched_entity_ids!=""]
ex=gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m"); tmap=dict(zip(ex.m,ex.source1_entity_id))
cols=["entity_id","raw_name","raw_addr","country"]
s1=pd.read_parquet(f"{W}/train_s1.parquet",columns=cols)
q=pd.concat([pd.read_parquet(f"{W}/train_s{k}.parquet",columns=cols) for k in (2,3)],ignore_index=True)
q=q[~q.raw_name.str.contains("[ऀ-ൿ]")]
m=SentenceTransformer("sentence-transformers/LaBSE",device="cuda"); m.half()
f=lambda d:(d.raw_name+" | "+d.raw_addr).tolist()
for c in ["US","India"]:
    s=s1[s1.country==c].reset_index(drop=True); qs=q[q.country==c].sample(20000,random_state=0)
    qs=qs[qs.entity_id.map(tmap).notna()]
    pos={e:i for i,e in enumerate(s.entity_id)}; tp=qs.entity_id.map(tmap).map(pos).values
    E1=m.encode(f(s),batch_size=2048,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
    Eq=m.encode(f(qs),batch_size=2048,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
    log(c,"encoded",E1.shape)
    hits={1:0,2:0,3:0,5:0,10:0}
    for i in range(0,len(Eq),2048):
        top=torch.topk(Eq[i:i+2048]@E1.T,10,dim=1).indices.cpu().numpy(); tt=tp[i:i+2048]
        for k in hits: hits[k]+=(top[:,:k]==tt[:,None]).any(1).sum()
    log(c,"LaBSE name+addr non-native",{k:round(v/len(qs),4) for k,v in hits.items()})
    np.save(f"../emb/exp2_{c}_ids.npy",qs.entity_id.values.astype(str))
    del E1,Eq; torch.cuda.empty_cache()
