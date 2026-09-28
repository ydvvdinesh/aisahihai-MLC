import time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
t=time.time(); log=lambda *a: print(f"{time.time()-t:6.0f}s",*a,flush=True)
G="../emb"; W="../work_v3"
s1=pd.read_parquet(f"{G}/train_s1_names.parquet"); q=pd.read_parquet(f"{G}/train_native_q.parquet").sample(20000,random_state=0)
gt=pd.read_parquet(f"{W}/train_gt.parquet"); gt=gt[gt.matched_entity_ids!=""]
ex=gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m"); tmap=dict(zip(ex.m,ex.source1_entity_id))
q["true"]=q.entity_id.map(tmap); q=q[q.true.notna()]
s1pos={e:i for i,e in enumerate(s1.entity_id)}; tpos=q.true.map(s1pos).values
for name in ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","sentence-transformers/LaBSE"]:
    m=SentenceTransformer(name,device="cuda"); m.half()
    for mode in ["name","name+addr"]:
        f=(lambda d: d.raw_name) if mode=="name" else (lambda d: d.raw_name+" | "+d.raw_addr)
        E1=m.encode(f(s1).tolist(),batch_size=1024,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
        Eq=m.encode(f(q).tolist(),batch_size=1024,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
        log(name,mode,"encoded",E1.shape)
        hits={1:0,5:0,20:0,100:0}
        for i in range(0,len(Eq),2048):
            top=torch.topk(Eq[i:i+2048]@E1.T,100,dim=1).indices.cpu().numpy()
            tp=tpos[i:i+2048]
            for k in hits: hits[k]+=(top[:,:k]==tp[:,None]).any(1).sum()
        log(name,mode,{k:round(v/len(q),4) for k,v in hits.items()})
        del E1,Eq; torch.cuda.empty_cache()
