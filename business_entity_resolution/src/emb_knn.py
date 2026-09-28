"""Usage: python emb_knn.py <emb_dir> train test   (needs a CUDA GPU)

LaBSE (Apache-2.0) name+address embeddings: top-20 Source-1 neighbours for every native-script record."""
import sys, time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
t = time.time(); log = lambda *a: print(f"{time.time()-t:6.0f}s", *a, flush=True)
G = sys.argv[1]
m = SentenceTransformer("sentence-transformers/LaBSE", device="cuda"); m.half()
K = 20
for sp in sys.argv[2:]:
    s1 = pd.read_parquet(f"{G}/{sp}_s1_names.parquet"); q = pd.read_parquet(f"{G}/{sp}_native_q.parquet")
    idx_all, cos_all = [], []
    for c in sorted(q.country.unique()):
        s1c = s1[s1.country == c]; qc_mask = (q.country == c).values
        E1 = m.encode((s1c.raw_name + ' | ' + s1c.raw_addr).tolist(), batch_size=2048, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        Eq = m.encode((q.raw_name + ' | ' + q.raw_addr)[qc_mask].tolist(), batch_size=2048, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        log(sp, c, "encoded", E1.shape, Eq.shape)
        ids = s1c.index.values  # positional index into s1 file (reset below)
        I = np.empty((len(Eq), K), np.int32); C = np.empty((len(Eq), K), np.float16)
        for i in range(0, len(Eq), 4096):
            v, ix = torch.topk(Eq[i:i+4096] @ E1.T, K, dim=1)
            I[i:i+4096] = ix.cpu().numpy(); C[i:i+4096] = v.float().cpu().numpy()
        # map to s1 entity ids
        idx_all.append(pd.DataFrame({"q_id": np.repeat(q.entity_id.values[qc_mask], K),
                                     "s1_id": s1c.entity_id.values[I.ravel()],
                                     "cos": C.ravel(), "erank": np.tile(np.arange(K, dtype=np.int8), len(Eq))}))
        del E1, Eq; torch.cuda.empty_cache()
    out = pd.concat(idx_all, ignore_index=True)
    out.to_parquet(f"{G}/{sp}_labse_knn.parquet", index=False)
    log(sp, "saved", len(out))
