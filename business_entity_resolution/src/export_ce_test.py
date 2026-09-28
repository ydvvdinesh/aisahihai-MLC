"""Export test pairs in the uncertain band of the v7 model for cross-encoder scoring."""
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from v6 import load
from v7 import prune
W, OUT = sys.argv[1], sys.argv[2]; LO, HI = 0.02, 0.98
meta = json.load(open(f"{W}/model_meta_v7.json"))
s1, q, cand, F = load(W, "test", 44); cand, F = prune(cand, F, meta["alpha"])
p = lgb.Booster(model_file=f"{W}/model_v7.txt").predict(F[meta["feats"]].to_numpy(np.float32), num_threads=44)
m = (p > LO) & (p < HI); c = cand[m]
t = lambda d, i: d.raw_name.values[i] + " | " + d.raw_addr.values[i]
pd.DataFrame({"a": t(q, c.q_idx.values), "b": t(s1, c.s1_idx.values), "q_idx": c.q_idx.values,
              "s1_idx": c.s1_idx.values, "p7": p[m]}).to_parquet(f"{OUT}/ce_test.parquet", index=False)
print("test uncertain pairs", m.sum(), "of", len(cand))
