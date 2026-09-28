"""Export pair texts for the cross-encoder: uncertain train pairs (from v7 OOF) with labels,
and all test pairs under the v7 candidate set whose v7 probability is in the uncertain band."""
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from common import load_split, gt_pairs
W = sys.argv[1]; OUT = sys.argv[2]; LO, HI = 0.02, 0.98
def text(df, idx):
    return (df.raw_name.values[idx] + " | " + df.raw_addr.values[idx])
# train
s1, q = load_split(W, "train"); truth = gt_pairs(W, s1, q)
c = pd.read_parquet(f"{W}/train_cand_v7.parquet"); p = np.load(f"{W}/train_oof_v7.npy")
m = (p > LO) & (p < HI)
c = c[m].reset_index(drop=True); p = p[m]
y = (truth[c.q_idx.values] == c.s1_idx.values).astype(np.int8)
fold = (c.s1_idx.values * 2654435761 % 2**32) % 2
pd.DataFrame({"a": text(q, c.q_idx.values), "b": text(s1, c.s1_idx.values), "y": y, "fold": fold,
              "q_idx": c.q_idx.values, "s1_idx": c.s1_idx.values, "p7": p}).to_parquet(f"{OUT}/ce_train.parquet", index=False)
print("train uncertain pairs", len(c), "pos rate", y.mean())
