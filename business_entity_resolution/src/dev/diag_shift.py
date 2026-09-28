"""Diagnostic: score saved stage-1 / stage-2 models on train with and without the simulated shift."""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_split, gt_pairs, fbeta_from_assign
from features import pair_features
from pipeline import assign_from_prob
from stage2 import group_features, decide_expected_f
from shift import drop_s1

W = sys.argv[1] if len(sys.argv) > 1 else "../work_v3"
t = time.time()
log = lambda *a: print(f"{time.time()-t:6.0f}s", *a, flush=True)
meta = json.load(open(f"{W}/model_meta.json"))
s1, q = load_split(W, "train"); truth = gt_pairs(W, s1, q)
cand = pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet")
fa = f"{W}/train_feat_K5_all.parquet"
if os.path.exists(fa):
    F = pd.read_parquet(fa)
else:
    F3 = pd.read_parquet(f"{W}/train_feat_K5_cap2000_use3.parquet")
    lo = cand["rank"].values < 3
    ext = cand[~lo].reset_index(drop=True)
    Fx = pair_features(ext, s1, q, procs=36)
    F = pd.DataFrame(index=np.arange(len(cand)), columns=F3.columns, dtype=np.float32)
    F.loc[np.flatnonzero(lo)] = F3.values
    F.loc[np.flatnonzero(~lo)] = Fx[F3.columns].values
    F = F.astype(np.float32)
    F.to_parquet(fa, index=False)
log("features all ranks", F.shape)
m1 = lgb.Booster(model_file=f"{W}/model.txt"); m2 = lgb.Booster(model_file=f"{W}/model2.txt")
feats = meta["features"]
for frac in (0.0, 0.19):
    c, F2, t2, keep = drop_s1(cand, F, truth, len(s1), frac, kuse=3)
    p1 = m1.predict(F2[feats].values, num_threads=36)
    a1 = assign_from_prob(c, p1, len(q), meta["threshold"])
    G = group_features(c, p1, s1, q)
    p2 = m2.predict(np.hstack([F2[feats].values, G.values]).astype(np.float32), num_threads=36)
    a2 = decide_expected_f(c, p2, len(q), len(s1), miss_rate=meta["stage2"]["param"])
    npred1 = (a1 >= 0).sum() / keep.sum(); npred2 = (a2 >= 0).sum() / keep.sum()
    log(f"frac={frac}: recs/S1={len(q)/keep.sum():.2f}  stage1 F={fbeta_from_assign(a1, t2, len(s1), keep):.4f} (pred/S1 {npred1:.2f})"
        f"  stage2 F={fbeta_from_assign(a2, t2, len(s1), keep):.4f} (pred/S1 {npred2:.2f})")
