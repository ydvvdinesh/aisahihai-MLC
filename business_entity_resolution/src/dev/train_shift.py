"""Train stage-1 + stage-2 on a density-shifted training set that mimics test.

Test has ~5.75 S2/S3 records per S1 entity vs 4.68 in train (extra distractors whose
S1 entity is absent).  We drop a random fraction of S1 entities from train, turning
their records into realistic look-alike distractors, then train/validate as usual.
Different seeds are used for CV-validation and for the final model so both see a shift.
"""
import argparse, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_split, gt_pairs, fbeta_from_assign
from pipeline import assign_from_prob, PARAMS
from stage2 import group_features, decide_expected_f
from shift import drop_s1
from features import pair_features
import os

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--frac", type=float, default=0.19)
    ap.add_argument("--Kuse", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--tag", default="shift")
    a = ap.parse_args()
    W = a.work
    s1, q = load_split(W, "train"); truth = gt_pairs(W, s1, q)
    cand = pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet")
    fa = f"{W}/train_feat_K5_all.parquet"
    if os.path.exists(fa):
        F = pd.read_parquet(fa)
    else:
        F3 = pd.read_parquet(f"{W}/train_feat_K5_cap2000_use3.parquet")
        lo = cand["rank"].values < 3
        Fx = pair_features(cand[~lo].reset_index(drop=True), s1, q, procs=a.procs)
        arr = np.zeros((len(cand), F3.shape[1]), np.float32)
        arr[lo] = F3.to_numpy(dtype=np.float32)
        arr[~lo] = Fx[list(F3.columns)].to_numpy(dtype=np.float32)
        F = pd.DataFrame(arr, columns=F3.columns)
        del F3, Fx, arr
        F.to_parquet(fa, index=False)
    feats = json.load(open(f"{W}/model_meta.json"))["features"]
    F = F[feats]
    log("features all ranks", F.shape)
    c, F2, t2, keep = drop_s1(cand, F, truth, len(s1), a.frac, seed=1, kuse=a.Kuse)
    del F
    y = (t2[c.q_idx.values] == c.s1_idx.values).astype(np.int8)
    log("shifted: S1 kept", keep.sum(), "records/S1", len(q) / keep.sum(), "pairs", len(c), "pos", y.mean())
    fold = (c.s1_idx.values * 2654435761 % 2**32) % 2
    X1 = F2[feats].values.astype(np.float32); del F2
    meta0 = json.load(open(f"{W}/model_meta.json"))
    p1 = lgb.Booster(model_file=f"{W}/model.txt").predict(X1, num_threads=a.procs)
    a1 = assign_from_prob(c, p1, len(q), meta0["threshold"])
    G0 = group_features(c, p1, s1, q)
    p2 = lgb.Booster(model_file=f"{W}/model2.txt").predict(np.hstack([X1, G0.values.astype(np.float32)]), num_threads=a.procs)
    del G0
    a2 = decide_expected_f(c, p2, len(q), len(s1), miss_rate=meta0["stage2"]["param"])
    log(f"DIAG old models on shifted train: stage1 F={fbeta_from_assign(a1, t2, len(s1), keep):.4f} "
        f"(assigned/S1 {(a1 >= 0).sum() / keep.sum():.2f})  stage2 F={fbeta_from_assign(a2, t2, len(s1), keep):.4f} "
        f"(assigned/S1 {(a2 >= 0).sum() / keep.sum():.2f})  true/S1 {(t2 >= 0).sum() / keep.sum():.2f}")
    oof = np.zeros(len(c), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X1[tr], y[tr], feature_name=feats), a.rounds)
        oof[va] = m.predict(X1[va], num_threads=a.procs); log("s1 fold", k)
    best1 = max((fbeta_from_assign(assign_from_prob(c, oof, len(q), th), t2, len(s1), keep), float(th))
                for th in np.arange(0.3, 0.95, 0.025))
    a1 = assign_from_prob(c, oof, len(q), best1[1])
    log("STAGE1 shifted OOF", best1, "assigned/S1", (a1 >= 0).sum() / keep.sum(),
        "true/S1", (t2 >= 0).sum() / keep.sum())
    G = group_features(c, oof, s1, q)
    feats2 = feats + list(G.columns)
    X2 = np.hstack([X1, G.values.astype(np.float32)]); del G
    oof2 = np.zeros(len(c), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X2[tr], y[tr], feature_name=feats2), a.rounds)
        oof2[va] = m.predict(X2[va], num_threads=a.procs); log("s2 fold", k)
    res = [(fbeta_from_assign(assign_from_prob(c, oof2, len(q), th), t2, len(s1), keep), float(th), "thr")
           for th in np.arange(0.3, 0.95, 0.025)]
    for mr in (0.0, 0.1):
        res.append((fbeta_from_assign(decide_expected_f(c, oof2, len(q), len(s1), miss_rate=mr), t2, len(s1), keep), mr, "expf"))
    best2 = max(res)
    log("STAGE2 shifted OOF", best2)
    # final models on the full shifted set
    m1 = lgb.train(PARAMS, lgb.Dataset(X1, y, feature_name=feats), a.rounds)
    m1.save_model(f"{W}/model_{a.tag}.txt")
    m2 = lgb.train(PARAMS, lgb.Dataset(X2, y, feature_name=feats2), a.rounds)
    m2.save_model(f"{W}/model2_{a.tag}.txt")
    meta = json.load(open(f"{W}/model_meta.json"))
    meta.update(threshold=best1[1], oof_f05_shift=best1[0], frac=a.frac, Kuse=a.Kuse,
                model=f"model_{a.tag}.txt", model2=f"model2_{a.tag}.txt",
                stage2={"features": feats2, "oof_f05": best2[0], "param": best2[1], "rule": best2[2]})
    json.dump(meta, open(f"{W}/model_meta_{a.tag}.json", "w"), indent=1)
    log("saved", f"model_meta_{a.tag}.json")


if __name__ == "__main__":
    main()
