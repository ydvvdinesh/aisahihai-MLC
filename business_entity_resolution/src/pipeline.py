"""End-to-end pipeline.

  python pipeline.py train   --work W            # blocking + features + CV + final model
  python pipeline.py predict --work W --out O    # test blocking + matching + output files
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import run_blocking
from common import load_split, gt_pairs, fbeta_from_assign
from features import pair_features
from stage2 import group_features, decide_expected_f

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=127, verbose=-1, num_threads=4)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_candidates(work, split, s1, q, K, df_cap, procs):
    path = f"{work}/{split}_cand_K{K}_cap{df_cap}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    cand = run_blocking(s1, q, K=K, df_cap=df_cap, procs=procs, log=log)
    cand.to_parquet(path, index=False)
    return cand


def get_features(work, split, tag, cand, s1, q, procs):
    path = f"{work}/{split}_feat_{tag}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    F = pair_features(cand, s1, q, procs=procs)
    F.to_parquet(path, index=False)
    return F


def assign_from_prob(cand, prob, n_q, thr):
    """Each record goes to its highest-probability S1 candidate if prob > thr."""
    order = np.lexsort((-prob, cand.q_idx.values))
    qs = cand.q_idx.values[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    sel = order[first]
    sel = sel[prob[sel] > thr]
    assign = np.full(n_q, -1, dtype=np.int64)
    assign[cand.q_idx.values[sel]] = cand.s1_idx.values[sel]
    return assign


def train(a):
    import gc
    s1, q = load_split(a.work, "train")
    truth = gt_pairs(a.work, s1, q)
    n_s1, n_q = len(s1), len(q)
    s1_countries = s1.country.values.copy()
    country_unique = list(s1.country.unique())
    log("loaded", n_s1, n_q)
    cand = get_candidates(a.work, "train", s1, q, a.K, a.df_cap, a.procs)
    cand = cand[cand["rank"] < a.Kuse].reset_index(drop=True)
    log("candidates", len(cand), "per S1", len(cand) / n_s1)
    F = get_features(a.work, "train", f"K{a.K}_cap{a.df_cap}_use{a.Kuse}", cand, s1, q, a.procs)
    del s1, q
    gc.collect()

    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    log("features", F.shape, "pos rate", y.mean(), "recall ceiling", y.sum() / (truth >= 0).sum())
    feats = list(F.columns)
    X = F[feats].to_numpy(dtype=np.float32)
    del F
    gc.collect()

    # 2-fold CV grouped by S1 entity -> out-of-fold probabilities
    fold = (cand.s1_idx.values * 2654435761 % 2**32) % 2
    oof = np.zeros(len(cand), dtype=np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True), a.rounds)
        oof[va] = m.predict(X[va], num_threads=a.procs)
        del m
        gc.collect()
        log(f"fold {k} done")

    # threshold sweep on macro F0.5 over ALL S1 entities (singletons included)
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.96, 0.025):
        f = fbeta_from_assign(assign_from_prob(cand, oof, n_q, thr), truth, n_s1)
        if f > best[0]:
            best = (f, float(thr))
        log(f"thr={thr:.3f} F0.5={f:.5f}")
    log("BEST OOF", best)
    by_c = {}
    assign = assign_from_prob(cand, oof, n_q, best[1])
    for c in country_unique:
        by_c[c] = fbeta_from_assign(assign, truth, n_s1, s1_countries == c)
    log("per-country", by_c)

    model = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats, free_raw_data=True), a.rounds)
    del X
    gc.collect()
    model.save_model(f"{a.work}/model.txt")
    imp = sorted(zip(model.feature_importance("gain"), feats), reverse=True)
    log("importance", [(f, int(g)) for g, f in imp])
    meta = {"threshold": best[1], "oof_f05": best[0], "K": a.K, "Kuse": a.Kuse, "df_cap": a.df_cap,
            "features": feats, "per_country": by_c, "stage2": None}
    np.save(f"{a.work}/train_oof.npy", oof)
    del model

    if a.stage2:
        # stage 2: group-consistency features built on stage-1 out-of-fold probabilities
        G = group_features(cand, oof, s1, q)
        feats2 = feats + list(G.columns)
        X = np.hstack([F.values.astype(np.float32), G.values.astype(np.float32)])
        del F, G
        oof2 = np.zeros(len(cand), dtype=np.float32)
        for k in (0, 1):
            tr, va = fold != k, fold == k
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feats2), a.rounds)
            oof2[va] = m.predict(X[va], num_threads=a.procs)
            log(f"stage2 fold {k} done")
        np.save(f"{a.work}/train_oof2.npy", oof2)
        best2 = (0, 0.5, "thr")
        for thr in np.arange(0.3, 0.9, 0.025):
            f = fbeta_from_assign(assign_from_prob(cand, oof2, len(q), thr), truth, len(s1))
            if f > best2[0]:
                best2 = (f, float(thr), "thr")
        f = fbeta_from_assign(decide_expected_f(cand, oof2, len(q), len(s1), miss_rate=0.1), truth, len(s1))
        if f > best2[0]:
            best2 = (f, 0.1, "expf")
        log("BEST STAGE2 OOF", best2)
        asg = (assign_from_prob(cand, oof2, len(q), best2[1]) if best2[2] == "thr"
               else decide_expected_f(cand, oof2, len(q), len(s1), miss_rate=best2[1]))
        log("stage2 per-country", {c: fbeta_from_assign(asg, truth, len(s1), s1.country.values == c)
                                   for c in s1.country.unique()})
        m2 = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats2), a.rounds)
        m2.save_model(f"{a.work}/model2.txt")
        meta["stage2"] = {"features": feats2, "oof_f05": best2[0], "param": best2[1], "rule": best2[2]}
    json.dump(meta, open(f"{a.work}/model_meta.json", "w"), indent=1)


def write_lists(path, col, s1_ids, s1_idx, q_ids):
    groups = pd.Series(q_ids).groupby(s1_idx).agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids,
                        col: pd.Series(groups).reindex(np.arange(len(s1_ids))).fillna("").values})
    out.to_csv(path, sep="\t", index=False)


def predict(a):
    import gc
    meta = json.load(open(f"{a.work}/{a.meta}"))
    if a.no_stage2:
        meta["stage2"] = None
    s1, q = load_split(a.work, "test")
    log("loaded test", len(s1), len(q), s1.country.value_counts().to_dict())
    cand = get_candidates(a.work, "test", s1, q, meta["K"], meta["df_cap"], a.procs)
    cand = cand[cand["rank"] < meta["Kuse"]].reset_index(drop=True)
    F = get_features(a.work, "test", f"K{meta['K']}_cap{meta['df_cap']}_use{meta['Kuse']}", cand, s1, q, a.procs)
    s1_ids = s1.entity_id.values
    q_ids = q.entity_id.values
    n_s1, n_q = len(s1), len(q)
    del s1, q
    gc.collect()

    model = lgb.Booster(model_file=f"{a.work}/{meta.get('model', 'model.txt')}")
    X = F[meta["features"]].to_numpy(dtype=np.float32)
    del F
    gc.collect()
    prob = model.predict(X, num_threads=a.procs)
    del X, model
    gc.collect()

    thr = a.thr if a.thr is not None else meta["threshold"]
    assign = assign_from_prob(cand, prob, n_q, thr)
    del prob
    gc.collect()

    os.makedirs(a.out, exist_ok=True)
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", s1_ids,
                cand.s1_idx.values, q_ids[cand.q_idx.values])
    m = assign >= 0
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", s1_ids,
                assign[m], q_ids[m])
    log(f"wrote outputs: cands/S1={len(cand) / n_s1:.2f} matched records={m.sum()} "
        f"matches/S1={m.sum() / n_s1:.2f} empty S1={(np.bincount(assign[m], minlength=n_s1) == 0).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="../output")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--Kuse", type=int, default=2)
    ap.add_argument("--df_cap", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--meta", default="model_meta.json")
    ap.add_argument("--no_stage2", action="store_true")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
