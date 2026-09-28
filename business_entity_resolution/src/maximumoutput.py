"""v5: blocking candidates + LaBSE candidates for native-script names, stage-1 LightGBM,
and a *robust* stage 2 whose group features cannot be inflated by sibling-business clusters.

Why robust: the test set contains many "sibling" businesses (same name, same street,
neighbouring house number, several mutually consistent records, Source-1 entity absent).
The original stage-2 features counting support for the record's *own* house number / name
let such a cluster vouch for itself, flipping stage-1 rejections into matches (+10 % test
assignments, leaderboard 0.956 -> 0.921).  The robust variant only keeps agreement with the
*Source-1 entity's* own values, so a sibling cluster can only lower confidence.

  python v5.py --work ../work_v3 --knn_dir ../../gpu --out_root ..
"""
import argparse, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_split, gt_pairs, fbeta_from_assign
from features import pair_features, add_context
from emb_candidates import add_emb_candidates
from pipeline import assign_from_prob, write_lists, PARAMS
from stage2 import group_features, decide_expected_f

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)
SELF_SUPPORT = ["g_num_me_w", "g_num_me_n", "g_name_me_w", "g_name_me_n", "g_psum_other", "g_n_claim"]


def build(W, split, knn_dir, feats, extra, procs):
    path = f"{W}/{split}_v5_cand_feat.parquet"
    s1, q = load_split(W, split)
    if os.path.exists(path):
        d = pd.read_parquet(path)
        return s1, q, d[["q_idx", "s1_idx", "score", "rank"]], d.drop(columns=["q_idx", "s1_idx", "score", "rank"])
    cand = pd.read_parquet(f"{W}/{split}_cand_K5_cap2000.parquet")
    cand = cand[cand["rank"] < 3].reset_index(drop=True)
    F = pd.read_parquet(f"{W}/{split}_feat_K5_cap2000_use3.parquet")[feats]
    knn = pd.read_parquet(f"{knn_dir}/{split}_labse_knn.parquet")
    cand2, cos, ise = add_emb_candidates(cand, knn, s1, q, extra=extra, procs=procs)
    new = cand2.iloc[len(cand):].reset_index(drop=True)
    log(split, "LaBSE added", len(new), "pairs")
    Fn = pair_features(new, s1, q, procs=procs)[feats]
    F = pd.concat([F, Fn], ignore_index=True)
    add_context(F, cand2)
    F["emb_cos"] = cos
    F["is_emb"] = ise
    d = pd.concat([cand2.reset_index(drop=True), F], axis=1)
    d.to_parquet(path, index=False)
    return s1, q, cand2, F


def sweep(cand, prob, truth, n_s1, n_q, keep=None):
    res = [(fbeta_from_assign(assign_from_prob(cand, prob, n_q, th), truth, n_s1, keep), float(th), "thr")
           for th in np.arange(0.3, 0.95, 0.025)]
    res.append((fbeta_from_assign(decide_expected_f(cand, prob, n_q, n_s1, miss_rate=0.1), truth, n_s1, keep), 0.1, "expf"))
    return max(res)


def decide(cand, prob, n_q, n_s1, rule):
    return (assign_from_prob(cand, prob, n_q, rule[1]) if rule[2] == "thr"
            else decide_expected_f(cand, prob, n_q, n_s1, miss_rate=rule[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--knn_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--extra", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args()
    W = a.work
    base_feats = json.load(open(f"{W}/model_meta.json"))["features"]
    P = dict(PARAMS, num_threads=a.procs)

    # ---------------- train ----------------
    s1, q, cand, F = build(W, "train", a.knn_dir, base_feats, a.extra, a.procs)
    truth = gt_pairs(W, s1, q)
    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    feats1 = list(F.columns)
    log("train pairs", len(cand), "cands/S1", round(len(cand) / len(s1), 2), "recall ceiling", y.sum() / (truth >= 0).sum())
    X1 = F.to_numpy(np.float32); del F
    fold = (cand.s1_idx.values * 2654435761 % 2**32) % 2
    oof1 = np.zeros(len(cand), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(P, lgb.Dataset(X1[tr], y[tr], feature_name=feats1), a.rounds)
        oof1[va] = m.predict(X1[va], num_threads=a.procs)
    r1 = sweep(cand, oof1, truth, len(s1), len(q))
    log("STAGE1 OOF", r1, {c: round(fbeta_from_assign(decide(cand, oof1, len(q), len(s1), r1), truth, len(s1), s1.country.values == c), 5)
                           for c in ("US", "India")})
    G = group_features(cand, oof1, s1, q).drop(columns=SELF_SUPPORT)
    feats2 = feats1 + list(G.columns)
    X2 = np.hstack([X1, G.to_numpy(np.float32)]); del G
    oof2 = np.zeros(len(cand), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(P, lgb.Dataset(X2[tr], y[tr], feature_name=feats2), a.rounds)
        oof2[va] = m.predict(X2[va], num_threads=a.procs)
    r2 = sweep(cand, oof2, truth, len(s1), len(q))
    a1 = decide(cand, oof1, len(q), len(s1), r1); a2 = decide(cand, oof2, len(q), len(s1), r2)
    log("STAGE2-robust OOF", r2, "train assigned stage1", (a1 >= 0).sum(), "stage2", (a2 >= 0).sum())
    m1 = lgb.train(P, lgb.Dataset(X1, y, feature_name=feats1), a.rounds)
    m2 = lgb.train(P, lgb.Dataset(X2, y, feature_name=feats2), a.rounds)
    m1.save_model(f"{W}/model_v5_s1.txt"); m2.save_model(f"{W}/model_v5_s2.txt")
    json.dump({"feats1": feats1, "feats2": feats2, "rule1": r1, "rule2": r2, "extra": a.extra,
               "self_support_dropped": SELF_SUPPORT}, open(f"{W}/model_meta_v5.json", "w"), indent=1)
    del X1, X2, oof1, oof2

    # ---------------- test ----------------
    s1, q, cand, F = build(W, "test", a.knn_dir, base_feats, a.extra, a.procs)
    X1 = F[feats1].to_numpy(np.float32); del F
    p1 = m1.predict(X1, num_threads=a.procs)
    G = group_features(cand, p1, s1, q).drop(columns=SELF_SUPPORT)
    p2 = m2.predict(np.hstack([X1, G.to_numpy(np.float32)]), num_threads=a.procs)
    cc = s1.country.values
    qid = q.entity_id.values
    for name, asg in (("v5a", decide(cand, p1, len(q), len(s1), r1)), ("v5b", decide(cand, p2, len(q), len(s1), r2))):
        n = np.bincount(asg[asg >= 0], minlength=len(s1))
        log(name, "TEST assigned", (asg >= 0).sum(),
            {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in sorted(set(cc))})
        out = f"{a.out_root}/output_{name}"
        os.makedirs(out, exist_ok=True)
        write_lists(f"{out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values,
                    cand.s1_idx.values, qid[cand.q_idx.values])
        m = asg >= 0
        write_lists(f"{out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, asg[m], qid[m])
    log("test cands/S1", round(len(cand) / len(s1), 2), "done")


if __name__ == "__main__":
    main()
