"""Prior-shift correction for sibling-like pairs (v5c).

Test contains many more "sibling" look-alikes than train: records with the same core name as a
Source-1 entity but a different house number, whose own Source-1 entity is absent.  A classifier
trained on train has train's class prior inside that sub-population, so its probabilities there
are too optimistic on test.

For each sub-population S (house-number mismatch x same/different core name), among the
records' top-ranked blocking candidates:
  - train: n_tr pairs in S, of which pos_tr are true matches → prior π_tr
  - test:  n_te pairs in S; the number of true matches per Source-1 entity is assumed unchanged,
           so pos_te ≈ pos_tr * (N_S1_test / N_S1_train) → π_te = pos_te / n_te
The odds of every test pair in S are multiplied by w = odds(π_te) / odds(π_tr) (a standard
label-shift correction), then the usual decision rule is re-applied.
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_split, gt_pairs
from pipeline import assign_from_prob, write_lists
from stage2 import decide_expected_f

t0 = time.time()
log = lambda *a: print(f"{time.time()-t0:6.0f}s", *a, flush=True)
W = sys.argv[1] if len(sys.argv) > 1 else "../work"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../output"


def first_num(s):
    return s.str.split(" ", n=1).str[0].fillna("").values


def groups(cand, s1, q):
    qi, si = cand.q_idx.values, cand.s1_idx.values
    a, b = first_num(q.a_nums)[qi], first_num(s1.a_nums)[si]
    mism = (a != "") & (b != "") & (a != b)
    same = q.n_core.values[qi] == s1.n_core.values[si]
    g = np.full(len(cand), -1, np.int8)
    g[mism & same] = 0
    g[mism & ~same] = 1
    return g


def main():
    s1t, qt = load_split(W, "train")
    truth = gt_pairs(W, s1t, qt)
    ct = pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet")
    ct = ct[ct["rank"] == 0].reset_index(drop=True)
    gt_ = groups(ct, s1t, qt)
    pos = truth[ct.q_idx.values] == ct.s1_idx.values
    n_s1_tr = len(s1t)
    del s1t, qt, truth

    s1, q = load_split(W, "test")
    d = pd.read_parquet(f"{W}/test_v5_cand_feat.parquet")
    cand = d[["q_idx", "s1_idx", "score", "rank"]]
    meta = json.load(open(f"{W}/model_meta_v5.json"))
    X = d[meta["feats1"]].to_numpy(np.float32)
    del d
    p = lgb.Booster(model_file=f"{W}/model_v5_s1.txt").predict(X, num_threads=40)
    del X
    top = (cand["rank"].values == 0)
    g = groups(cand, s1, q)
    scale = len(s1) / n_s1_tr
    w = {}
    for k, name in ((0, "mismatch+same-name"), (1, "mismatch+diff-name")):
        n_tr, pos_tr = (gt_ == k).sum(), (pos & (gt_ == k)).sum()
        n_te = (top & (g == k)).sum()
        pi_tr = pos_tr / n_tr
        pi_te = min(pos_tr * scale / n_te, pi_tr)
        w[k] = (pi_te / (1 - pi_te)) / (pi_tr / (1 - pi_tr))
        log(f"{name}: train n={n_tr} prior={pi_tr:.3f} ({n_tr / n_s1_tr:.3f}/S1) | test n={n_te} "
            f"({n_te / len(s1):.3f}/S1) est.prior={pi_te:.3f} → odds weight {w[k]:.3f}")
    odds = p / np.clip(1 - p, 1e-6, None)
    for k, wk in w.items():
        odds[g == k] *= wk
    p2 = odds / (1 + odds)
    rule = meta["rule1"]
    asg0 = (assign_from_prob(cand, p, len(q), rule[1]) if rule[2] == "thr"
            else decide_expected_f(cand, p, len(q), len(s1), miss_rate=rule[1]))
    asg = (assign_from_prob(cand, p2, len(q), rule[1]) if rule[2] == "thr"
           else decide_expected_f(cand, p2, len(q), len(s1), miss_rate=rule[1]))
    cc = s1.country.values
    n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log("v5a assigned", (asg0 >= 0).sum(), "| v5c assigned", (asg >= 0).sum(),
        {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in sorted(set(cc))})
    out = OUT
    os.makedirs(out, exist_ok=True)
    qid = q.entity_id.values
    m = asg >= 0
    write_lists(f"{out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, asg[m], qid[m])
    write_lists(f"{out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values,
                cand.s1_idx.values, qid[cand.q_idx.values])
    json.dump({str(k): float(v) for k, v in w.items()}, open(f"{W}/sibling_prior_weights.json", "w"))
    log("written", out)


if __name__ == "__main__":
    main()
