"""Why does stage 2 add so many more assignments on test than on train?  Compare distributions,
and write diagnostic submission variants."""
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from common import load_split
from pipeline import assign_from_prob, write_lists
from stage2 import group_features, decide_expected_f
t = time.time(); log = lambda *a: print(f"{time.time()-t:6.0f}s", *a, flush=True)
W = "../work_v3"; meta = json.load(open(f"{W}/model_meta.json")); feats = meta["features"]
s1, q = load_split(W, "test")
cand = pd.read_parquet(f"{W}/test_cand_K5_cap2000.parquet"); cand = cand[cand["rank"] < 3].reset_index(drop=True)
F = pd.read_parquet(f"{W}/test_feat_K5_cap2000_use3.parquet")
X1 = F[feats].to_numpy(np.float32); del F
p1 = lgb.Booster(model_file=f"{W}/model.txt").predict(X1, num_threads=44)
G = group_features(cand, p1, s1, q)
p2 = lgb.Booster(model_file=f"{W}/model2.txt").predict(np.hstack([X1, G.to_numpy(np.float32)]), num_threads=44)
log("predicted")
cc = s1.country.values
variants = {
    "v3a_stage1": assign_from_prob(cand, p1, len(q), meta["threshold"]),
    "v3b_stage2_thr": assign_from_prob(cand, p2, len(q), 0.675),
    "v3_stage2_expf": decide_expected_f(cand, p2, len(q), len(s1), miss_rate=meta["stage2"]["param"]),
}
for name, a in variants.items():
    n = np.bincount(a[a >= 0], minlength=len(s1))
    log(name, "assigned", (a >= 0).sum(), {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in ["US", "India", "France"]},
        "hist", np.bincount(np.minimum(n, 12)).tolist())
# what does stage 2 add on test?
a1, a2 = variants["v3a_stage1"], variants["v3_stage2_expf"]
added = np.flatnonzero((a2 >= 0) & (a1 < 0))
log("added by stage2:", len(added), "removed:", ((a1 >= 0) & (a2 < 0)).sum(), "changed:", ((a1 >= 0) & (a2 >= 0) & (a1 != a2)).sum())
best = cand.assign(p1=p1, p2=p2)
bm = best.loc[best.groupby("q_idx").p2.idxmax()].set_index("q_idx")
rng = np.random.default_rng(0)
for i in rng.choice(added, 25, replace=False):
    j = a2[i]; r = bm.loc[i]
    print(f"p1={r.p1:.2f} p2={r.p2:.2f} rank={int(r['rank'])} | Q: {q.raw_name.values[i]} | {q.raw_addr.values[i]}\n      S1: {s1.raw_name.values[j]} | {s1.raw_addr.values[j]}", flush=True)
print("G means test:", G.mean().round(3).to_dict(), flush=True)
out = ".."
for name in ["v3a_stage1", "v3b_stage2_thr"]:
    a = variants[name]; m = a >= 0
    import os; os.makedirs(f"{out}/output_{name}", exist_ok=True)
    write_lists(f"{out}/output_{name}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, a[m], q.entity_id.values[m])
log("written")
