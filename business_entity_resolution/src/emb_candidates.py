from multiprocessing import Pool

import numpy as np
import pandas as pd

_G = {}


def _jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rerank_chunk(bounds):
    lo, hi = bounds
    qi = _G["qi"][lo:hi]; si = _G["si"][lo:hi]
    qn, qw, sn, sw = _G["qn"], _G["qw"], _G["sn"], _G["sw"]
    out = np.empty(hi - lo, np.float32)
    for k, (a, b) in enumerate(zip(qi, si)):
        out[k] = 0.5 * _jac(qn[a], sn[b]) + 0.5 * _jac(qw[a], sw[b])
    return out


def add_emb_candidates(cand, knn, s1, q, extra=2, procs=36):
    q_pos = pd.Series(np.arange(len(q)), index=q.entity_id.values)
    s_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    k = pd.DataFrame({"q_idx": q_pos.reindex(knn.q_id.values).values,
                      "s1_idx": s_pos.reindex(knn.s1_id.values).values,
                      "cos": knn.cos.values.astype(np.float32)}).dropna()
    k = k.astype({"q_idx": np.int64, "s1_idx": np.int64})
    _G.update(qi=k.q_idx.values, si=k.s1_idx.values,
              qn=[set(x.split()) for x in q.a_nums.values], qw=[set(x.split()) for x in q.a_words.values],
              sn=[set(x.split()) for x in s1.a_nums.values], sw=[set(x.split()) for x in s1.a_words.values])
    step = 200000
    with Pool(procs) as pool:
        parts = pool.map(_rerank_chunk, [(i, min(i + step, len(k))) for i in range(0, len(k), step)])
    _G.clear()
    k["rr"] = k.cos.values + np.concatenate(parts)
    key = lambda d: d.q_idx.values * 10_000_000 + d.s1_idx.values
    present = np.isin(key(k), key(cand))
    new = k[~present].sort_values(["q_idx", "rr"], ascending=[True, False])
    new = new[new.groupby("q_idx").cumcount() < extra]
    maxrank = cand.groupby("q_idx")["rank"].max()
    add = pd.DataFrame({"q_idx": new.q_idx.values, "s1_idx": new.s1_idx.values,
                        "score": np.zeros(len(new), np.float32),
                        "rank": (maxrank.reindex(new.q_idx.values).fillna(-1).values + 1
                                 + new.groupby("q_idx").cumcount().values).astype(np.int8)})
    out = pd.concat([cand[["q_idx", "s1_idx", "score", "rank"]], add], ignore_index=True)
    cosmap = pd.Series(k.cos.values, index=key(k))
    emb_cos = cosmap.reindex(key(out)).values.astype(np.float32)
    native = np.zeros(len(q), bool); native[k.q_idx.values] = True
    emb_cos = np.where(native[out.q_idx.values] & np.isnan(emb_cos), 0.0, emb_cos)
    is_emb = np.r_[np.zeros(len(cand), np.int8), np.ones(len(add), np.int8)]
    return out, emb_cos, is_emb
