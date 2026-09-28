"""Stage 2: group-consistency features + per-entity expected-F0.5 decision.

The records of one real business agree with each other; a look-alike
"sibling" business (e.g. same street, house number 287 vs 2870) forms its
own mutually consistent cluster.  For every (record, S1) pair we therefore
describe the *other* candidates of the same S1 entity: how many share this
record's house number / name vs. the S1 entity's own house number / name,
weighted by their stage-1 probabilities.
"""
import numpy as np
import pandas as pd


def _first(s):
    return s.str.split(" ", n=1).str[0].fillna("")


def group_features(cand, p1, s1, q):
    qi, si = cand.q_idx.values, cand.s1_idx.values
    d = pd.DataFrame({"s": si, "p": p1})
    # stage-1 competition within the record
    g = d.assign(q=qi).groupby("q").p
    pmax = g.transform("max").values
    d_sorted = np.sort(p1)  # noqa: F841 (kept simple; second best below)
    rank_in_q = pd.Series(-p1).groupby(qi).rank(method="first").values - 1
    second = pd.Series(np.where(rank_in_q == 1, p1, 0.0)).groupby(qi).transform("max").values
    out = pd.DataFrame({
        "p1": p1,
        "p1_gap_best": pmax - p1,
        "p1_is_best": (rank_in_q == 0).astype(np.int8),
        "p1_second_q": np.where(rank_in_q == 0, second, pmax),
    })
    best = rank_in_q == 0
    pb = np.where(best, p1, 0.0)
    gs = pd.Series(pb).groupby(si)
    out["g_psum_other"] = gs.transform("sum").values - pb
    out["g_n_claim"] = pd.Series((pb > 0.5).astype(np.float32)).groupby(si).transform("sum").values - (pb > 0.5)
    out["g_n"] = pd.Series(np.ones(len(si))).groupby(si).transform("size").values
    out["g_rank"] = pd.Series(-pb).groupby(si).rank(method="first").values

    # agreement counts on house number and on name
    for col, tag in (("a_nums", "num"), ("n_core", "name")):
        kq = (_first(q[col]) if tag == "num" else q[col]).values[qi]
        ks = (_first(s1[col]) if tag == "num" else s1[col]).values[si]
        kq = np.where(kq == "", "<e>q", kq)
        ks = np.where(ks == "", "<e>s", ks)
        key_df = pd.DataFrame({"s": si, "k": kq, "w": pb})
        agg = key_df.groupby(["s", "k"]).w.agg(["sum", "size"])
        idx_me = pd.MultiIndex.from_arrays([si, kq])
        idx_s1 = pd.MultiIndex.from_arrays([si, ks])
        me = agg.reindex(idx_me)
        s1m = agg.reindex(idx_s1)
        same = kq == ks
        out[f"g_{tag}_me_w"] = me["sum"].fillna(0).values - pb
        out[f"g_{tag}_s1_w"] = s1m["sum"].fillna(0).values - np.where(same, pb, 0)
        out[f"g_{tag}_me_n"] = me["size"].fillna(0).values - 1
        out[f"g_{tag}_s1_n"] = s1m["size"].fillna(0).values - same
        out[f"g_{tag}_eq"] = same.astype(np.int8)
    return out


def assign_best(cand, prob, n_q):
    order = np.lexsort((-prob, cand.q_idx.values))
    qs = cand.q_idx.values[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    return order[first]


def decide_expected_f(cand, prob, n_q, n_s1, beta=0.5, miss_rate=0.0, min_p=0.05):
    """Per S1 entity choose the top-m of its (best-assigned) records that
    maximises expected F-beta, including the option of predicting nothing.
    Returns assignment array (q -> s1 or -1)."""
    b2 = beta * beta
    sel = assign_best(cand, prob, n_q)
    sel = sel[prob[sel] > min_p]
    s = cand.s1_idx.values[sel]
    p = prob[sel]
    qv = cand.q_idx.values[sel]
    order = np.lexsort((-p, s))
    s, p, qv = s[order], p[order], qv[order]
    start = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    end = np.r_[start[1:], len(s)]
    assign = np.full(n_q, -1, dtype=np.int64)
    for a, b in zip(start, end):
        pp = p[a:b]
        exp_true = pp.sum() + miss_rate
        # option: predict nothing -> F = P(no true match) ~ prod(1-p)
        best_f = np.prod(1 - pp) * np.exp(-miss_rate)
        best_m = 0
        cum = np.cumsum(pp)
        for m in range(1, b - a + 1):
            # E[F] ~ (1+b2) E[tp] / (b2 E[n_true] + m)
            f = (1 + b2) * cum[m - 1] / (b2 * exp_true + m)
            if f > best_f:
                best_f, best_m = f, m
        if best_m:
            assign[qv[a:a + best_m]] = s[a]
    return assign
