"""Pairwise features for (S2/S3 record, S1 candidate) pairs.

All features are language/country agnostic string similarities computed on
the normalised views, plus "context" features describing the competition
among candidates (a record belongs to at most one S1 entity).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

PAIR_COLS = ["n_core", "n_full", "n_alias", "n_nospace", "n_skel", "a_norm", "a_words", "a_nums"]

FEATS = [
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_full_ratio", "n_full_tset", "n_alias_best",
    "n_nospace_jw", "n_skel_ratio", "n_jacc", "n_first_eq", "n_len_q", "n_len_s",
    "a_ratio", "a_tset", "a_tsort", "a_words_tset", "a_words_jacc", "a_partial",
    "num_first_eq", "num_first_suffix", "num_jacc", "num_any", "num_q_n", "num_s_n",
    "a_empty_q", "a_empty_s", "acro", "num_first_in", "a_words_cover_q", "a_words_cover_s",
]


def _jacc(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _pair_feats(qr, sr):
    qn, qf, qa, qns, qk, qad, qaw, qnum = qr
    sn, sf, sa, sns, sk, sad, saw, snum = sr
    out = []
    out.append(fuzz.ratio(qn, sn))
    out.append(fuzz.token_set_ratio(qn, sn))
    out.append(fuzz.token_sort_ratio(qn, sn))
    out.append(fuzz.partial_ratio(qn, sn) if qn and sn else 0.0)
    out.append(fuzz.ratio(qf, sf))
    out.append(fuzz.token_set_ratio(qf, sf))
    best = 0.0
    for x in qa.split("|"):
        for y in sa.split("|"):
            v = fuzz.token_set_ratio(x, y)
            if v > best:
                best = v
    out.append(best)
    best = 0.0
    for x in qns.split("|"):
        for y in sns.split("|"):
            v = JaroWinkler.similarity(x, y)
            if v > best:
                best = v
    out.append(best)
    out.append(fuzz.ratio(qk, sk))
    qt, st = set(qn.split()), set(sn.split())
    out.append(_jacc(qt, st))
    qs, ss = qn.split(), sn.split()
    out.append(float(bool(qs) and bool(ss) and qs[0] == ss[0]))
    out.append(len(qn))
    out.append(len(sn))
    # address
    if qad and sad:
        out.append(fuzz.ratio(qad, sad))
        out.append(fuzz.token_set_ratio(qad, sad))
        out.append(fuzz.token_sort_ratio(qad, sad))
        out.append(fuzz.token_set_ratio(qaw, saw) if qaw and saw else -1.0)
        out.append(_jacc(set(qaw.split()), set(saw.split())))
        out.append(fuzz.partial_ratio(qad, sad))
    else:
        out.extend([-1.0] * 6)
    qnums, snums = qnum.split(), snum.split()
    if qnums and snums:
        out.append(float(qnums[0] == snums[0]))
        a, b = qnums[0], snums[0]
        out.append(float(a != b and (a.endswith(b) or b.endswith(a))))
        out.append(_jacc(set(qnums), set(snums)))
        out.append(float(bool(set(qnums) & set(snums))))
    else:
        out.extend([-1.0] * 4)
    out.append(len(qnums))
    out.append(len(snums))
    out.append(float(not qad))
    out.append(float(not sad))
    acro = 0.0
    if len(qs) == 1 and len(ss) >= 2 and qs[0] == "".join(t[0] for t in ss):
        acro = 1.0
    elif len(ss) == 1 and len(qs) >= 2 and ss[0] == "".join(t[0] for t in qs):
        acro = 1.0
    out.append(acro)
    out.append(float(qnums[0] in set(snums)) if qnums and snums else -1.0)
    qw, sw = set(qaw.split()), set(saw.split())
    if qw and sw:
        inter = len(qw & sw)
        out.append(inter / len(qw))
        out.append(inter / len(sw))
    else:
        out.extend([-1.0, -1.0])
    return out


def _chunk(args):
    qcols, scols = args
    return np.asarray([_pair_feats(a, b) for a, b in zip(zip(*qcols), zip(*scols))], dtype=np.float32)


def pair_features(cand, s1, q, procs=36, step=50000):
    """cand: DataFrame with q_idx, s1_idx, score, rank. Returns feature DataFrame."""
    qi, si = cand.q_idx.values, cand.s1_idx.values
    qcols_all = [q[c].values for c in PAIR_COLS]
    scols_all = [s1[c].values for c in PAIR_COLS]
    jobs = []
    for i in range(0, len(cand), step):
        jobs.append(([c[qi[i:i + step]] for c in qcols_all], [c[si[i:i + step]] for c in scols_all]))
    with Pool(procs) as pool:
        parts = pool.map(_chunk, jobs)
    F = pd.DataFrame(np.vstack(parts), columns=FEATS)
    F["q_web"] = q.is_web.values[qi]
    F["s_web"] = s1.is_web.values[si]
    F["q_alias"] = q.has_alias.values[qi]
    F["s_alias"] = s1.has_alias.values[si]
    add_context(F, cand)
    return F


def add_context(F, cand):
    """Competition features: how this candidate compares with the record's other
    candidates, and how crowded the S1 entity's candidate list is."""
    sc = cand.score.values
    g = pd.Series(sc).groupby(cand.q_idx.values)
    top = g.transform("max").values
    F["blk_score"] = sc
    F["blk_rank"] = cand["rank"].values
    F["blk_margin"] = sc - top
    F["q_ncand"] = g.transform("size").values
    # gap between the record's best and second-best candidate
    r1 = cand["rank"].values == 1
    sec = pd.Series(sc[r1], index=cand.q_idx.values[r1])
    F["blk_gap12"] = top - sec.reindex(cand.q_idx.values).fillna(0).values
    F["s1_ncand"] = pd.Series(np.ones(len(cand))).groupby(cand.s1_idx.values).transform("size").values
    F["s1_rank0_cnt"] = pd.Series((cand["rank"].values == 0).astype(np.float32)).groupby(cand.s1_idx.values).transform("sum").values
