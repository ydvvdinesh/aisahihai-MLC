"""Candidate generation (blocking).

Key data property: every Source 2/3 record belongs to at most one Source 1
entity.  So blocking runs in the *reverse* direction: for each S2/S3 record
we retrieve its top-K most similar S1 records (within the same country
label) through an inverted index over blocking keys.  The per-S1 candidate
list is the set of records whose top-K contains that S1 entity.

Single tokens are poor keys here (the business vocabulary is small and
heavily reused), so most keys are *compound*:

  N:<full name skeleton>          whole name (typo/transliteration tolerant)
  b:<skel_i>_<skel_j>             pairs of name-token skeletons
  i:<initials>                    acronym of the name ("tci" = Talava Certified Ishares)
  p:<prefix>                      first 5 chars of the space-less name (web domains)
  k:<skeleton>                    single name-token skeletons
  a:<word>, #:<number>            single address words / numbers
  h:<number>_<word>               house number x street/area word
  w:<word_i>_<word_i+1>           consecutive address word bigrams
  x:<name skel>_<address word>    name token x address word

Keys with S1 document frequency above `df_cap` are dropped (stop keys),
which bounds the work per query exactly like a blocking-key size cap.
Scores are IDF-weighted cosine similarities over the key sets.
"""
import re
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

from normalize import skeleton_word

KEY_COLS = ["n_core", "n_nospace", "a_words", "a_nums", "has_alias", "n_alias"]
_TRAIL = re.compile(r"[sj]$")


def _sk(t):
    s = skeleton_word(t)
    s2 = _TRAIL.sub("", s)
    return s2 or s


def record_keys(n_core, n_nospace, a_words, a_nums, has_alias, n_alias):
    keys = set()
    names = n_alias.split("|") if has_alias else [n_core]
    first_sk = []
    for nm in names:
        toks = [t for t in nm.split() if t]
        sk = [s for s in (_sk(t) for t in toks) if s]
        if not sk:
            continue
        keys.add("N:" + " ".join(sorted(sk)))
        for s in sk:
            keys.add("k:" + s)
        u = sorted(set(sk))[:6]
        for x, y in combinations(u, 2):
            keys.add("b:" + x + "_" + y)
        if len(toks) >= 2:
            keys.add("i:" + "".join(t[0] for t in toks))
        elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
            keys.add("i:" + toks[0])
        if not first_sk:
            first_sk = sk
    for ns in n_nospace.split("|"):
        if len(ns) >= 4:
            keys.add("p:" + ns[:5])
    words = a_words.split()
    for t in words:
        keys.add("a:" + t)
    for x, y in zip(words, words[1:]):
        keys.add("w:" + x + "_" + y)
    nums = a_nums.split()
    for t in nums:
        keys.add("#:" + t)
    for n in nums[:2]:
        for w in words[:6]:
            keys.add("h:" + n + "_" + w)
    for s in first_sk[:3]:
        for w in words[:5]:
            keys.add("x:" + s + "_" + w)
    return keys


def _hash_rows(cols):
    rows, hs = [], []
    for i, row in enumerate(zip(*cols)):
        ks = record_keys(*row)
        rows.extend([i] * len(ks))
        hs.extend(hash(k) for k in ks)
    return np.asarray(rows, dtype=np.int32), np.asarray(hs, dtype=np.int64)


_G = {}


def _s1_chunk(bounds):
    lo, hi = bounds
    r, h = _hash_rows([c[lo:hi] for c in _G["s1cols"]])
    return r + lo, h


def _q_chunk(bounds):
    lo, hi = bounds
    rq, hq = _hash_rows([c[lo:hi] for c in _G["qcols"]])
    vocab, ok, idf = _G["vocab"], _G["ok"], _G["idf"]
    m = hi - lo
    pos = np.searchsorted(vocab, hq)
    pos[pos >= len(vocab)] = 0
    known = vocab[pos] == hq
    qnorm = np.sqrt(np.bincount(rq[known], weights=idf[pos[known]] ** 2, minlength=m))
    qnorm[qnorm == 0] = 1
    use = known & ok[pos]
    Q = sp.csr_matrix((idf[pos[use]] / qnorm[rq[use]], (rq[use], pos[use])),
                      shape=(m, len(vocab)), dtype=np.float32)
    C = (Q @ _G["ST"]).tocsr()
    K = _G["K"]
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(m, dtype=np.int32), counts)
    order = np.lexsort((-C.data, rows))
    rows_s = rows[order]
    rank = np.arange(len(order)) - C.indptr[rows_s]
    keep = rank < K
    return (rows_s[keep] + lo, C.indices[order][keep].astype(np.int32),
            C.data[order][keep].astype(np.float32), rank[keep].astype(np.int8))


def block_country(s1, q, K, df_cap, procs, step=5000):
    _G["s1cols"] = [s1[c].values for c in KEY_COLS]
    N = len(s1)
    with Pool(procs) as pool:
        parts = pool.map(_s1_chunk, [(i, min(i + 20000, N)) for i in range(0, N, 20000)])
    r1 = np.concatenate([p[0] for p in parts])
    h1 = np.concatenate([p[1] for p in parts])
    del parts
    vocab, inv, df = np.unique(h1, return_inverse=True, return_counts=True)
    del h1
    idf = (np.log((N + 1) / (df + 1)) + 1.0).astype(np.float32)
    ok = df <= df_cap
    norm1 = np.sqrt(np.bincount(r1, weights=idf[inv] ** 2, minlength=N)).astype(np.float32)
    norm1[norm1 == 0] = 1
    m1 = ok[inv]
    S = sp.csr_matrix((idf[inv[m1]] / norm1[r1[m1]], (r1[m1], inv[m1])),
                      shape=(N, len(vocab)), dtype=np.float32)
    del r1, inv, m1
    _G.update(vocab=vocab, ok=ok, idf=idf, ST=S.T.tocsr(), K=K,
              qcols=[q[c].values for c in KEY_COLS])
    del S
    _G.pop("s1cols")
    bounds = [(i, min(i + step, len(q))) for i in range(0, len(q), step)]
    with Pool(procs) as pool:
        parts = pool.map(_q_chunk, bounds, chunksize=2)
    _G.clear()
    return [np.concatenate([p[j] for p in parts]) for j in range(4)]


def run_blocking(s1, q, K=5, df_cap=2000, procs=36, log=print):
    """s1, q: normalised frames. Returns DataFrame(q_idx, s1_idx, score, rank)
    with positional indices into s1 / q."""
    out = []
    countries = sorted(set(s1.country.unique()) & set(q.country.unique()))
    for c in countries:
        i1 = np.flatnonzero(s1.country.values == c)
        iq = np.flatnonzero(q.country.values == c)
        t = time.time()
        rq, r1, sc, rk = block_country(s1.iloc[i1], q.iloc[iq], K, df_cap, procs)
        out.append(pd.DataFrame({"q_idx": iq[rq], "s1_idx": i1[r1], "score": sc, "rank": rk}))
        log(f"  blocking {c}: S1={len(i1)} Q={len(iq)} pairs={len(rq)} in {time.time() - t:.0f}s")
    return pd.concat(out, ignore_index=True)
