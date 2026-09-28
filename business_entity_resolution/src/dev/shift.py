"""Simulate the train->test distractor-density shift on the training split.

Test has ~5.75 S2/S3 records per S1 entity vs ~4.68 in train, consistent with a
fraction of S1 entities being removed while their records stay behind as
distractors.  We reproduce that on train by dropping a random fraction of S1
entities: their records become distractors, their candidate rows disappear
and the remaining candidates are re-ranked (as re-running blocking would).
"""
import numpy as np
import pandas as pd

from features import add_context


def drop_s1(cand, F, truth, n_s1, frac, seed=0, kuse=3):
    rng = np.random.default_rng(seed)
    dropped = rng.random(n_s1) < frac
    keep = ~dropped[cand.s1_idx.values]
    c = cand[keep].copy()
    # re-rank the surviving candidates of each record
    c = c.sort_values(["q_idx", "score"], ascending=[True, False])
    c["rank"] = c.groupby("q_idx").cumcount().astype(np.int8)
    c = c[c["rank"] < kuse]
    F2 = F.loc[c.index].reset_index(drop=True)
    c = c.reset_index(drop=True)
    add_context(F2, c)  # recompute competition features after re-ranking
    t2 = truth.copy()
    t2[(t2 >= 0) & dropped[np.maximum(t2, 0)]] = -1
    return c, F2, t2, ~dropped
