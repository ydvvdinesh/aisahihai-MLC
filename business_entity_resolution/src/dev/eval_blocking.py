"""Dev tool: run blocking on the train split and report recall / candidate sizes."""
import argparse
import time

import numpy as np
import pandas as pd

from blocking import run_blocking
from common import load_split, gt_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--df_cap", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=36)
    a = ap.parse_args()
    t = time.time()
    s1, q = load_split(a.work, "train")
    truth = gt_pairs(a.work, s1, q)
    print(f"loaded {time.time() - t:.0f}s", flush=True)
    cand = run_blocking(s1, q, K=a.K, df_cap=a.df_cap, procs=a.procs)
    cand.to_parquet(f"{a.work}/train_cand_K{a.K}_cap{a.df_cap}.parquet", index=False)
    cand["hit"] = truth[cand.q_idx.values] == cand.s1_idx.values
    matched = truth >= 0
    print(f"total time {time.time() - t:.0f}s  pairs={len(cand)}")
    for k in range(1, a.K + 1):
        c = cand[cand["rank"] < k]
        rec = c.hit.sum() / matched.sum()
        print(f"K={k}: pair recall={rec:.4f}  cands/S1={len(c) / len(s1):.2f}")
    # misses by country / empty-address
    found = np.zeros(len(q), bool)
    found[cand.q_idx.values[cand.hit.values]] = True
    miss = matched & ~found
    print("miss rate by country:", pd.Series(miss[matched]).groupby(q.country.values[matched]).mean().to_dict())
    print("miss rate empty addr:", miss[matched & (q.a_norm.values == "")].mean(),
          "non-empty:", miss[matched & (q.a_norm.values != "")].mean())
    ex = np.flatnonzero(miss)[:25]
    for i in ex:
        print("Q :", q.raw_name.values[i], "|", q.raw_addr.values[i])
        print("S1:", s1.raw_name.values[truth[i]], "|", s1.raw_addr.values[truth[i]])


if __name__ == "__main__":
    main()
