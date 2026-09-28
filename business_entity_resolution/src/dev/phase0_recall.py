"""Why does blocking miss true pairs whose record HAS an address?  Categorise the misses."""
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from common import load_split, gt_pairs
W = "../work_v3"
s1, q = load_split(W, "train"); truth = gt_pairs(W, s1, q)
c7 = pd.read_parquet(f"{W}/train_cand_v7.parquet")
full = pd.read_parquet(f"{W}/train_cand_K5_cap2000.parquet")
inC = np.zeros(len(q), bool); inC[c7.q_idx.values[truth[c7.q_idx.values] == c7.s1_idx.values]] = True
inK5 = np.zeros(len(q), bool); inK5[full.q_idx.values[truth[full.q_idx.values] == full.s1_idx.values]] = True
anyc = np.zeros(len(q), bool); anyc[full.q_idx.values] = True
miss = np.flatnonzero((truth >= 0) & ~inC & (q.a_norm.values != ""))
print("missed true records with an address:", len(miss), "of", (truth >= 0).sum())
print(f"  in blocking top-5 but pruned/cut to top-3: {inK5[miss].mean():.1%} | record got no candidates at all: {(~anyc[miss]).mean():.1%}")
rng = np.random.default_rng(0); smp = rng.choice(miss, min(60000, len(miss)), replace=False)
t = truth[smp]
ns = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(q.n_core.values[smp], s1.n_core.values[t])])
ad = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(q.a_words.values[smp], s1.a_words.values[t])])
qn = [set(x.split()) for x in q.a_nums.values[smp]]; sn = [set(x.split()) for x in s1.a_nums.values[t]]
num = np.array([bool(a & b) for a, b in zip(qn, sn)])
nat = q.raw_name.str.contains("[ऀ-ൿ]").values[smp]
cat = np.where(ns >= 80, np.where(ad >= 70, "name OK + address OK (ranking loss)", "name OK, address different"),
       np.where(ad >= 70, "address OK, name different (DBA/brand)", "both different"))
df = pd.DataFrame({"cat": cat, "num": num, "nat": nat, "country": q.country.values[smp]})
print(df.groupby("cat").agg(share=("num", "size"), shares_a_number=("num", "mean"), native=("nat", "mean"))
      .assign(share=lambda d: (d.share / len(df)).round(3)).round(3).to_string())
print(df.groupby(["country", "cat"]).size().unstack(fill_value=0).to_string())
for label in df.cat.unique():
    print(f"\n--- {label} ---")
    for i in np.flatnonzero(cat == label)[:6]:
        j = smp[i]; k = truth[j]
        print(f"Q : {q.raw_name.values[j]} | {q.raw_addr.values[j]}\nS1: {s1.raw_name.values[k]} | {s1.raw_addr.values[k]}")
