"""For vocabulary-adapted (low-coverage) countries, restore exact-identity pairs dropped by the
adapted model: pairs with an identical core name AND an identical first house number that the
vocabulary-free model (v7) accepted.  On train this pattern is a true match 98.6 % of the time.
A record already assigned elsewhere keeps its assignment (one entity per record).

  python restore_exact.py <work> <base_out> <fallback_out> <out> [min_coverage=0.8]

Countries are chosen automatically: those whose test name words are covered by the training
word-odds vocabulary less than min_coverage (the same criterion as v10.py; open set of labels).
"""
import os, shutil, sys
import pandas as pd
from common import load_split

W, BASE, FB, OUT = sys.argv[1:5]
MIN_COV = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
s1, q = load_split(W, "test")
V = set(pd.read_parquet(f"{W}/v8_word_odds.parquet").index)
t = q[["n_full", "country"]].assign(t=q.n_full.str.split()).explode("t").dropna()
t = t[t.t.str.len() > 1]
cov = t.groupby("country").t.apply(lambda x: x.isin(V).mean())
COUNTRIES = set(cov[cov < MIN_COV].index)
print("vocabulary coverage:", cov.round(3).to_dict(), "| low-coverage:", sorted(COUNTRIES))
S = s1.set_index("entity_id"); Q = q.set_index("entity_id")
def load(p):
    m = pd.read_csv(f"{p}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    e = m[m.matched_entity_ids != ""].assign(q=lambda d: d.matched_entity_ids.str.split(",")).explode("q")
    return m, e[["source1_entity_id", "q"]].rename(columns={"source1_entity_id": "s"})
base_m, base_e = load(BASE); _, fb_e = load(FB)
keep = set(S.index[S.country.isin(COUNTRIES)])
cand = fb_e[fb_e.s.isin(keep)].merge(base_e, how="left", on=["s", "q"], indicator=True)
cand = cand[cand._merge == "left_only"][["s", "q"]]
fq = Q.a_nums.str.split(" ", n=1).str[0]; fs = S.a_nums.str.split(" ", n=1).str[0]
exact = ((Q.n_core.reindex(cand.q).values == S.n_core.reindex(cand.s).values)
         & (fq.reindex(cand.q).values == fs.reindex(cand.s).values) & (fq.reindex(cand.q).values != ""))
add = cand[exact & ~cand.q.isin(set(base_e.q))]
out = pd.concat([base_e, add]).groupby("s").q.agg(",".join)
res = base_m[["source1_entity_id"]].copy()
res["matched_entity_ids"] = res.source1_entity_id.map(out).fillna("")
os.makedirs(OUT, exist_ok=True)
res.to_csv(f"{OUT}/matching_results.tsv", sep="\t", index=False)
shutil.copy(f"{BASE}/candidate_pairs.tsv", f"{OUT}/candidate_pairs.tsv")
print(f"restored {len(add)} exact-identity pairs in {sorted(COUNTRIES)} (skipped {int(exact.sum()) - len(add)} whose record is assigned elsewhere)")
