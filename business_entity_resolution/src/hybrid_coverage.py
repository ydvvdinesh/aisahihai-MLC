"""Coverage-based fallback: use the model with learned word-difference features (v8) only where
the learned vocabulary covers the records' name words; elsewhere fall back to the model without
vocabulary-dependent features (v7).  Coverage is measured per country label on the test
Source-2/3 records (open set: no country is named in the code).

  python hybrid_coverage.py <work> <out_hi_cov> <out_lo_cov> <out> [min_coverage=0.8]
"""
import os, shutil, sys
import pandas as pd

W, HI, LO, OUT = sys.argv[1:5]
MIN_COV = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
V = set(pd.read_parquet(f"{W}/v8_word_odds.parquet").index)
q = pd.concat([pd.read_parquet(f"{W}/test_s{k}.parquet", columns=["n_full", "country"]) for k in (2, 3)])
t = q.assign(t=q.n_full.str.split()).explode("t").dropna()
t = t[t.t.str.len() > 1]
cov = t.groupby("country").t.apply(lambda s: s.isin(V).mean())
low = set(cov[cov < MIN_COV].index)
print("vocabulary coverage:", cov.round(3).to_dict(), "| fallback countries:", sorted(low))
s1 = pd.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "country"])
use_lo = s1.entity_id[s1.country.isin(low)]
hi = pd.read_csv(f"{HI}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
lo = pd.read_csv(f"{LO}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
out = hi.copy()
out.loc[use_lo, "matched_entity_ids"] = lo.loc[use_lo, "matched_entity_ids"]
os.makedirs(OUT, exist_ok=True)
out.reset_index().to_csv(f"{OUT}/matching_results.tsv", sep="\t", index=False)
shutil.copy(f"{HI}/candidate_pairs.tsv", f"{OUT}/candidate_pairs.tsv")
print("rows from fallback model:", len(use_lo), "of", len(out))
