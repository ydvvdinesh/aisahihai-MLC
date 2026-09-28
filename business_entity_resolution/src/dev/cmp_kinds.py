"""Break the pairs that differ between two outputs down by house-number mismatch kind,
alongside the train prior (share of true matches) for each kind."""
import sys, numpy as np, pandas as pd
from common import load_split
W = "../work_v3"; s1, q = load_split(W, "test")
qf = pd.Series(q.a_nums.str.split(" ", n=1).str[0].values, index=q.entity_id.values)
sf = pd.Series(s1.a_nums.str.split(" ", n=1).str[0].values, index=s1.entity_id.values)
def pairs(p):
    m = pd.read_csv(f"../{p}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False); m = m[m.matched_entity_ids != ""]
    e = m.assign(q=m.matched_entity_ids.str.split(",")).explode("q"); return set(zip(e.source1_entity_id, e.q))
def kind(a, b):
    if not a or not b: return "missing-num"
    if a == b: return "equal"
    if len(a) == len(b):
        d = abs(int(a[:15]) - int(b[:15])); return "same-len d<=2" if d <= 2 else ("same-len d3-50 (SIBLING)" if d <= 50 else "same-len d>50")
    if a.startswith(b) or b.startswith(a): return "prefix (digit dropped end)"
    if a.endswith(b) or b.endswith(a): return "suffix (digit dropped front)"
    if a in b or b in a: return "substring"
    return "other"
prior = {"equal": None, "same-len d<=2": .193, "same-len d3-50 (SIBLING)": .06, "same-len d>50": .52,
         "prefix (digit dropped end)": .715, "suffix (digit dropped front)": .843, "substring": .607, "other": .401, "missing-num": None}
A, B = sys.argv[1], sys.argv[2]; PA, PB = pairs(A), pairs(B)
if len(sys.argv) > 3:
    keep = set(s1.entity_id.values[s1.country.values == sys.argv[3]])
    PA = {p for p in PA if p[0] in keep}; PB = {p for p in PB if p[0] in keep}
rows = {}
for lab, ps in ((f"{B} adds", PB - PA), (f"{B} removes", PA - PB)):
    d = pd.DataFrame(list(ps), columns=["s", "q"])
    k = [kind(a, b) for a, b in zip(qf.reindex(d.q).fillna("").values, sf.reindex(d.s).fillna("").values)]
    rows[lab] = pd.Series(k).value_counts()
out = pd.DataFrame(rows).fillna(0).astype(int); out["train P(true)"] = [prior.get(i) for i in out.index]
print(out.sort_values(f"{B} adds", ascending=False).to_string())
