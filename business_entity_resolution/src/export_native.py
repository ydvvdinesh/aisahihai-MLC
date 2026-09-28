"""Export the (small) name/address bundle used by the LaBSE embedding step:
all records whose business name is written in an Indic script, plus the Source-1 records of
the countries those records belong to."""
import argparse, os
import pandas as pd

NATIVE = r"[ऀ-ൿ]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--emb_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.emb_dir, exist_ok=True)
    cols = ["entity_id", "raw_name", "raw_addr", "country"]
    for sp in ("train", "test"):
        s1 = pd.read_parquet(f"{a.work}/{sp}_s1.parquet", columns=cols)
        q = pd.concat([pd.read_parquet(f"{a.work}/{sp}_s{k}.parquet", columns=cols) for k in (2, 3)], ignore_index=True)
        qn = q[q.raw_name.str.contains(NATIVE)]
        s1[s1.country.isin(set(qn.country))].to_parquet(f"{a.emb_dir}/{sp}_s1_names.parquet", index=False)
        qn.to_parquet(f"{a.emb_dir}/{sp}_native_q.parquet", index=False)
        print(sp, "native-script records:", len(qn))


if __name__ == "__main__":
    main()
