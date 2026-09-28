"""Load raw TSVs, normalise every record in parallel, cache as parquet."""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd

from normalize import normalize_record

COLS = ["n_full", "n_core", "n_alias", "n_nospace", "is_web", "has_alias",
        "a_norm", "a_nums", "a_words", "n_skel"]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _work(args):
    names, addrs = args
    rows = [normalize_record(n, a) for n, a in zip(names, addrs)]
    return pd.DataFrame(rows, columns=COLS)


def normalize_df(df, procs):
    n = len(df)
    step = 20000
    chunks = [(df.business_name.values[i:i + step], df.business_address.values[i:i + step])
              for i in range(0, n, step)]
    with Pool(procs) as pool:
        parts = pool.map(_work, chunks)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "entity_id", df.entity_id.values)
    out["country"] = df.country.values
    out["raw_name"] = df.business_name.values
    out["raw_addr"] = df.business_address.values
    for c in ("is_web", "has_alias"):
        out[c] = out[c].astype("int8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset/ directory")
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for s in (1, 2, 3):
        t = time.time()
        df = read_tsv(os.path.join(a.data, a.split, f"{a.split}_source{s}.tsv"))
        out = normalize_df(df, a.procs)
        out.to_parquet(os.path.join(a.work, f"{a.split}_s{s}.parquet"), index=False)
        print(f"{a.split} source{s}: {len(out)} rows in {time.time() - t:.0f}s", flush=True)
    if a.split == "train":
        gt = read_tsv(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        gt.to_parquet(os.path.join(a.work, "train_gt.parquet"), index=False)


if __name__ == "__main__":
    main()
