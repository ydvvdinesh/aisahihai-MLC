import numpy as np
import pandas as pd


def load_split(work, split):
    s1 = pd.read_parquet(f"{work}/{split}_s1.parquet")
    q = pd.concat([pd.read_parquet(f"{work}/{split}_s2.parquet"),
                   pd.read_parquet(f"{work}/{split}_s3.parquet")], ignore_index=True)
    return s1, q


def gt_pairs(work, s1, q):
    """Positional (q_idx -> true s1_idx) array, -1 for unmatched records."""
    gt = pd.read_parquet(f"{work}/train_gt.parquet")
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    q_pos = pd.Series(np.arange(len(q)), index=q.entity_id.values)
    truth = np.full(len(q), -1, dtype=np.int64)
    truth[q_pos.loc[ex.m.values].values] = s1_pos.loc[ex.source1_entity_id.values].values
    return truth


def fbeta_macro(pred_sets, true_sets, beta=0.5):
    """pred_sets/true_sets: lists of python sets, aligned per S1 entity."""
    b2 = beta * beta
    tot = 0.0
    for p, t in zip(pred_sets, true_sets):
        if not t and not p:
            tot += 1.0
            continue
        if not p or not t:
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += (1 + b2) * pr * rc / (b2 * pr + rc)
    return tot / len(true_sets)


def fbeta_from_assign(assign, truth, n_s1, s1_subset=None, beta=0.5):
    """Vectorised macro F-beta.
    assign: predicted s1 idx for every q record (-1 = unassigned)
    truth:  true s1 idx for every q record (-1 = distractor)
    s1_subset: optional boolean mask of S1 entities to evaluate on."""
    b2 = beta * beta
    npred = np.bincount(assign[assign >= 0], minlength=n_s1).astype(np.float64)
    ntrue = np.bincount(truth[truth >= 0], minlength=n_s1).astype(np.float64)
    ok = (assign >= 0) & (assign == truth)
    tp = np.bincount(assign[ok], minlength=n_s1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(npred > 0, tp / npred, 0)
        rc = np.where(ntrue > 0, tp / ntrue, 0)
        f = np.where(tp > 0, (1 + b2) * pr * rc / (b2 * pr + rc), 0.0)
    f = np.where((npred == 0) & (ntrue == 0), 1.0, f)
    if s1_subset is not None:
        f = f[s1_subset]
    return f.mean()
