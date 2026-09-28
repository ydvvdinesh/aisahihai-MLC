"""Cross-encoder pair scorer (xlm-roberta-base, MIT license), fine-tuned on *uncertain* pairs.

Input text: "<name> | <address>" of the Source-2/3 record and of the Source-1 candidate, encoded
together so the model can compare them token by token (one changed word, injected numbers,
transliteration).  Trained with the same 2 folds (grouped by Source-1 entity) as LightGBM, so the
train scores are out-of-fold and can be used as a stacked feature without leakage; test scores
average the two fold models.

  python cross_encoder.py <emb_dir> train     # writes ce_train_scores.npy (OOF logits)
  python cross_encoder.py <emb_dir> test      # writes ce_test_scores.npy
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "xlm-roberta-base"
MAXLEN, BS, LR, EPOCHS = 128, 256, 3e-5, 1
t0 = time.time()
log = lambda *a: print(f"{time.time()-t0:6.0f}s", *a, flush=True)
D, MODE = sys.argv[1], sys.argv[2]
dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)


def batches(a, b, y=None, shuffle=False):
    idx = np.random.permutation(len(a)) if shuffle else np.arange(len(a))
    for i in range(0, len(idx), BS):
        j = idx[i:i + BS]
        enc = tok(list(a[j]), list(b[j]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield {k: v.to(dev) for k, v in enc.items()}, (None if y is None else torch.tensor(y[j], dtype=torch.float32, device=dev))


@torch.no_grad()
def predict(model, a, b):
    model.eval()
    out = []
    for enc, _ in batches(a, b):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def train(a, b, y):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(a) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    for ep in range(EPOCHS):
        for enc, yy in batches(a, b, y, shuffle=True):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(**enc).logits.float().squeeze(-1), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 500 == 0:
                log(f"step {step}/{steps} loss {loss.item():.4f}")
    return model


if MODE == "train":
    d = pd.read_parquet(f"{D}/ce_train.parquet")
    a, b, y, fold = d.a.values, d.b.values, d.y.values, d.fold.values
    oof = np.zeros(len(d), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        log(f"fold {k}: train {tr.sum()} / predict {va.sum()}")
        model = train(a[tr], b[tr], y[tr])
        oof[va] = predict(model, a[va], b[va])
        model.save_pretrained(f"{D}/ce_fold{k}")
        from sklearn.metrics import roc_auc_score
        log(f"fold {k} AUC {roc_auc_score(y[va], oof[va]):.4f}  (v7 prob AUC {roc_auc_score(y[va], d.p7.values[va]):.4f})")
        del model; torch.cuda.empty_cache()
    np.save(f"{D}/ce_train_scores.npy", oof)
    log("saved train OOF")
else:
    d = pd.read_parquet(f"{D}/ce_test.parquet")
    s = np.zeros(len(d), np.float32)
    for k in (0, 1):
        model = AutoModelForSequenceClassification.from_pretrained(f"{D}/ce_fold{k}").to(dev)
        s += predict(model, d.a.values, d.b.values) / 2
        log(f"test fold {k} done"); del model; torch.cuda.empty_cache()
    np.save(f"{D}/ce_test_scores.npy", s)
    log("saved test scores", len(s))
