#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML baseline trainer (chunked, leak-safe, keep ALL features)
----------------------------------------------------------
- Labels:           candidates_lr.csv  (from your labeler; includes entry_time + features_key_dt)
- Features source:  features.csv (1s unified features; HUGE) -> streamed in chunks; we keep only rows matching features_key_dt
- Merge key:        features_key_dt (exact entry second)  ==  features.datetime

Training:
- Chronological split: 60% train / 20% valid / 20% test
- Cost-sensitive weights (pos weighted by SL, neg by TP)
- Optional probability calibration (isotonic) on VALID
- Threshold picked on VALID to maximize expected value (EV)

Outputs:
- Console metrics + EV/trade
- ml_preds_test.csv (entry_time, prob, label, side, regime if present)

Leak-safety:
- We only join features at the *entry* timestamp (the second immediately after the event minute),
  which your labeler already guarantees uses *past* info via shifting/merge_asof.
"""

import os, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# ----------------------- Config -----------------------
CAND_PATH  = "candidates_lr.csv"                          # labels
FEATS_PATH = "/content/drive/MyDrive/data/features.csv"   # unified features (1s)
OUT_PRED_CSV = "ml_preds_test.csv"

# Payoff model
TP, SL, COST = 30.0, 15.0, 0.0

# Streaming config (tune for RAM)
CHUNK_ROWS  = 400_000     # lower to 200k if RAM is tight
PRINT_EVERY = 5           # progress print frequency (in chunks)

# Model/Training config
CALIBRATE_PROBS = True    # isotonic calibration on VALID (recommended)
TRAIN_FRACTION  = 0.60
VALID_FRACTION  = 0.20    # TEST is the remaining 20%

# ----------------------- Helpers -----------------------
def _downcast_numeric_inplace(df: pd.DataFrame) -> None:
    """Shrink numeric dtypes to save RAM (in-place)."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        # int32 is a safe default; change to int16 if certain
        df[c] = df[c].astype(np.int32)

def _load_matching_feature_rows(feats_path: str,
                                needed_datetimes_ns: np.ndarray,
                                chunksize: int,
                                print_every: int) -> pd.DataFrame:
    """
    Stream features.csv and keep ONLY rows whose datetime is in needed_datetimes_ns (int64 epoch ns).
    Returns a deduplicated DataFrame with all original columns.
    """
    need = np.unique(needed_datetimes_ns)  # ensure unique
    kept = []
    for ci, chunk in enumerate(pd.read_csv(
        feats_path,
        parse_dates=["datetime"],
        chunksize=chunksize
    ), start=1):
        # vectorized membership via np.isin on int64 epoch ns
        dt_i64 = chunk["datetime"].values.astype("datetime64[ns]").astype("int64")
        mask = np.isin(dt_i64, need, assume_unique=False)
        if mask.any():
            sub = chunk.loc[mask].copy()
            _downcast_numeric_inplace(sub)
            kept.append(sub)

        if ci % print_every == 0:
            matched = sum(len(k) for k in kept)
            print(f"  …processed {ci} chunks, matched rows so far: {matched}")

    if not kept:
        raise RuntimeError("No matching datetimes found in features for the candidate entries.")
    fe_small = pd.concat(kept, axis=0, ignore_index=True)
    fe_small = fe_small.drop_duplicates(subset=["datetime"], keep="last")
    return fe_small

def _ev_from_precision(precision: float, tp=TP, sl=SL, cost=COST) -> float:
    """Expected value per trade given precision."""
    return precision*tp - (1.0 - precision)*sl - cost

# ----------------------- Main -----------------------
def main():
    if not os.path.exists(CAND_PATH):
        raise FileNotFoundError(f"Missing {CAND_PATH}")
    if not os.path.exists(FEATS_PATH):
        raise FileNotFoundError(f"Missing {FEATS_PATH}")

    print("Loading candidates…")
    cand = pd.read_csv(
        CAND_PATH,
        parse_dates=["datetime_event","entry_time","features_key_dt"]
    )
    if cand.empty:
        raise RuntimeError("candidates_lr.csv is empty.")

    # y and time ordering (for chronological split)
    y = cand["label"].astype("int8").values
    entry_time = cand["entry_time"].values.astype("datetime64[ns]")

    # Datetimes we need from features table (as int64 epoch ns for fast matching)
    need_dt_ns = cand["features_key_dt"].values.astype("datetime64[ns]").astype("int64")

    # Quick header check
    hdr = pd.read_csv(FEATS_PATH, nrows=0)
    if "datetime" not in hdr.columns:
        raise ValueError("features.csv must contain a 'datetime' column.")

    print("Streaming features.csv (ALL columns) in chunks…")
    fe_small = _load_matching_feature_rows(FEATS_PATH, need_dt_ns, CHUNK_ROWS, PRINT_EVERY)
    del hdr; gc.collect()

    # Merge leak-safe at the exact entry second
    df = cand.merge(
        fe_small, left_on="features_key_dt", right_on="datetime",
        how="inner", suffixes=("","_feat")
    )
    if len(df) < len(cand):
        miss = len(cand) - len(df)
        print(f"⚠️  {miss} candidates had no matching features row (e.g., last row of file).")

    # Build feature matrix X: keep ALL numeric features except labels/ids/prices
    drop_cols = {
        "label","hit","t_hit_s","mfe","mae","tp","sl","horizon_s",
        "datetime_event","entry_time","features_key_dt","datetime",
        "price_event","entry_price","side"  # side used for diagnostics, not as a feature
    }
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = feat_df.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0.0)
    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype(np.float32)

    # Chronological split: 60% train / 20% valid / 20% test (by entry_time)
    order = np.argsort(entry_time)
    n = len(order)
    n_tr = int(TRAIN_FRACTION * n)
    n_va = int((TRAIN_FRACTION + VALID_FRACTION) * n)

    idx_tr = order[:n_tr]
    idx_va = order[n_tr:n_va]
    idx_te = order[n_va:]

    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_va, y_va = X.iloc[idx_va], y[idx_va]
    X_te, y_te = X.iloc[idx_te], y[idx_te]

    # Cost-sensitive sample weights: positives ~ SL, negatives ~ TP
    w_tr = np.where(y_tr == 1, SL, TP).astype("float32")

    # Model
    clf = HistGradientBoostingClassifier(
        max_depth=None,
        max_leaf_nodes=31,
        learning_rate=0.05,
        min_samples_leaf=200,
        l2_regularization=0.0,
        random_state=42
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)

    # Optional calibration on VALID (no leakage)
    if CALIBRATE_PROBS:
        cal = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
        cal.fit(X_va, y_va)
        prob_tr = cal.predict_proba(X_tr)[:, 1]
        prob_va = cal.predict_proba(X_va)[:, 1]
        prob_te = cal.predict_proba(X_te)[:, 1]
    else:
        prob_tr = clf.predict_proba(X_tr)[:, 1]
        prob_va = clf.predict_proba(X_va)[:, 1]
        prob_te = clf.predict_proba(X_te)[:, 1]

    # AUC/AP
    print(f"Train AUC {roc_auc_score(y_tr, prob_tr):.3f} | Valid AUC {roc_auc_score(y_va, prob_va):.3f} | Test AUC {roc_auc_score(y_te, prob_te):.3f}")
    print(f"Train AP  {average_precision_score(y_tr, prob_tr):.3f} | Valid AP  {average_precision_score(y_va, prob_va):.3f} | Test AP  {average_precision_score(y_te, prob_te):.3f}")

    # Choose threshold on VALID to maximize EV
    prec, rec, ths = precision_recall_curve(y_va, prob_va)
    # thresholds has len = len(prec)-1; use matching slice prec[1:]
    if len(ths) > 0:
        ev = prec[1:]*TP - (1.0 - prec[1:])*SL - COST
        best_idx = int(np.nanargmax(ev))
        thr = float(ths[best_idx])
    else:
        # degenerate: fallback to EV-neutral probability
        thr = (SL + COST) / (TP + SL)
    print(f"Chosen threshold (from VALID): {thr:.3f}")

    # Test evaluation at EV-optimal threshold
    pred_te = prob_te >= thr
    n_trades = int(pred_te.sum())
    if n_trades > 0:
        precision = float((y_te[pred_te] == 1).mean())
        ev_trade = _ev_from_precision(precision, TP, SL, COST)
    else:
        precision, ev_trade = float("nan"), float("nan")

    print(f"TEST → trades: {n_trades}, precision: {precision:.3f}, EV/trade: {ev_trade:.2f} pts")

    # Optional per-regime & per-side diagnostics (if present)
    te_meta = df.iloc[idx_te].copy()
    te_meta["prob"] = prob_te
    te_meta["pred"] = pred_te
    if "regime" in te_meta.columns:
        for r in sorted(pd.unique(te_meta["regime"].dropna())):
            mask = te_meta["regime"] == r
            if mask.sum() == 0 or te_meta.loc[mask, "pred"].sum() == 0:
                continue
            p = (y_te[mask & te_meta["pred"].values] == 1).mean()
            ev_r = _ev_from_precision(float(p), TP, SL, COST)
            print(f"Regime {int(r)} → trades: {int(te_meta.loc[mask,'pred'].sum())}, precision: {p:.3f}, EV/trade: {ev_r:.2f}")

    if "side" in df.columns:
        for s in ["long","short"]:
            mask = (df.iloc[idx_te]["side"].astype(str) == s)
            if mask.sum() == 0:
                continue
            sub_pred = pred_te[mask.values]
            if sub_pred.sum() == 0:
                continue
            sub_y = y_te[mask.values]
            p = (sub_y[sub_pred] == 1).mean()
            ev_s = _ev_from_precision(float(p), TP, SL, COST)
            print(f"Side {s:<5} → trades: {int(sub_pred.sum())}, precision: {p:.3f}, EV/trade: {ev_s:.2f}")

    # Save test predictions for inspection
    out = te_meta[["entry_time","prob","pred"]].copy()
    out["label"] = y_te
    for col in ["side","regime"]:
        if col in te_meta.columns:
            out[col] = te_meta[col].values
    out.to_csv(OUT_PRED_CSV, index=False)
    print(f"✅ Wrote {OUT_PRED_CSV} with {len(out)} rows.")

if __name__ == "__main__":
    main()
