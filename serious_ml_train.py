#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serious ML Trainer (chunked, per-side models, calibrated, EV thresholds)
- Input:
    candidates_lr.csv  (from your labeler; includes features_key_dt, side, label, entry_time)
    features.csv       (unified 1s features; leak-safe)
- Split: chronological by day → Train 60%, Valid 20%, Test 20%
- Models: HistGradientBoostingClassifier (one per side)
- Calibration: Isotonic on VALID only (no leakage)
- Thresholds: per-side, chosen on VALID to maximize EV with min-trades floor
- Outputs:
    models_long.joblib, models_short.joblib
    calib_long.joblib,  calib_short.joblib
    thresholds.json
    ml_preds_test.csv   (per-trade test scores/preds)
    run_summary.json
"""

import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump

# ---------------- Config ----------------
CAND_PATH  = "candidates_lr.csv"
# FEATS_PATH = "/content/drive/MyDrive/data/features.csv"   # change if local
FEATS_PATH = "features.csv"   # change if local
TP, SL, COST = 100.0, 10.0, 0.0
MIN_TRADES_VALID_PER_SIDE = 30        # floor to avoid overfitting thresholds
CHUNK_ROWS  = 400_000                  # reduce if RAM is tight
PRINT_EVERY = 5
RANDOM_STATE = 42

# ---------------- Helpers ----------------
def _downcast_numeric_inplace(df: pd.DataFrame) -> None:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = df[c].astype(np.int32)

def _ev(tp_rate: float) -> float:
    """Expected value per trade given precision."""
    return tp_rate*TP - (1.0 - tp_rate)*SL - COST

def pick_threshold_with_floor(prob: np.ndarray, y: np.ndarray,
                              min_trades: int, grid: int = 500):
    """Pick threshold maximizing EV on VALID with a min-trades floor."""
    if len(prob) == 0:
        return 1.0, float("-inf"), 0, 0.0
    ps = np.sort(prob)
    step = max(1, len(ps)//grid)
    best_thr, best_ev, best_n = 1.0, float("-inf"), 0
    for thr in ps[::step]:
        m = prob >= thr
        n = int(m.sum())
        if n < min_trades:
            continue
        prec = (y[m] == 1).mean() if n else 0.0
        ev = _ev(float(prec))
        if ev > best_ev:
            best_ev, best_thr, best_n = ev, float(thr), n
    if best_ev == float("-inf"):
        # fallback to EV-neutral threshold
        thr = (SL + COST) / (TP + SL)
        m = prob >= thr
        n = int(m.sum())
        prec = (y[m] == 1).mean() if n else 0.0
        best_ev, best_thr, best_n = _ev(float(prec)), float(thr), n
    return best_thr, best_ev, best_n, float(prec if best_n else 0.0)

def stream_join_features(cand: pd.DataFrame, feats_path: str) -> pd.DataFrame:
    """Chunked join of all features at features_key_dt; returns merged df."""
    need_dt = pd.to_datetime(cand["features_key_dt"]).values.astype("datetime64[ns]")
    need_i64 = set(need_dt.astype("int64"))

    hdr = pd.read_csv(feats_path, nrows=0)
    if "datetime" not in hdr.columns:
        raise ValueError("features.csv must contain 'datetime' column.")

    kept = []
    for ci, chunk in enumerate(pd.read_csv(
        feats_path,
        parse_dates=["datetime"],
        chunksize=CHUNK_ROWS
    ), start=1):
        dt_i64 = chunk["datetime"].values.astype("datetime64[ns]").astype("int64")
        mask = np.fromiter((v in need_i64 for v in dt_i64), count=len(dt_i64), dtype=bool)
        if mask.any():
            sub = chunk.loc[mask].copy()
            _downcast_numeric_inplace(sub)
            kept.append(sub)
        if ci % PRINT_EVERY == 0:
            matched = sum(len(k) for k in kept)
            print(f"  …processed {ci} chunks, matched rows so far: {matched}")
    if not kept:
        raise RuntimeError("No matching datetimes found in features for the candidate entries.")
    fe_small = pd.concat(kept, axis=0, ignore_index=True)
    fe_small = fe_small.drop_duplicates(subset=["datetime"])
    merged = cand.merge(fe_small, left_on="features_key_dt", right_on="datetime",
                        how="inner", suffixes=("","_feat"))
    if len(merged) < len(cand):
        print(f"⚠️ {len(cand)-len(merged)} candidates had no matching features (likely end-of-file last second).")
    return merged

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {
        "label","hit","t_hit_s","mfe","mae","tp","sl","horizon_s",
        "datetime_event","entry_time","features_key_dt","datetime",
        "price_event","entry_price"  # keep 'side' for splitting
    }
    keep_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = keep_num.select_dtypes(include=[np.number]).copy().fillna(0.0)
    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype(np.float32)
    return X

def side_mask(series):
    s = series.astype(str).str.lower().values
    return (s == "long"), (s == "short")

# ---------------- Main ----------------
def main():
    np.random.seed(RANDOM_STATE)

    # Load candidates
    print("Loading candidates…")
    cand = pd.read_csv(
        CAND_PATH,
        parse_dates=["datetime_event","entry_time","features_key_dt"]
    )
    if cand.empty:
        raise RuntimeError("candidates_lr.csv is empty.")
    cand["side"] = cand["side"].astype(str).str.lower()
    if not set(cand["side"].unique()).issubset({"long","short"}):
        raise ValueError("Unexpected 'side' values; expected only 'long'/'short'.")

    # Chunked join
    print("Streaming features.csv (ALL columns) in chunks…")
    df = stream_join_features(cand, FEATS_PATH)

    # Sort by time; create day groups for splitting
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["day"] = pd.to_datetime(df["entry_time"]).dt.normalize()

    # Split by contiguous days: 60/20/20
    days = df["day"].drop_duplicates().to_list()
    n = len(days)
    d_tr, d_va, d_te = days[:int(0.6*n)], days[int(0.6*n):int(0.8*n)], days[int(0.8*n):]
    idx_tr = df["day"].isin(d_tr).values
    idx_va = df["day"].isin(d_va).values
    idx_te = df["day"].isin(d_te).values

    y = df["label"].astype("int8").values
    X = build_X(df)

    # Per-side masks
    m_long, m_short = side_mask(df["side"])

    # -------- Train LONG model --------
    def train_side(name, m_side):
        X_tr, y_tr = X[idx_tr & m_side], y[idx_tr & m_side]
        X_va, y_va = X[idx_va & m_side], y[idx_va & m_side]
        X_te, y_te = X[idx_te & m_side], y[idx_te & m_side]

        if len(y_tr) == 0 or len(y_va) == 0 or len(y_te) == 0:
            print(f"⚠️ Not enough {name} samples in one of the splits. Skipping this side.")
            return None

        # class weights via inverse frequency
        pos = float(y_tr.mean()) if y_tr.mean() > 0 else 1e-6
        w_tr = np.where(y_tr == 1, 0.5/pos, 0.5/(1.0-pos)).astype(np.float32)

        clf = HistGradientBoostingClassifier(
            max_depth=None,
            max_leaf_nodes=31,
            learning_rate=0.05,
            min_samples_leaf=200,
            l2_regularization=0.0,
            random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        # raw scores
        pr_tr = clf.predict_proba(X_tr)[:,1]
        pr_va = clf.predict_proba(X_va)[:,1]
        pr_te = clf.predict_proba(X_te)[:,1]

        # metrics pre-calibration
        auc_tr = roc_auc_score(y_tr, pr_tr) if len(np.unique(y_tr))>1 else np.nan
        auc_va = roc_auc_score(y_va, pr_va) if len(np.unique(y_va))>1 else np.nan
        auc_te = roc_auc_score(y_te, pr_te) if len(np.unique(y_te))>1 else np.nan
        ap_tr  = average_precision_score(y_tr, pr_tr) if len(np.unique(y_tr))>1 else np.nan
        ap_va  = average_precision_score(y_va, pr_va) if len(np.unique(y_va))>1 else np.nan
        ap_te  = average_precision_score(y_te, pr_te) if len(np.unique(y_te))>1 else np.nan
        print(f"[{name}] AUC tr/va/te: {auc_tr:.3f}/{auc_va:.3f}/{auc_te:.3f} | AP tr/va/te: {ap_tr:.3f}/{ap_va:.3f}/{ap_te:.3f}")

        # isotonic calibration on VALID only
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pr_va, y_va)
        pb_tr = iso.transform(pr_tr)
        pb_va = iso.transform(pr_va)
        pb_te = iso.transform(pr_te)

        # choose threshold on VALID
        thr, ev_va, n_va, prec_va = pick_threshold_with_floor(pb_va, y_va,
                                                              min_trades=MIN_TRADES_VALID_PER_SIDE)
        # Evaluate on TEST at chosen thr
        pred_te = (pb_te >= thr)
        n_te = int(pred_te.sum())
        prec_te = (y_te[pred_te] == 1).mean() if n_te else np.nan
        ev_te = _ev(float(prec_te)) if n_te else np.nan
        print(f"[{name}] thr={thr:.3f} | VALID trades={n_va}, prec={prec_va:.3f}, EV={ev_va:.2f} | "
              f"TEST trades={n_te}, prec={prec_te:.3f}, EV/trade={ev_te:.2f}")

        # Save artifacts
        dump(clf, f"model_{name}.joblib")
        dump(iso, f"calib_{name}.joblib")

        # Return everything needed for final reporting
        return {
            "name": name,
            "clf_path": f"model_{name}.joblib",
            "calib_path": f"calib_{name}.joblib",
            "thr": float(thr),
            "valid_trades": int(n_va),
            "valid_ev": float(ev_va),
            "test_trades": int(n_te),
            "test_precision": float(prec_te) if n_te else np.nan,
            "test_ev_per_trade": float(ev_te) if n_te else np.nan,
            "pb_te": pb_te, "y_te": y_te,
            "mask_te": (idx_te & m_side),
        }

    res_long  = train_side("long",  m_long)
    res_short = train_side("short", m_short)

    # Build test predictions CSV
    rows = []
    if res_long is not None:
        te_idx = np.where(res_long["mask_te"])[0]
        rows.append(pd.DataFrame({
            "entry_time": df.loc[te_idx, "entry_time"].values,
            "side": "long",
            "prob": res_long["pb_te"],
            "label": res_long["y_te"],
            "pred": (res_long["pb_te"] >= res_long["thr"]).astype(int)
        }))
    if res_short is not None:
        te_idx = np.where(res_short["mask_te"])[0]
        rows.append(pd.DataFrame({
            "entry_time": df.loc[te_idx, "entry_time"].values,
            "side": "short",
            "prob": res_short["pb_te"],
            "label": res_short["y_te"],
            "pred": (res_short["pb_te"] >= res_short["thr"]).astype(int)
        }))
    if rows:
        preds = pd.concat(rows, axis=0, ignore_index=True).sort_values("entry_time")
        preds.to_csv("ml_preds_test.csv", index=False)
        print("✅ Wrote ml_preds_test.csv")

    # Save thresholds & run summary
    summary = {
        "tp": TP, "sl": SL, "cost": COST,
        "min_trades_valid_per_side": MIN_TRADES_VALID_PER_SIDE,
        "chunk_rows": CHUNK_ROWS,
        "random_state": RANDOM_STATE,
        "results": {
            "long":  {k: v for k, v in (res_long or {}).items() if k not in ("pb_te","y_te","mask_te")},
            "short": {k: v for k, v in (res_short or {}).items() if k not in ("pb_te","y_te","mask_te")},
        }
    }
    with open("thresholds.json", "w") as f:
        json.dump({
            "long":  res_long["thr"]  if res_long  else None,
            "short": res_short["thr"] if res_short else None
        }, f, indent=2)
    with open("run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Saved thresholds.json and run_summary.json")

if __name__ == "__main__":
    main()
