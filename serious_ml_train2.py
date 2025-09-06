#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serious trainer v2:
- Per-side models (HGBT)
- Isotonic calibration on VALID (no leakage)
- EV-max thresholds (no hard min-trades; you can set a soft minimum)
- Regime gating (trade only HMM regimes with positive VALID EV for that side)
- Chunked feature-join; keeps ALL numeric features + new labeler context columns
"""

import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump

CAND_PATH  = "candidates_lr.csv"
FEATS_PATH = "/content/drive/MyDrive/data/features.csv"
TP, SL, COST = 30.0, 15.0, 0.0
CHUNK_ROWS, PRINT_EVERY = 400_000, 5
RANDOM_STATE = 42
SOFT_MIN_VALID_TRADES = 50   # allow fewer if EV is strongly positive

def _downcast(df):
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = df[c].astype(np.int32)

def EV(prec): return prec*TP - (1-prec)*SL - COST

def pick_thr(prob, y, soft_min=SOFT_MIN_VALID_TRADES, grid=500):
    if len(prob)==0: return 1.0, -1e9, 0, 0.0
    ps = np.sort(prob); step = max(1, len(ps)//grid)
    best = (1.0, -1e9, 0, 0.0)
    for thr in ps[::step]:
        m = prob >= thr; n = int(m.sum())
        if n==0: continue
        if n < soft_min and len(ps) > 10:  # allow but prefer larger n if EV ties
            pass
        prec = (y[m]==1).mean()
        ev = EV(float(prec))
        if ev > best[1] or (ev==best[1] and n>best[2]):
            best = (float(thr), float(ev), int(n), float(prec))
    # fallback to EV-neutral if nothing chosen
    if best[1] == -1e9:
        thr = (SL + COST) / (TP + SL)
        m = prob >= thr; n = int(m.sum())
        prec = (y[m]==1).mean() if n else 0.0
        best = (float(thr), float(EV(prec)), int(n), float(prec))
    return best

def stream_join_features(cand, feats_path):
    need_dt = pd.to_datetime(cand["features_key_dt"]).values.astype("datetime64[ns]")
    need_i64 = set(need_dt.astype("int64"))

    hdr = pd.read_csv(feats_path, nrows=0)
    if "datetime" not in hdr.columns:
        raise ValueError("features.csv must contain 'datetime' column.")

    kept = []
    for ci, chunk in enumerate(pd.read_csv(feats_path, parse_dates=["datetime"], chunksize=CHUNK_ROWS), start=1):
        dt_i64 = chunk["datetime"].values.astype("datetime64[ns]").astype("int64")
        mask = np.fromiter((v in need_i64 for v in dt_i64), count=len(dt_i64), dtype=bool)
        if mask.any():
            sub = chunk.loc[mask].copy()
            _downcast(sub)
            kept.append(sub)
        if ci % PRINT_EVERY == 0:
            print(f"  …processed {ci} chunks, matched rows so far: {sum(len(k) for k in kept)}")
    if not kept:
        raise RuntimeError("No matching feature rows for candidates.")
    fe_small = pd.concat(kept, axis=0, ignore_index=True)
    fe_small = fe_small.drop_duplicates(subset=["datetime"])
    merged = cand.merge(fe_small, left_on="features_key_dt", right_on="datetime", how="inner")
    return merged

def build_X(df):
    drop_cols = {
        # do NOT drop lr_mid_prev/lr_slope_prev/overshoot/min_* (we want them!)
        "label","hit","t_hit_s","mfe","mae","tp","sl","horizon_s",
        "datetime_event","entry_time","features_key_dt","datetime",
        "price_event","entry_price"
    }
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = keep.select_dtypes(include=[np.number]).copy().fillna(0.0)
    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype(np.float32)
    return X

def side_mask(s):
    v = s.astype(str).str.lower().values
    return (v=="long"), (v=="short")

def main():
    np.random.seed(RANDOM_STATE)

    # Load candidates with context
    print("Loading candidates…")
    cand = pd.read_csv(CAND_PATH, parse_dates=["datetime_event","entry_time","features_key_dt"])
    if cand.empty: raise RuntimeError("candidates_lr.csv is empty.")
    cand["side"] = cand["side"].astype(str).str.lower()

    print("Streaming features.csv (ALL columns) in chunks…")
    df = stream_join_features(cand, FEATS_PATH)

    # Sort & split by day
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["day"] = pd.to_datetime(df["entry_time"]).dt.normalize()
    days = df["day"].drop_duplicates().to_list()
    n = len(days)
    d_tr, d_va, d_te = days[:int(0.6*n)], days[int(0.6*n):int(0.8*n)], days[int(0.8*n):]

    idx_tr = df["day"].isin(d_tr).values
    idx_va = df["day"].isin(d_va).values
    idx_te = df["day"].isin(d_te).values

    y = df["label"].astype("int8").values
    X = build_X(df)
    m_long, m_short = side_mask(df["side"])

    def train_side(name, m_side):
        X_tr, y_tr = X[idx_tr & m_side], y[idx_tr & m_side]
        X_va, y_va = X[idx_va & m_side], y[idx_va & m_side]
        X_te, y_te = X[idx_te & m_side], y[idx_te & m_side]
        reg_tr = df.loc[idx_tr & m_side, "regime"].values if "regime" in df.columns else None
        reg_va = df.loc[idx_va & m_side, "regime"].values if "regime" in df.columns else None
        reg_te = df.loc[idx_te & m_side, "regime"].values if "regime" in df.columns else None

        if len(y_tr)==0 or len(y_va)==0 or len(y_te)==0:
            print(f"[{name}] not enough samples; skipping.")
            return None

        pos = float(y_tr.mean()) if y_tr.mean()>0 else 1e-6
        w_tr = np.where(y_tr==1, 0.5/pos, 0.5/(1.0-pos)).astype(np.float32)

        clf = HistGradientBoostingClassifier(
            max_depth=None, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=200, l2_regularization=0.0, random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        pr_tr = clf.predict_proba(X_tr)[:,1]
        pr_va = clf.predict_proba(X_va)[:,1]
        pr_te = clf.predict_proba(X_te)[:,1]

        auc_tr = roc_auc_score(y_tr, pr_tr) if len(np.unique(y_tr))>1 else np.nan
        auc_va = roc_auc_score(y_va, pr_va) if len(np.unique(y_va))>1 else np.nan
        auc_te = roc_auc_score(y_te, pr_te) if len(np.unique(y_te))>1 else np.nan
        ap_tr  = average_precision_score(y_tr, pr_tr) if len(np.unique(y_tr))>1 else np.nan
        ap_va  = average_precision_score(y_va, pr_va) if len(np.unique(y_va))>1 else np.nan
        ap_te  = average_precision_score(y_te, pr_te) if len(np.unique(y_te))>1 else np.nan
        print(f"[{name}] AUC tr/va/te: {auc_tr:.3f}/{auc_va:.3f}/{auc_te:.3f} | AP tr/va/te: {ap_tr:.3f}/{ap_va:.3f}/{ap_te:.3f}")

        # Isotonic on VALID
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(pr_va, y_va)
        pb_tr, pb_va, pb_te = iso.transform(pr_tr), iso.transform(pr_va), iso.transform(pr_te)

        # Threshold (EV-max on VALID)
        thr, ev_va, n_va, prec_va = pick_thr(pb_va, y_va)
        print(f"[{name}] VALID: thr={thr:.3f}, trades={n_va}, prec={prec_va:.3f}, EV/trade={ev_va:.2f}")

        # ---------- Regime gating (using VALID) ----------
        allowed_regs = None
        if reg_va is not None:
            allowed = []
            for r in np.unique(reg_va):
                m = (reg_va == r)
                if m.sum() == 0: continue
                pred_r = pb_va[m] >= thr
                n = int(pred_r.sum())
                if n == 0: continue
                prec = (y_va[m][pred_r] == 1).mean()
                ev = EV(float(prec))
                if ev > 0: allowed.append(int(r))
            allowed_regs = allowed
            if allowed_regs:
                print(f"[{name}] Regimes kept (VALID EV>0): {allowed_regs}")
            else:
                print(f"[{name}] No regime had positive VALID EV; no gating will be applied.")

        # TEST evaluation with gating
        if reg_te is not None and allowed_regs:
            gate = np.isin(reg_te, allowed_regs)
        else:
            gate = np.ones_like(y_te, dtype=bool)

        pred_te = (pb_te >= thr) & gate
        n_te = int(pred_te.sum())
        prec_te = (y_te[pred_te] == 1).mean() if n_te else np.nan
        ev_te = EV(float(prec_te)) if n_te else np.nan
        print(f"[{name}] TEST (gated): trades={n_te}, precision={prec_te:.3f}, EV/trade={ev_te:.2f}")

        dump(clf, f"model_{name}.joblib"); dump(iso, f"calib_{name}.joblib")
        return {
            "name": name, "thr": float(thr),
            "valid_trades": int(n_va), "valid_ev": float(ev_va),
            "test_trades": int(n_te), "test_precision": float(prec_te) if n_te else np.nan,
            "test_ev_per_trade": float(ev_te) if n_te else np.nan,
            "pb_te": pb_te, "y_te": y_te, "gate": gate
        }

    res_long  = train_side("long",  (df["side"].values=="long"))
    res_short = train_side("short", (df["side"].values=="short"))

    # Write test preds
    rows = []
    if res_long is not None:
        rows.append(pd.DataFrame({
            "entry_time": df.loc[df["side"].values=="long", "entry_time"].values,
            "side": "long",
            "prob": res_long["pb_te"],
            "label": res_long["y_te"],
            "pred": (res_long["pb_te"] >= res_long["thr"]).astype(int),
            "gate": res_long["gate"].astype(int)
        }))
    if res_short is not None:
        rows.append(pd.DataFrame({
            "entry_time": df.loc[df["side"].values=="short", "entry_time"].values,
            "side": "short",
            "prob": res_short["pb_te"],
            "label": res_short["y_te"],
            "pred": (res_short["pb_te"] >= res_short["thr"]).astype(int),
            "gate": res_short["gate"].astype(int)
        }))
    if rows:
        out = pd.concat(rows, axis=0, ignore_index=True).sort_values("entry_time")
        out.to_csv("ml_preds_test.csv", index=False)
        print("✅ Wrote ml_preds_test.csv")

    with open("run_summary.json","w") as f:
        json.dump({
            "tp": TP, "sl": SL, "cost": COST,
            "results": {
                "long":  {k:v for k,v in (res_long or {}).items() if k not in ("pb_te","y_te","gate")},
                "short": {k:v for k,v in (res_short or {}).items() if k not in ("pb_te","y_te","gate")},
            }
        }, f, indent=2)
    print("✅ Saved run_summary.json")

if __name__ == "__main__":
    main()
