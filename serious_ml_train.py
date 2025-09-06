#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ML Entry Filter (RF / XGBoost / CatBoost) with EV threshold search
- Robust to missing columns in candidates: sec_dt, entry_time, features_key_dt
"""

import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

# Optional imports (guarded)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False


# ----------------------- Utilities -----------------------

def chronological_split(df: pd.DataFrame, frac_train=0.7):
    n = len(df)
    ntr = int(n * frac_train)
    return df.iloc[:ntr].copy(), df.iloc[ntr:].copy()

def build_feature_matrix(feats: pd.DataFrame):
    """
    Keep everything numeric except obvious leakage/ids/prices.
    """
    drop_like = {
        "open","high","low","close",
        "day_open","day_high","day_low","day_close",
        "prev_day_open","prev_day_high","prev_day_low","prev_day_close",
        "entry_price","price_event","mfe","mae","tp","sl","horizon_s",
        "label","side","side_enc",
        "datetime","datetime_event","sec_dt","entry_time","features_key_dt",
    }
    cols = [c for c in feats.columns if (c not in drop_like) and (not c.startswith("Unnamed"))]
    X = feats[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    for c in cols:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(np.int8)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, cols

def expected_value_from_probs(p: np.ndarray, tp: float, sl: float, fee: float) -> np.ndarray:
    return p*tp - (1.0 - p)*sl - fee

def search_threshold(proba: np.ndarray, y_true: np.ndarray, tp: float, sl: float, fee: float, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    evs = expected_value_from_probs(proba, tp, sl, fee)
    best = {
        "best_total_thr": None, "best_total_ev": -1e18, "best_total_n": 0, "best_total_avg_ev": 0.0,
        "best_avg_thr": None,   "best_avg_ev": -1e18,   "best_avg_n": 0,   "best_avg_total_ev": 0.0
    }
    rows = []
    for t in grid:
        mask = proba >= t
        n = int(mask.sum())
        total_ev = float(evs[mask].sum()) if n > 0 else 0.0
        avg_ev = float(total_ev / n) if n > 0 else -1e9
        rows.append({"threshold": float(t), "n_selected": n, "total_ev": total_ev, "avg_ev": avg_ev})
        if total_ev > best["best_total_ev"]:
            best.update({"best_total_thr": float(t), "best_total_ev": total_ev, "best_total_n": n, "best_total_avg_ev": avg_ev})
        if avg_ev > best["best_avg_ev"]:
            best.update({"best_avg_thr": float(t), "best_avg_ev": avg_ev, "best_avg_n": n, "best_avg_total_ev": total_ev})
    return best, pd.DataFrame(rows)

def search_threshold_by_group(df_test_scored: pd.DataFrame, group_col: str, tp: float, sl: float, fee: float, grid=None):
    all_rows, best_by = [], {}
    for g, block in df_test_scored.groupby(group_col):
        proba = block["proba_pred"].to_numpy()
        y = block["label"].to_numpy()
        best, table = search_threshold(proba, y, tp, sl, fee, grid)
        table[group_col] = g
        all_rows.append(table)
        best_by[g] = best
    tab = pd.concat(all_rows, axis=0).reset_index(drop=True) if all_rows else pd.DataFrame()
    return best_by, tab

def _coerce_datetimes(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _ensure_entry_keys(C: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have entry_time and features_key_dt.
    If missing, derive best-effort defaults:
      - entry_time: prefer existing; else datetime_event + 1s; else try 'datetime' + 1s
      - features_key_dt: entry_time
    """
    have = set(C.columns)
    # Coerce any present date-like columns
    C = _coerce_datetimes(C, ["datetime_event", "sec_dt", "entry_time", "features_key_dt", "datetime"])

    if "entry_time" not in have:
        if "datetime_event" in have:
            C["entry_time"] = C["datetime_event"] + pd.Timedelta(seconds=1)
            print("⚠️  candidates: 'entry_time' missing → using datetime_event+1s as fallback.")
        elif "datetime" in have:
            C["entry_time"] = C["datetime"] + pd.Timedelta(seconds=1)
            print("⚠️  candidates: 'entry_time' & 'datetime_event' missing → using 'datetime'+1s as fallback.")
        else:
            raise ValueError("No 'entry_time' or time column to derive it from (need at least 'datetime_event' or 'datetime').")

    if "features_key_dt" not in have:
        C["features_key_dt"] = C["entry_time"]
        print("ℹ️  candidates: 'features_key_dt' missing → set to 'entry_time'.")

    # sec_dt is NOT required for training; ignore if missing
    return C

def _parse_thr_grid(spec: str):
    try:
        s, e, step = [float(x) for x in spec.split(":")]
        grid = np.arange(s, e + 1e-9, step)
        grid = grid[(grid >= 0.0) & (grid <= 1.0)]
        if len(grid) == 0:
            raise ValueError
        return grid
    except Exception:
        return np.linspace(0.05, 0.95, 19)


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features_with_regimes*.csv (1s features)")
    ap.add_argument("--candidates", default="candidates_lr.csv", help="LR-cross candidates file")
    ap.add_argument("--model", choices=["rf", "xgb", "cat"], default="rf")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--class_weight", default="balanced")  # RF only

    # XGB
    ap.add_argument("--xgb_lr", type=float, default=0.05)
    ap.add_argument("--xgb_max_leaves", type=int, default=64)
    ap.add_argument("--xgb_depth", type=int, default=8)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    ap.add_argument("--xgb_scale_pos_weight", default="auto")

    # CatBoost
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_lr", type=float, default=0.05)
    ap.add_argument("--cat_l2", type=float, default=3.0)

    ap.add_argument("--seed", type=int, default=42)

    # EV threshold search
    ap.add_argument("--tp", type=float, default=25.0)
    ap.add_argument("--sl", type=float, default=15.0)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--group_threshold_by", default="")
    ap.add_argument("--thr_grid", default="0.05:0.95:0.05")

    args = ap.parse_args()

    # -------- Load candidates (robust) --------
    print(f"Loading candidates: {args.candidates}")
    # Read without parse_dates; coerce later for whatever exists
    C = pd.read_csv(args.candidates)
    if C.empty:
        raise SystemExit("No candidates found. Generate with entry_candidates_lr.py first.")

    # Ensure we have usable keys
    C = _ensure_entry_keys(C)
    C = C.sort_values("entry_time").reset_index(drop=True)

    # -------- Load features --------
    print(f"Loading features: {args.features}")
    F = pd.read_csv(args.features, parse_dates=["datetime"])
    F = F.sort_values("datetime").reset_index(drop=True)
    F_join = F.set_index("datetime")

    # Join features at entry_time (exact 1s bar, fallback to asof)
    try:
        feats_at_entry = F_join.loc[C["features_key_dt"].values].reset_index().rename(columns={"datetime":"features_key_dt"})
    except KeyError:
        print("Exact match failed for some rows; falling back to merge_asof backward join.")
        tmpF = F.sort_values("datetime")
        tmpC = C[["features_key_dt"]].rename(columns={"features_key_dt":"datetime"}).sort_values("datetime")
        joined = pd.merge_asof(tmpC, tmpF, on="datetime", direction="backward")
        feats_at_entry = joined.rename(columns={"datetime":"features_key_dt"})

    # Combine
    D = pd.concat([C.reset_index(drop=True), feats_at_entry.reset_index(drop=True)], axis=1)

    # Encode side
    if "side" in D.columns:
        D["side_enc"] = np.where(D["side"].astype(str)=="long", 1, -1).astype(np.int8)
    else:
        D["side_enc"] = 1  # fallback if side missing

    # Labels
    if "label" not in D.columns:
        raise SystemExit("Candidates file has no 'label' column. Re-run entry_candidates_lr.py to label events.")

    # Chronological split
    Dtr, Dte = chronological_split(D, 0.70)

    # Feature matrix
    Xtr, cols = build_feature_matrix(Dtr)
    ytr = Dtr["label"].astype(int).values
    Xte = Dte[cols].replace([np.inf,-np.inf], 0.0).fillna(0.0)
    yte = Dte["label"].astype(int).values

    print(f"Train size: {Xtr.shape[0]}  |  Test size: {Xte.shape[0]}  |  Features: {len(cols)}")

    # -------- Build model --------
    model_name = args.model.lower()
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=(args.max_depth if args.max_depth else None),
            n_jobs=-1,
            random_state=args.seed,
            class_weight=(None if args.class_weight in ["None","none",""] else args.class_weight)
        )
    elif model_name == "xgb":
        if not _HAS_XGB:
            raise SystemExit("XGBoost not installed. pip install xgboost")
        if args.xgb_scale_pos_weight == "auto":
            pos = max(ytr.sum(), 1)
            neg = len(ytr) - pos
            spw = float(neg / pos)
        else:
            spw = float(args.xgb_scale_pos_weight)
        max_depth = 0 if args.xgb_max_leaves and args.xgb_max_leaves > 0 else args.xgb_depth
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.xgb_lr,
            max_depth=max_depth,
            max_leaves=(args.xgb_max_leaves if args.xgb_max_leaves>0 else 0),
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=args.seed,
            n_jobs=-1,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="auc",
            scale_pos_weight=spw,
        )
    elif model_name == "cat":
        if not _HAS_CAT:
            raise SystemExit("CatBoost not installed. pip install catboost")
        clf = CatBoostClassifier(
            iterations=args.n_estimators,
            depth=args.max_depth if args.max_depth else 8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False
        )
    else:
        raise ValueError("Unknown model.")

    # -------- Train --------
    print(f"Training {model_name.upper()} …")
    clf.fit(Xtr, ytr)

    # -------- Evaluate --------
    proba_tr = clf.predict_proba(Xtr)[:,1]
    proba_te = clf.predict_proba(Xte)[:,1]

    auc_tr = roc_auc_score(ytr, proba_tr)
    auc_te = roc_auc_score(yte, proba_te)
    ap_te  = average_precision_score(yte, proba_te)

    print(f"AUC  train: {auc_tr:.3f}")
    print(f"AUC  test : {auc_te:.3f}")
    print(f"AP   test : {ap_te:.3f}")
    print("\nTEST classification report @ default 0.5 threshold:")
    print(classification_report(yte, (proba_te>=0.5).astype(int), digits=3))

    # -------- Save model --------
    dump(clf, "ml_entry_model.joblib")
    print("Saved ml_entry_model.joblib")

    # -------- Feature importance --------
    fi = None
    try:
        if hasattr(clf, "feature_importances_") and clf.feature_importances_ is not None:
            fi = pd.DataFrame({"feature": cols, "importance": clf.feature_importances_.astype(float)})
        elif _HAS_CAT and isinstance(clf, CatBoostClassifier):
            imp = clf.get_feature_importance(type="PredictionValuesChange")
            fi = pd.DataFrame({"feature": cols, "importance": imp})
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

    if fi is not None:
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.to_csv("ml_entry_feature_importance.csv", index=False)
        print("Wrote ml_entry_feature_importance.csv")
    else:
        print("No native feature_importances_; skipping export.")

    # -------- Score TEST & save --------
    Dte_out = Dte.copy()
    Dte_out["proba_pred"] = proba_te
    Dte_out.to_csv("ml_entry_candidates_scored.csv", index=False)
    print("Wrote ml_entry_candidates_scored.csv (TEST split).")

    # -------- EV-optimal thresholds --------
    grid = _parse_thr_grid(args.thr_grid)

    best_global, table_global = search_threshold(proba_te, yte, tp=args.tp, sl=args.sl, fee=args.fee, grid=grid)
    table_global.to_csv("ml_entry_thresholds_global_curve.csv", index=False)
    print("Wrote ml_entry_thresholds_global_curve.csv")

    thresholds_rows = [{"group_by": "GLOBAL", **best_global}]

    if args.group_threshold_by:
        gcol = args.group_threshold_by
        if gcol not in Dte_out.columns:
            print(f"Group column '{gcol}' not found in data; skipping per-group thresholds.")
        else:
            best_by, table_by = search_threshold_by_group(Dte_out, group_col=gcol, tp=args.tp, sl=args.sl, fee=args.fee, grid=grid)
            table_by.to_csv(f"ml_entry_thresholds_by_{gcol}_curve.csv", index=False)
            print(f"Wrote ml_entry_thresholds_by_{gcol}_curve.csv")
            for g, dct in best_by.items():
                thresholds_rows.append({"group_by": f"{gcol}={g}", **dct})

    thr_df = pd.DataFrame(thresholds_rows)
    thr_df.to_csv("ml_entry_thresholds.csv", index=False)
    print("Wrote ml_entry_thresholds.csv (EV-optimal).")

    print("\nEV-optimal thresholds (summary):")
    print(thr_df.to_string(index=False))

if __name__ == "__main__":
    main()
