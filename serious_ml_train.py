#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ML Entry Filter (RF / XGBoost / CatBoost) with EV threshold search
- Leak-proof: hard/keyword drops for outcome/future/path-dependent fields
- Chronological splits (train/test, and train_fit / train_cal)
- Optional isotonic/sigmoid calibration on *later* train slice (time-aware)
- XGBoost early-stopping:
    * try sklearn wrapper
    * else fall back to xgboost.train with DMatrix
    * else no early stopping
- Fully numeric matrices (no object dtypes), NaN/Inf safe
- Saves: model+calibrator, scored test, feature importance, EV curves, thresholds
"""

import argparse
import json
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Optional imports (guarded)
try:
    from xgboost import XGBClassifier
    import xgboost as xgb  # for DMatrix / train fallback
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    xgb = None

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

def chronological_2split(df: pd.DataFrame, frac_fit=0.85):
    """Split a TRAIN dataframe into fit and calibration subsets chronologically."""
    n = len(df)
    nfit = int(n * frac_fit)
    return df.iloc[:nfit].copy(), df.iloc[nfit:].copy()

def _coerce_numeric_inplace(df: pd.DataFrame):
    """Coerce all columns to numeric (bool->int8, non-numeric -> numeric), sanitize NaN/Inf."""
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(np.int8)
        elif not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

def build_feature_matrix(feats: pd.DataFrame):
    """
    Keep everything numeric except prices/ids and leak-like columns.
    """
    hard_drop = {
        # prices / identifiers / labels
        "open","high","low","close",
        "day_open","day_high","day_low","day_close",
        "prev_day_open","prev_day_high","prev_day_low","prev_day_close",
        "entry_price","price_event","tp","sl","horizon_s",
        "label","side","side_enc",
        "datetime","datetime_event","sec_dt","entry_time","features_key_dt",
        "Unnamed: 0",
    }
    leakage_keywords = [
        # outcome / future / forward path info
        "hit","t_hit","time_to","bars_to","future","fwd","ahead",
        "outcome","target","pnl","realized",
        "mfe","mae","rtn_fwd","ret_fwd","fut_","_fut","forward",
    ]

    keep = []
    dropped = []
    for c in feats.columns:
        cl = c.lower()
        if c in hard_drop or c.startswith("Unnamed"):
            continue
        if any(k in cl for k in leakage_keywords):
            dropped.append(c); continue
        keep.append(c)

    if dropped:
        print(f"Leak guard: dropped {len(dropped)} suspicious columns "
              f"(showing up to 8): {sorted(dropped)[:8]}{'...' if len(dropped)>8 else ''}")

    X = feats[keep].copy()
    _coerce_numeric_inplace(X)  # TRAIN numeric
    return X, keep

def expected_value_from_probs(p: np.ndarray, tp: float, sl: float, fee: float) -> np.ndarray:
    # EV per trade at probability p:
    # EV = p*TP - (1-p)*SL - fee
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
    """Ensure we have entry_time and features_key_dt; derive if missing."""
    have = set(C.columns)
    C = _coerce_datetimes(C, ["datetime_event", "sec_dt", "entry_time", "features_key_dt", "datetime"])
    if "entry_time" not in have:
        if "datetime_event" in have:
            C["entry_time"] = C["datetime_event"] + pd.Timedelta(seconds=1)
            print("⚠️  candidates: 'entry_time' missing → using datetime_event+1s as fallback.")
        elif "datetime" in have:
            C["entry_time"] = C["datetime"] + pd.Timedelta(seconds=1)
            print("⚠️  candidates: 'entry_time' & 'datetime_event' missing → using 'datetime'+1s as fallback.")
        else:
            raise ValueError("No 'entry_time' or time column to derive it from.")
    if "features_key_dt" not in have:
        C["features_key_dt"] = C["entry_time"]
        print("ℹ️  candidates: 'features_key_dt' missing → set to 'entry_time'.")
    return C

def _parse_thr_grid(spec: str):
    try:
        s, e, step = [float(x) for x in spec.split(":")]
        grid = np.arange(s, e + 1e-9, step)
        grid = grid[(grid >= 0.0) & (grid <= 1.0)]
        return grid if len(grid) else np.linspace(0.05, 0.95, 19)
    except Exception:
        return np.linspace(0.05, 0.95, 19)


# ----- Adapter to use xgboost.core.Booster like a sklearn estimator -----

class XGBBoosterAdapter:
    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names = list(feature_names)
        # Try to expose a sklearn-like attr for convenience
        try:
            score = booster.get_score(importance_type="gain")
            self.feature_importances_ = np.array([float(score.get(name, 0.0)) for name in self.feature_names])
        except Exception:
            self.feature_importances_ = None

    def predict_proba(self, X):
        D = xgb.DMatrix(X.values, feature_names=self.feature_names)
        p = self.booster.predict(D)
        # ensure 2-column proba
        return np.column_stack([1.0 - p, p])


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features_with_regimes*.csv (1s features)")
    ap.add_argument("--candidates", default="candidates_lr.csv", help="LR-cross candidates file")
    ap.add_argument("--model", choices=["rf", "xgb", "cat"], default="xgb")

    # Splits
    ap.add_argument("--train_frac", type=float, default=0.70, help="Chronological train fraction")
    ap.add_argument("--cal_frac", type=float, default=0.15, help="Fraction of TRAIN kept for calibration (rest for fitting)")

    # RF
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
    ap.add_argument("--early_stopping_rounds", type=int, default=100)

    # CatBoost
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_lr", type=float, default=0.05)
    ap.add_argument("--cat_l2", type=float, default=3.0)

    # Calibration
    ap.add_argument("--calibrate", choices=["none","isotonic","sigmoid"], default="isotonic")

    ap.add_argument("--seed", type=int, default=42)

    # EV threshold search
    ap.add_argument("--tp", type=float, default=25.0)
    ap.add_argument("--sl", type=float, default=15.0)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--group_threshold_by", default="regime")  # if present in data
    ap.add_argument("--thr_grid", default="0.05:0.95:0.05")

    args = ap.parse_args()
    outdir = Path(".")

    # -------- Load candidates (safe) --------
    print(f"Loading candidates: {args.candidates}")
    C_raw = pd.read_csv(args.candidates)
    if C_raw.empty:
        raise SystemExit("No candidates found. Generate with entry_candidates_lr.py first.")
    C_raw = _ensure_entry_keys(C_raw)
    C_raw = C_raw.sort_values("entry_time").reset_index(drop=True)

    # Whitelist minimal metadata to avoid any accidental leakage
    keep_from_C = [c for c in ["label", "side", "entry_time", "features_key_dt", "datetime_event", "sec_dt"] if c in C_raw.columns]
    C = C_raw[keep_from_C].copy()

    # -------- Load features --------
    print(f"Loading features: {args.features}")
    F = pd.read_csv(args.features, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    F_join = F.set_index("datetime")

    # Join features at exact second (fallback asof-backward)
    try:
        feats_at_entry = F_join.loc[C["features_key_dt"].values].reset_index().rename(columns={"datetime":"features_key_dt"})
    except KeyError:
        print("Exact match failed for some rows; using merge_asof backward join.")
        tmpF = F.sort_values("datetime")
        tmpC = C[["features_key_dt"]].rename(columns={"features_key_dt":"datetime"}).sort_values("datetime")
        feats_at_entry = pd.merge_asof(tmpC, tmpF, on="datetime", direction="backward").rename(columns={"datetime":"features_key_dt"})

    # Avoid duplicate key column; combine
    feats_at_entry = feats_at_entry.drop(columns=["features_key_dt"], errors="ignore")
    D = pd.concat([C.reset_index(drop=True), feats_at_entry.reset_index(drop=True)], axis=1)

    # Encode side
    if "side" in D.columns:
        D["side_enc"] = np.where(D["side"].astype(str)=="long", 1, -1).astype(np.int8)
    else:
        D["side_enc"] = 1

    # Labels check
    if "label" not in D.columns:
        raise SystemExit("Candidates file has no 'label' column. Re-run entry_candidates_lr.py to label events.")

    # -------- Chronological splits --------
    Dtr, Dte = chronological_split(D, args.train_frac)
    Dfit, Dcal = chronological_2split(Dtr, 1.0 - args.cal_frac)  # e.g. 85% fit, 15% cal inside TRAIN

    # Feature matrices
    Xfit, cols = build_feature_matrix(Dfit)
    yfit = Dfit["label"].astype(int).values

    Xcal = Dcal[cols].copy(); _coerce_numeric_inplace(Xcal)
    ycal = Dcal["label"].astype(int).values

    Xte  = Dte[cols].copy();  _coerce_numeric_inplace(Xte)
    yte  = Dte["label"].astype(int).values

    # Final leak sanity
    leak_terms = ["hit","t_hit","time_to","bars_to","future","fwd","ahead","outcome","target","pnl","realized","mfe","mae","rtn_fwd","ret_fwd","fut_","_fut","forward"]
    leaked = [c for c in cols if any(k in c.lower() for k in leak_terms)]
    if leaked:
        raise RuntimeError(f"Leak guard failed; suspicious columns in features: {leaked}")

    print(f"Fit size: {Xfit.shape[0]} | Cal size: {Xcal.shape[0]} | Test size: {Xte.shape[0]} | Features: {len(cols)}")

    # -------- Build model --------
    model_name = args.model.lower()
    model_obj = None  # this will be either a sklearn model or an XGBBoosterAdapter

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=(args.max_depth if args.max_depth else None),
            n_jobs=-1,
            random_state=args.seed,
            class_weight=(None if args.class_weight in ["None","none",""] else args.class_weight)
        )
        model_obj = clf

    elif model_name == "xgb":
        if not _HAS_XGB:
            raise SystemExit("XGBoost not installed. pip install xgboost")

        # scale_pos_weight from FIT split
        if args.xgb_scale_pos_weight == "auto":
            pos = max(yfit.sum(), 1)
            neg = len(yfit) - pos
            spw = float(neg / pos)
        else:
            spw = float(args.xgb_scale_pos_weight)

        max_depth = 0 if args.xgb_max_leaves and args.xgb_max_leaves > 0 else args.xgb_depth

        # Try sklearn wrapper first
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
            enable_categorical=False,
        )
        model_obj = clf

    elif model_name == "cat":
        if not _HAS_CAT:
            raise SystemExit("CatBoost not installed. pip install catboost")
        clf = CatBoostClassifier(
            iterations=args.n_estimators,
            depth=args.cat_depth if args.cat_depth else args.max_depth if args.max_depth else 8,
            learning_rate=args.cat_lr,
            l2_leaf_reg=args.cat_l2,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False
        )
        model_obj = clf
    else:
        raise ValueError("Unknown model.")

    # -------- Train (with robust early stopping for XGB) --------
    print(f"Training {model_name.upper()} …")

    if model_name != "xgb":
        model_obj.fit(Xfit, yfit)

    else:
        used_fallback_train = False
        # 1) Try sklearn wrapper with early_stopping_rounds
        tried = False
        if args.early_stopping_rounds > 0 and len(Xcal) > 0:
            tried = True
            try:
                model_obj.fit(
                    Xfit, yfit,
                    eval_set=[(Xcal, ycal)],
                    early_stopping_rounds=args.early_stopping_rounds,
                    verbose=False
                )
            except TypeError:
                # 2) Try xgboost.train with DMatrix
                try:
                    if xgb is None:
                        raise RuntimeError("xgboost core module unavailable.")
                    dtrain = xgb.DMatrix(Xfit.values, label=yfit, feature_names=cols)
                    dvalid = xgb.DMatrix(Xcal.values, label=ycal, feature_names=cols)
                    params = {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "eta": args.xgb_lr,
                        # depth/leaves: pick whichever user chose
                        **({"max_depth": args.xgb_depth} if (args.xgb_max_leaves==0 or args.xgb_max_leaves is None) else {}),
                        **({"max_leaves": args.xgb_max_leaves} if (args.xgb_max_leaves and args.xgb_max_leaves>0) else {}),
                        "subsample": args.xgb_subsample,
                        "colsample_bytree": args.xgb_colsample,
                        "lambda": 1.0,
                        "alpha": 0.0,
                        "random_state": args.seed,
                        "tree_method": "hist",
                        "scale_pos_weight": spw,
                    }
                    booster = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=args.n_estimators,
                        evals=[(dvalid, "valid")],
                        early_stopping_rounds=args.early_stopping_rounds,
                        verbose_eval=False
                    )
                    model_obj = XGBBoosterAdapter(booster, cols)
                    used_fallback_train = True
                except Exception as e:
                    print(f"Early stopping via xgboost.train failed ({e}); training without early stopping.")
                    model_obj = clf  # back to sklearn wrapper
                    model_obj.fit(Xfit, yfit)

        if not tried:
            # No early stopping requested or no cal slice
            model_obj.fit(Xfit, yfit)

    # -------- Optional probability calibration (time-aware) --------
    calibrator = None
    if args.calibrate != "none" and len(Xcal) > 0:
        method = args.calibrate  # "isotonic" or "sigmoid"
        print(f"Calibrating probabilities using {method} on train-cal slice …")
        # sklearn 1.1+: CalibratedClassifierCV(estimator=..., cv="prefit")
        # older: base_estimator=...
        try:
            cal = CalibratedClassifierCV(estimator=model_obj, method=method, cv="prefit")
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=model_obj, method=method, cv="prefit")
        cal.fit(Xcal, ycal)
        calibrator = cal

    # -------- Evaluate --------
    def _proba(est, X):
        return (calibrator if calibrator is not None else est).predict_proba(X)[:, 1]

    proba_fit = _proba(model_obj, Xfit)
    proba_cal = _proba(model_obj, Xcal) if len(Xcal) else np.zeros(len(Xcal))
    proba_te  = _proba(model_obj, Xte)

    def _scores(y, p, tag):
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        bs  = brier_score_loss(y, p)
        print(f"{tag}  AUC: {auc:.3f} | AP: {ap:.3f} | Brier: {bs:.4f}")
        return {"auc": auc, "ap": ap, "brier": bs}

    print("Performance (probabilities may be calibrated):")
    m_fit = _scores(yfit, proba_fit, "FIT ")
    if len(Xcal):
        m_cal = _scores(ycal, proba_cal, "CAL ")
    m_te  = _scores(yte,  proba_te,  "TEST")

    print("\nTEST classification report @ default 0.5 threshold:")
    print(classification_report(yte, (proba_te>=0.5).astype(int), digits=3))

    # -------- Save model (+ calibration if present) --------
    bundle = {
        "model_type": model_name,
        "model": model_obj,          # sklearn model OR XGBBoosterAdapter
        "calibration": (args.calibrate if calibrator is not None else "none"),
        "calibrator": calibrator,
        "features": cols,
        "train_frac": args.train_frac,
        "cal_frac": args.cal_frac,
        "metrics": {"fit": m_fit, "test": m_te},
        "params": vars(args),
    }
    dump(bundle, outdir / "ml_entry_model.joblib")
    print("Saved ml_entry_model.joblib (includes model/adapter, optional calibrator, and feature list).")

    # -------- Feature importance --------
    fi = None
    try:
        if hasattr(model_obj, "feature_importances_") and model_obj.feature_importances_ is not None:
            imp = np.asarray(model_obj.feature_importances_, dtype=float)
            fi = pd.DataFrame({"feature": cols, "importance": imp})
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

    if fi is not None:
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.to_csv(outdir / "ml_entry_feature_importance.csv", index=False)
        print("Wrote ml_entry_feature_importance.csv")
    else:
        print("No native feature_importances_; skipping export (use permutation importance if needed).")

    # -------- Score TEST & save --------
    Dte_out = Dte.copy()
    Dte_out["proba_pred"] = proba_te
    Dte_out.to_csv(outdir / "ml_entry_candidates_scored.csv", index=False)
    print("Wrote ml_entry_candidates_scored.csv (TEST split).")

    # -------- EV-optimal thresholds --------
    grid = _parse_thr_grid(args.thr_grid)
    best_global, table_global = search_threshold(proba_te, yte, tp=args.tp, sl=args.sl, fee=args.fee, grid=grid)
    table_global.to_csv(outdir / "ml_entry_thresholds_global_curve.csv", index=False)
    print("Wrote ml_entry_thresholds_global_curve.csv")

    thr_rows = [{"group_by": "GLOBAL", **best_global}]

    # Per-group thresholds (e.g., per HMM 'regime')
    gcol = args.group_threshold_by
    if gcol and gcol in Dte_out.columns:
        best_by, table_by = search_threshold_by_group(Dte_out.assign(proba_pred=proba_te), group_col=gcol, tp=args.tp, sl=args.sl, fee=args.fee, grid=grid)
        table_by.to_csv(outdir / f"ml_entry_thresholds_by_{gcol}_curve.csv", index=False)
        print(f"Wrote ml_entry_thresholds_by_{gcol}_curve.csv")
        for g, dct in best_by.items():
            thr_rows.append({"group_by": f"{gcol}={g}", **dct})
    else:
        if gcol:
            print(f"Group column '{gcol}' not found in TEST; skipping per-group thresholds.")

    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(outdir / "ml_entry_thresholds.csv", index=False)
    print("Wrote ml_entry_thresholds.csv (EV-optimal).")

    # -------- Print EV summary + break-even reference --------
    be = (args.sl + args.fee) / (args.tp + args.sl)
    print(f"\nBreak-even success probability ≈ {be:.3f} (given TP={args.tp}, SL={args.sl}, fee={args.fee}).")
    print("\nEV-optimal thresholds (summary):")
    print(thr_df.to_string(index=False))

if __name__ == "__main__":
    main()
