#!/usr/bin/env python3
# train_exit_policy.py (robust datetime join)
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---------------- helpers ----------------

def chronological_split(df, frac_train=0.7):
    n = len(df); ntr = int(n*frac_train)
    return df.iloc[:ntr].copy(), df.iloc[ntr:].copy()

def coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(np.int8)
        elif not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_X(df):
    hard_drop = {
        "open","high","low","close",
        "day_open","day_high","day_low","day_close",
        "prev_day_open","prev_day_high","prev_day_low","prev_day_close",
        "entry_price","price_event","tp","sl","horizon_s",
        "label","side","side_enc",
        "datetime","datetime_event","sec_dt","entry_time","features_key_dt",
        "Unnamed: 0",
        # outcome-ish
        "mfe","mae","mfe_pts","mae_pts","t_mfe_s","t_mae_s","t_hit_s"
    }
    leak_kw = ["future","fwd","ahead","hit","mfe","mae","pnl","realized",
               "time_to","ret_fwd","rtn_fwd","_fut","fut_"]
    cols = []
    for c in df.columns:
        if c in hard_drop or c.startswith("Unnamed"): 
            continue
        if any(k in c.lower() for k in leak_kw): 
            continue
        cols.append(c)
    X = coerce_numeric(df[cols].copy())
    return X, cols

def first_touch_win(row, tp, sl):
    mfe = float(row["mfe_pts"]); mae = float(row["mae_pts"])
    tp_reached = (mfe >= tp); sl_reached = (mae >= sl)
    t_mfe = row.get("t_mfe_s", None); t_mae = row.get("t_mae_s", None)
    if tp_reached and sl_reached and (t_mfe is not None) and (t_mae is not None):
        if np.isfinite(t_mfe) and np.isfinite(t_mae):
            if t_mfe < t_mae: return 1
            if t_mae < t_mfe: return 0
            return -1
    if tp_reached and not sl_reached: return 1
    if sl_reached and not tp_reached: return 0
    if tp_reached and sl_reached:
        rt = tp/(mfe+1e-9); rs = sl/(mae+1e-9)
        return 1 if rt < rs else 0
    return -1

def best_exit_class(row, tps, sls, fee):
    best_ev, best_label = -1e18, "no_trade"
    for tp in tps:
        for sl in sls:
            r = first_touch_win(row, tp, sl)
            if r == 1: ev = tp - fee
            elif r == 0: ev = -sl - fee
            else: ev = 0.0
            if ev > best_ev:
                best_ev = ev
                best_label = f"tp{int(tp)}_sl{int(sl)}"
    if best_ev <= 0.0:
        return "no_trade", best_ev
    return best_label, best_ev

def _to_dt64ns_naive(s):
    """
    Normalize any datetime-like series to timezone-naive datetime64[ns].
    Handles object, string, datetime64[ns, tz], etc.
    """
    s = pd.to_datetime(s, errors="coerce", utc=True)
    # now tz-aware; drop tz to make naive ns
    s = s.dt.tz_convert(None)
    return s.astype("datetime64[ns]")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Candidates with MFE/MAE (+ times if available)")
    ap.add_argument("--features", required=True, help="1s features with 'datetime'")
    ap.add_argument("--merge_key", default="entry_time", help="Join key with features (via features_key_dt)")
    ap.add_argument("--tps", default="20,25,30,35,40,50")
    ap.add_argument("--sls", default="8,10,12,15")
    ap.add_argument("--fee", type=float, default=0.5)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--model", choices=["xgb","rf"], default="xgb")
    args = ap.parse_args()

    # --- load candidates
    C = pd.read_csv(args.candidates)
    mfe_col = next((c for c in ["mfe_points","mfe_pts","mfe","MFE"] if c in C.columns), None)
    mae_col = next((c for c in ["mae_points","mae_pts","mae","MAE"] if c in C.columns), None)
    if mfe_col is None or mae_col is None:
        raise SystemExit("Need MFE/MAE columns in candidates (e.g., 'mfe_points','mae_points').")

    C = C.copy()
    C["mfe_pts"] = np.abs(pd.to_numeric(C[mfe_col], errors="coerce"))
    C["mae_pts"] = np.abs(pd.to_numeric(C[mae_col], errors="coerce"))
    if "t_mfe_s" not in C.columns:
        for k in ["t_mfe","time_to_mfe","t_hit_mfe_s"]:
            if k in C.columns: C["t_mfe_s"] = pd.to_numeric(C[k], errors="coerce"); break
    if "t_mae_s" not in C.columns:
        for k in ["t_mae","time_to_mae","t_hit_mae_s"]:
            if k in C.columns: C["t_mae_s"] = pd.to_numeric(C[k], errors="coerce"); break

    # ensure we have features_key_dt
    if "features_key_dt" not in C.columns:
        if args.merge_key in C.columns:
            C["features_key_dt"] = C[args.merge_key]
        else:
            raise SystemExit("Need features_key_dt or a merge_key column in candidates.")

    # --- load features
    F = pd.read_csv(args.features)
    if "datetime" not in F.columns:
        raise SystemExit("Features file must contain a 'datetime' column.")

    # --- HARD NORMALIZE BOTH KEYS TO NAIVE datetime64[ns] AND INT64 NS
    C["features_key_dt"] = _to_dt64ns_naive(C["features_key_dt"])
    F["datetime"]       = _to_dt64ns_naive(F["datetime"])

    # drop rows where join key is NaT (merge_asof requires non-null, sorted)
    C = C[C["features_key_dt"].notna()].copy()
    F = F[F["datetime"].notna()].copy()

    # build monotonic int64 ns join key
    C["_ts_ns"] = C["features_key_dt"].view("int64")
    F["_ts_ns"] = F["datetime"].view("int64")

    C.sort_values("_ts_ns", inplace=True)
    F.sort_values("_ts_ns", inplace=True)

    # --- asof merge on int64 key (bulletproof vs dtype mismatches)
    feats = pd.merge_asof(
        C,
        F,
        on="_ts_ns",
        direction="backward",
        allow_exact_matches=True
    )
    # keep the original timestamp columns for reference
    # (drop helper key if you prefer)
    # feats = feats.drop(columns=["_ts_ns"])

    # --- build labels: best (TP,SL)
    tps = [float(x) for x in args.tps.split(",") if x]
    sls = [float(x) for x in args.sls.split(",") if x]
    labels, best_evs = zip(*[best_exit_class(row, tps, sls, args.fee) for _,row in feats.iterrows()])
    feats["exit_label"] = labels
    feats["exit_label_ev"] = best_evs

    # --- chronological split
    TR, TE = chronological_split(feats, args.train_frac)

    # --- features/targets
    Xtr, cols = build_X(TR); ytr = TR["exit_label"].astype(str).values
    Xte = coerce_numeric(TE[cols].copy()); yte = TE["exit_label"].astype(str).values

    # --- model
    if args.model == "xgb":
        if not _HAS_XGB:
            raise SystemExit("Install xgboost or use --model rf")
        clf = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=600, max_depth=14, n_jobs=-1, class_weight="balanced_subsample",
            random_state=42
        )
    clf.fit(Xtr, ytr)

    # --- evaluate realized EV following predicted exit on TEST
    yhat = clf.predict(Xte)
    TE = TE.copy(); TE["pred_exit"] = yhat

    def payoff(row, fee):
        lab = row["pred_exit"]
        if lab == "no_trade": return 0.0
        try:
            tp = float(lab.split("_")[0].replace("tp",""))
            sl = float(lab.split("_")[1].replace("sl",""))
        except Exception:
            return 0.0
        r = first_touch_win(row, tp, sl)
        if r == 1: return tp - fee
        if r == 0: return -sl - fee
        return 0.0

    TE["realized_payoff"] = [payoff(row, args.fee) for _,row in TE.iterrows()]

    # --- report
    print("\nMulti-class exit policy report (TEST):")
    print(classification_report(yte, yhat, digits=3))
    print(f"TEST total EV (following predicted exits): {TE['realized_payoff'].sum():.2f}  "
          f"| trades: {len(TE)}  | avg/trade: {TE['realized_payoff'].mean():.3f}")

    # --- save
    dump({"model": clf, "features": cols, "classes": np.unique(ytr)}, "exit_policy_model.joblib")
    TE[["features_key_dt","pred_exit","realized_payoff"]].to_csv("exit_policy_eval.csv", index=False)
    print("✅ Saved exit_policy_model.joblib and exit_policy_eval.csv")

if __name__ == "__main__":
    main()
