#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Gaussian HMM for Intraday Regimes (overnight-safe, 70:30 split)
---------------------------------------------------------------------
- Builds HMM inputs *within each day* (no cross-day rolling/returns).
- Uses 60s returns as primary, plus 300s realized vol & slope context by default.
- Burns the first max(window) seconds of each day for TRAINING to avoid open distortion.
- Finds state count K via BIC on train (grid states_min..states_max).
- Saves:
    - hmm_model_intra.joblib
    - hmm_scaler_intra.joblib
    - regimes_hmm_intra.csv (datetime, regime, regime_p{0..K-1})
    - features_with_regimes_intra.csv (your features.csv + regime probs)
- Prints TEST split regime summary.

Install:
    pip install hmmlearn scikit-learn joblib
"""

import argparse
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import dump

# ---------------- Config helpers ----------------
def chronological_split(df: pd.DataFrame, frac_train: float = 0.7):
    n = len(df)
    n_train = int(n * frac_train)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()

def bic_gaussian_hmm(model: GaussianHMM, X: np.ndarray) -> float:
    ll = model.score(X)
    n, d = X.shape
    k = model.n_components
    # Parameter count for full-cov HMM (common approximation)
    p = (k - 1) + k * (k - 1) + k * d + k * (d * (d + 1) // 2)
    return -2.0 * ll + p * np.log(max(n, 1))

def pick_hmm_state_count(X_train: np.ndarray, state_grid=(2,3,4,5,6), random_state=42):
    best_bic, best_model, best_k = np.inf, None, None
    for k in state_grid:
        m = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=200,
            tol=1e-3,
            random_state=random_state,
            verbose=False,
        )
        m.fit(X_train)
        bic = bic_gaussian_hmm(m, X_train)
        print(f"K={k} BIC={bic:.2f}")
        if bic < best_bic:
            best_bic, best_model, best_k = bic, m, k
    print(f"Selected K={best_k} (min BIC={best_bic:.2f})")
    return best_model

# ---------------- Intraday-safe feature builders ----------------
def _groupby_day(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["datetime"]).dt.normalize()

def make_intra_returns(df: pd.DataFrame, windows_s=(60,)):
    """Within-day percentage returns over specified second windows."""
    out = pd.DataFrame(index=df.index)
    day = _groupby_day(df)
    for w in windows_s:
        out[f"ret_{w}s_intra"] = df.groupby(day)["close"].pct_change(w)
    return out

def make_intra_realized_vol(df: pd.DataFrame, windows_s=(60, 300)):
    """Within-day realized volatility (std of 1s returns) over windows."""
    out = pd.DataFrame(index=df.index)
    day = _groupby_day(df)
    ret_1s = df.groupby(day)["close"].pct_change().fillna(0.0)
    for w in windows_s:
        out[f"rv_{w}s_intra"] = ret_1s.groupby(day).rolling(w, min_periods=w).std().reset_index(level=0, drop=True)
    return out

def make_intra_sma_slopes(df: pd.DataFrame, windows_s=(60, 300)):
    """Within-day SMA slopes over second windows (1s sampling)."""
    out = pd.DataFrame(index=df.index)
    day = _groupby_day(df)
    for w in windows_s:
        sma = df.groupby(day)["close"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        out[f"sma_{w}s_intra"] = sma
        out[f"sma_{w}s_slope_intra"] = sma.diff()
    return out

def time_features_from_features_csv(df: pd.DataFrame):
    """Pick time-of-day and simple regime cues already in features.csv."""
    cols = []
    for c in ["tod_sin","tod_cos","dow_sin","dow_cos","adx","rsi_14","chop_14"]:
        if c in df.columns:
            cols.append(c)
    return df[cols].copy()

# ---------------- TEST summary ----------------
def summarize_states(df: pd.DataFrame, states: np.ndarray, name: str):
    tmp = df.copy()
    tmp["state"] = states
    # Forward 60s return for EVAL ONLY (not fed to model)
    tmp["fwd_ret_60s"] = tmp["close"].pct_change(-60)
    agg = tmp.groupby("state").agg(
        rows=("state","size"),
        fwd60s_mean=("fwd_ret_60s","mean"),
        fwd60s_std =("fwd_ret_60s","std"),
    )
    print(f"\nState summary ({name}):")
    print(agg.sort_index())
    return agg

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv", help="Path to features CSV with datetime/open/high/low/close + engineered fields")
    ap.add_argument("--states_min", type=int, default=2)
    ap.add_argument("--states_max", type=int, default=6)
    ap.add_argument("--ret_windows", default="60", help="Comma sep seconds for returns, e.g. '60,120'")
    ap.add_argument("--vol_windows", default="60,300", help="Comma sep seconds for realized vol windows")
    ap.add_argument("--sma_windows", default="60,300", help="Comma sep seconds for SMA slope windows")
    ap.add_argument("--burnin_seconds", type=int, default=300, help="Drop first N seconds of each day from TRAIN only")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--pca_components", type=int, default=0, help="0 = no PCA")
    args = ap.parse_args()

    ret_w = tuple(int(x) for x in args.ret_windows.split(",") if x.strip())
    vol_w = tuple(int(x) for x in args.vol_windows.split(",") if x.strip())
    sma_w = tuple(int(x) for x in args.sma_windows.split(",") if x.strip())
    max_w = max(ret_w + vol_w + sma_w + (args.burnin_seconds,))

    print(f"Loading {args.features} …")
    df = pd.read_csv(args.features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["datetime","open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from features CSV.")

    print("Building intraday-safe HMM inputs …")
    X1 = make_intra_returns(df, ret_w)
    X2 = make_intra_realized_vol(df, vol_w)
    X3 = make_intra_sma_slopes(df, sma_w)
    X4 = time_features_from_features_csv(df)

    # Concatenate and clean
    hmm_df = pd.concat([df[["datetime","close"]], X1, X2, X3, X4], axis=1)
    hmm_cols = [c for c in hmm_df.columns if c not in ("datetime","close")]
    hmm_df[hmm_cols] = hmm_df[hmm_cols].replace([np.inf,-np.inf], np.nan)

    # Chronological split (70/30) BEFORE any per-day drops so indexing stays simple
    train_df, test_df = chronological_split(hmm_df, 0.70)

    # Burn-in: drop first max_w seconds of each day from TRAIN only
    def day_pos(s: pd.Series) -> pd.Series:
        d = pd.to_datetime(s).dt.normalize()
        return s.groupby(d).cumcount()

    train_pos = day_pos(train_df["datetime"])
    keep_train = train_pos >= max_w
    train_df = train_df.loc[keep_train].copy()

    # Fill NaNs after burn-in
    train_df[hmm_cols] = train_df[hmm_cols].fillna(0.0).astype(np.float32)
    test_df[hmm_cols]  = test_df[hmm_cols].fillna(0.0).astype(np.float32)

    # Prepare arrays
    X_train = train_df[hmm_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[hmm_cols].to_numpy(dtype=np.float32)

    # Preprocess (scaler + optional PCA) fit on train only
    steps = [("scaler", StandardScaler())]
    if args.pca_components and args.pca_components > 0:
        steps.append(("pca", PCA(n_components=args.pca_components, random_state=args.random_state)))
    preproc = Pipeline(steps)
    X_train_t = preproc.fit_transform(X_train)
    X_test_t  = preproc.transform(X_test)

    # Pick K via BIC
    k_grid = tuple(range(args.states_min, args.states_max + 1))
    best_model = pick_hmm_state_count(X_train_t, state_grid=k_grid, random_state=args.random_state)

    # Decode on FULL set (train+test with train-fitted model)
    X_full_t = preproc.transform(hmm_df[hmm_cols].to_numpy(dtype=np.float32))
    logprob, post = best_model.score_samples(X_full_t)  # (N, K)
    states = best_model.predict(X_full_t)
    K = best_model.n_components
    print(f"\nFitted HMM with K={K}")

    # TEST summary
    test_states = states[len(train_df.index) + (len(hmm_df) - len(hmm_df)) : ]  # placeholder, we’ll recompute robustly
    # Better: slice by original indices
    test_idx = test_df.index
    test_states = pd.Series(states, index=hmm_df.index).loc[test_idx].to_numpy()
    summarize_states(df.loc[test_idx], test_states, "TEST")

    # Save artifacts
    dump(best_model, "hmm_model_intra.joblib")
    dump(preproc,    "hmm_scaler_intra.joblib")
    print("Saved hmm_model_intra.joblib & hmm_scaler_intra.joblib")

    # Write regimes CSV
    regimes = pd.DataFrame({"datetime": hmm_df["datetime"].values, "regime": states.astype(np.int16)})
    for i in range(K):
        regimes[f"regime_p{i}"] = post[:, i].astype(np.float32)
    regimes.to_csv("regimes_hmm_intra.csv", index=False)
    print("Wrote regimes_hmm_intra.csv")

    # Merge back into original features file for downstream models
    merged = df.merge(regimes, on="datetime", how="left")
    merged.to_csv("features_with_regimes_intra.csv", index=False)
    print("Wrote features_with_regimes_intra.csv")

    # Simple ranking by forward returns (TEST only)
    tmp = df.loc[test_idx].copy()
    tmp["state"] = test_states
    tmp["fwd_ret_60s"] = tmp["close"].pct_change(-60)
    srank = tmp.groupby("state")["fwd_ret_60s"].mean().sort_values(ascending=False)
    print("\nState ranking by mean forward 60s return (TEST):")
    print(srank.to_string())

if __name__ == "__main__":
    main()


'''
python train_hmm_regimes_intra.py \
  --features features.csv \
  --states_min 2 --states_max 6 \
  --ret_windows 60 \
  --vol_windows 60,300 \
  --sma_windows 60,300 \
  --burnin_seconds 300 \
  --random_state 42 \
  --pca_components 0

'''