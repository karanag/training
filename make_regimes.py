#!/usr/bin/env python3
"""
Make Regime Features (HMM soft states)
--------------------------------------
- Loads features.csv (with 1s + 1m engineered features)
- Trains Gaussian HMM on 1m features
- Produces per-bar regime probabilities (soft states)
- Broadcasts back to 1s data (delayed by 1m to avoid lookahead)
- Saves features_regimes.csv
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# ------------------ Config ------------------
N_STATES = 6   # number of regimes (tweak 4–7)
SEED = 42


# ------------------ Main ------------------
if __name__ == "__main__":
    print("Loading features.csv…")
    df = pd.read_csv("features.csv", parse_dates=["datetime"])
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

    # 1m resample for HMM training
    df_1m = df.resample("1min", on="datetime").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "atr": "last", "adx": "last",
        "roc_1": "last", "roc_5": "last", "roc_10": "last",
        "twap": "last", "dist_twap_sd1": "last", "dist_twap_sd2": "last", "dist_twap_sd3": "last",
        "dist_prev_day_high": "last", "dist_prev_day_low": "last",
        "dist_day_high": "last", "dist_day_low": "last",
        "dist_pivot": "last",
    }).dropna()

    print(f"HMM training rows: {len(df_1m)}")

    # Features for HMM training
    hmm_features = df_1m[[
        "atr", "adx", "roc_1", "roc_5", "roc_10",
        "twap", "dist_twap_sd1", "dist_twap_sd2", "dist_twap_sd3",
        "dist_prev_day_high", "dist_prev_day_low",
        "dist_day_high", "dist_day_low",
        "dist_pivot"
    ]].to_numpy(dtype=np.float64)

    # Fit Gaussian HMM
    print(f"Training GaussianHMM with {N_STATES} states…")
    hmm = GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=100, random_state=SEED)
    hmm.fit(hmm_features)

    # Predict probabilities
    probs = hmm.predict_proba(hmm_features)
    regime_cols = [f"regime_p{i+1}" for i in range(N_STATES)]
    df_probs = pd.DataFrame(probs, columns=regime_cols, index=df_1m.index)

    # Shift by 1 bar (only known after bar closes)
    df_probs = df_probs.shift(1)

    # Merge back to 1s data
    print("Broadcasting regime probabilities to 1s level…")
    df_out = pd.merge_asof(
        df.sort_values("datetime"),
        df_probs.reset_index().rename(columns={"datetime": "datetime"}),
        on="datetime",
        direction="backward"
    )

    # Fill NaN with 0 until first bar is available
    df_out[regime_cols] = df_out[regime_cols].fillna(0.0)

    print(f"Final output rows: {len(df_out)}, cols: {len(df_out.columns)}")
    df_out.to_csv("features_regimes.csv", index=False)
    print("✅ Saved to features_regimes.csv")
