#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze HMM Regimes
-------------------
- Loads features_with_regimes_intra.csv
- Groups by regime to check:
    * Rows per regime
    * Mean/Std forward returns (1m, 5m, 15m)
    * Realized volatility
- Saves summary CSV + prints quick stats
"""

import pandas as pd

def main():
    print("Loading features_with_regimes_intra.csv …")
    df = pd.read_csv("features_with_regimes_intra.csv", parse_dates=["datetime"])
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

    if "regime" not in df.columns:
        raise ValueError("No 'regime' column found. Did you merge regimes?")

    # Forward returns (evaluation only, not for training!)
    for horizon in [60, 300, 900]:
        df[f"fwd_ret_{horizon}s"] = df["close"].pct_change(-horizon)

    # Realized vol (rolling std of 1s returns)
    df["ret_1s"] = df["close"].pct_change().fillna(0)
    df["rv_60s"] = df["ret_1s"].rolling(60).std()
    df["rv_300s"] = df["ret_1s"].rolling(300).std()

    # Group by regime
    agg = df.groupby("regime").agg(
        rows=("regime","size"),
        fwd1m_mean=("fwd_ret_60s","mean"),
        fwd1m_std =("fwd_ret_60s","std"),
        fwd5m_mean=("fwd_ret_300s","mean"),
        fwd15m_mean=("fwd_ret_900s","mean"),
        rv_60s_mean=("rv_60s","mean"),
        rv_300s_mean=("rv_300s","mean"),
    ).sort_index()

    print("\n=== Regime Summary ===")
    print(agg)

    # Save for deeper analysis
    agg.to_csv("regime_summary.csv")
    print("✅ Saved regime_summary.csv")

if __name__ == "__main__":
    main()
