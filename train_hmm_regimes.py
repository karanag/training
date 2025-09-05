import pandas as pd

def main():
    df = pd.read_csv("features_with_regimes_intra.csv", parse_dates=["datetime"])
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

    if "regime" not in df.columns:
        raise ValueError("No 'regime' column found.")

    # --- Chronological split: 70% train, 30% test ---
    n = len(df)
    n_train = int(n * 0.7)
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    # Forward returns (evaluation only, not for training!)
    for horizon in [60, 300, 900]:
        df_test[f"fwd_ret_{horizon}s"] = df_test["close"].pct_change(-horizon)

    # Realized vol
    df_test["ret_1s"] = df_test["close"].pct_change().fillna(0)
    df_test["rv_60s"] = df_test["ret_1s"].rolling(60).std()
    df_test["rv_300s"] = df_test["ret_1s"].rolling(300).std()

    # Group by regime (OOS only)
    agg = df_test.groupby("regime").agg(
        rows=("regime","size"),
        fwd1m_mean=("fwd_ret_60s","mean"),
        fwd1m_std =("fwd_ret_60s","std"),
        fwd5m_mean=("fwd_ret_300s","mean"),
        fwd15m_mean=("fwd_ret_900s","mean"),
        rv_60s_mean=("rv_60s","mean"),
        rv_300s_mean=("rv_300s","mean"),
    ).sort_index()

    print("\n=== OOS Regime Summary ===")
    print(agg)

    agg.to_csv("regime_summary_oos.csv")
    print("✅ Saved regime_summary_oos.csv")

if __name__ == "__main__":
    main()
