#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe (no lookahead).
Hardcoded for Colab run:
    features = /content/drive/MyDrive/data/features.csv
    lr_window = 7
    cross_on = "wick"
    tp = 30
    sl = 15
    horizon_s = 600
"""

import numpy as np
import pandas as pd

# ---------------- Resample helpers ----------------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open=("open","first"),
        high=("high","max"),
        low =("low","min"),
        close=("close","last")
    ).dropna()
    m.index.name = "datetime"
    return m

def rolling_lr_midline_close(df_1m: pd.DataFrame, window: int) -> pd.DataFrame:
    y = df_1m["close"].to_numpy(dtype=np.float64)
    n = len(y)
    x = np.arange(n, dtype=np.float64)

    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    for i in range(window-1, n):
        xs = x[i-window+1:i+1] - x[i-window+1]
        ys = y[i-window+1:i+1]
        b, a = np.polyfit(xs, ys, 1)
        slope[i] = b
        intercept[i] = a

    mid = intercept + slope * (window - 1)
    out = df_1m.copy()
    out["lr_mid"] = mid
    out["lr_slope"] = slope
    return out

def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    df = df_lr.copy()
    lr = df["lr_mid"]
    if cross_on == "wick":
        prev_below = (df["low"].shift(1) <= lr.shift(1))
        curr_above = (df["high"] > lr)
        prev_above = (df["high"].shift(1) >= lr.shift(1))
        curr_below = (df["low"] < lr)
    else:
        prev_below = (df["close"].shift(1) <= lr.shift(1))
        curr_above = (df["close"] > lr)
        prev_above = (df["close"].shift(1) >= lr.shift(1))
        curr_below = (df["close"] < lr)

    long_sig  = prev_below & curr_above
    short_sig = prev_above & curr_below

    return pd.DataFrame({
        "datetime": df.index,
        "close": df["close"],
        "lr_mid": df["lr_mid"],
        "lr_slope": df["lr_slope"],
        "long_sig": long_sig,
        "short_sig": short_sig
    })

def make_labels_1s(df_1s: pd.DataFrame,
                   events: pd.DataFrame,
                   tp: float,
                   sl: float,
                   horizon_s: int) -> pd.DataFrame:
    s = df_1s.copy().sort_values("datetime").reset_index(drop=True)
    events = events.copy()
    events["minute_dt"] = pd.to_datetime(events["datetime"])
    s["minute_dt"] = s["datetime"].dt.floor("min")

    last_sec_idx = s.groupby("minute_dt").tail(1).set_index("minute_dt").index
    sec_at_min_end = s[s["minute_dt"].isin(last_sec_idx)][["minute_dt","datetime"]].drop_duplicates("minute_dt")
    events = events.merge(sec_at_min_end, on="minute_dt", how="left")
    events.rename(columns={"datetime_y":"sec_dt"}, inplace=True)

    s = s.set_index("datetime")
    rows = []
    for _, ev in events.iterrows():
        if pd.isna(ev.get("sec_dt")):
            continue
        sec_dt = ev["sec_dt"]
        try:
            entry_row = s.loc[s.index > sec_dt].iloc[0]
        except IndexError:
            continue
        entry_time = entry_row.name
        entry_px   = float(entry_row["open"])

        end_time = entry_time + pd.Timedelta(seconds=horizon_s)
        fwd = s.loc[(s.index > sec_dt) & (s.index <= end_time)]
        if fwd.empty:
            continue

        mfe_long  = fwd["high"].max() - entry_px
        mae_long  = fwd["low"].min()  - entry_px
        mfe_short = entry_px - fwd["low"].min()
        mae_short = entry_px - fwd["high"].max()

        side = ev["side"]
        if side == "long":
            label = int(mfe_long >= tp and mae_long > -sl)
            mfe, mae = float(mfe_long), float(mae_long)
        else:
            label = int(mfe_short >= tp and mae_short > -sl)
            mfe, mae = float(mfe_short), float(mae_short)

        rows.append({
            "datetime_event": ev["datetime"],
            "sec_dt": sec_dt,
            "side": side,
            "price_event": float(ev["close"]),
            "entry_time": entry_time,
            "entry_price": entry_px,
            "tp": tp, "sl": sl, "horizon_s": horizon_s,
            "label": int(label),
            "mfe": mfe, "mae": mae
        })

    return pd.DataFrame(rows)

# ---------------- Main (hardcoded args) ----------------
def main():
    features = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on = "wick"
    tp = 30.0
    sl = 15.0
    horizon_s = 600

    print(f"Loading {features} …")
    df = pd.read_csv(features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    m = resample_1m(df)
    lr = rolling_lr_midline_close(m, window=lr_window)
    lr["lr_mid_prev"] = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    lr = lr.dropna(subset=["lr_mid_prev","lr_slope_prev"])

    # Merge OHLC back so detect_crosses has 'high' and 'low'
    lr = lr.merge(m[["open","high","low","close"]], left_index=True, right_index=True)

    crosses = detect_crosses(
        lr.rename(columns={"lr_mid_prev":"lr_mid","lr_slope_prev":"lr_slope"}),
        cross_on=cross_on
    )


    ev_long = crosses[crosses["long_sig"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses[crosses["short_sig"]].copy()
    ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short]).sort_values("datetime")

    print(f"Found {len(events)} raw LR-cross events.")

    lab = make_labels_1s(df[["datetime","open","high","low","close"]], events, tp, sl, horizon_s)
    if lab.empty:
        print("No labelable events.")
        return

    out = lab.copy()
    out["features_key_dt"] = out["entry_time"]
    out.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
