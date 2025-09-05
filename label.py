#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorized LR-cross candidate labeling (Colab friendly, no lookahead).
- Computes TP/SL labels using precomputed forward 600s high/low windows.
- No Python loops over events -> memory safe, fast.
"""

import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- Resample helpers ----------------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
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
    for i in range(window - 1, n):
        xs = x[i - window + 1:i + 1] - x[i - window + 1]
        ys = y[i - window + 1:i + 1]
        b, a = np.polyfit(xs, ys, 1)
        slope[i] = b
        intercept[i] = a

    mid = intercept + slope * (window - 1)
    out = df_1m.copy()
    out["lr_mid"] = mid
    out["lr_slope"] = slope
    return out

def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    lr = df_lr["lr_mid"]
    if cross_on == "wick":
        prev_below = (df_lr["low"].shift(1)  <= lr)
        curr_above = (df_lr["high"]         >  lr)
        prev_above = (df_lr["high"].shift(1) >= lr)
        curr_below = (df_lr["low"]          <  lr)
    else:
        prev_below = (df_lr["close"].shift(1) <= lr)
        curr_above = (df_lr["close"]         >  lr)
        prev_above = (df_lr["close"].shift(1) >= lr)
        curr_below = (df_lr["close"]         <  lr)

    long_sig  = (prev_below & curr_above).fillna(False)
    short_sig = (prev_above & curr_below).fillna(False)

    out = pd.DataFrame({
        "datetime": df_lr.index,
        "close": df_lr["close"].to_numpy(),
        "high": df_lr["high"].to_numpy(),
        "low": df_lr["low"].to_numpy(),
        "lr_mid": df_lr["lr_mid"].to_numpy(),
        "lr_slope": df_lr["lr_slope"].to_numpy(),
        "long_sig": long_sig.to_numpy(),
        "short_sig": short_sig.to_numpy(),
    })
    out.index = pd.RangeIndex(len(out))
    return out

# ---------------- Vectorized forward-window ----------------
def compute_forward_windows(df_1s: pd.DataFrame, horizon_s: int):
    """
    Precompute forward max(high) and min(low) over horizon_s seconds
    using sliding_window_view. No lookahead leak: strictly forward.
    """
    highs = df_1s["high"].to_numpy()
    lows  = df_1s["low"].to_numpy()

    # Pad with NaN at the end so window view is same length
    pad = np.full(horizon_s, np.nan)
    highs_p = np.concatenate([highs, pad])
    lows_p  = np.concatenate([lows,  pad])

    win_high = sliding_window_view(highs_p, horizon_s)
    win_low  = sliding_window_view(lows_p,  horizon_s)

    fwd_max_high = np.nanmax(win_high, axis=1)
    fwd_min_low  = np.nanmin(win_low, axis=1)

    out = df_1s.copy()
    out["fwd_max_high"] = fwd_max_high
    out["fwd_min_low"]  = fwd_min_low
    return out

# ---------------- Labeling ----------------
def label_events(events: pd.DataFrame, df_fwd: pd.DataFrame, tp: float, sl: float):
    """
    Vectorized labeling: join events on entry_time and use precomputed fwd_max_high/fwd_min_low.
    """
    # Entry at NEXT second open after event minute
    df_fwd = df_fwd.sort_values("datetime").reset_index(drop=True)
    s = df_fwd[["datetime","open","fwd_max_high","fwd_min_low"]].set_index("datetime")

    labeled = []
    for _, ev in events.iterrows():
        # Entry time = first tick strictly after event minute end
        after = s.loc[s.index > ev["datetime"]]
        if after.empty:
            continue
        entry_time = after.index[0]
        row = after.iloc[0]
        entry_px = row["open"]

        fwd_max = row["fwd_max_high"]
        fwd_min = row["fwd_min_low"]

        if ev["side"] == "long":
            tp_px = entry_px + tp
            sl_px = entry_px - sl
            hit_tp = fwd_max >= tp_px
            hit_sl = fwd_min <= sl_px
        else:
            tp_px = entry_px - tp
            sl_px = entry_px + sl
            hit_tp = fwd_min <= tp_px
            hit_sl = fwd_max >= sl_px

        if hit_tp and not hit_sl:
            label, hit = 1, "tp"
        elif hit_sl and not hit_tp:
            label, hit = 0, "sl"
        elif hit_tp and hit_sl:
            # Ambiguity: conservatively mark as 0 (SL-first assumption)
            label, hit = 0, "both"
        else:
            label, hit = 0, "none"

        labeled.append({
            "datetime_event": ev["datetime"],
            "side": ev["side"],
            "entry_time": entry_time,
            "entry_price": entry_px,
            "label": label,
            "hit": hit,
        })

    return pd.DataFrame(labeled)

# ---------------- Main ----------------
def main():
    features = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"
    tp, sl, horizon_s = 30.0, 15.0, 600

    print(f"Loading {features} …")
    df = pd.read_csv(features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1m bars + LR
    m = resample_1m(df)
    lr = rolling_lr_midline_close(m, lr_window)
    lr["lr_mid_prev"] = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    merged = m.join(lr[["lr_mid_prev","lr_slope_prev"]], how="inner")
    merged = merged.rename(columns={"lr_mid_prev":"lr_mid","lr_slope_prev":"lr_slope"})

    # Cross events
    crosses = detect_crosses(merged, cross_on)
    ev_long = crosses[crosses["long_sig"]].assign(side="long")
    ev_short = crosses[crosses["short_sig"]].assign(side="short")
    events = pd.concat([ev_long, ev_short], ignore_index=True).sort_values("datetime")

    print(f"Found {len(events)} raw LR-cross events.")

    # Precompute forward windows on 1s data
    print("Precomputing forward windows …")
    df_fwd = compute_forward_windows(df, horizon_s)

    # Label events
    print("Labeling …")
    lab = label_events(events, df_fwd, tp, sl)

    out = lab.sort_values("entry_time").reset_index(drop=True)
    out.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(out)} rows.")
    print(out.head())

if __name__ == "__main__":
    main()
