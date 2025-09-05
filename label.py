#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe:
- LR is computed on 1m bars with a rolling window; the *current bar's* LR value
  uses only the window bars up to *previous minute* (we enter on next second).
- Labels (TP/SL) use forward 1s path starting from the *next second* after the event.

Outputs:
- candidates_lr.csv : one row per candidate with:
    datetime_event, side (long/short), price_event, entry_price, label, mfe, mae, t_hit, features_key (join key)

Usage:
    python entry_candidates_lr.py --features features.csv \
        --lr_window 20 --cross_on wick --tp 25 --sl 15 --horizon_s 300
"""

import argparse
import numpy as np
import pandas as pd

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
    """
    Linear regression (y ~ a + b*x) over rolling window on CLOSE.
    Returns midline value aligned to each minute index using only *past* window
    (we will shift by 1 to enforce 'previous minute' LR when generating candidates).
    """
    y = df_1m["close"].to_numpy(dtype=np.float64)
    n = len(y)
    x = np.arange(n, dtype=np.float64)

    # Compute slope & intercept using rolling polyfit (simple & robust)
    # For speed on huge datasets, replace with convolution-based sums.
    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    for i in range(window-1, n):
        xs = x[i-window+1:i+1] - x[i-window+1]  # rebase to 0..W-1 to improve conditioning
        ys = y[i-window+1:i+1]
        b, a = np.polyfit(xs, ys, 1)  # y = a + b*x
        slope[i] = b
        intercept[i] = a

    # Midline at the *last* x in the window (x = window-1 after rebase)
    mid = intercept + slope * (window - 1)
    out = df_1m.copy()
    out["lr_mid"] = mid
    out["lr_slope"] = slope
    return out

def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    """
    cross_on:
        - 'close': previous close below lr AND current close above lr (long), mirror for short
        - 'wick' : previous low below lr AND current high above lr (long), mirror for short
    """
    df = df_lr.copy()
    lr = df["lr_mid"]
    if cross_on == "wick":
        prev_below = (df["low"].shift(1) <= lr.shift(1))
        curr_above = (df["high"] > lr)
        prev_above = (df["high"].shift(1) >= lr.shift(1))
        curr_below = (df["low"] < lr)
    else:  # close
        prev_below = (df["close"].shift(1) <= lr.shift(1))
        curr_above = (df["close"] > lr)
        prev_above = (df["close"].shift(1) >= lr.shift(1))
        curr_below = (df["close"] < lr)

    long_sig  = prev_below & curr_above
    short_sig = prev_above & curr_below

    out = pd.DataFrame({
        "datetime": df.index,
        "close": df["close"],
        "lr_mid": df["lr_mid"],
        "lr_slope": df["lr_slope"],
        "long_sig": long_sig,
        "short_sig": short_sig
    })
    return out

def make_labels_1s(df_1s: pd.DataFrame,
                   events: pd.DataFrame,
                   tp: float,
                   sl: float,
                   horizon_s: int) -> pd.DataFrame:
    """
    For each minute event, align to the last second inside that minute,
    enter at the NEXT second's open (if available), then compute MFE/MAE
    over the next 'horizon_s' seconds. Label=1 if MFE>=tp before MAE<=-sl.
    """
    s = df_1s.copy()
    s = s.sort_values("datetime").reset_index(drop=True)

    # Minute alignment: event_minute_end = minute timestamp (already index)
    events = events.copy()
    events["minute_dt"] = pd.to_datetime(events["datetime"])
    # Align to the last second within that minute
    s["minute_dt"] = s["datetime"].dt.floor("min")
    last_sec_idx = s.groupby("minute_dt").tail(1).set_index("minute_dt").index
    # Map minute end to an actual 1s row index
    sec_at_min_end = s[s["minute_dt"].isin(last_sec_idx)][["minute_dt","datetime"]].drop_duplicates("minute_dt")
    events = events.merge(sec_at_min_end, on="minute_dt", how="left", suffixes=("",""))
    events.rename(columns={"datetime_y":"sec_dt"}, inplace=True)

    # Entry at next second open
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
        entry_px   = float(entry_row["open"])  # enter at next second open

        # Forward slice
        end_time = entry_time + pd.Timedelta(seconds=horizon_s)
        fwd = s.loc[(s.index > sec_dt) & (s.index <= end_time)]
        if fwd.empty:
            continue

        # MFE/MAE for long/short
        mfe_long = (fwd["high"].max() - entry_px)
        mae_long = (fwd["low"].min()  - entry_px)
        mfe_short = (entry_px - fwd["low"].min())
        mae_short = (entry_px - fwd["high"].max())

        side = ev["side"]
        if side == "long":
            # label 1 if TP hit before SL (approx using extremes)
            label = int(mfe_long >= tp and mae_long > -sl)
            mfe = float(mfe_long)
            mae = float(mae_long)
        else:
            label = int(mfe_short >= tp and mae_short > -sl)
            mfe = float(mfe_short)
            mae = float(mae_short)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.csv or features_with_regimes.csv (1s)")
    ap.add_argument("--lr_window", type=int, default=20, help="Rolling LR window (minutes)")
    ap.add_argument("--cross_on", choices=["close","wick"], default="wick", help="Cross detection basis")
    ap.add_argument("--tp", type=float, default=25.0, help="Take-profit in points")
    ap.add_argument("--sl", type=float, default=15.0, help="Stop-loss in points")
    ap.add_argument("--horizon_s", type=int, default=300, help="Forward path horizon in seconds")
    ap.add_argument("--min_slope", type=float, default=0.0, help="Optional slope filter (>= for longs, <= for shorts)")
    args = ap.parse_args()

    print(f"Loading {args.features} …")
    df = pd.read_csv(args.features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["datetime","open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in {args.features}")

    # 1) 1m bars & LR midline (shift by 1 to avoid peeking the forming bar)
    m = resample_1m(df)
    lr = rolling_lr_midline_close(m, window=args.lr_window)
    lr["lr_mid_prev"] = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    lr = lr.dropna(subset=["lr_mid_prev","lr_slope_prev"])

    # 2) Cross events using previous-minute LR
    crosses = detect_crosses(lr.rename(columns={"lr_mid_prev":"lr_mid","lr_slope_prev":"lr_slope"}), cross_on=args.cross_on)

    # Convert to events
    ev_long = crosses[crosses["long_sig"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses[crosses["short_sig"]].copy()
    ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short], axis=0).sort_values("datetime")
    # Optional slope gating
    if args.min_slope != 0.0:
        events = events[
            ((events["side"]=="long") & (events["lr_slope"]>=args.min_slope)) |
            ((events["side"]=="short")& (events["lr_slope"]<=-args.min_slope))
        ]

    print(f"Found {len(events)} raw LR-cross events.")

    # 3) Label on forward 1s path, entry at next second
    lab = make_labels_1s(df[["datetime","open","high","low","close"]], events, args.tp, args.sl, args.horizon_s)
    if lab.empty:
        print("No labelable events (insufficient forward data). Exiting.")
        return

    # Join back any keys you want to preserve
    out = lab.copy()
    # This feature join key will let us merge 1s features at entry_time in the trainer
    out["features_key_dt"] = out["entry_time"]

    out.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
