#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe (no lookahead), Colab-optimized with Numba.

Defaults (just run `!python label.py` in Colab):
    features  = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"
    tp        = 30.0
    sl        = 15.0
    horizon_s = 600

Output:
    candidates_lr.csv with columns:
        datetime_event, side, price_event,
        entry_time, entry_price,
        label (1 TP-first, 0 SL-first/none),
        hit ("tp" / "sl" / "none"),
        t_hit_s (seconds to first hit, NaN if none),
        mfe, mae,
        features_key_dt
"""
import os
import numpy as np
import pandas as pd
from numba import njit

# ---------------- 1m Resample ----------------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open = ("open", "first"),
        high = ("high", "max"),
        low  = ("low",  "min"),
        close= ("close","last")
    ).dropna()
    m.index.name = "datetime"
    return m

# ---------------- Rolling LR ----------------
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

# ---------------- Cross Detection ----------------
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
        "high":  df_lr["high"].to_numpy(),
        "low":   df_lr["low"].to_numpy(),
        "lr_mid":   df_lr["lr_mid"].to_numpy(),
        "lr_slope": df_lr["lr_slope"].to_numpy(),
        "long_sig":  long_sig.to_numpy().astype(bool),
        "short_sig": short_sig.to_numpy().astype(bool),
    })
    out.index = pd.RangeIndex(len(out))
    return out

# ---------------- Numba Labeling Core ----------------
@njit
def label_events_numba(entry_idx, entry_px, sides, highs, lows, times, tp, sl, horizon_s):
    n = len(entry_idx)
    labels = np.zeros(n, dtype=np.int32)
    hits = np.empty(n, dtype=np.int32)  # 0=none,1=tp,2=sl
    t_hits = np.full(n, np.nan)
    mfes = np.zeros(n)
    maes = np.zeros(n)

    for i in range(n):
        idx = entry_idx[i]
        px = entry_px[i]
        side = sides[i]
        entry_time = times[idx]
        end_time = entry_time + horizon_s * 1_000_000_000

        mfe = -1e9
        mae = 1e9
        hit = 0
        label = 0
        t_hit = np.nan

        j = idx
        while j < len(times) and times[j] <= end_time:
            hi = highs[j]
            lo = lows[j]
            if side == 1:  # long
                mfe = max(mfe, hi - px)
                mae = min(mae, lo - px)
                if hit == 0:
                    if hi >= px + tp:
                        hit, label, t_hit = 1, 1, (times[j] - entry_time)//1_000_000_000
                        break
                    elif lo <= px - sl:
                        hit, label, t_hit = 2, 0, (times[j] - entry_time)//1_000_000_000
                        break
            else:  # short
                mfe = max(mfe, px - lo)
                mae = min(mae, px - hi)
                if hit == 0:
                    if lo <= px - tp:
                        hit, label, t_hit = 1, 1, (times[j] - entry_time)//1_000_000_000
                        break
                    elif hi >= px + sl:
                        hit, label, t_hit = 2, 0, (times[j] - entry_time)//1_000_000_000
                        break
            j += 1

        if hit == 0:
            mfe = max(mfe, 0.0)
            mae = min(mae, 0.0)

        labels[i] = label
        hits[i] = hit
        t_hits[i] = t_hit
        mfes[i] = mfe
        maes[i] = mae

    return labels, hits, t_hits, mfes, maes

# ---------------- Fast Label Wrapper ----------------
def make_labels_1s_fast(df_1s, events, tp, sl, horizon_s):
    s = df_1s.sort_values("datetime").reset_index(drop=True)
    times = s["datetime"].astype("int64").to_numpy()
    highs = s["high"].to_numpy()
    lows  = s["low"].to_numpy()
    opens = s["open"].to_numpy()

    # map events to entry indices
    s["minute_dt"] = s["datetime"].dt.floor("min")
    last_secs = (
        s.groupby("minute_dt").tail(1)[["minute_dt","datetime"]]
        .rename(columns={"datetime":"sec_dt"})
    )
    ev = events.copy()
    ev["minute_dt"] = pd.to_datetime(ev["datetime"])
    ev = ev.merge(last_secs, on="minute_dt", how="left")

    entry_idx, entry_px, sides = [], [], []
    entry_times = []
    for _, e in ev.iterrows():
        if pd.isna(e.get("sec_dt")): continue
        sec_ns = np.int64(e["sec_dt"].value)
        idx = np.searchsorted(times, sec_ns)
        if idx >= len(times)-1: continue
        entry_idx.append(idx+1)
        entry_px.append(opens[idx+1])
        sides.append(1 if e["side"]=="long" else -1)
        entry_times.append(times[idx+1])

    if not entry_idx:
        return pd.DataFrame()

    labels, hits, t_hits, mfes, maes = label_events_numba(
        np.array(entry_idx), np.array(entry_px), np.array(sides),
        highs, lows, times, tp, sl, horizon_s
    )

    hit_map = {0:"none",1:"tp",2:"sl"}
    out = pd.DataFrame({
        "datetime_event": ev["datetime"].values[:len(entry_idx)],
        "side": ev["side"].values[:len(entry_idx)],
        "price_event": ev["close"].values[:len(entry_idx)],
        "entry_time": pd.to_datetime(entry_times),
        "entry_price": entry_px,
        "tp": tp, "sl": sl, "horizon_s": horizon_s,
        "label": labels,
        "hit": [hit_map[h] for h in hits],
        "t_hit_s": t_hits,
        "mfe": mfes,
        "mae": maes,
    })
    out["features_key_dt"] = out["entry_time"]
    return out

# ---------------- Main ----------------
def main():
    features  = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"
    tp, sl, horizon_s = 30.0, 15.0, 600

    print(f"Loading {features} …")
    if not os.path.exists(features):
        raise FileNotFoundError(features)

    df = pd.read_csv(features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    needed = {"datetime","open","high","low","close"}
    if not needed.issubset(df.columns):
        raise ValueError("Missing required OHLC columns")

    m = resample_1m(df)
    lr = rolling_lr_midline_close(m, window=lr_window)
    lr["lr_mid_prev"] = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    lr = lr.dropna()
    merged = m.join(lr[["lr_mid_prev","lr_slope_prev"]], how="inner")
    merged = merged.rename(columns={"lr_mid_prev":"lr_mid","lr_slope_prev":"lr_slope"})

    crosses = detect_crosses(merged, cross_on=cross_on)

    ev_long = crosses.loc[crosses["long_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses.loc[crosses["short_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short], ignore_index=True).sort_values("datetime")

    print(f"Found {len(events)} raw LR-cross events.")
    if events.empty:
        return

    lab = make_labels_1s_fast(df[["datetime","open","high","low","close"]], events, tp, sl, horizon_s)
    if lab.empty:
        print("No labelable events.")
        return

    lab = lab.sort_values("entry_time").reset_index(drop=True)
    lab.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(lab)} rows.")
    print(lab.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
