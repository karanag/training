#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe (no lookahead).

Colab-friendly defaults (just run `!python label.py`):
    features  = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"     # "close" also supported
    tp        = 30.0       # points
    sl        = 15.0       # points
    horizon_s = 600        # seconds

Output:
    candidates_lr.csv with columns:
        datetime_event, side, price_event,
        entry_time, entry_price,
        label (1 TP-first, 0 SL-first/none),
        hit ("tp" / "sl" / "none"),
        t_hit_s (seconds to first hit, NaN if none),
        mfe, mae,
        features_key_dt (same as entry_time)
"""
import os
import numpy as np
import pandas as pd

# ---------------- Resample helpers ----------------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open = ("open", "first"),
        high = ("high", "max"),
        low  = ("low",  "min"),
        close= ("close","last")
    ).dropna()
    m.index.name = "datetime"
    return m

def rolling_lr_midline_close(df_1m: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling linear regression y = a + b*x on CLOSE over 'window' minutes.
    Returns midline at the last x in each window (aligned to the minute index).
    """
    y = df_1m["close"].to_numpy(dtype=np.float64)
    n = len(y)
    x = np.arange(n, dtype=np.float64)

    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    for i in range(window - 1, n):
        xs = x[i - window + 1:i + 1] - x[i - window + 1]
        ys = y[i - window + 1:i + 1]
        b, a = np.polyfit(xs, ys, 1)  # returns slope b, intercept a
        slope[i] = b
        intercept[i] = a

    mid = intercept + slope * (window - 1)

    out = df_1m.copy()
    out["lr_mid"] = mid
    out["lr_slope"] = slope
    return out

def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    """
    Detect cross of price vs (previous-minute) LR midline.

    Expect df_lr to have columns:
      open, high, low, close, lr_mid, lr_slope
    where lr_mid is already the previous-minute LR midline (i.e., shifted).

    'wick' logic (recommended):
      long  if (prev low <= lr_mid) and (curr high > lr_mid)
      short if (prev high >= lr_mid) and (curr low  < lr_mid)

    'close' logic:
      long  if (prev close <= lr_mid) and (curr close > lr_mid)
      short if (prev close >= lr_mid) and (curr close < lr_mid)
    """
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
        "datetime": df_lr.index,   # explicit column (not index)
        "close": df_lr["close"].to_numpy(),
        "high":  df_lr["high"].to_numpy(),
        "low":   df_lr["low"].to_numpy(),
        "lr_mid":   df_lr["lr_mid"].to_numpy(),
        "lr_slope": df_lr["lr_slope"].to_numpy(),
        "long_sig":  long_sig.to_numpy().astype(bool),
        "short_sig": short_sig.to_numpy().astype(bool),
    })
    # Ensure there's no index named 'datetime' to avoid sort ambiguity
    out.index = pd.RangeIndex(len(out))
    return out

# ---------------- Labeling ----------------
def make_labels_1s(
    df_1s: pd.DataFrame,
    events: pd.DataFrame,
    tp: float,
    sl: float,
    horizon_s: int
) -> pd.DataFrame:
    """
    For each minute event (datetime), align to the last second inside that minute,
    enter at the NEXT second's open, simulate for 'horizon_s' seconds.

    Label = 1 if TP is hit before SL, otherwise 0.
    Also returns which hit, time-to-hit in seconds, MFE/MAE (absolute P&L in points).
    """
    s = df_1s.copy().sort_values("datetime").reset_index(drop=True)

    # Map each minute to the last second in that minute
    s["minute_dt"] = s["datetime"].dt.floor("min")
    last_second_per_min = (
        s.groupby("minute_dt")
         .tail(1)[["minute_dt", "datetime"]]
         .drop_duplicates("minute_dt")
         .rename(columns={"datetime": "sec_dt"})
    )

    ev = events.copy()
    ev["minute_dt"] = pd.to_datetime(ev["datetime"])
    ev = ev.merge(last_second_per_min, on="minute_dt", how="left")

    # For speed, index the 1s frame
    s = s.set_index("datetime")

    rows = []
    for _, e in ev.iterrows():
        if pd.isna(e.get("sec_dt")):
            continue

        minute_end = e["sec_dt"]

        # Entry at the FIRST tick strictly after minute_end
        after = s.loc[s.index > minute_end]
        if after.empty:
            continue
        entry_time = after.index[0]
        entry_px   = float(after.iloc[0]["open"])

        # Forward path
        end_time = entry_time + pd.Timedelta(seconds=horizon_s)
        fwd = s.loc[(s.index > minute_end) & (s.index <= end_time)]
        if fwd.empty:
            continue

        # Thresholds
        if e["side"] == "long":
            tp_px = entry_px + tp
            sl_px = entry_px - sl

            tp_hits = fwd.index[fwd["high"] >= tp_px]
            sl_hits = fwd.index[fwd["low"]  <= sl_px]
        else:  # short
            tp_px = entry_px - tp
            sl_px = entry_px + sl

            tp_hits = fwd.index[fwd["low"]  <= tp_px]
            sl_hits = fwd.index[fwd["high"] >= sl_px]

        # Which hit first?
        first_tp = tp_hits[0] if len(tp_hits) else None
        first_sl = sl_hits[0] if len(sl_hits) else None

        if first_tp is not None and (first_sl is None or first_tp <= first_sl):
            label = 1
            hit = "tp"
            t_hit_s = int((first_tp - entry_time).total_seconds())
        elif first_sl is not None:
            label = 0
            hit = "sl"
            t_hit_s = int((first_sl - entry_time).total_seconds())
        else:
            label = 0
            hit = "none"
            t_hit_s = np.nan

        # MFE/MAE in absolute points (not %)
        if e["side"] == "long":
            mfe = float(fwd["high"].max() - entry_px)
            mae = float(fwd["low"].min()  - entry_px)
        else:
            mfe = float(entry_px - fwd["low"].min())
            mae = float(entry_px - fwd["high"].max())

        rows.append({
            "datetime_event": e["datetime"],
            "side": e["side"],
            "price_event": float(e["close"]),
            "entry_time": entry_time,
            "entry_price": entry_px,
            "tp": tp, "sl": sl, "horizon_s": horizon_s,
            "label": int(label),
            "hit": hit,
            "t_hit_s": t_hit_s,
            "mfe": mfe,
            "mae": mae
        })

    return pd.DataFrame(rows)

# ---------------- Main ----------------
def main():
    # -------- Colab-friendly defaults --------
    features  = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"   # or "close"
    tp        = 30.0
    sl        = 15.0
    horizon_s = 600

    print(f"Loading {features} …")
    if not os.path.exists(features):
        raise FileNotFoundError(f"Features file not found at: {features}")

    df = pd.read_csv(features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Ensure required columns exist
    needed = {"datetime", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features CSV: {sorted(missing)}")

    # 1) 1-minute OHLC
    m = resample_1m(df)

    # 2) Rolling LR on 1m close (no lookahead)
    lr = rolling_lr_midline_close(m, window=lr_window)
    # Use previous-minute LR only
    lr["lr_mid_prev"]   = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    lr = lr.dropna(subset=["lr_mid_prev", "lr_slope_prev"])

    # 3) Merge OHLC with previous-minute LR (ensures perfect alignment)
    merged = m.join(lr[["lr_mid_prev", "lr_slope_prev"]], how="inner")
    merged = merged.rename(columns={"lr_mid_prev": "lr_mid", "lr_slope_prev": "lr_slope"})
    # from here on, merged has: open, high, low, close, lr_mid, lr_slope

    # 4) Detect crosses (minute-level)
    crosses = detect_crosses(merged, cross_on=cross_on)

    # Build events with sides
    ev_long = crosses.loc[crosses["long_sig"], ["datetime", "close", "lr_mid", "lr_slope"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses.loc[crosses["short_sig"], ["datetime", "close", "lr_mid", "lr_slope"]].copy()
    ev_short["side"] = "short"

    events = pd.concat([ev_long, ev_short], ignore_index=True)
    events = events.sort_values("datetime").reset_index(drop=True)

    print(f"Found {len(events)} raw LR-cross events.")

    if events.empty:
        print("No events found. Nothing to label.")
        return

    # 5) Label using forward 1s path (enter next second after event minute)
    lab = make_labels_1s(
        df[["datetime", "open", "high", "low", "close"]],
        events.rename(columns={"datetime": "datetime"}),  # explicit
        tp, sl, horizon_s
    )

    if lab.empty:
        print("No labelable events (insufficient forward data).")
        return

    # 6) Write out
    out = lab.copy()
    out["features_key_dt"] = out["entry_time"]  # convenient join key at entry time
    out = out.sort_values("entry_time").reset_index(drop=True)
    out.to_csv("candidates_lr.csv", index=False)

    print(f"✅ Wrote candidates_lr.csv with {len(out)} rows.")
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
