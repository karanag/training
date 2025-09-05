#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe (no lookahead), with CHUNKED labeling for Colab (low RAM).

Colab-friendly defaults (just run `!python label.py`):
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
        label, hit, t_hit_s, mfe, mae,
        features_key_dt
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
    """Rolling linear regression midline on CLOSE (uses only past window)."""
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
    """Detect LR crosses (wick or close logic)."""
    lr = df_lr["lr_mid"]
    if cross_on == "wick":
        prev_below = (df_lr["low"].shift(1)  <= lr)
        curr_above = (df_lr["high"]          > lr)
        prev_above = (df_lr["high"].shift(1) >= lr)
        curr_below = (df_lr["low"]           < lr)
    else:
        prev_below = (df_lr["close"].shift(1) <= lr)
        curr_above = (df_lr["close"]          > lr)
        prev_above = (df_lr["close"].shift(1) >= lr)
        curr_below = (df_lr["close"]          < lr)

    long_sig  = (prev_below & curr_above).fillna(False)
    short_sig = (prev_above & curr_below).fillna(False)

    out = pd.DataFrame({
        "datetime": df_lr.index,
        "close": df_lr["close"].to_numpy(),
        "high":  df_lr["high"].to_numpy(),
        "low":   df_lr["low"].to_numpy(),
        "lr_mid":   df_lr["lr_mid"].to_numpy(),
        "lr_slope": df_lr["lr_slope"].to_numpy(),
        "long_sig":  long_sig.to_numpy(),
        "short_sig": short_sig.to_numpy(),
    })
    out.index = pd.RangeIndex(len(out))
    return out

# ---------------- Chunked labeling ----------------
def make_labels_chunked(df_1s, events, tp, sl, horizon_s, chunk_size=20_000):
    """
    Chunked version of labeling to save RAM.
    """
    s = df_1s.copy().sort_values("datetime").reset_index(drop=True)
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

    s = s.set_index("datetime")

    results = []
    for start in range(0, len(ev), chunk_size):
        batch = ev.iloc[start:start+chunk_size]
        rows = []
        for _, e in batch.iterrows():
            if pd.isna(e.get("sec_dt")):
                continue
            minute_end = e["sec_dt"]
            after = s.loc[s.index > minute_end]
            if after.empty:
                continue
            entry_time = after.index[0]
            entry_px   = float(after.iloc[0]["open"])
            end_time = entry_time + pd.Timedelta(seconds=horizon_s)
            fwd = s.loc[(s.index > minute_end) & (s.index <= end_time)]
            if fwd.empty:
                continue

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

            first_tp = tp_hits[0] if len(tp_hits) else None
            first_sl = sl_hits[0] if len(sl_hits) else None

            if first_tp is not None and (first_sl is None or first_tp <= first_sl):
                label = 1; hit = "tp"; t_hit_s = int((first_tp - entry_time).total_seconds())
            elif first_sl is not None:
                label = 0; hit = "sl"; t_hit_s = int((first_sl - entry_time).total_seconds())
            else:
                label = 0; hit = "none"; t_hit_s = np.nan

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
        results.append(pd.DataFrame(rows))
        print(f"Processed {start+len(batch)} / {len(ev)} events")

    return pd.concat(results, ignore_index=True)

# ---------------- Main ----------------
def main():
    features  = "/content/drive/MyDrive/data/features.csv"
    lr_window = 7
    cross_on  = "wick"
    tp        = 30.0
    sl        = 15.0
    horizon_s = 600

    print(f"Loading {features} …")
    if not os.path.exists(features):
        raise FileNotFoundError(f"Features file not found at: {features}")

    df = pd.read_csv(features, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    needed = {"datetime", "open", "high", "low", "close"}
    if missing := (needed - set(df.columns)):
        raise ValueError(f"Missing columns in features CSV: {sorted(missing)}")

    m = resample_1m(df)
    lr = rolling_lr_midline_close(m, window=lr_window)
    lr["lr_mid_prev"]   = lr["lr_mid"].shift(1)
    lr["lr_slope_prev"] = lr["lr_slope"].shift(1)
    lr = lr.dropna(subset=["lr_mid_prev", "lr_slope_prev"])
    merged = m.join(lr[["lr_mid_prev", "lr_slope_prev"]], how="inner")
    merged = merged.rename(columns={"lr_mid_prev":"lr_mid", "lr_slope_prev":"lr_slope"})

    crosses = detect_crosses(merged, cross_on=cross_on)
    ev_long = crosses.loc[crosses["long_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses.loc[crosses["short_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short], ignore_index=True).sort_values("datetime").reset_index(drop=True)

    print(f"Found {len(events)} raw LR-cross events.")
    if events.empty: return

    lab = make_labels_chunked(df[["datetime","open","high","low","close"]], events, tp, sl, horizon_s)
    if lab.empty:
        print("No labelable events."); return

    lab["features_key_dt"] = lab["entry_time"]
    lab = lab.sort_values("entry_time").reset_index(drop=True)
    lab.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(lab)} rows.")
    print(lab.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
