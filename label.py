#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry candidate generator via Linear Regression (LR) midline cross on 1-minute bars.
Leak-safe (no lookahead). Fast & RAM-safe for Colab.

- Reads only datetime/open/high/low/close (float32).
- Auto-prefers a local copy (/content/features.csv) to avoid Drive disconnects.
- Rolling LR midline via vectorized convolution (no slow per-minute polyfit loop).
- Previous-minute LR is used for signals (shift by 1m).
- Entry = first second AFTER the signal minute.
- Labels via chunked, Numba-accelerated search for first TP/SL hit inside horizon.

Defaults (change with env vars if you like):
  FEATURES_PATH    = /content/features.csv (falls back to /content/drive/MyDrive/data/features.csv)
  LR_WINDOW        = 7          (minutes)
  CROSS_ON         = "wick"     ("close" also supported)
  TP_POINTS        = 30.0
  SL_POINTS        = 15.0
  HORIZON_SECONDS  = 600
  LABEL_CHUNK_SIZE = 20000

Output: candidates_lr.csv
Columns:
  datetime_event, side, price_event, entry_time, entry_price,
  label, hit, t_hit_s, mfe, mae, features_key_dt
"""

import os
import numpy as np
import pandas as pd

# ---------- Config (env overrides) ----------
FEATURES_PATH   = os.getenv("FEATURES_PATH", "/content/features.csv")
FALLBACK_PATH   = "/content/drive/MyDrive/data/features.csv"
LR_WINDOW       = int(os.getenv("LR_WINDOW", "7"))
CROSS_ON        = os.getenv("CROSS_ON", "wick")  # "wick" or "close"
TP_POINTS       = float(os.getenv("TP_POINTS", "30.0"))
SL_POINTS       = float(os.getenv("SL_POINTS", "15.0"))
HORIZON_SECONDS = int(os.getenv("HORIZON_SECONDS", "600"))
CHUNK_SIZE      = int(os.getenv("LABEL_CHUNK_SIZE", "20000"))

# ---------- Fast rolling LR midline (vectorized) ----------
def rolling_lr_midline_close_vec(close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling OLS y = a + b*x on equidistant x=0..W-1 for CLOSE.
    Returns (midline_at_last_x, slope) aligned to the original minute index (NaN for first W-1).
    """
    y = close.astype(np.float64, copy=False)
    n = y.shape[0]
    W = window
    if n < W:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Precompute constants
    sum_x  = W * (W - 1) / 2.0
    sum_x2 = (W - 1) * W * (2 * W - 1) / 6.0
    denom  = W * sum_x2 - sum_x * sum_x

    # Rolling sums via convolution
    ones = np.ones(W, dtype=np.float64)
    weights = np.arange(W, dtype=np.float64)  # 0..W-1

    sum_y  = np.convolve(y, ones, mode="valid")            # length n-W+1
    sum_xy = np.convolve(y, weights, mode="valid")

    # Slope and intercept for the valid positions
    b_valid = (W * sum_xy - sum_x * sum_y) / denom
    a_valid = (sum_y - b_valid * sum_x) / W

    # Midline at x = W-1 (the last bar in the window)
    mid_valid = a_valid + b_valid * (W - 1)

    # Pad to length n
    mid = np.full(n, np.nan, dtype=np.float64)
    slp = np.full(n, np.nan, dtype=np.float64)
    mid[W-1:] = mid_valid
    slp[W-1:] = b_valid
    return mid, slp

# ---------- 1m resample ----------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open = ("open","first"),
        high = ("high","max"),
        low  = ("low","min"),
        close= ("close","last"),
    ).dropna()
    m.index.name = "datetime"
    return m

# ---------- Detect crosses (previous-minute LR already provided) ----------
def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    lr = df_lr["lr_mid"]
    if cross_on == "wick":
        prev_below = (df_lr["low"].shift(1)  <= lr)
        curr_above = (df_lr["high"]          >  lr)
        prev_above = (df_lr["high"].shift(1) >= lr)
        curr_below = (df_lr["low"]           <  lr)
    else:
        prev_below = (df_lr["close"].shift(1) <= lr)
        curr_above = (df_lr["close"]          >  lr)
        prev_above = (df_lr["close"].shift(1) >= lr)
        curr_below = (df_lr["close"]          <  lr)

    long_sig  = (prev_below & curr_above).fillna(False)
    short_sig = (prev_above & curr_below).fillna(False)

    out = pd.DataFrame({
        "datetime": df_lr.index,
        "close": df_lr["close"].to_numpy(),
        "lr_mid": df_lr["lr_mid"].to_numpy(),
        "lr_slope": df_lr["lr_slope"].to_numpy(),
        "long_sig": long_sig.to_numpy(),
        "short_sig": short_sig.to_numpy(),
    })
    out.index = pd.RangeIndex(len(out))
    return out

# ---------- Numba-accelerated labeling ----------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

@njit(cache=True)
def _label_batch_numba(
    ts_ns: np.ndarray,     # int64 ns, ascending, 1s sampling
    o: np.ndarray, h: np.ndarray, l: np.ndarray,
    sec_idx: np.ndarray,   # index of last second of signal minute
    side: np.ndarray,      # +1 long, -1 short
    price_event: np.ndarray,
    tp: float, sl: float, horizon: int
):
    n_events = sec_idx.shape[0]
    N = ts_ns.shape[0]
    label = np.zeros(n_events, np.int8)
    hit = np.zeros(n_events, np.int8)  # 1=tp, 2=sl, 0=none
    t_hit = np.empty(n_events, np.float64)
    t_hit.fill(np.nan)
    entry_idx = np.full(n_events, -1, np.int64)
    entry_price = np.empty(n_events, np.float64)

    mfe = np.zeros(n_events, np.float64)
    mae = np.zeros(n_events, np.float64)

    for i in range(n_events):
        si = sec_idx[i]
        if si < 0 or si >= N-1:
            continue
        eidx = si + 1
        entry_idx[i] = eidx
        ep = o[eidx]
        entry_price[i] = ep

        end_idx = eidx + horizon
        if end_idx >= N:
            end_idx = N - 1
        if end_idx <= eidx:
            continue

        first_tp = -1
        first_sl = -1

        if side[i] == 1:  # long
            tp_px = ep + tp
            sl_px = ep - sl
            # scan forward
            for t in range(eidx, end_idx + 1):
                if h[t] >= tp_px:
                    first_tp = t
                    break
                if l[t] <= sl_px:
                    first_sl = t
                    break
            # MFE/MAE
            maxh = h[eidx:end_idx+1].max()
            minl = l[eidx:end_idx+1].min()
            mfe[i] = maxh - ep
            mae[i] = minl - ep
        else:  # short
            tp_px = ep - tp
            sl_px = ep + sl
            for t in range(eidx, end_idx + 1):
                if l[t] <= tp_px:
                    first_tp = t
                    break
                if h[t] >= sl_px:
                    first_sl = t
                    break
            maxh = h[eidx:end_idx+1].max()
            minl = l[eidx:end_idx+1].min()
            mfe[i] = ep - minl
            mae[i] = ep - maxh

        if first_tp != -1 and (first_sl == -1 or first_tp <= first_sl):
            label[i] = 1
            hit[i] = 1
            t_hit[i] = float(first_tp - eidx)
        elif first_sl != -1:
            label[i] = 0
            hit[i] = 2
            t_hit[i] = float(first_sl - eidx)
        else:
            label[i] = 0
            hit[i] = 0
            t_hit[i] = np.nan

    return entry_idx, entry_price, label, hit, t_hit, mfe, mae

# ---------- Utilities to map minute -> last second index ----------
def minute_last_second_index(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ts: datetime64[ns] ascending
    Returns (minute_keys_int64, last_idx_per_minute)
    """
    minute = ts.astype('datetime64[m]').astype('int64')
    # Positions where minute changes; the element BEFORE change is last of prev minute.
    change = minute[1:] != minute[:-1]
    last_idx = np.flatnonzero(change)
    # plus the very last row is last of its minute
    last_idx = np.append(last_idx, len(ts) - 1)
    minute_keys = minute[last_idx]
    return minute_keys, last_idx

# ---------- Main ----------
def main():
    path = FEATURES_PATH if os.path.exists(FEATURES_PATH) else FALLBACK_PATH
    print(f"Loading {path} …")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found at: {path}")

    # Read only what we need, downcast to save RAM
    usecols = ["datetime", "open", "high", "low", "close"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=["datetime"],
        dtype={"open":"float32","high":"float32","low":"float32","close":"float32"},
        engine="c",
        memory_map=True,
    ).sort_values("datetime").reset_index(drop=True)

    # 1) 1-minute OHLC
    m = resample_1m(df)

    # 2) Rolling LR on 1m close (vectorized, no lookahead), then SHIFT by 1 minute
    mid, slp = rolling_lr_midline_close_vec(m["close"].to_numpy(), LR_WINDOW)
    m_lr = m.copy()
    m_lr["lr_mid"]   = pd.Series(mid, index=m.index).shift(1)
    m_lr["lr_slope"] = pd.Series(slp, index=m.index).shift(1)
    m_lr = m_lr.dropna(subset=["lr_mid","lr_slope"])

    # 3) Cross detection on minutes (using previous-minute LR)
    crosses = detect_crosses(m_lr, cross_on=CROSS_ON)

    # Build events
    ev_long = crosses.loc[crosses["long_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_long["side"] = "long"
    ev_short = crosses.loc[crosses["short_sig"], ["datetime","close","lr_mid","lr_slope"]].copy()
    ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    print(f"Found {len(events)} raw LR-cross events.")
    if events.empty:
        print("No events found. Exiting.")
        return

    # 4) Prepare 1s arrays for fast labeling
    s = df.copy()
    ts = s["datetime"].values  # datetime64[ns]
    ts_ns = ts.view("int64")   # for numba; still monotonic
    o = s["open"].to_numpy(np.float64, copy=False)
    h = s["high"].to_numpy(np.float64, copy=False)
    l = s["low"].to_numpy(np.float64, copy=False)

    # Map each minute to the index of its LAST second
    minute_keys, last_idx = minute_last_second_index(ts)
    minute_to_last = pd.Series(last_idx, index=minute_keys)

    # Events minute keys -> last second index
    ev_minute_keys = events["datetime"].values.astype("datetime64[m]").astype("int64")
    # Use pandas align to keep vectorized mapping; non-existing minutes -> NaN
    ev_sec_idx = minute_to_last.reindex(ev_minute_keys).to_numpy()
    # Drop events we cannot map (e.g., partial last minute)
    ok = ~np.isnan(ev_sec_idx)
    events = events.loc[ok].reset_index(drop=True)
    ev_sec_idx = ev_sec_idx[ok].astype(np.int64)

    # Encode side as +1/-1
    side = np.where(events["side"].values == "long", 1, -1).astype(np.int8)
    price_event = events["close"].to_numpy(np.float64, copy=False)

    # 5) Chunked + Numba labeling
    out_path = "candidates_lr.csv"
    if os.path.exists(out_path):
        os.remove(out_path)

    out_cols = [
        "datetime_event","side","price_event","entry_time","entry_price",
        "label","hit","t_hit_s","mfe","mae","features_key_dt"
    ]

    total = len(events)
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        ev_idx = np.arange(start, end)
        sec_idx_batch = ev_sec_idx[ev_idx]
        side_batch = side[ev_idx]
        price_batch = price_event[ev_idx]

        entry_idx, entry_price, label, hit, t_hit, mfe, mae = _label_batch_numba(
            ts_ns, o, h, l,
            sec_idx_batch, side_batch, price_batch,
            TP_POINTS, SL_POINTS, HORIZON_SECONDS
        )

        # Build DataFrame for this batch
        batch = pd.DataFrame({
            "datetime_event": events.loc[ev_idx, "datetime"].values,
            "side": np.where(side_batch == 1, "long", "short"),
            "price_event": price_batch.astype(np.float32),
            "entry_time": ts[entry_idx.clip(min=0)],
            "entry_price": entry_price.astype(np.float32),
            "label": label.astype(np.int8),
            "hit": pd.Series(hit).map({0:"none",1:"tp",2:"sl"}).values,
            "t_hit_s": t_hit.astype(np.float32),
            "mfe": mfe.astype(np.float32),
            "mae": mae.astype(np.float32),
        })
        batch["features_key_dt"] = batch["entry_time"]

        # Append to disk
        batch.to_csv(out_path, index=False, mode=("a" if start else "w"), header=(start == 0), columns=out_cols)
        print(f"Processed {end}/{total} events")

    print(f"✅ Wrote {out_path}")
    # Show a peek
    peek = pd.read_csv(out_path, nrows=5, parse_dates=["datetime_event","entry_time","features_key_dt"])
    print(peek.to_string(index=False))

if __name__ == "__main__":
    main()
