#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leak-safe LR-cross labeler + context features (prev-minute line, slope, overshoot).
Writes candidates_lr.csv with extra columns:
  lr_mid_prev, lr_slope_prev, min_close, min_high, min_low,
  overshoot, cross_type
"""

import os, numpy as np, pandas as pd

# ---------------- Config (change if needed) ----------------
FEATURES  = "features.csv"
LR_WINDOW = 7
CROSS_ON  = "wick"    # or "close"
TP, SL    = 30.0, 15.0
HORIZON_S = 600

# ---------------- Helpers ----------------
def resample_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    m = df_1s.resample("1min", on="datetime").agg(
        open = ("open","first"),
        high = ("high","max"),
        low  = ("low", "min"),
        close= ("close","last")
    ).dropna()
    m.index.name = "datetime"
    return m

def rolling_lr_midline_close(df_1m: pd.DataFrame, window: int) -> pd.DataFrame:
    y = df_1m["close"].to_numpy(np.float64)
    n = len(y); x = np.arange(n, dtype=np.float64)
    slope = np.full(n, np.nan); intercept = np.full(n, np.nan)
    for i in range(window-1, n):
        xs = x[i-window+1:i+1] - x[i-window+1]
        ys = y[i-window+1:i+1]
        b, a = np.polyfit(xs, ys, 1)    # y = a + b*x
        slope[i] = b; intercept[i] = a
    mid = intercept + slope*(window-1)
    out = df_1m.copy()
    out["lr_mid"] = mid
    out["lr_slope"] = slope
    return out

def detect_crosses(df_lr: pd.DataFrame, cross_on: str = "close") -> pd.DataFrame:
    """df_lr has: open, high, low, close, lr_mid, lr_slope  (prev-minute values)."""
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
        "open":  df_lr["open"].to_numpy(),
        "high":  df_lr["high"].to_numpy(),
        "low":   df_lr["low"].to_numpy(),
        "close": df_lr["close"].to_numpy(),
        "lr_mid_prev":   df_lr["lr_mid"].to_numpy(),
        "lr_slope_prev": df_lr["lr_slope"].to_numpy(),
        "long_sig":  long_sig.to_numpy().astype(bool),
        "short_sig": short_sig.to_numpy().astype(bool),
    })
    out.index = pd.RangeIndex(len(out))
    return out

def make_labels_1s(df_1s: pd.DataFrame, events: pd.DataFrame,
                   tp: float, sl: float, horizon_s: int) -> pd.DataFrame:
    s = df_1s.copy().sort_values("datetime").reset_index(drop=True)
    s["minute_dt"] = s["datetime"].dt.floor("min")
    last_sec = (s.groupby("minute_dt")
                  .tail(1)[["minute_dt","datetime"]]
                  .drop_duplicates("minute_dt")
                  .rename(columns={"datetime":"sec_dt"}))
    ev = events.copy()
    ev["minute_dt"] = pd.to_datetime(ev["datetime"])
    ev = ev.merge(last_sec, on="minute_dt", how="left")

    s = s.set_index("datetime")
    rows = []
    for _, e in ev.iterrows():
        sec_dt = e.get("sec_dt")
        if pd.isna(sec_dt): continue
        after = s.loc[s.index > sec_dt]
        if after.empty: continue
        entry_time = after.index[0]
        entry_px   = float(after.iloc[0]["open"])

        fwd = s.loc[(s.index > sec_dt) & (s.index <= entry_time + pd.Timedelta(seconds=horizon_s))]
        if fwd.empty: continue

        if e["side"] == "long":
            tp_px = entry_px + tp; sl_px = entry_px - sl
            tp_hits = fwd.index[fwd["high"] >= tp_px]
            sl_hits = fwd.index[fwd["low"]  <= sl_px]
        else:
            tp_px = entry_px - tp; sl_px = entry_px + sl
            tp_hits = fwd.index[fwd["low"]  <= tp_px]
            sl_hits = fwd.index[fwd["high"] >= sl_px]

        first_tp = tp_hits[0] if len(tp_hits) else None
        first_sl = sl_hits[0] if len(sl_hits) else None

        if first_tp is not None and (first_sl is None or first_tp <= first_sl):
            label, hit, t_hit = 1, "tp", int((first_tp - entry_time).total_seconds())
        elif first_sl is not None:
            label, hit, t_hit = 0, "sl", int((first_sl - entry_time).total_seconds())
        else:
            label, hit, t_hit = 0, "none", np.nan

        # MFE/MAE (absolute points)
        if e["side"] == "long":
            mfe = float(fwd["high"].max() - entry_px)
            mae = float(fwd["low"].min()  - entry_px)
        else:
            mfe = float(entry_px - fwd["low"].min())
            mae = float(entry_px - fwd["high"].max())

        # Extra context (safe: from minute that triggered the event)
        overshoot = float(e["close"] - e["lr_mid_prev"])  # sign matters
        rows.append({
            "datetime_event": e["datetime"],
            "side": e["side"],
            "price_event": float(e["close"]),
            "entry_time": entry_time,
            "entry_price": entry_px,
            "tp": tp, "sl": sl, "horizon_s": horizon_s,
            "label": int(label), "hit": hit, "t_hit_s": t_hit,
            "mfe": mfe, "mae": mae,
            "lr_mid_prev": float(e["lr_mid_prev"]),
            "lr_slope_prev": float(e["lr_slope_prev"]),
            "min_close": float(e["close"]),
            "min_high": float(e["high"]),
            "min_low":  float(e["low"]),
            "overshoot": overshoot,
            "cross_type": CROSS_ON,
            "features_key_dt": entry_time
        })

    return pd.DataFrame(rows)

def main():
    print(f"Loading {FEATURES} …")
    if not os.path.exists(FEATURES):
        raise FileNotFoundError(FEATURES)
    df = pd.read_csv(FEATURES, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df.columns: raise ValueError(f"Missing {c} in features.csv")

    m  = resample_1m(df)
    lr = rolling_lr_midline_close(m, window=LR_WINDOW)
    lr = lr.assign(lr_mid_prev=lr["lr_mid"].shift(1),
                   lr_slope_prev=lr["lr_slope"].shift(1)).dropna(subset=["lr_mid_prev","lr_slope_prev"])
    merged = m.join(lr[["lr_mid_prev","lr_slope_prev"]], how="inner")

    crosses = detect_crosses(merged.rename(columns={"lr_mid_prev":"lr_mid","lr_slope_prev":"lr_slope"}), cross_on=CROSS_ON)
    ev_long  = crosses.loc[crosses["long_sig"]].copy();  ev_long["side"]  = "long"
    ev_short = crosses.loc[crosses["short_sig"]].copy(); ev_short["side"] = "short"
    events = pd.concat([ev_long, ev_short], ignore_index=True)
    events = events.sort_values("datetime").reset_index(drop=True)

    print(f"Found {len(events)} raw LR-cross events.")
    if events.empty:
        print("No events found."); return

    lab = make_labels_1s(df[["datetime","open","high","low","close"]], events, TP, SL, HORIZON_S)
    if lab.empty:
        print("No labelable events."); return

    lab.to_csv("candidates_lr.csv", index=False)
    print(f"✅ Wrote candidates_lr.csv with {len(lab)} rows.")
    print(lab.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
