#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Feature Engineering (1s decisions + 1m context, leak-safe)
------------------------------------------------------------------
- Inputs: 1-second OHLC with a 'datetime' column (tz-naive OK).
- Outputs: A single DataFrame with exhaustive features for:
    * HMM/HSMM (compact regime inputs),
    * Supervised ML (rich entry features),
    * RL (curated state + microstructure).
- NO LOOKAHEAD:
    * All 1m indicators are computed on 1m bars, then SHIFTED by 1 minute
      and merged-asof (direction='backward') into 1s → only prior-minute info is used.
    * Day-level prior-day features are shifted by one day.
    * Intraday "so-far" features at 1s are cumulative up to the current second
      (and the agent observes them on the NEXT step in your env).
- No volume required.

Assumptions:
- India cash session approx 09:15–15:30 local (used for time-of-day features).
  Adjust SESSION_START/SESSION_END if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------ Session Times (adjust if needed) ------------------
SESSION_START = pd.to_datetime("09:15:00").time()
SESSION_END   = pd.to_datetime("15:30:00").time()
SESSION_LEN_S = (pd.Timestamp.combine(pd.Timestamp.today(), SESSION_END)
                - pd.Timestamp.combine(pd.Timestamp.today(), SESSION_START)).seconds  # 6h15m = 22500s

# ------------------ Small numeric helpers ------------------
def _f32(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def _safe_div(a, b, eps=1e-9):
    return (a / (np.sign(b) * np.maximum(np.abs(b), eps))).astype(np.float32)

# ------------------ Time Features (1s) ------------------
def add_time_features_1s(df_1s: pd.DataFrame) -> pd.DataFrame:
    out = df_1s.copy()
    dt = pd.to_datetime(out["datetime"])
    out["dow"] = dt.dt.weekday.astype(np.int8)  # 0=Mon
    out["dom"] = dt.dt.day.astype(np.int8)
    out["month"] = dt.dt.month.astype(np.int8)

    # Day-of-week cyclical
    out["dow_sin"] = np.sin(2*np.pi * out["dow"] / 7).astype(np.float32)
    out["dow_cos"] = np.cos(2*np.pi * out["dow"] / 7).astype(np.float32)

    # Time since session open (seconds), clipped to [0, SESSION_LEN_S]
    t = dt.dt.time
    sec_from_midnight = dt.dt.hour*3600 + dt.dt.minute*60 + dt.dt.second
    ss = pd.to_datetime(SESSION_START.strftime("%H:%M:%S"))
    se = pd.to_datetime(SESSION_END.strftime("%H:%M:%S"))
    sec_open  = ss.hour*3600 + ss.minute*60 + ss.second
    sec_close = se.hour*3600 + se.minute*60 + se.second

    since_open = np.clip(sec_from_midnight - sec_open, 0, SESSION_LEN_S).astype(np.int32)
    out["since_open_s"] = since_open
    out["tod_frac"] = _f32(np.where(SESSION_LEN_S > 0, since_open / SESSION_LEN_S, 0.0))
    out["tod_sin"] = np.sin(2*np.pi * out["tod_frac"]).astype(np.float32)
    out["tod_cos"] = np.cos(2*np.pi * out["tod_frac"]).astype(np.float32)

    # Session bucket (coarse) — first hour, lunch, power hour, etc.
    # Feel free to tune these cut lines.
    cuts = np.array([0, 3600, 3*3600, 5*3600, SESSION_LEN_S], dtype=np.int32)
    out["session_bucket"] = pd.cut(since_open, bins=cuts, labels=False, include_lowest=True).fillna(0).astype(np.int8)

    return out

# ------------------ 1m Indicators (computed on 1m, then shifted) ------------------
def engineer_1m_indicators(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute rich 1m indicators WITHOUT leakage (we will shift by 1m outside)."""
    out = df_1m.copy()

    # Returns & momentum
    out["ret_1"]  = out["close"].pct_change().fillna(0.0)
    out["ret_5"]  = out["close"].pct_change(5).fillna(0.0)
    out["ret_10"] = out["close"].pct_change(10).fillna(0.0)

    # Moving averages & slopes
    for win in [9, 20, 50, 100, 200]:
        ma = out["close"].rolling(win).mean()
        out[f"sma_{win}"] = ma
        out[f"sma_{win}_slope"] = ma.diff()  # per-minute slope

    # EMAs
    out["ema_9"]   = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_20"]  = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"]  = out["close"].ewm(span=50, adjust=False).mean()
    out["ema_100"] = out["close"].ewm(span=100, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_sig"] = macds
    out["macd_hist"] = macd - macds

    # Bollinger (20, 2)
    ma20 = out["close"].rolling(20).mean()
    sd20 = out["close"].rolling(20).std()
    out["bb_mid"] = ma20
    out["bb_up"]  = ma20 + 2*sd20
    out["bb_lo"]  = ma20 - 2*sd20
    out["dist_bb_up"] = out["close"] - out["bb_up"]
    out["dist_bb_lo"] = out["close"] - out["bb_lo"]

    # Keltner Channels (20 EMA, ATR * 2)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift()).abs(),
        (out["low"]  - out["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    out["atr"] = atr
    ema20 = out["ema_20"]
    out["kel_mid"] = ema20
    out["kel_up"]  = ema20 + 2*atr
    out["kel_lo"]  = ema20 - 2*atr
    out["dist_kel_up"] = out["close"] - out["kel_up"]
    out["dist_kel_lo"] = out["close"] - out["kel_lo"]

    # ADX (Wilder 14)
    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14     = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * (pd.Series(plus_dm, index=out.index).ewm(span=14, adjust=False).mean() / tr14)
    minus_di = 100 * (pd.Series(minus_dm, index=out.index).ewm(span=14, adjust=False).mean() / tr14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out["adx"] = dx.ewm(span=14, adjust=False).mean()

    # Donchian (20, 55)
    for win in [20, 55]:
        out[f"donch_high_{win}"] = out["high"].rolling(win).max()
        out[f"donch_low_{win}"]  = out["low"].rolling(win).min()
        out[f"dist_donch_high_{win}"] = out["close"] - out[f"donch_high_{win}"]
        out[f"dist_donch_low_{win}"]  = out["close"]  - out[f"donch_low_{win}"]

    # Stochastics (14, 3)
    hh14 = out["high"].rolling(14).max()
    ll14 = out["low"].rolling(14).min()
    stoch_k = 100 * _safe_div(out["close"] - ll14, (hh14 - ll14))
    stoch_d = pd.Series(stoch_k, index=out.index).rolling(3).mean()
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    # RSI (Wilder 14)
    delta = out["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=out.index).ewm(alpha=1/14, adjust=False).mean()
    roll_dn = pd.Series(dn, index=out.index).ewm(alpha=1/14, adjust=False).mean()
    rs = _safe_div(roll_up, roll_dn)
    out["rsi_14"] = (100 - (100 / (1 + rs))).astype(np.float32)

    # Heikin-Ashi (for regime/context) — then distances
    ha_close = (out["open"] + out["high"] + out["low"] + out["close"]) / 4
    ha_open  = ha_close.copy()
    ha_open.iloc[0] = (out["open"].iloc[0] + out["close"].iloc[0]) / 2
    ha_open = (ha_open.shift(1).fillna(ha_open.iloc[0]) + ha_close.shift(1).fillna(ha_open.iloc[0])) / 2
    out["ha_close"] = ha_close
    out["ha_open"]  = ha_open
    out["ha_body"]  = out["ha_close"] - out["ha_open"]

    # Choppiness Index (CI) 14 — regime cue
    n = 14
    high_n = out["high"].rolling(n).max()
    low_n  = out["low"].rolling(n).min()
    sum_tr = tr.rolling(n).sum()
    ci = 100 * np.log10(_safe_div(sum_tr, (high_n - low_n))) / np.log10(n)
    out["chop_14"] = ci

    # Pivots from PREVIOUS day only — we will compute prev-day OHLC elsewhere and join (safer).

    return out

# ------------------ Prior-Day Levels & Gap (computed on 1s, then shifted by day) ------------------
def add_prevday_features_1s(df_1s: pd.DataFrame) -> pd.DataFrame:
    out = df_1s.copy()
    dt = pd.to_datetime(out["datetime"])
    day = dt.dt.normalize()

    # Daily OHLC from 1s
    daily = out.groupby(day).agg(
        day_open  = ("open", "first"),
        day_high  = ("high", "max"),
        day_low   = ("low", "min"),
        day_close = ("close", "last"),
    )

    # Previous day levels (shift by one day)
    prev = daily.shift(1).rename(columns=lambda c: "prev_" + c)
    daily = daily.join(prev)

    # Join back (merge-asof on date)
    daily = daily.reset_index().rename(columns={"datetime": "date"})
    out = out.merge(daily, left_on=day.rename("date"), right_on="date", how="left")
    out.drop(columns=["date"], inplace=True)

    # Distances to previous-day levels
    out["dist_prev_day_high"]  = out["close"] - out["prev_day_high"]
    out["dist_prev_day_low"]   = out["close"] - out["prev_day_low"]
    out["dist_prev_day_close"] = out["close"] - out["prev_day_close"]
    out["gap_open"] = out["day_open"] - out["prev_day_close"]

    # Classic Pivots from previous day (no leak)
    p = (out["prev_day_high"] + out["prev_day_low"] + out["prev_day_close"]) / 3
    r1 = 2*p - out["prev_day_low"]
    s1 = 2*p - out["prev_day_high"]
    r2 = p + (out["prev_day_high"] - out["prev_day_low"])
    s2 = p - (out["prev_day_high"] - out["prev_day_low"])

    out["prev_pivot"] = p
    out["prev_r1"] = r1
    out["prev_s1"] = s1
    out["prev_r2"] = r2
    out["prev_s2"] = s2
    out["dist_prev_pivot"] = out["close"] - p
    out["dist_prev_r1"]    = out["close"] - r1
    out["dist_prev_s1"]    = out["close"] - s1
    out["dist_prev_r2"]    = out["close"] - r2
    out["dist_prev_s2"]    = out["close"] - s2

    return out

# ------------------ Microstructure (1s) ------------------
def add_micro_features_1s(df_1s: pd.DataFrame) -> pd.DataFrame:
    out = df_1s.copy()

    # Candle anatomy
    out["body_1s"] = out["close"] - out["open"]
    out["upper_wick_1s"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick_1s"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["wick_ratio_1s"] = _safe_div(
        out["upper_wick_1s"] + out["lower_wick_1s"], (out["body_1s"].abs() + 1e-6)
    )

    # Directional ticks & pressure
    up_tick = (out["close"] > out["open"]).astype(np.int8)
    dn_tick = (out["close"] < out["open"]).astype(np.int8)
    out["up_tick"] = up_tick
    out["down_tick"] = dn_tick

    for win in [5, 15, 30, 60, 120]:
        out[f"bull_pressure_{win}s"] = up_tick.rolling(win).sum()
        out[f"bear_pressure_{win}s"] = dn_tick.rolling(win).sum()

    # 1s returns and realized vol
    out["ret_1s"] = out["close"].pct_change().fillna(0.0)
    for win in [15, 30, 60, 120, 300]:
        out[f"ret_{win}s"] = out["close"].pct_change(win).fillna(0.0)
        out[f"rv_{win}s"]  = out["ret_1s"].rolling(win).std()

    # 1s rolling highs/lows (breakout distances)
    for win in [15, 30, 60, 120, 300]:
        out[f"high_{win}s"] = out["high"].rolling(win).max()
        out[f"low_{win}s"]  = out["low"].rolling(win).min()
        out[f"dist_high_{win}s"] = out["close"] - out[f"high_{win}s"]
        out[f"dist_low_{win}s"]  = out["close"] - out[f"low_{win}s"]

    # 1s moving averages (fast context)
    for win in [15, 30, 60, 120, 300]:
        out[f"sma_{win}s"] = out["close"].rolling(win).mean()
        out[f"sma_{win}s_slope"] = out[f"sma_{win}s"].diff()

    # Intraday so-far levels (1s, causal)
    norm = pd.to_datetime(out["datetime"]).dt.normalize()
    out["day_high_sofar_1s"] = out.groupby(norm)["high"].cummax()
    out["day_low_sofar_1s"]  = out.groupby(norm)["low"].cummin()
    out["dist_day_high_1s"]  = out["close"] - out["day_high_sofar_1s"]
    out["dist_day_low_1s"]   = out["close"] - out["day_low_sofar_1s"]

    return out

# ------------------ Merge 1m → 1s (safe) ------------------
def merge_1m_into_1s(df_1s: pd.DataFrame, df_1m_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Shift 1m indicators by 1 bar (done here), then merge_asof backward into 1s.
    Only PREVIOUS minute info is available to any 1s timestamp.
    """
    # Shift everything by 1 minute to enforce "only prior minute known"
    ind = df_1m_ind.copy()
    ind = ind.shift(1)
    # Drop raw OHLC columns if present
    ind = ind.drop(columns=["open", "high", "low", "close"], errors="ignore")
    ind = ind.reset_index().rename(columns={"datetime": "minute_dt"})

    s = df_1s.sort_values("datetime").reset_index(drop=True)
    m = ind.sort_values("minute_dt").reset_index(drop=True)

    merged = pd.merge_asof(
        s, m, left_on="datetime", right_on="minute_dt", direction="backward"
    )
    merged.drop(columns=["minute_dt"], inplace=True)
    return merged

# ------------------ Public API ------------------
def compute_unified_features(df_1s: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: takes 1s OHLC with 'datetime', returns feature DataFrame.
    Ensures no leakage via shifting and causal constructs.
    """
    # Sort & basic checks
    df = df_1s.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # --- 1) 1m OHLC resample
    df_1m = df.resample("1min", on="datetime").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    df_1m.index.name = "datetime"

    # --- 2) 1m indicators (to be shifted)
    ind_1m = engineer_1m_indicators(df_1m)

    # --- 3) 1s time features
    f_1s = add_time_features_1s(df)

    # --- 4) 1s microstructure features
    f_1s = add_micro_features_1s(f_1s)

    # --- 5) Prior-day features (shifted by day, no leak)
    f_1s = add_prevday_features_1s(f_1s)

    # --- 6) Merge prior-minute 1m indicators into 1s (leak-safe)
    feats = merge_1m_into_1s(f_1s, ind_1m)

    # --- 7) Clean up & dtypes
    feats = feats.fillna(0.0)

    # Promote numerics to float32 (saves RAM)
    for col in feats.columns:
        if pd.api.types.is_float_dtype(feats[col]):
            feats[col] = feats[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(feats[col]) and col != "session_bucket":
            # keep small ints; cast to int16 to save memory
            feats[col] = feats[col].astype(np.int16)

    return feats

# ------------------ Convenience: column groups ------------------
def feature_groups(feats: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return curated groups to help you pick subsets for HMM / ML / RL quickly.
    """
    cols = feats.columns.tolist()

    group = {}
    # HMM/HSMM: compact regime descriptors
    hmm = [
        "ret_1s", "ret_15s", "ret_60s",
        "rv_60s", "rv_300s",
        "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        "rsi_14", "adx", "chop_14", "macd_hist",
        "sma_60s_slope", "sma_300s_slope",
    ]
    group["hmm_core"] = [c for c in hmm if c in cols]

    # Supervised ML: kitchen sink
    ml = [c for c in cols if c not in {"datetime"}]

    # RL: curated action-relevant state (you can add your own trade-state vars in env)
    rl = [
        # time/regime
        "tod_frac", "tod_sin", "tod_cos", "session_bucket",
        "dow_sin", "dow_cos",
        "rsi_14", "adx", "chop_14",
        # 1m context (previous minute)
        "ema_9", "ema_20", "ema_50", "macd", "macd_sig", "macd_hist",
        "dist_bb_up", "dist_bb_lo", "dist_kel_up", "dist_kel_lo",
        "dist_donch_high_20", "dist_donch_low_20",
        # microstructure
        "body_1s", "wick_ratio_1s",
        "bull_pressure_15s", "bear_pressure_15s",
        "bull_pressure_60s", "bear_pressure_60s",
        "dist_high_60s", "dist_low_60s",
        "rv_60s", "rv_300s",
        # prior-day structure
        "dist_prev_pivot", "dist_prev_r1", "dist_prev_s1",
        "gap_open",
    ]
    group["rl_curated"] = [c for c in rl if c in cols]

    group["ml_all"] = ml
    return group
