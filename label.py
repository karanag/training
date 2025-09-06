#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relabel existing candidates_lr.csv with NEW TP/SL (leak-safe, chunked).
- Reads 1s OHLC from features.csv
- Uses each row's (entry_time, entry_price, side)
- Checks next horizon_s seconds for first TP/SL hit
- Writes candidates_lr_relabelled.csv

Usage:
    python label.py \
      --features /content/drive/MyDrive/data/features.csv \
      --candidates candidates_lr.csv \
      --tp 100 --sl 10 --horizon_s 1800
"""

import argparse
import numpy as np
import pandas as pd

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)
    return df

def relabel_block(ohlc: pd.DataFrame, block: pd.DataFrame, tp: float, sl: float, horizon_s: int) -> pd.DataFrame:
    # ohlc indexed by datetime; columns: open, high, low, close
    idx = ohlc.index
    # Map entry_time -> position
    pos = idx.get_indexer(block["entry_time"], method="backfill")
    # If -1, entry time is beyond the last tick → drop
    ok = pos >= 0
    block = block.loc[ok].copy()
    pos = pos[ok]

    highs = ohlc["high"].to_numpy()
    lows  = ohlc["low"].to_numpy()
    opens = ohlc["open"].to_numpy()

    # Ensure entry_price comes from series (robustness)
    entry_px = opens[pos].astype(np.float64)
    side_arr = block["side"].to_numpy()

    # Compute forward window end indices
    end_pos = np.minimum(pos + horizon_s, len(idx)-1)

    labels = np.zeros(len(block), dtype=np.int8)
    hit = np.array(["none"]*len(block), dtype=object)
    t_hit = np.full(len(block), np.nan, dtype=np.float32)
    mfe = np.zeros(len(block), dtype=np.float32)
    mae = np.zeros(len(block), dtype=np.float32)

    for i in range(len(block)):
        a = pos[i]
        b = end_pos[i]
        if b <= a:
            continue

        if side_arr[i] == "long":
            tp_px = entry_px[i] + tp
            sl_px = entry_px[i] - sl

            # First TP/SL time
            tp_idx = np.flatnonzero(highs[a+1:b+1] >= tp_px)
            sl_idx = np.flatnonzero(lows[a+1:b+1]  <= sl_px)

            first_tp = (a+1+tp_idx[0]) if tp_idx.size else None
            first_sl = (a+1+sl_idx[0]) if sl_idx.size else None

            if first_tp is not None and (first_sl is None or first_tp <= first_sl):
                labels[i] = 1; hit[i] = "tp"; t_hit[i] = float(first_tp - a)
            elif first_sl is not None:
                labels[i] = 0; hit[i] = "sl"; t_hit[i] = float(first_sl - a)
            else:
                labels[i] = 0; hit[i] = "none"; t_hit[i] = np.nan

            mfe[i] = float(highs[a+1:b+1].max() - entry_px[i])
            mae[i] = float(lows[a+1:b+1].min()  - entry_px[i])

        else:  # short
            tp_px = entry_px[i] - tp
            sl_px = entry_px[i] + sl

            tp_idx = np.flatnonzero(lows[a+1:b+1]  <= tp_px)
            sl_idx = np.flatnonzero(highs[a+1:b+1] >= sl_px)

            first_tp = (a+1+tp_idx[0]) if tp_idx.size else None
            first_sl = (a+1+sl_idx[0]) if sl_idx.size else None

            if first_tp is not None and (first_sl is None or first_tp <= first_sl):
                labels[i] = 1; hit[i] = "tp"; t_hit[i] = float(first_tp - a)
            elif first_sl is not None:
                labels[i] = 0; hit[i] = "sl"; t_hit[i] = float(first_sl - a)
            else:
                labels[i] = 0; hit[i] = "none"; t_hit[i] = np.nan

            mfe[i] = float(entry_px[i] - lows[a+1:b+1].min())
            mae[i] = float(entry_px[i] - highs[a+1:b+1].max())

    block["tp"] = tp; block["sl"] = sl; block["horizon_s"] = horizon_s
    block["label"] = labels
    block["hit"] = hit
    block["t_hit_s"] = t_hit
    block["mfe"] = mfe
    block["mae"] = mae
    return block

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--candidates", default="candidates_lr.csv")
    ap.add_argument("--tp", type=float, required=True)
    ap.add_argument("--sl", type=float, required=True)
    ap.add_argument("--horizon_s", type=int, default=1800)
    ap.add_argument("--chunk_days", type=int, default=3, help="process N trading days at a time to cap RAM")
    args = ap.parse_args()

    print(f"Loading candidates: {args.candidates}")
    cand = pd.read_csv(args.candidates, parse_dates=["entry_time","datetime_event"])
    if {"entry_time","side"}.difference(cand.columns):
        raise ValueError("candidates file must have 'entry_time' and 'side' columns")
    cand["entry_time"] = pd.to_datetime(cand["entry_time"])
    cand = cand.sort_values("entry_time").reset_index(drop=True)

    print(f"Loading features (OHLC only): {args.features}")
    use_cols = ["datetime","open","high","low","close"]
    fe = pd.read_csv(args.features, usecols=use_cols, parse_dates=["datetime"])
    fe = fe.sort_values("datetime").reset_index(drop=True)
    fe = downcast(fe)
    fe = fe.dropna(subset=["open","high","low","close"])
    fe = fe.set_index("datetime")

    # Process in day-chunks (assumes continuous session per day)
    days = pd.DatetimeIndex(cand["entry_time"]).normalize()
    uniq_days = pd.to_datetime(sorted(days.unique()))
    out_parts = []

    for i in range(0, len(uniq_days), args.chunk_days):
        chunk_days = uniq_days[i:i+args.chunk_days]
        # candidates in these days
        msk = days.isin(chunk_days)
        block = cand.loc[msk].copy()
        if block.empty:
            continue

        # corresponding OHLC slice with a small pad on both sides
        start = chunk_days.min()
        end   = chunk_days.max() + pd.Timedelta(days=1)
        ohlc = fe.loc[(fe.index >= start) & (fe.index < end)]
        if ohlc.empty:
            continue

        print(f"Relabeling {len(block)} entries for {len(chunk_days)} day(s) [{start.date()} .. {end.date()}]")
        block = relabel_block(ohlc, block, args.tp, args.sl, args.horizon_s)
        out_parts.append(block)

    if not out_parts:
        raise RuntimeError("No candidates relabeled (check date ranges).")

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    # Keep same columns as original + updated label fields
    # And preserve features_key_dt if you had it
    if "features_key_dt" not in out.columns and "entry_time" in out.columns:
        out["features_key_dt"] = out["entry_time"]

    out = out.sort_values("entry_time").reset_index(drop=True)
    out.to_csv("candidates_lr_relabelled.csv", index=False)
    print(f"✅ Wrote candidates_lr_relabelled.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
