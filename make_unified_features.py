#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Unified Features CSV
-------------------------
Usage:
    python make_unified_features.py --input ../latest.csv --output features.csv

Expects input CSV with columns: datetime, open, high, low, close
(No volume needed.)
"""

import argparse
import pandas as pd
from unified_features import compute_unified_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Path to 1s OHLC CSV (with datetime, open, high, low, close)")
    ap.add_argument("--output", required=True, help="Where to write features.csv")
    args = ap.parse_args()

    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input, parse_dates=["datetime"])
    keep = ["datetime", "open", "high", "low", "close"]
    df = df[keep].sort_values("datetime").reset_index(drop=True)
    print(f"Rows: {len(df)}")

    print("Computing unified (leak-safe) features …")
    feats = compute_unified_features(df)
    print(f"Output rows: {len(feats)}, columns: {len(feats.columns)}")

    print(f"Writing to {args.output} …")
    feats.to_csv(args.output, index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()
