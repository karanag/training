#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Features with LR Candidates
---------------------------------
- Joins candidates_lr.csv with features_with_regimes_intra.csv
- Produces trainable dataset with X = features, y = label
"""

import pandas as pd

def main():
    feats = pd.read_csv("features_with_regimes_intra.csv", parse_dates=["datetime"])
    cands = pd.read_csv("candidates_lr.csv", parse_dates=["features_key_dt"])

    # Merge: align candidate entry time with features row
    merged = cands.merge(
        feats,
        left_on="features_key_dt",
        right_on="datetime",
        how="inner",
        suffixes=("_event", "")
    )

    # Target = label from candidates
    merged["profit"] = merged["label"]

    merged.to_csv("train_dataset.csv", index=False)
    print(f"✅ Wrote train_dataset.csv with {len(merged)} rows.")

if __name__ == "__main__":
    main()
