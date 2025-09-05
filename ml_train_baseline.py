# ml_train_baseline.py  — keep ALL features, chunked + leak-safe
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

CAND_PATH  = "candidates_lr.csv"                        # labels
FEATS_PATH = "/content/drive/MyDrive/data/features.csv" # unified features (1s)
TP, SL, COST = 30.0, 15.0, 0.0

# Streaming config (tune for RAM)
CHUNK_ROWS  = 400_000   # lower if RAM is tight (e.g., 200k)
PRINT_EVERY = 5

def _downcast_numeric_inplace(df: pd.DataFrame) -> None:
    """Shrink numeric dtypes to save RAM (in-place)."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    # Keep session_bucket etc. as ints but shrink width
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        # choose int32 to be safe; change to int16 if you’re sure bounds fit
        df[c] = df[c].astype(np.int32)

def main():
    print("Loading candidates…")
    cand = pd.read_csv(
        CAND_PATH,
        parse_dates=["datetime_event","entry_time","features_key_dt"]
    )
    if cand.empty:
        raise RuntimeError("candidates_lr.csv is empty.")

    # Target
    y = cand["label"].astype("int8").values

    # Datetimes we need from the features table
    need_dt  = pd.to_datetime(cand["features_key_dt"]).values.astype("datetime64[ns]")
    need_i64 = set(need_dt.astype("int64"))  # fast membership test

    # Read header to confirm datetime exists
    hdr = pd.read_csv(FEATS_PATH, nrows=0)
    if "datetime" not in hdr.columns:
        raise ValueError("features.csv must contain a 'datetime' column.")

    print("Streaming features.csv (ALL columns) in chunks…")
    kept = []
    for ci, chunk in enumerate(pd.read_csv(
        FEATS_PATH,
        parse_dates=["datetime"],
        chunksize=CHUNK_ROWS
    ), start=1):
        # membership on epoch-ns
        dt_i64 = chunk["datetime"].values.astype("datetime64[ns]").astype("int64")
        mask = np.fromiter((v in need_i64 for v in dt_i64), count=len(dt_i64), dtype=bool)
        if mask.any():
            sub = chunk.loc[mask].copy()
            _downcast_numeric_inplace(sub)   # shrink RAM before keeping
            kept.append(sub)

        if ci % PRINT_EVERY == 0:
            matched = sum(len(k) for k in kept)
            print(f"  …processed {ci} chunks, matched rows so far: {matched}")

    if not kept:
        raise RuntimeError("No matching datetimes found in features for the candidate entries.")
    fe_small = pd.concat(kept, axis=0, ignore_index=True)
    fe_small = fe_small.drop_duplicates(subset=["datetime"])

    # Merge features at the exact entry second (leak-safe by construction)
    df = cand.merge(
        fe_small, left_on="features_key_dt", right_on="datetime",
        how="inner", suffixes=("","_feat")
    )
    if len(df) < len(cand):
        miss = len(cand) - len(df)
        print(f"⚠️ {miss} candidates had no matching features row (e.g., final second of file).")

    # Build X: drop labels/identifiers; keep all remaining numeric features
    drop_cols = {
        "label","hit","t_hit_s","mfe","mae","tp","sl","horizon_s","side",
        "datetime_event","entry_time","features_key_dt","datetime",
        "price_event","entry_price"
    }
    keep_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Select numeric only; fill NaNs; ensure float32 for ML speed
    X = keep_num.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0.0)
    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype(np.float32)
    # (ints can stay as-is)

    # Chronological split by entry_time (no lookahead)
    order = np.argsort(df["entry_time"].values.astype("datetime64[ns]"))
    cut = int(0.70 * len(order))
    idx_tr, idx_te = order[:cut], order[cut:]
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # Simple inverse-frequency weights
    pos = float(y_tr.mean()) if y_tr.mean() > 0 else 1e-6
    w = np.where(y_tr == 1, 0.5/pos, 0.5/(1.0-pos)).astype(np.float32)

    # Train robust baseline
    clf = HistGradientBoostingClassifier(
        max_depth=None,
        max_leaf_nodes=31,
        learning_rate=0.05,
        min_samples_leaf=200,
        l2_regularization=0.0,
        random_state=42
    )
    clf.fit(X_tr, y_tr, sample_weight=w)

    # Evaluate
    p_tr = clf.predict_proba(X_tr)[:,1]
    p_te = clf.predict_proba(X_te)[:,1]
    print(f"Train AUC: {roc_auc_score(y_tr, p_tr):.3f} | Test AUC: {roc_auc_score(y_te, p_te):.3f}")
    print(f"Train AP : {average_precision_score(y_tr, p_tr):.3f} | Test AP : {average_precision_score(y_te, p_te):.3f}")

    # EV-neutral threshold
    p_star = (SL + COST) / (TP + SL)
    pred = (p_te >= p_star)
    n_pred = int(pred.sum())
    if n_pred > 0:
        prec = (y_te[pred] == 1).mean()
        ev_trade = prec*TP - (1-prec)*SL - COST
    else:
        prec, ev_trade = np.nan, np.nan
    print(f"Threshold p*>={p_star:.3f} → trades: {n_pred}, precision: {prec:.3f}, EV/trade: {ev_trade:.2f} pts")

if __name__ == "__main__":
    main()
