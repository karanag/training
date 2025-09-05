# ml_train_baseline.py
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

CAND_PATH = "candidates_lr.csv"                      # from your labeler
FEATS_PATH = "/content/drive/MyDrive/data/features.csv"  # unified features (1s, leak-safe)
TP = 30.0
SL = 15.0
COST = 0.0  # set >0 if you want to include costs per trade (points)

print("Loading…")
cand = pd.read_csv(CAND_PATH, parse_dates=["datetime_event","entry_time","features_key_dt"])
fe   = pd.read_csv(FEATS_PATH, parse_dates=["datetime"])

# Merge features at the entry timestamp (leak-safe)
df = cand.merge(fe, left_on="features_key_dt", right_on="datetime", how="inner", suffixes=("","_feat"))

# Target
y = df["label"].astype("int8")

# Columns to DROP (labels, identifiers, forward info, prices you don’t want, etc.)
drop_cols = {
    "label","hit","t_hit_s","mfe","mae","tp","sl","horizon_s","side",
    "datetime_event","entry_time","features_key_dt","datetime",
    "price_event","entry_price"
}
keep_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
X = keep_num.select_dtypes(include=[np.number]).copy()

# Fill any residual NaNs (unified features already minimize these)
X = X.fillna(0.0).astype(np.float32)

# Time-based split (chronological)
order = np.argsort(df["entry_time"].values.astype("datetime64[ns]"))
cut = int(0.7 * len(order))
idx_tr, idx_te = order[:cut], order[cut:]

X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]

# Handle class imbalance with simple inverse-frequency sample weights
pos = y_tr.mean() if y_tr.mean() > 0 else 1e-6
w_pos = 0.5 / pos
w_neg = 0.5 / (1.0 - pos)
w = np.where(y_tr.values == 1, w_pos, w_neg)

# Train fast, robust baseline (scale-free)
clf = HistGradientBoostingClassifier(
    max_depth=None,
    max_leaf_nodes=31,
    learning_rate=0.05,
    min_samples_leaf=200,
    l2_regularization=0.0,
    random_state=42
)
clf.fit(X_tr, y_tr, sample_weight=w)

# Eval (probability = decision_function mapped via sigmoid in HGBT)
p_tr = clf.predict_proba(X_tr)[:,1]
p_te = clf.predict_proba(X_te)[:,1]

print(f"Train AUC: {roc_auc_score(y_tr, p_tr):.3f} | Test AUC: {roc_auc_score(y_te, p_te):.3f}")
print(f"Train AP : {average_precision_score(y_tr, p_tr):.3f} | Test AP : {average_precision_score(y_te, p_te):.3f}")

# EV-based threshold: p > (SL + COST) / (TP + SL)
p_star = (SL + COST) / (TP + SL)
thr = float(p_star)
pred = (p_te >= thr)

# Quick trading summary at EV-neutral threshold
n_pred = int(pred.sum())
if n_pred > 0:
    prec = (y_te[pred] == 1).mean()
    exp_per_trade = prec*TP - (1-prec)*SL - COST
else:
    prec = np.nan
    exp_per_trade = np.nan
print(f"Threshold p*>={thr:.3f} → trades: {n_pred}, precision: {prec:.3f}, EV/trade: {exp_per_trade:.2f} pts")

# (Optional) choose threshold to maximize EV on a validation slice:
# ps, rs, ths = precision_recall_curve(y_te, p_te)
# ev = ps*TP - (1-ps)*SL - COST
# best = np.nanargmax(ev)
# best_thr, best_ev = ths[best], ev[best]
# print(f"Best EV threshold ~ {best_thr:.3f}, EV/trade ~ {best_ev:.2f} pts")
