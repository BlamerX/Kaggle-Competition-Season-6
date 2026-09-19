"""
S6E9 V15 - K-Nearest Neighbors (CPU)
================================================================================
Strategy: Exact-match lookup + KNN k=10 distance-weighted fallback

Why V15 = KNN:
- XGBoost/LGBM/CatBoost all saturate at ~0.946 OOF (6 versions confirmed)
- Forensics Analysis 1 proved data is DETERMINISTIC (every unique feature
  combo has exactly one label). Theoretical AUC ceiling = 1.0.
- KNN has a different inductive bias: "find nearest train row, copy label"
- For deterministic data, KNN is the correct prior for locally-constant functions

Two-Layer Strategy:
1. EXACT-MATCH LAYER: For each val/test row, check if its exact 13-feature
   tuple exists in train. If yes → use that label (prob 0/1).
2. KNN FALLBACK: For rows with no exact match, use k=10 distance-weighted
   soft probability.

Device: CPU (KDTree is fast enough for 668k rows)
Est. Time: ~10-15 min

Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"Device: CPU (KNN with KDTree)")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v15"
    EXP_ID = "S6E9_V15_KNN"
    DEVICE = "CPU"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'
    N_FOLDS = 5
    RANDOM_SEED = 42
    KNN_K = 10
    KNN_ALGO = 'kd_tree'

# =============================================================================
# 3. SEED
# =============================================================================
np.random.seed(CFG.RANDOM_SEED)

# =============================================================================
# 4. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 5. FEATURE ENCODING — one-hot cats + standardize nums
# =============================================================================
def encode_for_knn(train_df, test_df, orig_df, cat_cols, num_cols):
    """One-hot encode cats, standardize numerics. Returns encoded matrices."""
    n_train = len(train_df); n_test = len(test_df); n_orig = len(orig_df)
    combined = pd.concat([train_df[cat_cols + num_cols],
                          test_df[cat_cols + num_cols],
                          orig_df[cat_cols + num_cols]], axis=0).reset_index(drop=True)

    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)
    scaler = StandardScaler()
    combined[num_cols] = scaler.fit_transform(combined[num_cols].fillna(combined[num_cols].median()))

    X_train = combined.iloc[:n_train].values.astype('float32')
    X_test  = combined.iloc[n_train:n_train+n_test].values.astype('float32')
    X_orig  = combined.iloc[n_train+n_test:].values.astype('float32')

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    X_orig  = np.nan_to_num(X_orig,  nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, X_test, X_orig

# =============================================================================
# 6. EXACT-MATCH LOOKUP
# =============================================================================
def build_exact_match_lookup(train_df, cat_cols, num_cols, target):
    """Build dict: feature-tuple-string → label. For deterministic data = perfect predictor."""
    key_series = train_df[cat_cols + num_cols].astype(str).agg('|'.join, axis=1)
    return dict(zip(key_series.values, train_df[target].values))

def exact_match_predict(query_df, cat_cols, num_cols, lookup, default_prob=0.5):
    """Returns (probs_array, matched_mask). Probs = 0/1 for matched, default for unmatched."""
    keys = query_df[cat_cols + num_cols].astype(str).agg('|'.join, axis=1).values
    probs = np.full(len(keys), default_prob, dtype='float32')
    matched = np.zeros(len(keys), dtype=bool)
    for i, k in enumerate(keys):
        if k in lookup:
            probs[i] = float(lookup[k])
            matched[i] = True
    return probs, matched

# =============================================================================
# 7. KNN SOFT PROBABILITY (distance-weighted)
# =============================================================================
def knn_predict(X_train, y_train, X_query, k=10, chunk_size=5000):
    """Distance-weighted KNN: prob = sum(1/dist * label) / sum(1/dist)."""
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k, algorithm=CFG.KNN_ALGO,
                          metric='euclidean', n_jobs=-1)
    nn.fit(X_train)
    print(f"      KDTree built in {time.time()-t0:.1f}s on {X_train.shape[0]} rows")

    n_query = X_query.shape[0]
    probs = np.zeros(n_query, dtype='float32')

    t0 = time.time()
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        distances, indices = nn.kneighbors(X_query[start:end])
        weights = 1.0 / (distances + 1e-6)
        neighbor_labels = y_train[indices]
        probs[start:end] = (weights * neighbor_labels).sum(axis=1) / weights.sum(axis=1)

    print(f"      KNN query done in {time.time()-t0:.1f}s ({n_query:,} rows)")
    return probs

# =============================================================================
# 8. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Strategy: Exact-match + KNN k={CFG.KNN_K} fallback")
    print("="*80)

    # [1/5] LOAD
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx).astype('int8')
    orig[CFG.TARGET]  = orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx).astype('int8')

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])

    keep = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
            'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
            'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    orig_aligned = orig[keep].copy()

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    print(f"   Train: {train.shape} | Test: {test.shape} | Orig: {orig_aligned.shape}")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f} | (orig): {orig[CFG.TARGET].mean():.4f}")

    # [2/5] ENCODE
    print("\n[2/5] Encoding features for KNN...")
    X_train_enc, X_test_enc, X_orig_enc = encode_for_knn(train, test, orig_aligned, CATS, NUMS)
    print(f"   Encoded: train={X_train_enc.shape}, test={X_test_enc.shape}, orig={X_orig_enc.shape}")

    y_train = train[CFG.TARGET].values.astype('float32')
    y_orig_arr = y_orig.values.astype('float32')

    # [3/5] EXACT-MATCH CHECK
    print("\n[3/5] Exact-match analysis (train vs test)...")
    train_lookup = build_exact_match_lookup(train, CATS, NUMS, CFG.TARGET)
    _, test_matched = exact_match_predict(test, CATS, NUMS, train_lookup)
    n_test_matched = test_matched.sum()
    print(f"   Test rows with exact match in train: {n_test_matched:,} / {len(test):,} ({100*n_test_matched/len(test):.2f}%)")

    train_keys = train[CATS + NUMS].astype(str).agg('|'.join, axis=1)
    n_unique = train_keys.nunique()
    print(f"   Unique train keys: {n_unique:,} / {len(train):,} ({100*n_unique/len(train):.2f}%)")
    if n_unique == len(train):
        print(f"   ✅ Every train row has unique key — confirms deterministic data")

    # [4/5] 5-FOLD OOF + TEST
    print(f"\n[4/5] Training KNN ({CFG.N_FOLDS}-Fold CV)...")

    oof_probs = np.zeros(len(train), dtype='float32')
    fold_scores = []
    fold_exact_rates = []

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    t0 = time.time()
    for fold, ((train_idx, val_idx), (or_train_idx, _)) in enumerate(
            zip(kf.split(train), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_tr_fold = np.vstack([X_train_enc[train_idx], X_orig_enc[or_train_idx]])
        y_tr_fold = np.concatenate([y_train[train_idx], y_orig_arr[or_train_idx]])
        X_val_fold = X_train_enc[val_idx]
        y_val_fold = y_train[val_idx]

        # Per-fold exact-match lookup
        train_fold_df = pd.concat([train.iloc[train_idx],
                                    orig_aligned.iloc[or_train_idx].assign(**{CFG.TARGET: y_orig_arr[or_train_idx].astype(int)})],
                                   axis=0).reset_index(drop=True)
        fold_lookup = build_exact_match_lookup(train_fold_df, CATS, NUMS, CFG.TARGET)

        val_exact_probs, val_matched = exact_match_predict(
            train.iloc[val_idx], CATS, NUMS, fold_lookup, default_prob=0.5
        )
        n_val_matched = val_matched.sum()
        fold_exact_rates.append(n_val_matched / len(val_idx))
        print(f"      Val: {len(val_idx):,} | Exact matches: {n_val_matched:,} ({100*n_val_matched/len(val_idx):.2f}%)")

        if (~val_matched).sum() > 0:
            val_knn_probs = knn_predict(X_tr_fold, y_tr_fold, X_val_fold[~val_matched],
                                         k=CFG.KNN_K, chunk_size=5000)
            val_probs = val_exact_probs.copy()
            val_probs[~val_matched] = val_knn_probs
        else:
            val_probs = val_exact_probs

        oof_probs[val_idx] = val_probs
        fold_auc = auc_score(y_val_fold, val_probs)
        fold_scores.append(fold_auc)

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

        del X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, val_probs, val_exact_probs
        gc.collect()

    oof_cv = auc_score(y_train, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Avg exact-match rate: {100*np.mean(fold_exact_rates):.2f}%")

    # [5/5] TEST PREDICTIONS
    print(f"\n[5/5] Predicting test with full train as lookup...")
    full_train_df = pd.concat([train, orig_aligned.assign(**{CFG.TARGET: y_orig_arr.astype(int)})],
                               axis=0).reset_index(drop=True)
    full_lookup = build_exact_match_lookup(full_train_df, CATS, NUMS, CFG.TARGET)

    test_exact_probs, test_matched_final = exact_match_predict(test, CATS, NUMS, full_lookup, default_prob=0.5)
    n_test_matched_final = test_matched_final.sum()
    print(f"   Test exact matches: {n_test_matched_final:,} / {len(test):,} ({100*n_test_matched_final/len(test):.2f}%)")

    if (~test_matched_final).sum() > 0:
        X_full = np.vstack([X_train_enc, X_orig_enc])
        y_full = np.concatenate([y_train, y_orig_arr])
        test_knn_probs = knn_predict(X_full, y_full, X_test_enc[~test_matched_final],
                                      k=CFG.KNN_K, chunk_size=5000)
        test_probs = test_exact_probs.copy()
        test_probs[~test_matched_final] = test_knn_probs
    else:
        test_probs = test_exact_probs

    # SAVE
    print(f"\nSaving outputs...")
    out_dir = "/kaggle/working"; os.makedirs(out_dir, exist_ok=True)
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv + sub_{CFG.VERSION_NAME}.csv")

    # FINAL RESULTS
    print(f"\n{'='*80}")
    print(f"V15 RESULTS — KNN (Exact-Match + k={CFG.KNN_K} fallback)")
    print(f"{'='*80}")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"Avg exact-match rate (val): {100*np.mean(fold_exact_rates):.2f}%")
    print(f"Test exact-match rate:      {100*n_test_matched_final/len(test):.2f}%")
    print(f"V14 XGB baseline: 0.946+ OOF → V15 KNN: {oof_cv:.5f} OOF")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
