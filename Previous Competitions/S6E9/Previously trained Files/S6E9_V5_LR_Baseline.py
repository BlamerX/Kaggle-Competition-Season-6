"""
S6E9 V5 - LogisticRegression Baseline (CPU)
================================================================================
Strategy: LogisticRegression with V3 FE Pipeline + Triple TE + StandardScaler
          + Original Data (per-fold concat)

Reference: S6E4 V6 LogisticRegression Baseline (proven pattern)
- Same FE pipeline as V3 (digit features + engineered interactions + smooth keys
  + magic flags + org means + numeric-as-string + frequency encoding)
- Per-fold Triple Target Encoding (auto, 10, 100) on cat + num-as-string cols
- StandardScaler (critical for LR convergence with LBFGS solver)
- KFold (not StratifiedKFold) — avoids CV overfitting with sample weights
- Sample weights for class imbalance (inverse class frequency)
- No Optuna class weight optimization (binary AUC doesn't need it)

Device: CPU ONLY (no GPU, no cuDF)
- LogisticRegression is fast on CPU (multi-core with n_jobs=-1)
- No GPU needed for linear models

Key advantages of LogisticRegression for ensemble diversity:
1. Linear model — captures linear/additive relationships tree models may miss
2. Completely different algorithm class (linear vs tree) → maximum diversity
3. Public evidence: "LR beats advanced models on original dataset" (broccoli beef)
4. Very fast training (~2-3 min total)
5. ROC AUC rewards probability ranking — LR's strength is calibration

Adapted from S6E4 V6:
- Binary classification (not 3-class) — solver='lbfgs', no multi_class param
- ROC AUC metric (not balanced accuracy)
- No idx2target mapping (output probabilities directly)
- No Optuna (binary AUC is threshold-free)
- Save only sub_v5.csv + oof_v5.csv (no .npy)

Pipeline: Same FE as V3/V4
- Digit features (8 per numeric column)
- Engineered interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety, etc.)
- Hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
- Synthetic-artifact flags (is_30k_spike, is_millionaire_cliff, is_dead_zone, is_env_hater)
- Multi-Scale Smooth Keys (income_exact_int, income100_floor, etc.)
- Original dataset target means
- Numeric->string + frequency encoding
- Triple Target Encoding (auto, 10, 100) per fold
- Per-fold orig concat
- Feature selection (drop constants + corr=1.0)

Expected: OOF ~0.943-0.945, LB ~0.944-0.946
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS (CPU ONLY — no GPU, no cuDF)
# =============================================================================
import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"Device: CPU ONLY (no GPU, no cuDF)")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v5"
    EXP_ID = "S6E9_V5_LR_Baseline"
    DEVICE = "CPU"  # Explicit: CPU only

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — KFold (not StratifiedKFold) per S6E4 V6 bug fix
    # Avoids CV overfitting when using sample weights
    N_FOLDS = 5
    RANDOM_SEED = 42

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# LogisticRegression: linear model, needs StandardScaler for convergence
# No class_weight — sample_weights handles imbalance per-sample
# Binary classification: solver='lbfgs' (default for binary)
# ElasticNet would require solver='saga' (slower); using L2 (default) for speed

LR_PARAMS = {
    'C': 1.0,                        # Inverse regularization (default, fine for baseline)
    'max_iter': 2000,                # Plenty for convergence
    'solver': 'lbfgs',               # Best for binary on CPU
    'random_state': CFG.RANDOM_SEED,
    'n_jobs': -1,                    # Parallel
    'verbose': 0,
}

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (Same as V3/V4 — proven pipeline)
# =============================================================================
def add_digit_features(df, num_cols):
    """Digit Feature Extraction - 8 features per numerical column."""
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    """EDA-validated engineered features (kept from V1/V2/V3/V4)."""
    df = df.copy()

    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')
    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

    df['_ECL_x_Subsidy'] = (df['Environmental_Concern_Level'] * df['_Subsidy_bin']).astype('float32')
    df['_ECL_x_RangeAnxiety'] = (df['Environmental_Concern_Level'] * (3 - df['_RA_code'])).astype('float32')
    df['_Income_x_Subsidy'] = (df['Annual_Income_USD'] * df['_Subsidy_bin']).astype('float32')
    df['_Charging_Total'] = (df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']).astype('float32')

    df['_log_Income'] = np.log1p(df['Annual_Income_USD'].clip(lower=0)).astype('float32')
    df['_log_Commute'] = np.log1p(df['Daily_Commute_km'].clip(lower=0)).astype('float32')
    df['_log_Charging_Total'] = np.log1p(df['_Charging_Total'].clip(lower=0)).astype('float32')

    df['_high_income'] = (df['Annual_Income_USD'] > 170537).astype('int8')
    df['_range_anxiety_high'] = (df['Range_Anxiety_Level'] == 'High').astype('int8')
    df['_ecl_max'] = (df['Environmental_Concern_Level'] == 5).astype('int8')
    df['_ev_recipe'] = ((df['Environmental_Concern_Level'] == 5) & (df['Range_Anxiety_Level'] == 'Low')).astype('int8')

    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])
    return df


def add_synthetic_artifact_flags(df):
    """Synthetic-artifact magic flags from public discussions."""
    df = df.copy()
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')
    df['is_dead_zone'] = ((df['Annual_Income_USD'] >= 38000.0) & (df['Annual_Income_USD'] <= 42000.0)).astype('int8')
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
    return df


def add_smooth_keys(df):
    """Markus's Multi-Scale Smooth Keys — coarse-to-fine string bins."""
    df = df.copy()
    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)
    return df


def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
    """Map each value to its mean target in the original 10K dataset."""
    orig_global_mean = orig_df[target].mean()
    orig_stats = {}
    for col in cat_cols + num_cols:
        if col in orig_df.columns:
            orig_stats[col] = orig_df.groupby(col, observed=False)[target].mean()

    for col, stats in orig_stats.items():
        map_name = f"{col}_org_mean"
        train_df[map_name] = train_df[col].map(stats).fillna(orig_global_mean).astype('float32')
        test_df[map_name]  = test_df[col].map(stats).fillna(orig_global_mean).astype('float32')

    return train_df, test_df


def add_numeric_as_string(df, num_cols):
    """Convert numeric columns to string columns for TE."""
    df = df.copy()
    num_to_cat_cols = []
    for col in num_cols:
        if col not in df.columns:
            continue
        cat_name = f"{col}_cat"
        df[cat_name] = df[col].fillna('NaN').astype(str)
        num_to_cat_cols.append(cat_name)
    return df, num_to_cat_cols


def add_frequency_encoding(train_df, test_df, cols_to_encode):
    """Global frequency encoding on specified columns."""
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df


def drop_redundant_features(train_df, test_df, target):
    """Feature selection: drop constant + perfectly-correlated (corr=1.0)."""
    eval_cols = [c for c in train_df.columns
                 if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]
    to_drop_corr = []
    if len(eval_cols) > 1:
        corr_matrix = train_df[eval_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
        del corr_matrix, upper_tri
        gc.collect()

    to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + \
                    [c for c in test_df.columns if test_df[c].nunique() == 1]

    DROP = set(to_drop_corr).union(set(to_drop_const))
    DROP = [c for c in DROP if c not in ['id', target]]

    if len(DROP) > 0:
        print(f"   Dropping {len(DROP)} redundant/constant features")
        train_df = train_df.drop(columns=DROP, errors='ignore')
        test_df  = test_df.drop(columns=DROP, errors='ignore')
    else:
        print("   No redundant features found.")

    return train_df, test_df, DROP


def align_original_schema(orig_df):
    """Align original dataset schema with competition train."""
    keep_cols = [
        'Age', 'Annual_Income_USD', 'Daily_Commute_km',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level',
        'Gender', 'City_Type', 'Current_Car_Type',
        'Home_Charging_Possible', 'Subsidy_Available',
        'Range_Anxiety_Level',
    ]
    return orig_df[keep_cols].copy()

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} ONLY (no GPU, no cuDF) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat + static target means)")
    print(f"Triple TE: auto, 10, 100 smoothing on all cat + num-as-string cols")
    print(f"StandardScaler: applied per fold (critical for LR convergence)")
    print(f"KFold (not StratifiedKFold): avoids CV overfitting with sample weights")
    print("="*80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    if train[CFG.TARGET].dtype == object:
        train[CFG.TARGET] = (train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx))
    if orig[CFG.TARGET].dtype == object:
        orig[CFG.TARGET] = (orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx))

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()

    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]

    print(f"   Categorical columns ({len(CATS)}): {CATS}")
    print(f"   Numerical columns ({len(NUMS)}): {NUMS}")

    print(f"\n   Target mapping: {target2idx}")
    print("\n   Class Distribution (train):")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")
    print(f"   Pos rate (orig):  {y_orig.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (Same as V3/V4)
    # =========================================================================
    print("\n[2/5] Feature Engineering...")

    print("   Adding digit features...")
    train = add_digit_features(train, NUMS)
    test  = add_digit_features(test,  NUMS)
    orig_aligned = add_digit_features(orig_aligned, NUMS)

    digit_cols = [c for c in train.columns if 'digit' in c]
    NUMS_extended = NUMS + digit_cols

    print("   Adding engineered interactions + flags...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig_aligned = add_engineered_features(orig_aligned)

    print("   Adding synthetic-artifact flags...")
    train = add_synthetic_artifact_flags(train)
    test  = add_synthetic_artifact_flags(test)
    orig_aligned = add_synthetic_artifact_flags(orig_aligned)

    print("   Adding multi-scale smooth keys...")
    train = add_smooth_keys(train)
    test  = add_smooth_keys(test)
    orig_aligned = add_smooth_keys(orig_aligned)

    print("   Adding original dataset target means...")
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean
            orig_aligned[org_mean_name] = orig_aligned[org_mean_name].astype('float32')

    print("   Converting numerics to string categories...")
    all_numeric_cols = [c for c in train.columns
                        if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c])
                        and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
                          '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    all_numeric_to_str = all_numeric_cols + engineered_numeric

    train, train_num_cat_cols = add_numeric_as_string(train, all_numeric_to_str)
    test, _ = add_numeric_as_string(test, all_numeric_to_str)
    orig_aligned, _ = add_numeric_as_string(orig_aligned, all_numeric_to_str)
    print(f"   Created {len(train_num_cat_cols)} numeric-as-string columns")

    print("   Adding frequency encoding on all cat + num-as-string cols...")
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold KFold with per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Training LogisticRegression ({CFG.N_FOLDS}-Fold KFold, orig concat + Triple TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []

    # KFold (not StratifiedKFold) per S6E4 V6 bug fix
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    t0 = time.time()
    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Per-fold: concat competition train + original
        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # Sample weights for class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        avg = len(y_train) / 2
        w_neg = avg / neg
        w_pos = avg / pos
        train_weights = np.where(y_train == 0, w_neg, w_pos).astype('float32')

        # ---- Triple Target Encoding (auto, 10, 100) ----
        te_feature_names = []
        for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
            te = TargetEncoder(target_type='binary', smooth=smooth_val,
                               cv=CFG.N_FOLDS, shuffle=True, random_state=42)
            X_train_enc = te.fit_transform(X_train[TARGET_ENCODE_COLS], y_train).astype('float32')
            X_val_enc   = te.transform(X_val[TARGET_ENCODE_COLS]).astype('float32')
            X_test_enc  = te.transform(X_test_fold[TARGET_ENCODE_COLS]).astype('float32')

            for i, col in enumerate(TARGET_ENCODE_COLS):
                te_name = f"TE_{col}_{smooth_name}"
                X_train[te_name] = X_train_enc[:, i]
                X_val[te_name]   = X_val_enc[:, i]
                X_test_fold[te_name] = X_test_enc[:, i]
                if te_name not in te_feature_names:
                    te_feature_names.append(te_name)

        print(f"      Triple TE: {len(te_feature_names)} features added")

        # Drop ALL string columns (LR can't handle strings)
        # After Triple TE, string columns are redundant — TE columns replace them
        string_cols = [c for c in X_train.columns
                       if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')

        # Fill NaN and convert to float32
        X_train = X_train.fillna(0).astype('float32')
        X_val   = X_val.fillna(0).astype('float32')
        X_test_fold = X_test_fold.fillna(0).astype('float32')

        # Normalize column names to string (TargetEncoder outputs int names)
        X_train.columns = X_train.columns.astype(str)
        X_val.columns = X_val.columns.astype(str)
        X_test_fold.columns = X_test_fold.columns.astype(str)

        if fold == 0:
            print(f"      Final feature count (numeric only): {len(X_train.columns)}")

        # ---- StandardScaler (critical for LR convergence with LBFGS) ----
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        X_test_sc  = scaler.transform(X_test_fold)

        # ---- Train LogisticRegression ----
        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_train_sc, y_train, sample_weight=train_weights)

        # Predictions
        val_probs = model.predict_proba(X_val_sc)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_sc)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        converged = "✓" if model.n_iter_[0] < LR_PARAMS['max_iter'] else "✗"
        print(f"      AUC: {fold_auc:.5f} | Conv: {converged} ({model.n_iter_[0]}/{LR_PARAMS['max_iter']}) | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        # Feature importance (fold 1 only) — LR coefficients as importance
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(model.coef_[0]),  # |coefficient| as importance
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1, |coefficient|):")
            print(imp.head(15).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model, scaler
        gc.collect()

    # Overall OOF score
    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (only CSV files — no .npy per new rule)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_path = os.path.join(out_dir, f"oof_{CFG.VERSION_NAME}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"   [SAVED] {oof_path}")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_path = os.path.join(out_dir, f"sub_{CFG.VERSION_NAME}.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"   [SAVED] {sub_path}")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V5 RESULTS — LogisticRegression Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  - After dropping string cols: numeric only (StandardScaled)")
    print(f"Original data: concatenated per-fold + static target means")
    print(f"Hyperparameters: C=1.0, max_iter=2000, solver=lbfgs, sample weights")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)