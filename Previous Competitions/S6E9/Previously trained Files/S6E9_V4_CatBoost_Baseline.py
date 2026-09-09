"""
S6E9 V4 - CatBoost Baseline (GPU)
================================================================================
Strategy: CatBoost with Native Categorical Handling + Multi-Scale Smooth Keys
          + Triple Target Encoding + Original Dataset Target Means
          + Synthetic-Artifact Flags + Digit Features + Engineered Interactions
          + Original Data (per-fold concat)

References:
- https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637
  (V3 LightGBM port — CV 0.94607 / LB 0.94638; same FE pipeline reused)
- https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94583
  (Hyperparameter style — lr=0.005, depth=7, weak reg)
- https://www.kaggle.com/competitions/playground-series-s6e9/discussion
  (Anhadm's 0.94621 ensemble used XGB+LGB+CatBoost; V4 = our CatBoost ingredient)

V4 Changes from V3:
1. Switched from LightGBM (CPU) to CatBoost (GPU)
   - CatBoost has excellent GPU support (unlike LGBM)
   - Different tree growth (oblivious/symmetric trees) → different inductive bias
   - Native categorical handling — pass strings directly, no label encoding needed
2. Native cat_features handling — original CATS passed as strings to CatBoost
   - CatBoost uses target statistics internally (similar to TE but built-in)
   - Different mechanism than LGB/XGB label encoding → more diverse
3. Kept V3's full FE pipeline (344 features) — Triple TE, smooth keys, magic flags, etc.
4. GPU training (task_type='GPU') — CatBoost GPU is reliable and fast
5. CatBoost-specific params: iterations=10000, learning_rate=0.005, depth=7,
   l2_leaf_reg=2.0, bootstrap_type='Bayesian', early_stopping_rounds=500

Kept from V3 (proven FE pipeline):
- Digit features (8 per numeric column, positions -4 to 3)
- Engineered interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety, _Income_x_Subsidy,
  _Charging_Total, _log_Income, _log_Commute, _log_Charging_Total)
- Hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
- Synthetic-artifact flags (is_30k_spike, is_millionaire_cliff, is_dead_zone, is_env_hater)
- Multi-Scale Smooth Keys (income_exact_int, income100_floor, income1000_floor, commute_integer)
- Original dataset target means (static lookup table per cat+num column)
- Numeric->string conversion + frequency encoding on all cats
- Triple Target Encoding (auto, 10, 100) on cat + num-as-string + smooth-key cols
- Feature selection: drop constant + perfectly-correlated (corr=1.0)
- Per-fold orig concat (10K rows, Buyer_ID dropped, schema aligned)

Dropped from V3 (CatBoost-specific):
- Label encoding of original CATS (CatBoost handles strings natively)
- Dropping original CATS after TE (CatBoost uses them directly via cat_features)

CatBoost Parameters (Evgen-style adapted for CatBoost):
  - iterations=10000, learning_rate=0.005, depth=7
  - l2_leaf_reg=2.0, bootstrap_type='Bayesian' (CatBoost default; good for tabular)
  - random_seed=42, eval_metric='AUC', task_type='GPU'
  - early_stopping_rounds=500, verbose=200

Dataset Structure:
- 13 features + 1 target
- Categorical: Gender, City_Type, Current_Car_Type, Home_Charging_Possible,
               Subsidy_Available, Range_Anxiety_Level
- Numerical: Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
             Charging_Stations_Near_Home, Charging_Stations_Near_Work,
             Environmental_Concern_Level
- Target: Will_Buy_EV (No=0, Yes=1) — binary classification, ROC AUC metric

Expected: OOF ~0.9455-0.9465, LB ~0.9460-0.9465
Golden Rules: SKF(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Check sklearn version for TargetEncoder compatibility
print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

import catboost
print(f"Device: GPU (CatBoost)")
print(f"CatBoost version: {catboost.__version__}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v4"
    EXP_ID = "S6E9_V4_CatBoost_Baseline"
    DEVICE = "GPU"  # CatBoost GPU (reliable, unlike LGBM GPU)

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV
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
# 4. MODEL PARAMETERS (Evgen-style adapted for CatBoost)
# =============================================================================
# CatBoost with oblivious/symmetric trees (different from LGB leaf-wise, XGB depth-wise)
# Key differences from V3 LightGBM:
# - depth=7 (CatBoost uses symmetric trees; can go deeper than LGB's 5)
# - learning_rate=0.005 (slower than LGB's 0.02; CatBoost benefits from slow LR)
# - l2_leaf_reg=2.0 (CatBoost-specific L2 reg; matches Evgen's reg_lambda)
# - bootstrap_type='Bayesian' (CatBoost default; good for tabular)
# - task_type='GPU' (CatBoost GPU is reliable; unlike LGBM)
# - Native cat_features handling (no label encoding needed)

CATBOOST_PARAMS = {
    'iterations': 10000,
    'learning_rate': 0.005,
    'depth': 7,
    'l2_leaf_reg': 2.0,
    'bootstrap_type': 'Bayesian',
    'random_seed': CFG.RANDOM_SEED,
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'early_stopping_rounds': 500,
    'verbose': 2000,
    'use_best_model': True,
}

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (Same as V3 — proven pipeline)
# =============================================================================
def add_digit_features(df, num_cols):
    """
    Digit Feature Extraction - 8 features per numerical column (positions -4 to 3).
    Reference notebook's exact approach: fillna(0) before digit extraction.
    """
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    """
    EDA-validated engineered features (kept from V1/V2/V3 — proven signals).

    Adds:
    - 4 arithmetic interactions (ECL x Subsidy, ECL x RangeAnxiety,
      Income x Subsidy, Charging_Total)
    - 3 log transforms (log_Income, log_Commute, log_Charging_Total)
    - 4 hard-edge flags (high_income, range_anxiety_high, ecl_max, ev_recipe)
    """
    df = df.copy()

    # Ordinal encode Range_Anxiety_Level (Low=0, Medium=1, High=2)
    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')

    # Binary Yes/No -> 1/0 for Subsidy
    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

    # ---- Arithmetic interactions (EDA-validated) ----
    df['_ECL_x_Subsidy'] = (
        df['Environmental_Concern_Level'] * df['_Subsidy_bin']
    ).astype('float32')

    df['_ECL_x_RangeAnxiety'] = (
        df['Environmental_Concern_Level'] * (3 - df['_RA_code'])
    ).astype('float32')

    df['_Income_x_Subsidy'] = (
        df['Annual_Income_USD'] * df['_Subsidy_bin']
    ).astype('float32')

    df['_Charging_Total'] = (
        df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    ).astype('float32')

    # ---- Log transforms (right-skewed numerics) ----
    df['_log_Income'] = np.log1p(
        df['Annual_Income_USD'].clip(lower=0)
    ).astype('float32')
    df['_log_Commute'] = np.log1p(
        df['Daily_Commute_km'].clip(lower=0)
    ).astype('float32')
    df['_log_Charging_Total'] = np.log1p(
        df['_Charging_Total'].clip(lower=0)
    ).astype('float32')

    # ---- Hard-edge flags (public discussion findings) ----
    df['_high_income'] = (
        df['Annual_Income_USD'] > 170537
    ).astype('int8')

    df['_range_anxiety_high'] = (
        df['Range_Anxiety_Level'] == 'High'
    ).astype('int8')

    df['_ecl_max'] = (
        df['Environmental_Concern_Level'] == 5
    ).astype('int8')

    df['_ev_recipe'] = (
        (df['Environmental_Concern_Level'] == 5) &
        (df['Range_Anxiety_Level'] == 'Low')
    ).astype('int8')

    # Drop helper columns
    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df


def add_synthetic_artifact_flags(df):
    """
    Synthetic-artifact magic flags from public discussions (reference notebook).
    Community-validated deterministic flaws in the CTGAN synthetic generator.
    """
    df = df.copy()

    # The Mode Collapse Spike — income == 30000 (mode collapse artifact)
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')

    # The Millionaire Cliff — income >= 170537 (100% buy rate region)
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')

    # The Dead Zone — income in [38000, 42000] (0% buy rate region)
    df['is_dead_zone'] = (
        (df['Annual_Income_USD'] >= 38000.0) &
        (df['Annual_Income_USD'] <= 42000.0)
    ).astype('int8')

    # Environmental Concern Extremes — ECL == 1 (env haters)
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')

    return df


def add_smooth_keys(df):
    """
    Markus's Multi-Scale "Smooth Keys" — coarse-to-fine string bins for
    income and commute. These get frequency-encoded and triple-TE'd, giving
    the model multiple resolution views of the same numeric value.

    Reference notebook's V3 update: +0.0002 OOF gain from this technique.
    """
    df = df.copy()

    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)

    return df


def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
    """
    Map each value in cat+num columns to its mean target in the original 10K dataset.
    This is a STATIC lookup table (different from per-fold orig concat).
    Values not in orig get the global orig mean.

    Reference notebook's approach: anchors model to real-world statistics.
    """
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
    """
    Convert every numeric column to a string column ({col}_cat).
    These string columns then get frequency-encoded and triple-TE'd.
    Reference notebook's approach: gives model a different view of numerics.

    Skips columns that don't exist in df (defensive — orig_aligned may not
    have all the derived columns that train/test have).
    """
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
    """
    Global frequency encoding on specified columns (reference notebook's approach).
    Computes normalized frequency across train+test combined (no target leakage).
    Adds a _fe column for each input column.
    """
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)

    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)

    return train_df, test_df


def drop_redundant_features(train_df, test_df, target):
    """
    Feature selection (reference notebook's approach):
    1. Drop constant columns (nunique == 1 in train OR test)
    2. Drop perfectly correlated columns (corr == 1.0 in train)
    """
    # Find perfectly correlated features (numeric only)
    eval_cols = [c for c in train_df.columns
                 if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]

    to_drop_corr = []
    if len(eval_cols) > 1:
        corr_matrix = train_df[eval_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
        del corr_matrix, upper_tri
        gc.collect()

    # Find constant features in train or test
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
    """
    Align original dataset schema with competition train.
    Drops Buyer_ID; reorders columns to match competition schema.
    """
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
    print(f"Device: {CFG.DEVICE} (CatBoost) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat + static target means)")
    print(f"Triple TE: auto, 10, 100 smoothing on all cat + num-as-string cols")
    print(f"Native cat_features: original CATS passed as strings to CatBoost")
    print(f"Multi-Scale Smooth Keys: income/commute bins (Markus's technique)")
    print(f"Synthetic-artifact flags: 30k_spike, millionaire_cliff, dead_zone, env_hater")
    print("="*80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Encode target: Yes/No -> 1/0 (S6E9 stores target as strings)
    target2idx = {'No': 0, 'Yes': 1}
    if train[CFG.TARGET].dtype == object:
        train[CFG.TARGET] = (train[CFG.TARGET].astype(str).str.strip()
                               .str.title().map(target2idx))
    if orig[CFG.TARGET].dtype == object:
        orig[CFG.TARGET] = (orig[CFG.TARGET].astype(str).str.strip()
                              .str.title().map(target2idx))

    # Store IDs
    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()

    # Drop id from train/test; align original schema
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)  # drops Buyer_ID + target

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")

    # Identify column types (from test, which has no target)
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
    print(f"   Missing in orig: {orig_aligned.isna().sum().sum()} (CatBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (Same as V3 — proven pipeline)
    # =========================================================================
    print("\n[2/5] Feature Engineering...")

    # 2a. Digit features (8 per numeric column) — reference notebook's exact approach
    print("   Adding digit features...")
    train = add_digit_features(train, NUMS)
    test  = add_digit_features(test,  NUMS)
    orig_aligned = add_digit_features(orig_aligned, NUMS)

    # Update NUMS to include digit features (they're numeric, will be converted to string later)
    digit_cols = [c for c in train.columns if 'digit' in c]
    NUMS_extended = NUMS + digit_cols

    # 2b. Engineered interactions + log transforms + hard-edge flags (from V1/V2/V3)
    print("   Adding engineered interactions + flags...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig_aligned = add_engineered_features(orig_aligned)

    # 2c. Synthetic-artifact magic flags (from reference notebook)
    print("   Adding synthetic-artifact flags...")
    train = add_synthetic_artifact_flags(train)
    test  = add_synthetic_artifact_flags(test)
    orig_aligned = add_synthetic_artifact_flags(orig_aligned)

    # 2d. Multi-Scale Smooth Keys (Markus's technique from reference notebook)
    print("   Adding multi-scale smooth keys...")
    train = add_smooth_keys(train)
    test  = add_smooth_keys(test)
    orig_aligned = add_smooth_keys(orig_aligned)

    # 2e. Original dataset target means (static lookup table)
    print("   Adding original dataset target means...")
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    # Add the same _org_mean columns to orig_aligned (static lookups; values from orig itself)
    # This ensures orig_aligned has the same schema as train/test for per-fold concat
    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean
            orig_aligned[org_mean_name] = orig_aligned[org_mean_name].astype('float32')

    # 2f. Numeric -> string conversion (reference notebook's approach)
    # Convert ALL numeric columns (original + digit + engineered + flags + org_means)
    # to string columns, then frequency-encode and triple-TE them
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

    # 2g. Frequency encoding on all categorical + numeric-as-string + smooth-key cols
    print("   Adding frequency encoding on all cat + num-as-string cols...")
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]

    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0  # Will be overwritten per-fold

    # 2h. Feature selection: drop constant + perfectly-correlated
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # 2i. Define feature groups
    # TARGET_ENCODE_COLS = all cat + num-as-string + smooth-key cols (still in train)
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]) if c in train.columns]

    # FEATURES = all columns in test (after FE + feature selection)
    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Original cat: {len([c for c in CATS if c in TARGET_ENCODE_COLS])}")
    print(f"     - Numeric-as-string: {len([c for c in train_num_cat_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - Smooth keys: {len([c for c in ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'] if c in TARGET_ENCODE_COLS])}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV with per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Training CatBoost ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    t0 = time.time()
    # Split orig in lockstep with train for per-fold concat
    kf_orig = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X, y), orig_splits)):

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

        print(f"      Train (comp+orig): {X_train.shape} | "
              f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- Triple Target Encoding (auto, 10, 100) on TARGET_ENCODE_COLS ----
        # Same as V3 — applied per-fold to prevent leakage
        te_feature_names = []
        for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
            te = TargetEncoder(
                target_type='binary', smooth=smooth_val,
                cv=CFG.N_FOLDS, shuffle=True, random_state=42,
            )
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

        print(f"      Triple TE: {len(te_feature_names)} features "
              f"(3 smoothings x {len(TARGET_ENCODE_COLS)} cols)")

        # ---- CatBoost native cat_features handling ----
        # CatBoost can use string columns directly via cat_features parameter.
        # We pass: original CATS + num-as-string (_cat) + smooth keys (all string cols)
        # These are KEPT (not dropped) — CatBoost uses them natively.
        # Only drop the TARGET_ENCODE_COLS that are redundant with TE features
        # BUT keep original CATS and smooth keys for CatBoost native handling.
        cols_to_drop_after_te = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
        # CRITICAL: Keep original CATS for CatBoost native cat_features
        # (CatBoost handles them via target statistics internally — different from TE)
        cols_to_keep_for_catboost = CATS + [
            'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
        ]
        cols_to_drop_after_te = [c for c in cols_to_drop_after_te
                                  if c not in cols_to_keep_for_catboost]

        X_train = X_train.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop_after_te, errors='ignore')

        # Identify cat_features for CatBoost (all string/object columns)
        cat_features = [c for c in X_train.columns
                        if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        print(f"      CatBoost cat_features: {len(cat_features)} columns")

        # Ensure all cat_features are strings (CatBoost requirement)
        for col in cat_features:
            X_train[col] = X_train[col].astype(str).fillna('NaN')
            X_val[col] = X_val[col].astype(str).fillna('NaN')
            X_test_fold[col] = X_test_fold[col].astype(str).fillna('NaN')

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")
            print(f"      cat_features ({len(cat_features)}): {cat_features[:10]}{'...' if len(cat_features) > 10 else ''}")

        # ---- Train CatBoost (GPU) ----
        # Use Pool for explicit cat_features specification
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Predictions
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = model.get_best_iteration() if model.get_best_iteration() is not None else model.tree_count_
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        # Feature importance (fold 1 only, for visibility)
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.get_feature_importance(),
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model, train_pool, val_pool
        gc.collect()

    # Overall OOF score
    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Best iters: {best_iters}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (only CSV files — no .npy per new rule)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    # Save OOF as CSV (id, pred) — for hill climber and OOF tracking
    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_probs,
    })
    oof_path = os.path.join(out_dir, f"oof_{CFG.VERSION_NAME}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"   [SAVED] {oof_path}")

    # Save submission (probability of class 1)
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_probs,
    })
    sub_path = os.path.join(out_dir, f"sub_{CFG.VERSION_NAME}.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"   [SAVED] {sub_path}")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V4 RESULTS — CatBoost Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  - Triple TE target cols: {len(TARGET_ENCODE_COLS)}")
    print(f"  - Triple TE features: {len(te_feature_names)} (3 smoothings)")
    print(f"  - CatBoost native cat_features: {len(cat_features)} (kept as strings)")
    print(f"Original data: concatenated per-fold + static target means")
    print(f"Hyperparameters: lr=0.005, depth=7, l2_leaf_reg=2.0, bootstrap=Bayesian (GPU)")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)