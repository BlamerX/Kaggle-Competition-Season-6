"""
S6E9 V10 Modified - LightGBM + Targeted Bigram + Selective Groupby (CPU)
================================================================================
Strategy: V3 LightGBM + TARGETED new features (not bulk addition)

Why V10 Modified (not original V10):
- Original V10 added 174 features (150 groupby + 24 bigram) → -0.00008 LB (worse)
- Root cause 1: MISSED the #1 interaction (ECL_bin × Range_Anxiety — 415× rate flip)
- Root cause 2: TOO MANY groupby features (150, only ~10 useful)
- Root cause 3: WRONG groupby groups (Gender/City_Type, not ECL_bin/income_bin)

V10 Modified fixes all 3:
1. Add the 2 MISSING critical bigrams:
   - bigram_ECL_bin_x_RangeAnxiety (the 415× rate flip interaction!)
   - bigram_ECL_bin_x_Subsidy (the strongest gate interaction)
2. Add only TOP 10-15 groupby DEVIATION features (not 150):
   - Group by ECL_bin and income_bin (not Gender/City_Type)
   - Only deviation (not mean/std/min/max)
3. Keep V3's 344 features — don't touch what works

Total: ~360-370 features (not 479) — less saturation, more targeted

Device: CPU | Est. Time: ~32-38 min
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"Device: CPU ONLY")
print(f"LightGBM version: {lgb.__version__}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v10"
    EXP_ID = "S6E9_V10_LGBM_TargetedFeatures"
    DEVICE = "CPU"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'
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
# 4. MODEL PARAMETERS (V3 LightGBM — proven 0.94634)
# =============================================================================
LGBM_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.02,
    'max_depth': 5,
    'num_leaves': 32,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'reg_alpha': 0.071,
    'reg_lambda': 2.0,
    'max_bin': 1024,
    'random_state': CFG.RANDOM_SEED,
    'feature_pre_filter': False,
    'metric': 'auc',
    'n_jobs': -1,
    'verbose': -1,
    'device': 'cpu',
}

# =============================================================================
# 5. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (V3 base + TARGETED new features)
# =============================================================================
def add_digit_features(df, num_cols):
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    df = df.copy()

    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

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
    df = df.copy()
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')
    df['is_dead_zone'] = ((df['Annual_Income_USD'] >= 38000.0) & (df['Annual_Income_USD'] <= 42000.0)).astype('int8')
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
    return df


def add_smooth_keys(df):
    df = df.copy()
    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)
    return df


def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
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
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df


def drop_redundant_features(train_df, test_df, target):
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


# NEW: Targeted Bigram Interactions (the 2 MISSING critical pairs)
def add_targeted_bigrams(train_df, test_df):
    """
    Add the 2 MISSING critical bigram interactions from EDA:
    1. ECL_bin × Range_Anxiety (415× rate flip — the #1 interaction!)
    2. ECL_bin × Subsidy (the strongest gate)

    Original V10 missed these because ECL is numerical — we bin it first.

    These are then Triple TE'd in the training loop.
    """
    print("   Adding TARGETED bigram interactions (ECL_bin × RA, ECL_bin × Subsidy)...")

    # Bin ECL into 5 levels (1-5) — already integer, just convert to string
    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    # Bigram 1: ECL_bin × Range_Anxiety (the 415× rate flip interaction!)
    bigram_name_1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bigram_name_1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bigram_name_1] = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)

    # Bigram 2: ECL_bin × Subsidy (the strongest gate interaction)
    bigram_name_2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bigram_name_2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bigram_name_2] = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)

    bigram_cols = [bigram_name_1, bigram_name_2]
    print(f"      Added {len(bigram_cols)} TARGETED bigram columns (ECL_bin × RA, ECL_bin × Subsidy)")
    return train_df, test_df, bigram_cols


# NEW: Selective Groupby Statistics (only deviation, only ECL_bin/income_bin groups)
def add_selective_groupby(train_df, test_df):
    """
    Add only TOP groupby DEVIATION features (not 150 like original V10).

    Groups by ECL_bin and income_bin (the strongest signals per EDA),
    not Gender/City_Type (weak signals).

    Only deviation (not mean/std/min/max) — V10 showed deviation is what matters.
    """
    print("   Adding SELECTIVE groupby deviation features (ECL_bin, income_bin groups)...")

    # Ensure ECL_bin exists
    if 'ECL_bin' not in train_df.columns:
        for df in [train_df, test_df]:
            df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    # Create income_bin (5 levels)
    for df in [train_df, test_df]:
        df['income_bin'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                   bins=5, labels=['1', '2', '3', '4', '5']).astype(str)

    combined = pd.concat([train_df, test_df], axis=0)

    # Group columns (the strong signals per EDA)
    group_cols = ['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level']

    # Numerical columns to aggregate (the top signals)
    agg_num_cols = ['Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']

    new_features = []
    for group_col in group_cols:
        for num_col in agg_num_cols:
            if group_col == num_col or group_col not in combined.columns:
                continue
            # Compute group mean and std on train+test combined (no target leakage)
            grp_mean = combined.groupby(group_col, observed=False)[num_col].mean()
            grp_std = combined.groupby(group_col, observed=False)[num_col].std().fillna(0)

            # Deviation only (V10 showed this is what matters)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            train_df[dev_name] = ((train_df[num_col] - train_df[group_col].map(grp_mean)) /
                                  (train_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            test_df[dev_name] = ((test_df[num_col] - test_df[group_col].map(grp_mean)) /
                                 (test_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            new_features.append(dev_name)

    print(f"      Added {len(new_features)} SELECTIVE groupby deviation features")
    return train_df, test_df


def align_original_schema(orig_df):
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
    print(f"Device: {CFG.DEVICE} ONLY | Folds: {CFG.N_FOLDS}")
    print(f"FIX 1: Add MISSING bigrams (ECL_bin×RA, ECL_bin×Subsidy)")
    print(f"FIX 2: Selective groupby (only deviation, only ECL_bin/income_bin)")
    print(f"FIX 3: Keep V3's 344 features, add only ~20 targeted features")
    print(f"Base: V3 LightGBM (0.94634) + V3 full FE pipeline")
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
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (V3 base + TARGETED new features)
    # =========================================================================
    print("\n[2/5] Feature Engineering (V3 base + TARGETED features)...")

    # V3 base FE
    print("   Adding digit features...")
    train = add_digit_features(train, NUMS)
    test  = add_digit_features(test,  NUMS)
    orig_aligned = add_digit_features(orig_aligned, NUMS)

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

    # NEW FIX 1: Targeted Bigrams (the 2 MISSING critical interactions)
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    # Add bigram cols + ECL_bin to orig_aligned
    for df in [orig_aligned]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        # Parse the bigram name to get source cols
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    # NEW FIX 2: Selective Groupby (only deviation, only ECL_bin/income_bin groups)
    train, test = add_selective_groupby(train, test)
    # Add ECL_bin and income_bin to orig_aligned for groupby computation
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    orig_aligned['income_bin'] = pd.cut(orig_aligned['Annual_Income_USD'].fillna(orig_aligned['Annual_Income_USD'].median()),
                                         bins=5, labels=['1', '2', '3', '4', '5']).astype(str)
    # Compute groupby deviation for orig using train+test stats
    combined_for_grp = pd.concat([train[['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level',
                                          'Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']],
                                   test[['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level',
                                          'Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']]],
                                  axis=0)
    group_cols = ['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level']
    agg_num_cols = ['Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']
    for group_col in group_cols:
        for num_col in agg_num_cols:
            if group_col == num_col or group_col not in combined_for_grp.columns:
                continue
            grp_mean = combined_for_grp.groupby(group_col, observed=False)[num_col].mean()
            grp_std = combined_for_grp.groupby(group_col, observed=False)[num_col].std().fillna(0)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            orig_aligned[dev_name] = ((orig_aligned[num_col] - orig_aligned[group_col].map(grp_mean)) /
                                       (orig_aligned[group_col].map(grp_std) + 1e-6)).astype('float32')

    # Drop helper columns (ECL_bin, income_bin) — they were only needed for groupby computation
    # LightGBM can't handle object dtype
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Define TE columns: original + TARGETED bigrams
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Original cat + num-as-string + smooth keys: {len(TARGET_ENCODE_COLS) - len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - TARGETED Bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")

    # =========================================================================
    # [3/5] TRAINING
    # =========================================================================
    print(f"\n[3/5] Training LightGBM ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []

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

        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | "
              f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

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

        print(f"      Triple TE: {len(te_feature_names)} features "
              f"(3 smoothings x {len(TARGET_ENCODE_COLS)} cols)")

        # Drop original string columns (TE columns replace them)
        cols_to_drop_after_te = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop_after_te, errors='ignore')

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")

        # ---- Train LightGBM (V3 params) ----
        clf = lgb.LGBMClassifier(**LGBM_PARAMS)

        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=False),
                lgb.log_evaluation(period=1000),
            ],
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += clf.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = clf.best_iteration_ if clf.best_iteration_ is not None else clf.n_estimators
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        # Feature importance (fold 1 only)
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': clf.feature_importances_,
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            # Show TARGETED NEW features specifically
            new_feats = imp[imp['feature'].str.contains('grp_|bigram_|TE_bigram_')]
            if len(new_feats) > 0:
                print(f"\n      Top-10 TARGETED NEW features:")
                print(new_feats.head(10).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, clf
        gc.collect()

    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Best iters: {best_iters}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS
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
    print(f"V10 MODIFIED RESULTS — LightGBM + Targeted Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  - FIX 1: TARGETED bigrams (ECL_bin×RA, ECL_bin×Subsidy) = 6 TE features")
    print(f"  - FIX 2: SELECTIVE groupby deviation (~12 features, ECL_bin/income_bin groups)")
    print(f"Base: V3 LightGBM (0.94634) + V3 full FE pipeline")
    print(f"V10 original (479 features): 0.94626 → V10 modified (~370 features): {oof_cv:.5f} OOF")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)