"""
S6E9 V11 - LightGBM + Deep Analysis Targeted Features (CPU)
================================================================================
Strategy: V10 Modified (NEW HIGH 0.94636) + 5 features from deep data analysis

Instead of adding generic features (conditional stats, distribution, etc.),
this version adds ONLY 5 highly targeted features that directly encode
patterns found in the deep data analysis.

DEEP ANALYSIS FINDINGS → 5 TARGETED FEATURES:

1. _never_buy_flag = (Subsidy=No) OR (RA=High)
   → Analysis: Subsidy=No has 0.58% buy rate (248K people)
   → Analysis: RA=High has 0.14% buy rate (2.2K people)
   → Combined: captures 250K+ people at <0.6% buy rate (near-deterministic)

2. _always_buy_flag = (Income >= 170537)
   → Analysis: 100% buy rate (n=393) — hard cliff, every single person buys

3. _income_round00 = (Income % 100 == 0)
   → Analysis: Income ending in "00" → 5.39% buy rate vs 17.46% overall (3× lower)
   → CTGAN artifact: round numbers are synthetic non-buyers

4. trigram_Sub_ECL_RA = 3-way interaction string ("Yes_5_Low", etc.)
   → Analysis: The ENTIRE signal is in Sub×ECL×RA 3-way interaction
   → V10 only had 2-way bigrams (ECL×RA, ECL×Subsidy) — MISSING the 3-way
   → Triple TE'd: captures the full 30-cell interaction table

5. _recipe_residual = recipe_score - 5.5
   → Analysis: recipe_score alone has AUC 0.93769
   → Threshold at 5.5: above=69.13% buy, below=6.28% buy
   → Residual = distance from decision boundary (positive=likely buyer)

Kept from V10 Modified (proven NEW HIGH 0.94636):
- V3's full FE pipeline (digit, interactions, smooth keys, magic flags, org means, freq)
- Targeted bigrams: ECL_bin×RA, ECL_bin×Subsidy (Triple TE'd)
- Selective groupby deviation (12 features: ECL_bin/income_bin groups, deviation only)
- V3's LightGBM params (lr=0.02, depth=5, num_leaves=32, colsample=0.3, CPU)
- Triple TE (auto, 10, 100) on all cat + num-as-string + smooth-key + bigram cols
- Per-fold orig concat
- Feature selection (drop constants + corr=1.0)

Device: CPU | Est. Time: ~36-40 min
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
from sklearn.preprocessing import TargetEncoder
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
    VERSION_NAME = "v11"
    EXP_ID = "S6E9_V11_LGBM_DeepAnalysis"
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
# 4. MODEL PARAMETERS (V3/V10 LightGBM — proven 0.94634/0.94636)
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
# 6. FEATURE ENGINEERING
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


# V10 Modified: Targeted Bigrams
def add_targeted_bigrams(train_df, test_df):
    print("   Adding TARGETED bigram interactions (ECL_bin×RA, ECL_bin×Subsidy)...")

    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    bigram_name_1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bigram_name_1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bigram_name_1] = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)

    bigram_name_2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bigram_name_2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bigram_name_2] = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)

    bigram_cols = [bigram_name_1, bigram_name_2]
    print(f"      Added {len(bigram_cols)} TARGETED bigram columns")
    return train_df, test_df, bigram_cols


# V10 Modified: Selective Groupby
def add_selective_groupby(train_df, test_df):
    print("   Adding SELECTIVE groupby deviation features...")

    if 'ECL_bin' not in train_df.columns:
        for df in [train_df, test_df]:
            df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    for df in [train_df, test_df]:
        df['income_bin'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                   bins=5, labels=['1', '2', '3', '4', '5']).astype(str)

    combined = pd.concat([train_df, test_df], axis=0)

    group_cols = ['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level']
    agg_num_cols = ['Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']

    new_features = []
    for group_col in group_cols:
        for num_col in agg_num_cols:
            if group_col == num_col or group_col not in combined.columns:
                continue
            grp_mean = combined.groupby(group_col, observed=False)[num_col].mean()
            grp_std = combined.groupby(group_col, observed=False)[num_col].std().fillna(0)

            dev_name = f'grp_{group_col}_{num_col}_dev'
            train_df[dev_name] = ((train_df[num_col] - train_df[group_col].map(grp_mean)) /
                                  (train_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            test_df[dev_name] = ((test_df[num_col] - test_df[group_col].map(grp_mean)) /
                                 (test_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            new_features.append(dev_name)

    print(f"      Added {len(new_features)} SELECTIVE groupby deviation features")
    return train_df, test_df


# NEW V11: Deep Analysis Targeted Features (5 features)
def add_deep_analysis_features(train_df, test_df, orig_df):
    """
    5 highly targeted features from deep data analysis.

    Each directly encodes a pattern found in the data — no generic features.
    """
    print("   Adding DEEP ANALYSIS targeted features (5 features)...")

    for df in [train_df, test_df, orig_df]:
        # Feature 1: _never_buy_flag = (Subsidy=No) OR (RA=High)
        # Analysis: Subsidy=No → 0.58% buy rate (248K people)
        #           RA=High → 0.14% buy rate (2.2K people)
        # Combined: captures 250K+ people at <0.6% buy rate
        df['_never_buy_flag'] = (
            (df['Subsidy_Available'] == 'No') | (df['Range_Anxiety_Level'] == 'High')
        ).astype('int8')

        # Feature 2: _always_buy_flag = (Income >= 170537)
        # Analysis: 100% buy rate (n=393) — hard cliff
        df['_always_buy_flag'] = (df['Annual_Income_USD'] >= 170537).astype('int8')

        # Feature 3: _income_round00 = (Income % 100 == 0)
        # Analysis: Income ending in "00" → 5.39% buy rate vs 17.46% overall (3× lower)
        # CTGAN artifact: round numbers are synthetic non-buyers
        df['_income_round00'] = (
            df['Annual_Income_USD'].fillna(0).astype(int) % 100 == 0
        ).astype('int8')

        # Feature 5: _recipe_residual = recipe_score - 5.5
        # Analysis: recipe_score alone has AUC 0.93769
        # Threshold at 5.5: above=69.13% buy, below=6.28% buy
        # Residual = distance from decision boundary
        recipe_score = (
            1.2 * (df['Annual_Income_USD'] / 1e5)
            + 0.6 * df['Environmental_Concern_Level']
            + 2.0 * (df['Subsidy_Available'] == 'Yes').astype(float)
            - 1.0 * (df['Range_Anxiety_Level'] == 'Medium').astype(float)
            - 3.0 * (df['Range_Anxiety_Level'] == 'High').astype(float)
        )
        df['_recipe_residual'] = (recipe_score - 5.5).astype('float32')

    # Feature 4: trigram_Sub_ECL_RA = 3-way interaction string
    # Analysis: The ENTIRE signal is in Sub×ECL×RA 3-way interaction
    # V10 only had 2-way bigrams — MISSING the 3-way
    # This gets Triple TE'd in the training loop
    trigram_name = 'trigram_Sub_ECL_RA'
    for df in [train_df, test_df, orig_df]:
        df[trigram_name] = (
            df['Subsidy_Available'].astype(str) + '_' +
            df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            df['Range_Anxiety_Level'].astype(str)
        )

    # Feature 6: bigram_income_band_x_Subsidy = "Poor + Subsidy = Buy EV"
    # Discussion finding: $30k + Subsidy=Yes → 7.01% buy vs $30k + Subsidy=No → 0.13% (54×)
    # Our _Income_x_Subsidy (continuous, AUC 0.814) captures SOME of this,
    # but the relationship is a STEP FUNCTION, not smooth.
    # Categorical bigram captures the discrete jumps.
    bigram_inc_sub_name = 'bigram_income_band_x_Subsidy'
    income_bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
    income_labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
    for df in [train_df, test_df, orig_df]:
        df['_inc_band'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=income_bins, labels=income_labels).astype(str)
        df[bigram_inc_sub_name] = df['_inc_band'] + '_' + df['Subsidy_Available'].astype(str)
        df.drop(columns=['_inc_band'], inplace=True)

    trigram_cols = [trigram_name, bigram_inc_sub_name]
    print(f"      Added 6 DEEP ANALYSIS features:")
    print(f"        1. _never_buy_flag (Subsidy=No OR RA=High → 0.6% buy rate)")
    print(f"        2. _always_buy_flag (Income >= $170,537 → 100% buy rate)")
    print(f"        3. _income_round00 (Income ending in 00 → 5.39% vs 17.46%)")
    print(f"        4. trigram_Sub_ECL_RA (3-way interaction → Triple TE'd)")
    print(f"        5. _recipe_residual (recipe_score - 5.5 → distance from boundary)")
    print(f"        6. bigram_income_band_x_Subsidy (Poor+Subsidy=Buy → Triple TE'd)")

    return train_df, test_df, orig_df, trigram_cols


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
    print(f"NEW: 6 Deep Analysis targeted features (from data + discussion)")
    print(f"  1. _never_buy_flag (Subsidy=No OR RA=High = 0.6% buy)")
    print(f"  2. _always_buy_flag (Income>=$170,537 = 100% buy)")
    print(f"  3. _income_round00 (Income ends 00 = 5.4% vs 17.5%)")
    print(f"  4. trigram_Sub_ECL_RA (3-way interaction = Triple TE)")
    print(f"  5. _recipe_residual (recipe_score - 5.5)")
    print(f"  6. bigram_income_band_x_Subsidy (Poor+Subsidy=Buy = Triple TE)")
    print(f"Base: V10 Modified (0.94636) + V3 full FE + targeted bigrams + groupby")
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
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V10 Modified + Deep Analysis features)...")

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

    print("   Adding frequency encoding...")
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    # V10 Modified: Targeted Bigrams
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    for df in [orig_aligned]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    # V10 Modified: Selective Groupby
    train, test = add_selective_groupby(train, test)
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    orig_aligned['income_bin'] = pd.cut(orig_aligned['Annual_Income_USD'].fillna(orig_aligned['Annual_Income_USD'].median()),
                                         bins=5, labels=['1', '2', '3', '4', '5']).astype(str)
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

    # Drop helper columns
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    # NEW V11: Deep Analysis Targeted Features
    train, test, orig_aligned, trigram_cols = add_deep_analysis_features(train, test, orig_aligned)

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Define TE columns: original + bigrams + NEW trigram
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols + trigram_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Original cat + num-as-string + smooth keys: {len(TARGET_ENCODE_COLS) - len([c for c in bigram_cols if c in TARGET_ENCODE_COLS]) - len([c for c in trigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - V10 Bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - NEW V11 Trigram cols: {len([c for c in trigram_cols if c in TARGET_ENCODE_COLS])}")

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

        # ---- Train LightGBM ----
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
            # Show NEW V11 Deep Analysis features
            new_feats = imp[imp['feature'].str.contains('_never_buy|_always_buy|_income_round|trigram|_recipe_residual|TE_trigram|TE_bigram_income')]
            if len(new_feats) > 0:
                print(f"\n      NEW V11 Deep Analysis features:")
                print(new_feats.head(10).to_string(index=False))
            # Show V10 targeted features
            v10_feats = imp[imp['feature'].str.contains('grp_|TE_bigram_')]
            if len(v10_feats) > 0:
                print(f"\n      V10 Targeted features (bigram + groupby):")
                print(v10_feats.head(5).to_string(index=False))
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
    print(f"V11 RESULTS — LightGBM + Deep Analysis Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  NEW V11: 6 Deep Analysis targeted features:")
    print(f"    1. _never_buy_flag (Subsidy=No OR RA=High → 0.6% buy rate)")
    print(f"    2. _always_buy_flag (Income>=$170,537 → 100% buy rate)")
    print(f"    3. _income_round00 (Income ends in 00 → 5.4% vs 17.5%)")
    print(f"    4. trigram_Sub_ECL_RA (3-way interaction → Triple TE'd)")
    print(f"    5. _recipe_residual (recipe_score - 5.5 → distance from boundary)")
    print(f"  V10: Targeted bigrams (ECL_bin×RA, ECL_bin×Subsidy)")
    print(f"  V10: Selective groupby deviation (12 features)")
    print(f"  V3: Full FE pipeline (digit, interactions, smooth keys, magic flags, org means, freq)")
    print(f"Base: V10 Modified (0.94636) + 6 Deep Analysis features")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
