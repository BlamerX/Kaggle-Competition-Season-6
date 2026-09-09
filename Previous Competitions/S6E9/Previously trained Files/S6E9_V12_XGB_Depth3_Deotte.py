"""
S6E9 V12 - XGBoost Depth=3 + V3 FE + Proven Discussion Features (GPU)
================================================================================
Strategy: XGBoost (depth=3, Deotte-style) + V3 full FE + V10/V11 proven features

Why V12 (new model, not another LGBM):
- V9 XGBoost (depth=7) scored 0.94612 — our 4th best
- S6E2 evidence: depth=3 (V16 Deotte, 0.95382) > depth=2 (V11 Stumps, 0.95377)
- S6E2 insight: "The data wants sharp cuts" — S6E9's signal is rule-based
- Different inductive bias from LGBM (depth-wise vs leaf-wise)

Key differences from V9 (our previous XGBoost):
1. depth=3 (V9: depth=7) — shallower, sharper cuts, less overfitting
2. lr=0.0025 (V9: lr=0.005) — slower, more iterations
3. n_estimators=50000 (V9: 10000) — more trees for shallow depth
4. 15-fold CV (V9: 5-fold) — more training data per fold
5. Inner K-Fold TE with mean stat (V9: sklearn TargetEncoder)
6. Frequency encoding on numericals (V9: only on cats)

FE Pipeline (V3 base + V10/V11 proven features):
- V3 full FE: digit, interactions, smooth keys, magic flags, org means, freq
- V10 proven: trigram_Sub_ECL_RA (3-way interaction, V11: 339 importance)
- V10 proven: bigram_income_band_x_Subsidy (Poor+Subsidy, V11: 121 importance)
- V10 targeted: bigram_ECL_bin×RA, bigram_ECL_bin×Subsidy
- V10 selective: groupby deviation (12 features)
- Inner K-Fold TE (mean stat) on all cat + num-as-cat + interaction cols
- Per-fold orig concat

Discussion Features (from competition discussions — ALL implemented):
1. Subsidy on/off switch → _ECL_x_Subsidy, _never_buy_flag (via V3 FE)
2. ECL=5+Subsidy=69% → _ev_recipe, bigram_ECL_bin_x_Subsidy
3. RA=High hard wall → _range_anxiety_high, is_env_hater
4. Income≥$170,537=100% → is_millionaire_cliff
5. Income $38-42k dead zone → is_dead_zone
6. ECL==1=0.57% → is_env_hater
7. $30k spike → is_30k_spike
8. Poor+Subsidy=Buy → bigram_income_band_x_Subsidy (NEW V10)
9. 3-way Sub×ECL×RA → trigram_Sub_ECL_RA (NEW V10)
10. Recipe score → _recipe_residual (NOT included — V11 proved cannibalization)

Device: GPU | Est. Time: ~30-40 min
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
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"xgboost version: {xgb.__version__}")
print(f"Device: GPU (cuda)")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v12"
    EXP_ID = "S6E9_V12_XGB_Depth3_Deotte"
    DEVICE = "GPU"

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
# 4. MODEL PARAMETERS (S6E2 V16 Deotte-style — depth=3, proven 0.95382)
# =============================================================================
# Key differences from V9 (depth=7):
# - depth=3: shallower, sharper cuts (S6E2: "data wants sharp cuts")
# - lr=0.0025: slower learning, more iterations
# - n_estimators=50000: more trees for shallow depth
# - subsample=0.8, colsample=0.8: moderate regularization
# - early_stopping=1000: patient early stopping
# - 5-fold (not 15-fold — for speed; S6E2 used 15 but 5 is fine for 668K rows)

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.0025,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': CFG.RANDOM_SEED,
    'early_stopping_rounds': 1000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda',
    'tree_method': 'hist',
    'verbosity': 0,
}

# =============================================================================
# 5. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (V3 base + V10/V11 proven features)
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
    """S6E2 V16 pattern: frequency encoding on train+test+orig combined."""
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


# V10 proven: Targeted Bigrams
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


# V10 proven: Selective Groupby
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


# V10/V11 proven: Trigram + Income-band bigram
def add_proven_deep_analysis_features(train_df, test_df, orig_df):
    """V11 proved only 2 features useful: trigram + income-band bigram."""
    print("   Adding 2 PROVEN deep analysis features (V11 evidence)...")

    trigram_name = 'trigram_Sub_ECL_RA'
    for df in [train_df, test_df, orig_df]:
        df[trigram_name] = (
            df['Subsidy_Available'].astype(str) + '_' +
            df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            df['Range_Anxiety_Level'].astype(str)
        )

    bigram_inc_sub_name = 'bigram_income_band_x_Subsidy'
    income_bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
    income_labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
    for df in [train_df, test_df, orig_df]:
        df['_inc_band'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=income_bins, labels=income_labels).astype(str)
        df[bigram_inc_sub_name] = df['_inc_band'] + '_' + df['Subsidy_Available'].astype(str)
        df.drop(columns=['_inc_band'], inplace=True)

    proven_cols = [trigram_name, bigram_inc_sub_name]
    print(f"      Added 2 PROVEN features: trigram_Sub_ECL_RA + bigram_income_band_x_Subsidy")
    return train_df, test_df, orig_df, proven_cols


# S6E2 V16 pattern: sklearn TargetEncoder with cv=5 (replaces slow manual inner_kfold_te)
# Both do the same thing (inner K-fold mean encoding) but sklearn is 100x faster (C vectorized)
def fast_inner_kfold_te(X_train, y_train, X_val, X_test, te_cols, n_folds=5, seed=42):
    """
    sklearn TargetEncoder with cv=n_folds — equivalent to manual inner K-fold TE
    but 100x faster (C implementation vs Python groupby.apply loops).
    """
    te = TargetEncoder(
        target_type='binary', smooth='auto',
        cv=n_folds, shuffle=True, random_state=seed,
    )
    X_train_enc = te.fit_transform(X_train[te_cols], y_train).astype('float32')
    X_val_enc   = te.transform(X_val[te_cols]).astype('float32')
    X_test_enc  = te.transform(X_test[te_cols]).astype('float32')

    te_features = []
    for i, col in enumerate(te_cols):
        te_name = f"TE_{col}"
        X_train[te_name] = X_train_enc[:, i]
        X_val[te_name]   = X_val_enc[:, i]
        X_test[te_name]  = X_test_enc[:, i]
        te_features.append(te_name)

    return X_train, X_val, X_test, te_features


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
    print(f"Device: {CFG.DEVICE} (XGBoost depth=3) | Folds: {CFG.N_FOLDS}")
    print(f"Model: XGBoost depth=3 (S6E2 V16 Deotte-style, 0.95382 on S6E2)")
    print(f"FE: V3 full pipeline + V10/V11 proven features + Inner K-Fold TE")
    print(f"Discussion features: ALL 10 patterns implemented")
    print(f"  1. Subsidy on/off → _ECL_x_Subsidy, is_millionaire_cliff")
    print(f"  2. ECL=5+Subsidy → _ev_recipe, bigram_ECL_bin_x_Subsidy")
    print(f"  3. RA=High wall → _range_anxiety_high")
    print(f"  4. Income≥$170k → is_millionaire_cliff")
    print(f"  5. $38-42k dead zone → is_dead_zone")
    print(f"  6. ECL==1 → is_env_hater")
    print(f"  7. $30k spike → is_30k_spike")
    print(f"  8. Poor+Subsidy → bigram_income_band_x_Subsidy")
    print(f"  9. 3-way Sub×ECL×RA → trigram_Sub_ECL_RA")
    print(f" 10. Recipe score → NOT included (V11 proved cannibalization)")
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
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V3 + V10/V11 proven + Inner K-Fold TE)...")

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

    print("   Adding frequency encoding (V16 pattern: on numericals too)...")
    # S6E2 V16: frequency encoding on ALL columns (not just cats)
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + NUMS  # V16 also freq-encodes raw numericals
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    # V10 proven: Targeted Bigrams
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    for df in [orig_aligned]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    # V10 proven: Selective Groupby
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

    # V10/V11 proven: Trigram + Income-band bigram
    train, test, orig_aligned, proven_cols = add_proven_deep_analysis_features(train, test, orig_aligned)

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Define TE columns: original cats + num-as-cat + smooth keys + bigrams + trigrams
    # S6E2 V16 used: NUM_AS_CAT + CATS + NEW_CATS (all string columns)
    TE_COLUMNS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols + proven_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   TE columns (Inner K-Fold TE): {len(TE_COLUMNS)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold KFold with per-fold orig concat + Inner K-Fold TE)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost depth=3 ({CFG.N_FOLDS}-Fold CV, orig concat + Inner K-Fold TE)...")

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

        # Per-fold: concat competition train + original (S6E2 V16 pattern)
        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | "
              f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- sklearn TargetEncoder (cv=5 — same as inner K-fold TE, 100x faster) ----
        print(f"      Running sklearn TargetEncoder (cv=5) on {len(TE_COLUMNS)} columns...")
        X_train, X_val, X_test_fold, te_features = fast_inner_kfold_te(
            X_train, y_train, X_val, X_test_fold, TE_COLUMNS,
            n_folds=5, seed=42
        )

        # Drop original string columns (TE replaces them)
        cols_to_drop = [c for c in TE_COLUMNS if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop, errors='ignore')

        # Drop any remaining string columns
        string_cols = [c for c in X_train.columns
                       if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')

        # Fill NaN and convert to float32
        X_train = X_train.fillna(0).astype('float32')
        X_val   = X_val.fillna(0).astype('float32')
        X_test_fold = X_test_fold.fillna(0).astype('float32')

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")

        # ---- Train XGBoost (depth=3, Deotte-style) ----
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (model.best_iteration if hasattr(model, 'best_iteration')
                     and model.best_iteration is not None
                     else model.n_estimators)
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        # Feature importance (fold 1 only)
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            # Show proven features
            proven_feats = imp[imp['feature'].str.contains('trigram|TE_trigram|TE_bigram_income|TE_bigram_ECL')]
            if len(proven_feats) > 0:
                print(f"\n      Proven discussion features:")
                print(proven_feats.head(10).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model
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
    print(f"V12 RESULTS — XGBoost Depth=3 + Deotte-style ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_features)} Inner K-Fold TE = {len(FEATURES) + len(te_features)} total")
    print(f"Model: XGBoost depth=3, lr=0.0025, n_est=50000 (S6E2 V16 Deotte-style)")
    print(f"FE: V3 full pipeline + V10/V11 proven features + Inner K-Fold TE")
    print(f"Discussion features: ALL 10 patterns implemented")
    print(f"V9 XGBoost depth=7: 0.94612 → V12 XGBoost depth=3: {oof_cv:.5f} OOF")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
