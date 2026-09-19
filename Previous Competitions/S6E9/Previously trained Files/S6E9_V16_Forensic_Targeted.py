"""
S6E9 V16 - XGBoost + Forensic-Targeted Features (GPU)
================================================================================
Strategy: V14 XGB depth=3 base + 4 new feature groups from V16 forensic report

Forensic Findings (V16_ECL_BLINDSPOT_REPORT.md):
- ECL=1,2,3 subgroups (61% of data) have AUC 0.84-0.87 vs 0.946 global
- Universal disambiguator: Subsidy_Available (already in V14)
- MISSING pair: Subsidy × HomeCharging (gain +0.025 to +0.031 per subgroup)
- Distance-to-buyer-centroid AUC 0.80+ in every subgroup
- (Subsidy=Yes AND Income>median) has 2.4x lift in all subgroups
- Range_Anxiety=High has 0% pos_rate (perfect non-buyer signal)

NEW Features (5 total):
1. bigram_Sub_HomeCharging — Subsidy × HomeCharging string concat + TE
   (Proven pair interaction in all 3 ECL subgroups, gain +0.025-0.031)

2. _dist_to_buyer_centroid_ECL1 — Distance to ECL=1 buyer centroid
3. _dist_to_buyer_centroid_ECL2 — Distance to ECL=2 buyer centroid
4. _dist_to_buyer_centroid_ECL3 — Distance to ECL=3 buyer centroid
   (Per-fold computed on train+orig only; AUC 0.80+ standalone per subgroup)
   (Geometric feature — trees cannot reconstruct from raw inputs)

5. _recipe_sub_income — Binary: (Subsidy=Yes AND Income>median)
   (2.4x lift in all 3 ECL subgroups)

Kept from V14 (proven):
- XGB depth=3, lr=0.0025, n=50000, early_stop=1000
- V3 full FE: digits, smooth keys, magic flags, org means, freq, num-as-string
- V10/V12 proven: trigram_Sub_ECL_RA, bigram_income_band_x_Subsidy
- V10 targeted: bigram_ECL_bin×RA, bigram_ECL_bin×Subsidy
- sklearn TargetEncoder cv=5, smooth='auto'
- Per-fold orig concat (no leakage)
- NO pseudo-labels (clean OOF for ensemble)

Device: GPU | Est. Time: ~30 min
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
from sklearn.preprocessing import TargetEncoder, StandardScaler, LabelEncoder
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
    VERSION_NAME = "v16"
    EXP_ID = "S6E9_V16_Forensic_Targeted"
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
# 4. MODEL PARAMETERS (V14 XGB depth=3 — proven)
# =============================================================================
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
# 6. FEATURE ENGINEERING — V14 base pipeline (unchanged)
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
    for col in cat_cols + num_cols:
        if col in orig_df.columns:
            stats = orig_df.groupby(col, observed=False)[target].mean()
            train_df[f"{col}_org_mean"] = train_df[col].map(stats).fillna(orig_global_mean).astype('float32')
            test_df[f"{col}_org_mean"] = test_df[col].map(stats).fillna(orig_global_mean).astype('float32')
    return train_df, test_df

def add_numeric_as_string(df, num_cols):
    df = df.copy()
    num_to_cat_cols = []
    for col in num_cols:
        if col not in df.columns: continue
        cat_name = f"{col}_cat"
        df[cat_name] = df[col].fillna('NaN').astype(str)
        num_to_cat_cols.append(cat_name)
    return df, num_to_cat_cols

def add_frequency_encoding(train_df, test_df, cols_to_encode):
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"] = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df

def drop_redundant_features(train_df, test_df, target):
    eval_cols = [c for c in train_df.columns
                 if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]
    to_drop_corr = []
    if len(eval_cols) > 1:
        corr_matrix = train_df[eval_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
        del corr_matrix, upper_tri; gc.collect()
    to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + \
                    [c for c in test_df.columns if test_df[c].nunique() == 1]
    DROP = set(to_drop_corr).union(set(to_drop_const))
    DROP = [c for c in DROP if c not in ['id', target]]
    if len(DROP) > 0:
        print(f"   Dropping {len(DROP)} redundant/constant features")
        train_df = train_df.drop(columns=DROP, errors='ignore')
        test_df  = test_df.drop(columns=DROP, errors='ignore')
    return train_df, test_df, DROP

def add_targeted_bigrams(train_df, test_df):
    print("   Adding TARGETED bigram interactions (V10 proven)...")
    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    bn1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bn1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bn1] = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)
    bn2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bn2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bn2] = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)
    return train_df, test_df, [bn1, bn2]

def add_proven_features(train_df, test_df, orig_df):
    print("   Adding 2 PROVEN features (trigram + income-band bigram)...")
    tn = 'trigram_Sub_ECL_RA'
    for df in [train_df, test_df, orig_df]:
        df[tn] = (df['Subsidy_Available'].astype(str) + '_' +
            df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            df['Range_Anxiety_Level'].astype(str))
    bis = 'bigram_income_band_x_Subsidy'
    bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
    labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
    for df in [train_df, test_df, orig_df]:
        df['_inc_band'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=bins, labels=labels).astype(str)
        df[bis] = df['_inc_band'] + '_' + df['Subsidy_Available'].astype(str)
        df.drop(columns=['_inc_band'], inplace=True)
    return train_df, test_df, orig_df, [tn, bis]

def align_original_schema(orig_df):
    keep = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
            'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
            'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    return orig_df[keep].copy()

# =============================================================================
# 7. NEW FORENSIC-TARGETED FEATURES (V16 = the only difference from V14)
# =============================================================================
def add_forensic_pair_bigram(train_df, test_df, orig_df):
    """[F1] Subsidy × HomeCharging bigram — proven pair in all 3 ECL subgroups."""
    print("   Adding forensic pair: Subsidy × HomeCharging...")
    bn = 'bigram_Sub_HomeCharging'
    for df in [train_df, test_df, orig_df]:
        df[bn] = df['Subsidy_Available'].astype(str) + '_' + df['Home_Charging_Possible'].astype(str)
    return train_df, test_df, orig_df, [bn]

def add_recipe_sub_income_flag(df, income_median):
    """[F2] Binary flag: (Subsidy=Yes AND Income>median). 2.4x lift in all ECL subgroups."""
    df = df.copy()
    df['_recipe_sub_income'] = (
        (df['Subsidy_Available'] == 'Yes') & (df['Annual_Income_USD'] > income_median)
    ).astype('int8')
    return df

def compute_buyer_centroids_per_ecl(train_df, target, cat_cols, num_cols, ecl_values=(1, 2, 3)):
    """
    [F3] Compute buyer centroid (mean feature vector of buyers) per ECL subgroup.
    Returns dict {ecl_value: centroid_vector}.

    Used per-fold to avoid leakage: centroids computed on training rows only,
    then distances computed for val/test rows.
    """
    # Encode features
    enc = train_df[cat_cols + num_cols].copy()
    for c in num_cols:
        enc[c] = enc[c].fillna(enc[c].median())
    for c in cat_cols:
        enc[c] = enc[c].fillna('NaN').astype(str)
        enc[c] = LabelEncoder().fit_transform(enc[c])

    # Standardize
    X = enc.values.astype('float32')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-ECL buyer centroid
    centroids = {}
    ecl_col = train_df['Environmental_Concern_Level'].values
    y = train_df[target].values
    for ecl in ecl_values:
        mask = (ecl_col == ecl) & (y == 1)
        if mask.sum() >= 10:
            centroids[ecl] = X[mask].mean(axis=0)
        else:
            centroids[ecl] = np.zeros(X.shape[1], dtype='float32')

    return centroids, scaler

def compute_distance_to_centroids(df, cat_cols, num_cols, centroids, scaler):
    """
    [F3] For each row, compute distance to its ECL subgroup's buyer centroid.
    Adds 3 features: _dist_to_buyer_centroid_ECL1, ECL2, ECL3.

    Rows where ECL != k get distance 0 (the feature is only meaningful for
    that subgroup; model can learn to ignore it for other ECL values).
    """
    df = df.copy()
    enc = df[cat_cols + num_cols].copy()
    for c in num_cols:
        enc[c] = enc[c].fillna(enc[c].median())
    for c in cat_cols:
        enc[c] = enc[c].fillna('NaN').astype(str)
        enc[c] = LabelEncoder().fit_transform(enc[c])

    X = enc.values.astype('float32')
    X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ecl_col = df['Environmental_Concern_Level'].values
    for ecl, centroid in centroids.items():
        dists = np.linalg.norm(X - centroid, axis=1)
        # Zero out distances for rows not in this ECL subgroup
        # (model can learn this feature is only active for ECL=k)
        feat_name = f'_dist_to_buyer_centroid_ECL{ecl}'
        df[feat_name] = np.where(ecl_col == ecl, dists, 0.0).astype('float32')

    return df

# =============================================================================
# 8. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} (XGBoost depth=3) | Folds: {CFG.N_FOLDS}")
    print(f"Base: V14 XGB depth=3 (proven) — NO pseudo-labels")
    print(f"NEW forensic-targeted features (5):")
    print(f"  [F1] bigram_Sub_HomeCharging (pair AUC 0.72-0.75 in ECL subgroups)")
    print(f"  [F2] _recipe_sub_income (2.4x lift in all ECL subgroups)")
    print(f"  [F3] _dist_to_buyer_centroid_ECL1 (centroid AUC 0.801)")
    print(f"  [F4] _dist_to_buyer_centroid_ECL2 (centroid AUC 0.804)")
    print(f"  [F5] _dist_to_buyer_centroid_ECL3 (centroid AUC 0.824)")
    print("="*80)

    # [1/5] LOAD
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)
    orig[CFG.TARGET]  = orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    print(f"   Train: {train.shape} | Pos rate: {train[CFG.TARGET].mean():.4f}")

    # Compute income median for _recipe_sub_income flag
    income_median = train['Annual_Income_USD'].median()
    print(f"   Income median: ${income_median:.0f}")

    # [2/5] FEATURE ENGINEERING (V14 base + V16 forensic additions)
    print("\n[2/5] Feature Engineering (V14 base + V16 forensic)...")
    train = add_digit_features(train, NUMS); test = add_digit_features(test, NUMS); orig_aligned = add_digit_features(orig_aligned, NUMS)
    train = add_engineered_features(train); test = add_engineered_features(test); orig_aligned = add_engineered_features(orig_aligned)
    train = add_synthetic_artifact_flags(train); test = add_synthetic_artifact_flags(test); orig_aligned = add_synthetic_artifact_flags(orig_aligned)
    train = add_smooth_keys(train); test = add_smooth_keys(test); orig_aligned = add_smooth_keys(orig_aligned)
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean

    all_numeric_cols = [c for c in train.columns if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c]) and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy', '_Charging_Total',
                          '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    train, train_num_cat_cols = add_numeric_as_string(train, all_numeric_cols + engineered_numeric)
    test, _ = add_numeric_as_string(test, all_numeric_cols + engineered_numeric)
    orig_aligned, _ = add_numeric_as_string(orig_aligned, all_numeric_cols + engineered_numeric)

    freq_target_cols = CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns: orig_aligned[f"{col}_fe"] = 0.0

    train, test, bigram_cols = add_targeted_bigrams(train, test)
    for df in [orig_aligned]: df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bn in bigram_cols:
        if 'RangeAnxiety' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'Subsidy' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test, orig_aligned, proven_cols = add_proven_features(train, test, orig_aligned)

    # [F1] NEW: Subsidy × HomeCharging bigram
    train, test, orig_aligned, forensic_bigram_cols = add_forensic_pair_bigram(train, test, orig_aligned)

    # [F2] NEW: _recipe_sub_income flag
    train = add_recipe_sub_income_flag(train, income_median)
    test  = add_recipe_sub_income_flag(test, income_median)
    orig_aligned = add_recipe_sub_income_flag(orig_aligned, income_median)

    # Drop helper columns
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin'], errors='ignore', inplace=True)

    # Feature selection on train+test
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Define TE columns (include new forensic bigram)
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
                                       + bigram_cols + proven_cols + forensic_bigram_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']
    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   TE columns: {len(TARGET_ENCODE_COLS)}")

    # Verify forensic features survived
    forensic_check = ['bigram_Sub_HomeCharging', '_recipe_sub_income']
    print(f"   Forensic features surviving selection: {sum(1 for f in forensic_check if f in FEATURES)}/{len(forensic_check)}")
    print(f"   (Distance features added per-fold to avoid leakage)")

    # [3/5] TRAINING
    print(f"\n[3/5] Training XGBoost depth=3 ({CFG.N_FOLDS}-Fold CV)...")

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

        # Per-fold: concat competition train + original
        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        # [F3-F5] NEW: Per-fold buyer centroid distances (computed on TRAIN ONLY to avoid leakage)
        train_with_target = X_train.copy()
        train_with_target[CFG.TARGET] = y_train.values
        centroids, scaler = compute_buyer_centroids_per_ecl(
            train_with_target, CFG.TARGET, CATS, NUMS, ecl_values=(1, 2, 3)
        )
        X_train = compute_distance_to_centroids(X_train, CATS, NUMS, centroids, scaler)
        X_val   = compute_distance_to_centroids(X_val, CATS, NUMS, centroids, scaler)
        X_test_fold = compute_distance_to_centroids(X_test_fold, CATS, NUMS, centroids, scaler)

        print(f"      Train (comp+orig): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # Target Encoding (single smooth='auto' — V12 lesson)
        te_cols_active = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
        te_feature_names = []
        te = TargetEncoder(target_type='binary', smooth='auto',
                           cv=CFG.N_FOLDS, shuffle=True, random_state=42)
        X_train_enc = te.fit_transform(X_train[te_cols_active], y_train).astype('float32')
        X_val_enc   = te.transform(X_val[te_cols_active]).astype('float32')
        X_test_enc  = te.transform(X_test_fold[te_cols_active]).astype('float32')
        for i, col in enumerate(te_cols_active):
            te_name = f"TE_{col}"
            X_train[te_name] = X_train_enc[:, i]
            X_val[te_name]   = X_val_enc[:, i]
            X_test_fold[te_name] = X_test_enc[:, i]
            te_feature_names.append(te_name)

        # Drop string columns
        cols_to_drop = [c for c in te_cols_active if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop, errors='ignore')
        string_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')
        X_train = X_train.fillna(0).astype('float32')
        X_val   = X_val.fillna(0).astype('float32')
        X_test_fold = X_test_fold.fillna(0).astype('float32')

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")

        # Train XGBoost depth=3
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (model.best_iteration if hasattr(model, 'best_iteration')
                     and model.best_iteration is not None else model.n_estimators)
        best_iters.append(best_iter)

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

        if fold == 0:
            imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            # Show V16 forensic features specifically
            forensic_feats = imp[imp['feature'].str.contains('bigram_Sub_HomeCharging|_recipe_sub_income|_dist_to_buyer_centroid')]
            if len(forensic_feats) > 0:
                print(f"\n      V16 FORENSIC features in top-importance:")
                print(forensic_feats.head(10).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model
        gc.collect()

    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Best iters: {best_iters}")

    # [4/5] SAVE
    print(f"\n[4/5] Saving outputs...")
    out_dir = "/kaggle/working"; os.makedirs(out_dir, exist_ok=True)
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv + sub_{CFG.VERSION_NAME}.csv")

    # [5/5] RESULTS
    print(f"\n{'='*80}")
    print(f"V16 RESULTS — XGBoost depth=3 + Forensic-Targeted Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Base: V14 XGB depth=3 (proven 0.946+ OOF)")
    print(f"NEW forensic features (5):")
    print(f"  [F1] bigram_Sub_HomeCharging (pair AUC 0.72-0.75 in ECL subgroups)")
    print(f"  [F2] _recipe_sub_income (2.4x lift in all ECL subgroups)")
    print(f"  [F3] _dist_to_buyer_centroid_ECL1 (centroid AUC 0.801)")
    print(f"  [F4] _dist_to_buyer_centroid_ECL2 (centroid AUC 0.804)")
    print(f"  [F5] _dist_to_buyer_centroid_ECL3 (centroid AUC 0.824)")
    print(f"")
    print(f"V14 baseline: 0.946+ OOF → V16 forensic: {oof_cv:.5f} OOF")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
