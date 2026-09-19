"""
S6E9 V14 - Pseudo-Labeling (Semi-Supervised Learning, GPU)
================================================================================
Strategy: XGBoost depth=3 (V12 config) + Pseudo-labels from V10's test predictions

Why V14:
- S6E3 V53-V57 proved pseudo-labeling works: "Pseudo-labeling is NOT dead.
  Conservative thresholds (p>=0.98 or p<=0.02) + half-weighting prevents
  signal corruption and adds value"
- V10 is our best model (0.94636 LB) — its test predictions are the best teacher
- V12 XGB depth=3 is our best XGBoost (0.94629 LB) — proven depth=3 > depth=7
- 286K unlabeled test rows contain signal that training doesn't

How Pseudo-labeling Works:
1. V10 (teacher) predicts test probabilities → sub_v10.csv (already saved)
2. Select high-confidence test rows:
   - p >= 0.98 → pseudo-label = 1 (likely buyer)
   - p <= 0.02 → pseudo-label = 0 (likely non-buyer)
3. Add these pseudo-labeled rows to training with HALF weight (0.5)
4. V12 (student) trains on: original train (weight=1.0) + pseudo-labeled (weight=0.5)
5. If OOF improves → submit

Why XGBoost depth=3 as student (not LGBM):
- V12 XGB depth=3 scored 0.94629 (nearly matched V10's 0.94636)
- XGBoost GPU is fast (~30 min)
- depth=3 "sharp cuts" proven for S6E9's rule-based signal
- Different model family from teacher (V10 LGBM) → more diverse

Teacher: V10's test predictions (sub_v10.csv)
Student: V12 XGB depth=3 (V3 FE + V10/V12 proven features)
Threshold: p >= 0.98 (buyer) or p <= 0.02 (non-buyer)
Pseudo-weight: 0.5 (half-weight, S6E3 proven)

FE Pipeline (V10/V12 proven):
- V3 full FE + V10 targeted bigrams + V10 selective groupby
- V10/V12 proven: trigram_Sub_ECL_RA + bigram_income_band_x_Subsidy
- sklearn TargetEncoder cv=5 (fast, V12 lesson)
- All 10 discussion patterns (except recipe_score — V11 proved cannibalization)

Device: GPU | Est. Time: ~35-40 min
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
    VERSION_NAME = "v14"
    EXP_ID = "S6E9_V14_PseudoLabel_XGB"
    DEVICE = "GPU"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # V10's test predictions (teacher) — must be uploaded as dataset or in /kaggle/working
    TEACHER_PREDS_PATH = "/kaggle/working/sub_v10.csv"

    TARGET = 'Will_Buy_EV'
    N_FOLDS = 5
    RANDOM_SEED = 42

    # Pseudo-labeling params (S6E3 V53 proven)
    PL_THRESHOLD_HIGH = 0.98   # p >= 0.98 → pseudo-label = 1
    PL_THRESHOLD_LOW = 0.02    # p <= 0.02 → pseudo-label = 0
    PL_WEIGHT = 0.5            # Half-weight (S6E3 proven)

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (V12 XGB depth=3 — proven 0.94629)
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
# 6. FEATURE ENGINEERING (V10/V12 proven pipeline)
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
        test_df = test_df.drop(columns=DROP, errors='ignore')
    return train_df, test_df, DROP

def add_targeted_bigrams(train_df, test_df):
    print("   Adding TARGETED bigram interactions...")
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
# 7. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} (XGBoost depth=3) | Folds: {CFG.N_FOLDS}")
    print(f"Teacher: V10 test predictions (sub_v10.csv)")
    print(f"Student: V12 XGB depth=3 (proven 0.94629)")
    print(f"Pseudo-labels: p>={CFG.PL_THRESHOLD_HIGH} or p<={CFG.PL_THRESHOLD_LOW}, weight={CFG.PL_WEIGHT}")
    print("="*80)

    # =========================================================================
    # [1/6] LOAD DATA + TEACHER PREDICTIONS
    # =========================================================================
    print("\n[1/6] Loading data + teacher predictions...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)
    orig[CFG.TARGET] = orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)

    # Load teacher predictions (V10's test predictions)
    if os.path.exists(CFG.TEACHER_PREDS_PATH):
        teacher_sub = pd.read_csv(CFG.TEACHER_PREDS_PATH)
        teacher_preds = teacher_sub.set_index('id').loc[test_id, CFG.TARGET].values
        print(f"   Teacher predictions loaded: {len(teacher_preds)} rows")
    else:
        print(f"   ⚠️ Teacher predictions not found at {CFG.TEACHER_PREDS_PATH}")
        print(f"   Please upload sub_v10.csv as a dataset and update TEACHER_PREDS_PATH")
        print(f"   Falling back to V12 without pseudo-labels (baseline)")
        teacher_preds = None

    # Create pseudo-labels
    if teacher_preds is not None:
        pl_high = teacher_preds >= CFG.PL_THRESHOLD_HIGH
        pl_low = teacher_preds <= CFG.PL_THRESHOLD_LOW
        n_pl_high = pl_high.sum()
        n_pl_low = pl_low.sum()
        print(f"   Pseudo-labels: {n_pl_high} high-confidence buyers (p>={CFG.PL_THRESHOLD_HIGH})")
        print(f"                  {n_pl_low} high-confidence non-buyers (p<={CFG.PL_THRESHOLD_LOW})")
        print(f"                  Total: {n_pl_high + n_pl_low} pseudo-labeled rows ({(n_pl_high+n_pl_low)/len(test)*100:.1f}% of test)")

        # Create pseudo-labeled test subset
        pl_mask = pl_high | pl_low
        test_pl = test[pl_mask].copy()
        test_pl[CFG.TARGET] = np.where(teacher_preds[pl_mask] >= CFG.PL_THRESHOLD_HIGH, 1, 0)
        print(f"   Pseudo-labeled subset shape: {test_pl.shape}")
    else:
        test_pl = pd.DataFrame()

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")

    # =========================================================================
    # [2/6] FEATURE ENGINEERING (V10/V12 proven)
    # =========================================================================
    print("\n[2/6] Feature Engineering (V10/V12 proven)...")
    train = add_digit_features(train, NUMS); test = add_digit_features(test, NUMS); orig_aligned = add_digit_features(orig_aligned, NUMS)
    if len(test_pl) > 0: test_pl = add_digit_features(test_pl, NUMS)
    train = add_engineered_features(train); test = add_engineered_features(test); orig_aligned = add_engineered_features(orig_aligned)
    if len(test_pl) > 0: test_pl = add_engineered_features(test_pl)
    train = add_synthetic_artifact_flags(train); test = add_synthetic_artifact_flags(test); orig_aligned = add_synthetic_artifact_flags(orig_aligned)
    if len(test_pl) > 0: test_pl = add_synthetic_artifact_flags(test_pl)
    train = add_smooth_keys(train); test = add_smooth_keys(test); orig_aligned = add_smooth_keys(orig_aligned)
    if len(test_pl) > 0: test_pl = add_smooth_keys(test_pl)
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    if len(test_pl) > 0:
        # Add org means to test_pl (same mapping as test)
        for col in CATS + NUMS:
            org_mean_name = f"{col}_org_mean"
            if org_mean_name in test.columns and col in test_pl.columns:
                test_pl[org_mean_name] = test_pl[col].map(
                    orig.groupby(col, observed=False)[CFG.TARGET].mean()
                ).fillna(orig[CFG.TARGET].mean()).astype('float32')
            elif org_mean_name in test.columns:
                test_pl[org_mean_name] = orig[CFG.TARGET].mean()

    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean

    all_numeric_cols = [c for c in train.columns if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c]) and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy', '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    train, train_num_cat_cols = add_numeric_as_string(train, all_numeric_cols + engineered_numeric)
    test, _ = add_numeric_as_string(test, all_numeric_cols + engineered_numeric)
    orig_aligned, _ = add_numeric_as_string(orig_aligned, all_numeric_cols + engineered_numeric)
    if len(test_pl) > 0: test_pl, _ = add_numeric_as_string(test_pl, all_numeric_cols + engineered_numeric)

    freq_target_cols = CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    if len(test_pl) > 0:
        combined_freq = pd.concat([train[freq_target_cols], test[freq_target_cols]], axis=0)
        for col in freq_target_cols:
            fm = combined_freq[col].value_counts(normalize=True).to_dict()
            test_pl[f"{col}_fe"] = test_pl[col].map(fm).astype('float32').fillna(0.0)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns: orig_aligned[f"{col}_fe"] = 0.0

    # V10/V12 proven features
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    if len(test_pl) > 0:
        test_pl['ECL_bin'] = test_pl['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
        for bn in bigram_cols:
            if 'RangeAnxiety' in bn: test_pl[bn] = test_pl['ECL_bin'] + '_' + test_pl['Range_Anxiety_Level'].astype(str)
            elif 'Subsidy' in bn: test_pl[bn] = test_pl['ECL_bin'] + '_' + test_pl['Subsidy_Available'].astype(str)
        test_pl.drop(columns=['ECL_bin'], inplace=True)
    for df in [orig_aligned]: df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bn in bigram_cols:
        if 'RangeAnxiety' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'Subsidy' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test, orig_aligned, proven_cols = add_proven_features(train, test, orig_aligned)
    if len(test_pl) > 0:
        tn = proven_cols[0]; bis = proven_cols[1]
        test_pl[tn] = (test_pl['Subsidy_Available'].astype(str) + '_' +
            test_pl['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            test_pl['Range_Anxiety_Level'].astype(str))
        bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
        labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
        test_pl['_inc_band'] = pd.cut(test_pl['Annual_Income_USD'].fillna(test_pl['Annual_Income_USD'].median()), bins=bins, labels=labels).astype(str)
        test_pl[bis] = test_pl['_inc_band'] + '_' + test_pl['Subsidy_Available'].astype(str)
        test_pl.drop(columns=['_inc_band'], inplace=True)

    # Drop helper columns
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin'], errors='ignore', inplace=True)
    if len(test_pl) > 0: test_pl.drop(columns=['ECL_bin'], errors='ignore', inplace=True)

    # Define TE columns
    TE_COLUMNS = [c for c in (CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'] + bigram_cols + proven_cols) if c in train.columns]

    # Feature selection on train+test (before adding pseudo-labels)
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')
    if len(test_pl) > 0: test_pl = test_pl.drop(columns=[c for c in dropped if c in test_pl.columns], errors='ignore')

    FEATURES = [c for c in test.columns if c != 'id']
    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   TE columns: {len(TE_COLUMNS)}")

    # =========================================================================
    # [3/6] PREPARE PSEUDO-LABELED DATA
    # =========================================================================
    print(f"\n[3/6] Preparing pseudo-labeled data...")

    # Ensure test_pl has same columns as train (minus target)
    if len(test_pl) > 0:
        # Add missing columns to test_pl
        for col in FEATURES:
            if col not in test_pl.columns:
                test_pl[col] = 0
        # Keep only FEATURES + TARGET
        test_pl = test_pl[FEATURES + [CFG.TARGET]]
        print(f"   Pseudo-labeled data ready: {test_pl.shape}")
        print(f"   Pos rate in pseudo-labels: {test_pl[CFG.TARGET].mean():.4f}")
    else:
        print(f"   No pseudo-labels (teacher predictions not available)")

    # =========================================================================
    # [4/6] TRAINING (5-Fold KFold with pseudo-labels + per-fold orig concat)
    # =========================================================================
    print(f"\n[4/6] Training XGBoost depth=3 ({CFG.N_FOLDS}-Fold CV, pseudo-labels + orig concat)...")

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

        # Add pseudo-labeled test rows (with half weight)
        if len(test_pl) > 0:
            X_train = pd.concat([X_train, test_pl[FEATURES]], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train, test_pl[CFG.TARGET]], axis=0).reset_index(drop=True)
            # Sample weights: 1.0 for original, 0.5 for pseudo-labeled
            n_orig = len(train_idx) + len(or_train_idx)
            n_pl = len(test_pl)
            weights = np.concatenate([
                np.ones(n_orig, dtype='float32'),
                np.ones(n_pl, dtype='float32') * CFG.PL_WEIGHT
            ])
        else:
            weights = np.ones(len(y_train), dtype='float32')

        X_test_fold = test_X.copy()
        print(f"      Train (comp+orig+PL): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")
        print(f"      Weights: {n_orig} full + {n_pl} half = {len(weights)}")

        # sklearn TargetEncoder (V12 lesson: fast, cv=5)
        # Filter TE_COLUMNS to only columns that exist in X_train (after concat)
        te_cols_active = [c for c in TE_COLUMNS if c in X_train.columns]
        te_feature_names = []
        for smooth_val, smooth_name in [('auto', 'auto')]:
            te = TargetEncoder(target_type='binary', smooth=smooth_val,
                               cv=CFG.N_FOLDS, shuffle=True, random_state=42)
            X_train_enc = te.fit_transform(X_train[te_cols_active], y_train).astype('float32')
            X_val_enc   = te.transform(X_val[te_cols_active]).astype('float32')
            X_test_enc  = te.transform(X_test_fold[te_cols_active]).astype('float32')
            for i, col in enumerate(te_cols_active):
                te_name = f"TE_{col}"
                X_train[te_name] = X_train_enc[:, i]
                X_val[te_name]   = X_val_enc[:, i]
                X_test_fold[te_name] = X_test_enc[:, i]
                if te_name not in te_feature_names:
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

        # Train XGBoost depth=3 (V12 config)
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (model.best_iteration if hasattr(model, 'best_iteration')
                     and model.best_iteration is not None else model.n_estimators)
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        if fold == 0:
            imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            print(f"\n      Top-10 feature importances (fold 1):")
            print(imp.head(10).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model, weights
        gc.collect()

    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Best iters: {best_iters}")

    # =========================================================================
    # [5/6] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[5/6] Saving outputs...")
    out_dir = "/kaggle/working"; os.makedirs(out_dir, exist_ok=True)
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv + sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [6/6] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V14 RESULTS — Pseudo-Labeling + XGBoost depth=3 ({CFG.DEVICE})")
    print(f"{'='*80}")
    if len(test_pl) > 0:
        print(f"Teacher: V10 (0.94636 LB)")
        print(f"Student: V12 XGB depth=3 (0.94629 LB)")
        print(f"Pseudo-labels: {n_pl_high + n_pl_low} rows ({(n_pl_high+n_pl_low)/len(test)*100:.1f}% of test)")
        print(f"  - Buyers (p>={CFG.PL_THRESHOLD_HIGH}): {n_pl_high}")
        print(f"  - Non-buyers (p<={CFG.PL_THRESHOLD_LOW}): {n_pl_low}")
        print(f"  - Weight: {CFG.PL_WEIGHT} (half-weight, S6E3 proven)")
    else:
        print(f"No pseudo-labels (teacher not available) — baseline V12 rerun")
    print(f"V12 baseline: 0.94629 LB → V14 pseudo-label: {oof_cv:.5f} OOF")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
