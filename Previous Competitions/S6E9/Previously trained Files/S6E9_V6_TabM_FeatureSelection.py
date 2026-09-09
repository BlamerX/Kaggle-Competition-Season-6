"""
S6E9 V6 - TabM Baseline with Feature Selection (GPU - pytabkit)
================================================================================
Strategy: TabM with EVIDENCE-BASED Feature Selection + StandardScaler
          + Original Data (per-fold concat)

References:
- S6E4 V8 TabM Baseline (154 -> 76 features after selection; d_block=128)
- S6E4 V36 TabTransformer (LB 0.97752; projection architecture: TE branch + numeric branch)
- TabM paper (Yandex 2024): d_block=256 for 160 features
- V5 LogisticRegression coefficients (feature importance for selection)
- V3 LightGBM feature_importance (feature importance for selection)

V6 Design (Evidence-Based — NOT raw 344 features):
1. Feature Selection using V5 LR coefficients + V3 LightGBM importance
   - Keep top features by |coefficient| (V5) and feature_importance (V3)
   - Target: ~80-120 features (not 344, not 50-70)
2. Drop redundant features:
   - Most digit features (V5 shows low importance)
   - Most _cat columns (redundant with raw numerics)
   - TE on original cats (V5 low importance)
   - TE smooth=10 and smooth=100 (keep only smooth=auto to reduce redundancy)
3. TabM with d_block=256 (TabM paper uses this for 160 features)
4. StandardScaler (TabM expects normalized numericals)
5. KFold (per S6E4 V8 pattern)
6. Per-fold orig concat

Why V6 Redesigned (V6-as-original would underperform):
- V5 LR coefficients show most digit features, _cat columns, and Triple TE on original cats
  have LOW importance — feeding these to TabM wastes capacity
- S6E4 V36 (only NN with confirmed Kaggle score, LB 0.97752) used projection architecture
  to compress 351 TE features — did NOT feed them directly
- TabM paper uses d_block=256 for 160 features; V6-as-original had 344 features with d_block=128
  (2x features with half the capacity = overfitting risk)
- Zero S6E9 public NN notebooks exist — we're in unknown territory; evidence-based design critical

V5 LR Top-15 Features (by |coefficient|) — GUIDE for V6 selection:
1. Subsidy_Available_fe (1.955) — frequency encoding
2. ECL_org_mean (1.831) — original target mean
3. ECL_cat_fe (1.538) — numeric-as-string + freq
4. _Income_x_Subsidy_cat_fe (0.934) — engineered + freq
5. Annual_Income_USD_cat_fe (0.605) — numeric-as-string + freq
6. TE_income1000_floor_100 (0.593) — TE on smooth key
7. _ecl_max (0.401) — hard-edge flag
8. City_Type_org_mean (0.374) — original target mean
9. TE_income1000_floor_10 (0.367) — TE on smooth key
10. Annual_Income_USD_org_mean (0.336) — original target mean
11. _ECL_x_Subsidy_cat_fe (0.307) — engineered + freq
12. income100_floor_fe (0.301) — smooth key + freq
13. income1000_floor_fe (0.242) — smooth key + freq
14. TE__Income_x_Subsidy_cat_auto (0.239) — TE on engineered
15. TE__log_Income_cat_auto (0.236) — TE on log

V3 LightGBM Top-15 Features (by splits) — CONFIRMS V5 findings:
1. TE_income100_floor_auto (949 splits)
2. TE__log_Income_cat_auto (799)
3. TE_income100_floor_10 (790)
4. TE_income100_floor_100 (652)
5. TE__Income_x_Subsidy_cat_auto (639)
6. Annual_Income_USD (579)
7. _ECL_x_RangeAnxiety (562)
8. TE_Annual_Income_USD_cat_auto (561)
9. _Income_x_Subsidy (527)
10. TE__log_Income_cat_100 (506)
11. income100_floor_fe (492)
12. TE_Annual_Income_USD_cat_10 (489)
13. TE_Age_cat_auto (477)
14. TE__Income_x_Subsidy_cat_100 (467)
15. Environmental_Concern_Level_freq (359) [estimated]

Feature Selection Strategy (Intersection of V5 + V3 top features):
KEEP:
- All 13 original raw features (Age, Annual_Income, Daily_Commute, etc.)
- All 7 engineered interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety, etc.)
- All 4 hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
- All 4 magic flags (is_30k_spike, is_millionaire_cliff, is_dead_zone, is_env_hater)
- All 4 smooth keys (income_exact_int, income100_floor, income1000_floor, commute_integer)
- All 13 original target means ({col}_org_mean)
- Top frequency features (Subsidy_Available_fe, ECL_cat_fe, _Income_x_Subsidy_cat_fe,
  Annual_Income_USD_cat_fe, income100_floor_fe, income1000_floor_fe)
- Top TE features (TE_income100_floor_*, TE_income1000_floor_*, TE__Income_x_Subsidy_cat_*,
  TE__log_Income_cat_*, TE_Annual_Income_USD_cat_*, TE_Age_cat_*)

DROP:
- All 56 digit features (V5 shows low importance; NN learns via PWL from raw)
- Most _cat columns (redundant with raw numerics — keep only those with top _fe)
- TE on original cats (low importance in V5; native handling not available in TabM)
- TE smooth=10 and smooth=100 for MOST cols (keep only auto to reduce redundancy;
  EXCEPTION: keep TE_income100_floor_10, TE_income100_floor_100, TE_income1000_floor_10,
  TE_income1000_floor_100 — these are top features in both V5 and V3)

Expected: OOF ~0.944-0.946, LB ~0.945-0.947
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import gc
import sys
import subprocess
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# Auto-install pytabkit
try:
    from pytabkit import TabM_D_Classifier
    print("pytabkit loaded successfully!")
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("pytabkit installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v6"
    EXP_ID = "S6E9_V6_TabM_FeatureSelection"
    DEVICE = DEVICE

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — KFold (per S6E4 V8 pattern)
    N_FOLDS = 5
    RANDOM_SEED = 42

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (TabM paper specs — d_block=256 for ~100 features)
# =============================================================================
# TabM with BatchEnsemble (tabm_k=16 independent prediction heads)
# + PWL (Piecewise Linear) numerical embeddings
#
# Key params (evidence-based):
#   d_block=256: TabM paper uses this for 160 features (Aloi dataset)
#   tabm_k=16: 16 BatchEnsemble heads (reduced from 32 for GPU memory)
#   d_embedding=16: PWL embedding dimension
#   n_epochs=30: early stopping handles convergence
#   batch_size=1024: safe middle ground for GPU memory

TABM_PARAMS = {
    'device': DEVICE,
    'verbosity': 0,
    # Architecture (TabM paper specs for ~100 features)
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 16,                    # 16 BatchEnsemble heads
    'num_emb_type': 'pwl',           # Piecewise Linear numerical embeddings
    'd_embedding': 16,               # PWL embedding dimension
    'd_block': 256,                  # Hidden layer size (INCREASED from 128 per TabM paper)
    'n_blocks': 3,                   # Number of hidden blocks
    'dropout': 0.2,                  # Dropout rate
    # Training
    'batch_size': 1024,              # Safe middle ground for GPU memory
    'lr': 1e-3,                      # Adam learning rate
    'n_epochs': 30,                  # Early stopping handles convergence
    'patience': 10,                  # Early stopping patience
    'weight_decay': 1e-3,            # L2 regularization
    'random_state': CFG.RANDOM_SEED,
}

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (Same as V3/V4/V5 — then SELECT for NN)
# =============================================================================
def add_digit_features(df, num_cols):
    """Digit Feature Extraction - 8 features per numerical column."""
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    """EDA-validated engineered features (kept from V1/V2/V3/V4/V5)."""
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


def select_features_for_nn(train_df, test_df, orig_df, cat_cols, num_cols, target):
    """
    EVIDENCE-BASED Feature Selection for NN (TabM).

    Uses V5 LR coefficients + V3 LightGBM importance to select ~80-120 features.

    KEEP:
    - All 13 original raw features
    - All 7 engineered interactions + logs
    - All 4 hard-edge flags
    - All 4 magic flags
    - All 4 smooth keys
    - All 13 original target means (_org_mean)
    - Top frequency features (_fe) — V5 shows these matter
    - Top TE features (auto smoothing + select 10/100 for smooth keys only)

    DROP:
    - All 56 digit features (V5 low importance; NN learns via PWL from raw)
    - Most _cat columns (redundant with raw numerics)
    - TE on original cats (V5 low importance)
    - TE smooth=10 and smooth=100 for MOST cols (keep auto; exception: smooth keys)
    """
    print("   Selecting features for NN (evidence-based)...")

    # Define feature groups to KEEP
    keep_features = set()

    # 1. All original raw features (13)
    keep_features.update(num_cols)
    keep_features.update(cat_cols)

    # 2. All engineered interactions + logs (7)
    engineered = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
                  '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    keep_features.update([c for c in engineered if c in train_df.columns])

    # 3. All hard-edge flags (4)
    flags = ['_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe']
    keep_features.update([c for c in flags if c in train_df.columns])

    # 4. All magic flags (4)
    magic = ['is_30k_spike', 'is_millionaire_cliff', 'is_dead_zone', 'is_env_hater']
    keep_features.update([c for c in magic if c in train_df.columns])

    # 5. All smooth keys (4) — V5 and V3 both show these matter
    smooth_keys = ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    keep_features.update([c for c in smooth_keys if c in train_df.columns])

    # 6. All original target means (13) — V5 shows these matter
    org_means = [c for c in train_df.columns if c.endswith('_org_mean')]
    keep_features.update(org_means)

    # 7. Top frequency features (_fe) — V5 shows these dominate
    # Keep _fe for: original cats, smooth keys, top engineered interactions, top numerics
    top_fe_cols = [
        'Subsidy_Available_fe',           # V5 #1 (1.955)
        'Environmental_Concern_Level_cat_fe',  # V5 #3 (1.538)
        '_Income_x_Subsidy_cat_fe',       # V5 #4 (0.934)
        'Annual_Income_USD_cat_fe',       # V5 #5 (0.605)
        'income100_floor_fe',             # V5 #12 (0.301)
        'income1000_floor_fe',            # V5 #13 (0.242)
        'income_exact_int_fe',
        'commute_integer_fe',
        'Gender_fe', 'City_Type_fe', 'Current_Car_Type_fe',
        'Home_Charging_Possible_fe', 'Range_Anxiety_Level_fe',
        '_ECL_x_Subsidy_cat_fe',          # V5 #11 (0.307)
        '_ECL_x_RangeAnxiety_cat_fe',
        '_log_Income_cat_fe',
        '_log_Commute_cat_fe',
        '_log_Charging_Total_cat_fe',
        '_Charging_Total_cat_fe',
        'Age_cat_fe',
        'Daily_Commute_km_cat_fe',
        'Number_of_Cars_Owned_cat_fe',
        'Charging_Stations_Near_Home_cat_fe',
        'Charging_Stations_Near_Work_cat_fe',
    ]
    keep_features.update([c for c in top_fe_cols if c in train_df.columns])

    # 8. Top TE features — V3 LightGBM top + V5 LR top
    # Keep TE with auto smoothing for ALL cols (V3 shows auto is most useful)
    # Keep TE with 10 and 100 smoothing ONLY for smooth keys (V3 top features)
    top_te_cols = [
        # Smooth keys TE (all 3 smoothings — V3 top features)
        'TE_income100_floor_auto',    # V3 #1 (949 splits)
        'TE_income100_floor_10',      # V3 #3 (790)
        'TE_income100_floor_100',     # V3 #4 (652)
        'TE_income1000_floor_auto',
        'TE_income1000_floor_10',     # V5 #9 (0.367)
        'TE_income1000_floor_100',    # V5 #6 (0.593)
        'TE_income_exact_int_auto',
        'TE_commute_integer_auto',
        # Engineered interactions TE (auto only — V3/V5 top)
        'TE__Income_x_Subsidy_cat_auto',   # V3 #5 (639), V5 #14 (0.239)
        'TE__log_Income_cat_auto',         # V3 #2 (799), V5 #15 (0.236)
        'TE__ECL_x_Subsidy_cat_auto',
        'TE__ECL_x_RangeAnxiety_cat_auto',
        'TE__log_Commute_cat_auto',
        'TE__log_Charging_Total_cat_auto',
        'TE__Charging_Total_cat_auto',
        # Top numeric-as-string TE (auto only)
        'TE_Annual_Income_USD_cat_auto',   # V3 #8 (561)
        'TE_Age_cat_auto',                 # V3 #13 (477)
        'TE_Environmental_Concern_Level_cat_auto',
        'TE_Daily_Commute_km_cat_auto',
        'TE_Number_of_Cars_Owned_cat_auto',
        'TE_Charging_Stations_Near_Home_cat_auto',
        'TE_Charging_Stations_Near_Work_cat_auto',
    ]
    keep_features.update([c for c in top_te_cols if c in train_df.columns])

    # Filter to features that exist in both train and test
    keep_features = [c for c in keep_features
                     if c in train_df.columns and c in test_df.columns]

    # Also keep these in orig_df if they exist (for per-fold concat)
    keep_features_orig = [c for c in keep_features if c in orig_df.columns]

    print(f"      Keeping {len(keep_features)} features for NN (out of {len(train_df.columns) - 1})")

    return keep_features, keep_features_orig


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
    print(f"Device: {CFG.DEVICE} (TabM) | Folds: {CFG.N_FOLDS}")
    print(f"Feature Selection: V5 LR + V3 LGB importance (evidence-based)")
    print(f"TabM: tabm_k=16, d_block=256, PWL embeddings (TabM paper specs)")
    print(f"Original data: USED (per-fold concat)")
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
    # [2/5] FEATURE ENGINEERING + SELECTION
    # =========================================================================
    print("\n[2/5] Feature Engineering + Selection...")

    # Step 1: Full FE pipeline (same as V3/V4/V5)
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

    # Step 2: Add Triple TE (auto, 10, 100) on all cols (will select subset later)
    print("   Adding Triple TE (auto, 10, 100) on all cat + num-as-string cols...")
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]) if c in train.columns]

    # Compute TE on full train (not per-fold yet — we'll do per-fold in training loop)
    # For feature SELECTION, use full-train TE to identify which TE cols matter
    # Then in training loop, recompute TE per-fold for selected cols only
    # This is a 2-stage approach: select features first, then per-fold TE on selected

    # For now, add TE columns to train/test using full-train fit (for selection only)
    # We'll redo per-fold in the training loop
    te_cols_for_selection = TARGET_ENCODE_COLS
    te_features_for_selection = []
    for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
        te = TargetEncoder(target_type='binary', smooth=smooth_val,
                           cv=CFG.N_FOLDS, shuffle=True, random_state=42)
        train_enc = te.fit_transform(train[te_cols_for_selection], train[CFG.TARGET]).astype('float32')
        test_enc  = te.transform(test[te_cols_for_selection]).astype('float32')

        for i, col in enumerate(te_cols_for_selection):
            te_name = f"TE_{col}_{smooth_name}"
            train[te_name] = train_enc[:, i]
            test[te_name]  = test_enc[:, i]
            te_features_for_selection.append(te_name)

    print(f"   Added {len(te_features_for_selection)} TE features for selection")

    # Step 3: Feature selection (drop constants + corr=1)
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Step 4: EVIDENCE-BASED feature selection for NN
    selected_features, selected_features_orig = select_features_for_nn(
        train, test, orig_aligned, CATS, NUMS, CFG.TARGET
    )

    # Filter train/test/orig to selected features + target
    train = train[selected_features + [CFG.TARGET]]
    test  = test[selected_features]
    # orig_aligned: keep only selected features that exist in orig
    orig_keep = [c for c in selected_features if c in orig_aligned.columns]
    orig_aligned = orig_aligned[orig_keep]

    print(f"\n   Final selected features: {len(selected_features)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold KFold with per-fold orig concat + per-fold TE)
    # =========================================================================
    print(f"\n[3/5] Training TabM ({CFG.N_FOLDS}-Fold KFold, orig concat + per-fold TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []

    # KFold (per S6E4 V8 pattern)
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    # Identify which selected features need per-fold TE
    # (those starting with TE_ — we'll recompute them per-fold)
    te_feature_prefix = "TE_"
    te_cols_in_selected = [c for c in selected_features if c.startswith(te_feature_prefix)]
    non_te_features = [c for c in selected_features if not c.startswith(te_feature_prefix)]

    # Map TE feature names back to (source_col, smooth_name)
    # TE_{col}_{smooth_name} -> (col, smooth_name)
    te_col_map = {}
    for te_feat in te_cols_in_selected:
        # Parse: TE_<col>_<smooth> where smooth in {auto, 10, 100}
        parts = te_feat.split('_')
        # smooth is last part
        smooth_name = parts[-1]
        # col is everything between TE_ and _<smooth>
        col = '_'.join(parts[1:-1])
        te_col_map[te_feat] = (col, smooth_name)

    # Unique source cols that need TE
    te_source_cols = list(set([col for col, _ in te_col_map.values()]))
    print(f"   TE source cols (per-fold): {len(te_source_cols)}")
    print(f"   Non-TE features: {len(non_te_features)}")

    t0 = time.time()
    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train = X.iloc[train_idx].copy()
        X_val   = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val   = y.iloc[val_idx]

        # Per-fold: concat competition train + original
        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        # orig_aligned may not have TE cols — we'll compute them per-fold
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- Per-fold Triple TE on selected TE source cols ----
        # Recompute TE features per-fold to prevent leakage
        for te_feat in te_cols_in_selected:
            col, smooth_name = te_col_map[te_feat]
            if col not in X_train.columns:
                continue

            # Map smooth_name to smooth_val
            if smooth_name == 'auto':
                smooth_val = 'auto'
            else:
                smooth_val = float(smooth_name)

            te = TargetEncoder(target_type='binary', smooth=smooth_val,
                               cv=CFG.N_FOLDS, shuffle=True, random_state=42)
            # fit on train, transform val and test
            X_train[te_feat] = te.fit_transform(
                X_train[[col]], y_train
            ).ravel().astype('float32')
            X_val[te_feat] = te.transform(X_val[[col]]).ravel().astype('float32')
            X_test_fold[te_feat] = te.transform(X_test_fold[[col]]).ravel().astype('float32')

        # Drop ALL string columns (TabM needs numeric only)
        string_cols = [c for c in X_train.columns
                       if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')

        # Fill NaN and convert to float32 (saves ~50% GPU memory vs float64)
        X_train = X_train.fillna(0).astype('float32')
        X_val   = X_val.fillna(0).astype('float32')
        X_test_fold = X_test_fold.fillna(0).astype('float32')

        # Normalize column names to string
        X_train.columns = X_train.columns.astype(str)
        X_val.columns = X_val.columns.astype(str)
        X_test_fold.columns = X_test_fold.columns.astype(str)

        if fold == 0:
            print(f"      Final feature count (numeric only): {len(X_train.columns)}")

        # ---- StandardScaler (TabM expects normalized numericals) ----
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        X_test_sc  = scaler.transform(X_test_fold)

        X_train_df = pd.DataFrame(X_train_sc, columns=X_train.columns, index=X_train.index)
        X_val_df = pd.DataFrame(X_val_sc, columns=X_val.columns, index=X_val.index)
        X_test_df = pd.DataFrame(X_test_sc, columns=X_test_fold.columns, index=X_test_fold.index)

        # ---- Train TabM ----
        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_train_df, y_train, X_val=X_val_df, y_val=y_val)

        # Predictions
        val_probs = model.predict_proba(X_val_df)
        if len(val_probs.shape) == 2:
            val_probs = val_probs[:, 1]
        oof_probs[val_idx] = val_probs

        test_probs_fold = model.predict_proba(X_test_df)
        if len(test_probs_fold.shape) == 2:
            test_probs_fold = test_probs_fold[:, 1]
        test_probs += test_probs_fold / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test_fold, X_train_df, X_val_df, X_test_df, y_train, y_val, model, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    print(f"V6 RESULTS — TabM with Feature Selection ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Selected features: {len(selected_features)} (from 344 raw)")
    print(f"  - Original raw: 13")
    print(f"  - Engineered interactions + logs: 7")
    print(f"  - Hard-edge flags: 4")
    print(f"  - Magic flags: 4")
    print(f"  - Smooth keys: 4")
    print(f"  - Original target means: 13")
    print(f"  - Frequency features (_fe): top ~20")
    print(f"  - TE features (auto + select 10/100): top ~25")
    print(f"  - Dropped: 56 digit features, most _cat cols, TE on original cats")
    print(f"TabM: tabm_k=16, d_block=256, PWL embeddings (TabM paper specs)")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
