"""
S6E9 V7 Option B - FT-Transformer on Full Feature Set (PyTorch/GPU)
================================================================================
Strategy: FT-Transformer (Yandex rtdl) on V6's selected features (~80 features)

Why Option B (not original V7 with 10 formula features):
- Original V7 (10 features) scored 0.93849 — too few tokens for self-attention
- FT-Transformer needs more feature tokens to learn cross-feature interactions
- V6 TabM proved NN works with ~83 features (OOF 0.94585 / LB 0.94606)
- This version uses V6's feature selection pipeline + FT-Transformer architecture

Architecture:
  Feature Tokenizer:
    - Categorical features -> Embedding(cardinality, 64) per feature
    - Numerical features -> Linear(1, 64) per feature
    -> N tokens of dim 64
  Transformer blocks (×2):
    - Pre-norm
    - Multi-head attention (4 heads)
    - FFN with ReGLU
    - Dropout
  CLS token -> Linear -> Sigmoid -> binary output

Feature Pipeline (same as V6):
- Digit features (8 per numeric) — then drop low-importance ones
- Engineered interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety, etc.)
- Hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
- Synthetic-artifact flags (is_30k_spike, is_millionaire_cliff, etc.)
- Multi-Scale Smooth Keys
- Original dataset target means
- Frequency encoding on selected cols
- Triple TE (auto only — reduce redundancy for NN)
- Feature selection (V5 LR + V3 LGB importance)

Key Difference from V6 TabM:
- V6: MLP + BatchEnsemble + PWL embeddings (no attention)
- V7: Feature Tokenizer + Self-Attention (true transformer)
- This gives MAXIMUM algorithm diversity from V6 for ensemble

Training: AdamW(lr=1e-4, weight_decay=1e-5), CosineAnnealingLR,
          BCEWithLogitsLoss with pos_weight, batch_size=2048

Device: GPU | Est. Time: ~40-60 min
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# Auto-install rtdl-revisiting-models
try:
    import rtdl_revisiting_models as rtdl
    print("rtdl_revisiting_models loaded successfully!")
except ImportError:
    print("Installing rtdl-revisiting-models...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rtdl-revisiting-models", "-q"])
    import rtdl_revisiting_models as rtdl
    print("rtdl-revisiting-models installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
print(f"scikit-learn version: {sklearn_version}")
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v7"
    EXP_ID = "S6E9_V7_FTTransformer_FullFeatures"
    DEVICE = DEVICE
    N_FOLDS = 5
    RANDOM_SEED = 42
    TARGET = 'Will_Buy_EV'

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # FT-Transformer hyperparams (larger config for ~80 features)
    D_BLOCK = 128                # Increased from 64 (more features need more capacity)
    N_BLOCKS = 2
    ATTENTION_N_HEADS = 8        # Increased from 4 (more features to attend across)
    ATTENTION_DROPOUT = 0.2
    FFN_DROPOUT = 0.1

    # Training
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 1024              # Reduced from 2048 to avoid OOM (69 tokens × 128 dim)
    MAX_EPOCHS = 50              # Reduced from 100 (more features converge faster)
    ES_PATIENCE = 10
    PRED_BATCH_SIZE = 8192       # Batch size for validation/test predictions

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
# 4. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 5. FEATURE ENGINEERING (Same as V6 — proven pipeline + selection)
# =============================================================================
def add_digit_features(df, num_cols):
    """Digit Feature Extraction - 8 features per numerical column."""
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    """EDA-validated engineered features (kept from V1/V2/V3/V4/V5/V6)."""
    df = df.copy()

    # Fill NaN on source columns (orig has 543 intentional NaN)
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
    EVIDENCE-BASED Feature Selection for NN (same as V6).
    Uses V5 LR coefficients + V3 LightGBM importance to select ~80 features.
    """
    print("   Selecting features for NN (evidence-based)...")

    keep_features = set()

    # 1. All original raw features
    keep_features.update(num_cols)
    keep_features.update(cat_cols)

    # 2. All engineered interactions + logs
    engineered = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
                  '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    keep_features.update([c for c in engineered if c in train_df.columns])

    # 3. All hard-edge flags
    flags = ['_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe']
    keep_features.update([c for c in flags if c in train_df.columns])

    # 4. All magic flags
    magic = ['is_30k_spike', 'is_millionaire_cliff', 'is_dead_zone', 'is_env_hater']
    keep_features.update([c for c in magic if c in train_df.columns])

    # 5. All smooth keys
    smooth_keys = ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    keep_features.update([c for c in smooth_keys if c in train_df.columns])

    # 6. All original target means
    org_means = [c for c in train_df.columns if c.endswith('_org_mean')]
    keep_features.update(org_means)

    # 7. Top frequency features (_fe)
    top_fe_cols = [
        'Subsidy_Available_fe', 'Environmental_Concern_Level_cat_fe',
        '_Income_x_Subsidy_cat_fe', 'Annual_Income_USD_cat_fe',
        'income100_floor_fe', 'income1000_floor_fe',
        'income_exact_int_fe', 'commute_integer_fe',
        'Gender_fe', 'City_Type_fe', 'Current_Car_Type_fe',
        'Home_Charging_Possible_fe', 'Range_Anxiety_Level_fe',
        '_ECL_x_Subsidy_cat_fe', '_ECL_x_RangeAnxiety_cat_fe',
        '_log_Income_cat_fe', '_log_Commute_cat_fe',
        '_log_Charging_Total_cat_fe', '_Charging_Total_cat_fe',
        'Age_cat_fe', 'Daily_Commute_km_cat_fe',
        'Number_of_Cars_Owned_cat_fe',
        'Charging_Stations_Near_Home_cat_fe',
        'Charging_Stations_Near_Work_cat_fe',
    ]
    keep_features.update([c for c in top_fe_cols if c in train_df.columns])

    # 8. Top TE features (auto smoothing only — reduce redundancy for NN)
    top_te_cols = [
        'TE_income100_floor_auto', 'TE_income1000_floor_auto',
        'TE_income_exact_int_auto', 'TE_commute_integer_auto',
        'TE__Income_x_Subsidy_cat_auto', 'TE__log_Income_cat_auto',
        'TE__ECL_x_Subsidy_cat_auto', 'TE__ECL_x_RangeAnxiety_cat_auto',
        'TE__log_Commute_cat_auto', 'TE__log_Charging_Total_cat_auto',
        'TE__Charging_Total_cat_auto',
        'TE_Annual_Income_USD_cat_auto', 'TE_Age_cat_auto',
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

    keep_features_orig = [c for c in keep_features if c in orig_df.columns]

    print(f"      Keeping {len(keep_features)} features for NN")

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
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: FT-Transformer (d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS})")
    print(f"Feature Selection: V5 LR + V3 LGB importance (same as V6)")
    print(f"BREAKS Algorithm Lock: Self-attention vs MLP (V6 TabM) vs trees")
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
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")
    print(f"   Pos rate (orig):  {y_orig.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING + SELECTION
    # =========================================================================
    print("\n[2/5] Feature Engineering + Selection...")

    # Step 1: Full FE pipeline (same as V3/V4/V5/V6)
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

    # Step 2: Add Triple TE (auto only — reduce redundancy for NN)
    print("   Adding TE (auto smoothing only — reduce redundancy for NN)...")
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]) if c in train.columns]

    te_features_for_selection = []
    te = TargetEncoder(target_type='binary', smooth='auto',
                       cv=CFG.N_FOLDS, shuffle=True, random_state=42)
    train_enc = te.fit_transform(train[TARGET_ENCODE_COLS], train[CFG.TARGET]).astype('float32')
    test_enc  = te.transform(test[TARGET_ENCODE_COLS]).astype('float32')

    for i, col in enumerate(TARGET_ENCODE_COLS):
        te_name = f"TE_{col}_auto"
        train[te_name] = train_enc[:, i]
        test[te_name]  = test_enc[:, i]
        te_features_for_selection.append(te_name)

    print(f"   Added {len(te_features_for_selection)} TE features (auto only)")

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
    orig_keep = [c for c in selected_features if c in orig_aligned.columns]
    orig_aligned = orig_aligned[orig_keep]

    print(f"\n   Final selected features: {len(selected_features)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold KFold with per-fold orig concat + per-fold TE)
    # =========================================================================
    print(f"\n[3/5] Training FT-Transformer ({CFG.N_FOLDS}-Fold KFold, orig concat + per-fold TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    # Identify TE features (need per-fold recompute)
    te_cols_in_selected = [c for c in selected_features if c.startswith("TE_")]
    non_te_features = [c for c in selected_features if not c.startswith("TE_")]

    # Map TE feature names to source cols
    te_col_map = {}
    for te_feat in te_cols_in_selected:
        parts = te_feat.split('_')
        # TE_<col>_auto -> col
        col = '_'.join(parts[1:-1])
        te_col_map[te_feat] = col

    te_source_cols = list(set(te_col_map.values()))
    print(f"   TE source cols (per-fold): {len(te_source_cols)}")
    print(f"   Non-TE features: {len(non_te_features)}")

    # Class weight for imbalanced binary classification
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
    print(f"   pos_weight: {pos_weight.item():.4f}")

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
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- Per-fold TE on selected TE source cols ----
        for te_feat in te_cols_in_selected:
            col = te_col_map[te_feat]
            if col not in X_train.columns:
                continue
            te = TargetEncoder(target_type='binary', smooth='auto',
                               cv=CFG.N_FOLDS, shuffle=True, random_state=42)
            X_train[te_feat] = te.fit_transform(X_train[[col]], y_train).ravel().astype('float32')
            X_val[te_feat] = te.transform(X_val[[col]]).ravel().astype('float32')
            X_test_fold[te_feat] = te.transform(X_test_fold[[col]]).ravel().astype('float32')

        # Drop ALL string columns (FT-Transformer needs numeric)
        # We'll treat original CATS as categorical via cat_cardinalities
        # But first, identify which columns are categorical (low cardinality integers)
        string_cols = [c for c in X_train.columns
                       if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')

        # Fill NaN and convert to float32
        X_train = X_train.fillna(0).astype('float32')
        X_val   = X_val.fillna(0).astype('float32')
        X_test_fold = X_test_fold.fillna(0).astype('float32')

        # Identify categorical features (low cardinality integer-like)
        # Treat features with < 20 unique values as categorical
        cat_features_idx = []
        num_features_idx = []
        for i, col in enumerate(X_train.columns):
            n_unique = X_train[col].nunique()
            if n_unique < 20:
                cat_features_idx.append(i)
            else:
                num_features_idx.append(i)

        print(f"      Categorical features: {len(cat_features_idx)} (cardinality < 20)")
        print(f"      Numerical features: {len(num_features_idx)}")

        # Get cardinalities for categorical features
        cat_cardinalities = []
        for i in cat_features_idx:
            col = X_train.columns[i]
            # +1 for safety (unseen values in val/test)
            card = int(X_train[col].max()) + 2
            cat_cardinalities.append(card)

        # Convert to numpy arrays
        x_num_train = X_train.iloc[:, num_features_idx].values.astype(np.float32)
        x_num_val   = X_val.iloc[:, num_features_idx].values.astype(np.float32)
        x_num_test  = X_test_fold.iloc[:, num_features_idx].values.astype(np.float32)

        x_cat_train = X_train.iloc[:, cat_features_idx].values.astype(np.int64)
        x_cat_val   = X_val.iloc[:, cat_features_idx].values.astype(np.int64)
        x_cat_test  = X_test_fold.iloc[:, cat_features_idx].values.astype(np.int64)

        # Clip categorical values to valid range
        for i, card in enumerate(cat_cardinalities):
            x_cat_train[:, i] = np.clip(x_cat_train[:, i], 0, card - 1)
            x_cat_val[:, i] = np.clip(x_cat_val[:, i], 0, card - 1)
            x_cat_test[:, i] = np.clip(x_cat_test[:, i], 0, card - 1)

        # StandardScaler on numerical features (per-fold)
        scaler = StandardScaler()
        x_num_train = scaler.fit_transform(x_num_train)
        x_num_val   = scaler.transform(x_num_val)
        x_num_test  = scaler.transform(x_num_test)

        # Convert to tensors
        x_num_train_t = torch.tensor(x_num_train, dtype=torch.float32)
        x_num_val_t   = torch.tensor(x_num_val, dtype=torch.float32)
        x_num_test_t  = torch.tensor(x_num_test, dtype=torch.float32)

        x_cat_train_t = torch.tensor(x_cat_train, dtype=torch.long)
        x_cat_val_t   = torch.tensor(x_cat_val, dtype=torch.long)
        x_cat_test_t  = torch.tensor(x_cat_test, dtype=torch.long)

        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        y_val_np  = y_val.values.astype(int)

        if fold == 0:
            print(f"      Total features: {len(num_features_idx)} num + {len(cat_features_idx)} cat = {len(X_train.columns)}")

        # DataLoader
        train_ds = TensorDataset(x_num_train_t, x_cat_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)

        # Build FT-Transformer
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        backbone_kwargs = rtdl.FTTransformer.get_default_kwargs(n_blocks=CFG.N_BLOCKS)
        backbone_kwargs['d_block'] = CFG.D_BLOCK
        backbone_kwargs['attention_n_heads'] = CFG.ATTENTION_N_HEADS
        backbone_kwargs['attention_dropout'] = CFG.ATTENTION_DROPOUT
        backbone_kwargs['ffn_dropout'] = CFG.FFN_DROPOUT

        model = rtdl.FTTransformer(
            n_cont_features=len(num_features_idx),
            cat_cardinalities=cat_cardinalities,
            d_out=1,  # Binary: single logit
            **backbone_kwargs,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"      FT-Transformer params: {n_params/1e3:.1f}K")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.MAX_EPOCHS)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            model.train()
            for x_num, x_cat, y_batch in train_loader:
                x_num, x_cat, y_batch = x_num.to(DEVICE), x_cat.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_cont=x_num, x_cat=x_cat).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation (batched to avoid OOM — 133K rows × 69 tokens too large for single pass)
            model.eval()
            val_probs_fold = np.zeros(len(y_val_np), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(x_num_val_t), CFG.PRED_BATCH_SIZE):
                    end = min(start + CFG.PRED_BATCH_SIZE, len(x_num_val_t))
                    x_num_batch = x_num_val_t[start:end].to(DEVICE)
                    x_cat_batch = x_cat_val_t[start:end].to(DEVICE)
                    val_logits = model(x_cont=x_num_batch, x_cat=x_cat_batch).squeeze(-1)
                    val_probs_fold[start:end] = torch.sigmoid(val_logits).cpu().numpy()

            # Handle NaN (safety)
            val_probs_fold = np.nan_to_num(val_probs_fold, nan=0.5)

            val_auc = auc_score(y_val_np, val_probs_fold)

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= CFG.ES_PATIENCE:
                    print(f"      [ES@{epoch}] ", end="")
                    break

        # Restore best and predict (batched to avoid OOM)
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        # Val predictions (batched)
        val_probs_final = np.zeros(len(y_val_np), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x_num_val_t), CFG.PRED_BATCH_SIZE):
                end = min(start + CFG.PRED_BATCH_SIZE, len(x_num_val_t))
                x_num_batch = x_num_val_t[start:end].to(DEVICE)
                x_cat_batch = x_cat_val_t[start:end].to(DEVICE)
                val_logits = model(x_cont=x_num_batch, x_cat=x_cat_batch).squeeze(-1)
                val_probs_final[start:end] = torch.sigmoid(val_logits).cpu().numpy()

        # Test predictions (batched)
        test_probs_fold = np.zeros(len(x_num_test_t), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x_num_test_t), CFG.PRED_BATCH_SIZE):
                end = min(start + CFG.PRED_BATCH_SIZE, len(x_num_test_t))
                x_num_batch = x_num_test_t[start:end].to(DEVICE)
                x_cat_batch = x_cat_test_t[start:end].to(DEVICE)
                test_logits = model(x_cont=x_num_batch, x_cat=x_cat_batch).squeeze(-1)
                test_probs_fold[start:end] = torch.sigmoid(test_logits).cpu().numpy()

        # Handle NaN (safety)
        val_probs_final = np.nan_to_num(val_probs_final, nan=0.5)
        test_probs_fold = np.nan_to_num(test_probs_fold, nan=0.5)

        oof_probs[val_idx] = val_probs_final
        test_probs += test_probs_fold / CFG.N_FOLDS
        fold_scores.append(best_auc)

        del model, optimizer, scheduler, criterion, best_state, scaler
        del x_num_train_t, x_cat_train_t, x_num_val_t, x_cat_val_t, x_num_test_t, x_cat_test_t
        del y_train_t, train_ds, train_loader
        gc.collect()
        torch.cuda.empty_cache()

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"AUC={best_auc:.5f} | Time={fold_time:.0f}s | Total={elapsed:.1f}min")

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
    print(f"V7 RESULTS -- FT-Transformer Full Features ({DEVICE})")
    print(f"{'='*80}")
    print(f"Selected features: {len(selected_features)} (from 344 raw)")
    print(f"  - Numerical: {len(num_features_idx)}")
    print(f"  - Categorical: {len(cat_features_idx)}")
    print(f"Architecture: FT-Transformer (d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS})")
    print(f"Training: AdamW(lr={CFG.LR}, wd={CFG.WEIGHT_DECAY}), CosineAnnealingLR, BCEWithLogitsLoss")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
