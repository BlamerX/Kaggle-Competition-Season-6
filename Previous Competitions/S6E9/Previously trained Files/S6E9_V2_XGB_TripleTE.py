"""
S6E9 V2 - XGBoost with Triple Target Encoding
================================================================================
Strategy: XGBoost with Digit Features + Engineered Interactions + Triple Target
          Encoding + True Frequency Encoding + Recipe Score + Original Data
          (per-fold concat)

References:
- https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94583 (CV 0.94583, LB 0.94590)
  -> Triple Target Encoding (auto, 10, 100 smoothing) on 13 core cols: +0.00095 OOF, +0.00130 LB
  -> Hyperparameter upgrade (lr=0.005, depth=7, weak reg, max_bin=1024): +0.00022 OOF
  -> True frequency encoding on all columns: +0.00119 OOF (with digits)
  -> Feature selection (drop constants + corr=1): cleaner model, same score
- https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter/notebook
  -> Recipe score feature (cdeotte's "buy score" formula from EDA)

V2 Changes from V1 (OOF 0.94535 / LB 0.94559):
1. Triple Target Encoding (auto, 10, 100) on 13 core cols -> 39 TE features per fold
   (V1 used single TE on all features; Evgen measured +0.00095 OOF gain from Triple TE)
2. Hyperparameter upgrade: lr=0.005, depth=7, weak reg, max_bin=1024, n_est=10000
   (V1 used lr=0.05, depth=6, strong reg=10/10; Evgen measured +0.00022 OOF gain)
3. True frequency encoding on ALL columns (adds _freq column per feature)
   (V1 used frequency-rank encoding on cats/digits only; Evgen measured +0.00119 OOF gain)
4. Recipe score feature (cdeotte's buy_score formula as extra column)
   (cdeotte: "small but real, consistent across folds")

Kept from V1:
- Per-fold orig concat (10K rows, Buyer_ID dropped, schema aligned)
- Digit features (8 per numeric column)
- Engineered interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety, _Income_x_Subsidy, _Charging_Total)
- Log transforms (_log_Income, _log_Commute, _log_Charging_Total)
- Quantile bins (Income_10q, Commute_7q, ECL_5q, ChargingHome_5q)
- Hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
- Sample weights for class imbalance (inverse class frequency)
- 5-fold StratifiedKFold (Evgen proved 10-fold gives same LB; 5-fold saves compute)

Dropped from V1:
- Single TE on all features (replaced by Triple TE on 13 core cols)
- Count encoding (replaced by true frequency encoding)
- Frequency-rank encoding on cats (replaced by label-encoding + true freq columns)

Dataset Structure:
- 13 features + 1 target
- Categorical: Gender, City_Type, Current_Car_Type, Home_Charging_Possible,
               Subsidy_Available, Range_Anxiety_Level
- Numerical: Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
             Charging_Stations_Near_Home, Charging_Stations_Near_Work,
             Environmental_Concern_Level
- Target: Will_Buy_EV (No=0, Yes=1) — binary classification, ROC AUC metric

Expected: OOF ~0.9463-0.9470, LB ~0.9465 (matches Anhadm's 0.94621 public ceiling)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
except Exception as e:
    print(f"⚠️ cuDF failed: {e}. Using standard pandas.")

import os
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Check sklearn version for TargetEncoder compatibility
print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v2"
    EXP_ID = "S6E9_V2_XGB_TripleTE"

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
# 4. MODEL PARAMETERS (Evgen Dvorkin's proven config — CV 0.94583, LB 0.94590)
# =============================================================================
# Key differences from V1:
# - learning_rate: 0.005 (V1: 0.05) — 10x slower, deeper learning
# - n_estimators: 10000 (V1: 6000) — more room with low LR
# - max_depth: 7 (V1: 6) — slightly deeper
# - min_child_weight: 10 (V1: 12) — less conservative
# - subsample: 0.9 (V1: 0.7) — more rows per tree
# - colsample_bytree: 0.9 (V1: 0.6) — more features per split
# - reg_alpha: 0.071 (V1: 10) — much weaker L1 (V1 was over-regularized)
# - reg_lambda: 2.0 (V1: 10) — much weaker L2
# - max_bin: 1024 (V1: 512) — better split resolution
# - early_stopping_rounds: 500 (V1: 250) — more patience for slow LR

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'n_estimators': 10000,
    'max_depth': 7,
    'learning_rate': 0.005,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.071,
    'reg_lambda': 2.0,
    'min_child_weight': 10,
    'max_bin': 1024,
    'early_stopping_rounds': 500,
    'n_jobs': -1,
    'verbosity': 0,
}

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
def add_digit_features(df, num_cols, M):
    """
    Digit Feature Extraction - 8 features per numerical column (positions -4 to 3).
    Also rounds original numerical columns based on magnitude.
    """
    df = df.copy()

    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')

        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)

    return df


def add_engineered_features(df):
    """
    EDA-validated engineered features for S6E9.

    Adds:
    - 4 arithmetic interactions (ECL x Subsidy, ECL x RangeAnxiety,
      Income x Subsidy, Charging_Total)
    - 3 log transforms (log_Income, log_Commute, log_Charging_Total)
    - 4 hard-edge flags (high_income, range_anxiety_high, ecl_max, ev_recipe)
    - 1 recipe score (cdeotte's buy_score formula from EDA)
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
    # "Everyone above $170,537 buys an EV" -- starkhushi's thread
    df['_high_income'] = (
        df['Annual_Income_USD'] > 170537
    ).astype('int8')

    # Range_Anxiety == High -> near-zero buy rate (0.1%)
    df['_range_anxiety_high'] = (
        df['Range_Anxiety_Level'] == 'High'
    ).astype('int8')

    # ECL == 5 -> 54.5% buy rate (vs 17.5% baseline)
    df['_ecl_max'] = (
        df['Environmental_Concern_Level'] == 5
    ).astype('int8')

    # cdeotte's "recipe": ECL==5 & Range_Anxiety==Low -> highest buy segment
    df['_ev_recipe'] = (
        (df['Environmental_Concern_Level'] == 5) &
        (df['Range_Anxiety_Level'] == 'Low')
    ).astype('int8')

    # ---- Recipe score (cdeotte's buy_score formula from EDA) ----
    # buy_score = 1.2 * (income/100k) + 0.6 * ECL + 2.0 * subsidy
    #           - 1.0 * (RA=Medium) - 3.0 * (RA=High)
    # Person buys if buy_score + N(0,1) > 5.5
    df['_recipe_score'] = (
        1.2 * (df['Annual_Income_USD'] / 1e5)
        + 0.6 * df['Environmental_Concern_Level']
        + 2.0 * df['_Subsidy_bin']
        - 1.0 * (df['Range_Anxiety_Level'] == 'Medium').astype('float32')
        - 3.0 * (df['Range_Anxiety_Level'] == 'High').astype('float32')
    ).astype('float32')

    # Drop helper columns
    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df


def add_quantile_bins(train_df, test_df, orig_df, bin_config):
    """
    Add quantile-binned features using KBinsDiscretizer.
    Fit on train, transform test and orig.
    """
    for col, n_bins in bin_config.items():
        bin_name = f"{col}_{n_bins}q_bin"
        kb = KBinsDiscretizer(
            n_bins=n_bins, encode='ordinal',
            strategy='quantile', subsample=None,
        )
        med = train_df[col].median()
        kb.fit(train_df[[col]].fillna(med))
        train_df[bin_name] = kb.transform(
            train_df[[col]].fillna(med)
        ).ravel().astype('int32')
        test_df[bin_name] = kb.transform(
            test_df[[col]].fillna(med)
        ).ravel().astype('int32')
        orig_df[bin_name] = kb.transform(
            orig_df[[col]].fillna(med)
        ).ravel().astype('int32')
    return train_df, test_df, orig_df


def add_frequency_encoding(train_df, test_df, orig_df, skip_cols=None):
    """
    True frequency encoding (Evgen Dvorkin's approach).
    Adds a _freq column for EVERY column with normalized frequency across train+test.
    Original columns are kept; this adds new signal about how rare/common each value is.

    This is different from V1's frequency-rank encoding (which replaced values).
    Here we ADD frequency as new columns while keeping originals.

    Args:
        train_df, test_df, orig_df: DataFrames
        skip_cols: set/list of column names to skip (e.g., target column still in train_df)
    """
    if skip_cols is None:
        skip_cols = set()
    else:
        skip_cols = set(skip_cols)

    # Compute frequency maps on train+test combined (no target leakage — uses only features)
    # Skip columns that don't exist in test (e.g., target column still in train)
    cols_to_encode = [c for c in train_df.columns if c not in skip_cols and c in test_df.columns]

    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)

    freq_maps = {}
    for col in cols_to_encode:
        freq_maps[col] = combined[col].value_counts(normalize=True).to_dict()

    # Apply to train, test, orig
    for col in cols_to_encode:
        freq_map = freq_maps[col]
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).astype('float32')
        test_df[f"{col}_freq"]  = test_df[col].map(freq_map).astype('float32')
        # For orig, values not in train+test get frequency 0 (rare/unknown)
        orig_df[f"{col}_freq"]  = orig_df[col].map(freq_map).fillna(0).astype('float32')

    return train_df, test_df, orig_df


def label_encode_categoricals(train_df, test_df, orig_df, cat_cols):
    """
    Label-encode categorical columns to integers (for XGBoost).
    Uses train+test combined mapping for consistency.
    Returns the label-encoded DataFrames and the list of LE column names.
    """
    le_cols = []
    for col in cat_cols:
        le_name = f"LE_{col}"
        le_cols.append(le_name)

        # Build mapping from train+test combined
        combined_vals = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        unique_vals = sorted(combined_vals.unique())
        mapping = {val: i for i, val in enumerate(unique_vals)}

        train_df[le_name] = train_df[col].astype(str).map(mapping).astype('int32')
        test_df[le_name]  = test_df[col].astype(str).map(mapping).astype('int32')
        # For orig, unknown values get -1 (will be handled by TE/freq)
        orig_df[le_name]  = orig_df[col].astype(str).map(mapping).fillna(-1).astype('int32')

    # Drop original string categorical columns
    train_df = train_df.drop(columns=cat_cols)
    test_df  = test_df.drop(columns=cat_cols)
    orig_df  = orig_df.drop(columns=cat_cols)

    return train_df, test_df, orig_df, le_cols


def drop_redundant_features(train_df, test_df, orig_df):
    """
    Feature selection (Evgen's approach):
    1. Drop constant columns (nunique == 1)
    2. Drop perfectly correlated columns (corr == 1.0)
    """
    # Drop constants
    const_cols = [c for c in train_df.columns if train_df[c].nunique() == 1]
    print(f"   Dropping {len(const_cols)} constant columns")
    train_df = train_df.drop(columns=const_cols)
    test_df  = test_df.drop(columns=const_cols)
    orig_df  = orig_df.drop(columns=const_cols)

    # Drop perfectly correlated (corr == 1.0)
    # Only check numeric columns to avoid issues
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = train_df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        corr_drop = [col for col in upper_tri.columns
                     if any(upper_tri[col] == 1.0)]
        print(f"   Dropping {len(corr_drop)} perfectly-correlated columns")
        train_df = train_df.drop(columns=corr_drop)
        test_df  = test_df.drop(columns=corr_drop)
        orig_df  = orig_df.drop(columns=corr_drop)
        del corr_matrix, upper_tri
        gc.collect()

    return train_df, test_df, orig_df


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
    print(f"Device: GPU (cuda) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Buyer_ID dropped)")
    print(f"Triple TE: auto, 10, 100 smoothing on 13 core cols")
    print(f"True Frequency Encoding: on ALL columns")
    print(f"Recipe Score: cdeotte's buy_score formula")
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
    orig  = align_original_schema(orig)  # drops Buyer_ID + target

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig.shape}")

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
    print(f"   Missing in orig: {orig.isna().sum().sum()} (XGBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering...")

    # 2a. Digit features (8 per numeric column)
    print("   Adding digit features...")
    M = train[NUMS].max()
    train = add_digit_features(train, NUMS, M)
    test  = add_digit_features(test,  NUMS, M)
    orig  = add_digit_features(orig,  NUMS, M)

    # 2b. Engineered interactions + log transforms + hard-edge flags + recipe score
    print("   Adding engineered interactions + flags + recipe score...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig  = add_engineered_features(orig)

    # 2c. Quantile bins (fit on train, transform all)
    print("   Adding quantile bins...")
    bin_config = {
        'Annual_Income_USD': 10,
        'Daily_Commute_km': 7,
        'Environmental_Concern_Level': 5,
        'Charging_Stations_Near_Home': 5,
    }
    train, test, orig = add_quantile_bins(train, test, orig, bin_config)

    # 2d. Label-encode categorical columns (replaces V1's frequency-rank encoding)
    print(f"   Label-encoding {len(CATS)} categorical columns...")
    train, test, orig, le_cols = label_encode_categoricals(train, test, orig, CATS)
    print(f"   LE columns: {le_cols}")

    # 2e. True frequency encoding on ALL columns (Evgen's approach)
    # Skip target column (still in train at this point; not in test)
    print("   Adding true frequency encoding on all columns...")
    train, test, orig = add_frequency_encoding(
        train, test, orig, skip_cols=[CFG.TARGET]
    )

    # 2f. Feature selection (drop constants + corr=1)
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, orig = drop_redundant_features(train, test, orig)

    # 2g. Define feature groups for Triple TE
    # Triple TE targets: 7 original numerics + 6 label-encoded cats = 13 cols
    # (Evgen's proven set)
    te_core_cols = NUMS + le_cols
    # Filter to columns that still exist after feature selection
    te_core_cols = [c for c in te_core_cols if c in train.columns]
    print(f"   Triple TE target columns ({len(te_core_cols)}): {te_core_cols}")

    # All features = all columns in train (minus target)
    FEATURES = [c for c in train.columns if c != CFG.TARGET]
    print(f"\n   Total features: {len(FEATURES)}")
    digit_cols   = [c for c in FEATURES if 'digit' in c]
    bin_cols     = [c for c in FEATURES if c.endswith('_q_bin')]
    flag_cols    = [c for c in FEATURES if c in [
        '_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe'
    ]]
    interaction_cols = [c for c in FEATURES if c in [
        '_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
        '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total',
        '_recipe_score'
    ]]
    freq_cols = [c for c in FEATURES if c.endswith('_freq')]
    print(f"     - Original numeric: {len([c for c in NUMS if c in FEATURES])}")
    print(f"     - Label-encoded cat: {len([c for c in le_cols if c in FEATURES])}")
    print(f"     - Digit features: {len(digit_cols)}")
    print(f"     - Quantile bins: {len(bin_cols)}")
    print(f"     - Hard-edge flags: {len(flag_cols)}")
    print(f"     - Interactions + logs + recipe: {len(interaction_cols)}")
    print(f"     - Frequency features: {len(freq_cols)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV with per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")

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
    orig_splits = list(kf_orig.split(orig, y_orig))

    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X, y), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Per-fold: concat competition train + original
        orig_tr = orig.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | "
              f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # Sample weights for class imbalance (re-computed on concatenated train)
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        avg = len(y_train) / 2
        w_neg = avg / neg
        w_pos = avg / pos
        train_weights = np.where(y_train == 0, w_neg, w_pos).astype('float32')

        # ---- Triple Target Encoding (auto, 10, 100) on 13 core cols ----
        # Evgen's approach: 3 smoothings give model different views of each feature
        # Applied per-fold to prevent leakage
        te_feature_names = []
        for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
            te = TargetEncoder(
                target_type='binary', smooth=smooth_val,
                cv=5, shuffle=True, random_state=42,
            )
            X_train_enc = te.fit_transform(X_train[te_core_cols], y_train).astype('float32')
            X_val_enc   = te.transform(X_val[te_core_cols]).astype('float32')
            X_test_enc  = te.transform(X_test_fold[te_core_cols]).astype('float32')

            for i, col in enumerate(te_core_cols):
                te_name = f"TE_{col}_{smooth_name}"
                X_train[te_name] = X_train_enc[:, i]
                X_val[te_name]   = X_val_enc[:, i]
                X_test_fold[te_name] = X_test_enc[:, i]
                te_feature_names.append(te_name)

        print(f"      Triple TE: {len(te_feature_names)} features "
              f"(3 smoothings x {len(te_core_cols)} cols)")

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")

        # Train model
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predictions
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

        # Feature importance (fold 1 only, for visibility)
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            print()

        del X_train, X_val, X_test_fold, y_train, y_val, model
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
    print(f"V2 RESULTS — XGBoost with Triple Target Encoding (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  - Original numeric: {len([c for c in NUMS if c in FEATURES])}")
    print(f"  - Label-encoded cat: {len([c for c in le_cols if c in FEATURES])}")
    print(f"  - Digit features: {len(digit_cols)}")
    print(f"  - Quantile bins: {len(bin_cols)}")
    print(f"  - Hard-edge flags: {len(flag_cols)}")
    print(f"  - Interactions + logs + recipe: {len(interaction_cols)}")
    print(f"  - Frequency features: {len(freq_cols)}")
    print(f"  - Triple TE (auto/10/100 x {len(te_core_cols)} cols): {len(te_feature_names)}")
    print(f"Original data: concatenated per-fold (Buyer_ID dropped)")
    print(f"Hyperparameters: lr=0.005, depth=7, weak reg, max_bin=1024")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
