"""
S6E9 V1 - XGBoost Baseline
================================================================================
Strategy: XGBoost with Digit Features + Engineered Interactions + Target Encoding
          + Original Data (per-fold concat)

Reference: https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94583
- Single XGBoost CV 0.94583
- Using 5-fold StratifiedKFold

Reference: https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943
- Digit Feature Extraction (8 features per numerical column)
- Frequency Encoding for categorical + digit features
- Target Encoding (per-fold to avoid leakage)
- Sample weights for class imbalance

Device: GPU (cuda)

Key Techniques:
1. Digit Feature Extraction (8 features per numerical column)
2. Engineered interactions (EDA-validated):
   - _ECL_x_Subsidy, _ECL_x_RangeAnxiety (415x rate flip per EDA pivot)
   - _Income_x_Subsidy, _Charging_Total
   - Log transforms: _log_Income, _log_Commute, _log_Charging_Total
   - Quantile bins: Income_10q, Commute_7q, ECL_5q, ChargingHome_5q
   - Hard-edge flags: _high_income (>$170,537), _range_anxiety_high,
     _ecl_max, _ev_recipe (ECL=5 & RA=Low)
3. Frequency Encoding for categorical + digit features
4. Target Encoding (per-fold to avoid leakage)
5. Original EV Adoption dataset concatenated per-fold (Buyer_ID dropped)
6. Sample weights for class imbalance (inverse class frequency)

Dataset Structure:
- 13 features + 1 target
- Categorical: Gender, City_Type, Current_Car_Type, Home_Charging_Possible,
               Subsidy_Available, Range_Anxiety_Level
- Numerical: Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
             Charging_Stations_Near_Home, Charging_Stations_Near_Work,
             Environmental_Concern_Level
- Target: Will_Buy_EV (No=0, Yes=1) — binary classification, ROC AUC metric

EDA-Confirmed Signal Concentration (drives FE priorities):
  - Subsidy_Available:           70.4% importance (No -> 0.58%, Yes -> 27.47%)
  - Environmental_Concern_Level: 23.3% importance (1->0.6%, 5->54.5%)
  - Range_Anxiety_Level:          3.2% importance (Low->18.9%, High->0.1%)
  - All other 10 features:        3.1% combined
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
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer
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
    VERSION_NAME = "v1"
    EXP_ID = "S6E9_V1_XGB_Baseline"

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
# 4. MODEL PARAMETERS
# =============================================================================
# Parameters aligned with S6E4 V1 XGB baseline + S6E9 EDA findings
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'n_estimators': 6000,
    'max_depth': 6,           # S6E9 signal concentrated; deeper than S6E4's 4
    'learning_rate': 0.05,
    'subsample': 0.7,         # = bagging_fraction
    'colsample_bytree': 0.6,  # = feature_fraction
    'reg_alpha': 10,          # = lambda_l1
    'reg_lambda': 10,         # = lambda_l2
    'min_child_weight': 12,   # = min_child_samples
    'max_bin': 512,
    'early_stopping_rounds': 250,
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
    Digit Feature Extraction - Key technique from baseline.
    Extracts 8 digit features per numerical column (positions -4 to 3).
    Also rounds original numerical columns based on magnitude.
    """
    df = df.copy()

    for c in num_cols:
        # Add 8 digit features per numerical column
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')

        # Round original columns based on max value
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
    - 4 quantile-binned numerics (Income_10q, Commute_7q, ECL_5q, ChargingHome_5q)
    - 4 hard-edge flags (high_income, range_anxiety_high, ecl_max, ev_recipe)
    """
    df = df.copy()

    # Ordinal encode Range_Anxiety_Level (Low=0, Medium=1, High=2)
    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')

    # Binary Yes/No -> 1/0 for Subsidy
    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

    # ---- Arithmetic interactions (EDA-validated) ----
    # ECL x Subsidy: biggest interaction (ECL=5 & Subsidy=Yes -> ~80% buy)
    df['_ECL_x_Subsidy'] = (
        df['Environmental_Concern_Level'] * df['_Subsidy_bin']
    ).astype('float32')

    # ECL x RangeAnxiety: 415x rate flip (ECL=5 & RA=Low -> 54.5%)
    df['_ECL_x_RangeAnxiety'] = (
        df['Environmental_Concern_Level'] * (3 - df['_RA_code'])
    ).astype('float32')

    # Income x Subsidy: per "Poor + Subsidy = Buy EV" public thread
    df['_Income_x_Subsidy'] = (
        df['Annual_Income_USD'] * df['_Subsidy_bin']
    ).astype('float32')

    # Charging_Total: stations near home + work (correlated pair, r=0.51)
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

    # ---- Quantile bins (binned numerics) ----
    # Done in main() after train/test concat for proper quantile fitting

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

    # Drop helper columns
    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df


def add_quantile_bins(train_df, test_df, orig_df, bin_config):
    """
    Add quantile-binned features using KBinsDiscretizer.
    Fit on train, transform test and orig.

    Args:
        train_df, test_df, orig_df: DataFrames
        bin_config: dict of {col: n_bins}
    Returns:
        (train_df, test_df, orig_df) with new _bin_ columns added
    """
    for col, n_bins in bin_config.items():
        bin_name = f"{col}_{n_bins}q_bin"
        kb = KBinsDiscretizer(
            n_bins=n_bins, encode='ordinal',
            strategy='quantile', subsample=None,
        )
        # Fit on train (fillna with median; original has NaNs)
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

    # 2b. Engineered interactions + log transforms + hard-edge flags
    print("   Adding engineered interactions + flags...")
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

    # 2d. Drop constant columns
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)
    orig.drop(columns=DROP, inplace=True)

    # 2e. Define feature groups
    # Digit features + quantile bins + engineered flags are categorical-like
    digit_cols = [c for c in test.columns if 'digit' in c]
    bin_cols   = [c for c in test.columns if c.endswith('_q_bin')]
    flag_cols  = [c for c in test.columns if c in [
        '_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe'
    ]]
    CATEGORY = CATS + digit_cols + bin_cols + flag_cols

    # Engineered numeric features (interactions + log transforms)
    engineered_num = [c for c in test.columns if c.startswith('_') and c not in flag_cols]

    # All numerical (original + engineered)
    NUMS_ALL = NUMS + engineered_num

    # 2f. Frequency encoding for categorical + digit + bin + flag features
    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c]  = test[c].map(lambda x: mapping.get(x, mapping_default))
        orig[c]  = orig[c].map(lambda x: mapping.get(x, mapping_default))

    FEATURES = CATEGORY + NUMS_ALL
    print(f"   Total features: {len(FEATURES)}")
    print(f"     - Original categorical: {len(CATS)}")
    print(f"     - Digit features: {len(digit_cols)}")
    print(f"     - Quantile bins: {len(bin_cols)}")
    print(f"     - Hard-edge flags: {len(flag_cols)}")
    print(f"     - Original + engineered numeric: {len(NUMS_ALL)}")

    # 2g. Sample weights for class imbalance (computed on train only;
    #     re-computed per fold after orig concat for proper weighting)
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    print(f"\n   Class Weights (train): {weights_dict}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV with per-fold orig concat + Target Encoding)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, orig concat + TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    t0 = time.time()
    # We need to split orig in lockstep with train for per-fold concat
    # Use a separate StratifiedKFold on orig with same seed
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

        # Target Encoding (per-fold to avoid leakage) — applied to all FEATURES
        te = TargetEncoder(
            target_type='binary', smooth='auto',
            cv=5, random_state=42,
        )
        X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
        X_val_enc   = te.transform(X_val[FEATURES])
        X_test_enc  = te.transform(X_test_fold[FEATURES])

        # Convert to DataFrame and concatenate
        X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
        X_val_enc   = pd.DataFrame(X_val_enc,   index=X_val.index)
        X_test_enc  = pd.DataFrame(X_test_enc,  index=X_test_fold.index)

        X_train = pd.concat([X_train, X_train_enc], axis=1)
        X_val   = pd.concat([X_val,   X_val_enc],   axis=1)
        X_test  = pd.concat([X_test_fold, X_test_enc], axis=1)

        # Drop original categorical columns (TE columns replace them)
        X_train = X_train.drop(CATS, axis=1)
        X_val   = X_val.drop(CATS, axis=1)
        X_test  = X_test.drop(CATS, axis=1)

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
        test_probs += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

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

        del X_train, X_val, X_test, y_train, y_val, model, te
        gc.collect()

    # Overall OOF score
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

    # Save OOF probabilities (for hill climber)
    np.save(f"{out_dir}/oof_probs_{CFG.VERSION_NAME}.npy", oof_probs)
    np.save(f"{out_dir}/test_probs_{CFG.VERSION_NAME}.npy", test_probs)
    print(f"   [SAVED] {out_dir}/test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs.shape})")
    print(f"   [SAVED] {out_dir}/oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_probs.shape})")

    # Save OOF as CSV (id, pred)
    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_probs,
    })
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # Save submission (probability of class 1)
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_probs,
    })
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V1 RESULTS — XGBoost Baseline (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + TE columns")
    print(f"  - Original categorical: {len(CATS)}")
    print(f"  - Digit features: {len(digit_cols)}")
    print(f"  - Quantile bins: {len(bin_cols)}")
    print(f"  - Hard-edge flags: {len(flag_cols)}")
    print(f"  - Original + engineered numeric: {len(NUMS_ALL)}")
    print(f"Original data: concatenated per-fold (Buyer_ID dropped)")
    print(f"Target Encoding: per-fold on all FEATURES")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
