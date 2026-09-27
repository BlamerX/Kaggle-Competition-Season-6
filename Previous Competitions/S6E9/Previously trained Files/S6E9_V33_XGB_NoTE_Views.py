"""
S6E9 V33 - Zero-target-encoding, numerics-as-categorical view (GPU)
In this competition's structural twins the top-weighted members carried no target
encoding, and our residual-exhaustion test only covers the TE-bearing matrix, so this
representation is unprobed. USE_TRIPLE_TE=False and the string columns survive as
global label codes; max_bin 1024 -> 8192 is the only param change, so those
micro-value codes have bins to split on.
Change vs V30: representation plus bin resolution; config, FE and split otherwise equal.
Outputs: /kaggle/working/oof_v33.csv (id, pred), sub_v33.csv (id, Will_Buy_EV)
Estimate: 45-60 min.
"""

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
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# 1. CONFIG
class CFG:
    VERSION_NAME = "v33"
    EXP_ID = "S6E9_V33_XGB_NoTE_Views"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'

    N_FOLDS = 10
    RANDOM_SEED = 42

    NUM_ROUND = 8000
    ES_ROUNDS = 400

    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

    USE_TRIPLE_TE = False

# 2. SEED EVERYTHING
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'device': 'cuda',
    'seed': CFG.RANDOM_SEED,
    'nthread': -1,
    'verbosity': 0,
    'learning_rate': 0.01,
    'max_depth': 3,
    'max_leaves': 8,
    'grow_policy': 'lossguide',
    'min_child_weight': 4.532387806880492,
    'subsample': 0.7400402414525654,
    'colsample_bytree': 0.85,
    'alpha': 0.7523885021652775,
    'reg_lambda': 0.6189949691705282,
    'gamma': 1.0,

    'max_bin': 8192,
}

# 3. METRIC
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# 4. FEATURE ENGINEERING
def add_digit_features(df, num_cols):
    """
    Digit Feature Extraction — the FIXED version (V19 onwards).

    V1-V18 used `col // (10**k) % 10`; for k < 0 that is a float floor-division
    that reads the IEEE-754 bit pattern and corrupts 89.63% of the
    Daily_Commute_km tenths. Scaling once to int64 and using integer divmods is
    exact.
    """
    df = df.copy()

    for c in num_cols:
        scaled = np.rint(df[c].fillna(0).astype('float64') * 1e4).astype('int64')
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = ((scaled // np.int64(10 ** (k + 4))) % 10).astype('int8')

    return df

def add_engineered_features(df):
    """
    EDA-validated engineered features (unchanged from V3/V10).

    Adds 4 arithmetic interactions, 3 log transforms and 4 hard-edge flags.
    """
    df = df.copy()

    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')

    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

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

    df['_log_Income'] = np.log1p(df['Annual_Income_USD'].clip(lower=0)).astype('float32')
    df['_log_Commute'] = np.log1p(df['Daily_Commute_km'].clip(lower=0)).astype('float32')
    df['_log_Charging_Total'] = np.log1p(df['_Charging_Total'].clip(lower=0)).astype('float32')

    df['_high_income'] = (df['Annual_Income_USD'] > 170537).astype('int8')
    df['_range_anxiety_high'] = (df['Range_Anxiety_Level'] == 'High').astype('int8')
    df['_ecl_max'] = (df['Environmental_Concern_Level'] == 5).astype('int8')
    df['_ev_recipe'] = (
        (df['Environmental_Concern_Level'] == 5) &
        (df['Range_Anxiety_Level'] == 'Low')
    ).astype('int8')

    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df

def add_synthetic_artifact_flags(df):
    """CTGAN anomaly flags mapped by starkhushi / Aryan Kaisth (V10 set)."""
    df = df.copy()
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')
    df['is_dead_zone'] = ((df['Annual_Income_USD'] >= 38000.0) &
                          (df['Annual_Income_USD'] <= 42000.0)).astype('int8')
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
    return df

def add_structural_flags(df):
    """
    Exact structural bounds decoded from the generator recipe.

    buy_score = 1.2*(income/100k) + 0.6*ECL + 2*subsidy - anxiety_penalty, so at
    income < 41,667 the score can never reach 5.5 whatever the other columns say.
    """
    df = df.copy()
    df['_below_buy_bound'] = (df['Annual_Income_USD'] < 41667).astype('int8')
    df['_dead_zone_exact'] = ((df['Annual_Income_USD'] >= 31004) &
                              (df['Annual_Income_USD'] <= 41970)).astype('int8')
    return df

def add_smooth_keys(df):
    """Markus's multi-scale 'Smooth Keys' (binned numerics as strings)."""
    df = df.copy()
    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)
    return df

def add_lift_features(train_df, test_df, orig_df):
    """
    Generator-Lift Features — the reason V19 broke the 0.94636 plateau.

        lift = freq_pool(value) / freq_orig(value)      pool = train + test

    Never touches y, so it is leak-free and defined for test rows.
    """
    print("   Adding generator-lift features (pool freq / orig freq, target-free)...")
    new_cols = []

    def _lift(keys_tr, keys_te, keys_og, col_name):
        pool = pd.concat([keys_tr, keys_te], ignore_index=True)
        comp_freq = pool.value_counts(normalize=True)
        orig_freq = keys_og.value_counts(normalize=True)
        for df_, keys in ((train_df, keys_tr), (test_df, keys_te), (orig_df, keys_og)):
            raw = keys.map(comp_freq).fillna(0.0) / keys.map(orig_freq)
            df_[col_name] = raw.fillna(0.0).astype('float32')
        new_cols.append(col_name)

    _lift(train_df['income_exact_int'], test_df['income_exact_int'],
          orig_df['income_exact_int'], 'lift_inc_exact')
    _lift(train_df['income100_floor'], test_df['income100_floor'],
          orig_df['income100_floor'], 'lift_income100')
    _lift(train_df['commute_integer'], test_df['commute_integer'],
          orig_df['commute_integer'], 'lift_commute')
    for c in CFG.LIFT_CATS:
        _lift(train_df[c].astype(str), test_df[c].astype(str),
              orig_df[c].astype(str), f'lift_{c}')

    for df_ in (train_df, test_df, orig_df):
        df_['_tri_key'] = (df_['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
                           + '_' + df_['Subsidy_Available'].astype(str)
                           + '_' + df_['Range_Anxiety_Level'].astype(str))
    _lift(train_df['_tri_key'], test_df['_tri_key'], orig_df['_tri_key'], 'lift_trigram')

    orig_inc_set = set(orig_df['income_exact_int'].unique())
    train_df['novel_inc'] = (~train_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    test_df['novel_inc']  = (~test_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    orig_df['novel_inc']  = np.int8(0)
    new_cols.append('novel_inc')

    for df_ in (train_df, test_df, orig_df):
        df_.drop(columns=['_tri_key'], errors='ignore', inplace=True)

    print(f"      Added {len(new_cols)} lift/novelty columns: {new_cols}")
    return train_df, test_df, orig_df, new_cols

def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
    """Map the original dataset's per-value target means onto train and test."""
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
    """Numeric-as-string columns so micro-values get their own TE category."""
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
    """Frequency encoding computed on the train+test pool (no target used)."""
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df

def add_targeted_bigrams(train_df, test_df):
    """
    The two bigrams V10 found were missing (V10 = 0.94636 LB):
    ECL_bin x Range_Anxiety and ECL_bin x Subsidy. Both are Triple TE'd later.
    """
    print("   Adding TARGETED bigram interactions (ECL_bin x RA, ECL_bin x Subsidy)...")

    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    bigram_name_1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bigram_name_1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bigram_name_1]  = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)

    bigram_name_2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bigram_name_2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bigram_name_2]  = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)

    bigram_cols = [bigram_name_1, bigram_name_2]
    return train_df, test_df, bigram_cols

def add_selective_groupby(train_df, test_df):
    """
    Selective groupby DEVIATIONS only (V10's finding). Group statistics come from
    the train+test pool, never from the target.
    """
    print("   Adding SELECTIVE groupby deviation features (ECL_bin, income_bin groups)...")

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
            grp_std  = combined.groupby(group_col, observed=False)[num_col].std().fillna(0)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            train_df[dev_name] = ((train_df[num_col] - train_df[group_col].map(grp_mean)) /
                                  (train_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            test_df[dev_name]  = ((test_df[num_col] - test_df[group_col].map(grp_mean)) /
                                  (test_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            new_features.append(dev_name)

    print(f"      Added {len(new_features)} SELECTIVE groupby deviation features")
    return train_df, test_df

def drop_redundant_features(train_df, test_df, target):
    """Drop constant columns and columns perfectly correlated (|r| = 1.0)."""
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

def align_original_schema(orig_df):
    """Align original dataset schema with competition train (drops Buyer_ID)."""
    keep_cols = [
        'Age', 'Annual_Income_USD', 'Daily_Commute_km',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level',
        'Gender', 'City_Type', 'Current_Car_Type',
        'Home_Charging_Possible', 'Subsidy_Available',
        'Range_Anxiety_Level',
    ]
    return orig_df[keep_cols].copy()

def encode_string_columns_as_codes(dfs):
    str_cols = [c for c in dfs[0].columns if not pd.api.types.is_numeric_dtype(dfs[0][c])]
    if not str_cols:
        return [], {}
    union = pd.concat([df[str_cols] for df in dfs], ignore_index=True)
    card = {}
    for c in str_cols:
        values = union[c].astype(str)
        codes, uniques = pd.factorize(values, sort=True)
        mapping = {v: int(i) for i, v in enumerate(uniques)}
        card[c] = len(uniques)
        for df in dfs:
            df[c] = df[c].astype(str).map(mapping).fillna(-1).astype('int32')
    return str_cols, card

# 5. MAIN EXECUTION
if __name__ == "__main__":
    t0_all = time.time()

    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    if not pd.api.types.is_numeric_dtype(train[CFG.TARGET]):
        train[CFG.TARGET] = (train[CFG.TARGET].astype(str).str.strip()
                             .str.title().map(target2idx))
    if not pd.api.types.is_numeric_dtype(orig[CFG.TARGET]):
        orig[CFG.TARGET] = (orig[CFG.TARGET].astype(str).str.strip()
                            .str.title().map(target2idx))

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy().reset_index(drop=True)

    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig).reset_index(drop=True)

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")

    CATS = [c for c in test.columns if train[c].dtype == object or train[c].dtype.name == 'str']
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
    print(f"   Orig share of the training matrix: {len(orig_aligned)/(len(train)+len(orig_aligned)):.4%}")
    print(f"   Rows held out per fold: {len(train)//CFG.N_FOLDS:,}")
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} (XGBoost handles NaN natively)")

    print(f"\n[2/5] Feature Engineering (V19 pipeline, TE block behind USE_TRIPLE_TE={CFG.USE_TRIPLE_TE})...")

    print("   Adding FIXED digit features...")
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

    print("   Adding STRUCTURAL flags (buy bound 41,667 + exact dead zone)...")
    train = add_structural_flags(train)
    test  = add_structural_flags(test)
    orig_aligned = add_structural_flags(orig_aligned)

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

    train, test, orig_aligned, lift_cols = add_lift_features(train, test, orig_aligned)

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

    train, test, bigram_cols = add_targeted_bigrams(train, test)
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test = add_selective_groupby(train, test)
    orig_aligned['income_bin'] = pd.cut(
        orig_aligned['Annual_Income_USD'].fillna(orig_aligned['Annual_Income_USD'].median()),
        bins=5, labels=['1', '2', '3', '4', '5']).astype(str)
    combined_for_grp = pd.concat([
        train[['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level',
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
            grp_std  = combined_for_grp.groupby(group_col, observed=False)[num_col].std().fillna(0)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            orig_aligned[dev_name] = ((orig_aligned[num_col] - orig_aligned[group_col].map(grp_mean)) /
                                      (orig_aligned[group_col].map(grp_std) + 1e-6)).astype('float32')

    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols) if c in train.columns]

    str_cols_coded, str_card = encode_string_columns_as_codes([train, test, orig_aligned])
    print(f"   Label-coded {len(str_cols_coded)} string columns as integer codes")

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns TE WOULD target (kept as codes, TE is OFF): {len(TARGET_ENCODE_COLS)}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in FEATURES])}")

    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, orig concat, TE={'ON' if CFG.USE_TRIPLE_TE else 'OFF'})...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    best_iters = []
    fold_scores = []
    n_te = 0
    cols_used = 0
    fold1_importances = {}

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

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

        te_feature_names = []
        if CFG.USE_TRIPLE_TE:
            for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
                te = TargetEncoder(target_type='binary', smooth=smooth_val,
                                   cv=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
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

            n_te = len(te_feature_names)

            cols_to_drop_after_te = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
            X_train = X_train.drop(columns=cols_to_drop_after_te, errors='ignore')
            X_val   = X_val.drop(columns=cols_to_drop_after_te, errors='ignore')
            X_test_fold = X_test_fold.drop(columns=cols_to_drop_after_te, errors='ignore')

        string_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        if string_cols:
            X_train = X_train.drop(columns=string_cols)
            X_val   = X_val.drop(columns=string_cols)
            X_test_fold = X_test_fold.drop(columns=string_cols)

        cols_fold = list(X_train.columns)
        if fold == 0:
            cols_used = len(cols_fold)
            print(f"      Feature count: {len(cols_fold)} "
                  f"(TE {'ON' if CFG.USE_TRIPLE_TE else 'OFF'}: {n_te} TE cols) | "
                  f"train rows {len(train_idx):,} comp + {len(or_train_idx):,} orig")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=cols_fold)
        dva = xgb.DMatrix(X_val, label=y_val, feature_names=cols_fold)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=CFG.NUM_ROUND,
                            evals=[(dva, 'valid')], verbose_eval=False,
                            callbacks=[xgb.callback.EarlyStopping(rounds=CFG.ES_ROUNDS,
                                                                  save_best=True)])
        val_probs = booster.predict(dva)
        oof_probs[val_idx] = val_probs
        dtest = xgb.DMatrix(X_test_fold, feature_names=cols_fold)
        test_probs += booster.predict(dtest) / CFG.N_FOLDS
        if fold == 0:
            fold1_importances = booster.get_score(importance_type='gain')
        del dtrain, dva, dtest
        gc.collect()

        best_iter = booster.num_boosted_rounds()
        best_iters.append(best_iter)
        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        print(f"      AUC: {fold_auc:.5f} | BestIter {best_iter} | "
              f"time {(time.time()-fold_start)/60:.1f} min | total {(time.time()-t0_all)/60:.1f} min")

        del booster, X_train, X_val, X_test_fold
        gc.collect()

    cv_score = auc_score(y_np, oof_probs)
    print(f"\n   {'='*70}")
    print(f"   {CFG.N_FOLDS}-Fold OOF CV (AUC): {cv_score:.6f}")
    print(f"   Fold mean {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print(f"   Best iterations: {best_iters}")

    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)  — probabilities")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv  — probabilities, "
          f"range {test_probs.min():.4f} .. {test_probs.max():.4f}")

    print(f"\n{'='*80}")
    print(f"[5/5] V33 RESULTS — No-TE View, zero target encoding + NUMS_AS_CATS (GPU)")
    print(f"{'='*80}")
    print(f"Columns actually used: {cols_used} (TE OFF, string cols kept as integer codes)")
    print(f"Config: V30 params with max_bin {XGB_PARAMS['max_bin']} as the only change "
          f"(depth 3, leaves 8, gamma 1.0, colsample 0.85, lr 0.01)")
    print(f"OOF CV (AUC): {cv_score:.6f}  |  Fold mean {np.mean(fold_scores):.6f} "
          f"+/- {np.std(fold_scores):.6f}")
    print(f"Fold AUCs: {[round(s,5) for s in fold_scores]}")
    print(f"Best iterations: {best_iters}")
    print(f"Test prediction min/max: {test_probs.min():.6f} / {test_probs.max():.6f}")

    print(f"\nFold-1 top-15 gain importances:")
    for rank, (feat, gain) in enumerate(
            sorted(fold1_importances.items(), key=lambda kv: kv[1], reverse=True)[:15], 1):
        print(f"   {rank:>2}. {feat:<44} {gain:,.1f}")

    print(f"\nCOMPARABILITY: V30's 10-fold OOF was 0.946171 on the TE-bearing matrix.")
    print("   This run changes the representation (zero TE, codes kept, 8192 bins),")
    print("   so read the delta as the whole-view effect, not a single-factor one.")
    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*80}")
