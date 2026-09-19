"""
S6E9 V21 - XGBoost depth=4 + Explicit Conditional Cross Keys
================================================================================
Strategy: V20's model (0.94606 OOF, best single-model OOF we have) with the
          feature budget moved from dead weight into explicit crosses

Reference: https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738991
- Deotte's Simpson's paradox: more nearby chargers looks NEGATIVE overall
  (0-2 stations 19.00% vs 11-14 stations 17.27%) but flips POSITIVE inside
  both Home_Charging groups (no home charging 9.20% -> 14.70%)
- A commenter who swept every column pair found City_Type doing the same thing
  (Rural 19.34% vs Urban 16.11% overall; among no-home-charging rows Rural
  4.64% vs Urban 13.56%), and the within-group slope gap is larger than an
  additive-model simulation reproduces -> a real interaction, not just mix

Device: GPU (cuda) | Est. Time: ~20-25 min (V20 took 20.6 min)

Why V21 reallocates features instead of adding them:
V20 fold-1 XGBoost GAIN shares showed exactly where a depth-4 model spends
its limited capacity:
  - TE_lift_trigram_cat_auto/10/100 + lift_trigram  ~35% of ALL gain, #1 slot
  - _ECL_x_Subsidy / bigram ECL x Subsidy / _ev_recipe  also top-10
  -> the model is STARVED for pre-computed crosses; it cannot build 3- and
     4-way ones at depth 4
  - every digit column and all five structural flags scored <= 0.08% and were
     effectively ignored (the reference notebook prunes its equivalents too)
So V21 ADDS six conditional cross keys (each Triple-TE'd like the V10 bigrams,
plus its own generator-lift column) and DROPS the digit block and the four
flags that earned nothing.

Key Techniques:
1. Explicit Conditional Cross Keys (NEW in V21), each a string key that is
   frequency-encoded and Triple-TE'd, plus a generator-lift column:
   - cx_chg_x_home    charging-station total bins x Home_Charging_Possible
   - cx_city_x_home   City_Type x Home_Charging_Possible
   - cx_ecl_x_home    ECL x Home_Charging_Possible
   - cx_ecl_x_city    ECL x City_Type
   - cx_inc_x_ecl_x_sub   fixed income bands x ECL x Subsidy  (extends #1)
   - cx_ecl_x_sub_x_comm  ECL x Subsidy x commute bins (5.0 km isolated)
2. Generator-Lift Features (from V19): pool (train+test) frequency divided by
   original-dataset frequency, target-free, per value and per joint key
3. Multi-scale "Smooth Keys" (Markus / maiernator) + numeric-as-string columns
4. Frequency Encoding on all categorical-like columns (train+test pool)
5. Original Dataset target-mean mapping (itzzomkar's public dataset)
6. Targeted Bigrams + Selective Groupby Deviations (V10, 0.94636 LB)
7. Triple Target Encoding (auto, 10.0, 100.0) applied per-fold to avoid leakage
8. XGBoost depth=4 lossguide with the reference recipe's tuned regularisation
9. REMOVED here: digit feature block and the 30k/cliff/dead-zone/bound flags
   (measured <= 0.08% gain in V20). is_env_hater is kept — it is the one flag
   the reference notebook also kept.

Dataset Structure:
- 13 features + 1 target
- Categorical: Gender, City_Type, Current_Car_Type, Home_Charging_Possible,
               Subsidy_Available, Range_Anxiety_Level
- Numerical: Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
             Charging_Stations_Near_Home, Charging_Stations_Near_Work,
             Environmental_Concern_Level
- Target: Will_Buy_EV (No=0, Yes=1) — binary classification, ROC AUC metric

Generator-Artifact Signal Map (verified against the local data):
  - income >= 170,537:        393 train rows, 100.00% buy  (millionaire cliff)
  - income 31,004-41,970:   1,257 train rows,   0.00% buy  (dead zone)
  - income == 30,000:      61,605 train rows,   4.43% buy  (mode-collapse spike)
  - ECL == 1:             147,476 train rows,   0.57% buy
  - commute == 5.0 km:    144,280 train rows,  18.44% buy
  - commute >= 83 km:         186 train rows,   0.00% buy
  - Subsidy No / Yes:      0.576% vs 27.47% buy
  - buy-rate by income-lift bucket (target-free):
      lift <= 0.5 -> 23.19% | (0.5,0.9] -> 21.74% | (0.9,1.1] -> 21.50%
      (1.1,2.0] -> 15.25% | (2.0,5.0] -> 16.95% | >5 -> 17.12%
    NON-MONOTONE, so it is not a proxy for income level
  - novel income values: 13,742 rows (2.06%), 19.56% buy vs 17.46% base
  - recipe-only AUC: 0.9377 | adversarial train-vs-test AUC: 0.501 (no shift)

Score discipline: public LB covers 20% of test and cannot resolve a +0.00007
change (V19 and V20 both scored exactly 0.94639 despite different OOFs), so
versions are ranked by honest OOF and accepted only when paired DeLong against
V20 gives delta > +0.00006 AND z > 3.
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

# Check sklearn version for TargetEncoder compatibility
print(f"scikit-learn version: {sklearn_version}")
print(f"xgboost version: {xgb.__version__}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v21"
    EXP_ID = "S6E9_V21_XGB_CrossKeys"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — 5 folds kept identical to V19 so the OOF stays directly comparable
    # (the reference notebook's 10-fold OOF is NOT comparable across fold counts)
    N_FOLDS = 5
    RANDOM_SEED = 42

    # Categorical columns that get a generator-lift feature
    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

    # Fixed bin edges for the cross keys — hardcoded so train, test and the
    # original frame are binned identically (no quantile fitted on one split)
    INCOME_EDGES  = [0, 42000, 55000, 70000, 85000, 100000, 120000, 145000, 170536.9, 1e9]
    CHARGER_EDGES = [-0.1, 2, 5, 8, 11, 1e9]
    # 4.999 / 5.001 isolates the 144,280-row exactly-5.0 km cluster in its own bin
    COMMUTE_EDGES = [-0.1, 2.999, 4.999, 5.001, 8, 12, 20, 30, 1e9]

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
# Adopted verbatim from najiama's XGBoost recipe (credited there to Tilii).
# max_depth=4 caps the trees: heavy regularisation is what keeps a shallow
# model from fitting synthetic-generator noise while lossguide still lets it
# reach for the strongest interactions.
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'n_estimators': 50000,
    'learning_rate': 0.01,
    'max_depth': 4,               # "massive regularization against synthetic noise"
    'max_leaves': 16,             # only read under lossguide
    'grow_policy': 'lossguide',
    'min_child_weight': 4.532387806880492,
    'subsample': 0.7400402414525654,
    'colsample_bytree': 0.5695776529558766,
    'alpha': 0.7523885021652775,   # = L1
    'reg_lambda': 0.6189949691705282,  # = L2
    'gamma': 3.673225596759869,
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
def add_engineered_features(df):
    """
    EDA-validated engineered features (unchanged from V3/V10).

    Adds:
    - 4 arithmetic interactions (_ECL_x_Subsidy, _ECL_x_RangeAnxiety,
      _Income_x_Subsidy, _Charging_Total)
    - 3 log transforms (_log_Income, _log_Commute, _log_Charging_Total)
    - 4 hard-edge flags (_high_income, _range_anxiety_high, _ecl_max, _ev_recipe)
    """
    df = df.copy()

    # The original dataset carries NaNs in these three columns
    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

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
    df['_log_Income'] = np.log1p(df['Annual_Income_USD'].clip(lower=0)).astype('float32')
    df['_log_Commute'] = np.log1p(df['Daily_Commute_km'].clip(lower=0)).astype('float32')
    df['_log_Charging_Total'] = np.log1p(df['_Charging_Total'].clip(lower=0)).astype('float32')

    # ---- Hard-edge flags (public discussion findings) ----
    df['_high_income'] = (df['Annual_Income_USD'] > 170537).astype('int8')
    df['_range_anxiety_high'] = (df['Range_Anxiety_Level'] == 'High').astype('int8')
    df['_ecl_max'] = (df['Environmental_Concern_Level'] == 5).astype('int8')
    df['_ev_recipe'] = (
        (df['Environmental_Concern_Level'] == 5) &
        (df['Range_Anxiety_Level'] == 'Low')
    ).astype('int8')

    # Drop helper columns
    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df


def add_synthetic_artifact_flags(df):
    """
    Only is_env_hater survives from the V10 flag set.

    V20 measured <= 0.08% gain for is_30k_spike / is_millionaire_cliff /
    is_dead_zone / _below_buy_bound / _dead_zone_exact — a depth-4 tree reads
    the raw income column and forms those splits itself, so the flags are dead
    weight (the reference notebook prunes its copies of them too). ECL == 1 is
    kept because ECL is the one column whose anomaly the tree under-weights.
    """
    df = df.copy()
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
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

    The synthetic pool over- and under-produces specific values relative to the
    10k-row original dataset (rejection shaping), and that shaping tracks the
    target. For each key:

        lift = freq_pool(value) / freq_orig(value)      pool = train + test

    This never touches y, so it is leak-free, and it is defined for test rows.
    Values absent from the original get lift = 0 and a separate novel flag.

    Returns (train_df, test_df, orig_df, lift_cols).
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

    # Multi-scale income keys + commute + every categorical
    _lift(train_df['income_exact_int'], test_df['income_exact_int'],
          orig_df['income_exact_int'], 'lift_inc_exact')
    _lift(train_df['income100_floor'], test_df['income100_floor'],
          orig_df['income100_floor'], 'lift_income100')
    _lift(train_df['commute_integer'], test_df['commute_integer'],
          orig_df['commute_integer'], 'lift_commute')
    for c in CFG.LIFT_CATS:
        _lift(train_df[c].astype(str), test_df[c].astype(str),
              orig_df[c].astype(str), f'lift_{c}')

    # Trigram-cell lift: ECL=3 cells out-perform the linear recipe by 1.22-1.31x
    for df_ in (train_df, test_df, orig_df):
        df_['_tri_key'] = (df_['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
                           + '_' + df_['Subsidy_Available'].astype(str)
                           + '_' + df_['Range_Anxiety_Level'].astype(str))
    _lift(train_df['_tri_key'], test_df['_tri_key'], orig_df['_tri_key'], 'lift_trigram')

    # Novelty flag: income value that never appears in the original 10k
    orig_inc_set = set(orig_df['income_exact_int'].unique())
    train_df['novel_inc'] = (~train_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    test_df['novel_inc']  = (~test_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    orig_df['novel_inc']  = np.int8(0)
    new_cols.append('novel_inc')

    # Helper column is object dtype — LightGBM/XGBoost cannot take it
    for df_ in (train_df, test_df, orig_df):
        df_.drop(columns=['_tri_key'], errors='ignore', inplace=True)

    print(f"      Added {len(new_cols)} lift/novelty columns: {new_cols}")
    return train_df, test_df, orig_df, new_cols


def _binned(d, col, edges):
    """Fixed-edge bin codes as strings, identical for train / test / original."""
    return pd.cut(d[col].astype('float64'), bins=edges, labels=False,
                  include_lowest=True).fillna(-1).astype('int32').astype(str)


def add_cross_keys(train_df, test_df, orig_df):
    """
    Explicit Conditional Cross Keys — the whole point of V21.

    V20 gave ~35% of its total XGBoost gain to the four lift-trigram columns,
    i.e. a depth-4 model leans hard on crosses it is handed pre-built. These
    six keys add the crossings the discussion threads documented as Simpson
    reversals, plus one deeper extension of the current #1 feature:

      cx_chg_x_home   station total x home charging   (9.20% -> 14.70% flip)
      cx_city_x_home  city type x home charging       (Rural 4.64% vs Urban 13.56%)
      cx_ecl_x_home   ECL x home charging             (the confounder itself)
      cx_ecl_x_city   ECL x city type
      cx_inc_x_ecl_x_sub     income band x ECL x subsidy  (extends lift_trigram)
      cx_ecl_x_sub_x_comm    ECL x subsidy x commute (5.0 km isolated)

    Each key is returned as a string column (so it gets frequency encoded and
    Triple-TE'd like the V10 bigrams) AND as a generator-lift column, which is
    target-free: pool frequency / original frequency, never y.

    Returns (train_df, test_df, orig_df, key_cols, lift_cols).
    """
    print("   Adding EXPLICIT CROSS KEYS (Simpson reversals + deeper recipe keys)...")

    def build(d):
        ecl  = d['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
        home = d['Home_Charging_Possible'].astype(str)
        sub  = d['Subsidy_Available'].astype(str)
        city = d['City_Type'].astype(str)
        chg  = d['Charging_Stations_Near_Home'] + d['Charging_Stations_Near_Work']
        return {
            'cx_chg_x_home':       _binned(d.assign(_c=chg), '_c', CFG.CHARGER_EDGES) + '_' + home,
            'cx_city_x_home':      city + '_' + home,
            'cx_ecl_x_home':       ecl + '_' + home,
            'cx_ecl_x_city':       ecl + '_' + city,
            'cx_inc_x_ecl_x_sub':  _binned(d, 'Annual_Income_USD', CFG.INCOME_EDGES) + '_' + ecl + '_' + sub,
            'cx_ecl_x_sub_x_comm': ecl + '_' + sub + '_' + _binned(d, 'Daily_Commute_km', CFG.COMMUTE_EDGES),
        }

    ktr, kte, kog = build(train_df), build(test_df), build(orig_df)
    key_cols, lift_cols = [], []

    for name in ktr:
        train_df[name] = ktr[name]; test_df[name] = kte[name]; orig_df[name] = kog[name]
        key_cols.append(name)

        # target-free lift of the same key
        pool = pd.concat([ktr[name], kte[name]], ignore_index=True)
        comp_freq = pool.value_counts(normalize=True)
        orig_freq = kog[name].value_counts(normalize=True)
        lift_name = f'lift_{name}'
        for df_, keys in ((train_df, ktr[name]), (test_df, kte[name]), (orig_df, kog[name])):
            raw = keys.map(comp_freq).fillna(0.0) / keys.map(orig_freq)
            df_[lift_name] = raw.fillna(0.0).astype('float32')
        lift_cols.append(lift_name)

    print(f"      Added {len(key_cols)} cross keys: {key_cols}")
    print(f"      Added {len(lift_cols)} cross-lift columns: {lift_cols}")
    return train_df, test_df, orig_df, key_cols, lift_cols


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
    Selective groupby DEVIATIONS only (V10's finding: deviation is the part that
    matters, and only ECL_bin / income_bin groups are worth grouping by).
    Group statistics come from the train+test pool, never from the target.
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
    print(f"Model: XGBoost depth=4 lossguide (V20 recipe, OOF 0.94606 / LB 0.94639)")
    print(f"ADDS  : 6 explicit cross keys + their 6 generator-lift columns")
    print(f"DROPS : digit block and 4 dead flags (<= 0.08% gain each in V20)")
    print(f"KEEPS : lift/novelty set, smooth keys, bigrams, groupby devs, Triple TE")
    print(f"Original data: USED (per-fold concat, Buyer_ID dropped)")
    print(f"Self-contained: no previous-version OOF or submission is read")
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
    if not pd.api.types.is_numeric_dtype(train[CFG.TARGET]):
        train[CFG.TARGET] = (train[CFG.TARGET].astype(str).str.strip()
                             .str.title().map(target2idx))
    if not pd.api.types.is_numeric_dtype(orig[CFG.TARGET]):
        orig[CFG.TARGET] = (orig[CFG.TARGET].astype(str).str.strip()
                            .str.title().map(target2idx))

    # Store IDs
    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()

    # Drop id from train/test; align original schema
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)  # drops Buyer_ID + target

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")

    # Identify column types (from test, which has no target)
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
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} (XGBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V20 pipeline, digits/flags swapped for crosses)...")

    # NOTE: the V19/V20 digit block is intentionally NOT built here. V20 measured
    # <= 0.08% gain for every digit column and every income-anomaly flag.

    # 2a. Engineered interactions + log transforms + hard-edge flags
    print("   Adding engineered interactions + flags...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig_aligned = add_engineered_features(orig_aligned)

    # 2b. Kept anomaly flag (ECL == 1)
    print("   Adding synthetic-artifact flag (is_env_hater only)...")
    train = add_synthetic_artifact_flags(train)
    test  = add_synthetic_artifact_flags(test)
    orig_aligned = add_synthetic_artifact_flags(orig_aligned)

    # 2c. Multi-scale smooth keys
    print("   Adding multi-scale smooth keys...")
    train = add_smooth_keys(train)
    test  = add_smooth_keys(test)
    orig_aligned = add_smooth_keys(orig_aligned)

    # 2d. Original dataset target means
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

    # 2e. Generator-lift features (target-free; needs the smooth keys above)
    train, test, orig_aligned, lift_cols = add_lift_features(train, test, orig_aligned)

    # 2f. NEW in V21: explicit conditional cross keys + their lifts
    train, test, orig_aligned, cross_cols, cross_lift_cols = add_cross_keys(train, test, orig_aligned)
    lift_cols = lift_cols + cross_lift_cols

    # 2g. Numeric-as-string columns
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

    # 2h. Frequency encoding on all categorical-like columns
    print("   Adding frequency encoding on all cat + num-as-string + cross keys...")
    freq_target_cols = CATS + train_num_cat_cols + cross_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    # 2i. Targeted bigrams (V10) + mirrored onto the original frame
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    # 2j. Selective groupby deviations (V10) + mirrored onto the original frame
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

    # 2k. Drop the helper bins (object dtype, only needed for the groupby maths)
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    # 2l. Redundancy selection
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # 2m. Define feature groups — the V21 cross keys are Triple-TE'd like the bigrams
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols + cross_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - V21 cross keys TE'd: {len([c for c in cross_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"     - Dropped in V21: digit block + 4 income-anomaly flags")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV with per-fold orig concat + Triple Target Encoding)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []

    # Golden rule: KFold(5, shuffle=True, rs=42) — same split as V19
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    t0 = time.time()
    # orig is split in lockstep with train for the per-fold concat
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

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

        print(f"      Train (comp+orig): {X_train.shape} | "
              f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- Triple Target Encoding (auto, 10.0, 100.0), per-fold ----
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

        # Any categorical left over is dropped; XGBoost needs a numeric frame
        string_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        if string_cols:
            X_train = X_train.drop(columns=string_cols)
            X_val   = X_val.drop(columns=string_cols)
            X_test_fold = X_test_fold.drop(columns=string_cols)

        X_train = X_train.astype('float32')
        X_val   = X_val.astype('float32')
        X_test_fold = X_test_fold.astype('float32')

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)}")

        # Train model
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predictions
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

        # Feature importance (fold 1 only, for visibility)
        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            artifact_feats = imp[imp['feature'].str.contains('lift_|cx_|novel_')]
            if len(artifact_feats) > 0:
                print(f"\n      Top-15 ARTIFACT features (lift / cross-key / novelty):")
                print(artifact_feats.head(15).to_string(index=False))
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
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    # Save OOF as CSV (id, pred)
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # Save submission (probability of class 1)
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V21 RESULTS — XGBoost depth=4 + Explicit Cross Keys (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {len(te_feature_names)} Triple TE = {len(FEATURES) + len(te_feature_names)} total")
    print(f"  - ADDED  : {len(cross_cols)} cross keys (Triple TE = {3*len(cross_cols)} cols) + {len(cross_lift_cols)} cross-lift cols")
    print(f"  - KEPT   : {len(lift_cols)} generator-lift/novelty cols, {len(bigram_cols)} bigrams, 12 groupby deviations")
    print(f"  - DROPPED: digit block (V20 gain <= 0.08%) + 4 income-anomaly flags")
    print(f"Model: XGBoost depth=4, lossguide, max_leaves=16, lr=0.01, gamma=3.673")
    print(f"Original data: concatenated per-fold (Buyer_ID dropped)")
    print(f"Target Encoding: per-fold, 3 smoothings (auto/10/100)")
    print(f"Comparison — V20 XGBoost: OOF 0.94606 / LB 0.94639")
    print(f"             V19 LightGBM: OOF 0.94599 / LB 0.94639")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"Accept gate: paired DeLong vs V20 needs delta > +0.00006 AND z > 3")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
