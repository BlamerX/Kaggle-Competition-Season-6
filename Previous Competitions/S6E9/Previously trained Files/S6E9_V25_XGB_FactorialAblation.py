"""
S6E9 V25 - XGBoost Factorial Ablation: Cross Keys / Recipe Prior / Gain Pruning (GPU)
================================================================================
Strategy: one script, one build per fold, a control arm equal to V22's winning
          config, and four arms that each change exactly ONE factor vs control

Reference: V22 (XGBoost tuned search, OOF 0.94608 / LB 0.94640). V21 taught us
that bundling an addition with a removal destroys attribution; this harness
avoids that by keeping every arm single-factor against an in-run control.

Device: GPU (cuda) | Est. Time: ~160-190 min

Arms (all share the fold matrices, so only training time differs):
  a0 control            = V22 winner c1 (depth 3, 8 leaves, gamma 1.0, colsample 0.85)
  a1 + cross keys       = a0 + V21's six Simpson/recipe crosses, PURE ADDITION
  a2 + recipe prior     = a0 + base_margin = logit(LR fitted on the original 10k)
  a3 + gain pruning     = a0 + per-fold in-run pruning to the top-94 gain features
  a4 all three          = a1 + a2 + a3, to see whether the arms interact

Why these three: V21's crosses took 5 of the top 8 gain slots but were judged
inside a bundle that also deleted the digit block, so the addition was never
tested alone; the decoded label recipe gives a per-row prior no tree can compute
in 3 levels; and the reference 0.94639 recipe matched us with 94 features.

Diagnostics: every arm is scored against a0 with a paired DeLong test computed
from this run's own OOF vectors (no external file is read). Paired SE between
our models is 0.00001-0.00003, so z > 3 is the accept gate, not the delta.

Design notes:
- Stage A runs all 5 arms on the canonical KFold(rs=42); the top 3 are re-run on
  KFold(rs=7) and the winner is the best mean of the two splits (V22's protocol).
- Cross-key columns are appended last and are excluded from the redundancy scan,
  so a0's feature matrix is byte-identical to V22's and must reproduce 0.94608.
- OOF stays purely out-of-fold for every arm; the submission is the winner's test
  predictions averaged over both splits. V23's full-data refit blend is NOT
  applied here, so any LB move is attributable to the arms alone.
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
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    VERSION_NAME = "v25"
    EXP_ID = "S6E9_V25_XGB_FactorialAblation"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — 5 folds kept identical to V19/V20/V22 so the OOF stays comparable
    N_FOLDS = 5
    RANDOM_SEED = 42

    # Search control (V22 protocol)
    SEARCH_SEED = 42     # canonical split
    CONFIRM_SEED = 7     # second split to confirm the leaders on
    TOP_K = 3            # how many arms advance to the confirmation stage

    # Categorical columns that get a generator-lift feature
    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

    # Fixed bin edges for the cross keys — identical for train / test / original
    INCOME_EDGES  = [0, 42000, 55000, 70000, 85000, 100000, 120000, 145000, 170536.9, 1e9]
    CHARGER_EDGES = [-0.1, 2, 5, 8, 11, 1e9]
    COMMUTE_EDGES = [-0.1, 2.999, 4.999, 5.001, 8, 12, 20, 30, 1e9]

    # Prefixes that identify the cross-key block (added last, excluded from a0)
    CROSS_PREFIXES = ('cx_', 'lift_cx_', 'TE_cx_', 'TE_lift_cx_')

    # arm a3: in-run gain pruning
    PRUNE_TOP_N = 94
    PRUNE_ROUNDS = 600
    PRUNE_ES = 100
    PRUNE_LR = 0.05

    # arm a2: recipe prior from the original dataset. logit(0.993) ~ 5, so the
    # clip keeps the offset inside a range the trees can still argue with.
    MARGIN_CLIP = 5.0

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS — control config and arms
# =============================================================================
# Booster-API params (DMatrix + xgb.train), because arm a2 needs a per-row
# base_margin and the sklearn wrapper cannot pass one.
XGB_BASE = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'device': 'cuda',
    'seed': CFG.RANDOM_SEED,
    'nthread': -1,
    'verbosity': 0,
    'learning_rate': 0.01,
    'max_depth': 3,                # V22 winner c1
    'max_leaves': 8,
    'grow_policy': 'lossguide',
    'min_child_weight': 4.532387806880492,
    'subsample': 0.7400402414525654,
    'colsample_bytree': 0.85,
    'alpha': 0.7523885021652775,
    'reg_lambda': 0.6189949691705282,
    'gamma': 1.0,
    'max_bin': 1024,
}
NUM_ROUND = 8000
ES_ROUNDS = 400

# Each arm changes exactly one factor vs the control.
ARMS = [
    {'name': "a0 control (V22 winner c1)",     'cross': False, 'margin': False, 'prune': 0},
    {'name': "a1 + six cross keys",            'cross': True,  'margin': False, 'prune': 0},
    {'name': "a2 + LR-on-original base_margin", 'cross': False, 'margin': True,  'prune': 0},
    {'name': "a3 + in-run gain pruning",       'cross': False, 'margin': False, 'prune': CFG.PRUNE_TOP_N},
    {'name': "a4 cross keys + prior + pruning", 'cross': True,  'margin': True,  'prune': CFG.PRUNE_TOP_N},
]

# =============================================================================
# 5. METRICS — ROC AUC and a paired DeLong test on this run's own OOFs
# =============================================================================
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)


def _midrank(x):
    """Average ranks with ties resolved, in Sun & Xu's (2014) DeLong form."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype='float64')
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype='float64')
    out[order] = ranks
    return out


def delong_paired_cov(y_true, score_matrix):
    """
    Fast DeLong (Sun & Xu 2014) covariance of several AUCs on the same labels.

    score_matrix is (k, n) predictions. The placement values are
        V10_i = (#negatives below positive i + half the ties) / n
              = (midrank of X_i in the pooled sample - its midrank among the
                 positives) / n
        V01_j = (m - (pooled midrank of Y_j - its midrank among negatives)) / m
    and Var = cov(V10)/m + cov(V01)/n, so SE of the difference between arms i
    and j is sqrt(cov[i,i] + cov[j,j] - 2*cov[i,j]).
    """
    y = np.asarray(y_true).astype('int8')
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    k = score_matrix.shape[0]

    X = np.asarray(score_matrix[:, pos], dtype='float64')
    Y = np.asarray(score_matrix[:, neg], dtype='float64')

    v10 = np.empty((k, m), dtype='float64')
    v01 = np.empty((k, n), dtype='float64')
    aucs = np.empty(k, dtype='float64')

    for r in range(k):
        tz = _midrank(np.concatenate([X[r], Y[r]]))
        tx = _midrank(X[r])
        ty = _midrank(Y[r])
        v10[r] = (tz[:m] - tx) / n
        v01[r] = (m - (tz[m:] - ty)) / m
        aucs[r] = v10[r].mean()

    s10 = np.atleast_2d(np.cov(v10)) / m
    s01 = np.atleast_2d(np.cov(v01)) / n
    return aucs, s10 + s01


def paired_z(y_true, s_a, s_b):
    """DeLong z for AUC(a) - AUC(b) when both are predictions on the same rows."""
    aucs, cov = delong_paired_cov(y_true, np.vstack([s_a, s_b]))
    var_diff = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    se = float(np.sqrt(max(var_diff, 0.0)))
    z = float('inf') if se == 0.0 else (aucs[0] - aucs[1]) / se
    return aucs[0], aucs[1], se, z

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
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
    income < 41,667 the score can never reach 5.5 whatever the other columns
    say. Sharper than the observed [31,004 - 41,970] dead zone (1,257 train rows,
    0 buyers).
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

    The synthetic pool over- and under-produces specific values relative to the
    10k-row original dataset, and that shaping tracks the target:

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

    # Helper column is object dtype — XGBoost cannot take it
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


def _binned(d, col, edges):
    """Fixed-edge bin codes as strings, identical for train / test / original."""
    return pd.cut(d[col].astype('float64'), bins=edges, labels=False,
                  include_lowest=True).fillna(-1).astype('int32').astype(str)


def add_cross_keys(train_df, test_df, orig_df):
    """
    ARM a1 — the six explicit conditional cross keys V21 introduced, kept here as
    a PURE ADDITION (V21 also deleted the digit block and 4 anomaly flags, which
    is what broke it).

      cx_chg_x_home        station total x home charging   (9.20% -> 14.70% flip)
      cx_city_x_home       city type x home charging       (Rural 4.64% vs Urban 13.56%)
      cx_ecl_x_home        ECL x home charging             (the confounder itself)
      cx_ecl_x_city        ECL x city type
      cx_inc_x_ecl_x_sub   income band x ECL x subsidy     (extends lift_trigram)
      cx_ecl_x_sub_x_comm  ECL x subsidy x commute         (5.0 km cluster isolated)

    Each key is emitted as a string column (so it is frequency encoded and
    Triple-TE'd like the V10 bigrams), plus its own target-free pool/orig lift and
    a string view of that lift. Appended last so the pre-existing columns, and
    therefore arm a0, stay identical to V22.
    """
    print("   Adding EXPLICIT CROSS KEYS (arm a1, pure addition)...")

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
        train_df[name] = ktr[name]
        test_df[name]  = kte[name]
        orig_df[name]  = kog[name]
        key_cols.append(name)

        # target-free lift of the same key
        pool = pd.concat([ktr[name], kte[name]], ignore_index=True)
        comp_freq = pool.value_counts(normalize=True)
        orig_freq = kog[name].value_counts(normalize=True)
        lift_name = f'lift_{name}'
        for df_, keys in ((train_df, ktr[name]), (test_df, kte[name]), (orig_df, kog[name])):
            raw = keys.map(comp_freq).fillna(0.0) / keys.map(orig_freq)
            df_[lift_name] = raw.fillna(0.0).astype('float32')
            # string view so the lift gets its own TE category, like lift_trigram
            df_[f'{lift_name}_cat'] = df_[lift_name].fillna(-1.0).astype(str)
        lift_cols.append(lift_name)

    print(f"      Added {len(key_cols)} cross keys + {len(lift_cols)} cross-lift columns")
    return train_df, test_df, orig_df, key_cols, lift_cols


def drop_redundant_features(train_df, test_df, target):
    """
    Drop constant columns and columns perfectly correlated (|r| = 1.0).

    The correlation scan is restricted to the non-cross columns on purpose: that
    keeps arm a0's matrix bit-identical to V22's. Cross columns are only dropped
    when they are constant, never when they duplicate a base column, so nothing
    V22 kept can disappear here.
    """
    is_cross = lambda c: c.startswith(CFG.CROSS_PREFIXES)

    base_cols_all = [c for c in train_df.columns if not is_cross(c)]
    eval_cols = [c for c in base_cols_all
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


def build_recipe_design(df):
    """
    Arm a2's design matrix: the raw drivers of the decoded label recipe, so the
    logistic fit can express buy_score = 1.2*(inc/100k) + 0.6*ECL + 2*subsidy
    - 1*RA_Medium - 3*RA_High almost exactly.
    """
    d = pd.DataFrame(index=df.index)
    inc = df['Annual_Income_USD'].astype('float64')
    d['income_100k']  = (inc / 100000.0).astype('float32')
    d['log_income']   = np.log1p(inc.clip(lower=0)).astype('float32')
    d['ecl']          = df['Environmental_Concern_Level'].astype('float64').fillna(3.0).astype('float32')
    d['subsidy']      = (df['Subsidy_Available'].astype(str) == 'Yes').astype('int8')
    d['ra_medium']    = (df['Range_Anxiety_Level'].astype(str) == 'Medium').astype('int8')
    d['ra_high']      = (df['Range_Anxiety_Level'].astype(str) == 'High').astype('int8')
    d['age']          = df['Age'].astype('float64').fillna(df['Age'].median()).astype('float32')
    d['commute']      = df['Daily_Commute_km'].astype('float64').fillna(15.0).astype('float32')
    d['cars']         = df['Number_of_Cars_Owned'].astype('float64').fillna(1.0).astype('float32')
    d['chg_home']     = df['Charging_Stations_Near_Home'].astype('float64').fillna(0.0).astype('float32')
    d['chg_work']     = df['Charging_Stations_Near_Work'].astype('float64').fillna(0.0).astype('float32')
    d['home_charging'] = (df['Home_Charging_Possible'].astype(str) == 'Yes').astype('int8')

    # The original 10k has ~543 missing cells (income 178, commute 181, ECL 184)
    # and LogisticRegression rejects NaN, so impute from this frame's own medians.
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.fillna(d.median(numeric_only=True)).fillna(0.0)
    return d.to_numpy('float32')


def fit_recipe_margin(design_tr, y_tr, *designs):
    """Standardised logistic fit on the ORIGINAL rows only -> per-row logit prior."""
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(design_tr, y_tr)
    out = []
    for d in designs:
        m = pipe.decision_function(d)
        out.append(np.clip(m, -CFG.MARGIN_CLIP, CFG.MARGIN_CLIP).astype('float32'))
    return pipe, out


def train_arm(params, num_round, es_rounds, X, y, margin, cols, Xv, yv, marginv):
    """
    Fit through the booster API so a per-row base_margin can be supplied.

    Returns (booster, val_probs, val_auc). Early stopping slices the booster to
    the best iteration, so predict() matches the sklearn wrapper's behaviour.
    """
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(cols),
                         base_margin=None if margin is None else margin)
    dval   = xgb.DMatrix(Xv, label=yv, feature_names=list(cols),
                         base_margin=None if marginv is None else marginv)
    callbacks = [xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)]
    booster = xgb.train(params, dtrain, num_boost_round=num_round,
                        evals=[(dval, 'valid')], callbacks=callbacks, verbose_eval=False)
    probs = booster.predict(dval)
    del dtrain, dval
    gc.collect()
    return booster, probs

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: GPU (cuda) | Folds: {CFG.N_FOLDS}")
    print(f"Arms: {len(ARMS)} single-factor ablations on one build, top-{CFG.TOP_K} confirmed on rs={CFG.CONFIRM_SEED}")
    print(f"Control a0 = V22 winner (depth 3, 8 leaves, gamma 1.0, colsample 0.85)")
    print(f"a1 cross keys | a2 LR-on-original base_margin | a3 in-run gain pruning to {CFG.PRUNE_TOP_N} | a4 all three")
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
    y_orig   = orig[CFG.TARGET].copy().reset_index(drop=True)

    # Drop id from train/test; align original schema
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig).reset_index(drop=True)

    # Raw design matrices for arm a2 (taken BEFORE feature engineering mutates)
    D_TRAIN_RAW = build_recipe_design(train)
    D_TEST_RAW  = build_recipe_design(test)
    D_ORIG_RAW  = build_recipe_design(orig_aligned)

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")
    print(f"   Recipe design: {D_TRAIN_RAW.shape[1]} columns for the base_margin arm")

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
    print("\n[2/5] Feature Engineering (V19/V22 pipeline + cross keys appended last)...")

    # 2a. Digit features (FIXED extraction, 8 per numeric column)
    print("   Adding FIXED digit features...")
    train = add_digit_features(train, NUMS)
    test  = add_digit_features(test,  NUMS)
    orig_aligned = add_digit_features(orig_aligned, NUMS)

    # 2b. Engineered interactions + log transforms + hard-edge flags
    print("   Adding engineered interactions + flags...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig_aligned = add_engineered_features(orig_aligned)

    # 2c. CTGAN anomaly flags
    print("   Adding synthetic-artifact flags...")
    train = add_synthetic_artifact_flags(train)
    test  = add_synthetic_artifact_flags(test)
    orig_aligned = add_synthetic_artifact_flags(orig_aligned)

    # 2d. Structural flags decoded from the recipe
    print("   Adding STRUCTURAL flags (buy bound 41,667 + exact dead zone)...")
    train = add_structural_flags(train)
    test  = add_structural_flags(test)
    orig_aligned = add_structural_flags(orig_aligned)

    # 2e. Multi-scale smooth keys
    print("   Adding multi-scale smooth keys...")
    train = add_smooth_keys(train)
    test  = add_smooth_keys(test)
    orig_aligned = add_smooth_keys(orig_aligned)

    # 2f. Original dataset target means
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

    # 2g. Generator-lift features (target-free; needs the smooth keys above)
    train, test, orig_aligned, lift_cols = add_lift_features(train, test, orig_aligned)

    # 2h. Numeric-as-string columns
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

    # 2i. Frequency encoding on all categorical-like columns
    print("   Adding frequency encoding on all cat + num-as-string cols...")
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    # 2j. Targeted bigrams (V10) + mirrored onto the original frame
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    # 2k. Selective groupby deviations (V10) + mirrored onto the original frame
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

    # 2l. Drop the helper bins (object dtype, only needed for the groupby maths)
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    # 2m. ARM a1 cross keys — appended LAST so a0's columns stay V22's
    train, test, orig_aligned, cx_key_cols, cx_lift_cols = add_cross_keys(train, test, orig_aligned)
    cx_str_cols = cx_key_cols + [f'{c}_cat' for c in cx_lift_cols]
    train, test = add_frequency_encoding(train, test, cx_str_cols)
    for col in cx_str_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    # 2n. Redundancy selection (base columns only, see docstring)
    print("   Feature selection (drop constants + perfectly-correlated, base block)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # 2o. Define feature groups
    BASE_TE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols) if c in train.columns]

    CROSS_TE_COLS = [c for c in cx_str_cols if c in train.columns]
    TARGET_ENCODE_COLS = BASE_TE_COLS + CROSS_TE_COLS

    FEATURES = [c for c in test.columns if c != 'id']
    CROSS_FEATURES = [c for c in FEATURES if c.startswith(CFG.CROSS_PREFIXES)]
    BASE_FEATURES  = [c for c in FEATURES if c not in set(CROSS_FEATURES)]

    print(f"\n   Total features: {len(FEATURES)}  ({len(BASE_FEATURES)} base = V22 set + {len(CROSS_FEATURES)} cross-key block)")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)} "
          f"({len(BASE_TE_COLS)} base + {len(CROSS_TE_COLS)} cross)")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in BASE_TE_COLS])}")

    # =========================================================================
    # [3/5] FACTORIAL ABLATION (5-Fold CV, per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Factorial ablation ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")
    print(f"   Stage A: {len(ARMS)} arms on KFold(rs={CFG.SEARCH_SEED}) — canonical split")
    print(f"   Stage B: top-{CFG.TOP_K} confirmed on KFold(rs={CFG.CONFIRM_SEED})")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    def run_split(seed, arm_indices):
        """
        Run one KFold split for the given arms.

        The fold matrices (Triple TE of the superset) are built ONCE per fold and
        shared by every arm; an arm simply trains on its column subset, so the
        ablation costs training time only, never re-encoding.
        Returns (oof by arm, test by arm, iters by arm, n_te).
        """
        names = ", ".join(ARMS[i]['name'] for i in arm_indices)
        print(f"\n   ===== SPLIT rs={seed} =====")
        print(f"   arms: {names}")

        kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        orig_splits = list(kf_orig.split(orig_aligned, y_orig))

        oof_by_arm   = {ai: np.zeros(len(y)) for ai in arm_indices}
        test_by_arm  = {ai: np.zeros(len(test_X)) for ai in arm_indices}
        iters_by_arm = {ai: [] for ai in arm_indices}
        n_te = 0
        t_seed = time.time()

        for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
                zip(kf.split(X), orig_splits)):

            fold_start = time.time()
            print(f"\n      Fold {fold+1}/{CFG.N_FOLDS}:")

            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Per-fold: concat competition train + original
            orig_tr = orig_aligned.iloc[or_train_idx].copy()
            y_orig_tr = y_orig.iloc[or_train_idx].copy()
            X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
            X_test_fold = test_X.copy()

            n_comp = len(train_idx)
            y_train_np = y_train.values.astype('float32')
            y_val_np   = y_val.values.astype('float32')

            # ---- Triple Target Encoding (auto, 10.0, 100.0), built once ----
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

            n_te = len(te_feature_names)

            # Drop the encoded string columns (TE columns replace them)
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

            cols_fold = list(X_train.columns)
            A_tr = X_train.to_numpy('float32')
            A_va = X_val.to_numpy('float32')
            A_te = X_test_fold.to_numpy('float32')
            del X_train, X_val, X_test_fold
            gc.collect()

            col_index = {c: i for i, c in enumerate(cols_fold)}
            # classify by NAME PREFIX so the TE children follow their parent:
            # TE_cx_* / TE_lift_cx_* belong to the cross block, everything else
            # (including TE_ of the base columns) to the V22 block.
            base_idx   = [i for i, c in enumerate(cols_fold) if not c.startswith(CFG.CROSS_PREFIXES)]
            cross_idx  = [i for i, c in enumerate(cols_fold) if c.startswith(CFG.CROSS_PREFIXES)]

            if fold == 0:
                print(f"         Final feature count: {len(cols_fold)} "
                      f"({n_te} Triple TE of {len(TARGET_ENCODE_COLS)} cols; "
                      f"{len(base_idx)} base-block, {len(cross_idx)} cross-block)")

            # ---- arm a2's per-row prior, fitted on ORIGINAL rows only ----
            margin_tr = margin_va = margin_te = None
            if any(ARMS[ai]['margin'] for ai in arm_indices):
                _pipe, (m_comp, m_orig, m_test) = fit_recipe_margin(
                    D_ORIG_RAW[or_train_idx], y_orig.iloc[or_train_idx].values,
                    D_TRAIN_RAW, D_ORIG_RAW, D_TEST_RAW)
                margin_tr = np.concatenate([m_comp[train_idx], m_orig[or_train_idx]]).astype('float32')
                margin_va = m_comp[val_idx].astype('float32')
                margin_te = m_test.astype('float32')
                if fold == 0:
                    lr_val_auc = auc_score(y_val_np, margin_va)
                    clipped = int((np.abs(m_comp) >= CFG.MARGIN_CLIP - 1e-6).sum())
                    print(f"         Recipe prior alone on this fold's val: AUC {lr_val_auc:.5f} | "
                          f"clipped margins {clipped:,} / {len(m_comp):,}")
                del m_comp, m_orig, m_test
                gc.collect()

            # ---- Train every arm on this fold's matrices ----
            for ai in arm_indices:
                arm = ARMS[ai]
                idxs = base_idx + (cross_idx if arm['cross'] else [])
                arm_cols = [cols_fold[i] for i in idxs]

                # each arm decides for itself whether the recipe prior is active
                if arm['margin']:
                    m_tr, m_va, m_te_arm = margin_tr, margin_va, margin_te
                else:
                    m_tr = m_va = m_te_arm = None

                if arm['prune'] and len(idxs) > arm['prune']:
                    qparams = dict(XGB_BASE)
                    qparams['learning_rate'] = CFG.PRUNE_LR
                    Xq = np.ascontiguousarray(A_tr[:, idxs])
                    Xqv = np.ascontiguousarray(A_va[:, idxs])
                    quick, _ = train_arm(qparams, CFG.PRUNE_ROUNDS, CFG.PRUNE_ES,
                                         Xq, y_train_np, m_tr, arm_cols,
                                         Xqv, y_val_np, m_va)
                    gain = quick.get_score(importance_type='gain')
                    keep_set = {c for c, _ in sorted(gain.items(),
                                                     key=lambda kv: (-kv[1], kv[0]))[:arm['prune']]}
                    del quick, Xq, Xqv
                    gc.collect()

                    keep = [c for c in arm_cols if c in keep_set]
                    if len(keep) < arm['prune']:
                        spare = [c for c in arm_cols if c not in keep_set]
                        keep = keep + spare[:arm['prune'] - len(keep)]
                    idxs = [col_index[c] for c in keep]
                    arm_cols = keep

                Xtr = np.ascontiguousarray(A_tr[:, idxs])
                Xvl = np.ascontiguousarray(A_va[:, idxs])
                booster, val_probs = train_arm(XGB_BASE, NUM_ROUND, ES_ROUNDS,
                                              Xtr, y_train_np, m_tr, arm_cols,
                                              Xvl, y_val_np, m_va)
                del Xtr, Xvl
                gc.collect()

                oof_by_arm[ai][val_idx] = val_probs

                dtest = xgb.DMatrix(np.ascontiguousarray(A_te[:, idxs]),
                                    feature_names=list(arm_cols),
                                    base_margin=m_te_arm)
                test_by_arm[ai] += booster.predict(dtest) / CFG.N_FOLDS
                del dtest
                gc.collect()

                best_iter = booster.num_boosted_rounds()
                iters_by_arm[ai].append(best_iter)

                fold_auc = auc_score(y_val_np, val_probs)
                print(f"         {arm['name']:<34} AUC {fold_auc:.5f} | "
                      f"trees {best_iter} | feats {len(arm_cols)}")

                if fold == 0 and ai == arm_indices[0]:
                    gain = booster.get_score(importance_type='gain')
                    total_gain = float(sum(gain.values()))
                    imp = pd.DataFrame({
                        'feature': arm_cols,
                        'importance': [gain.get(c, 0.0) / total_gain for c in arm_cols],
                    }).sort_values('importance', ascending=False)
                    print(f"\n         Top-10 feature importances ({arm['name']}, fold 1):")
                    print(imp.head(10).to_string(index=False))
                    print()

                del booster
                gc.collect()

            del A_tr, A_va, A_te
            gc.collect()

            print(f"      Fold {fold+1} done in {(time.time()-fold_start)/60:.1f} min | "
                  f"split elapsed {(time.time()-t_seed)/60:.1f} min")

        for ai in arm_indices:
            oof_auc = auc_score(y_np, oof_by_arm[ai])
            print(f"   rs={seed} OOF | {ARMS[ai]['name']:<34} {oof_auc:.5f} | "
                  f"iters {iters_by_arm[ai]}")

        # ---- paired DeLong of every arm against the control (in-run OOF only) ----
        if 0 in oof_by_arm:
            print(f"\n   Paired DeLong vs a0 control (rs={seed}):")
            for ai in arm_indices:
                if ai == 0:
                    continue
                a_arm, a_ctl, se, z = paired_z(y_np, oof_by_arm[ai], oof_by_arm[0])
                verdict = 'ACCEPT' if z > 3 else ('reject' if z < -3 else 'tie')
                print(f"      {ARMS[ai]['name']:<34} delta {a_arm - a_ctl:+.5f} | "
                      f"SE {se:.5f} | z {z:+.2f} {verdict}")

        return oof_by_arm, test_by_arm, iters_by_arm, n_te

    # ---- Stage A: all arms on the canonical split ----
    oofA, testA, itersA, n_te = run_split(CFG.SEARCH_SEED, list(range(len(ARMS))))
    aucA = {ai: auc_score(y_np, oofA[ai]) for ai in oofA}

    print(f"\n{'='*78}")
    print("STAGE A RANKING — OOF on the canonical split (rs=%d)" % CFG.SEARCH_SEED)
    print(f"{'='*78}")
    rankedA = sorted(aucA.items(), key=lambda kv: -kv[1])
    for rank, (ai, a) in enumerate(rankedA):
        print(f"   {rank+1}. {ARMS[ai]['name']:<36} {a:.5f}  (delta vs a0 {a - aucA[0]:+.5f})")
    print(f"   Reference — V22 winner OOF: 0.94608 | V20/V23: 0.94606 | V19: 0.94599")

    top_k = [ai for ai, _ in rankedA[:CFG.TOP_K]]
    if 0 not in top_k:
        top_k.append(0)   # keep the control in Stage B as the reference run
    top_k = sorted(set(top_k))
    print(f"\n   Advancing to confirmation: {[ARMS[ai]['name'] for ai in top_k]}")

    # ---- Stage B: confirm the leaders on a second split ----
    oofB, testB, itersB, _ = run_split(CFG.CONFIRM_SEED, top_k)
    aucB = {ai: auc_score(y_np, oofB[ai]) for ai in oofB}

    print(f"\n{'='*78}")
    print("STAGE B — two-split means (rs=%d and rs=%d)" % (CFG.SEARCH_SEED, CFG.CONFIRM_SEED))
    print(f"{'='*78}")
    combined = []
    for ai in top_k:
        combined.append((ai, aucA[ai], aucB[ai], 0.5 * (aucA[ai] + aucB[ai])))
    combined.sort(key=lambda t: -t[3])
    for rank, (ai, a42, a7, ma) in enumerate(combined):
        print(f"   {rank+1}. {ARMS[ai]['name']:<36} rs{CFG.SEARCH_SEED} {a42:.5f} | "
              f"rs{CFG.CONFIRM_SEED} {a7:.5f} | mean {ma:.5f}")

    # ---- Winner ----
    winner_ai, w_a42, w_a7, w_mean = combined[0]
    winner_arm = ARMS[winner_ai]
    print(f"\n   WINNER: {winner_arm['name']}")
    print(f"      cross keys: {winner_arm['cross']} | base_margin: {winner_arm['margin']} | "
          f"prune top-{winner_arm['prune'] or 'off'}")

    # OOF reported on the canonical split (comparable with V19/V20/V22);
    # test predictions average both splits -> variance reduction on the submission
    oof_probs  = oofA[winner_ai]
    test_probs = 0.5 * (testA[winner_ai] + testB[winner_ai])
    oof_cv = auc_score(y_np, oof_probs)
    fold_scores = [auc_score(y_np[vi], oof_probs[vi])
                   for _, vi in KFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                      random_state=CFG.SEARCH_SEED).split(X)]
    best_iters = itersA[winner_ai]

    print(f"\n   Winner OOF (rs={CFG.SEARCH_SEED}): {oof_cv:.5f}")
    print(f"   Winner OOF (rs={CFG.CONFIRM_SEED}): {w_a7:.5f}")
    print(f"   Mean of splits: {w_mean:.5f}  (V22 reference 0.94608)")
    print(f"   Control a0 on rs={CFG.SEARCH_SEED}: {aucA[0]:.5f}")
    print(f"   Best iters (rs={CFG.SEARCH_SEED}): {best_iters}")

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
    print(f"V25 RESULTS — XGBoost Factorial Ablation (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(BASE_FEATURES)} base + {len(CROSS_FEATURES)} cross block + {n_te} Triple TE")
    print(f"Control a0 = V22 winner config | rs{CFG.SEARCH_SEED} OOF: {aucA[0]:.5f}")
    for ai in sorted(aucA, key=lambda k: -aucA[k]):
        print(f"   {ARMS[ai]['name']:<36} rs{CFG.SEARCH_SEED} {aucA[ai]:.5f} "
              f"(delta {aucA[ai] - aucA[0]:+.5f})")
    print(f"Winner: {winner_arm['name']} | rs{CFG.SEARCH_SEED} {w_a42:.5f} | "
          f"rs{CFG.CONFIRM_SEED} {w_a7:.5f} | mean {w_mean:.5f}")
    print(f"Gain over control (rs={CFG.SEARCH_SEED}): {w_a42 - aucA[0]:+.5f}")
    print(f"OOF CV (AUC, rs={CFG.SEARCH_SEED}): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} | iters {best_iters}")
    print(f"Submission: winner's test predictions averaged over both CV splits (no refit blend)")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
