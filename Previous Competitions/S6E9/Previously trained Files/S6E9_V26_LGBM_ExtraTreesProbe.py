"""
S6E9 V26 - LightGBM ExtraTrees-Bias Probe on the Artifact Matrix (CPU)
================================================================================
Strategy: test the one inductive bias we have never used — fully randomised split
          thresholds — against an in-run LightGBM control on V22's exact matrix

Reference: V25 (factorial ablation, OOF 0.94609 / LB 0.94628). Every axis that
shares V25's verdict is closed: features (cross keys rejected at z = -3.98),
gain pruning (z = -2.77), recipe prior (a tie), and model families that still
pick axis-parallel thresholds by gain (XGBoost > LightGBM > CatBoost).

Device: CPU (LightGBM) | Est. Time: ~150-200 min

Why random splits could be the missing bias: the training pool is 668k rows
produced by rejection sampling from a 10k-row original, so exact input values
recur thousands of times and the label is close to a deterministic lookup on
them. A gain-seeking depth-3/4 tree can only approximate that lookup with shared
thresholds; an infinitely-grown tree with RANDOM thresholds over many bins can
memorise value boundaries instead of averaging over them. Greedy trees are not
allowed to do that (they always pick the best split), so this is not reachable by
tuning the boosting we already run.

Arms (one fold build shared by all, each arm changes one factor vs control):
  a0 control                 = V19's LightGBM params verbatim, so a0 must
                               reproduce V19's stored OOF 0.945987
  a1 extra_trees             = a0 + randomised split thresholds (the pure test)
  a2 extra_trees + lookup    = a1 + RF-style capacity: unlimited depth, 128
                               leaves, per-node feature bagging, no column cap
  a3 extra_trees + wide/weak = a1 + V22's winning direction (shallow, many
                               columns, low learning rate) to see whether
                               random splits need more regularisation, not less

Protocol: Stage A on the canonical KFold(rs=42) with a paired DeLong of every arm
against a0, computed from this run's own OOF vectors. Stage B (rs=7) runs ONLY if
an arm clears the z > 3 gate, so a negative answer costs one split, not two.
Outputs: oof_v26.csv / sub_v26.csv of the best arm. NOTE: this is a CV probe —
unless a printed verdict says a GATE CLEARED, the submission line stays V23.
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
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Check sklearn version for TargetEncoder compatibility
print(f"scikit-learn version: {sklearn_version}")
print(f"LightGBM version: {lgb.__version__}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v26"
    EXP_ID = "S6E9_V26_LGBM_ExtraTreesProbe"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — 5 folds kept identical to V19/V20/V22/V25 so the OOF stays comparable
    N_FOLDS = 5
    RANDOM_SEED = 42

    # Probe control
    SEARCH_SEED = 42      # canonical split
    CONFIRM_SEED = 7      # only run when an arm clears the gate
    ACCEPT_Z = 3.0        # our promotion gate: paired DeLong z on honest OOF

    # Categorical columns that get a generator-lift feature
    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS — V19 control and the extra_trees arms
# =============================================================================
# LGB_BASE is V19's recipe unchanged (that is what makes a0 a control worth
# trusting), with a shorter tree budget so four arms fit in one notebook session.
# Known quirk carried over from V3/V10/V19 on purpose: LightGBM only applies
# `subsample` when `subsample_freq > 0`, so bagging has been silently off in our
# whole LightGBM lineage. It is NOT enabled here — that would confound the arm
# comparison with a second, unrelated change.
LGB_BASE = {
    'n_estimators': 10000,
    'learning_rate': 0.02,
    'max_depth': 5,
    'num_leaves': 32,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'reg_alpha': 0.071,
    'reg_lambda': 2.0,
    'max_bin': 1024,
    'min_data_in_bin': 3,
    'random_state': CFG.RANDOM_SEED,
    'feature_pre_filter': False,
    'metric': 'auc',
    'importance_type': 'gain',
    'n_jobs': -1,
    'verbose': -1,
    'device': 'cpu',
}
ES_ROUNDS = 400

# In LightGBM 4.x the ExtraTrees path is implemented through sampled-bin
# histograms, so both keys together are what "randomised split thresholds" means
# for this engine; a0 keeps greedy full-histogram splits.
EXTRA_TREES = {'extra_trees': True, 'split_histogram_sampling': True}

ARMS = [
    {'name': "a0 control (V19 params)",          'params': {}},
    {'name': "a1 extra_trees random splits",     'params': dict(EXTRA_TREES)},
    {'name': "a2 extra_trees + lookup capacity", 'params': dict(EXTRA_TREES, **{
        'max_depth': -1,          # let trees grow to the leaf floor: the lookup bias
        'num_leaves': 128,
        'min_child_samples': 200,
        'colsample_bytree': 1.0,  # no per-tree cap; randomness does that job
        'feature_fraction_bynode': 0.3,
        'learning_rate': 0.05,
    })},
    {'name': "a3 extra_trees + wide weak many",  'params': dict(EXTRA_TREES, **{
        'max_depth': 3,           # V22's winning direction, applied to random splits
        'num_leaves': 8,
        'min_child_samples': 50,
        'colsample_bytree': 0.85,
        'learning_rate': 0.01,
    })},
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
# 6. FEATURE ENGINEERING (V19/V22 pipeline, unchanged)
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

    # Helper column is object dtype — LightGBM cannot take it
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

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: CPU (LightGBM) | Folds: {CFG.N_FOLDS}")
    print(f"Probe: does randomised split thresholds (ExtraTrees bias) beat greedy LightGBM?")
    print(f"Control a0 = V19's params verbatim, so a0 must land on V19's stored OOF 0.945987")
    print(f"Arms: {', '.join(a['name'] for a in ARMS)}")
    print(f"Features: V19/V22 artifact matrix unchanged (fixed digits + lift/novelty + structural flags)")
    print(f"Original data: USED (per-fold concat, Buyer_ID dropped)")
    print(f"Self-contained: no previous-version OOF or submission is read")
    print(f"Gate: an arm advances only on paired DeLong z > {CFG.ACCEPT_Z} vs the in-run control")
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
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} (LightGBM handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V19 pipeline, unchanged)...")

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

    # 2m. Redundancy selection
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # 2n. Define feature groups
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")

    # =========================================================================
    # [3/5] FACTORIAL PROBE (5-Fold CV, per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] ExtraTrees-bias probe ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")
    print(f"   Stage A: {len(ARMS)} arms on KFold(rs={CFG.SEARCH_SEED}) — canonical split")
    print(f"   Stage B: only if an arm clears z > {CFG.ACCEPT_Z}")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    def run_split(seed, arm_indices):
        """
        Run one KFold split for the given arms.

        The fold matrices (Triple TE) are built ONCE per fold and shared by every
        arm, so the probe costs training time only, never re-encoding.
        Returns (oof by arm, test by arm, iters by arm, n_te, gate-passed arms).
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

            print(f"         Train (comp+orig): {X_train.shape} | "
                  f"Val: {X_val.shape} | Test: {X_test_fold.shape}")

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

            # Drop original string columns (TE columns replace them)
            cols_to_drop_after_te = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
            X_train = X_train.drop(columns=cols_to_drop_after_te, errors='ignore')
            X_val   = X_val.drop(columns=cols_to_drop_after_te, errors='ignore')
            X_test_fold = X_test_fold.drop(columns=cols_to_drop_after_te, errors='ignore')

            # Any categorical left over is dropped; LightGBM gets a numeric frame
            string_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
            if string_cols:
                X_train = X_train.drop(columns=string_cols)
                X_val   = X_val.drop(columns=string_cols)
                X_test_fold = X_test_fold.drop(columns=string_cols)

            X_train = X_train.astype('float32')
            X_val   = X_val.astype('float32')
            X_test_fold = X_test_fold.astype('float32')

            if fold == 0:
                print(f"         Final feature count: {len(X_train.columns)} "
                      f"({n_te} Triple TE of {len(TARGET_ENCODE_COLS)} cols)")

            # ---- Train every arm on this fold's matrices ----
            for ai in arm_indices:
                arm = ARMS[ai]
                params = dict(LGB_BASE)
                params.update(arm['params'])

                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=False),
                                   lgb.log_evaluation(period=0)])

                val_probs = clf.predict_proba(X_val)[:, 1]
                oof_by_arm[ai][val_idx] = val_probs
                test_by_arm[ai] += clf.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

                best_iter = (clf.best_iteration_ if clf.best_iteration_
                             else params['n_estimators'])
                iters_by_arm[ai].append(best_iter)

                fold_auc = auc_score(y_val.values, val_probs)
                print(f"         {arm['name']:<34} AUC {fold_auc:.5f} | trees {best_iter}")

                if fold == 0 and ai == arm_indices[0]:
                    shares = clf.feature_importances_
                    total = float(np.nansum(shares))
                    imp = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': shares / total if total > 0 else shares,
                    }).sort_values('importance', ascending=False)
                    print(f"\n         Top-10 feature importances ({arm['name']}, fold 1):")
                    print(imp.head(10).to_string(index=False))
                    print()

                del clf
                gc.collect()

            del X_train, X_val, X_test_fold, y_train, y_val
            gc.collect()

            print(f"      Fold {fold+1} done in {(time.time()-fold_start)/60:.1f} min | "
                  f"split elapsed {(time.time()-t_seed)/60:.1f} min")

        for ai in arm_indices:
            oof_auc = auc_score(y_np, oof_by_arm[ai])
            print(f"   rs={seed} OOF | {ARMS[ai]['name']:<34} {oof_auc:.5f} | "
                  f"iters {iters_by_arm[ai]}")

        # ---- paired DeLong of every arm against the control (in-run OOF only) ----
        z_by_arm = {}
        if 0 in oof_by_arm:
            print(f"\n   Paired DeLong vs a0 control (rs={seed}):")
            for ai in arm_indices:
                if ai == 0:
                    continue
                a_arm, a_ctl, se, z = paired_z(y_np, oof_by_arm[ai], oof_by_arm[0])
                z_by_arm[ai] = z
                verdict = 'GATE CLEARED' if z > CFG.ACCEPT_Z else (
                    'reject' if z < -CFG.ACCEPT_Z else 'tie')
                print(f"      {ARMS[ai]['name']:<34} delta {a_arm - a_ctl:+.5f} | "
                      f"SE {se:.5f} | z {z:+.2f} {verdict}")

        return oof_by_arm, test_by_arm, iters_by_arm, n_te, z_by_arm

    # ---- Stage A: all arms on the canonical split ----
    oofA, testA, itersA, n_te, zA = run_split(CFG.SEARCH_SEED, list(range(len(ARMS))))
    aucA = {ai: auc_score(y_np, oofA[ai]) for ai in oofA}

    print(f"\n{'='*78}")
    print("STAGE A RANKING — OOF on the canonical split (rs=%d)" % CFG.SEARCH_SEED)
    print(f"{'='*78}")
    rankedA = sorted(aucA.items(), key=lambda kv: -kv[1])
    for rank, (ai, a) in enumerate(rankedA):
        ztxt = f" | z {zA[ai]:+.2f}" if ai in zA else ""
        print(f"   {rank+1}. {ARMS[ai]['name']:<34} {a:.5f}  "
              f"(delta vs a0 {a - aucA[0]:+.5f}{ztxt})")
    print(f"   Reference — V19 LightGBM stored OOF: 0.945987 | V22 XGBoost 0.946076 | V20/V23 0.946060")

    passed = [ai for ai in zA if zA[ai] > CFG.ACCEPT_Z]
    top_k = sorted(set([ai for ai, _ in rankedA[:3]] + [0] + passed))

    if passed:
        print(f"\n   Arms clearing z > {CFG.ACCEPT_Z}: {[ARMS[ai]['name'] for ai in passed]}")
        print(f"   Advancing to confirmation: {[ARMS[ai]['name'] for ai in top_k]}")
        oofB, testB, itersB, _, zB = run_split(CFG.CONFIRM_SEED, top_k)
        aucB = {ai: auc_score(y_np, oofB[ai]) for ai in oofB}
        print(f"\n{'='*78}")
        print("STAGE B — two-split means (rs=%d and rs=%d)" % (CFG.SEARCH_SEED, CFG.CONFIRM_SEED))
        print(f"{'='*78}")
        combined = [(ai, aucA[ai], aucB[ai], 0.5 * (aucA[ai] + aucB[ai])) for ai in top_k]
        combined.sort(key=lambda t: -t[3])
        for rank, (ai, a42, a7, ma) in enumerate(combined):
            print(f"   {rank+1}. {ARMS[ai]['name']:<34} rs{CFG.SEARCH_SEED} {a42:.5f} | "
                  f"rs{CFG.CONFIRM_SEED} {a7:.5f} | mean {ma:.5f}")
        winner_ai, w_a42, w_a7, w_mean = combined[0]
        two_split = True
    else:
        print(f"\n   NO ARM CLEARED z > {CFG.ACCEPT_Z} on rs={CFG.SEARCH_SEED} — "
              f"Stage B skipped, the probe answers at one split.")
        winner_ai, w_a42 = rankedA[0][0], rankedA[0][1]
        w_a7, w_mean, two_split = float('nan'), w_a42, False

    winner_arm = ARMS[winner_ai]
    print(f"\n   WINNER: {winner_arm['name']}")
    print(f"      param override: {winner_arm['params']}")

    # OOF reported on the canonical split (comparable with V19/V20/V22/V25)
    oof_probs  = oofA[winner_ai]
    if two_split:
        test_probs = 0.5 * (testA[winner_ai] + testB[winner_ai])
    else:
        test_probs = testA[winner_ai]
    oof_cv = auc_score(y_np, oof_probs)
    fold_scores = [auc_score(y_np[vi], oof_probs[vi])
                   for _, vi in KFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                      random_state=CFG.SEARCH_SEED).split(X)]
    best_iters = itersA[winner_ai]

    print(f"\n   Winner OOF (rs={CFG.SEARCH_SEED}): {oof_cv:.5f}")
    if two_split:
        print(f"   Winner OOF (rs={CFG.CONFIRM_SEED}): {w_a7:.5f} | mean {w_mean:.5f}")
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
    print(f"V26 RESULTS — LightGBM ExtraTrees-Bias Probe (CPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {n_te} Triple TE = {len(FEATURES) + n_te} total")
    print(f"Control a0 (V19 params) rs{CFG.SEARCH_SEED}: {aucA[0]:.5f}  "
          f"(V19's stored OOF is 0.945987)")
    for ai in sorted(aucA, key=lambda k: -aucA[k]):
        ztxt = f" | z {zA[ai]:+.2f}" if ai in zA else ""
        print(f"   {ARMS[ai]['name']:<34} rs{CFG.SEARCH_SEED} {aucA[ai]:.5f} "
              f"(delta {aucA[ai] - aucA[0]:+.5f}{ztxt})")
    print(f"Winner: {winner_arm['name']} | OOF rs{CFG.SEARCH_SEED} {w_a42:.5f}")
    print(f"OOF CV (AUC, rs={CFG.SEARCH_SEED}): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} | iters {best_iters}")
    if passed:
        print(f"VERDICT: an arm cleared z > {CFG.ACCEPT_Z} — this model is a submission candidate")
    else:
        print(f"VERDICT: no arm cleared z > {CFG.ACCEPT_Z} — CV probe only, "
              f"the submission line stays V23 (LB 0.94641)")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
