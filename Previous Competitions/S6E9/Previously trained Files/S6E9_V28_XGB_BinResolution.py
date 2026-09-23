"""
S6E9 V28 - XGBoost Bin Resolution: does max_bin=1024 throttle the interacting model? (GPU)
================================================================================
Strategy: change ONE parameter — the histogram bin cap — on the real 366-feature
          pipeline, and measure it against an in-run control.

Reference: V22 winner (XGBoost depth 3 / 8 leaves / gamma 1.0 / colsample 0.85,
OOF 0.946076) reproduced inside a harness in V25 to 6e-6. Every XGBoost version
since V20 and every LightGBM since V10 pins max_bin to 1024; it has been 256
(V12/V14/V16), 512 (V1) and 1024, and never once raised.

Device: GPU (cuda) | Est. Time: ~40-55 min | Arms: 2 | Splits: 1 (no Stage B)

Arms (a1 differs from a0 in exactly one key):
  a0 control   = V22 winner config, max_bin = 1024      (must land on 0.946076)
  a1 no cap    = same model,         max_bin = 16384     (> income's 14,667 pool
                                                         values = unconstrained)

A first run of this version already answered the intermediate step: max_bin 4096
gave OOF 0.94607 against the control's 0.94607 — delta -0.00000 at z = -0.21, with
identical fold AUCs on folds 3 and 5 — for 1.5x the control's GPU time. That is a
null, not a closure, because 4096 still leaves income's 14,667 distinct pool values
compressed 3.6x. This run therefore goes straight to the only point on the axis
that cannot be answered by "try more bins": at 16,384 every distinct income value
owns a bin, so if the tie holds, the resolution axis is closed by construction and
never needs revisiting.

WHY THIS IS THE LAST MEASUREMENT THAT CAN ANSWER THE QUESTION. Annual_Income_USD
has 14,667 distinct values across the pool (13,214 in train) = 14.3x the cap, and
it is the ONLY column of 13 that exceeds 1024 distinct values (the next largest,
commute, is 0.81x), so every effect of this parameter is a pure income effect.
Offline screens against the frozen V23 OOF all came back at +0.000000:
  - stacking a full-resolution per-value income lookup, a 4,839-bin lookup, or
    raw/log/ordinal income onto logit(oof): -0.000003 .. +0.000010
  - corr(observed per-bin income rate, model bin-mean) = 0.987 at full resolution
    against a split-half reliability ceiling of 0.965 — the model already tracks
    the per-value rate as well as any estimator can against noisy labels
  - a median income value has only 8 training rows (69.6% have fewer than 50), so
    the binding constraint on a LOOKUP is sample size, not the histogram
But every one of those probes can only detect ADDITIVE residual income
information. None of them can see whether income's split CANDIDATES are
quantisation-starved when income interacts with ECL and Subsidy — and in a
13-column depth-3 tree, raising max_bin 1024 -> 12,000 measured +0.001675, versus
+0.000059 for a depth-1 tree: resolution is consumed ~28x more by the interacting
model. Externally, this episode's own ablation thread reports the raised cap worth
+0.00201 OOF on a 0.94172 baseline (and notes digits buy the same lever, so most
of it is already banked by our fixed digit block since V19).

INTEGRITY GUARD. XGBoost derives its quantile cuts when a DMatrix is first
binned, so an arm could silently inherit the control's 1024-bin cuts and the run
would report a false null. Each arm therefore gets its OWN DMatrix objects, and
on fold 1 the script counts the income-derived columns the trees actually split
on and the distinct thresholds they used. If a1's threshold count is not larger
than a0's, max_bin never reached the trees and the tie is meaningless — the
printout makes that visible in the log rather than hidden. (The first run of this
version probed four exact column names and reported "never used as a split" for
all of them, which was a matching bug, not an absence of income splits: the trees
label their Feature column with the DMatrix names. Both arms' differing stop
rounds and 1.5x time difference do confirm the cap took effect; the guard is here
so the log says so too.)

FREE EXTRA, at zero compute cost: both arms' OOF vectors are in memory, so the
run also reports the rank correlation between the two resolutions and the
equal-weight blend of them. Two bin settings partition the same column
differently, so they are the cheapest decorrelated pair we own — and a blend is
the one channel that measured +0.000135 at z = 9.1 in this project.

Rules honoured: 5-fold KFold(shuffle=True, rs=42), V1 house style, Kaggle paths,
oof/sub outputs only, self-contained (no previous-version OOF or submission is
read), one change per arm, and a paired DeLong gate at z > 3 computed from this
run's own OOF.
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
    VERSION_NAME = "v28"
    EXP_ID = "S6E9_V28_XGB_BinResolution"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — 5 folds kept identical to V19/V20/V22/V25/V27 so the OOF stays comparable
    N_FOLDS = 5
    RANDOM_SEED = 42

    SEARCH_SEED = 42      # canonical split — the only split this version runs
    ACCEPT_Z = 3.0        # promotion gate: paired DeLong z on honest OOF

    # Training budget: V22's winner stopped between 4105 and 5997 trees, so
    # 8000/400 reproduces it exactly (validated in V25, where a0 hit 0.94607).
    NUM_ROUND = 8000
    ES_ROUNDS = 400

    # The parameter under test. 16,384 exceeds income's 14,667 distinct pool
    # values, so this arm is NOT "more bins" — it is unconstrained bins, the last
    # point on this axis. If it ties the control there is nothing above it to try.
    BASE_BINS = 1024
    RAISED_BINS = 16384

    # Materiality gate for earning a submission slot (see the results block).
    MATERIALITY = 0.0003

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
# 4. MODEL PARAMETERS — the V22 winner, with max_bin as the only arm factor
# =============================================================================
# Booster API (DMatrix + xgb.train) because each arm needs its own freshly
# binned DMatrix; the sklearn wrapper would let XGBoost reuse cached cuts.
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
    'max_bin': CFG.BASE_BINS,
}

# a1 differs from a0 in exactly one key.
ARMS = [
    {'name': f"a0 control max_bin={CFG.BASE_BINS}",   'max_bin': CFG.BASE_BINS},
    {'name': f"a1 raised max_bin={CFG.RAISED_BINS}",  'max_bin': CFG.RAISED_BINS},
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


def rank_corr(a, b):
    """Spearman correlation via dense ranks (no scipy dependency)."""
    ra = np.empty(len(a)); ra[np.argsort(a, kind='mergesort')] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b, kind='mergesort')] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def resolution_probe(booster, cols_fold, tag):
    """
    Integrity guard: report what the trees actually split on.

    If a raised max_bin never reaches the quantile cuts, the two arms bin income
    identically and the comparison is a false null. Counting distinct thresholds
    and the tightest gap on the high-cardinality columns proves the parameter
    took effect, from the fitted model rather than from the request.
    """
    try:
        tdf = booster.trees_to_dataframe()
    except Exception as err:
        print(f"         [{tag}] resolution probe unavailable -> {type(err).__name__}")
        return
    if 'Feature' not in tdf.columns or 'Split' not in tdf.columns:
        print(f"         [{tag}] resolution probe: unexpected trees_to_dataframe schema")
        return

    # trees_to_dataframe labels the Feature column with the DMatrix feature names
    # when they were supplied, and with f{i} when they were not, so accept either.
    name_to_label = {c: (c, f"f{i}") for i, c in enumerate(cols_fold)}
    feats = tdf['Feature'].astype(str)
    probe_cols = [c for c in cols_fold
                  if any(k in c for k in ('ncome', 'income')) and c not in ('id',)]
    if not probe_cols:
        print(f"         [{tag}] resolution probe: no income-derived column in the matrix")
        return

    used_any = 0
    lines = []
    for probe in probe_cols:
        a, b = name_to_label[probe]
        rows = tdf[feats.isin([a, b])]['Split'].dropna()
        if len(rows) == 0:
            continue
        used_any += 1
        cuts = []
        for s in rows.astype(str):
            try:
                cuts.append(float(s.split()[-1]))
            except (ValueError, IndexError):
                continue
        if not cuts:
            continue
        uniq = np.unique(np.asarray(cuts, dtype='float64'))
        lines.append((uniq.size, len(cuts), probe))
    lines.sort(reverse=True)
    total_cuts = sum(n for n, _, _ in lines)
    print(f"         [{tag}] income cols used as splits: {used_any}/{len(probe_cols)} | "
          f"distinct thresholds across them: {total_cuts}")
    for n_uniq, n_splits, probe in lines[:4]:
        print(f"                 {probe:<40} distinct cuts {n_uniq:>6} | splits {n_splits}")
    del tdf
    gc.collect()

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

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: GPU (cuda) | Folds: {CFG.N_FOLDS} | Splits: 1 (no Stage B)")
    print(f"One parameter under test: max_bin {CFG.BASE_BINS} -> {CFG.RAISED_BINS}")
    print(f"Control a0 = V22 winner (depth 3, 8 leaves, gamma 1.0, colsample 0.85),")
    print(f"   so a0 must land on V22's stored OOF 0.946076 (V25 hit 0.94607 in-run)")
    print(f"Arms: {', '.join(a['name'] for a in ARMS)}")
    print(f"Annual_Income_USD carries 13,214 distinct train values = 12.9x the cap,")
    print(f"   and is the only column of 13 above it (commute is 0.81x)")
    print(f"Original data: per-fold concat, Buyer_ID dropped")
    print(f"Self-contained: no previous-version OOF or submission is read")
    print(f"Gate: paired DeLong z > {CFG.ACCEPT_Z} vs the in-run control, and "
          f"delta >= {CFG.MATERIALITY:.4f} to earn a submission slot")
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
    print(f"   Orig share of the training matrix: {len(orig_aligned)/(len(train)+len(orig_aligned)):.4%}")
    print(f"   Distinct income values — train {train['Annual_Income_USD'].nunique()}, "
          f"test {test['Annual_Income_USD'].nunique()}, "
          f"pool {pd.concat([train['Annual_Income_USD'], test['Annual_Income_USD']]).nunique()}")
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} (XGBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V19 pipeline, unchanged)...")

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

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")

    # =========================================================================
    # [3/5] BIN-RESOLUTION ABLATION (5-Fold CV, per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Bin-resolution ablation ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")
    print(f"   {len(ARMS)} arms on KFold(rs={CFG.SEARCH_SEED}) — canonical split only")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    def run_split(seed, arm_indices):
        """
        Run one KFold split for the given arms.

        The fold matrices (Triple TE) are built ONCE per fold and shared by every
        arm; arms differ only in the max_bin they hand to xgb.train, so the
        ablation costs training time only, never re-encoding. Each arm gets its
        own DMatrix objects so no arm can inherit another's quantile cuts.
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
        arm_minutes  = {ai: 0.0 for ai in arm_indices}
        # Fold 1 is the canary: if this XGBoost/CUDA build rejects a raised bin
        # cap, the arm is dropped for the whole split instead of aborting the run.
        dead = {}
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
            A_full = X_train.to_numpy('float32')     # comp train rows, then orig rows
            A_val  = X_val.to_numpy('float32')
            A_test = X_test_fold.to_numpy('float32')
            del X_train, X_val, X_test_fold
            gc.collect()

            n_comp = len(train_idx)
            n_orig = len(or_train_idx)
            y_full = y_train.values.astype('float32')
            y_val_np = y_val.values.astype('float32')

            if fold == 0:
                inc_c = [c for c in cols_fold if 'ncome' in c]
                n_hi = sum(1 for c in inc_c
                           if np.unique(A_full[:, cols_fold.index(c)]).size > CFG.BASE_BINS)
                print(f"         Final feature count: {len(cols_fold)} "
                      f"({n_te} Triple TE of {len(TARGET_ENCODE_COLS)} cols) | "
                      f"train rows {n_comp:,} comp + {n_orig:,} orig")
                print(f"         Income-derived columns whose distinct values exceed "
                      f"max_bin={CFG.BASE_BINS}: {n_hi} of {len(inc_c)} income columns")

            # ---- Train every arm on this fold's matrices ----
            for ai in arm_indices:
                arm = ARMS[ai]
                if ai in dead:
                    continue

                params = dict(XGB_BASE)
                params['max_bin'] = arm['max_bin']
                arm_start = time.time()

                try:
                    dtrain = xgb.DMatrix(A_full, label=y_full, feature_names=cols_fold)
                    dva = xgb.DMatrix(A_val, label=y_val_np, feature_names=cols_fold)

                    booster = xgb.train(params, dtrain, num_boost_round=CFG.NUM_ROUND,
                                        evals=[(dva, 'valid')], verbose_eval=False,
                                        callbacks=[xgb.callback.EarlyStopping(rounds=CFG.ES_ROUNDS,
                                                                              save_best=True)])
                    val_probs = booster.predict(dva)
                    del dtrain, dva
                    gc.collect()
                except Exception as err:
                    # A fold-1 failure is a platform limitation, not a bug in the
                    # arm: drop the arm and keep the run. Later folds must not be
                    # masked.
                    if fold > 0:
                        raise
                    dead[ai] = f"{type(err).__name__}: {str(err)[:120]}"
                    print(f"         {arm['name']:<34} DROPPED at fold 1 -> {dead[ai]}")
                    gc.collect()
                    continue

                oof_by_arm[ai][val_idx] = val_probs

                dtest = xgb.DMatrix(A_test, feature_names=cols_fold)
                test_by_arm[ai] += booster.predict(dtest) / CFG.N_FOLDS
                del dtest
                gc.collect()

                best_iter = booster.num_boosted_rounds()
                iters_by_arm[ai].append(best_iter)
                arm_minutes[ai] += (time.time() - arm_start) / 60.0

                fold_auc = auc_score(y_val_np, val_probs)
                print(f"         {arm['name']:<34} AUC {fold_auc:.5f} | trees {best_iter} "
                      f"| {arm_minutes[ai]:.1f} min cumulative")

                if fold == 0:
                    resolution_probe(booster, cols_fold, f"{arm['name'][:22]}")

                if fold == 0 and ai == arm_indices[0]:
                    gain = booster.get_score(importance_type='gain')
                    total = float(sum(gain.values())) or 1.0
                    imp = pd.DataFrame({
                        'feature': cols_fold,
                        'importance': [gain.get(c, 0.0) / total for c in cols_fold],
                    }).sort_values('importance', ascending=False)
                    print(f"\n         Top-10 feature importances ({arm['name']}, fold 1):")
                    print(imp.head(10).to_string(index=False))
                    print()

                del booster
                gc.collect()

            del A_full, A_val, A_test
            gc.collect()

            print(f"      Fold {fold+1} done in {(time.time()-fold_start)/60:.1f} min | "
                  f"split elapsed {(time.time()-t_seed)/60:.1f} min")

        if dead:
            print(f"\n   Dropped arms this split: "
                  + ", ".join(f"{ARMS[ai]['name']} ({dead[ai]})" for ai in sorted(dead)))

        for ai in arm_indices:
            if ai in dead:
                continue
            oof_auc = auc_score(y_np, oof_by_arm[ai])
            print(f"   rs={seed} OOF | {ARMS[ai]['name']:<34} {oof_auc:.5f} | "
                  f"iters {iters_by_arm[ai]} | {arm_minutes[ai]:.1f} min train time")

        # ---- paired DeLong of every arm against the control (in-run OOF only) ----
        z_by_arm = {}
        if 0 in oof_by_arm and 0 not in dead:
            print(f"\n   Paired DeLong vs a0 control (rs={seed}):")
            for ai in arm_indices:
                if ai == 0 or ai in dead:
                    continue
                a_arm, a_ctl, se, z = paired_z(y_np, oof_by_arm[ai], oof_by_arm[0])
                z_by_arm[ai] = z
                verdict = 'GATE CLEARED' if z > CFG.ACCEPT_Z else (
                    'reject' if z < -CFG.ACCEPT_Z else 'tie')
                print(f"      {ARMS[ai]['name']:<34} delta {a_arm - a_ctl:+.5f} | "
                      f"SE {se:.5f} | z {z:+.2f} {verdict}")

        return oof_by_arm, test_by_arm, iters_by_arm, arm_minutes, n_te, z_by_arm, dead

    # ---- the single split ----
    oofA, testA, itersA, minsA, n_te, zA, deadA = run_split(CFG.SEARCH_SEED, list(range(len(ARMS))))
    if 0 in deadA:
        raise RuntimeError(f"The control arm failed, so nothing in this run is comparable: {deadA[0]}")
    aucA = {ai: auc_score(y_np, oofA[ai]) for ai in oofA if ai not in deadA}

    print(f"\n{'='*78}")
    print("RANKING — OOF on the canonical split (rs=%d)" % CFG.SEARCH_SEED)
    print(f"{'='*78}")
    rankedA = sorted(aucA.items(), key=lambda kv: -kv[1])
    for rank, (ai, a) in enumerate(rankedA):
        ztxt = f" | z {zA[ai]:+.2f}" if ai in zA else ""
        print(f"   {rank+1}. {ARMS[ai]['name']:<34} {a:.5f}  "
              f"(delta vs a0 {a - aucA[0]:+.5f}{ztxt}) | {minsA[ai]:.1f} min")
    print(f"   Reference — V22 stored OOF 0.946076 | V20/V23 0.946060 | "
          f"V25 0.946094 | V26 0.946085")

    # ---- the axis verdict is read off the RAISED arm, not off the winner ----
    live = [ai for ai in range(len(ARMS)) if ai not in deadA]
    raised = [ai for ai in live if ai != 0]
    if raised:
        axis_ai = raised[0]
        d_axis = aucA[axis_ai] - aucA[0]
        z_axis = zA.get(axis_ai, float('nan'))
        axis_bins = ARMS[axis_ai]['max_bin']
    else:
        axis_ai, d_axis, z_axis, axis_bins = None, float('nan'), float('nan'), None

    winner_ai = rankedA[0][0]
    winner_arm = ARMS[winner_ai]
    print(f"\n   WINNER: {winner_arm['name']}")

    oof_probs  = oofA[winner_ai]
    test_probs = testA[winner_ai]
    oof_cv = auc_score(y_np, oof_probs)
    fold_scores = [auc_score(y_np[vi], oof_probs[vi])
                   for _, vi in KFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                      random_state=CFG.SEARCH_SEED).split(X)]
    best_iters = itersA[winner_ai]

    print(f"\n   Winner OOF (rs={CFG.SEARCH_SEED}): {oof_cv:.5f}")
    print(f"   Control a0 on rs={CFG.SEARCH_SEED}: {aucA[0]:.5f}")
    if axis_ai is not None:
        print(f"   Axis test (a{axis_ai} vs a0): delta {d_axis:+.6f} | z {z_axis:+.2f} | "
              f"max_bin {CFG.BASE_BINS} -> {axis_bins}")
    else:
        print("   Axis test UNAVAILABLE — the raised arm died on fold 1, so this run "
              "answers nothing about max_bin.")
    print(f"   Best iters: {best_iters}")
    print(f"   Winner test prediction range: {test_probs.min():.4f} .. {test_probs.max():.4f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    # Saved BEFORE any post-hoc diagnostic. In the previous run an unpacking bug
    # in the blend report threw after training and cost the artifacts of 31 min of
    # GPU work; the outputs are the product, the commentary is not.
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv")

    # ---- the free extra: are the two resolutions decorrelated enough to blend? ----
    blend_note = ""
    try:
        if len(live) >= 2:
            a_ctl = oofA[0]
            best_blend = None
            for ai in live:
                if ai == 0:
                    continue
                rho = rank_corr(oofA[0], oofA[ai])
                lg0 = np.log(np.clip(oofA[0], 1e-7, 1 - 1e-7) / (1 - np.clip(oofA[0], 1e-7, 1 - 1e-7)))
                lga = np.log(np.clip(oofA[ai], 1e-7, 1 - 1e-7) / (1 - np.clip(oofA[ai], 1e-7, 1 - 1e-7)))
                mid = 0.5 * (lg0 + lga)
                a_bl = auc_score(y_np, mid)
                a_b, a_c, se_b, z_b = paired_z(y_np, mid, a_ctl)
                d_b = a_b - a_c
                print(f"\n   Resolution pair a0 + a{ai}: rank corr {rho:.5f} | equal-logit blend "
                      f"{a_bl:.5f} (delta {d_b:+.5f}, z {z_b:+.2f})")
                if best_blend is None or a_bl > best_blend[0]:
                    best_blend = (a_bl, ai, d_b, z_b, rho)
            if best_blend is not None:
                blend_note = (f"best blend a0+a{best_blend[1]} OOF {best_blend[0]:.5f} "
                              f"(delta {best_blend[2]:+.5f}, z {best_blend[3]:+.2f}, "
                              f"rank corr {best_blend[4]:.5f})")
    except Exception as err:
        print(f"\n   Blend diagnostic failed ({type(err).__name__}: {str(err)[:90]}) — "
              f"the outputs above are unaffected.")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V28 RESULTS — XGBoost Bin Resolution (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {n_te} Triple TE = {len(FEATURES) + n_te} total")
    print(f"Control a0 (max_bin={CFG.BASE_BINS}) rs{CFG.SEARCH_SEED}: {aucA[0]:.5f}  "
          f"(V22's stored OOF is 0.946076)")
    for ai in sorted(aucA, key=lambda k: -aucA[k]):
        ztxt = f" | z {zA[ai]:+.2f}" if ai in zA else ""
        print(f"   {ARMS[ai]['name']:<34} rs{CFG.SEARCH_SEED} {aucA[ai]:.5f} "
              f"(delta {aucA[ai] - aucA[0]:+.5f}{ztxt})")
    if blend_note:
        print(f"Blend diagnostic: {blend_note}")
    print(f"Winner: {winner_arm['name']} | OOF rs{CFG.SEARCH_SEED} {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} | "
          f"iters {best_iters}")

    print(f"\nVERDICT")
    if axis_ai is None:
        print("   INCONCLUSIVE — no raised-resolution arm survived to be compared.")
    elif d_axis >= CFG.MATERIALITY and z_axis > CFG.ACCEPT_Z:
        print(f"   Resolution axis is LIVE: {d_axis:+.6f} at z {z_axis:+.2f} clears both the "
              f"significance gate and the {CFG.MATERIALITY:.4f} materiality gate.")
        print(f"   -> resolution is real even though every additive proxy said no. Adopt "
              f"max_bin={axis_bins} as the new base for V29/V30 and bank the gain.")
    elif z_axis > CFG.ACCEPT_Z:
        print(f"   Significant but immaterial: {d_axis:+.6f} at z {z_axis:+.2f} sits inside the "
              f"public-LB noise band (paired SE 0.000053-0.000129).")
        print(f"   -> keep max_bin={axis_bins} as free OOF for the later build; do NOT spend a "
              f"submission slot on it alone.")
    elif abs(d_axis) < CFG.MATERIALITY:
        print(f"   Resolution axis CLOSED for the interacting model: {axis_bins // CFG.BASE_BINS}x "
              f"more bins moved it {d_axis:+.6f}.")
        print(f"   -> every additive proxy plus this direct test now agree. Retire max_bin, "
              f"keep {CFG.BASE_BINS}, and move to the fold-count and two-view channels.")
    else:
        print(f"   Raised bins REGRESSED ({d_axis:+.6f} at z {z_axis:+.2f}) — 1024 was already "
              f"past the optimum; the axis is closed from this side too.")
    if winner_ai == 0:
        print(f"   The control won, so sub_{CFG.VERSION_NAME}.csv reproduces V22's estimator on "
              f"the V22 matrix and stays comparable with it.")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*80}")
