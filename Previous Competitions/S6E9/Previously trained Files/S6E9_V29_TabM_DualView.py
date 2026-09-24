"""
S6E9 V29 - TabM + In-Run XGBoost Control: does a non-tree view add anything? (GPU)
================================================================================
Strategy: put a neural tabular model on the V19/V22 artifact matrix for the first
          time, in the same script as the incumbent GBM, and ask two questions of
          it — can it stand near the GBM, and is it decorrelated enough to blend.

Reference: V22 winner (XGBoost depth 3 / 8 leaves / gamma 1.0 / colsample 0.85,
OOF 0.946076, reproduced in-run by V25 0.94607 and V28 a0 0.94607 with LB 0.94640
identical to V22's). V6 = TabM with pytabkit, OOF 0.94585 / LB 0.94606 in 43.9 min
on the pre-V19 feature set.

Device: GPU (cuda) | Est. Time: ~60-80 min | Arms: GBM control + TabM | Splits: 1

WHY A NON-TREE MODEL NOW. Four axes are closed by direct measurement, not by
assumption: features (91 pairwise lifts and 201 arithmetic derivatives at
+0.000000 each), distribution matching (an ORACLE cell recalibration costs
-0.0033), the objective (rank:pairwise -0.000329 at z = -42.6), and bin
resolution (V28: max_bin -> unconstrained moved the OOF -0.000002 at z = -0.40).
What is left is the estimator's inductive bias. And the one non-tree result we
actually measured in this project is a *blend*: V23's OOF paired with V6's TabM
gave +0.000093 at z = +7.5, the best non-tree leg of the 25 tested, while pairing
with V17's RealMLP gave +0.000034 and with V13's DCN-V2 +0.000038. The ranking of
those gains followed decorrelation x quality, exactly as in the literature (S6E5
5th: "a 0.918-AUC DeepFFM was worth more than a fourth 0.953 LightGBM"; S5E3 2nd:
+0.00118 from adding one SVC to two GBMs; S6E6 9th: +0.0019 from a 63-model
diverse stack).

BUT NO VERSION HAS EVER RUN A NEURAL MODEL ON THE V19+ ARTIFACT MATRIX. V6 used
83 pre-V19 features, V7's FT-Transformer and V13's DCN-V2 likewise, and V17's
RealMLP ran on the *corrupted* digit block (the `col // 10**k % 10` float
floor-div that mis-reads 89.63% of the commute tenths, fixed only in V19). So
"neural nets lose on this matrix" is an untested inference, not a result.

WHY TABM SPECIFICALLY. It is the only non-tree family that was simultaneously
strong and cheap here (OOF 0.94585 in 43.9 min vs FT-Transformer's 0.94545 in
159.1 min), its batch-ensemble k-heads give free variance reduction, and the
PLR/periodic numeric embeddings it supports are the right inductive bias for this
data: income is an integer lattice of 13,214 distinct values whose per-value rate
the trees approximate through 1024 bins. Externally, periodic embeddings measured
+0.001309 AUC over a scalar MLP in S6E2, PLR lifted an MLP's average benchmark
rank from 8.5 to 3.0, and S6E8 was won by a single RealMLP that beat its XGBoost.

BUDGET CONTROL, LEAK-FREE. TabM over all 312 fitted columns costs roughly k x
d_in x d_block, and at k=16 on 312 inputs that blows the ~130 min ceiling and
risks memory. So each fold feeds the NN only the **top CFG.NN_TOP_N columns by the
gain of that fold's own freshly trained control model**. The ranking therefore
comes from that fold's training rows alone — nested, no validation label is ever
seen — and it cuts both compute and memory by ~2.6x. TabM's k is 4 and the batch
is 4096 rather than V6's 16 and 1024 for the same reason. A fold-0 canary enforces
the ceiling: if the NN's first fold exceeds CFG.NN_BUDGET_MIN minutes the NN arm is
dropped for the remaining folds and the fold-0 diagnostic is still reported.

BLEND HONESTY. The weight is chosen CROSS-FITTED — on the four folds that are not
being scored — because on this same board a non-linear combiner measured CV +0.0005
against LB -0.0004 (~8 sigma) while a plain linear one transferred to within
0.00003 (thread 741447). That is exactly how V18 lost its way: forward-stepwise
weights fitted on the OOF they were scored on. One parameter, one linear weight,
no stacker, no hill climber.

Rules honoured: 5-fold KFold(shuffle=True, rs=42), V1 house style, Kaggle paths,
oof/sub outputs only, self-contained (both views trained in this script; no
previous-version OOF or submission is read), paired DeLong gate at z > 3.
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
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Auto-install pytabkit (the path V6 proved on Kaggle)
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
print(f"xgboost version: {xgb.__version__}")
print(f"PyTorch version: {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v29"
    EXP_ID = "S6E9_V29_TabM_DualView"
    DEVICE = DEVICE

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # Target
    TARGET = 'Will_Buy_EV'

    # CV — 5 folds kept identical to V19/V20/V22/V25/V27/V28 so the OOF stays comparable
    N_FOLDS = 5
    RANDOM_SEED = 42

    SEARCH_SEED = 42      # canonical split — the only split this version runs
    ACCEPT_Z = 3.0        # promotion gate: paired DeLong z on honest OOF
    MATERIALITY = 0.0003  # and the gain must exceed the public-LB noise band

    # GBM control budget (V22's winner stopped between 4105 and 5997 trees)
    NUM_ROUND = 8000
    ES_ROUNDS = 400

    # ---- TabM arm ----
    # V6 used tabm_k=16 / d_block=256 / n_blocks=3 / batch 1024 / pwl d_embedding=16
    # on 83 columns. Here the matrix is 312 columns, so k, the batch and the
    # embedding width are reduced and the input is budgeted to NN_TOP_N columns.
    NN_TOP_N = 120               # per-fold, by that fold's own control-model gain
    NN_BUDGET_MIN = 20.0         # fold-0 canary: drop the NN arm past this
    TABM_PARAMS = {
        'device': DEVICE,
        'verbosity': 0,
        'arch_type': 'tabm-mini-normal',
        'tabm_k': 4,                 # batch-ensemble heads (V6 used 16)
        'num_emb_type': 'pwl',       # piecewise-linear numeric embeddings
        'd_embedding': 16,
        'd_block': 128,
        'n_blocks': 2,
        'dropout': 0.2,
        'batch_size': 4096,          # V6 used 1024; 4x fewer steps per epoch
        'lr': 1e-3,
        'n_epochs': 25,
        'patience': 8,
        'weight_decay': 1e-3,
        'random_state': RANDOM_SEED,   # bare name: CFG is not bound inside its own body
    }
    # Blend weight grid, coarse on purpose: a one-parameter choice made on 535k
    # rows per fold has negligible estimation noise, and a coarse grid removes the
    # temptation to fit the scored data.
    BLEND_WEIGHTS = [round(w, 2) for w in np.arange(0.0, 0.71, 0.05)]

    # Categorical columns that get a generator-lift feature
    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS — the incumbent estimator
# =============================================================================
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
    'max_bin': 1024,               # V28 showed 16x more bins is worth -0.000002
}

# =============================================================================
# 5. METRICS, PAIRED DELONG, AND THE CROSS-FITTED BLEND
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


def _logit(p):
    p = np.clip(np.asarray(p, dtype='float64'), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def best_weight(y_true, zA, zB, weights):
    """The single linear weight that maximises AUC of (1-w)A + wB on these rows."""
    best_w, best_a = 0.0, -1.0
    for w in weights:
        a = auc_score(y_true, zA if w == 0.0 else (1.0 - w) * zA + w * zB)
        if a > best_a:
            best_a, best_w = a, w
    return best_w


def cross_fitted_blend(y_true, oof_gbm, oof_nn, fold_id, weights):
    """
    Blend whose weight is chosen on folds other than the one it is applied to.

    V18's lesson and thread 741447 both point the same way: a combiner fitted on
    the OOF it is then judged on reports gains that the board does not pay. So
    each fold's rows get the weight selected on the other four folds' rows.
    Returns (honest blend vector, per-fold weights, full-OOF weight).
    """
    zA, zB = _logit(oof_gbm), _logit(oof_nn)
    out = np.array(zA, dtype='float64')
    per_fold = []
    for f in np.unique(fold_id):
        tr = fold_id != f
        te = fold_id == f
        if tr.sum() == 0 or te.sum() == 0:
            per_fold.append(0.0)
            continue
        w = best_weight(y_true[tr], zA[tr], zB[tr], weights)
        per_fold.append(w)
        out[te] = (1.0 - w) * zA[te] + w * zB[te]
    w_full = best_weight(y_true, zA, zB, weights)
    return out, per_fold, w_full

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
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS} | Splits: 1 (no Stage B)")
    print("Question: does a non-tree view stand near the incumbent GBM on the")
    print("          artifact matrix, and is it decorrelated enough to blend?")
    print(f"View A = XGBoost V22 winner (max_bin 1024) — must land on 0.946076")
    print(f"View B = TabM k={CFG.TABM_PARAMS['tabm_k']} d_block={CFG.TABM_PARAMS['d_block']} "
          f"n_blocks={CFG.TABM_PARAMS['n_blocks']} "
          f"emb={CFG.TABM_PARAMS['num_emb_type']}x{CFG.TABM_PARAMS['d_embedding']}, "
          f"{CFG.TABM_PARAMS['n_epochs']} epochs")
    print(f"   per-fold input: top {CFG.NN_TOP_N} columns by that fold's own control gain")
    print(f"   fold-0 budget canary: drop the NN arm if it exceeds {CFG.NN_BUDGET_MIN:.0f} min")
    print(f"Blend: ONE linear weight, chosen cross-fitted (never on scored rows)")
    print(f"Original data: per-fold concat, Buyer_ID dropped")
    print(f"Self-contained: both views trained here; no previous OOF/submission is read")
    print(f"Gate: paired DeLong z > {CFG.ACCEPT_Z} vs the GBM and delta >= {CFG.MATERIALITY:.4f}")
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
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} "
          f"(XGBoost handles NaN natively; the NN arm median-fills them)")

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
    # [3/5] DUAL-VIEW RUN (5-Fold CV, per-fold orig concat + Triple TE)
    # =========================================================================
    print(f"\n[3/5] Dual-view run ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")
    print(f"   KFold(rs={CFG.SEARCH_SEED}) — canonical split only")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    oof_gbm   = np.zeros(len(y))
    oof_nn    = np.zeros(len(y))
    test_gbm  = np.zeros(len(test_X))
    test_nn   = np.zeros(len(test_X))
    fold_id   = np.full(len(y), -1, dtype='int32')
    iters_gbm = []
    nn_alive  = True
    nn_minutes = 0.0
    gbm_minutes = 0.0
    n_te = 0
    top_cols_seen = []
    t_run = time.time()

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEARCH_SEED)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEARCH_SEED)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X), orig_splits)):

        fold_start = time.time()
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        # ---- Triple Target Encoding (auto, 10.0, 100.0), built once per fold ----
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
        A_full = X_train.to_numpy('float32')
        A_val  = X_val.to_numpy('float32')
        A_test = X_test_fold.to_numpy('float32')
        del X_train, X_val, X_test_fold
        gc.collect()

        n_comp = len(train_idx)
        n_orig = len(or_train_idx)
        y_full = y_train.values.astype('float32')
        y_val_np = y_val.values.astype('float32')
        fold_id[val_idx] = fold

        if fold == 0:
            print(f"         Final feature count: {len(cols_fold)} "
                  f"({n_te} Triple TE of {len(TARGET_ENCODE_COLS)} cols) | "
                  f"train rows {n_comp:,} comp + {n_orig:,} orig")

        # ---------------- View A: the incumbent XGBoost control ----------------
        gbm_start = time.time()
        dtrain = xgb.DMatrix(A_full, label=y_full, feature_names=cols_fold)
        dva = xgb.DMatrix(A_val, label=y_val_np, feature_names=cols_fold)
        booster = xgb.train(XGB_BASE, dtrain, num_boost_round=CFG.NUM_ROUND,
                            evals=[(dva, 'valid')], verbose_eval=False,
                            callbacks=[xgb.callback.EarlyStopping(rounds=CFG.ES_ROUNDS,
                                                                  save_best=True)])
        val_probs = booster.predict(dva)
        oof_gbm[val_idx] = val_probs
        dtest = xgb.DMatrix(A_test, feature_names=cols_fold)
        test_gbm += booster.predict(dtest) / CFG.N_FOLDS
        del dtrain, dva, dtest
        gc.collect()
        iters_gbm.append(booster.num_boosted_rounds())
        gbm_minutes += (time.time() - gbm_start) / 60.0
        a_gbm = auc_score(y_val_np, val_probs)
        print(f"         XGBoost control                     AUC {a_gbm:.5f} | "
              f"trees {iters_gbm[-1]} | {gbm_minutes:.1f} min cumulative")

        # Columns for the NN, ranked by THIS fold's control model on THIS fold's
        # training rows only — nested, so no validation label informs the subset.
        gain = booster.get_score(importance_type='gain')
        order = sorted(cols_fold, key=lambda c: -gain.get(c, 0.0))
        nn_cols = order[:CFG.NN_TOP_N]
        nn_idx = [cols_fold.index(c) for c in nn_cols]
        if fold == 0:
            top_cols_seen = nn_cols
            hi = [c for c in nn_cols[:12]]
            print(f"         NN input = top {len(nn_cols)} of {len(cols_fold)} by fold-1 gain; "
                  f"first 12: {hi}")
        del booster, gain, order
        gc.collect()

        # ---------------- View B: TabM ----------------
        if not nn_alive:
            print(f"         TabM skipped (fold-1 time canary)")
            print(f"      Fold {fold+1} done in {(time.time()-fold_start)/60:.1f} min | "
                  f"elapsed {(time.time()-t_run)/60:.1f} min")
            del A_full, A_val, A_test
            gc.collect()
            continue

        nn_start = time.time()
        try:
            # The original rows carry NaNs; medians come from the fold's TRAINING
            # rows only (comp train itself has no missing values).
            col_mean = np.nanmean(A_full[:, nn_idx], axis=0)
            fills = np.where(np.isnan(col_mean), 0.0, col_mean)

            def nn_frame(A, rows):
                B = A[:, nn_idx]
                nan_mask = np.isnan(B)
                if nan_mask.any():
                    B = B.copy()
                    B[nan_mask] = np.broadcast_to(fills, B.shape)[nan_mask]
                return pd.DataFrame(B, columns=nn_cols, index=rows)

            nn_train = nn_frame(A_full, np.arange(A_full.shape[0]))
            nn_val   = nn_frame(A_val, np.arange(A_val.shape[0]))
            nn_test  = nn_frame(A_test, np.arange(A_test.shape[0]))
            y_tr_series = pd.Series(y_full)

            model = TabM_D_Classifier(**CFG.TABM_PARAMS)
            model.fit(nn_train, y_tr_series, X_val=nn_val, y_val=pd.Series(y_val_np))

            vp = model.predict_proba(nn_val)
            if hasattr(vp, 'ndim') and np.ndim(vp) == 2:
                vp = vp[:, 1]
            vp = np.asarray(vp, dtype='float64').ravel()
            tp = model.predict_proba(nn_test)
            if hasattr(tp, 'ndim') and np.ndim(tp) == 2:
                tp = tp[:, 1]
            tp = np.asarray(tp, dtype='float64').ravel()

            oof_nn[val_idx] = np.clip(vp, 1e-7, 1 - 1e-7)
            test_nn += np.clip(tp, 1e-7, 1 - 1e-7) / CFG.N_FOLDS

            nn_fold_min = (time.time() - nn_start) / 60.0
            nn_minutes += nn_fold_min
            a_nn = auc_score(y_val_np, oof_nn[val_idx])
            rho = rank_corr(_logit(val_probs), _logit(oof_nn[val_idx]))
            print(f"         TabM                                AUC {a_nn:.5f} | "
                  f"{nn_fold_min:.1f} min this fold | rank corr vs GBM {rho:.5f}")

            if fold == 0 and nn_fold_min > CFG.NN_BUDGET_MIN:
                print(f"         BUDGET CANARY: TabM fold 1 took {nn_fold_min:.1f} min, over the "
                      f"{CFG.NN_BUDGET_MIN:.0f} min ceiling (projected "
                      f"{nn_fold_min * CFG.N_FOLDS:.0f} min). Dropping the NN arm for the "
                      f"remaining folds; the fold-1 diagnostic above still stands.")
                nn_alive = False

            del nn_train, nn_val, nn_test, model
            gc.collect()
        except Exception as err:
            print(f"         TabM FAILED at fold {fold+1} -> {type(err).__name__}: {str(err)[:140]}")
            if fold == 0:
                nn_alive = False
                print(f"         NN arm abandoned — this run answers the control only.")
        finally:
            gc.collect()

        del A_full, A_val, A_test
        gc.collect()
        print(f"      Fold {fold+1} done in {(time.time()-fold_start)/60:.1f} min | "
              f"elapsed {(time.time()-t_run)/60:.1f} min")

    # Unfilled OOF slots are exactly 0.0, so this counts the folds TabM scored.
    nn_folds_done = len(np.unique(fold_id[oof_nn > 0])) if (oof_nn > 0).any() else 0
    print(f"\n   GBM control train time {gbm_minutes:.1f} min | TabM {nn_minutes:.1f} min over "
          f"{nn_folds_done} fold(s)")
    if nn_folds_done == CFG.N_FOLDS:
        print(f"   rs={CFG.SEARCH_SEED} OOF | XGBoost control {auc_score(y_np, oof_gbm):.5f} | "
              f"iters {iters_gbm}")
        print(f"   rs={CFG.SEARCH_SEED} OOF | TabM            {auc_score(y_np, oof_nn):.5f}")
    else:
        print(f"   rs={CFG.SEARCH_SEED} OOF | XGBoost control {auc_score(y_np, oof_gbm):.5f} | "
              f"iters {iters_gbm}  (TabM incomplete: {nn_folds_done}/{CFG.N_FOLDS} folds)")

    # =========================================================================
    # Decisions taken from this run's own OOF vectors
    # =========================================================================
    a_gbm_oof = auc_score(y_np, oof_gbm)
    blend_ok = False
    blend_oof = None
    blend_note = ""
    w_test = 0.0
    try:
        if nn_folds_done == CFG.N_FOLDS:
            rho_full = rank_corr(oof_gbm, oof_nn)
            zA, zB = _logit(oof_gbm), _logit(oof_nn)
            w_full = best_weight(y_np, zA, zB, CFG.BLEND_WEIGHTS)
            a_plain = auc_score(y_np, (1 - w_full) * zA + w_full * zB)
            blend_oof, per_fold_w, _ = cross_fitted_blend(y_np, oof_gbm, oof_nn, fold_id,
                                                         CFG.BLEND_WEIGHTS)
            a_honest = auc_score(y_np, blend_oof)
            a_h_b, a_h_c, se_h, z_h = paired_z(y_np, blend_oof, oof_gbm)
            d_h = a_h_b - a_h_c
            a_i_b, a_i_c, se_i, z_i = paired_z(y_np, (1 - w_full) * zA + w_full * zB, oof_gbm)
            d_i = a_i_b - a_i_c
            a_nn = auc_score(y_np, oof_nn)
            print(f"\n   TabM standalone {a_nn:.5f} vs control {a_gbm_oof:.5f} "
                  f"(delta {a_nn - a_gbm_oof:+.6f}) | rank corr {rho_full:.5f}")
            print(f"   Blend, weight fitted on scored rows: w={w_full:.2f} -> {a_plain:.5f} "
                  f"(delta {d_i:+.6f}, z {z_i:+.2f})")
            print(f"   Blend, weight CROSS-FITTED: per-fold w {per_fold_w} -> {a_honest:.5f} "
                  f"(delta {d_h:+.6f}, SE {se_h:.6f}, z {z_h:+.2f})")
            blend_note = (f"TabM standalone {a_nn:.5f}, rank corr {rho_full:.5f}, "
                          f"cross-fitted blend {a_honest:.5f} (delta {d_h:+.6f}, z {z_h:+.2f}, "
                          f"per-fold w {per_fold_w})")
            blend_ok = (z_h > CFG.ACCEPT_Z) and (d_h >= CFG.MATERIALITY)
            w_test = float(np.mean(per_fold_w))
        else:
            print("\n   Blend not computed: TabM did not complete all folds "
                  f"({nn_folds_done}/{CFG.N_FOLDS}).")
    except Exception as err:
        print(f"\n   Blend diagnostics failed ({type(err).__name__}: {str(err)[:110]}) — "
              f"the GBM outputs below are unaffected.")

    # What ships: the honest cross-fitted blend only if it clears both gates.
    if blend_ok:
        oof_out  = blend_oof
        test_out = (1 - w_test) * _logit(test_gbm) + w_test * _logit(test_nn)
        chosen = f"blend of XGBoost control + TabM at w={w_test:.2f}"
    else:
        oof_out, test_out = oof_gbm, test_gbm
        chosen = "XGBoost control alone"
    print(f"\n   SAVED MODEL: {chosen}")

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # Keep the house file format: pred is a probability. The blend lives in logit
    # space, and sigmoid is a monotone map, so the AUC of the saved file is the
    # AUC that was measured above.
    oof_probs  = _sigmoid(np.asarray(oof_out, dtype='float64'))
    test_probs = _sigmoid(np.asarray(test_out, dtype='float64'))
    oof_cv = auc_score(y_np, oof_probs)
    fold_scores = [auc_score(y_np[vi], oof_probs[vi])
                   for _, vi in KFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                      random_state=CFG.SEARCH_SEED).split(X)]

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv")
    print("   Both files hold probabilities; the blend was mapped back through the "
          "sigmoid, which is monotone and cannot change the AUC.")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V29 RESULTS — TabM + In-Run XGBoost Control (GPU)")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} base + {n_te} Triple TE = {len(FEATURES) + n_te} total; "
          f"NN input {CFG.NN_TOP_N}/fold")
    print(f"XGBoost control OOF rs{CFG.SEARCH_SEED}: {a_gbm_oof:.5f}  "
          f"(V22's stored OOF is 0.946076)")
    if blend_note:
        print(f"Non-tree view: {blend_note}")
    print(f"Saved model: {chosen} | OOF {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} | "
          f"GBM iters {iters_gbm}")
    print(f"Test prediction range: {test_probs.min():.4f} .. {test_probs.max():.4f}")

    print(f"\nVERDICT")
    if not blend_note:
        print("   INCONCLUSIVE for the non-tree axis: no honest blend weight was computed")
        print("   — either TabM did not complete every fold, or the blend block above")
        print("   raised (see its message). The standalone TabM OOF, the per-fold rank")
        print("   correlations and both views' predictions are still valid as printed.")
    elif blend_ok:
        print(f"   The non-tree view EARNES a slot: the cross-fitted blend clears z > "
              f"{CFG.ACCEPT_Z:.0f} AND the {CFG.MATERIALITY:.4f} materiality gate. This is")
        print(f"   the first submission-worthy gain since V19's artifact features.")
    else:
        print("   The non-tree view does NOT earn a slot: the cross-fitted blend gain is")
        print(f"   below the {CFG.MATERIALITY:.4f} materiality line even where its z is")
        print("   significant — the same wall V18 hit from the other side.")
        print("   -> keep the GBM alone. The family axis stays closed as a source of LB")
        print("   movement, and any diversity worth carrying has to come from a genuinely")
        print("   different FEATURE view, not from a second estimator on the same one.")
    print("   REMINDER: V28's control reproduced V22's LB to the last digit, so a saved "
          "submission here is comparable with the incumbent line.")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*80}")
