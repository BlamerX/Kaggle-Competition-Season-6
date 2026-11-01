"""
S6E10 V17 - RealMLP epoch ladder on V12's twin frame (budget axis) - GPU (cuda)
GOAL: V12 is this episode's best single (0.96117 CV / 0.96095 LB) and it is still starved: with
      use_early_stopping=False the best epoch came out at 3 = the budget in 5/5 folds, i.e. the model
      was still improving when training was cut off. V15 showed the opposite case (ES fired at
      669-1068 of 4000), so the net's ceiling has never been observed and this is the one knob our own
      data says is untested.
Axis: ONE change - n_epochs swept over the pre-declared ladder 6 -> 8 -> 10, one fitted model per
      rung, rungs NEVER combined and exactly one rung becomes the version's output (the V8 / V14 / V16
      pattern, so this stays a Single Model). REALMLP_PARAMS is otherwise byte-identical to V12
      (lr 0.053, flat_anneal, n_ens 8, batch 256, ES off) and the frame is byte-identical too - 17
      numeric->categorical twins + teacher-as-logit. NO learning-rate or bagging change (held for
      their own arms if a rung wins), and NO invented schedule knob: PyTabKit exposes no anneal-start
      fraction, which is what the pre-flight below proves rather than assumes.
Parent: Previously trained files/Archieve/S6E10_V12_NetTwinTeacher.py (0.96117 CV / 0.96095 LB).
Device: GPU (cuda) - RealMLP_TD via PyTabKit on a Kaggle T4 (16 GB); all numeric matrices float32.
Standalone (owner law 2026-10-11): no previous version's OOF is read as a feature or a target here.
      V12's per-fold AUCs below are recorded ledger NUMBERS used only to print fold deltas; the gate's
      paired DeLong runs OFFLINE on the saved OOFs after this run.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = SEED_REPRO=2027.
Gate: paired DeLong vs v12_oof.csv AND v2_oof.csv: dAUC > +0.00005 AND z > 3 AND >=4/5 folds
      improved, then reproduce at 2027. Tie band 0.0003 - decide on CV, audit on LB.
Run-of-record (pre-declared): the highest-CV rung; V12 is replaced only if the gate also passes.
Checkpoint selection (required disclosure): PyTabKit restores the best-validation epoch and the CV
      fold IS that validation set, so the fold's own labels pick the checkpoint - not a leak, but
      schedule-dependent; longer budgets sit further from the flat_anneal+3ep robust region, so each
      rung is reported against its own budget ceiling rather than blended.
Aggregation knob: `ens_av_before_softmax` stays unset (probability averaging = V12's behaviour).
Expected / Cost: +0.00013 ... +0.00021 CV over V12 (published RealMLP single 0.961344); fit time
      scales with the budget (V12: 154 s/fold at 3 ep) -> 3 rungs x 5 folds ~65-85 min on one T4.
Outputs: v17_ladder_6/8/10_oof.csv (audit trail) + v17_oof.csv / v17_sub.csv (rung of record).
Evidence: PLAN_V16_ONWARD.md §3 "Arm V17"; V12/V15 logs in Memory/training_logs.md.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings

import os
import gc
import time
import random
import inspect
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import torch
import subprocess, sys
from importlib.metadata import version
try:
    from pytabkit import RealMLP_TD_Classifier
    print("pytabkit loaded successfully!")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])
    from pytabkit import RealMLP_TD_Classifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"pytabkit version: {version('pytabkit')}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 17
    VERSION_NAME = "V17_RealMLPEpochLadder"
    EXP_ID = "S6E10_V17_RealMLPEpochLadder"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    # reproduction run (gate requirement): a second Kaggle run whose ONLY delta is
    # RANDOM_SEED = SEED_REPRO (2027) - same recipe, same frame, same fold count.
    SEED_REPRO = 2027

# THE AXIS: the pre-declared epoch ladder. One fitted model per rung, rungs never combined.
EPOCH_RUNGS = [6, 8, 10]
V12_RECORDED_CV = 0.96117
# V12's per-fold AUCs from Memory/training_logs.md - recorded numbers, used only to print fold deltas.
V12_RECORDED_FOLDS = [0.96134, 0.96022, 0.96199, 0.96124, 0.96110]

RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
           'Ease of Online booking', 'Gate location', 'Food and drink',
           'Online boarding', 'Seat comfort', 'Inflight entertainment',
           'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Cleanliness']
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SHORT = {'Inflight wifi service': 'wifi', 'Departure/Arrival time convenient': 'timehc',
         'Ease of Online booking': 'booking', 'Gate location': 'gate', 'Food and drink': 'food',
         'Online boarding': 'ob', 'Seat comfort': 'seat', 'Inflight entertainment': 'ient',
         'On-board service': 'obsrv', 'Leg room service': 'legroom', 'Baggage handling': 'baggage',
         'Checkin service': 'checkin', 'Cleanliness': 'clean',
         'Age': 'age', 'Flight Distance': 'fd', 'Gender': 'gender',
         'Customer Type': 'cust', 'Type of Travel': 'travel', 'Class': 'class',
         'Departure Delay in Minutes': 'depdelay', 'Arrival Delay in Minutes': 'arrdelay'}
INFLIGHT8 = ['Inflight wifi service', 'Food and drink', 'Seat comfort',
             'Inflight entertainment', 'On-board service', 'Leg room service',
             'Baggage handling', 'Cleanliness']
ALL_FEATURES = (['Age', 'Flight Distance'] + RATINGS +
                ['Departure Delay in Minutes', 'Arrival Delay in Minutes'] + CATS)

KEY_PAIRS = [('Class', 'Type of Travel'), ('Customer Type', 'Type of Travel'),
             ('Class', 'Customer Type'), ('Gender', 'Class'),
             ('Inflight wifi service', 'Online boarding'),
             ('Inflight wifi service', 'Type of Travel'),
             ('Online boarding', 'Type of Travel'),
             ('Seat comfort', 'Inflight entertainment'),
             ('Class', 'Inflight wifi service'), ('Class', 'Online boarding')]
RATING_PAIRS = [('Inflight wifi service', 'Ease of Online booking'),
                ('Ease of Online booking', 'Online boarding'),
                ('Seat comfort', 'Leg room service'),
                ('Food and drink', 'Cleanliness'),
                ('On-board service', 'Baggage handling'),
                ('Checkin service', 'Gate location'),
                ('Departure/Arrival time convenient', 'Gate location'),
                ('Inflight entertainment', 'Cleanliness')]

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
# PyTabKit RealMLP_TD. Argument names come from our own proven scripts (S6E3_V73 / S6E5_V21):
# 'epochs'/'batch'/'epoch_scheduler'/'auto_configure' are NOT this build's names, and invented
# kwargs cost three failed runs in V5 plus two in V15. The values below are V12's byte-for-byte;
# the ONLY thing this version varies is n_epochs, and it does so through EPOCH_RUNGS, never by
# editing a value in this dict (a silent second change would make the V12->V17 delta unattributable).
# n_ens=8 is INTERNAL BAGGING inside one model object (a single fit() call), not an ensemble.
# use_early_stopping=False keeps every rung on a FIXED budget, so 'best epoch' is read against the
# rung's own ceiling - see the checkpoint disclosure in the module docstring.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,        # UNCHANGED vs V12 - internal bagging of one model object
    'n_epochs': 3,     # V12's value; superseded per rung by EPOCH_RUNGS below, not edited here
    'batch_size': 256,
    'use_early_stopping': False,   # V12's fixed-epoch budget
    'early_stopping_additive_patience': 3,       # inert with ES off; V6 names kept
    'early_stopping_multiplicative_patience': 1,  # inert with ES off; V6 names kept
    'lr': 0.053,                   # V12 (held constant - a longer budget with a lower lr is a
                                   # SECOND axis and gets its own arm if a rung wins)
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',     # V12's schedule name, unchanged; no anneal-fraction knob exists
    'wd_sched': 'cos_log_15',
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'embedding_size': 5,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],   # architecture is NOT this axis
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'expm4t',
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',
    # 'ens_av_before_softmax' is deliberately NOT set: pytabkit's default averages probabilities.
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

# Teacher = ektarr ORIGINAL_MODEL_PARAMS (deeper + sparser than main model)
TEACHER_PARAMS = {
    'objective': 'binary:logistic',
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'tree_method': 'hist',
    'enable_categorical': True,
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 0,
    'verbosity': 0,
}

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING (fold-independent part) - byte-identical to V12
# =============================================================================
def add_digit_features(df, num_cols, M):
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('float32')
        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    return df


def add_aggregate_features(df):
    """
    Multi-column aggregates + delay arithmetic only. Deterministic 1-3 condition
    AND-flags (r_*, *_is0, trio_break) are deliberately excluded: trees recover
    those in <=3 splits and a single such flag hijacked 0.37 fold-1 gain in the
    pre-V1 development run.
    """
    df = df.copy()
    R = df[RATINGS]

    df['_mean_excl0'] = R.replace(0, np.nan).mean(axis=1).astype('float32')
    df['_std_excl0'] = R.replace(0, np.nan).std(axis=1).astype('float32')
    df['_min_excl0'] = R.replace(0, np.nan).min(axis=1).astype('float32')
    df['_count_ge4_all'] = (R >= 4).sum(axis=1).astype('float32')
    df['_count_ge4_inflight8'] = (df[INFLIGHT8] >= 4).sum(axis=1).astype('float32')
    df['_count_ge3'] = (R >= 3).sum(axis=1).astype('float32')
    df['_count_le2'] = ((R <= 2) & (R > 0)).sum(axis=1).astype('float32')
    df['_count_le1'] = ((R <= 1) & (R > 0)).sum(axis=1).astype('float32')
    df['_count_eq5'] = (R == 5).sum(axis=1).astype('float32')
    df['_count_0'] = (R == 0).sum(axis=1).astype('float32')

    dep = df['Departure Delay in Minutes']
    arr = df['Arrival Delay in Minutes']
    df['_dep_delayed'] = (dep > 0).astype('float32')
    df['_arr_delay_isna'] = arr.isna().astype('float32')
    df['_delay_gap'] = (arr - dep).astype('float32')
    df['_total_delay'] = (dep + arr.fillna(dep)).astype('float32')
    df['_delay_per_100mi'] = df['_total_delay'] / (df['Flight Distance'] / 100.0)

    df['_fd_250_bin'] = np.floor(df['Flight Distance'] / 250).astype('float32')
    df['_age_5_bin'] = np.floor(df['Age'] / 5).astype('float32')
    return df


def add_route_profile(train_df, test_df):
    """
    Wide target-free route profile (ektarr distance_profile + pool-based means):
    per-FD mean, std, residual and categorical shares over the train+test pool.
    """
    measures = RATINGS + ['Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    pool = pd.concat([train_df, test_df], ignore_index=True)
    fd = pool['Flight Distance']
    gmean = pool.groupby(fd)[measures].mean()
    gstd = pool.groupby(fd)[measures].std()
    cnt = pool.groupby(fd).size()
    shares = {c: pd.crosstab(fd, pool[c], normalize='index') for c in CATS}

    for d in (train_df, test_df):
        for c in measures:
            m = d['Flight Distance'].map(gmean[c])
            d[f'_route_{SHORT[c]}_mean'] = m.astype('float32')
            d[f'_route_{SHORT[c]}_std'] = d['Flight Distance'].map(gstd[c]).astype('float32')
            d[f'_route_{SHORT[c]}_resid'] = (d[c] - m).astype('float32')
        d['_route_count'] = d['Flight Distance'].map(cnt).fillna(1).astype('float32')
        for c, table in shares.items():
            prior = pool[c].value_counts(normalize=True)
            for cat in table.columns:
                cat_tag = str(cat).lower().replace(' ', '_')
                d[f'_route_{SHORT[c]}_{cat_tag}_share'] = \
                    d['Flight Distance'].map(table[cat]).fillna(prior[cat]).astype('float32')
    return train_df, test_df


def add_org_stats(train_df, test_df, orig_df):
    """
    Original-data label means per value for every feature (ektarr original_means
    protocol: raw group means, unseen -> orig prior) + Travel x Class x Customer
    triple. Ratings/FD keys additionally get k=20 smoothed variants.
    """
    yo = orig_df[CFG.TARGET].astype(int)
    base = float(yo.mean())

    def key_of(df, col):
        if col == 'Flight Distance':
            return df[col] // 10
        return df[col]

    for c in ALL_FEATURES:
        ok = key_of(orig_df, c).astype('string') if not pd.api.types.is_numeric_dtype(orig_df[c]) else key_of(orig_df, c)
        stat = pd.DataFrame({'k': ok, 'y': yo}).groupby('k')['y'].mean()
        name = f'_{SHORT[c]}_org_mean'
        for d in (train_df, test_df):
            ck = key_of(d, c)
            ck = ck.astype('string') if not pd.api.types.is_numeric_dtype(d[c]) else ck
            d[name] = ck.map(stat).fillna(base).astype('float32')

    trip = [('Travel', 'Class', 'Customer')]
    def triple(frame):
        a = frame['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        b = frame['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        c = frame['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
        return a * 100 + b * 10 + c
    stat = pd.DataFrame({'k': triple(orig_df), 'y': yo}).groupby('k')['y'].mean()
    for d in (train_df, test_df):
        d['_triple_org_mean'] = triple(d).map(stat).fillna(base).astype('float32')
    return train_df, test_df


def prepare_original(orig_df, train, test):
    """Exclude original rows whose predictor vector exactly matches a
    competition row (ektarr hygiene), then dedupe."""
    def hashes(frame):
        v = frame[ALL_FEATURES].copy()
        for c in ALL_FEATURES:
            v[c] = (v[c].astype('float64') if pd.api.types.is_numeric_dtype(v[c])
                    else v[c].astype('string'))
        return pd.util.hash_pandas_object(v, index=False)
    comp = set(pd.concat([hashes(train), hashes(test)], ignore_index=True).values)
    oh = hashes(orig_df)
    keep = ~oh.isin(comp).values
    out = orig_df.loc[keep].drop_duplicates(subset=ALL_FEATURES).reset_index(drop=True)
    print(f"      original rows: {len(orig_df):,} | exact-overlap removed: "
          f"{int((~keep).sum()):,} | clean: {len(out):,}")
    return out


def add_original_teacher(train_df, test_df, orig_clean):
    """
    Teacher = ektarr spec: XGB d8, cs0.5, ss0.8, 600 trees, ALL 21 features with
    native categoricals, fit on the overlap-cleaned original, prediction added
    as one column. Never touches competition labels.
    """
    vocab = {}
    def prep(frame):
        X = frame[ALL_FEATURES].copy()
        for c in ALL_FEATURES:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('float32')
            else:
                X[c] = X[c].astype('string').astype(
                    pd.CategoricalDtype(categories=vocab[c]))
        return X
    for c in CATS:
        vocab[c] = sorted(orig_clean[c].astype('string').unique().tolist())

    Xo = prep(orig_clean)
    yo = orig_clean[CFG.TARGET].astype(int)
    m = xgb.XGBClassifier(**TEACHER_PARAMS)
    m.fit(Xo, yo, verbose=False)
    train_df['_og_teacher'] = m.predict_proba(prep(train_df))[:, 1].astype('float32')
    test_df['_og_teacher'] = m.predict_proba(prep(test_df))[:, 1].astype('float32')
    del m
    gc.collect()
    return train_df, test_df

# =============================================================================
# 6b. INHERITED FROM V12 - THE NET'S INPUT REPRESENTATION (twins + teacher-as-logit)
#     This is V12's axis, carried unchanged: V17 varies the epoch budget only.
# =============================================================================
def add_numeric_categorical_twins(train_df, test_df, cols):
    """
    V12 axis (a), plan §2/V12 line 74 verbatim: "for every numeric column add
    `<col>_cat_ = factorize(...)` as a categorical while keeping every numeric".
    `cols` is the 17 NUMERIC raw predictors (13 RATINGS + Age + Flight Distance +
    both delays) = ALL_FEATURES minus CATS; the ~240 derived numeric columns (per-fold
    TE, freq/rarity, route profile, aggregates, digits, org means) deliberately get NO
    twin - they are continuous estimates, not levels, and factorising one `_tm_*` column
    alone would hand PyTabKit hundreds of thousands of singleton categories.
    NO numeric column is dropped anywhere; dropping them is the all-categorical variant
    measured at -0.00417.

    RNG-free by construction: pd.factorize over the concatenated train+test pool uses no
    RNG, no sampling and no labels; codes are int64 and NaN becomes code -1, i.e. missing
    becomes a level of its own. Both frames are cast to CategoricalDtype with the IDENTICAL
    level list, so train, validation and test share one code space.
    """
    twin_cols = []
    n_train = len(train_df)
    pool = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    oh_limit = REALMLP_PARAMS['max_one_hot_cat_size']
    for c in cols:
        codes = pd.factorize(pool[c])[0]
        levels = np.unique(codes)                      # sorted, shared by both frames
        tr = pd.Categorical(codes[:n_train], categories=levels)
        te = pd.Categorical(codes[n_train:], categories=levels)
        if int(np.asarray(tr.isna()).sum() + np.asarray(te.isna()).sum()):
            raise ValueError(f"twin of {c} produced an out-of-vocabulary level")
        name = f"{c}_cat_"                             # plan's `<col>_cat_` naming
        train_df[name] = tr
        test_df[name] = te
        twin_cols.append(name)
        if len(levels) <= oh_limit:
            treat = f"one-hot ({len(levels)} cols)"
        else:
            treat = f"embedding ({REALMLP_PARAMS['embedding_size']} cols)"
        print(f"      twin {name}: {len(levels):,} levels -> {treat}")
    return train_df, test_df, twin_cols


def teacher_to_logit(df):
    """
    V12 axis (b): feed `_og_teacher` as a logit rather than a probability. Called for the
    train frame and the test frame with the SAME two statements, so the transform is
    identical in both and identical for every fold.
    Clip 1e-6 is the house precedent from `S6E10_V3_DuoTeacher.py`. It is required, not
    cosmetic: the teacher saturates at exactly 0.0 and 1.0, log(0) = -inf, and PyTabKit
    robust-scales the continuous block, so an inf would poison the column's median/IQR.
    """
    p = np.clip(df['_og_teacher'].to_numpy(dtype='float64'), 1e-6, 1.0 - 1e-6)
    df['_og_teacher'] = np.log(p / (1.0 - p)).astype('float32')
    return df


def estimate_expanded_matrix(X_fit):
    """
    MEMORY GUARD estimate of the width PyTabKit's transform expands X_fit to, from the
    frame's own dtypes and category counts rather than from documentation memory: a
    categorical with at most max_one_hot_cat_size levels is one-hot (contributes n_levels
    columns), a wider one gets an embedding (contributes embedding_size columns).
    Returns (width, n_numeric, one_hot_dict, embedding_dict, worst_all_one_hot_width).
    """
    oh = REALMLP_PARAMS['max_one_hot_cat_size']
    emb = REALMLP_PARAMS['embedding_size']
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    n_num = int(X_fit.shape[1] - len(cat_cols))
    cards = {c: int(X_fit[c].cat.categories.size) for c in cat_cols}
    one_hot = {c: k for c, k in cards.items() if k <= oh}
    embedding = {c: k for c, k in cards.items() if k > oh}
    width = n_num + sum(one_hot.values()) + emb * len(embedding)
    worst = n_num + sum(cards.values())
    return width, n_num, one_hot, embedding, worst


def net_memory_report(tag, X_fit, X_test, width, worst_width):
    """
    MEMORY GUARD print, ABSOLUTE numbers only (no fabricated progress percentage). Reports the
    ESTIMATE from estimate_expanded_matrix as tensor GB, the 16 GB T4 / 12 GB alert headroom
    check, the all-one-hot worst-case bound that max_one_hot_cat_size=18 exists to prevent, and
    the device's own allocated/reserved counters.
    """
    gib = 1024 ** 3
    rows_fit, rows_test = int(X_fit.shape[0]), int(X_test.shape[0])
    gb_fit = rows_fit * width * 4 / gib
    gb_test = rows_test * width * 4 / gib
    gb_worst = rows_fit * worst_width * 4 / gib
    lines = [
        f"{tag}: ESTIMATE expanded matrix {width} cols from {X_fit.shape[1]} frame cols "
        f"({rows_fit:,} x {width} x 4 B = {gb_fit:.2f} GB train block, "
        f"{rows_test:,} x {width} x 4 B = {gb_test:.2f} GB test block)",
        f"{tag}: T4 16 GB headroom check (absolute): {gb_fit:.2f} + {gb_test:.2f} = "
        f"{gb_fit + gb_test:.2f} GB of the transformed matrix vs 16 GB device, "
        f"alert threshold 12 GB -> "
        + ("WITHIN BUDGET" if gb_fit + gb_test <= 12.0
           else "ALERT: OVER THE 12 GB LINE - ABORT, DO NOT SILENTLY SHRINK THE TWIN SET")
        + f" | all-one-hot worst case would be {gb_worst:.2f} GB "
          f"(width {worst_width}: prevented by max_one_hot_cat_size="
          f"{REALMLP_PARAMS['max_one_hot_cat_size']})",
    ]
    if gb_worst > 15.0:
        lines.append(
            f"{tag}: WORST-CASE NOTE (absolute): if this build one-hots every category instead of "
            f"embedding the wide ones, the train block ALONE is {gb_worst:.2f} GB vs a 16 GB device "
            f"-> OOM. Narrowing the twin set is a NEW axis, so it needs a NEW version, not an edit here.")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        peak_res = torch.cuda.max_memory_reserved(0) / gib
        lines.append(
            f"{tag}: device {props.name} total {props.total_memory / gib:.2f} GB "
            f"| allocated {torch.cuda.memory_allocated(0) / gib:.2f} GB "
            f"| reserved {torch.cuda.memory_reserved(0) / gib:.2f} GB "
            f"| peak allocated {torch.cuda.max_memory_allocated(0) / gib:.2f} GB "
            f"| peak reserved {peak_res:.2f} GB "
            + ("| ALERT: peak reserved > 12 GB" if peak_res > 12.0 else "| no >12 GB alert"))
    else:
        lines.append(f"{tag}: CUDA UNAVAILABLE - this run is not on the T4 the arm was "
                     f"budgeted for (plan §1.4: the GPU arms of this ladder need one)")
    for line in lines:
        print(f"      {line}")
    return lines

# =============================================================================
# 7. PER-FOLD FEATURE BUILDER (curated TE + frequency/rarity + native cats)
# =============================================================================
def feature_name(column):
    return column.lower().replace(' ', '_').replace('/', '_').replace('-', '_')


def make_key_frames(fit, frames):
    """
    Int-factorized key columns for TE + frequency (no string row-joins:
    dev-run prep cost was ~290s/fold from '__'.join over 18 pair keys x 3 frames).
    Pair code = code_a * n_cat_b + code_b using vocabularies fitted on the fit fold.
    """
    cat_codes = {}
    for c in CATS:
        cats = pd.Categorical(fit[c].astype('string'),
                              categories=sorted(fit[c].astype('string').dropna().unique().tolist()))
        cat_codes[c] = (cats, len(cats.categories) + 1)

    def codes(frame):
        out = {}
        for c in CATS:
            out[c] = frame[c].astype('string').map(
                {v: i + 1 for i, v in enumerate(cat_codes[c][0].categories)}).fillna(0).astype('int64')
        for c in RATINGS:
            out[c] = frame[c].astype('int64')
        return out

    cf = [codes(f) for f in frames]

    keys = []
    for f, c in zip(frames, cf):
        k = {feature_name(col): val for col, val in c.items()}
        k['fd_exact'] = f['Flight Distance'].astype('int64')
        k['fd_div10'] = (f['Flight Distance'] // 10).astype('int64')
        k['fd_div100'] = (f['Flight Distance'] // 100).astype('int64')
        k['fd_250_bin'] = np.floor(f['Flight Distance'] / 250).astype('int64')
        k['age_5_bin'] = np.floor(f['Age'] / 5).astype('int64')
        for a, b in KEY_PAIRS + RATING_PAIRS:
            na, nb_ = feature_name(a), feature_name(b)
            if a in RATINGS and b in RATINGS:
                k[f'pair_{na}__{nb_}'] = f[a].astype('int64') * 10 + f[b].astype('int64')
            elif a in RATINGS:
                k[f'pair_{na}__{nb_}'] = f[a].astype('int64') * cat_codes[b][1] + c[b]
            elif b in RATINGS:
                k[f'pair_{na}__{nb_}'] = c[a] * 10 + f[b].astype('int64')
            else:
                k[f'pair_{na}__{nb_}'] = c[a] * cat_codes[b][1] + c[b]
        keys.append(pd.DataFrame(k, index=f.index))
    return keys


def fill_continuous_nan(X_fit, X_val, X_test):
    """
    PyTabKit raises on NaN in continuous columns; V2 relied on XGBoost routing them.
    Medians come from the FIT FOLD only (no val/test statistics), cast back to
    float32. Inherited from V12 unchanged - forced by the architecture, not this axis.
    """
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X_fit.columns if c not in cat_cols]
    med = X_fit[num].median(numeric_only=True).astype('float32')
    out = []
    for f in (X_fit, X_val, X_test):
        g = f.copy()
        g[num] = g[num].fillna(med)
        g[num] = g[num].astype('float32')
        for c in cat_cols:                       # leftover unseen category levels
            cats = g[c].cat.categories
            g[c] = g[c].astype('object').fillna(cats[0]).astype(pd.CategoricalDtype(cats))
        left = g.isna().sum()
        left = left[left > 0]
        if len(left):                            # fail here, not inside PyTabKit
            raise ValueError(f"NaN survived fill: {dict(left)}")
        out.append(g)
    return out


def build_fold_features(X_fit, X_val, X_test, y_fit, fold):
    """Curated per-fold TE (smooth=20, inner cv=5), frequency+rarity, native
    categorical vocabularies from the fit fold. Returns (3 frames, n_added)."""
    frames = [X_fit.copy(), X_val.copy(), X_test.copy()]
    keys = make_key_frames(X_fit, frames)

    skv = tuple(map(int, sklearn_version.split('.')[:2]))
    if skv >= (1, 9):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1042 + fold)
        enc = TargetEncoder(target_type='binary', smooth=20.0, cv=inner)
    else:
        enc = TargetEncoder(target_type='binary', smooth=20.0, cv=5,
                            shuffle=True, random_state=1042 + fold)
    encoded = [enc.fit_transform(keys[0], y_fit)]
    encoded += [enc.transform(k) for k in keys[1:]]
    names = [f'_tm_{k}' for k in keys[0].columns]
    for f, e in zip(frames, encoded):
        for j, nm in enumerate(names):
            f[nm] = np.asarray(e)[:, j].astype('float32')

    for col in keys[0].columns:
        fm = keys[0][col].value_counts(normalize=True)
        for f, k in zip(frames, keys):
            fr = k[col].map(fm).fillna(0).astype('float32')
            f[f'_freq_{col}'] = fr
            f[f'_rarity_{col}'] = -np.log(fr.clip(lower=1.0 / len(X_fit)))

    for c in CATS:
        cats = sorted(X_fit[c].astype('string').dropna().unique().tolist())
        dt = pd.CategoricalDtype(categories=cats)
        for f in frames:
            f[c] = f[c].astype('string').astype(dt)
            # a tree routes NaN natively, an MLP cannot: values outside the fit-fold
            # vocabulary (~0.02% of cells) go to the first level so the encoder is total
            f[c] = f[c].astype('object').fillna(cats[0]).astype(dt)

    n_added = len(frames[0].columns) - len(X_fit.columns)
    return frames[0], frames[1], frames[2], n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED}")
    print(f"Axis: n_epochs ladder {EPOCH_RUNGS} on V12's byte-identical twin+logit frame and recipe "
          f"(rungs never combined)")
    print(f"Parent V12: {V12_RECORDED_CV} CV / 0.96095 LB | Control V2: 0.961009 CV / 0.96051 LB "
          f"| gate vs v12_oof AND v2_oof, offline")
    print("="*80)

    # ---- ARG-NAME LAW pre-flight (PLAN_V16_ONWARD.md rule 4), at import: V17's whole axis is one
    # kwarg name, and invented PyTabKit names have already cost this episode five failed runs. The
    # constructor takes **kwargs, so an unknown key would be silently absorbed rather than raise -
    # hence listing any REALMLP_PARAMS key the build does not name explicitly.
    _SIG = inspect.signature(RealMLP_TD_Classifier.__init__).parameters
    print(f"RealMLP_TD_Classifier accepts n_epochs: {'n_epochs' in _SIG} "
          f"| var_kw present: {any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _SIG.values())}")
    if 'n_epochs' not in _SIG:
        raise ImportError("This PyTabKit build has no n_epochs parameter - stopping before the "
                          "fold loop instead of letting the ladder run on a silently-ignored knob.")
    _unnamed = [k for k in REALMLP_PARAMS if k not in _SIG]
    if _unnamed:
        print(f"⚠️ keys absorbed by **kwargs (not named in the signature): {_unnamed}")

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    for df in (train, orig):
        if df[CFG.TARGET].dtype == object:
            df[CFG.TARGET] = df[CFG.TARGET].astype(str).str.strip().str.lower().eq('true')
        df[CFG.TARGET] = df[CFG.TARGET].astype(int)

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig  = orig.drop(columns=['id'], errors='ignore')

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig.shape}")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f} | (orig): {orig[CFG.TARGET].mean():.4f}")
    print(f"   Missing in train: {int(train.isna().sum().sum())} (XGBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering...")

    print("   Adding digit features...")
    DIGIT_NUMS = ['Age', 'Flight Distance',
                  'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    M = train[DIGIT_NUMS].max()
    train = add_digit_features(train, DIGIT_NUMS, M)
    test  = add_digit_features(test,  DIGIT_NUMS, M)

    print("   Adding multi-column aggregates (rule-region flags excluded)...")
    train = add_aggregate_features(train)
    test  = add_aggregate_features(test)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    print("   Adding wide target-free route profile (mean/std/resid/shares)...")
    train, test = add_route_profile(train, test)

    print("   Adding original-data org stats (all features + triple)...")
    train, test = add_org_stats(train, test, orig)

    print("   Preparing original (overlap removal) + ektarr-spec teacher...")
    orig_clean = prepare_original(orig, train, test)
    train, test = add_original_teacher(train, test, orig_clean)

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    # ---- INHERITED FROM V12 (a): numeric -> categorical TWINS of the 17 numeric raw
    # predictors, every numeric kept. Placed AFTER the float32 alignment loop on purpose:
    # the twins are CategoricalDtype columns and must never pass through a float32 cast.
    TWIN_SOURCE = [c for c in ALL_FEATURES if c not in CATS]
    train, test, twin_cols = add_numeric_categorical_twins(train, test, TWIN_SOURCE)
    print(f"   V12 axis (a) inherited: +{len(twin_cols)} numeric->categorical twins "
          f"(source {len(TWIN_SOURCE)} cols, no numeric dropped) -> frame {len(test.columns)} cols")

    # ---- INHERITED FROM V12 (b): the teacher enters as its LOGIT (same function, both frames).
    train = teacher_to_logit(train)
    test = teacher_to_logit(test)
    print(f"   V12 axis (b) inherited: _og_teacher -> logit (clip 1e-6) | "
          f"range [{float(train['_og_teacher'].min()):.4f}, {float(train['_og_teacher'].max()):.4f}]")

    FEATURES = list(test.columns)

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV x 3 epoch rungs, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training RealMLP_TD epoch ladder {EPOCH_RUNGS} ({CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_per_rung  = {e: np.zeros(len(y)) for e in EPOCH_RUNGS}
    test_per_rung = {e: np.zeros(len(test_X)) for e in EPOCH_RUNGS}
    rung_fold     = {e: [] for e in EPOCH_RUNGS}
    rung_best     = {e: [] for e in EPOCH_RUNGS}
    prep_times    = []
    fit_times     = {e: [] for e in EPOCH_RUNGS}   # print-support only, per rung
    exp_widths    = []
    frame_cols    = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        fold_start = time.time()

        prep_start = time.time()
        X_train, X_val, X_test, n_added = build_fold_features(
            X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), test_X.copy(),
            y.iloc[train_idx], fold)
        X_train, X_val, X_test = fill_continuous_nan(X_train, X_val, X_test)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s (shared by all rungs)")

        # MEMORY GUARD printed BEFORE the fits: the width PyTabKit expands this frame to.
        exp_width, n_num_cols, oh_map, emb_map, worst_width = estimate_expanded_matrix(X_train)
        exp_widths.append(exp_width)
        frame_cols.append(int(X_train.shape[1]))
        print(f"      expanded-matrix ESTIMATE (absolute): {X_train.shape[1]} frame cols = "
              f"{n_num_cols} numeric + {len(CATS) + len(twin_cols)} categorical "
              f"[{len(CATS)} passengers + {len(twin_cols)} twins], {len(oh_map)} one-hot "
              f"(<= max_one_hot_cat_size={REALMLP_PARAMS['max_one_hot_cat_size']}), "
              f"{len(emb_map)} embedded -> {exp_width} expanded cols "
              f"(all-one-hot worst case {worst_width})")
        net_memory_report(f"pre-fit fold {fold+1}", X_train, X_test, exp_width, worst_width)

        for ep in EPOCH_RUNGS:
            params = {**REALMLP_PARAMS, 'n_epochs': ep}
            model = RealMLP_TD_Classifier(**params)
            # X_val/y_val positional, as in our S6E2/S6E3/S6E5 RealMLP scripts. They are
            # REQUIRED: fit() without a validation block silently drops ~20% of the training
            # rows (measured +0.00061 when passed). No checkpoint reuse across folds/rungs.
            t_fit = time.time()
            model.fit(X_train, y_train.values, X_val, y_val.values)

            val_probs = np.asarray(model.predict_proba(X_val))[:, 1].astype('float64')
            oof_per_rung[ep][val_idx] = val_probs
            test_per_rung[ep] += np.asarray(model.predict_proba(X_test))[:, 1] / CFG.N_FOLDS

            fold_auc = auc_score(y_val.values, val_probs)
            rung_fold[ep].append(fold_auc)
            best_ep = (getattr(model, 'best_epoch', None)
                       or getattr(model, 'best_epoch_', None) or ep)
            rung_best[ep].append(best_ep)
            fit_times[ep].append(time.time() - t_fit)   # this rung's fit seconds, excluding prep

            print(f"      epochs={ep:>3}: AUC {fold_auc:.5f} | best epoch {best_ep} (ceiling {ep}) | "
                  f"elapsed {(time.time()-t0)/60:.1f}min")

            del model, val_probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if fold == 0:
            print(f"      Device: "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
                  f"| cols {X_train.shape[1]} | n_ens={REALMLP_PARAMS['n_ens']} "
                  f"(internal bagging of one model object)")
            net_memory_report("fold 1 end", X_train, X_test, exp_width, worst_width)

        del X_train, X_val, X_test, y_train, y_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- rung CVs and the pre-declared run-of-record rule --------------------
    rung_cv = {e: auc_score(y.values, oof_per_rung[e]) for e in EPOCH_RUNGS}
    improved_vs_v12 = {e: int(sum(1 for a, b in zip(rung_fold[e], V12_RECORDED_FOLDS) if a > b))
                       for e in EPOCH_RUNGS}
    record_ep = max(EPOCH_RUNGS, key=lambda e: rung_cv[e])
    saturated = {e: sum(1 for b in rung_best[e] if b == e) for e in EPOCH_RUNGS}

    oof_probs, test_probs = oof_per_rung[record_ep], test_per_rung[record_ep]
    fold_scores, best_iters = rung_fold[record_ep], rung_best[record_ep]
    oof_cv = rung_cv[record_ep]

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    for e in EPOCH_RUNGS:
        pd.DataFrame({'id': train_id, 'pred': oof_per_rung[e]}).to_csv(
            f"{out_dir}/{stem}_ladder_{e}_oof.csv", index=False)
        print(f"   [SAVED] {out_dir}/{stem}_ladder_{e}_oof.csv — per-rung audit trail")

    pd.DataFrame({'id': train_id, 'pred': oof_probs}).to_csv(
        f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred) — rung of record n_epochs={record_ep}")

    pd.DataFrame({'id': test_id, CFG.TARGET: test_probs}).to_csv(
        f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction) — same single model")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Rung of record: n_epochs={record_ep} (pre-declared: highest CV; adoption still needs "
          f"DeLong vs v12_oof AND v2_oof)")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best epochs: {best_iters} | {saturated[record_ep]}/{CFG.N_FOLDS} still pinned at the "
          f"ceiling {record_ep}")
    for e in EPOCH_RUNGS:
        print(f"   Rung n_epochs={e:>3}: OOF CV {rung_cv[e]:.5f} | folds over V12's recorded AUCs: "
              f"{improved_vs_v12[e]}/{CFG.N_FOLDS} | pinned at ceiling {saturated[e]}/{CFG.N_FOLDS}")
    print(f"   dCV vs V12 (recorded {V12_RECORDED_CV}): {oof_cv - V12_RECORDED_CV:+.6f}")
    print(f"   PyTabKit width per fold: cols handed to fit {frame_cols} | expanded estimate {exp_widths}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s/fold (mean) "
          f"| fit per rung (mean) "
          f"{ {e: round(float(np.mean(v))) for e, v in fit_times.items()} }"
          f" | total {total_time_min:.1f} min")
