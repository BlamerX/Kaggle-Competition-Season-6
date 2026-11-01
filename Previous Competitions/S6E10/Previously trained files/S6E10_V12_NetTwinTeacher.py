"""
S6E10 V12 - Net Input Representation: numeric->categorical twins + teacher-as-logit (frame axis) - GPU (cuda)
GOAL: hand the NET (a) a factorised `<col>_cat_` twin for each of the 17 numeric raw predictors while
      keeping every numeric column, and (b) `_og_teacher` as its logit log(p/(1-p)) instead of the
      probability - both halves are ONE axis, taken from goodpjw's `nn_frame` and yekenot's kernel.
      This only matters if V7's recipe lives: V7's CV IS PENDING (authored, unrun), a net below V2's
      0.961009 twice means V12 is SKIPPED, and no V12 number may be quoted before that check lands.
Axis: ONE change - the net's input representation. `REALMLP_PARAMS` is V7's, byte-identical (a value
      edit here would make the V7->V12 delta unattributable); no TE, aux, binning, fold, seed, float32
      or NaN-fill change, and trees are not trained by this script (twins on trees is V17's frame).
Parent: S6E10_V7_RealMLPRecipe.py (CV pending).
Device: GPU (cuda) - RealMLP_TD via PyTabKit on a Kaggle T4 (16 GB); all numeric matrices float32.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = RANDOM_SEED=2027
       (CFG.SEED_REPRO) as the only delta.
Gate: paired DeLong vs v7_oof.csv AND v2_oof.csv, computed externally: dAUC > +0.00005 AND z > 3 AND
      >= 4/5 folds improved, then reproduce at seed 2027; ship as control only at CV >= 0.9614.
Expected / Cost: +0.0001 ... +0.0004 over V7, ~15-20 min (plan estimate, not a measurement).
Outputs: v12_oof.csv (id, pred) + v12_sub.csv (id, satisfaction) to /kaggle/working.
Checkpoint selection: PyTabKit restores the best-validation epoch and the CV fold IS that validation set,
      so the fold's own labels pick the checkpoint - not a leak, but schedule-dependent; flat_anneal + 3
      epochs (V7's recipe, inherited unchanged) is the robust region, so V12 is comparable to V7 and V2.
Aggregation knob: `ens_av_before_softmax` is NOT set - probability averaging stays at the default.
Memory guard: the twins widen PyTabKit's expanded matrix, so the run prints the absolute GB readings -
      frame width vs the 16 GB device, the 12 GB alert line, the all-one-hot worst-case OOM bound and the
      torch.cuda device counters. Absolute values only; a progress meter is never fabricated.
Evidence behind this arm: `PLAN_V7_ONWARD.md` §2 "### V12" and §4 (planning file, kept outside this repo).
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings

import os
import gc
import time
import random
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
    VERSION_NUMBER = 12
    VERSION_NAME = "V12_NetTwinTeacher"
    EXP_ID = "S6E10_V12_NetTwinTeacher"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    # reproduction run (gate requirement 4): a second Kaggle run whose ONLY delta is
    # RANDOM_SEED = SEED_REPRO (2027) - same recipe, same frame, same fold count.
    SEED_REPRO = 2027

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
# kwargs cost three failed runs in V5. The values below are V7's byte-for-byte - V12's axis is the
# input frame, and editing a recipe value here would make the V7->V12 delta unattributable.
# n_ens=8 is INTERNAL BAGGING inside one model object (a single fit() call), not an ensemble.
# use_early_stopping=False makes n_epochs=3 the whole training plan; PyTabKit still restores the
# best-validation epoch - see the checkpoint-selection disclosure in the module docstring.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,        # UNCHANGED vs V6 - internal bagging of one model object
    'n_epochs': 3,     # V6 6 -> 3 (plan line 41; 3 ep reproduces the 6-ep score at half the time)
    'batch_size': 256,
    'use_early_stopping': False,   # V6 True -> False (plan line 41): fixed-epoch budget
    'early_stopping_additive_patience': 3,       # inert with ES off; V6 names kept
    'early_stopping_multiplicative_patience': 1,  # inert with ES off; V6 names kept
    'lr': 0.053,                   # V6 0.075 -> 0.053 (plan line 41)
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',     # unchanged (already V6's) - the robust-region schedule
    'wd_sched': 'cos_log_15',      # NEW (plan line 41) - weight-decay schedule
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,      # unchanged, stated explicitly by plan line 41
    'bias_init_mode': 'neg-uniform-dynamic-2',   # NEW (plan line 41)
    'embedding_size': 5,           # V6 8 -> 5 (plan line 41)
    'max_one_hot_cat_size': 18,    # unchanged (V6 already 18, per plan line 41)
    'hidden_sizes': [512, 256, 128],   # architecture NOT this axis
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'expm4t',      # V6 'flat_cos' -> 'expm4t' (plan line 41)
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'ls_eps': 0.01,                # V6 0.02 -> 0.01 (plan line 41)
    'ls_eps_sched': 'sqrt_cos',    # V6 'cos' -> 'sqrt_cos' (plan line 41)
    # 'ens_av_before_softmax' is deliberately NOT set (plan §2/V7 comparability knob,
    # runbook §3): pytabkit 1.7.3's default averages the bags' probabilities.
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
# 6. FEATURE ENGINEERING (fold-independent part)
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
# 6b. V12 AXIS: THE NET'S INPUT REPRESENTATION (twins + teacher-as-logit)
#     Net frame ONLY - trees are not trained by this script and the twin
#     representation on trees is V17's question (plan §2/V12 line 76).
# =============================================================================
def add_numeric_categorical_twins(train_df, test_df, cols):
    """
    V12 axis (a), plan §2/V12 line 74 verbatim: "for every numeric column add
    `<col>_cat_ = factorize(...)` as a categorical while keeping every numeric".
    `cols` is the 17 NUMERIC raw predictors (13 RATINGS + Age + Flight Distance +
    both delays) = ALL_FEATURES minus CATS; the ~240 derived numeric columns (per-fold
    TE, freq/rarity, route profile, aggregates, digits, org means) deliberately get NO
    twin - they are continuous estimates, not levels, and factorising one `_tm_*` column
    alone would hand PyTabKit hundreds of thousands of singleton categories (see MEMORY GUARD in the module
    docstring; runbook §3 budgets this arm at ~21 categorical columns).
    NO numeric column is dropped anywhere (runbook §5 T7 "every numeric kept"); dropping
    them is the all-categorical variant measured at -0.00417 (plan §2/V12 line 76).

    RNG-free by construction (runbook §3): pd.factorize over the concatenated train+test
    pool uses no RNG, no sampling and no labels; codes are int64 and NaN becomes code -1,
    i.e. missing becomes a level of its own (the published kernels' `__MISSING__`).
    Both frames are cast to CategoricalDtype with the IDENTICAL level list, so train,
    validation and test share one code space and no fold can meet an unseen twin level.
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
    V12 axis (b), plan §2/V12 line 74 verbatim: "feed `_og_teacher` as a logit rather
    than a probability". Called for the train frame and the test frame with the SAME two
    statements, so the transform is identical in both (and identical for every fold -
    nothing here is per-fold).
    Clip 1e-6 is the house precedent from `S6E10_V3_DuoTeacher.py` line 357
    (np.clip(p, 1e-6, 1 - 1e-6) before np.log(p/(1-p))). It is required, not cosmetic:
    the teacher is a 600-tree float32 XGB whose output really does saturate at exactly
    0.0 and 1.0, log(0) = -inf, and PyTabKit robust-scales the continuous block, so an
    inf would poison that column's median/IQR and everything downstream. The transform
    is order-preserving (1e-6 of probability resolution is invisible to AUC at 700k
    rows), so this half of the axis changes the representation, not the information.
    """
    p = np.clip(df['_og_teacher'].to_numpy(dtype='float64'), 1e-6, 1.0 - 1e-6)
    df['_og_teacher'] = np.log(p / (1.0 - p)).astype('float32')
    return df


def estimate_expanded_matrix(X_fit):
    """
    MEMORY GUARD (runbook §3: "the risk is the transformed/one-hot tensor width
    (max_one_hot_cat_size=18 keeps embeddings small; V12 twins add ~21 categorical
    columns - re-check pytabkit's expanded matrix fits)").
    Estimate of the width PyTabKit's transform expands X_fit to, from the frame's own
    dtypes and category counts rather than from documentation memory: a categorical with
    at most max_one_hot_cat_size levels is one-hot (contributes n_levels columns), a
    wider one gets an embedding (contributes embedding_size columns - the one term that
    cannot be verified without running pytabkit, hence the ESTIMATE label at the print).
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
    float32. This is forced by the architecture change, declared as part of this
    axis - V2 already ships _arr_delay_isna so the missingness signal is not lost.
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
    print(f"Axis: V7 recipe frozen; net input representation only (numeric twins + teacher-as-logit)")
    print("="*80)

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

    # ---- V12 AXIS, part (a): numeric -> categorical TWINS of the 17 numeric raw
    # predictors, every numeric kept (plan §2/V12 line 74). Applied to the frame this
    # script hands to the NET; trees are not trained here and the twin representation on
    # trees is V17's question (plan §2/V12 line 76). Placed AFTER the float32 alignment
    # loop on purpose: the twins are CategoricalDtype columns and must never pass through
    # a float32 cast.
    TWIN_SOURCE = [c for c in ALL_FEATURES if c not in CATS]
    train, test, twin_cols = add_numeric_categorical_twins(train, test, TWIN_SOURCE)
    print(f"   V12 axis (a): +{len(twin_cols)} numeric->categorical twins "
          f"(source {len(TWIN_SOURCE)} cols, no numeric dropped) -> frame {len(test.columns)} cols")

    # ---- V12 AXIS, part (b): the teacher enters as its LOGIT (same function, both
    # frames, no per-fold branch; clip 1e-6 is the archived V3 precedent).
    sat = int((train['_og_teacher'] <= 0.0).sum() + (train['_og_teacher'] >= 1.0).sum())
    p_min, p_max = float(train['_og_teacher'].min()), float(train['_og_teacher'].max())
    train = teacher_to_logit(train)
    test = teacher_to_logit(test)
    print(f"   V12 axis (b): _og_teacher -> logit (clip 1e-6) | prob range [{p_min:.6f}, {p_max:.6f}] "
          f"| saturated train cells {sat:,}")

    FEATURES = list(test.columns)

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training RealMLP_TD ({CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    fit_times   = []   # print-support only: the fit seconds V7 already computes inline,
                       # kept so the RESULTS block can print them per fold (see docstring)
    exp_widths  = []   # per-fold ESTIMATE of PyTabKit's expanded (one-hot + embedding) width
    frame_cols  = []   # per-fold column count actually handed to PyTabKit (width print)

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        prep_start = time.time()
        X_train, X_val, X_test, n_added = build_fold_features(
            X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), test_X.copy(),
            y.iloc[train_idx], fold)
        nan_before = int(X_train.isna().sum().sum() + X_val.isna().sum().sum()
                         + X_test.isna().sum().sum())
        X_train, X_val, X_test = fill_continuous_nan(X_train, X_val, X_test)
        nan_after = int(X_train.isna().sum().sum() + X_val.isna().sum().sum()
                        + X_test.isna().sum().sum())
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        print(f"      Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Prep: {prep_s:.0f}s")

        # V12 MEMORY GUARD, printed BEFORE the fit: the width PyTabKit is about to
        # expand the frame to (runbook §3 - the transformed/one-hot tensor is this
        # arm's only real risk; §7's 12 GB alert becomes an alert line, no trackio).
        exp_width, n_num_cols, oh_map, emb_map, worst_width = estimate_expanded_matrix(X_train)
        exp_widths.append(exp_width)
        frame_cols.append(int(X_train.shape[1]))
        print(f"      V12 expanded-matrix ESTIMATE (absolute): {X_train.shape[1]} frame cols = "
              f"{n_num_cols} numeric (every one kept) + "
              f"{len(CATS) + len(twin_cols)} categorical [{len(CATS)} passengers + "
              f"{len(twin_cols)} V12 twins], of which {len(oh_map)} one-hot "
              f"(<= max_one_hot_cat_size={REALMLP_PARAMS['max_one_hot_cat_size']}), "
              f"{len(emb_map)} embedded ({REALMLP_PARAMS['embedding_size']} cols each)"
              f" -> {exp_width} expanded cols "
              f"(all-one-hot worst case {worst_width})")
        net_memory_report(f"pre-fit fold {fold+1}", X_train, X_test, exp_width, worst_width)

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        # X_val/y_val positional, as in our S6E2/S6E3/S6E5 RealMLP scripts. They
        # are REQUIRED: fit() without a validation block silently drops ~20% of the
        # training rows (measured +0.00061 when passed). Epoch-honest: no checkpoint
        # reuse across folds, no pretrained weights.
        model.fit(X_train, y_train.values, X_val, y_val.values)

        val_probs = np.asarray(model.predict_proba(X_val))[:, 1].astype('float64')
        oof_probs[val_idx] = val_probs
        test_probs += np.asarray(model.predict_proba(X_test))[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (getattr(model, 'best_epoch', None)
                     or getattr(model, 'best_epoch_', None) or REALMLP_PARAMS['n_epochs'])
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        fit_times.append(fold_time - prep_s)
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s (train {fold_time-prep_s:.0f}s) | Total: {elapsed:.1f}min")

        if fold == 0:
            # a net has no feature_importances_; what must be logged here is the
            # config actually handed to PyTabKit and the device it trained on
            print(f"\n      RealMLP config (fold 1): n_ens={REALMLP_PARAMS['n_ens']} "
                  f"n_epochs={REALMLP_PARAMS['n_epochs']} "
                  f"lr_sched={REALMLP_PARAMS['lr_sched']} lr={REALMLP_PARAMS['lr']} "
                  f"hidden={REALMLP_PARAMS['hidden_sizes']}")
            n_cat_cols = len([c for c in X_train.columns
                              if isinstance(X_train[c].dtype, pd.CategoricalDtype)])
            print(f"      Device: "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
                  f"| cols {X_train.shape[1]} | cat cols {n_cat_cols} "
                  f"| best epoch {best_iter}")
            print(f"      NaN filled (fit-fold medians): {nan_before} -> 0")
            # V12 MEMORY GUARD at fold 1 END, absolute readings only.
            net_memory_report("fold 1 end", X_train, X_test, exp_width, worst_width)

        del X_train, X_val, X_test, y_train, y_val, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Best iters: {best_iters}")
    print(f"   Prep times/fold: {[f'{p:.0f}s' for p in prep_times]}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred)")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Axis: {len(twin_cols)} numeric->categorical twins (every numeric kept) + "
          f"_og_teacher as logit | REALMLP_PARAMS byte-identical to V7")
    print(f"   PyTabKit width per fold: cols handed to fit {frame_cols} | "
          f"expanded (one-hot + embedding) estimate {exp_widths}")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best epochs (budget {REALMLP_PARAMS['n_epochs']}; val set = the CV fold): {best_iters}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
