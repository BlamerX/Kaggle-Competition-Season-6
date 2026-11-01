"""
S6E10 V33 - V18 reproduced at SEED_REPRO=2027 (the control gate's second half; no axis, no new model) - GPU (cuda)
GOAL: this episode's gate has two halves - paired DeLong on the saved vector, then reproduction at
      RANDOM_SEED=2027. V18 passed the first half on 2026-10-12 (+0.000115 vs V12, z=+4.87, 5/5 folds;
      +0.000278 vs V2, z=+5.12, 5/5 - all recorded K1 measurements) and the second half has never run, so the standing control is only
      half-gated. It is numbered V33 so it gets scheduled, costed and logged like every other arm, but it
      owns no axis and is not a ladder rung: the distance between v33_oof and v18_oof is what "same config,
      different seed" actually costs here, which is the number every future ACCEPT must be read against.
Axis: none - ONE config line, RANDOM_SEED 42 -> 2027. The recipe, the frame, the EV block, the twins, the
      budget, the folds and every parameter are V18's byte-for-byte, and the outputs are v33_* so nothing
      overwrites the archived v18_oof.csv this run is measured against. `AUX_EV_PARAMS` keeps random_state 0 in both runs, exactly as
      V11 and V18 specify, so the seed swap is the only delta.
Parent: Previously trained files/Archieve/S6E10_V18_RealMLPFusionEV.py (frozen, recorded
         0.961287 CV / 0.96106 LB).
Device: GPU (cuda) - RealMLP_TD via PyTabKit on a Kaggle T4 (16 GB); all numeric matrices float32.
Folds: StratifiedKFold(5, shuffle=True, random_state=2027) - the reproduction seed, nothing else moves.
Gate: none to earn - offline paired DeLong of v33_oof against v18_oof IS the read.
Expected / Cost: one V18-length run (~18 min GPU) and a seed-noise datum, not a gain claim.
Read: the distance between the two vectors: Expected: a small |dAUC| with z inside +-3 and
      fold agreement around 3/5. A LARGE distance is not an improvement, it is a warning that V18's CV
      is seed-fragile, and the control bar would then have to be re-derived before any arm is promoted.
Outputs: v33_oof.csv (id, pred) + v33_sub.csv (id, satisfaction) to /kaggle/working.
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
    VERSION_NUMBER = 33   # own stem v33_* - it must never write v18_* (the vector it is read against)
    VERSION_NAME = "V33_V18SeedRepro"
    EXP_ID = "S6E10_V33_V18SeedRepro"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 2027       # V33: THE ONLY CONFIG CHANGE VS V18 - this run IS the reproduction
    SEED_REPRO = 2027        # reproduction run changes only RANDOM_SEED (42 -> 2027)

    # V11's builder switch, inherited unchanged: True = predictors fitted on the
    # overlap-cleaned original (~1 min), False = 5-fold cross-fit over ~1M pooled rows.
    AUX_EV_FIT_ON_ORIGINAL = True

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
# V18's axis is a FEATURE change, so the recipe is not touched: this dict is V12's, byte-for-byte,
# n_epochs 3 included - the epoch ladder is V17's axis, not this arm's.
# Argument names come from our own proven scripts - 'epochs'/'batch'/'epoch_scheduler'/'auto_configure'
# are NOT this build's names, and invented kwargs cost five failed runs in this episode.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,        # internal bagging inside one model object, not an ensemble
    'n_epochs': 3,     # V12's budget, held unchanged
    'batch_size': 256,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 3,
    'early_stopping_multiplicative_patience': 1,
    'lr': 0.053,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'wd_sched': 'cos_log_15',
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'embedding_size': 5,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
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
    # 'ens_av_before_softmax' stays unset (probability averaging = V12 behaviour).
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

# Inherited from V11 unchanged: the 13 EV predictors. multi:softprob over the rating levels.
AUX_EV_PARAMS = {
    'objective': 'multi:softprob',
    'n_estimators': 300,
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
# 6b. INHERITED FROM V11 - THE 13 LABEL-FREE AUX-EV COLUMNS (this version's axis)
# =============================================================================
def add_aux_ev_features(train_df, test_df, orig_clean):
    """
    V11's axis, inherited unchanged: add 13 `aux_ev_<rating>` columns, EV = sum_k k * P_hat(rating=k),
    where each P_hat comes from an XGB predicting that rating from the OTHER 20 ALL_FEATURES columns
    (never `satisfaction`). They are label-free feature producers, not a second competing model: only
    the net predicts satisfaction and nothing is blended at inference.

    LEAKAGE GUARD: the aux stage is fitted ONCE globally, never per outer fold, and no row's EV may
    come from a model that saw that row. DEFAULT (CFG.AUX_EV_FIT_ON_ORIGINAL=True, run-of-record)
    fits only on the overlap-cleaned original - prepare_original has dropped every original row equal
    to a competition row, so no competition row is in any fit set.
    """
    def prep(frame, cols, vocab):
        X = frame[cols].copy()
        for c in cols:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('float32')
            else:
                X[c] = X[c].astype('string').astype(
                    pd.CategoricalDtype(categories=vocab[c]))
        return X

    def encode(y):
        """Contiguous 0-based codes + the ascending rating levels actually present. The level vector
        is load-bearing: `Baggage handling` is 1-5 with NO 0, so EV must use the real k."""
        vals = y.to_numpy(dtype='int64')
        levels = np.unique(vals)
        return np.searchsorted(levels, vals).astype('int64'), levels.astype('float32')

    def ev_of(model, X, levels):
        return np.asarray(model.predict_proba(X), dtype='float32') @ levels

    if CFG.AUX_EV_FIT_ON_ORIGINAL:
        vocab = {c: sorted(orig_clean[c].astype('string').unique().tolist())
                 for c in CATS}
        for r in RATINGS:
            other = [c for c in ALL_FEATURES if c != r]
            yr = pd.to_numeric(orig_clean[r], errors='coerce')
            fit_rows = yr.notna()
            ycode, levels = encode(yr[fit_rows])
            m = xgb.XGBClassifier(**AUX_EV_PARAMS)
            m.fit(prep(orig_clean.loc[fit_rows], other, vocab), ycode, verbose=False)
            name = f'aux_ev_{feature_name(r)}'
            train_df[name] = ev_of(m, prep(train_df, other, vocab), levels).astype('float32')
            test_df[name] = ev_of(m, prep(test_df, other, vocab), levels).astype('float32')
            del m, other, yr, fit_rows, ycode, levels, name
            gc.collect()
    else:
        pool = pd.concat([train_df[ALL_FEATURES], test_df[ALL_FEATURES]],
                         ignore_index=True)
        vocab = {c: sorted(pool[c].astype('string').dropna().unique().tolist())
                 for c in CATS}
        n_train_rows = len(train_df)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1042)
        for r in RATINGS:
            other = [c for c in ALL_FEATURES if c != r]
            Xp = prep(pool, other, vocab)
            yr = pool[r].astype('int64')
            ev = np.zeros(len(pool), dtype='float32')
            for fit_idx, hold_idx in inner.split(Xp, yr.to_numpy()):
                ycode, levels = encode(yr.iloc[fit_idx])
                m = xgb.XGBClassifier(**AUX_EV_PARAMS)
                m.fit(Xp.iloc[fit_idx], ycode, verbose=False)
                ev[hold_idx] = ev_of(m, Xp.iloc[hold_idx], levels).astype('float32')
                del m, ycode, levels
                gc.collect()
            name = f'aux_ev_{feature_name(r)}'
            train_df[name] = ev[:n_train_rows]
            test_df[name] = ev[n_train_rows:]
            del Xp, yr, ev, other, name
            gc.collect()
        del pool
        gc.collect()

    print(f"      EV/raw-rating Pearson r (train) - instability check:")
    for r in RATINGS:
        name = f'aux_ev_{feature_name(r)}'
        a = train_df[name].to_numpy(dtype='float64')
        b = pd.to_numeric(train_df[r], errors='coerce').to_numpy(dtype='float64')
        ok = ~(np.isnan(a) | np.isnan(b))
        if np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            print(f"        {name:<34s} r = n/a (constant column)")
        else:
            print(f"        {name:<34s} r = {np.corrcoef(a[ok], b[ok])[0, 1]:+.4f}")
    return train_df, test_df

# =============================================================================
# 6c. INHERITED FROM V12 - THE NET'S INPUT REPRESENTATION (twins + teacher-as-logit)
# =============================================================================
def add_numeric_categorical_twins(train_df, test_df, cols):
    """
    V12's axis, carried unchanged. `cols` is the 17 NUMERIC raw predictors (ALL_FEATURES minus
    CATS) - the derived numeric columns, the EVs included, deliberately get NO twin: they are
    continuous estimates, not levels, and factorising one of them would hand PyTabKit hundreds of
    thousands of singleton categories. NO numeric column is dropped.
    RNG-free by construction: pd.factorize over the concatenated train+test pool uses no RNG and no
    labels; NaN becomes code -1, i.e. missing becomes a level of its own. Both frames share one code
    space, so no fold can meet an unseen twin level.
    """
    twin_cols = []
    n_train = len(train_df)
    pool = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    oh_limit = REALMLP_PARAMS['max_one_hot_cat_size']
    for c in cols:
        codes = pd.factorize(pool[c])[0]
        levels = np.unique(codes)
        tr = pd.Categorical(codes[:n_train], categories=levels)
        te = pd.Categorical(codes[n_train:], categories=levels)
        if int(np.asarray(tr.isna()).sum() + np.asarray(te.isna()).sum()):
            raise ValueError(f"twin of {c} produced an out-of-vocabulary level")
        name = f"{c}_cat_"
        train_df[name] = tr
        test_df[name] = te
        twin_cols.append(name)
        treat = (f"one-hot ({len(levels)} cols)" if len(levels) <= oh_limit
                 else f"embedding ({REALMLP_PARAMS['embedding_size']} cols)")
        print(f"      twin {name}: {len(levels):,} levels -> {treat}")
    return train_df, test_df, twin_cols


def teacher_to_logit(df):
    """V12's axis (b), unchanged: logit of the teacher probability, clip 1e-6 (V3's house value)
    because the teacher saturates at exactly 0/1 and an inf would poison robust-scaling."""
    p = np.clip(df['_og_teacher'].to_numpy(dtype='float64'), 1e-6, 1.0 - 1e-6)
    df['_og_teacher'] = np.log(p / (1.0 - p)).astype('float32')
    return df


def estimate_expanded_matrix(X_fit):
    """Width PyTabKit's transform expands X_fit to, from the frame's own dtypes and category counts.
    Returns (width, n_numeric, one_hot_dict, embedding_dict, worst_all_one_hot_width)."""
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
    """MEMORY GUARD print, ABSOLUTE numbers only. Tensor GB for the estimated expanded width, the
    16 GB T4 / 12 GB alert check, the all-one-hot worst-case bound, and the device counters.
    V18 raises V12's estimate by 13 numeric columns, so the alert is a hard abort."""
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
           else "ALERT: OVER THE 12 GB LINE - ABORT, DO NOT SILENTLY SHRINK THE FEATURE SET")
        + f" | all-one-hot worst case would be {gb_worst:.2f} GB "
          f"(width {worst_width}: prevented by max_one_hot_cat_size="
          f"{REALMLP_PARAMS['max_one_hot_cat_size']})",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        peak_res = torch.cuda.max_memory_reserved(0) / gib
        lines.append(
            f"{tag}: device {props.name} total {props.total_memory / gib:.2f} GB "
            f"| allocated {torch.cuda.memory_allocated(0) / gib:.2f} GB "
            f"| reserved {torch.cuda.memory_reserved(0) / gib:.2f} GB "
            f"| peak reserved {peak_res:.2f} GB")
    else:
        lines.append(f"{tag}: CUDA UNAVAILABLE - this run is not on the T4 the arm was budgeted for")
    for line in lines:
        print(f"      {line}")
    if gb_fit + gb_test > 12.0:
        raise MemoryError(f"{tag}: estimated transformed matrix {gb_fit + gb_test:.2f} GB exceeds "
                          f"the 12 GB alert line on a 16 GB device. Do not narrow the frame "
                          f"silently - that is a new axis and needs a new version.")
    return lines

# =============================================================================
# 7. PER-FOLD FEATURE BUILDER (curated TE + frequency/rarity + native cats)
# =============================================================================
def feature_name(column):
    return column.lower().replace(' ', '_').replace('/', '_').replace('-', '_')


def make_key_frames(fit, frames):
    """Int-factorized key columns for TE + frequency. Pair code = code_a * n_cat_b + code_b using
    vocabularies fitted on the fit fold only."""
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
    """PyTabKit raises on NaN in continuous columns; V2 relied on XGBoost routing them. Medians come
    from the FIT FOLD only (no val/test statistics). Inherited from V12 unchanged."""
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X_fit.columns if c not in cat_cols]
    med = X_fit[num].median(numeric_only=True).astype('float32')
    out = []
    for f in (X_fit, X_val, X_test):
        g = f.copy()
        g[num] = g[num].fillna(med)
        g[num] = g[num].astype('float32')
        for c in cat_cols:
            cats = g[c].cat.categories
            g[c] = g[c].astype('object').fillna(cats[0]).astype(pd.CategoricalDtype(cats))
        left = g.isna().sum()
        left = left[left > 0]
        if len(left):
            raise ValueError(f"NaN survived fill: {dict(left)}")
        out.append(g)
    return out


def build_fold_features(X_fit, X_val, X_test, y_fit, fold):
    """Curated per-fold TE (smooth=20, inner cv=5), frequency+rarity, native categorical
    vocabularies from the fit fold. Returns (3 frames, n_added)."""
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
            f[c] = f[c].astype('object').fillna(cats[0]).astype(dt)

    n_added = len(frames[0].columns) - len(X_fit.columns)
    return frames[0], frames[1], frames[2], n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # V12 is the parent, so REALMLP_PARAMS is handed over unchanged and EPOCH is only a print alias
    # for the budget actually in that dict. The epoch ladder is V17's axis, not this arm's.
    EPOCH = int(REALMLP_PARAMS['n_epochs'])

    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED}")
    print(f"Axis: none - V18 reproduced at RANDOM_SEED={CFG.RANDOM_SEED}; read offline against v18_oof")
    print(f"Parent V18: 0.961287 CV / 0.96106 LB (recorded) | this run is the gate's second half")
    print("="*80)

    # ARG-NAME LAW pre-flight: the recipe dict is inherited, but a silently-absorbed unknown key
    # would still invalidate the comparison, so any key the build does not name is printed.
    _SIG = inspect.signature(RealMLP_TD_Classifier.__init__).parameters
    _unnamed = [k for k in REALMLP_PARAMS if k not in _SIG]
    print(f"RealMLP_TD_Classifier accepts n_epochs: {'n_epochs' in _SIG} "
          f"| keys absorbed by **kwargs: {_unnamed if _unnamed else 'none'}")
    if 'n_epochs' not in _SIG:
        raise ImportError("This PyTabKit build has no n_epochs parameter - stopping before the fits.")

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

    # ---- V18 AXIS: the 13 aux-EV columns, added here - AFTER the teacher and BEFORE the float32
    # cast and the twin pass, exactly where V11 put them, so the fold driver / TE / recipe stay
    # byte-identical and the EVs can never be mistaken for twin sources.
    print("   Adding V18 axis: 13 label-free aux_ev columns...")
    t0_ev = time.time()
    train, test = add_aux_ev_features(train, test, orig_clean)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    assert len(EV_COLS) == 13, f"expected 13 aux_ev columns, got {len(EV_COLS)}"
    print(f"      {ev_build_s:.0f}s EV build | +{len(EV_COLS)} columns on the net frame")

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    # ---- INHERITED FROM V12 (a): twins of the 17 numeric RAW predictors only. EV columns are not
    # in ALL_FEATURES, so they are structurally excluded from TWIN_SOURCE - no EV twin exists here.
    TWIN_SOURCE = [c for c in ALL_FEATURES if c not in CATS]
    train, test, twin_cols = add_numeric_categorical_twins(train, test, TWIN_SOURCE)
    print(f"   V12 axis (a) inherited: +{len(twin_cols)} twins (source {len(TWIN_SOURCE)} cols) "
          f"-> frame {len(test.columns)} cols")

    # ---- INHERITED FROM V12 (b): teacher as logit.
    train = teacher_to_logit(train)
    test = teacher_to_logit(test)
    FEATURES = list(test.columns)

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training RealMLP_TD (n_epochs={EPOCH}, {CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    fit_times   = []
    exp_widths  = []
    frame_cols  = []

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
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s")

        exp_width, n_num_cols, oh_map, emb_map, worst_width = estimate_expanded_matrix(X_train)
        exp_widths.append(exp_width)
        frame_cols.append(int(X_train.shape[1]))
        print(f"      V18 expanded-matrix ESTIMATE (absolute): {X_train.shape[1]} frame cols = "
              f"{n_num_cols} numeric (incl. {len(EV_COLS)} EV) + "
              f"{len(CATS) + len(twin_cols)} categorical, {len(oh_map)} one-hot, "
              f"{len(emb_map)} embedded -> {exp_width} expanded cols "
              f"(all-one-hot worst case {worst_width})")
        net_memory_report(f"pre-fit fold {fold+1}", X_train, X_test, exp_width, worst_width)

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        # X_val/y_val positional, as in our S6E2/S6E3/S6E5 RealMLP scripts: fit() without a
        # validation block silently drops ~20% of the training rows. No checkpoint reuse across folds.
        t_fit = time.time()
        model.fit(X_train, y_train.values, X_val, y_val.values)
        fit_times.append(time.time() - t_fit)

        val_probs = np.asarray(model.predict_proba(X_val))[:, 1].astype('float64')
        oof_probs[val_idx] = val_probs
        test_probs += np.asarray(model.predict_proba(X_test))[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (getattr(model, 'best_epoch', None)
                     or getattr(model, 'best_epoch_', None) or EPOCH)
        best_iters.append(best_iter)

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | best epoch {best_iter} (ceiling {EPOCH}) | "
              f"fit {fit_times[-1]:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, y_train, y_val, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_cv = auc_score(y.values, oof_probs)

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    pd.DataFrame({'id': train_id, 'pred': oof_probs}).to_csv(
        f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred)")

    pd.DataFrame({'id': test_id, CFG.TARGET: test_probs}).to_csv(
        f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Read: distance to V18's vector is the seed-fragility datum, not a gain claim")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best epochs: {best_iters} | EV build {ev_build_s:.0f}s")
    print(f"   PyTabKit width per fold: cols handed to fit {frame_cols} | expanded estimate {exp_widths}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
