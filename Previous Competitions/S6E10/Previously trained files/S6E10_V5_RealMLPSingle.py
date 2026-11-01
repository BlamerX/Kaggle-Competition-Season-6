"""
S6E10 V5 - RealMLP Single (architecture axis only, V2 pipeline frozen)
================================================================================
Strategy: keep the V2 feature pipeline, folds, seed and orig-only teacher EXACTLY
          as they are and change one axis: the main model XGBoost -> PyTabKit
          RealMLP_TD. This tests the architecture axis, not the feature axis.
          Epoch-honest: no fold checkpointing, no pretrained weights carried
          between folds; X_val is passed (omitting it silently trains on 80%).

Reference: https://www.kaggle.com/code/ektarr/xgboost-baseline
- params, curated TE/freq/rarity, native cats, orig-overlap hygiene,
  teacher ORIGINAL_MODEL_PARAMS (d8/cs0.5/ss0.8/600, all 21 features),
  wide distance profile (mean+std+residual+shares), triple org means
Reference: https://www.kaggle.com/code/goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127
- route profile (+0.00160 CV), orig-teacher-as-feature (+0.00072 CV)
Reference: https://www.kaggle.com/code/evgendvorkin/lightgbm-cv-5-folds-0-96047
- digit features (+0.00069), org target means, fold-safe TE (+0.00038)

Reference: https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
- 62 votes, the most-voted kernel of this episode; RealMLP_TD + PyTabKit recipe
Reference: https://www.kaggle.com/code/goodpjw2008/s6e10-realmlp-aux-task-features-lb-0-96141
- RealMLP with flat_anneal is the best SINGLE model measured here: 0.961344 CV;
  aux-EV delivery to the most-different carrier, NOT to the best one
Reference: https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe
- S6E8 1st place was won by a STOCK RealMLP single: 0.97070 CV, +0.00033 over his
  own 380-model ensemble; our own Tier-A history agrees (S6E2 V48 0.95395 and
  S6E5 V28 0.95357 each beat that episode's best GBDT)

Control V2: 0.96101 CV / 0.96051 LB (v2_oof.csv = paired-DeLong control).
V5 changes exactly ONE axis vs V2:
    main model XGB (d6 hist, mcw5, ss0.9, cs0.85, lr0.04, max_bin256, es150)
    -> RealMLP_TD (PyTabKit, n_ens=1, flat_anneal, 6 epochs ES-limited)
Everything else is byte-identical to V2: FE functions, int-key pair builder, per-fold
TargetEncoder(smooth=20, inner cv=5), frequency/rarity, native categorical vocab, wide
route profile, org means + triple, orig-only teacher column, folds, seed, float32.
FORCED SUB-DECISION (account for it when reading the delta): PyTabKit raises on NaN in
continuous columns while XGBoost routes them, so V2's NaN (292 Arrival-Delay rows,
all-zero-rating rows, TE/route keys unseen in the fit fold) are filled with FIT-FOLD
MEDIANS. This is a second change, not a free one - Step 0 measured fill choices moving
AUC by up to 0.000415, so a delta smaller than that is not attributable to the network
alone. Fold 1 logs how many cells were filled.
Gate: paired DeLong vs v2_oof, accept only dAUC > +0.00005 AND z > 3, reproduce with
RANDOM_SEED=2027. Expected from the evidence above: +0.0003 to +0.0004 CV.
A RealMLP that lands at V2's CV is a RE-WRAP, not progress - record it, do not ship it.

Device: GPU (cuda)

Key Techniques:
1. RealMLP_TD (PyTabKit): n_ens=1, n_epochs=6 ES-limited (additive patience 3 on
   val metric '1-auc_ovr'), lr_sched='flat_anneal', hidden [512,256,128], act silu,
   PLR + label smoothing + tfms as in our S6E3_V73 / S6E5_V21 scripts; X_val block
   always passed (fit() without it silently trains on ~80% of the rows)
2. Native categorical: 4 cat cols via CategoricalDtype (fit-fold vocabulary,
   unseen -> NaN), enable_categorical=True
3. Curated per-fold TE (smooth=20, inner cv=5): 21 singletons + 10 pair keys
   (int-factorized) + FD exact/div10/div100 + FD 250-bin + Age 5-bin
4. Frequency + rarity (-log clip) on singletons + pairs, fit-fold maps
5. Wide target-free route profile: per-FD mean + std + residual + categorical
   shares + count over the train+test pool
6. Multi-column aggregates (tree-inexpressible): count_ge4_inflight8 family,
   mean/std/min_excl0, count_0, delay_gap/total/per_100mi, 8-position digits
7. Original data as statistics: org per-value means for all 21 features +
   (Travel,Class,Customer) triple + teacher XGB (d8, native cats, 21 cols) on
   data.csv with exact-overlap rows excluded (ektarr spec)
8. all numeric matrices float32 (columnar dtype discipline)
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
    VERSION_NUMBER = 5
    VERSION_NAME = "V5_RealMLPSingle"
    EXP_ID = "S6E10_V5_RealMLPSingle"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42

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
# PyTabKit RealMLP_TD. Argument names are copied from our own working scripts
# (S6E3_V73 / S6E5_V21 use n_epochs, batch_size, lr_sched, use_early_stopping,
# early_stopping_*_patience, val_metric_name, verbosity, tfms - 'epochs'/'batch'/
# 'epoch_scheduler'/'auto_configure' are NOT this build's names).
# n_ens=1 keeps the version a single model (the sources ran 8/24/32 = bagged).
# Epochs are ES-limited on the fold's validation rows, same in-fold ES bias as V2.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 1,
    'n_epochs': 6,
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 3,
    'early_stopping_multiplicative_patience': 1,
    'lr': 0.075,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,
    'embedding_size': 8,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'flat_cos',
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'ls_eps': 0.02,
    'ls_eps_sched': 'cos',
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
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | {CFG.EXP_ID}")
    print(f"Device: GPU (cuda) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: statistics + ektarr-spec teacher (no row concat)")
    print("V2 pipeline frozen; single axis: main model XGB -> RealMLP_TD (flat_anneal)")
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

    FEATURES = list(test.columns)
    print(f"   Total base features: {len(FEATURES)}")

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
            print(f"      Device: "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
                  f"| cols {X_train.shape[1]} | cat cols {len(CATS)} "
                  f"| best epoch {best_iter}")
            print(f"      NaN filled (fit-fold medians): {nan_before} -> 0")
            print()

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
    print(f"\n{'='*80}")
    print(f"{CFG.VERSION_NAME} RESULTS — RealMLP_TD single, V2 pipeline (GPU)")
    print(f"{'='*80}")
    print(f"Base features: {len(FEATURES)} + per-fold TE/freq/rarity additions")
    print(f"V2 control: 0.96101 CV / 0.96051 LB (v2_oof.csv = paired-DeLong control)")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
