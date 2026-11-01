"""
S6E10 V55 - RealRowsAugment: the original 129,880 real rows enter the FINAL FIT as training rows - GPU (cuda)
Axis: ONE change vs V18 - a held-out slice of the overlap-cleaned original is APPENDED to each fold's training rows.
      The estimator (RealMLP_TD), its parameter dict, the 289-column competition frame, the frozen
      StratifiedKFold(5, shuffle, seed 42) driver, n_epochs 3 and the validation block are all byte-identical to V18;
      competition rows' features are untouched, so the archived V18 run is the leave-real-rows-out control.
Why: a static audit of all 54 archived arms found this mechanism class never tested - originals have only ever been
      feature producers ([V18:791] `X = train.drop(TARGET)` fits competition rows; [V18:352-468] supply the teacher,
      the 13 aux-EV and the org-statistics; V27 fits RealMLP on originals but only as a teacher column, and V48 appends
      columns, never rows). The V51 appendix names this data regime as the one source of an inductive bias the pool
      does not contain. Evidence against, priced honestly: Sachith7's paired kit found appending originals negative in
      6/6 comparisons (-0.000446) - on a 180-tree LightGBM, far weaker than this net.
LEAKAGE DESIGN (the reason this arm is not a concat): the cleaned original is split R 80% / A 20% (seed 9121,
      stratified). A is the augmentation. Every producer that can see A's own `satisfaction` label is refitted on R
      ALONE before transforming A - org-statistics, the teacher, and the aux-EV block (V11's guard is "no row's EV may
      come from a model that saw that row", and A rows ARE original rows). The per-fold target encoder is fitted on the
      competition fit fold only and merely transforms A. Competition features and the route-profile pool stay V18's, so
      the axis is the appended rows and nothing else. Cost of that safety, declared: A's producer-fitted columns come
      from an 80% fit while competition rows' come from the 100% original, so the two halves differ slightly in
      semantics - the alternative (one shared fit) would confound the axis with a feature change.
Guard: the twin code space is the COMPETITION pool's (V12's axis), so an A value the pool never carried - or a NaN
      where the pool carries none - has no code: such rows are DROPPED and counted, because widening the
      vocabulary would change one-hot twin widths for the competition rows too (a second axis, on the model input)
      and the categorical fill sending them to categories[0] would dress an extreme delay as a zero. Measured on
      the real frames: 1,241 of 25,972 A rows (4.78%) over three columns, pos rate +0.0002. Above the 10% cap the
      design, not the threshold, is wrong and the run stops. Each producer runs ONCE over all three frames, so
      competition features cannot move; the A frame's schema is checked column-for-column.
Admission (pre-declared, and raised by this week's composition measurement): the arm ships only if its SOLO CV
      reaches member strength (>= 0.96140, the level of v36/v44) AND dCV vs v18_oof >= +0.00005 AND >=4/5 house folds
      improved AND paired DeLong z > 3, then reproduce at SEED_REPRO 2027. Rationale measured 2026-10-28: a blend leg
      10 rulers under the members cost -0.000039 and one 45 under cost -0.000206, so a sub-0.9614 result is
      pre-declared worthless as a leg and gets no board slot. The CV->LB gap is NOT predicted here - it is family- and
      regime-specific with an observed spread of ~0.0014 - so this arm's board score is the measurement, not a forecast.
Expected / Cost: ~19-25 min on one T4 (V18 was ~18 min; the 24,731 surviving A rows are +4.4% of ~560k fit rows);
      honest prior from the external paired test is a tie-to-negative solo result, and a tie is now KNOWN useless.
Outputs: v55_oof.csv (id, pred) + v55_sub.csv (id, satisfaction) to /kaggle/working - exactly two writes.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    VERSION_NUMBER = 55
    VERSION_NAME = "V55_RealRowsAugment"
    EXP_ID = "S6E10_V55_RealRowsAugment"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027        # reproduction run changes only RANDOM_SEED (42 -> 2027)

    # ---- V55 axis: the augmentation split, fixed here so a re-run cannot re-cut it ----
    AUG_FRAC = 0.20          # share of the overlap-cleaned original appended to each fold's training rows
    AUG_SEED = 9121          # stratified split seed, chosen for this arm and never for the result
    AUG_COVERAGE_CAP = 0.10  # max share of A the twin vocabulary cannot express, else the design is wrong
    MIN_SOLO = 0.96140       # member-strength admission, raised by this week's composition measurement

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
# This arm's axis is WHICH ROWS ARE FITTED, not what fits them, so the recipe is untouched: this dict is V18's
# (and V12's) byte-for-byte, n_epochs 3 included - verified by AST diff before this version was authored.
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


def _read_held(model, key):
    """What the fitted OBJECT records about a config key, in the order sklearn/PyTabKit actually keep it:
    get_params, then the attribute, then a stored params mapping. '<<absent>>' is the only reading that proves
    the key was dropped rather than forwarded. Copied from V53, which ran this against a real PyTabKit build;
    no archived arm before that read anything back, which is why its verdicts rested on banners."""
    try:
        p = model.get_params(deep=False)
        if key in p:
            return p[key]
    except (TypeError, AttributeError, KeyError):
        pass
    if hasattr(model, key):
        return getattr(model, key)
    for holder in ('params', '_params'):
        d = getattr(model, holder, None)
        if isinstance(d, dict) and key in d:
            return d[key]
    return '<<absent>>'

# =============================================================================
# 6. FEATURE ENGINEERING (fold-independent part) - V18's producers, each threaded to transform an A frame too.
#    The competition-row branch of every function here is V18's code path unchanged; check_v55_leak.py proves it.
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


def add_route_profile(train_df, test_df, augment_df=None):
    """
    Wide target-free route profile (ektarr distance_profile + pool-based means):
    per-FD mean, std, residual and categorical shares over the train+test pool.

    V55: the POOL STAYS train+test. `augment_df`, when given, is only TRANSFORMED by these statistics -
    putting A into the pool would change every competition row's features and turn the appended-rows
    axis into a feature axis as well.
    """
    measures = RATINGS + ['Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    pool = pd.concat([train_df, test_df], ignore_index=True)
    fd = pool['Flight Distance']
    gmean = pool.groupby(fd)[measures].mean()
    gstd = pool.groupby(fd)[measures].std()
    cnt = pool.groupby(fd).size()
    shares = {c: pd.crosstab(fd, pool[c], normalize='index') for c in CATS}

    targets = [d for d in (train_df, test_df, augment_df) if d is not None]
    for d in targets:
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
    if augment_df is not None:
        return train_df, test_df, augment_df
    return train_df, test_df


def _org_stats_into(d, ref_df):
    """Write the original-data label-mean columns into ONE frame, using `ref_df` as the only label source.
    Split out of V18's add_org_stats so the augmentation rows can be built from R alone: the per-value means
    are fitted on satisfaction labels, so a row must never be inside the frame whose means it reads."""
    yo = ref_df[CFG.TARGET].astype(int)
    base = float(yo.mean())

    def key_of(df, col):
        if col == 'Flight Distance':
            return df[col] // 10
        return df[col]

    for c in ALL_FEATURES:
        ok = key_of(ref_df, c).astype('string') if not pd.api.types.is_numeric_dtype(ref_df[c]) else key_of(ref_df, c)
        stat = pd.DataFrame({'k': ok, 'y': yo}).groupby('k')['y'].mean()
        name = f'_{SHORT[c]}_org_mean'
        ck = key_of(d, c)
        ck = ck.astype('string') if not pd.api.types.is_numeric_dtype(d[c]) else ck
        d[name] = ck.map(stat).fillna(base).astype('float32')

    def triple(frame):
        a = frame['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        b = frame['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        c = frame['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
        return a * 100 + b * 10 + c
    stat = pd.DataFrame({'k': triple(ref_df), 'y': yo}).groupby('k')['y'].mean()
    d['_triple_org_mean'] = triple(d).map(stat).fillna(base).astype('float32')
    return d


def add_org_stats(train_df, test_df, orig_df, augment_df=None, ref_df=None):
    """
    Original-data label means per value for every feature (ektarr original_means
    protocol: raw group means, unseen -> orig prior) + Travel x Class x Customer
    triple. Ratings/FD keys additionally get k=20 smoothed variants.

    V55: competition rows get V18's org stats unchanged, and the proof is the argument, not a re-fit: V18
    called this function with the RAW original frame (V18:752, before prepare_original runs at V18:755), so
    that is the object passed here. The augmentation rows are built from `ref_df` = the overlap-cleaned R
    slice, which excludes them, so an appended row cannot read its own label.
    """
    train_df = _org_stats_into(train_df, orig_df)
    test_df = _org_stats_into(test_df, orig_df)
    if augment_df is not None:
        assert ref_df is not None, "augment rows need an R-only label source"
        augment_df = _org_stats_into(augment_df, ref_df)
        return train_df, test_df, augment_df
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


def add_original_teacher(train_df, test_df, orig_clean, augment_df=None, ref_df=None):
    """
    Teacher = ektarr spec: XGB d8, cs0.5, ss0.8, 600 trees, ALL 21 features with
    native categoricals, fit on the overlap-cleaned original, prediction added
    as one column. Never touches competition labels.

    V55: the teacher CONSUMES satisfaction, so the augmentation rows cannot be scored by the model that
    was fitted on them. Competition rows keep V18's teacher (fitted on the whole cleaned original, which
    contains no competition row); a SECOND teacher, fitted on R only, produces A's column. Same
    vocabulary (category levels are label-free), so the encoding is identical and only the fit set moves.
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

    def fit_predict(fit_frame, target_frames):
        m = xgb.XGBClassifier(**TEACHER_PARAMS)
        m.fit(prep(fit_frame), fit_frame[CFG.TARGET].astype(int), verbose=False)
        for d in target_frames:
            d['_og_teacher'] = m.predict_proba(prep(d))[:, 1].astype('float32')
        del m
        gc.collect()

    fit_predict(orig_clean, [train_df, test_df])
    if augment_df is not None:
        assert ref_df is not None, "augment rows need an R-only teacher"
        fit_predict(ref_df, [augment_df])
        return train_df, test_df, augment_df
    return train_df, test_df

# =============================================================================
# 6b. INHERITED FROM V11/V18 - THE 13 LABEL-FREE AUX-EV COLUMNS (inherited, NOT this version's axis)
# =============================================================================
def add_aux_ev_features(train_df, test_df, orig_clean, augment_df=None, ref_df=None):
    """
    V11's axis, inherited unchanged: add 13 `aux_ev_<rating>` columns, EV = sum_k k * P_hat(rating=k),
    where each P_hat comes from an XGB predicting that rating from the OTHER 20 ALL_FEATURES columns
    (never `satisfaction`). They are label-free feature producers, not a second competing model: only
    the net predicts satisfaction and nothing is blended at inference.

    LEAKAGE GUARD: the aux stage is fitted ONCE globally, never per outer fold, and no row's EV may
    come from a model that saw that row. DEFAULT (CFG.AUX_EV_FIT_ON_ORIGINAL=True, run-of-record)
    fits only on the overlap-cleaned original - prepare_original has dropped every original row equal
    to a competition row, so no competition row is in any fit set.

    V55 applies that guard's own letter to the augmentation rows: they ARE original rows, so a second
    EV block fitted on R only produces their columns, exactly as the teacher is refitted. Competition
    rows keep V18's original-fit EVs, so the axis stays the appended rows and nothing else.
    """
    if not CFG.AUX_EV_FIT_ON_ORIGINAL:
        if augment_df is not None:
            raise NotImplementedError("V55 threads the augmentation only through the run-of-record "
                                      "original-fit EV branch (CFG.AUX_EV_FIT_ON_ORIGINAL=True)")
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
        # V55: two fit sources, one per population. Competition rows keep V18's block (fitted on the
        # whole cleaned original, which excludes every competition row); augmentation rows are scored
        # by a block fitted on R alone, so no appended row's EV comes from a model that saw it.
        sources = [(orig_clean, [train_df, test_df])]
        if augment_df is not None:
            assert ref_df is not None, "augment rows need an R-only EV source"
            sources.append((ref_df, [augment_df]))
        for r in RATINGS:
            other = [c for c in ALL_FEATURES if c != r]
            name = f'aux_ev_{feature_name(r)}'
            for src, targets in sources:
                yr = pd.to_numeric(src[r], errors='coerce')
                keep = yr.notna()
                ycode, levels = encode(yr[keep])
                m = xgb.XGBClassifier(**AUX_EV_PARAMS)
                m.fit(prep(src.loc[keep], other, vocab), ycode, verbose=False)
                for d in targets:
                    d[name] = ev_of(m, prep(d, other, vocab), levels).astype('float32')
                del m, yr, keep, ycode, levels
                gc.collect()
            del other, name
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
    if augment_df is not None:
        return train_df, test_df, augment_df
    return train_df, test_df

# =============================================================================
# 6c. INHERITED FROM V12 - THE NET'S INPUT REPRESENTATION (twins + teacher-as-logit)
# =============================================================================
def filter_unrepresentable_rows(augment_df, train_df, test_df, cols, cap):
    """V55: keep only the augmentation rows the parent's twin vocabulary can already express.

    Why DROP instead of re-encoding: the twin code space belongs to the train+test pool (V12's axis, RNG-free).
    Giving A's out-of-pool values a category of their own would widen one-hot twin columns for EVERY frame,
    including the competition rows - a second axis, on the model's input, not just on the extra rows. Mapping
    them to an existing level instead (which is what the categorical fill does silently, to categories[0])
    would dress an extreme delay as a zero delay. So an unrepresentable row is removed from the augmentation,
    and the count is reported: 531 rows at 2.04% on one column was the real figure this arm first met.

    `cap` is the honesty limit: if too large a share of A cannot be expressed, the augmentation is not
    representable in this frame at all and the design - not the threshold - has to change. Measured on the real
    frames (measure_v55_coverage_real.py, 2026-10-28): 1,241 of 25,972 rows = 4.78%, from Flight Distance 1.58%,
    Departure Delay 2.04% and Arrival Delay 2.32%, and the drop moves A's pos rate by +0.0002. So the cap is
    above the real figure with room to spare, and the surviving augmentation is not a label-biased subset.
    """
    unrepresentable = np.zeros(len(augment_df), dtype=bool)
    for c in cols:
        pool = pd.concat([train_df[c], test_df[c]], ignore_index=True)
        pool_codes, uniques = pd.factorize(pool)
        carries_missing = -1 in set(int(x) for x in np.unique(pool_codes))
        v = augment_df[c]
        novel = v.notna() & ~v.isin(set(uniques.tolist()))
        missing_unexpressible = v.isna() if not carries_missing else v.notna() & False
        if bool(missing_unexpressible.any()):
            print(f"      coverage [{c}]: the pool carries no missing level, so A's "
                  f"{int(v.isna().sum()):,} NaN rows cannot be encoded")
        bad = (novel | missing_unexpressible).to_numpy()
        n_bad = int(bad.sum())
        if n_bad:
            print(f"      coverage [{c}]: {n_bad:,} of {len(augment_df):,} A rows "
                  f"({n_bad / len(augment_df) * 100:.2f}%) cannot be expressed in the pool's twin codes")
        unrepresentable |= bad
    keep = ~unrepresentable
    rate = int(unrepresentable.sum()) / len(augment_df)
    y = augment_df[CFG.TARGET]
    print(f"      A coverage: {int(unrepresentable.sum()):,} of {len(augment_df):,} rows dropped "
          f"({rate * 100:.2f}%), {int(keep.sum()):,} appended | pos rate "
          f"{y.mean():.4f} -> {y[keep].mean():.4f} on the kept rows")
    if rate > cap:
        raise ValueError(f"{rate * 100:.2f}% of the augmentation cannot be expressed in the parent's twin "
                         f"vocabulary, above the {cap * 100:.0f}% cap - the appended rows are not "
                         f"representable in this frame and the design needs rethinking, not a wider cap.")
    return augment_df[keep].reset_index(drop=True)


def add_numeric_categorical_twins(train_df, test_df, cols, augment_df=None):
    """
    V12's axis, carried unchanged. `cols` is the 17 NUMERIC raw predictors (ALL_FEATURES minus
    CATS) - the derived numeric columns, the EVs included, deliberately get NO twin: they are
    continuous estimates, not levels, and factorising one of them would hand PyTabKit hundreds of
    thousands of singleton categories. NO numeric column is dropped.
    RNG-free by construction: pd.factorize over the concatenated train+test pool uses no RNG and no
    labels; NaN becomes code -1, i.e. missing becomes a level of its own. Both frames share one code
    space, so no fold can meet an unseen twin level.

    V55: the code space stays the COMPETITION pool, and the rows that cannot be expressed in it were
    already removed by filter_unrepresentable_rows - so what is asserted here is the post-condition of that
    step, not a tunable threshold. If any A row is still unmapped, the coverage pass is wrong and the run
    stops: the later categorical fill would route it to categories[0], dressing a novel level as the first.
    """
    twin_cols = []
    n_train = len(train_df)
    pool = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    oh_limit = REALMLP_PARAMS['max_one_hot_cat_size']
    for c in cols:
        codes, uniques = pd.factorize(pool[c])
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
        if augment_df is not None:
            # Post-condition of filter_unrepresentable_rows, expressed in the same terms that function used:
            # a value the pool carries maps to its pool code, and A's NaN maps to the pool's OWN missing
            # level (code -1) when the pool has one. Anything left unmapped means the two views disagree.
            code_of = {v: i for i, v in enumerate(uniques)}
            a_codes = augment_df[c].map(code_of)
            if -1 in set(int(x) for x in levels):
                a_codes = a_codes.where(augment_df[c].notna(), -1)
            unmapped = int(a_codes.isna().sum())
            if unmapped:
                raise AssertionError(f"twin {name}: {unmapped:,} kept A rows still have no pool code, so "
                                     f"coverage filtering and this mapping disagree - the categorical fill "
                                     f"would route them to categories[0]")
            augment_df[name] = pd.Categorical(a_codes.to_numpy(dtype=np.int64), categories=levels)
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
    V55 passes the AUGMENTED fit frame, so the reading includes the appended rows and the alert is a
    statement about this run's real matrix, not the parent's."""
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


def fill_continuous_nan(X_fit, X_val, X_test, X_aug):
    """PyTabKit raises on NaN in continuous columns; V2 relied on XGBoost routing them. Medians come
    from the FIT FOLD only (no val/test statistics). Inherited from V12 unchanged.

    V55: the augment frame is filled with those SAME fit-fold medians and category vocabularies, and
    contributes none of them - letting the appended rows move the imputation centre would be a second
    change to the competition rows' feature semantics, and this arm's axis is which rows are fitted.
    X_aug is required here (V12's optional fourth frame has no caller in this arm): the function returns
    the four filled frames as a literal tuple so the arity audit can count them - a wrong unpack count on
    a threaded call is what cost the first Kaggle attempt."""
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X_fit.columns if c not in cat_cols]
    med = X_fit[num].median(numeric_only=True).astype('float32')
    out = []
    for f in (X_fit, X_val, X_test, X_aug):
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
    return out[0], out[1], out[2], out[3]


def build_fold_features(X_fit, X_val, X_test, y_fit, fold, X_aug=None):
    """Curated per-fold TE (smooth=20, inner cv=5), frequency+rarity, native categorical
    vocabularies from the fit fold. Returns (3 frames, n_added).

    V55: the augment frame joins as a TRANSFORM-ONLY peer. The target encoder is fitted on the fit
    fold's COMPETITION rows and labels only (keys[0] with y_fit), so no appended row's TE column can
    carry its own label; the frequency table and the category vocabularies are likewise the fit fold's.
    n_added is measured on the fit fold, exactly as V18 reported it."""
    frames = [X_fit.copy(), X_val.copy(), X_test.copy()]
    if X_aug is not None:
        frames.append(X_aug.copy())
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
    if X_aug is not None:
        # The appended rows are concatenated positionally with the fit fold, so the two frames must be the
        # same columns in the same order. This is the check that turns "same pipeline" into a fact.
        if list(frames[3].columns) != list(frames[0].columns):
            missing = set(frames[0].columns) - set(frames[3].columns)
            extra = set(frames[3].columns) - set(frames[0].columns)
            raise AssertionError(f"augment frame schema differs from the fit fold: missing {sorted(missing)} "
                                 f"extra {sorted(extra)}")
        return frames[0], frames[1], frames[2], frames[3], n_added
    return frames[0], frames[1], frames[2], None, n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # V18 is the parent and the recipe is NOT touched: this dict is inherited byte-for-byte, because the
    # axis is which rows are fitted, not what fits them. The epoch ladder stays V17's axis.
    EPOCH = int(REALMLP_PARAMS['n_epochs'])

    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED}")
    print(f"Axis: original rows APPENDED to each fold's training rows - augment fraction "
          f"{CFG.AUG_FRAC:.2f} of the overlap-cleaned original, split seed {CFG.AUG_SEED}; "
          f"frame, recipe, budget and validation are V18's unchanged")
    print(f"Control: the archived V18 run IS the leave-real-rows-out control (same frame, same folds, "
          f"same params) | gate vs v18_oof, offline")
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

    print("   Preparing the original (overlap removal), then the V55 R/A cut...")
    orig_clean = prepare_original(orig, train, test)

    # ---- V55 AXIS: cut the augmentation AFTER overlap removal, so an appended row is never a duplicate of a
    # competition row. prepare_original cannot be hoisted above the digit pass (that pass ROUNDS
    # Age/Flight Distance/delays, which are the hashed columns), so the split lives at this point.
    orig_R, orig_A = train_test_split(orig_clean, test_size=CFG.AUG_FRAC, random_state=CFG.AUG_SEED,
                                      stratify=orig_clean[CFG.TARGET])
    print(f"      V55 split (seed {CFG.AUG_SEED}, stratified on {CFG.TARGET}): "
          f"R={len(orig_R):,} (fit source) | A={len(orig_A):,} (appended rows)")
    if len(orig_R) + len(orig_A) != len(orig_clean):
        raise AssertionError("the R/A split is not a partition of the cleaned original")
    # Every satisfaction-derived producer for A is fitted on R, so R must not contain a copy of an A row's
    # predictor vector. prepare_original deduplicates orig_clean on ALL_FEATURES, so this is provably 0 by
    # construction - which is exactly why it is worth asserting: a nonzero reading means the dedupe was
    # bypassed and the seed's cut would let R's fits inform the very rows being appended.
    shared = pd.merge(orig_R[ALL_FEATURES].drop_duplicates(), orig_A[ALL_FEATURES].drop_duplicates(),
                      on=ALL_FEATURES, how='inner')
    print(f"      predictor vectors shared between R and A: {len(shared):,} of "
          f"{orig_A[ALL_FEATURES].drop_duplicates().shape[0]:,} distinct A vectors")
    if len(shared):
        raise AssertionError(f"{len(shared):,} A predictor vectors also sit in R, so R's fit would inform those "
                             f"rows' own features - the augmentation is not leakage-safe at this seed")

    # A joins the SAME producers as the competition frames, from here down, and each producer runs ONCE per
    # population. It is built after the digit and aggregate passes so it inherits V18's rounding exactly.
    augment = orig_A.copy()
    augment = add_digit_features(augment, DIGIT_NUMS, M)
    augment = add_aggregate_features(augment)
    augment.drop(columns=[c for c in DROP if c in augment.columns], inplace=True)

    print("   Adding wide target-free route profile (pool stays train+test; A is only transformed)...")
    train, test, augment = add_route_profile(train, test, augment)

    print("   Adding original-data org stats (competition from the original, A from R only)...")
    train, test, augment = add_org_stats(train, test, orig, augment, ref_df=orig_R)

    print("   Adding ektarr-spec teacher (competition from the cleaned original, A from R only)...")
    train, test, augment = add_original_teacher(train, test, orig_clean, augment, ref_df=orig_R)

    # ---- INHERITED FROM V18: the 13 aux-EV columns, added here - AFTER the teacher and BEFORE the float32
    # cast and the twin pass, exactly where V11 put them, so the fold driver / TE / recipe stay
    # byte-identical and the EVs can never be mistaken for twin sources.
    print("   Adding V18's inherited EV block: 13 label-free aux_ev columns (two fit sources)...")
    t0_ev = time.time()
    train, test, augment = add_aux_ev_features(train, test, orig_clean, augment, ref_df=orig_R)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    assert len(EV_COLS) == 13, f"expected 13 aux_ev columns, got {len(EV_COLS)}"
    print(f"      {ev_build_s:.0f}s EV build (two fit sources) | +{len(EV_COLS)} columns on the net frame")

    # No idempotence snapshot lives here: each producer above runs exactly once, so there is no second call to
    # compare against, and duplicating the teacher/org-stat fits just to prove it would cost a GPU fit per run.
    # The equivalence to V18's competition features is measured offline instead (check_v55_leak.py test 3:
    # same producer called with and without an A frame, on the real CSVs).

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')
            if c in augment.columns:
                augment[c] = augment[c].astype('float32')

    # ---- INHERITED FROM V12 (a): twins of the 17 numeric RAW predictors only. EV columns are not
    # in ALL_FEATURES, so they are structurally excluded from TWIN_SOURCE - no EV twin exists here.
    TWIN_SOURCE = [c for c in ALL_FEATURES if c not in CATS]
    print("   V55: dropping the A rows the competition twin vocabulary cannot express...")
    augment = filter_unrepresentable_rows(augment, train, test, TWIN_SOURCE, CFG.AUG_COVERAGE_CAP)
    train, test, twin_cols = add_numeric_categorical_twins(train, test, TWIN_SOURCE, augment)
    print(f"   V12 axis (a) inherited: +{len(twin_cols)} twins (source {len(TWIN_SOURCE)} cols) "
          f"-> frame {len(test.columns)} cols")

    # ---- INHERITED FROM V12 (b): teacher as logit.
    train = teacher_to_logit(train)
    test = teacher_to_logit(test)
    augment = teacher_to_logit(augment)
    FEATURES = list(test.columns)

    # the appended frame must be the same columns in the same order as the competition frame
    aug_y = augment[CFG.TARGET].astype(int).to_numpy()
    augment = augment.drop(columns=[CFG.TARGET])
    miss = set(test.columns) - set(augment.columns)
    extra = set(augment.columns) - set(test.columns)
    if miss or extra:
        raise AssertionError(f"A frame schema differs: missing {sorted(miss)} extra {sorted(extra)}")
    # The original's CSV is not in the competition column order and V18 only ever selected columns by name, so the
    # A frame arrives in that other order. It is concatenated POSITIONALLY with the fit fold, so the layout is
    # normalised to the competition order here - after this, the per-fold schema check tests the pipeline, not
    # the input file's column layout.
    augment = augment[list(test.columns)]
    print(f"   V55 augment frame: {augment.shape} | pos rate {aug_y.mean():.4f} "
          f"| competition train {train.shape} (pos {train[CFG.TARGET].mean():.4f})")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training RealMLP_TD (n_epochs={EPOCH}, {CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    aug_X  = augment.copy()          # the appended rows: feature-only, no target column

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    fit_times   = []
    exp_widths  = []
    frame_cols  = []
    fit_rows    = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        fold_start = time.time()
        prep_start = time.time()
        X_train, X_val, X_test, X_aug, n_added = build_fold_features(
            X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), test_X.copy(),
            y.iloc[train_idx], fold, X_aug=aug_X)
        X_train, X_val, X_test, X_aug = fill_continuous_nan(X_train, X_val, X_test, X_aug)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # THE axis: appended after the fold's own features were built, so the TE that touched these
        # rows was fitted on competition fit-fold labels only, and no validation row is ever an A row.
        X_fit = pd.concat([X_train, X_aug], ignore_index=True)
        y_fit = np.concatenate([y_train.to_numpy(), aug_y])
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        fit_rows.append((len(X_train), len(X_aug), int(y_fit.sum())))
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_fit.shape} "
              f"(competition {len(X_train):,} + augment {len(X_aug):,}) | Val: {X_val.shape} (competition only) | "
              f"Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s")
        if len(X_val) != len(val_idx):
            raise AssertionError(f"validation frame holds {len(X_val)} rows, the fold expects "
                                 f"{len(val_idx)} - OOF must stay competition-only")
        if len(X_fit) != len(X_train) + len(X_aug):
            raise AssertionError("fit frame is not competition-fold rows plus the augmentation")

        exp_width, n_num_cols, oh_map, emb_map, worst_width = estimate_expanded_matrix(X_fit)
        exp_widths.append(exp_width)
        frame_cols.append(int(X_fit.shape[1]))
        print(f"      V18 expanded-matrix ESTIMATE (absolute): {X_fit.shape[1]} frame cols = "
              f"{n_num_cols} numeric (incl. {len(EV_COLS)} EV) + "
              f"{len(CATS) + len(twin_cols)} categorical, {len(oh_map)} one-hot, "
              f"{len(emb_map)} embedded -> {exp_width} expanded cols "
              f"(all-one-hot worst case {worst_width})")
        net_memory_report(f"pre-fit fold {fold+1}", X_fit, X_test, exp_width, worst_width)

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        # X_val/y_val positional, as in our S6E2/S6E3/S6E5 RealMLP scripts: fit() without a
        # validation block silently drops ~20% of the training rows. No checkpoint reuse across folds.
        # The fit frame is X_fit = fold's competition rows + ALL of A; validation stays competition-only,
        # which is what keeps the OOF vector and every fold AUC comparable with V18's.
        t_fit = time.time()
        model.fit(X_fit, y_fit, X_val, y_val.values)
        fit_times.append(time.time() - t_fit)

        val_probs = np.asarray(model.predict_proba(X_val))[:, 1].astype('float64')
        oof_probs[val_idx] = val_probs
        test_probs += np.asarray(model.predict_proba(X_test))[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (getattr(model, 'best_epoch', None)
                     or getattr(model, 'best_epoch_', None) or EPOCH)
        best_iters.append(best_iter)

        if fold == 0:
            # The arm's ONE axis is "did these rows reach the fit?", so it gets an instrument instead of a banner
            # claim. Row counts are not readable off a fitted PyTabKit object and no attribute name is invented
            # for them here (grep: no archived arm ever read one), so the measurement is in-sample AUC: if the
            # appended rows were fitted they are memorised at about the competition fit rows' level; if they were
            # silently dropped they sit at this fold's held-out level instead. _read_held is the read-back
            # mechanism V53 proved on a real PyTabKit object.
            _held = _read_held(model, 'n_epochs')
            # The competition-fit probe is capped at the first 100k rows: same reading, and it keeps the extra
            # prediction to ~0.15 GB of the transformed matrix instead of doubling what fold 1 already allocated.
            _probe = X_train.iloc[:100_000]
            _aug_in = auc_score(aug_y, np.asarray(model.predict_proba(X_aug))[:, 1])
            _fit_in = auc_score(y_train.to_numpy()[:len(_probe)],
                                np.asarray(model.predict_proba(_probe))[:, 1])
            print(f"      fold-1 fit composition: competition {len(X_train):,} + augment {len(X_aug):,} "
                  f"= {len(X_fit):,} rows | positives {int(y_fit.sum()):,} ({y_fit.mean():.4f}) | "
                  f"validation {len(X_val):,} competition rows only")
            print(f"      fold-1 read-back: n_epochs as the object holds it = {_held} "
                  f"(REALMLP_PARAMS says {REALMLP_PARAMS['n_epochs']})")
            print(f"      fold-1 did the augmentation reach the fit? in-sample AUC - appended rows {_aug_in:.5f} "
                  f"| competition fit rows ({len(_probe):,} probe) {_fit_in:.5f} | held-out (this fold) {fold_auc:.5f}")
            print(f"      pre-declared read: near the fit-row level means the rows WERE fitted, near the "
                  f"held-out level means they were not. Originals are a different regime, so a reading in "
                  f"between is evidence to weigh on the log, not a pass/fail.")
            # leave-real-rows-out control: V18 is exactly this recipe minus the appended rows, on the same
            # frozen partition, so its archived vector - not a second in-run fit - is the comparison.
            print(f"      control declared: V18 (archived v18_oof) = same frame, same folds, "
                  f"competition rows only; dCV per fold is the augmentation effect and nothing else")

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | best epoch {best_iter} (ceiling {EPOCH}) | "
              f"fit {fit_times[-1]:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, X_aug, X_fit, y_fit, y_train, y_val, model
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
    print(f"   Axis: {CFG.AUG_FRAC:.2f} of the overlap-cleaned original ({sum(r[1] for r in fit_rows) // CFG.N_FOLDS:,} "
          f"rows) appended to each fold's competition training rows, V18's frame and recipe unchanged")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Fit rows per fold (competition + augment): {[f'{c:,}+{a:,}' for c, a, _ in fit_rows]}")
    print(f"   Admission (all four must hold): CV >= {CFG.MIN_SOLO:.5f} member strength AND dCV vs v18_oof "
          f">= +0.00005 AND >=4/5 folds improved AND paired DeLong z > 3; below {CFG.MIN_SOLO:.5f} the leg "
          f"question is already answered (2026-10-28: a leg 10 rulers under cost the blend -0.000039)")
    print(f"   CV->LB gap is NOT predicted here: family- and regime-specific, observed spread ~0.0014 "
          f"(V53 +0.00005, V54 +0.00096, RealMLP -0.00018..-0.00043) - the board score is the measurement")
    print(f"   Best epochs: {best_iters} | EV build {ev_build_s:.0f}s")
    print(f"   PyTabKit width per fold: cols handed to fit {frame_cols} | expanded estimate {exp_widths}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
