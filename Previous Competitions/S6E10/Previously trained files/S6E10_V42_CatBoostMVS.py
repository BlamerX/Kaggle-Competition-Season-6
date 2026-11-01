"""
S6E10 V42 - V4's CatBoost with Minimum Variance Sampling instead of Bernoulli sampling (one scheme) - GPU (cuda)
GOAL: our own record says MVS is closed. It is not: V35 ran XGBoost `sampling_method='gradient_based'`, which is
      GOSS - a gradient-magnitude-biased sampling scheme - while the published +0.0012 (Ibragimov et al., KDD 2019)
      describes CatBoost's `bootstrap_type='MVS'`, a variance-minimising gradient estimator. The two share a name in
      our ledger and nothing else, so the claim the ledger rests on has never been tested. This arm tests it.
Axis: ONE key, `bootstrap_type` Bernoulli -> MVS, with the parent's `subsample=0.8` left exactly where V4 put it -
      subsample is the rate under MVS too, so the amount of data per tree IS held and only the estimator that picks it
      changes. Two parameter inventions died on this build and cost the runs: a float in `sampling_frequency`
      (constructor accepted it, core rejected at fit) and then the enum value `SqrtDegree`, which the core's own error
      says does not exist - `ESamplingFrequency` is 'PerTree' | 'PerTreeLevel'. So `sampling_frequency` is not set at
      all and the parent's default is inherited: naming a value from documentation memory is a second axis. Everything
      else is frozen: frame, depth, lr, iterations, loss, eval metric, task_type, fold protocol, seed.
Read-back, AFTER the fit, because construction accepts anything: fold 1 fits first, then prints the FITTED object's
      `bootstrap_type` / `sampling_frequency` / `subsample` and RAISES unless it kept `MVS` at rate 0.8; it also re-fits
      the parent's Bernoulli on the same matrix and prints the paired AUCs, so the reading has a same-machine control
      beside the parent's recorded run. An unobservable axis is how V37 was priced against someone else's config.
Metric trap avoided deliberately: `eval_metric` stays `Logloss`. V16 measured that `AUC` on this GPU build is
      computed on the CPU (15 fits = 63 min) and bought +0.00001, so fold AUC is computed offline here as the
      parent does.
Parent: Previously trained files/Archieve/S6E10_V4_CatBoostMember.py - the tree-side member whose CV sits below the
      control; this arm is about the sampling mechanism, not about becoming the shipped model.
Device: GPU (cuda), CatBoost task_type=GPU, devices=0; recipe byte-identical apart from the axis.
Standalone (owner law 2026-10-11): one fitted model, no previous version's saved OOF, nothing blended.
Gate: paired DeLong vs the parent's OOF: dAUC >= +0.0001 AND z > 3 AND >= 4/5 folds improved, then reproduce at
      SEED_REPRO=2027. The ruler is +-0.00006 CV, so a smaller delta is a tie about the lever.
Expected / Cost: the published +0.0012 was measured on sparse-gradient, large-iteration regimes; on a 700k frame at
      depth 8 and 6000 iterations with early stopping it may well be inert, and a tie closes MVS properly instead of
      by re-labelling GOSS. The parent's cost plus one extra fold-1 fit for the in-run control - a few minutes on a T4.
Outputs: v42_oof.csv (id, pred) + v42_sub.csv (id, satisfaction) to /kaggle/working - exactly two writes.
Evidence: PLAN_V36_ONWARD.md §10c (the mislabel this arm repairs) and §10b (where the candidate came from);
      Memory/training_logs.md V35 (the GOSS result that was filed under the wrong name).
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
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 42
    VERSION_NAME = "V42_CatBoostMVS"
    EXP_ID = "S6E10_V42_CatBoostMVS"
    PARENT = "V4_CatBoostMember"
    PARENT_SUBSAMPLE = 0.8      # the parent's Bernoulli rate, kept only so the fold-1 control can be rebuilt here

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
# CatBoost main model (V4); XGB no longer used for the competition model
CB_PARAMS = dict(
    loss_function='Logloss',
    eval_metric='Logloss',  # AUC not implemented for GPU on this build; fold AUC is computed offline
    iterations=6000, learning_rate=0.05, depth=8,
    l2_leaf_reg=5.0, random_strength=1.0,
    # THE AXIS: one key, Bernoulli -> MVS. The parent's `subsample=0.8` stays exactly where V4 put it, because
    # subsample is the rate under MVS too - so the amount of data per tree IS held and the only thing that changes is
    # the estimator that chooses it. `sampling_frequency` is left unset on purpose: the build's own error names its
    # enum as 'PerTree' | 'PerTreeLevel' (the parent runs on the default), so writing a value here would be a second
    # axis invented from documentation, which is what cost the last two runs.
    bootstrap_type='MVS', subsample=CFG.PARENT_SUBSAMPLE,
    task_type='GPU', devices='0',
    early_stopping_rounds=200, random_seed=CFG.RANDOM_SEED,
    verbose=0, allow_writing_files=False,
)

# Guard, in microseconds and before a row is read, against the two defects this build has now reported at fit time:
# a float where ESamplingFrequency is an enum, and an enum value the build does not define. Valid set is copied FROM
# THE ERROR THE BUILD PRINTED, not from memory ('SqrtDegree'/'GradientDegree' are not in it).
_KNOWN_SAMPLING = ('PerTree', 'PerTreeLevel')
if 'sampling_frequency' in CB_PARAMS and CB_PARAMS['sampling_frequency'] not in _KNOWN_SAMPLING:
    raise TypeError(f"sampling_frequency must be one of {_KNOWN_SAMPLING} or left unset, not "
                    f"{CB_PARAMS['sampling_frequency']!r}.")
if not (isinstance(CB_PARAMS['subsample'], (int, float)) and 0.0 < CB_PARAMS['subsample'] <= 1.0):
    raise TypeError(f"subsample is the rate MVS reads and must be a number in (0, 1], not "
                    f"{CB_PARAMS['subsample']!r}.")

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

    n_added = len(frames[0].columns) - len(X_fit.columns)
    return frames[0], frames[1], frames[2], n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | {CFG.EXP_ID}")
    print(f"Device: GPU (CatBoost GPU + cuda teacher) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: statistics + ektarr-spec teacher (no row concat)")
    print("Recipe: light params, curated TE/freq/rarity, native cats,")
    print("int-coded pair keys, wide route profile, no weights, no AND-flags")
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
    print(f"\n[3/5] Training CatBoost ({CFG.N_FOLDS}-Fold CV)...")

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
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        print(f"      Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Prep: {prep_s:.0f}s")

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  cat_features=CATS, use_best_model=True)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = model.get_best_iteration()
        best_iters.append(best_iter)

        if fold == 0:
            # The axis must be shown to be ACTIVE, and the proof belongs AFTER the fit: construction accepts any
            # type at all - the first version of this arm printed `sampling_frequency=0.8 subsample=None` happily
            # and only the C++ core objected at fit time - so the fitted object's own params are the read-back.
            _gp = model.get_params()
            _bt, _sf, _ss = (_gp.get('bootstrap_type'), _gp.get('sampling_frequency'),
                             _gp.get('subsample'))
            print(f'      READ-BACK after the fit: bootstrap_type={_bt!r} '
                  f'sampling_frequency={_sf!r} subsample={_ss!r}')
            if _bt != 'MVS':
                raise RuntimeError('CatBoost did not keep bootstrap_type=MVS - the axis is inert, so the run '
                                   'would measure something else and print a plausible AUC doing it.')
            if _ss is None or abs(float(_ss) - CFG.PARENT_SUBSAMPLE) > 1e-9:
                raise RuntimeError(f'the axis was supposed to hold the rate at {CFG.PARENT_SUBSAMPLE!r}; the fitted '
                                   f'object reports subsample={_ss!r} - something else moved, so the run would not '
                                   f'be a single-axis comparison.')
            # In-run control on the same fold, frame and machine state: the parent's Bernoulli scheme, fitted once,
            # so the MVS reading has a paired number beside it rather than only the parent's recorded run.
            _ctrl_p = dict(CB_PARAMS)
            _ctrl_p['bootstrap_type'] = 'Bernoulli'
            _ctrl_p['subsample'] = CFG.PARENT_SUBSAMPLE
            _ctrl_p.pop('sampling_frequency', None)
            _ctrl = CatBoostClassifier(**_ctrl_p)
            _ctrl.fit(X_train, y_train, eval_set=(X_val, y_val),
                      cat_features=CATS, use_best_model=True)
            _ctrl_auc = auc_score(y_val.values, _ctrl.predict_proba(X_val)[:, 1])
            # get_best_iteration is the call the parent already makes on this build; `len(model)` is not supported
            # by this build's wrapper, and a diagnostic print must never be able to kill a fold that trained fine.
            print(f'      IN-RUN CONTROL fold 1, parent scheme Bernoulli(subsample={CFG.PARENT_SUBSAMPLE}): '
                  f'AUC {_ctrl_auc:.5f} | MVS AUC {fold_auc:.5f} | paired d {fold_auc - _ctrl_auc:+.6f} | '
                  f'iters used {_ctrl.get_best_iteration()} vs {best_iter}')
            del _ctrl
            gc.collect()

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s (train {fold_time-prep_s:.0f}s) | Total: {elapsed:.1f}min")

        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.get_feature_importance(),
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))
            print()

        del X_train, X_val, X_test, y_train, y_val, model
        gc.collect()

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
    print(f"{CFG.VERSION_NAME} RESULTS — CatBoost Member (GPU)")
    print(f"{'='*80}")
    print(f"Base features: {len(FEATURES)} + per-fold TE/freq/rarity additions")
    print(f"Control: V2 0.96101 CV / 0.96051 LB (DeLong z=+7.44) — v2_oof.csv")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
