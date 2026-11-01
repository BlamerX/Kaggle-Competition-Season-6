"""
S6E10 V7 - RealMLP_TD on the proven in-episode training recipe (axis: REALMLP_PARAMS only) - GPU (cuda)
GOAL: swap ONLY the net's training RECIPE to goodpjw2008's measured in-episode config — 3 fixed epochs with
      ES off, lr 0.075->0.053, embedding 8->5, p_drop_sched->expm4t, ls_eps 0.02->0.01/sqrt_cos, new
      wd_sched='cos_log_15' + bias_init_mode='neg-uniform-dynamic-2' (n_ens stays 8); architecture, features,
      teacher, folds, float32 and the NaN median-fill stay frozen from the parent.
      Run it because V5/V6 cloned the recipe of the episode where RealMLP LOST to XGB, V6 measured that BAGGING
      rather than the class was the binding constraint (+0.000886, 5/5 folds), and the plan ranks this the
      highest-ceiling single-model bet: recipe ceiling 0.9612-0.9613 CV against V2's 0.961009.
Axis: ONE change — the REALMLP_PARAMS recipe dict. Declared confound: those values move together, so the delta
      belongs to THE RECIPE, not to any single knob; nothing else here differs from the parent.
Parent: Previously trained files/Archieve/S6E10_V6_RealMLPBagged.py (frozen, 0.96064 CV / 0.96045 LB).
Device: GPU (cuda) — all numeric matrices stay float32.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; the reproduction run swaps only
       RANDOM_SEED -> 2027 (CFG.SEED_REPRO) and nothing else.
Gate: paired DeLong on the saved OOFs, computed EXTERNALLY (no DeLong code lives here), vs BOTH v6_oof.csv and
      v2_oof.csv: ship at dAUC > +0.00005 AND z > 3 AND >= 4/5 folds improved, then reproduce at seed 2027; ship
      as control only at CV >= 0.9614. Two runs below V2's 0.961009 disprove the net and skip V12 (gated here).
Expected / Cost: +0.00026..+0.0003 over V6 => ~0.9609-0.9610 CV (published numbers sit on richer input views,
      so take the small end); ~13 min — 3 ep halves V6's ~248 s/fold fit, prep ~19 s/fold (plan estimate).
Outputs: v7_oof.csv (id, pred) + v7_sub.csv (id, satisfaction) to /kaggle/working.
Checkpoint selection (mandatory statement for net arms): PyTabKit restores the best-validation epoch and the CV
      fold IS that validation set, so the fold's own labels pick the checkpoint — not a leak, but
      schedule-dependent: ~0.00002-0.00004 for flat_anneal at 3-4 ep vs 0.000406 at 6 ep lin_cos_log_15.
Aggregation untouched: ens_av_before_softmax is NOT set anywhere — PyTabKit's default averages the bags'
      PROBABILITIES and V6 ran on it, so setting the revert flag would mix aggregation into the recipe delta.
      n_ens=8 is INTERNAL BAGGING of one model object (one fit() call), never an ensemble of two models.
Evidence: `PLAN_V7_ONWARD.md` §2 "### V7" (planning file, kept outside this repo).
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
    VERSION_NUMBER = 7
    VERSION_NAME = "V7_RealMLPRecipe"
    EXP_ID = "S6E10_V7_RealMLPRecipe"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027  # reproduction run changes only RANDOM_SEED (42 -> 2027)

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
# PyTabKit arg-name law (runbook §3): every key here is proven in the frozen V6 file or verbatim in plan
# §2/V7 — invented names cost three failed runs in V5. THE ONLY AXIS is the recipe VALUES below; renaming
# or adding a key here silently makes it a second axis.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,
    'n_epochs': 3,
    'batch_size': 256,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 3,       # inert with ES off; name kept (arg-name law)
    'early_stopping_multiplicative_patience': 1,  # inert with ES off; name kept (arg-name law)
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
    # 'ens_av_before_softmax' stays unset on purpose: probability averaging is V6's behaviour too.
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
    """Aggregates + delay arithmetic only; deterministic AND-flags stay out because
    trees recover them in <=3 splits and one hijacked 0.37 fold-1 gain pre-V1."""
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
    """Wide target-free route profile: per-FD mean, std, residual and categorical
    shares over the train+test pool (target-free, so fold-independent)."""
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
    """Original-data label means per value for every feature (raw group means,
    unseen -> orig prior) + Travel x Class x Customer triple."""
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
    """ektarr-spec teacher (XGB d8, cs0.5, 600 trees, all 21 features, native cats)
    fitted on the overlap-cleaned original only — never touches competition labels."""
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
    """Int-factorized key columns for TE + frequency (no string row-joins: they cost
    ~290s/fold in the dev run). Pair code = code_a * n_cat_b + code_b, vocabularies
    fitted on the fit fold only."""
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
    """PyTabKit raises on NaN in continuous columns (XGBoost routes them). Medians come
    from the FIT FOLD only — never val/test statistics — and everything is cast back to
    float32; this fill is frozen from V6, not part of this axis."""
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
            # an MLP cannot route NaN: off-vocabulary levels (~0.02% of cells) go to the first level
            f[c] = f[c].astype('object').fillna(cats[0]).astype(dt)

    n_added = len(frames[0].columns) - len(X_fit.columns)
    return frames[0], frames[1], frames[2], n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis: REALMLP_PARAMS recipe only | Parent V6: 0.96064 CV / 0.96045 LB (v6_oof.csv control), "
          f"original data = statistics + teacher (no row concat)")

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

    print(f"   Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape}")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f} | (orig): {orig[CFG.TARGET].mean():.4f} "
          f"| Missing in train: {int(train.isna().sum().sum())} (PyTabKit needs them filled, see fill step)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V6 frame: digits, aggregates, route profile, "
          "org stats, teacher)...")

    DIGIT_NUMS = ['Age', 'Flight Distance',
                  'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    M = train[DIGIT_NUMS].max()
    train = add_digit_features(train, DIGIT_NUMS, M)
    test  = add_digit_features(test,  DIGIT_NUMS, M)

    train = add_aggregate_features(train)
    test  = add_aggregate_features(test)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    train, test = add_route_profile(train, test)
    train, test = add_org_stats(train, test, orig)
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
    print(f"\n[3/5] Training RealMLP_TD ({CFG.N_FOLDS}-Fold CV, device=cuda)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    fit_times   = []   # print-support only: per-fold fit seconds for the RESULTS block

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    t0 = time.time()
    # per-fold loop, NOT PyTabKit's vectorized CV: that would rebuild the fold driver = a second axis
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        fold_start = time.time()

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
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s")

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        # X_val/y_val are REQUIRED: fit() without a validation block silently drops ~20% of the
        # training rows (measured +0.00061 when passed). Epoch-honest: no checkpoint reuse across folds.
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
            # a net has no feature_importances_: log the device it really trained on
            print(f"      Device: "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
                  f"| cols {X_train.shape[1]} | cat cols {len(CATS)} "
                  f"| best epoch {best_iter}")
            print(f"      NaN filled (fit-fold medians): {nan_before} -> 0")

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

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred)")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best epochs (budget {REALMLP_PARAMS['n_epochs']}): {best_iters}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s/fold (mean) "
          f"| total {total_time_min:.1f} min")
