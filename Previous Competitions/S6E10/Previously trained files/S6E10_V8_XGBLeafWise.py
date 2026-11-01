"""
S6E10 V8 - Leaf-wise XGBoost (growth-policy axis; parent V2) - GPU (cuda)
GOAL: ONE axis vs the frozen control V2 — XGBoost's growth policy. V2 grows depth-wise (no
      grow_policy key, max_depth=6); V8 sets grow_policy='lossguide' with max_depth=0 (leaf-wise
      forces the depth cap off, so max_leaves alone bounds tree shape) over the pre-declared
      max_leaves ladder 8 -> 16 -> 32. Run it because S6E5's V2->V7 leaf-wise step was +0.00089
      LB, the largest single-model move we have ever measured.
Declared confound: that S6E5 step bundled TE RowStats (now V9), so the isolated leaf-wise gain is unknown.
Axis: XGB_PARAMS only — +grow_policy='lossguide', max_depth 6->0, +max_leaves per rung (8/16/32);
      FE, folds, teacher, sampling, regularisation, tree_method='hist', max_bin=256 all stay V2's.
Parent: Previously trained files/Archieve/S6E10_V2_XGB_WideProfile.py (frozen, 0.961009 CV / 0.96051 LB).
Device: GPU (cuda)
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = RANDOM_SEED=2027
      (CFG.SEED_REPRO) as the only config delta.
Gate: paired DeLong vs v2_oof.csv (computed externally, no DeLong lives in this script):
      dAUC > +0.00005 AND z > 3 AND >=4/5 folds improved, then reproduce at RANDOM_SEED=2027;
      DeLong-only is a TIE, not an improvement.
Expected / Cost: 0 ... +0.0005 CV, ~4-6 min (plan estimate, not a measurement).
Ladder = coordinate search over one axis, not an ensemble: one fitted config per rung (one
      XGBClassifier per fold, fold-averaged test preds like V2); rungs are never combined.
Outputs: v8_oof.csv (id, pred) + v8_sub.csv (id, satisfaction) for the provisional rung of record
      (highest OOF CV; authoritative selection happens in the external gate), plus the per-rung
      audit files v8_ladder_<leaves>_oof.csv, to /kaggle/working.
Evidence behind this arm: `PLAN_V7_ONWARD.md` §2 "### V8" (planning file, kept outside this repo).
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

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

print(f"xgboost version: {xgb.__version__}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 8
    VERSION_NAME = "V8_XGBLeafWise"
    EXP_ID = "S6E10_V8_XGBLeafWise"

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
# THE AXIS vs V2: +grow_policy='lossguide', max_depth 6 -> 0, +max_leaves per rung (LEAF_LADDER).
# Any other key changed in this dict re-tunes beyond what leaf-wise forces = a second axis.
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'device': 'cuda',
    'n_estimators': 4000,
    'learning_rate': 0.04,
    'grow_policy': 'lossguide',
    'max_depth': 0,
    'min_child_weight': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_lambda': 5.0,
    'reg_alpha': 0.05,
    'max_bin': 256,
    'early_stopping_rounds': 150,
    'n_jobs': -1,
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
}

# Coordinate ladder over the ONE axis (plan §2/V8: 8 -> 16 -> 32); rungs never combined.
LEAF_LADDER = [8, 16, 32]

# Anti-silent-retune contract: the only place a retune could enter — printed pre-fit and fenced here.
LADDER_RETUNE = {
    8:  {},
    16: {},
    32: {},
}
assert set(LADDER_RETUNE) == set(LEAF_LADDER), "ladder/retune keys out of sync"
assert not any(set(r) - {'min_child_weight', 'reg_lambda', 'gamma',
                        'learning_rate', 'max_leaves'}
               for r in LADDER_RETUNE.values()), \
    "a rung retunes outside the four keys leaf-wise is allowed to force"

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
    unseen -> orig prior) + Travel x Class x Customer triple. Uses ONLY original
    labels, never competition labels."""
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
    """ektarr-spec teacher (XGB d8, cs0.5, 600 trees, all 21 features, native cats) fitted on
    the overlap-cleaned original only — never competition labels; the teacher stays XGBoost."""
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
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis vs V2: grow_policy='lossguide', max_depth 6->0, max_leaves ladder {LEAF_LADDER} "
          f"(one fitted config per rung, rungs never combined) | Parent V2: 0.961009 CV / "
          f"0.96051 LB, control OOF v2_oof.csv")

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
          f"| Missing in train: {int(train.isna().sum().sum())} (XGBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V2 recipe: digits, aggregates, route profile, "
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
    # [3/5] TRAINING (frozen 5-Fold CV, per-fold TE + native cats, leaf ladder)
    # =========================================================================
    print(f"\n[3/5] Training leaf-wise XGB ladder {LEAF_LADDER} "
          f"({CFG.N_FOLDS}-Fold CV, folds identical to V2)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    rung_oof         = {L: np.zeros(len(y)) for L in LEAF_LADDER}
    rung_test        = {L: np.zeros(len(test_X)) for L in LEAF_LADDER}
    rung_fold_scores = {L: [] for L in LEAF_LADDER}
    rung_best_iters  = {L: [] for L in LEAF_LADDER}
    rung_train_times = {L: [] for L in LEAF_LADDER}
    prep_times  = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    print(f"\n   Pre-declared per-rung retune table: {LADDER_RETUNE} "
          f"(empty = V2 baseline; max_leaves is the only key a rung moves)")
    print(f"      shared effective params: mcw={XGB_PARAMS['min_child_weight']} "
          f"reg_lambda={XGB_PARAMS['reg_lambda']} reg_alpha={XGB_PARAMS['reg_alpha']} "
          f"gamma=0(unset) lr={XGB_PARAMS['learning_rate']} "
          f"n_estimators={XGB_PARAMS['n_estimators']} "
          f"es={XGB_PARAMS['early_stopping_rounds']} max_bin={XGB_PARAMS['max_bin']} "
          f"tree_method={XGB_PARAMS['tree_method']} device={XGB_PARAMS['device']} | "
          f"max_leaves={LEAF_LADDER}")

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        fold_start = time.time()

        prep_start = time.time()
        X_train, X_val, X_test, n_added = build_fold_features(
            X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), test_X.copy(),
            y.iloc[train_idx], fold)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | "
              f"Val: {X_val.shape} | Prep: {prep_s:.0f}s (shared by all rungs)")

        for rung_i, leaves in enumerate(LEAF_LADDER):
            rung_params = dict(XGB_PARAMS)
            rung_params.update(LADDER_RETUNE[leaves])
            rung_params['max_leaves'] = leaves

            # One-axis guard: only max_leaves and the pre-declared retune table may move a key.
            changed = {k: (XGB_PARAMS[k], v) for k, v in rung_params.items()
                       if k in XGB_PARAMS and v != XGB_PARAMS[k]
                       and k not in LADDER_RETUNE[leaves] and k != 'max_leaves'}
            added = (set(rung_params) - set(XGB_PARAMS) - set(LADDER_RETUNE[leaves])
                     - {'max_leaves'})
            assert not changed and not added, (
                f"rung leaves={leaves} moved keys outside the declared axis: "
                f"changed={changed} added={added}")

            fit_start = time.time()
            model = xgb.XGBClassifier(**rung_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            fit_s = time.time() - fit_start
            rung_train_times[leaves].append(fit_s)

            val_probs = model.predict_proba(X_val)[:, 1]
            rung_oof[leaves][val_idx] = val_probs
            rung_test[leaves] += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

            fold_auc = auc_score(y_val.values, val_probs)
            rung_fold_scores[leaves].append(fold_auc)
            best_iter = (model.best_iteration if hasattr(model, 'best_iteration')
                         and model.best_iteration is not None else model.n_estimators)
            rung_best_iters[leaves].append(best_iter)

            print(f"      leaves={leaves}: AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
                  f"Train: {fit_s:.0f}s | Elapsed: {(time.time() - t0)/60:.1f}min")

            if fold == 0 and rung_i == 0:
                imp = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_,
                }).sort_values('importance', ascending=False)
                print(f"      Top-15 feature importances (fold 1, leaves={leaves}):")
                print(imp.head(15).to_string(index=False))

            del model
            gc.collect()

        del X_train, X_val, X_test, y_train, y_val
        gc.collect()

    rung_cv = {L: auc_score(y.values, rung_oof[L]) for L in LEAF_LADDER}
    best_leaves = max(LEAF_LADDER, key=lambda L: rung_cv[L])
    oof_cv = rung_cv[best_leaves]

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print("\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    for L in LEAF_LADDER:
        ladder_df = pd.DataFrame({'id': train_id, 'pred': rung_oof[L]})
        ladder_df.to_csv(f"{out_dir}/{stem}_ladder_{L}_oof.csv", index=False)
        print(f"   [SAVED] {out_dir}/{stem}_ladder_{L}_oof.csv (id, pred) — per-rung audit trail")

    oof_df = pd.DataFrame({'id': train_id, 'pred': rung_oof[best_leaves]})
    oof_df.to_csv(f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred) — provisional rung max_leaves={best_leaves}")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: rung_test[best_leaves]})
    sub_df.to_csv(f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction) — same provisional rung")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(rung_fold_scores[best_leaves]):.5f} +/- {np.std(rung_fold_scores[best_leaves]):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in rung_fold_scores[best_leaves]]}")
    print(f"   Best iters: {rung_best_iters[best_leaves]}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s/fold (mean) | total {total_time_min:.1f} min")
    for L in LEAF_LADDER:
        print(f"   Rung max_leaves={L}: OOF CV {rung_cv[L]:.5f}")
