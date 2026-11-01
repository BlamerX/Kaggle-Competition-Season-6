"""
S6E10 V20 - XGBoost on soft targets from an IN-RUN nested teacher (target axis) - GPU (cuda)
GOAL: attack the ~3.8% symmetric label flips at the one place they enter the loss - the target. Published
      single-model gains for soft targets are +0.00012 (LightGBM) / +0.00019 (CatBoost).
Axis: ONE change - the target. p = (1-ALPHA)*y + ALPHA*p_teacher, ALPHA = 0.20, where p_teacher is an XGB
      cross-fitted INSIDE the outer training fold only (5 inner folds; validation rows never fitted and
      never used to fit). Everything else is byte-identical to V11: XGB_PARAMS (lr 0.04, depth 6, 4000
      estimators, es 150), the 13 aux-EV columns and their builder, the TE/freq/rarity/route/org/teacher
      package, the frozen folds and the seed. PLAN §V20 said p_teacher = v12_oof; a saved OOF as a
      training input is banned by the owner's standalone law, so the teacher is recomputed in-run.
Mechanism: the student trains through xgb.DMatrix(label=p) + xgb.train, not the sklearn wrapper: this
      build infers classes from y's unique values and refuses a float target, and binary:logistic reads a
      float label as a probability natively. booster_params() drops only the three wrapper-only keys -
      v3.2.0 forwards reg_lambda / reg_alpha / random_state / n_jobs untranslated, so no key is renamed
      and none can be silently dropped. Deviation from V11: the calling API, not the config. The import
      probe falsifies that reading (functional vs wrapper on hard labels) and that predict() returns
      probabilities, not margins.
Safeguards: TRAIN SOFT / EVALUATE HARD - `auc` needs binary labels, so the eval DMatrix carries the
      untouched y_val and ES, fold AUC and OOF all score the real target. ANTI-COLLAPSE FALSIFIER -
      per-fold Spearman(student, teacher) > 0.990 means the student is a smoothed copy: REJECTED at any
      CV. The teacher never sees a validation row and p is asserted inside [0,1].
Parent: Previously trained files/Archieve/S6E10_V11_AuxEVFeatures.py (0.96113 CV / 0.96064 LB).
Device: GPU (cuda) - XGBoost 3.2.0 hist on a Kaggle T4; the ektarr teacher and EV predictors on cuda.
Standalone (owner law 2026-10-11): no previous version's OOF is a feature or a target here; v11_oof and
      v2_oof enter only OFFLINE, after the run, for the gate.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = SEED_REPRO=2027.
Gate: paired DeLong vs v11_oof AND v2_oof: dAUC > +0.00005 AND z > 3 AND >=4/5 folds improved, then
      reproduce at 2027. Tie band 0.0003 - decide on CV, audit on LB.
Expected / Cost: +0.00008 ... +0.00012 CV over V11 (5.8 min); the teacher adds 25 inner fits, so
      ~18-20 min on one T4.
Outputs: v20_oof.csv (id, pred) + v20_sub.csv (id, satisfaction) to /kaggle/working.
Evidence: PLAN_V16_ONWARD.md §3 "Arm V20"; Memory/feature_engineering.md (V11 row).
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

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"xgboost version: {xgb.__version__}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 20
    VERSION_NAME = "V20_SoftLabelDistill"
    EXP_ID = "S6E10_V20_SoftLabelDistill"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027        # reproduction run changes only RANDOM_SEED (42 -> 2027)

    # Aux-EV builder switch inherited from V11 (True = original-fitted predictors).
    AUX_EV_FIT_ON_ORIGINAL = True

# THIS AXIS: the mixing weight, the teacher's inner fold count and its seed offset.
ALPHA = 0.20
TEACHER_INNER_FOLDS = 5
TEACHER_SEED_BASE = 77       # deterministic; validation rows never enter these splits
COLLAPSE_LIMIT = 0.990       # falsifier: Spearman(student, teacher) above this = no diversity

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
# The student: byte-identical to V11's XGB_PARAMS. eval_metric='auc' stays valid because the
# early-stopping eval DMatrix always carries HARD binary labels (see the fit site).
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'device': 'cuda',
    'n_estimators': 4000,
    'learning_rate': 0.04,
    'max_depth': 6,
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

# The in-run teacher: deliberately a weaker, differently-regularised XGB (depth 4, 600 trees),
# so it is not the student's own configuration predicting itself. Specified here as part of the
# target-vector axis, not tuned further.
TEACHER_SOFT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'enable_categorical': True,
    'device': 'cuda',
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_lambda': 5.0,
    'reg_alpha': 0.05,
    'n_jobs': -1,
    'random_state': 1,
    'verbosity': 0,
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

# Inherited from V11 unchanged: the 13 EV predictors.
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


def spearman(a, b):
    """Rank correlation without a scipy dependency: Pearson on the ranks."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

# =============================================================================
# 6. FEATURE ENGINEERING (fold-independent part) - byte-identical to V11
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
    """Original-data label means per value for every feature (ektarr original_means
    protocol: raw group means, unseen -> orig prior) + Travel x Class x Customer triple.
    V2's unshrunk estimator is kept deliberately - the shrinkage is V13's axis, not this one."""
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
# 6b. INHERITED FROM V11 - THE 13 LABEL-FREE AUX-EV COLUMNS
# =============================================================================
def feature_name(column):
    return column.lower().replace(' ', '_').replace('/', '_').replace('-', '_')


def add_aux_ev_features(train_df, test_df, orig_clean):
    """
    V11's axis, inherited unchanged: 13 `aux_ev_<rating>` columns, EV = sum_k k * P_hat(rating=k),
    each P_hat from an XGB predicting that rating from the OTHER 20 ALL_FEATURES columns (never
    `satisfaction`). Label-free feature producers, not a second competing model: only the student
    predicts satisfaction, and nothing is blended at inference.

    LEAKAGE GUARD: fitted ONCE globally, never per outer fold, and DEFAULT (run-of-record) fits only
    on the overlap-cleaned original, so no competition row is in any fit set.
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
# 7. PER-FOLD FEATURE BUILDER - byte-identical to V11
# =============================================================================
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


def soft_targets(X_fit, y_fit, X_val, fold):
    """THIS VERSION'S AXIS. Cross-fits a weaker XGB inside the outer TRAINING fold only and returns
    y_soft = (1-ALPHA)*y + ALPHA*p_teacher, the teacher's OOF vector on the training rows, and its mean
    prediction on the VALIDATION rows. LEAKAGE GUARD: every training row's p_teacher is out-of-fold
    within those rows, and the validation rows are not in X_fit/y_fit at all, so no inner model ever
    sees one - the val prediction feeds only the collapse diagnostic, never a target or a feature."""
    p_teacher = np.zeros(len(y_fit), dtype='float32')
    val_preds = []
    inner = StratifiedKFold(n_splits=TEACHER_INNER_FOLDS, shuffle=True,
                            random_state=TEACHER_SEED_BASE + fold)
    y_np = y_fit.to_numpy(dtype='int64')
    for fit_idx, hold_idx in inner.split(X_fit, y_np):
        m = xgb.XGBClassifier(**TEACHER_SOFT_PARAMS)
        m.fit(X_fit.iloc[fit_idx], y_np[fit_idx], verbose=False)
        p_teacher[hold_idx] = m.predict_proba(X_fit.iloc[hold_idx])[:, 1].astype('float32')
        val_preds.append(m.predict_proba(X_val)[:, 1].astype('float32'))
        del m
        gc.collect()

    p_teacher_val = np.mean(val_preds, axis=0).astype('float32')
    y_soft = ((1.0 - ALPHA) * y_np.astype('float32') + ALPHA * p_teacher).astype('float32')
    if float(y_soft.min()) < 0.0 or float(y_soft.max()) > 1.0:
        raise ValueError(f"soft target left [0,1]: [{y_soft.min():.4f}, {y_soft.max():.4f}]")
    return y_soft, p_teacher, p_teacher_val


# Wrapper-only keys: n_estimators and early_stopping_rounds become xgb.train arguments and
# enable_categorical belongs to the DMatrix. Every other key is forwarded verbatim, exactly as
# XGBModel.get_xgb_params() does it (v3.2.0 source), so no name is hand-mapped and none can drift.
WRAPPER_ONLY = ('n_estimators', 'early_stopping_rounds', 'enable_categorical')


def booster_params(params):
    """XGB_PARAMS -> the params dict xgb.train takes: wrapper-only keys out, everything else verbatim."""
    return {k: v for k, v in params.items() if k not in WRAPPER_ONLY}


def soft_dmatrix(X, label=None):
    """DMatrix carrier - binary:logistic reads a float label in [0,1] as a probability, and unlike the
    sklearn wrapper this entry point never infers classes from y."""
    return xgb.DMatrix(X, label=label, enable_categorical=True)


def best_predict(bst, dm):
    """Early-stopped predict. The sklearn wrapper slices to best_iteration internally; a raw Booster
    does NOT - predict's default iteration_range=(0,0) uses EVERY trained round, and xgb.train keeps
    training for the full patience window after the best round. Without this slice V20 would be scored
    on ~150 trees more than V11's early-stopped model, i.e. a second axis. best_iteration is 0-based,
    so the inclusive range is (0, best_iteration + 1)."""
    bi = getattr(bst, 'best_iteration', None)
    return bst.predict(dm, iteration_range=(0, int(bi) + 1)) if bi is not None else bst.predict(dm)

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis: training target only - p = (1-{ALPHA})*y + {ALPHA}*p_in_run_nested_teacher, carried as "
          f"a DMatrix label through xgb.train | evaluate and early-stop on HARD y_val")
    print(f"Parent V11: 0.96113 CV / 0.96064 LB | Control V2: 0.961009 CV / 0.96051 LB "
          f"| gate vs v11_oof AND v2_oof, offline | no saved OOF is a model input")
    print("="*80)

    # ---- Soft-target pre-flight (plan rule 4 - prove the mechanism on the installed build, never on
    # documentation memory). Two claims, both fatal to this arm, on 500 synthetic rows before any prep
    # minute is spent: (1) EQUIVALENCE - the functional path must reproduce the wrapper path V11 ran.
    # reg_lambda=500 with the bootstrap off makes the case hostile to a silently dropped key: an
    # unregularised booster splits differently, so the predictions separate far past the 1e-3 abort.
    # (2) THE AXIS - a float target must train, a HARD-labelled eval DMatrix must drive `auc` early
    # stopping, and predict() must return probabilities inside [0,1] (margins would void every AUC).
    _pf = pd.DataFrame({'a': np.random.RandomState(0).rand(500).astype('float32'),
                        'b': np.random.RandomState(1).rand(500).astype('float32')})
    _py = (_pf['a'] > 0.5).astype('int64').to_numpy()
    _EQ = {**XGB_PARAMS, 'max_depth': 3, 'subsample': 1.0, 'colsample_bytree': 1.0,
           'reg_lambda': 500.0, 'n_estimators': 30, 'early_stopping_rounds': None}
    _wm = xgb.XGBClassifier(**_EQ)
    _wm.fit(_pf, _py, verbose=False)                       # V11's own path, hard labels
    _wp = _wm.predict_proba(_pf)[:, 1]
    try:
        _bst = xgb.train(booster_params(_EQ), soft_dmatrix(_pf, _py), num_boost_round=30)
    except Exception as e:
        raise ImportError(f"xgb.train rejected this params dict on xgboost {xgb.__version__} "
                          f"({type(e).__name__}: {e}) - fix the params, do not re-tune the recipe.")
    _dev = float(np.max(np.abs(best_predict(_bst, soft_dmatrix(_pf)) - _wp)))
    print(f"xgboost {xgb.__version__} wrapper-vs-functional equivalence at reg_lambda=500: "
          f"max |p_functional - p_wrapper| = {_dev:.6f} | rounds {_bst.num_boosted_rounds()}")
    if _dev > 1e-3:
        raise ImportError("the functional path does NOT reproduce V11's wrapper recipe (max deviation "
                          f"{_dev:.6f}) - a parameter is being ignored, so this arm would train a "
                          "different model than its parent. Fix the params; do not widen the tolerance.")

    _pt = np.clip(_py * (1.0 - ALPHA) + ALPHA, 0.0, 1.0).astype('float32')      # genuinely SOFT target
    _sf = xgb.train(booster_params(XGB_PARAMS), soft_dmatrix(_pf, _pt), num_boost_round=120,
                    evals=[(soft_dmatrix(_pf, _py), 'val')],                    # HARD labels for auc
                    early_stopping_rounds=XGB_PARAMS['early_stopping_rounds'], verbose_eval=False)
    _sp = best_predict(_sf, soft_dmatrix(_pf))
    if float(_sp.min()) < 0.0 or float(_sp.max()) > 1.0:
        raise ImportError(f"soft-label predict() left [0,1]: [{_sp.min():.4f}, {_sp.max():.4f}]")
    print(f"soft-target train + hard-label auc early stopping: OK | best iter "
          f"{getattr(_sf, 'best_iteration', 'n/a')} | p in [{float(_sp.min()):.4f}, "
          f"{float(_sp.max()):.4f}] | probe auc {auc_score(_py, _sp):.5f}")
    del _wm, _bst, _sf, _pf, _py, _pt, _wp, _sp
    gc.collect()

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
    print("\n[2/5] Feature Engineering (V11 recipe byte-equal)...")

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

    t0_ev = time.time()
    train, test = add_aux_ev_features(train, test, orig_clean)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    print(f"   V11 axis inherited: {ev_build_s:.0f}s EV build | +{len(EV_COLS)} aux_ev columns")

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    FEATURES = list(test.columns)
    print(f"   Total base features: {len(FEATURES)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost on soft targets ({CFG.N_FOLDS}-Fold CV, alpha={ALPHA})...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    train_times = []
    rho_fold    = []
    teacher_auc = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

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

        # THE AXIS: soft targets from a nested in-run teacher (training rows only). The soft vector is
        # the DMatrix label, and the eval DMatrix carries the untouched HARD y_val - so `auc`, early
        # stopping, the fold AUC and the saved OOF all score the real target, as V11's eval_set did.
        t_soft = time.time()
        y_soft, p_teacher_tr, p_teacher_val = soft_targets(X_train, y_train, X_val, fold)
        soft_s = time.time() - t_soft

        dtrain = soft_dmatrix(X_train, y_soft)
        dval   = soft_dmatrix(X_val, y_val)
        dtest  = soft_dmatrix(X_test)
        booster = xgb.train(booster_params(XGB_PARAMS), dtrain,
                            num_boost_round=XGB_PARAMS['n_estimators'],
                            evals=[(dval, 'val')],
                            early_stopping_rounds=XGB_PARAMS['early_stopping_rounds'],
                            verbose_eval=False)

        val_probs = np.asarray(best_predict(booster, dval), dtype='float64')
        oof_probs[val_idx] = val_probs
        test_probs += np.asarray(best_predict(booster, dtest), dtype='float64') / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (booster.best_iteration
                     if getattr(booster, 'best_iteration', None) is not None
                     else XGB_PARAMS['n_estimators'])
        best_iters.append(best_iter)

        # Collapse falsifier, measured on this fold's VALIDATION rows: both vectors are
        # out-of-sample there (no inner teacher model ever saw a validation row), so rho measures
        # imitation rather than leakage.
        rho = spearman(val_probs, p_teacher_val)
        rho_fold.append(rho)
        teacher_auc.append(auc_score(y_val.values, p_teacher_val))

        fold_time = time.time() - fold_start
        train_times.append(fold_time - prep_s)
        elapsed   = (time.time() - t0) / 60
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"cols added {n_added} | Prep {prep_s:.0f}s | teacher xfit {soft_s:.0f}s")
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"y_soft mean {float(np.mean(y_soft)):.4f} vs hard {float(np.mean(y_train.to_numpy())):.4f} "
              f"| teacher train-OOF AUC {auc_score(y_train.to_numpy(), p_teacher_tr):.5f} "
              f"| Spearman(student, teacher) {rho:.4f} "
              f"{'<-- COLLAPSE (> ' + str(COLLAPSE_LIMIT) + ')' if rho > COLLAPSE_LIMIT else ''}")
        print(f"      Time: {fold_time:.0f}s (train {fold_time-prep_s:.0f}s) | Total: {elapsed:.1f}min")

        if fold == 0:
            # A Booster has no feature_importances_; gain, with unsplit columns reindexed to 0.0.
            gain = booster.get_score(importance_type='gain')
            imp = (pd.Series(gain).reindex(booster.feature_names, fill_value=0.0)
                   .sort_values(ascending=False))
            print(f"\n      Top-15 feature importances (fold 1, gain; {len(gain)} of "
                  f"{len(booster.feature_names)} columns ever split on):")
            print(imp.head(15).to_frame('importance').to_string())

        del X_train, X_val, X_test, y_train, y_val, y_soft, p_teacher_tr, p_teacher_val
        del booster, dtrain, dval, dtest
        gc.collect()

    oof_cv = auc_score(y.values, oof_probs)
    collapsed = any(r > COLLAPSE_LIMIT for r in rho_fold)

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
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]} | Best iterations: {best_iters}")
    print(f"   dCV vs V11 (recorded 0.96113): {oof_cv - 0.96113:+.6f} | "
          f"dCV vs V2 (recorded 0.961009): {oof_cv - 0.961009:+.6f}")
    print(f"   Falsifier: max Spearman(student, teacher) {max(rho_fold):.4f} vs limit "
          f"{COLLAPSE_LIMIT} -> {'COLLAPSED: no diversity, reject regardless of CV' if collapsed else 'PASSES: student is not a copy'}")
    print(f"   Teacher fold AUC on the same validation rows: "
          f"{[f'{v:.5f}' for v in teacher_auc]}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(train_times):.0f}s per fold (mean) "
          f"| EV build {ev_build_s:.0f}s | total {total_time_min:.1f} min")
