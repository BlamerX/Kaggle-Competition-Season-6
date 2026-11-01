"""
S6E10 V40 - the original-data prior enters XGBoost as a base_margin OFFSET, not as a feature column - GPU (cuda)
GOAL: `_og_teacher` is 0.4177 of fold-1 gain in this frame - the tree spends roughly 42% of its splitting capacity
      re-deriving a law that could be handed to it as a starting point. XGBoost supports exactly that (`base_margin`),
      and no arm of the 37 has used it. S6E9's ledger is the external support: moving a prior from column to offset was
      +0.00005 CV there AND dissolved their mega-feature (fold-1 gain 9,642.9 -> 251.0) - the same capacity claim,
      measured on someone else's frame.
Axis: ONE change - HOW the prior enters. The teacher is built exactly as the parent builds it (same spec, same
      overlap-cleaned original, same call site); its probability becomes a logit with the house clip (1e-6), is passed
      as `base_margin` at fit / eval / predict, and the `_og_teacher` COLUMN is dropped - keeping both would
      double-count the prior, and dropping the column with no offset is the V21 ablation this campaign already ran.
Device: GPU (cuda) - XGBoost hist, per-fold curated TE + native cats, all byte-identical to the parent.
Parent: Previously trained files/Archieve/S6E10_V11_AuxEVFeatures.py (the best tree arm here: 13 aux-EV columns on
      V2's frame). Its feature count is therefore ours plus the teacher column, and both numbers print.
Arg-Name Law (this arm cannot skip it): `base_margin` appears NOWHERE in our 37 archived arms, and the installed build
      decides HOW it can be used - Kaggle's `fit` names `base_margin` but NOT `eval_base_margin` (measured 2026-10-24),
      which is the argument that keeps early stopping honest. So the pre-flight ranks three mechanisms and picks the
      first it can PROVE on synthetic rows, printing the choice: (1) `eval_base_margin`; (2) an eval `DMatrix` carrying
      its own margin, accepted only if the reported validation logloss moves > 0.005 when that margin is supplied vs
      withheld; (3) fit/predict margin with a margin-blind stopping point, which the run then labels its own reading
      as - a two-change arm, not a clean single-axis port. Naming is not adding (K5 died proving a signature can be
      blind to a live parameter), so each route is proved on coin-flip rows too: +2 logits must move the mean
      probability >= 0.20 at predict, and a model FIT against it must pull back toward 0.50; either failing raises.
Positive control (pre-declared): the arm prints the AUC of the offset ALONE (no model, just sigmoid of `base_margin`).
      The original-law teacher lands ~0.954 on this frame (K1b M2(c)), so a control below 0.80 means the margin carries
      almost nothing and the run RAISES rather than printing a clean-looking null.
Standalone (owner law 2026-10-11): one fitted model; no previous version's OOF is read; nothing is blended.
Gate: paired DeLong vs the parent's OOF and vs v2_oof: dAUC >= +0.0001 AND z > 3 AND >= 4/5 folds improved, then
      reproduce at SEED_REPRO=2027. The ruler is +-0.00006 CV, so a smaller delta is a tie about the lever.
Expected / Cost: direction unknown, size bounded - if capacity was the cost, the gain is the teacher block's own
      contribution re-expressed (the net's version of that channel was +0.00016); if not, a tie. ~6 min, the parent's.
Outputs: v40_oof.csv (id, pred) + v40_sub.csv (id, satisfaction) to /kaggle/working - exactly two writes.
Evidence: PLAN_V36_ONWARD.md §10b (the transfer reading) and §10d; Memory/training_logs.md V27 (the same channel added
      the feature-engineering way, which returned +0.00002 and is the reason this arm is about capacity, not signal).
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings

import os
import gc
import time
import inspect
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

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 40
    VERSION_NAME = "V40_PriorAsBaseMargin"
    EXP_ID = "S6E10_V40_PriorAsBaseMargin"
    PARENT = "V11_AuxEVFeatures"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027

    # Aux-EV builder switch (the axis is fixed: 13 EVs either way). True = run-of-record
    # original-fitted predictors (~1 min); False = plan's literal 5-fold cross-fit over the
    # ~1M pooled competition rows (~6 min). Flip ONLY when the two builds themselves are the
    # thing under test; measured equivalent (LGBM +0.00011).
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
# ektarr-aligned light regularization: the house alpha=10/lambda=10 was over-regularized here
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

# Aux-EV predictor template: every KEY copied from TEACHER_PARAMS (arg-name law - no invented
# kwarg); only two VALUES change - objective=multi:softprob (an EV needs P_hat(k) for every
# rating level) and the plan's 300-round budget. random_state stays 0 (as TEACHER_PARAMS does)
# so the EV block is byte-identical across the seed-42 and seed-2027 gate runs: CFG.RANDOM_SEED
# must remain the only config delta, so don't change this or the reproduction test breaks.
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


def add_aux_ev_features(train_df, test_df, orig_clean):
    """
    V11's ONE axis: add 13 `aux_ev_<rating>` columns, EV = sum_k k * P_hat(rating=k), where
    each P_hat comes from an XGB predicting that rating from the OTHER 20 ALL_FEATURES columns
    (never `satisfaction`). They are label-free feature producers, not a second competing model:
    only the main XGBoost predicts satisfaction, nothing is blended at inference.

    LEAKAGE GUARD (stated once, applies to both builders): the aux stage is fitted ONCE globally,
    never per outer fold, and no row's EV may come from a model that saw that row. DEFAULT
    (CFG.AUX_EV_FIT_ON_ORIGINAL=True, run-of-record) fits only on the overlap-cleaned original -
    `prepare_original` has dropped every original row equal to a competition row, so no competition
    row is in any fit set. ALT (False) keeps one 5-fold cross-fit over the pooled train+test rows
    and maps each row's OUT-OF-FOLD EV to both frames like V2 maps TE. Test labels cannot enter
    (the aux labels are ratings, not satisfaction). Returns (train_df, test_df).
    """
    def prep(frame, cols, vocab):
        """Same prep as `add_original_teacher`: numerics float32, cats as native
        categoricals over a vocabulary fitted on the predictor frame, unseen -> NaN."""
        X = frame[cols].copy()
        for c in cols:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('float32')
            else:
                X[c] = X[c].astype('string').astype(
                    pd.CategoricalDtype(categories=vocab[c]))
        return X

    def encode(y):
        """Contiguous 0-based codes + the ascending rating levels actually present. Codes (not
        raw labels) avoid the unverified `classes_` attribute (XGBoost multiclass handling shifted
        across 1.x/2.x); predict_proba then returns columns in code order = level order. The level
        vector is load-bearing: `Baggage handling` is 1-5 with NO 0, so EV must use the real k
        (sum_k k*P_hat(k)) not column indices, else that one column reads exactly 1.0 too low."""
        vals = y.to_numpy(dtype='int64')
        levels = np.unique(vals)
        return np.searchsorted(levels, vals).astype('int64'), levels.astype('float32')

    def ev_of(model, X, levels):
        """EV = sum_k k * P_hat(k) over the rating levels this model actually fitted."""
        return np.asarray(model.predict_proba(X), dtype='float32') @ levels

    if CFG.AUX_EV_FIT_ON_ORIGINAL:
        vocab = {c: sorted(orig_clean[c].astype('string').unique().tolist())
                 for c in CATS}
        for r in RATINGS:
            other = [c for c in ALL_FEATURES if c != r]
            yr = pd.to_numeric(orig_clean[r], errors='coerce')
            fit_rows = yr.notna()                       # ratings are ints; skip any null
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
                # Codes/levels come from THIS fit set so EV weights always match the class
                # space of the model that produced the probabilities (no column-count desync).
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
# 7b. THE AXIS'S OWN INSTRUMENT, PROVEN BEFORE A SINGLE ROW OF DATA IS READ
# =============================================================================
# K5's first version proved its argument with inspect.signature and died on Kaggle: the xgboost sklearn
# wrapper does not NAME constructor params like scale_pos_weight, it forwards them through **kwargs, so a
# signature can be blind to a parameter that works. base_margin is named on fit()/predict_proba(), so the
# signature check below is meaningful here - but "named" is not "added", and an ignored margin would print a
# perfectly plausible AUC while measuring nothing. So the arm proves the offset behaviourally, on synthetic
# rows, in about a second, before the EV build spends its minutes.
V40_MIN_OFFSET_MOVE = 0.20
V40_MIN_ES_DELTA = 0.005      # a margin that reaches the eval objective moves logloss by ~0.07; an ignored one, 0.000


def _lg(p):
    p = np.clip(np.asarray(p, dtype='float64'), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p)).astype('float32')


def v40_margin_preflight():
    """Prove the offset reaches fit and predict, then return the EARLY-STOPPING route this build can honour.

    Two separate questions, because Kaggle answered them differently: `fit` names `base_margin` but not
    `eval_base_margin`, so the named route is unavailable while the mechanism itself is fine. Fit/predict liveness is
    proved on coin-flip labels with one useless feature - no model can learn a ranking there, so only the margin can
    move the mean probability (+2 logits must move it by >= 0.20; a fit AGAINST that margin must pull the mean back
    toward 0.50, which is the fit-time half). The eval half is then settled by measurement on a second synthetic frame
    where the margin is the ONLY signal: whichever way the validation block is given its margin, a route the core
    honours must change the reported validation logloss, and a route it ignores must not.
    """
    _FIT = inspect.signature(xgb.XGBClassifier.fit).parameters
    _PRED = inspect.signature(xgb.XGBClassifier.predict_proba).parameters
    print(f"   Arg-Name guard: fit names base_margin={'base_margin' in _FIT} "
          f"eval_base_margin={'eval_base_margin' in _FIT} | "
          f"predict_proba names base_margin={'base_margin' in _PRED}")
    if not ('base_margin' in _FIT and 'base_margin' in _PRED):
        raise ImportError("This xgboost build cannot take a base_margin at fit or predict at all - the axis "
                          "is unavailable, not merely awkward.")

    rng = np.random.default_rng(3)
    n = 2000
    Xs = pd.DataFrame({'num': rng.normal(size=n).astype('float32'),
                       'cat': pd.Categorical(rng.integers(0, 3, size=n).astype('int8'))})
    ys = (rng.random(n) < 0.5).astype('int64')
    m0 = np.zeros(n, dtype='float32')
    m2 = np.full(n, 2.0, dtype='float32')
    base = dict(XGB_PARAMS)
    base.pop('early_stopping_rounds', None)
    base['n_estimators'] = 40

    fit0 = xgb.XGBClassifier(**base)
    fit0.fit(Xs, ys, base_margin=m0, verbose=False)
    p0_at0 = float(fit0.predict_proba(Xs, base_margin=m0)[:, 1].mean())
    p0_at2 = float(fit0.predict_proba(Xs, base_margin=m2)[:, 1].mean())
    fit2 = xgb.XGBClassifier(**base)
    fit2.fit(Xs, ys, base_margin=m2, verbose=False)
    p2_at2 = float(fit2.predict_proba(Xs, base_margin=m2)[:, 1].mean())
    print(f"   [V40 pre-flight] coin-flip labels, one useless feature | fit m=0: mean p {p0_at0:.4f} at "
          f"m=0 -> {p0_at2:.4f} at m=+2 (move {p0_at2 - p0_at0:+.4f}) | fit m=+2: mean p {p2_at2:.4f} at m=+2 "
          f"(a read fit-time margin pulls it back toward the 0.50 base rate)")
    del fit0, fit2
    gc.collect()
    if abs(p0_at2 - p0_at0) < V40_MIN_OFFSET_MOVE:
        raise RuntimeError(f"base_margin is INERT at predict time on this build: +2 logits of offset moved the "
                          f"mean probability by {p0_at2 - p0_at0:+.5f} (< {V40_MIN_OFFSET_MOVE}). The prior would "
                          f"not be entering the model at all, so a null here would measure a broken setup.")
    if abs(p2_at2 - 0.5) > V40_MIN_OFFSET_MOVE:
        raise RuntimeError(f"base_margin is INERT at fit time: fitted against a +2 offset yet the model still "
                          f"predicts {p2_at2:.4f} through it instead of correcting to ~0.50 - early stopping and "
                          f"the residuals would be scored against the wrong objective.")

    # ---- the eval half: three routes, ranked, each accepted only on a measured difference.
    if 'eval_base_margin' in _FIT:
        print("   [V40 route] eval_base_margin (named by fit) - early stopping sees the prior")
        return 'eval_base_margin'

    n_tr, n_va = 1200, 800
    ptr = 1.0 / (1.0 + np.exp(-rng.normal(size=n_tr) * 1.2))
    pva = 1.0 / (1.0 + np.exp(-rng.normal(size=n_va) * 1.2))
    Xa = pd.DataFrame({'num': rng.normal(size=n_tr).astype('float32'),
                       'cat': pd.Categorical(rng.integers(0, 3, size=n_tr).astype('int8'))})
    ya = (rng.random(n_tr) < ptr).astype('int64')
    ma = _lg(ptr)
    Xb = pd.DataFrame({'num': rng.normal(size=n_va).astype('float32'),
                       'cat': pd.Categorical(rng.integers(0, 3, size=n_va).astype('int8'))})
    yb = (rng.random(n_va) < pva).astype('int64')
    mb = _lg(pva)

    def _eval_ll(eval_pair):
        mdl = xgb.XGBClassifier(**base)
        mdl.fit(Xa, ya, base_margin=ma, eval_set=[eval_pair], verbose=False)
        r = mdl.evals_result()
        key = list(r.keys())[0]
        return float(np.mean(np.asarray(list(r[key].values())[-1], dtype='float64')[-3:]))

    try:
        dm_with = xgb.DMatrix(Xb, label=yb, base_margin=mb, enable_categorical=True)
        dm_wo = xgb.DMatrix(Xb, label=yb, base_margin=np.zeros(n_va, dtype='float32'),
                            enable_categorical=True)
        ll_with = _eval_ll((dm_with, None))
        ll_wo = _eval_ll((dm_wo, None))
        print(f"   [V40 route probe] eval DMatrix carrying the margin: validation logloss {ll_with:.5f} with it, "
              f"{ll_wo:.5f} without (delta {ll_with - ll_wo:+.5f})")
        if abs(ll_with - ll_wo) > V40_MIN_ES_DELTA:
            print("   [V40 route] dmatrix_eval_margin - early stopping sees the prior through the eval DMatrix")
            return 'dmatrix_eval_margin'
        print("   [V40 route probe] the eval DMatrix margin changed nothing - the core ignored it, so that route "
              "would be a decoration, not a fix")
    except Exception as e:
        print(f"   [V40 route probe] a DMatrix eval pair was rejected -> {type(e).__name__}: {str(e)[:110]}")

    print("   [V40 route] es_margin_blind - the prior enters at fit and predict only. EARLY STOPPING IS CHOSEN "
          "AGAINST A VALIDATION BLOCK SCORED WITHOUT THE OFFSET, so this run changes TWO things (the prior's route "
          "and the stopping objective) and its verdict is a mechanism reading, not a clean single-axis port.")
    return 'es_margin_blind'


def fit_margin_routes(model, X_tr, y_tr, m_tr, X_va, y_va, m_va, route):
    """One call site, three margin mechanisms - the pre-flight picks which one this build honours, and this is the
    only place the choice is used. Nothing about the model, the frame, the seed or the labels varies between them."""
    if route == 'eval_base_margin':
        model.fit(X_tr, y_tr, base_margin=m_tr, eval_set=[(X_va, y_va)],
                  eval_base_margin=[m_va], verbose=False)
    elif route == 'dmatrix_eval_margin':
        dva = xgb.DMatrix(X_va, label=y_va, base_margin=m_va, enable_categorical=True)
        model.fit(X_tr, y_tr, base_margin=m_tr, eval_set=[(dva, None)], verbose=False)
    else:
        model.fit(X_tr, y_tr, base_margin=m_tr, eval_set=[(X_va, y_va)], verbose=False)

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis: the original-data prior enters as a base_margin OFFSET and the `_og_teacher` column is dropped | "
          f"Parent {CFG.PARENT} | gate vs the parent's OOF offline - no saved OOF is a model input")
    V40_ROUTE = v40_margin_preflight()
    ev_variant = ('original-fitted predictors (plan default / run-of-record)'
                  if CFG.AUX_EV_FIT_ON_ORIGINAL
                  else '5-fold cross-fit over ~1M pooled competition rows')

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

    # V11's ONLY axis change vs V2: the 13 EV columns are added here, before the float32
    # cast and before FEATURES is fixed, so the fold driver / TE / params stay byte-identical.
    t0_ev = time.time()
    train, test = add_aux_ev_features(train, test, orig_clean)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    print(f"   V11 axis: {ev_build_s:.0f}s EV build | +{len(EV_COLS)} aux_ev columns: {EV_COLS}")

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    # ---- V40 AXIS: the prior changes route, not source. Teacher probability -> logit offset (the house clip is
    # V12's 1e-6), then the column itself leaves the frame so nothing about the prior is counted twice.
    _tp = np.clip(train['_og_teacher'].to_numpy('float64'), 1e-6, 1 - 1e-6)
    train_margin = np.log(_tp / (1 - _tp)).astype('float32')
    _tq = np.clip(test['_og_teacher'].to_numpy('float64'), 1e-6, 1 - 1e-6)
    test_margin = np.log(_tq / (1 - _tq)).astype('float32')
    train.drop(columns=['_og_teacher'], inplace=True)
    test.drop(columns=['_og_teacher'], inplace=True)
    assert np.isfinite(train_margin).all() and np.isfinite(test_margin).all(), "non-finite margin"
    _prior_auc = auc_score(train[CFG.TARGET].to_numpy('int64'), train_margin)
    print(f"   V40 axis: teacher column -> base_margin offset | margin mean {train_margin.mean():+.3f} sd {train_margin.std():.3f} | frame {len(test.columns)} cols (one fewer than the parent)")
    print(f"   POSITIVE CONTROL - the offset ALONE scores AUC {_prior_auc:.5f} on train (the orig-law teacher lands ~0.954; below 0.80 the margin carries nothing)")
    if _prior_auc < 0.80:
        raise RuntimeError(f"the base_margin carries almost nothing (solo AUC {_prior_auc:.5f}) - this would measure a broken setup, not the mechanism")

    FEATURES = list(test.columns)
    print(f"   Total base features: {len(FEATURES)}")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    train_times = []

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
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | "
              f"Val: {X_val.shape} | Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s")

        model = xgb.XGBClassifier(**XGB_PARAMS)
        # The mechanism that carries the margin into the EVAL block is whatever this build proves it honours;
        # eval_base_margin is the honest one, an eval DMatrix is the fallback, and neither existing at all is
        # printed as a two-change reading rather than passed off as the parent's objective.
        fit_margin_routes(model, X_train, y_train, train_margin[train_idx],
                          X_val, y_val, train_margin[val_idx], V40_ROUTE)

        val_probs = model.predict_proba(X_val, base_margin=train_margin[val_idx])[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test, base_margin=test_margin)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        best_iter = (model.best_iteration if hasattr(model, 'best_iteration')
                     and model.best_iteration is not None else model.n_estimators)
        best_iters.append(best_iter)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        train_times.append(fold_time - prep_s)
        print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
              f"Time: {fold_time:.0f}s (train {fold_time-prep_s:.0f}s) | Total: {elapsed:.1f}min")

        if fold == 0:
            imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_,
            }).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))

        del X_train, X_val, X_test, y_train, y_val, model
        gc.collect()

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
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Axis: prior as base_margin (an offset, not a column) on {CFG.PARENT}'s frame | margin route used: "
          f"{V40_ROUTE}{' - EARLY STOPPING WAS MARGIN-BLIND, so this is a two-change reading' if V40_ROUTE == 'es_margin_blind' else ''}"
          f" | {len(EV_COLS)} aux-EV columns | builder: {ev_variant} | EV build: {ev_build_s:.0f}s")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best iters: {best_iters}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(train_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
