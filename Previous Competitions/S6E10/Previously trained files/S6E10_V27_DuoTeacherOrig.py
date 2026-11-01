"""
S6E10 V27 - Duo Teacher On Original Data (teacher axis: `_og_teacher` gains a net member) - GPU (cuda)
GOAL: `_og_teacher` is 40.4% of fold-1 gain (recorded K1 census) and it has always been one XGBoost fitted on the clean
      original. V3 tried a second teacher and the ledger closed the axis as "tree+tree diversity adds
      nothing" - but V3 also overwrote a [0,1] probability column with a raw logit mean in the same step,
      so the ledger's own correction table says that run measured two things and proved one. This arm is
      the test V3 should have been: the second member is a network, so the two teachers disagree by
      inductive bias rather than by seed, and the column keeps the parent's scale exactly.
Axis: ONE change - `_og_teacher` becomes sigmoid(mean(logit_xgb, logit_net)) instead of the XGB
      probability alone. Both members are fitted on the same overlap-cleaned original, neither ever sees
      a competition label, and the column stays in [0,1] so nothing downstream has to be re-read. The
      net's own recipe is V12/V18's dict verbatim; the XGB member, the frame, the TE block, the 13 EVs,
      the main model, the folds and the seed are V11's byte-for-byte.
      Pre-falsifier, printed before any CV number matters: Spearman of the two teachers' logits over the
      competition TRAIN rows - rows neither member has seen. At >= 0.990 the net adds no inductive bias
      and this arm is a null at any CV.
Parent: Previously trained files/Archieve/S6E10_V11_AuxEVFeatures.py (frozen, recorded
         0.961133 CV / 0.96064 LB).
Device: GPU (cuda) - XGBoost with device='cuda' and RealMLP_TD via PyTabKit on a Kaggle T4 (16 GB).
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = RANDOM_SEED=2027.
Gate: paired DeLong vs v11_oof.csv, computed offline - this arm reads no saved OOF. Ship only at
      dAUC > +0.00005 AND z > 3 AND >=4/5 folds improved, then reproduce at RANDOM_SEED=2027.
Expected / Cost: +0.00000 ... +0.00015 CV; ~10 min.
Outputs: v27_oof.csv (id, pred) + v27_sub.csv (id, satisfaction) to /kaggle/working.
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
import inspect
import subprocess, sys
import torch
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
    VERSION_NUMBER = 27
    VERSION_NAME = "V27_DuoTeacherOrig"
    EXP_ID = "S6E10_V27_DuoTeacherOrig"

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

# V27's second teacher. This dict is V12/V18's, copied verbatim rather than re-derived:
# 'epochs'/'batch'/'auto_configure' are not this build's names and invented kwargs have cost
# this episode five failed runs.
REALMLP_PARAMS = {
    'random_state': CFG.RANDOM_SEED,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,        # internal bagging inside one model object, not an ensemble
    'n_epochs': 3,     # V12/V18's budget, transplanted unchanged
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


# Falsifier bar for teacher diversity, the same limit V20 uses for student-teacher collapse.
COLLAPSE_LIMIT = 0.990

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


def add_original_teacher(train_df, test_df, orig_clean, y_train=None):
    """ektarr-spec teacher (XGB d8, cs0.5, 600 trees, all 21 features, native cats)
    fitted on the overlap-cleaned original only — never touches competition labels.

    V27's ONE axis: the column this writes becomes sigmoid(mean(logit_xgb, logit_net)) for a
    RealMLP_TD net fitted on the same rows of the same clean original. The sigmoid back-projection is
    what keeps this a single axis: the parent stores a probability, so the slot keeps its [0,1] scale
    and only its value moves - V3 changed both at once and proved neither. The net takes no NaN, so
    numeric columns are filled with the ORIGINAL medians (counted and printed); the XGB member keeps
    the parent's untouched frame. `y_train` is the competition train label and is used ONLY for the
    printed teacher census; it never enters a fit, because both members are fitted on original rows.
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

    def logit(p):
        p = np.clip(np.asarray(p, dtype='float64'), 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    med = prep(orig_clean).median(numeric_only=True)

    def net_prep(frame):
        X = prep(frame)
        n_nan = int(X.isna().sum().sum())
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(med.get(c, 0.0)).astype('float32')
        return X, n_nan

    Xo = prep(orig_clean)
    yo = orig_clean[CFG.TARGET].astype(int)
    m = xgb.XGBClassifier(**TEACHER_PARAMS)
    m.fit(Xo, yo, verbose=False)

    # use_early_stopping is False in this recipe, so the val pair feeds monitoring only and never
    # selects a stopping point; passing the fit set keeps the net on the parent's exact rows.
    Xo_net, n_nan_fit = net_prep(orig_clean)
    left_fit = int(Xo_net.isna().sum().sum())
    if left_fit:
        raise AssertionError(f"the net teacher's fit matrix keeps {left_fit} NaN cells "
                             f"(numeric filled: {n_nan_fit}) - an unseen categorical level")
    m_net = RealMLP_TD_Classifier(**REALMLP_PARAMS)
    m_net.fit(Xo_net, yo.to_numpy(), Xo_net, yo.to_numpy())

    X_tr_net, n_nan_tr = net_prep(train_df)
    X_te_net, n_nan_te = net_prep(test_df)
    # Numeric NaN is filled from the original medians; what cannot be filled is a categorical level the
    # original never carried, and a NaN into RealMLP is a silently degenerate teacher column - so this
    # arm stops rather than trains on it.
    for tag, Xn in (("train", X_tr_net), ("test", X_te_net)):
        left = int(Xn.isna().sum().sum())
        if left:
            bad = [c for c in Xn.columns if int(Xn[c].isna().sum())]
            raise AssertionError(f"{tag} frame keeps {left} NaN cells for the net teacher in {bad}")
    p_xgb_tr = np.asarray(m.predict_proba(prep(train_df)), dtype='float64')[:, 1]
    p_xgb_te = np.asarray(m.predict_proba(prep(test_df)), dtype='float64')[:, 1]
    p_net_tr = np.asarray(m_net.predict_proba(X_tr_net), dtype='float64')[:, 1]
    p_net_te = np.asarray(m_net.predict_proba(X_te_net), dtype='float64')[:, 1]
    del Xo, Xo_net, X_tr_net, X_te_net
    gc.collect()

    duo_tr = sigmoid(0.5 * (logit(p_xgb_tr) + logit(p_net_tr)))
    duo_te = sigmoid(0.5 * (logit(p_xgb_te) + logit(p_net_te)))
    if float(duo_tr.min()) < 0.0 or float(duo_tr.max()) > 1.0:
        raise AssertionError("duo teacher left [0,1]")
    rho = float(np.corrcoef(pd.Series(logit(p_xgb_tr)).rank().to_numpy(),
                            pd.Series(logit(p_net_tr)).rank().to_numpy())[0, 1])
    print(f"      teacher census on {len(p_xgb_tr)} rows neither member was fitted on "
          f"| NaN filled for the net: fit {n_nan_fit}, train {n_nan_tr}, test {n_nan_te}")
    if y_train is not None:
        y_ref = np.asarray(y_train, dtype='int64')
        print(f"      teacher AUC alone: xgb {auc_score(y_ref, p_xgb_tr):.5f} | "
              f"net {auc_score(y_ref, p_net_tr):.5f} | "
              f"logit-mean {auc_score(y_ref, duo_tr):.5f}")
    print(f"      Spearman(xgb logit, net logit) = {rho:+.5f} | limit {COLLAPSE_LIMIT} -> "
          f"{'DIVERSITY OK' if rho < COLLAPSE_LIMIT else 'REJECTED: the net adds no inductive bias'}")
    if rho >= COLLAPSE_LIMIT:
        # Pre-declared here and in the plan: above the limit the arm is a null AT ANY CV, so it stops now
        # instead of spending five folds proving a premise the falsifier already rejected.
        raise RuntimeError(f"teacher diversity collapsed: Spearman {rho:+.5f} >= {COLLAPSE_LIMIT}")

    train_df['_og_teacher'] = duo_tr.astype('float32')
    test_df['_og_teacher'] = duo_te.astype('float32')
    del m, m_net
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
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    # ARG-NAME LAW pre-flight: the recipe is inherited from V12/V18, but a silently-absorbed unknown
    # key would still invalidate the comparison, so any key this build does not name is printed.
    _SIG = inspect.signature(RealMLP_TD_Classifier.__init__).parameters
    _unnamed = [k for k in REALMLP_PARAMS if k not in _SIG]
    print(f"RealMLP_TD_Classifier accepts n_epochs: {'n_epochs' in _SIG} "
          f"| keys absorbed by **kwargs: {_unnamed if _unnamed else 'none'}")
    if 'n_epochs' not in _SIG:
        raise ImportError("This PyTabKit build has no n_epochs parameter - stopping before the fits.")
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis: _og_teacher becomes sigmoid(mean logit) of the parent XGB + a RealMLP_TD net "
          f"| gated offline vs v11_oof.csv")
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
    train, test = add_original_teacher(train, test, orig_clean,
                                     y_train=train[CFG.TARGET])

    # INHERITED from V11, not this arm's axis: the 13 EV columns are added here, before the float32
    # cast and before FEATURES is fixed, so the fold driver / TE / params stay byte-identical.
    t0_ev = time.time()
    train, test = add_aux_ev_features(train, test, orig_clean)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    print(f"   EV block (inherited from V11): {ev_build_s:.0f}s build, {ev_variant} | "
          f"{len(EV_COLS)} aux_ev columns")

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

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
    print(f"   Axis: duo original teacher (xgb + RealMLP_TD), teacher of record logit-mean")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best iters: {best_iters}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(train_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
