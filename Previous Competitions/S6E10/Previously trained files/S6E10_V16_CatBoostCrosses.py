"""
S6E10 V16 - CatBoost: AUC early stopping + 39 native rating x context crosses - GPU (cuda)
GOAL: land the interaction block with the largest published in-episode FE delta (13 ratings x
      {Class, Type of Travel, Customer Type} as native `category` columns: +0.000337 / +0.000284,
      5/5 folds; +0.00013 replicated on a rich frame) and retire V4's falsified `eval_metric='Logloss'`.
Axis: LADDER over the interaction frame - one fitted model per rung, rungs NEVER combined, exactly one
      rung is the version's output (the V8 / V14 pattern, so this stays a Single Model):
      `none` = V4's frame under AUC early stopping (in-run control, re-fitted not loaded);
      `add`  = +39 crosses, manual pair TE kept;
      `swap` = +39 crosses with the manual `*_pair_*` TE/freq/rarity family dropped so CatBoost's own
               ordered target statistics own the interactions.
Reads: dCV(none vs V4's recorded 0.96088) = the metric delta (V16a); dCV(record vs none) = the frame
      delta (V16b) - both fall out of ONE standalone run, so neither is confounded with the other.
Parent: Previously trained files/Archieve/S6E10_V4_CatBoostMember.py (0.96088 CV / 0.96051 LB).
Device: GPU (cuda) - CatBoost task_type='GPU' devices='0' on a Kaggle T4 (16 GB); the ektarr
      teacher runs device='cuda'; all numeric matrices float32.
Params: CB_PARAMS byte-identical to V4 (6000 iters, lr 0.05, depth 8, l2_leaf_reg 5.0, random_strength
      1.0, Bernoulli subsample 0.8, es 200, seed 42, task_type='GPU'); only eval_metric -> 'AUC', and
      it is held CONSTANT across all three rungs so the rung delta belongs to the frame alone.
Standalone (owner law 2026-10-11): no previous version's OOF is read as a feature or a target here;
      the v4 / v2 OOFs are used only OFFLINE, after the run, for the gate below.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = SEED_REPRO=2027.
Gate: paired DeLong vs v4_oof AND v2_oof, externally: dAUC > +0.00005 AND z > 3 AND >=4/5 folds
      improved, then reproduce at 2027. Tie band 0.0003 - decide on CV, audit on LB.
Run-of-record (pre-declared): the highest-CV rung beating `none` on >=4/5 folds; else `none` stands.
Expected / Cost: +0.0001 ... +0.0003 CV over `none`; 3 rungs x 5 folds on a T4 ~25-30 min (V4: 7.9).
Outputs: v16_rung_none/add/swap_oof.csv (audit) + v16_oof.csv / v16_sub.csv (one model, one sub).
Evidence: PLAN_V16_ONWARD.md §3 (V16a + V16b fused), PLAN_V7_ONWARD.md §1 item 10.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings

import os
import gc
import time
import random
from importlib.metadata import version
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
print(f"catboost version: {version('catboost')} | xgboost version: {xgb.__version__}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = "16"
    VERSION_NAME = "V16_CatBoostCrosses"
    EXP_ID = "S6E10_V16_CatBoostCrosses"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027        # reproduction run changes only RANDOM_SEED (42 -> 2027)

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

# THE AXIS INPUT: 13 ratings x 3 context columns = 39 explicit interaction pairs.
CROSS_CONTEXTS = ['Class', 'Type of Travel', 'Customer Type']
CROSS_COLS = [f'_x_{SHORT[r]}__{SHORT[c]}' for r in RATINGS for c in CROSS_CONTEXTS]
RUNGS = ['none', 'add', 'swap']
V4_RECORDED_CV = 0.96088     # Memory/public_scores.md. A recorded number, not an OOF vector.

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
# Byte-identical to V4 except eval_metric, which is held constant across every rung so the rung delta
# is attributable to the frame alone. V4's comment claimed AUC was unavailable on GPU; the live kernel
# sweep disproved it (wangxintong111 ran loss='Logloss', eval_metric='AUC', task_type='GPU' on a
# Kaggle T4), and the fold-0 pre-flight below re-proves it on THIS build before any long fit.
CB_PARAMS = dict(
    loss_function='Logloss',
    eval_metric='AUC',        # V4: 'Logloss' - this version's metric change of record
    iterations=6000, learning_rate=0.05, depth=8,
    l2_leaf_reg=5.0, random_strength=1.0,
    bootstrap_type='Bernoulli', subsample=0.8,
    task_type='GPU', devices='0',
    early_stopping_rounds=200, random_seed=CFG.RANDOM_SEED,
    verbose=0, allow_writing_files=False,
)

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
# 6. FEATURE ENGINEERING (fold-independent part) - byte-identical to V4
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
    categorical vocabularies from the fit fold - the 39 crosses get exactly V4's
    cat treatment (fit-fold vocabulary, unseen level -> NaN). Returns (3 frames,
    n_added). Frame build happens ONCE per fold and all three rungs slice columns
    out of it: TargetEncoder fits each column independently and the frequency maps
    are per-column value_counts, so dropping the `*_pair_*` columns afterwards is
    identical to never having encoded them - which is what makes `swap` a clean
    single-variable contrast against `add`."""
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

    for c in CATS + CROSS_COLS:
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
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | {CFG.EXP_ID}")
    print(f"Axis: ladder {RUNGS} over the interaction frame (one fitted model per rung, "
          f"rungs never combined) | eval_metric='AUC' held constant")
    print(f"Parent V4: {V4_RECORDED_CV} CV / 0.96051 LB | Control V2: 0.961009 CV / 0.96051 LB "
          f"| standalone: no previous OOF is read as a model input")
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
    print(f"   Missing in train: {int(train.isna().sum().sum())} (CatBoost handles NaN natively)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering (V4 recipe byte-equal, plus the axis block)...")

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

    print("   Adding the V16 axis (rating x context crosses)...")
    for df in (train, test):
        for r in RATINGS:
            for c in CROSS_CONTEXTS:
                df[f'_x_{SHORT[r]}__{SHORT[c]}'] = (
                    df[r].astype('string') + '|' + df[c].astype('string'))
    print(f"      +{len(CROSS_COLS)} crosses | levels in fit-set terms: "
          f"{CROSS_COLS[0]} {train[CROSS_COLS[0]].nunique(dropna=True)}, "
          f"{CROSS_COLS[1]} {train[CROSS_COLS[1]].nunique(dropna=True)}, "
          f"{CROSS_COLS[2]} {train[CROSS_COLS[2]].nunique(dropna=True)}")

    print("   Adding wide target-free route profile (mean/std/resid/shares)...")
    train, test = add_route_profile(train, test)

    print("   Adding original-data org stats (all features + triple)...")
    train, test = add_org_stats(train, test, orig)

    print("   Preparing original (overlap removal) + ektarr-spec teacher...")
    orig_clean = prepare_original(orig, train, test)
    train, test = add_original_teacher(train, test, orig_clean)

    for c in test.columns:
        if c not in CATS + CROSS_COLS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    FEATURES = list(test.columns)
    print(f"   Total base features: {len(FEATURES)} (V4's 139 + {len(CROSS_COLS)} crosses)")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV x 3 rungs, per-fold curated TE + native cats)
    # =========================================================================
    print(f"\n[3/5] Training CatBoost ladder {RUNGS} ({CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_per_rung  = {r: np.zeros(len(y)) for r in RUNGS}
    test_per_rung = {r: np.zeros(len(test_X)) for r in RUNGS}
    rung_fold     = {r: [] for r in RUNGS}
    rung_best     = {r: [] for r in RUNGS}
    prep_times    = []
    fit_times     = []   # print-support only: per-fold fit seconds for the RESULTS block

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
        pair_cols = [c for c in X_train.columns if '_pair_' in c]
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Per-fold cols added: {n_added} | pair cols {len(pair_cols)} | "
              f"Prep: {prep_s:.0f}s (shared by all rungs)")

        # ---- Arg-Name Law pre-flight (PLAN_V16_ONWARD.md rule 4), fold 0 only. Both claims this arm
        # rests on - that this CatBoost build accepts eval_metric='AUC' on GPU, and that the 39
        # interaction columns pass through cat_features - are proven on one cheap fit before the real
        # rungs start. The probe model is discarded and the seeds restored, so the run-of-record stays
        # stream-identical to V4's.
        if fold == 0:
            probe = CatBoostClassifier(**{**CB_PARAMS, 'iterations': 30, 'depth': 4,
                                          'early_stopping_rounds': None})
            pX, py = X_train.iloc[:20000], y_train.iloc[:20000]
            probe.fit(pX, py, cat_features=CATS + CROSS_COLS, verbose=False)
            print(f"      [pre-flight] eval_metric='{CB_PARAMS['eval_metric']}' on "
                  f"task_type='{CB_PARAMS['task_type']}' with {len(CROSS_COLS)} cross cat_features: "
                  f"OK | probe AUC (30 iters, 20k rows, discarded) "
                  f"{auc_score(py.values, probe.predict_proba(pX)[:, 1]):.5f}")
            del probe, pX, py
            gc.collect()
            seed_everything(CFG.RANDOM_SEED)

        imp_fold0 = None
        for rung in RUNGS:
            if rung == 'none':
                Xr_tr = X_train.drop(columns=CROSS_COLS)
                Xr_va = X_val.drop(columns=CROSS_COLS)
                Xr_te = X_test.drop(columns=CROSS_COLS)
                cats_used = CATS
            elif rung == 'add':
                Xr_tr, Xr_va, Xr_te = X_train, X_val, X_test
                cats_used = CATS + CROSS_COLS
            else:   # 'swap': crosses in, the manual pair-TE family out
                Xr_tr = X_train.drop(columns=pair_cols)
                Xr_va = X_val.drop(columns=pair_cols)
                Xr_te = X_test.drop(columns=pair_cols)
                cats_used = CATS + CROSS_COLS

            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(Xr_tr, y_train, eval_set=(Xr_va, y_val),
                      cat_features=cats_used, use_best_model=True)

            val_probs = model.predict_proba(Xr_va)[:, 1]
            oof_per_rung[rung][val_idx] = val_probs
            test_per_rung[rung] += model.predict_proba(Xr_te)[:, 1] / CFG.N_FOLDS

            fold_auc = auc_score(y_val.values, val_probs)
            rung_fold[rung].append(fold_auc)
            best_iter = model.get_best_iteration()
            rung_best[rung].append(best_iter)
            if fold == 0 and rung == 'add':
                imp_fold0 = (Xr_tr.columns, model.get_feature_importance())

            print(f"      {rung:>4}: cols {Xr_tr.shape[1]:>3} | AUC {fold_auc:.5f} | "
                  f"BestIter {best_iter} | elapsed {(time.time()-t0)/60:.1f}min")

            del model, Xr_tr, Xr_va, Xr_te, val_probs
            gc.collect()

        fit_times.append((time.time() - fold_start) - prep_s)

        if imp_fold0 is not None:
            imp = pd.DataFrame({'feature': imp_fold0[0],
                                'importance': imp_fold0[1]}
                               ).sort_values('importance', ascending=False)
            print(f"\n      Top-15 feature importances (fold 1, rung 'add'):")
            print(imp.head(15).to_string(index=False))
            del imp, imp_fold0

        del X_train, X_val, X_test, y_train, y_val
        gc.collect()

    # ---- rung CVs, then the pre-declared run-of-record rule ------------------
    rung_cv = {r: auc_score(y.values, oof_per_rung[r]) for r in RUNGS}
    improved = {r: int(sum(1 for a, b in zip(rung_fold[r], rung_fold['none']) if a > b))
                for r in RUNGS}
    candidates = [r for r in RUNGS if r != 'none' and improved[r] >= 4
                  and rung_cv[r] > rung_cv['none']]
    record_rung = max(candidates, key=lambda r: rung_cv[r]) if candidates else 'none'

    oof_probs, test_probs = oof_per_rung[record_rung], test_per_rung[record_rung]
    fold_scores, best_iters = rung_fold[record_rung], rung_best[record_rung]
    oof_cv = rung_cv[record_rung]

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    for r in RUNGS:
        pd.DataFrame({'id': train_id, 'pred': oof_per_rung[r]}).to_csv(
            f"{out_dir}/{stem}_rung_{r}_oof.csv", index=False)
        print(f"   [SAVED] {out_dir}/{stem}_rung_{r}_oof.csv — per-rung audit trail")

    pd.DataFrame({'id': train_id, 'pred': oof_probs}).to_csv(
        f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred) — rung of record '{record_rung}'")

    pd.DataFrame({'id': test_id, CFG.TARGET: test_probs}).to_csv(
        f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction) — same single model")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Rung of record: '{record_rung}' (pre-declared: highest CV beating 'none' on >=4/5 folds)")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Best iters: {best_iters}")
    for r in RUNGS:
        print(f"   Rung {r:>4}: OOF CV {rung_cv[r]:.5f} | folds over 'none': {improved[r]}/{CFG.N_FOLDS}")
    print(f"   dCV metric (none vs V4 {V4_RECORDED_CV}): {rung_cv['none'] - V4_RECORDED_CV:+.6f} | "
          f"dCV frame ({record_rung} vs none): {oof_cv - rung_cv['none']:+.6f}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
