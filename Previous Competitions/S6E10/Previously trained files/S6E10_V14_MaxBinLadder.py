"""
S6E10 V14 - max_bin resolution ladder (one config value; parent V2) [CLOSURE TIER] - GPU (cuda)
GOAL: sweep XGBoost `max_bin` over the pre-declared ladder 256 -> 1024 -> 4096 on V2's exact
      frame, one single fitted model per rung. A win would prove bin resolution (not FD's coarse
      monotone label law) is the binding constraint, and re-create the demotion evidence
      (the external `v7b_out.txt` reading) via the fold-independent diagnostic below.
Axis: ONE change — `max_bin` via CFG.MAX_BIN_LADDER, overridden only at the fit site; every other
      XGB_PARAMS key is byte-identical to V2; FE/TE/teacher/folds/device all untouched.
Parent: Previously trained files/Archieve/S6E10_V2_XGB_WideProfile.py (frozen, 0.961009 CV / 0.96051 LB).
Device: GPU (cuda) — tree_method='hist', device='cuda' on a Kaggle T4; all numeric matrices float32.
Folds: StratifiedKFold(5, shuffle=True, random_state=42) frozen; reproduction run = RANDOM_SEED=2027 (CFG.SEED_REPRO).
Gate: paired DeLong vs v2_oof.csv (offline): ship at dAUC > +0.00005 AND z > 3 AND >=4/5 folds improved,
      then reproduce at RANDOM_SEED=2027. The in-script CV half only records a tie, not an improvement.
Expected / Cost: +0.00000 ... +0.0001, ~5 min (plan estimate, closure tier — run last).
Outputs: v14_oof.csv (id, pred) + v14_sub.csv (id, satisfaction) to /kaggle/working; per-rung OOF also written.
Evidence behind this arm: `PLAN_V7_ONWARD.md` §2 "### V14" (planning file, kept outside this repo).
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings

import os
import gc
import time
import random
import subprocess
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
    VERSION_NUMBER = 14
    VERSION_NAME = "V14_MaxBinLadder"
    EXP_ID = "S6E10_V14_MaxBinLadder"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027  # reproduction run changes ONLY this; fold seed stays frozen at 42

    # Pre-declared coordinate ladder (plan §2 V14): never extended, searched, or combined.
    MAX_BIN_LADDER = [256, 1024, 4096]

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
# ektarr-aligned light regularization; the house alpha=10/lambda=10 was over-regularized here
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

# =============================================================================
# 5. METRIC - ROC AUC (Competition Metric)
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 5b. V14 ONLY: D4 bin-resolution diagnostic + GPU memory guard
#     (fold-independent; nothing here is fitted on a fold or fed to the model)
# =============================================================================
# Diagnostic bin rows: plan's measured ladder (16/64/256/1024) + top rung (4096).
DIAG_BINS = [16, 64] + list(CFG.MAX_BIN_LADDER)


def equal_width_bin(values, n_bins):
    """Equal-width integer binning (plan's counter-measurement scheme); AUC of the
    binned vector is the rank-preserving fraction the D4 table reports."""
    v = np.asarray(values, dtype='float64')
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.zeros(len(v), dtype='int64')
    return np.floor((v - lo) * (float(n_bins) / (span + 1.0))).astype('int64')


def quantisation_table(name, values, y, n_levels_hint):
    """Print one vector's rank-preserving single-column AUC (raw, then per bin count)."""
    v = np.asarray(values, dtype='float64')
    finite = np.isfinite(v)
    raw_auc = auc_score(y[finite], v[finite])
    print(f"   {name}: distinct levels = {n_levels_hint} | "
          f"raw single-column AUC = {raw_auc:.5f}  (n = {int(finite.sum()):,})")
    if raw_auc < 0.5:
        print(f"      NOTE: raw AUC < 0.5, read the ratios below as |AUC-0.5| retention.")
    for b in DIAG_BINS:
        bv = equal_width_bin(v[finite], b)
        auc_b = auc_score(y[finite], bv)
        retention = auc_b / raw_auc * 100.0
        print(f"      {b:>5} bins: AUC {auc_b:.5f}  = {retention:6.2f} % of raw"
              f"  (dAUC {auc_b - raw_auc:+.5f}) | distinct binned = {len(np.unique(bv))}")
    return raw_auc


def te_fd_exact_diagnostic(train, y):
    """Build both `_tm_fd_exact` flavours (cross-fitted TE idiom with a fixed
    fold-independent inner seed rs=1042; and the in-sample smoothed group stat) for
    measurement only — neither touches a frozen fold nor enters the model matrix."""
    fd = train['Flight Distance'].astype('int64')
    key = pd.DataFrame({'fd_exact': fd})
    skv = tuple(map(int, sklearn_version.split('.')[:2]))
    if skv >= (1, 9):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1042)
        enc = TargetEncoder(target_type='binary', smooth=20.0, cv=inner)
    else:
        enc = TargetEncoder(target_type='binary', smooth=20.0, cv=5,
                            shuffle=True, random_state=1042)
    te_xfit = np.asarray(enc.fit_transform(key, y))[:, 0]

    base = float(np.mean(y))
    df = pd.DataFrame({'k': fd.values, 'y': np.asarray(y, dtype='int64')})
    agg = df.groupby('k')['y'].agg(['sum', 'count'])
    smooth = 20.0
    stat = (agg['sum'] + smooth * base) / (agg['count'] + smooth)
    te_insamp = fd.map(stat).astype('float64').values

    return te_xfit, te_insamp, int(fd.nunique())


def gpu_memory_mib(tag):
    """used/total GPU MiB via nvidia-smi (stdlib subprocess, no new dep); (None, None)
    if unreadable, then the guard degrades to estimate-only."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        used = sum(float(line.split(',')[0]) for line in out if line)
        total = sum(float(line.split(',')[1]) for line in out if line)
        return used, total
    except Exception as e:
        print(f"      GPU memory reading unavailable at {tag}: {type(e).__name__}: {e}")
        return None, None


def rung_hist_memory_gb(n_cols, max_bin, n_rows):
    """COARSE upper-bound XGBoost GPU demand (guard heuristic, not a guarantee):
    histogram buffers (n_cols x max_bin x float32 x ~256 workers) + banded DMatrix."""
    hist_gb = (n_cols * max_bin * 4.0 * 256.0) / 1e9
    dmatrix_gb = (n_rows * n_cols * 4.0) / 1e9
    return hist_gb, dmatrix_gb


GPU_MIN_FREE_GB = 3.0        # refuse a rung whose estimate leaves < 3 GB headroom
GPU_TOTAL_GB_T4 = 16.0       # documented T4 budget


def rung_memory_check(rung_params, n_cols, n_rows, tag):
    """Print GPU memory at rung start and decide fit/skip; never crashes."""
    used, total = gpu_memory_mib(tag)
    hist_gb, dm_gb = rung_hist_memory_gb(n_cols, rung_params['max_bin'], n_rows)
    est_gb = hist_gb + dm_gb
    total_txt = (f"{total:.0f} MiB" if total is not None
                 else f"n/a (T4 budget {GPU_TOTAL_GB_T4:.0f} GB)")
    used_txt = f"{used:.0f} MiB" if used is not None else "n/a"
    print(f"      [{tag}] GPU at start: used={used_txt} total={total_txt} | demand est "
          f"~{est_gb:.2f} GB (hist ~{hist_gb:.2f} @ max_bin={rung_params['max_bin']} "
          f"x {n_cols} cols x 256 workers; banded ~{dm_gb:.2f} GB / {n_rows:,} rows)")
    if used is not None and total is not None:
        free_gb = (total - used) / 1024.0
        need_gb = est_gb + GPU_MIN_FREE_GB
        print(f"      [{tag}] free ~{free_gb:.2f} GB | needed ~{need_gb:.2f} GB "
              f"(estimate + {GPU_MIN_FREE_GB:.0f} GB headroom)")
        if need_gb > free_gb:
            return False, est_gb, used, total
        return True, est_gb, used, total
    print(f"      [{tag}] memory unreadable -> proceeding (per-rung try/except protects it)")
    return True, est_gb, used, total

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
    unseen -> orig prior) + Travel x Class x Customer triple. NOTE: no k=20 smoothed
    variant is actually written here — the unshrunk means are V2's inherited defect (V13)."""
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
    print(f"Axis vs parent V2 (S6E10_V2_XGB_WideProfile.py, 0.961009 CV / 0.96051 LB): "
          f"max_bin ladder {CFG.MAX_BIN_LADDER} ONLY — one fitted model per rung, "
          f"rungs never combined, all other params byte-identical to V2. "
          f"Tier: CLOSURE (expected +0.00000...+0.0001, ~5 min).")

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

    # [Diag] D4 fold-independent bin-resolution diagnostic: runs before FE, uses
    # only the raw column + a label-side TE off the frozen folds, feeds nothing to
    # the model. Redirect stdout to v14_bin_diagnostic.txt in the Kaggle working dir.
    print("\n[Diag] D4 evidence re-creation: max_bin rank-preserving AUC table "
          "(whole-train single-column AUC, no fold, no model — re-creates v7b_out.txt)")
    y_diag = train[CFG.TARGET].values
    fd_diag = train['Flight Distance'].values
    fd_levels = int(train['Flight Distance'].nunique())
    fd_levels_pool = int(pd.concat([train['Flight Distance'],
                                   test['Flight Distance']]).nunique())
    fd_levels_orig = int(orig['Flight Distance'].nunique())
    print(f"   Distinct Flight Distance levels (plan said '~3,474 route levels'): "
          f"train = {fd_levels:,} | train+test pool = {fd_levels_pool:,} | original = {fd_levels_orig:,}")

    auc_fd_raw = quantisation_table("Flight Distance (raw, competition labels)",
                                    fd_diag, y_diag, f"{fd_levels:,} (train)")

    te_xfit_diag, te_insamp_diag, fd_key_levels = te_fd_exact_diagnostic(train, y_diag)
    print(f"   `_tm_fd_exact`-style TE column (cross-fitted V2 idiom: "
          f"TargetEncoder binary, smooth=20, inner SKF(5, rs=1042), no outer fold):")
    quantisation_table("_tm_fd_exact (cross-fitted)", te_xfit_diag, y_diag,
                       f"{len(np.unique(te_xfit_diag)):,} distinct values "
                       f"over {fd_key_levels:,} FD keys")
    print(f"   `_tm_fd_exact`-flavour in-sample smoothed group statistic "
          f"(smooth=20 shrink-to-prior; the ~{fd_key_levels:,}-level distribution "
          f"the histogram must bin):")
    quantisation_table("_tm_fd_exact (in-sample smoothed stat)", te_insamp_diag, y_diag,
                       f"{len(np.unique(te_insamp_diag)):,} distinct values")

    print("   FD -> label law shape (8 coarse equal-width buckets — why the arm is demoted):")
    coarse = equal_width_bin(fd_diag, 8)
    lo_f, hi_f = float(np.nanmin(fd_diag)), float(np.nanmax(fd_diag))
    width_f = (hi_f - lo_f + 1.0) / 8.0
    step_tbl = pd.DataFrame({'bucket': coarse, 'y': y_diag}).groupby('bucket')['y'].agg(['mean', 'size'])
    for b, row in step_tbl.iterrows():
        print(f"      bucket {int(b)} (FD < {lo_f + (b + 1) * width_f:.0f} mi): "
              f"rate {row['mean']:.4f} | n {int(row['size']):,}")
    print(f"   D4 consequence: >=99.8 % of FD's own ordering survives at 16 bins -> the only live "
          f"sub-mechanism is quantisation of the {fd_key_levels:,} distinct `_tm_fd_exact` values. "
          f"(k-means ~0.51 figure stays uncited: unordered indices = measurement artifact.)")
    gc.collect()

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
    # [3/5] TRAINING — RUNG LOOP over the pre-declared max_bin ladder,
    #         each rung re-running the frozen V2 5-fold loop unchanged.
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV), "
          f"max_bin ladder {CFG.MAX_BIN_LADDER}...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    rungs = {}   # max_bin -> per-rung record
    for rung_no, max_bin in enumerate(CFG.MAX_BIN_LADDER, start=1):
        rung_params = dict(XGB_PARAMS)
        rung_params['max_bin'] = int(max_bin)   # THE ONLY CONFIG DELTA VS PARENT V2
        tag = f"rung {rung_no}/{len(CFG.MAX_BIN_LADDER)} max_bin={max_bin}"

        print(f"\n   === RUNG max_bin={max_bin} ({rung_no}/{len(CFG.MAX_BIN_LADDER)}) "
              f"— one fitted model per fold, all other params byte-identical to V2 ===")
        # model width = base features + per-fold TE block (40 keys x _tm_/_freq_/_rarity_ = +120 cols).
        n_model_cols = len(FEATURES) + 3 * (len(CATS) + len(RATINGS) + 5
                                            + len(KEY_PAIRS) + len(RATING_PAIRS))
        fit_ok, est_gb, mem_used, mem_total = rung_memory_check(
            rung_params, n_model_cols, len(X) + len(test_X), tag)
        if not fit_ok:
            print(f"      [{tag}] SKIPPED (memory guard): estimate does not fit the T4 budget.")
            rungs[max_bin] = {'skipped': True, 'reason': 'memory guard estimate',
                              'est_gb': est_gb, 'mem_used': mem_used,
                              'mem_total': mem_total, 'fold_scores': [],
                              'best_iters': [], 'prep_times': [], 'train_times': [],
                              'oof': None, 'test': None, 'oof_cv': None,
                              'mem_end_gb': None}
            continue

        oof_probs  = np.zeros(len(y))
        test_probs = np.zeros(len(test_X))
        fold_scores = []
        best_iters  = []
        prep_times  = []
        train_times = []
        rung_start = time.time()

        try:
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
                      f"Val: {X_val.shape} | Prep: {prep_s:.0f}s")

                model = xgb.XGBClassifier(**rung_params)
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
                train_s = fold_time - prep_s
                train_times.append(train_s)
                elapsed   = (time.time() - t0) / 60
                print(f"      AUC: {fold_auc:.5f} | BestIter: {best_iter} | "
                      f"Time: {fold_time:.0f}s (train {train_s:.0f}s) | Total: {elapsed:.1f}min")

                if fold == 0:
                    imp = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_,
                    }).sort_values('importance', ascending=False)
                    print(f"\n      Top-15 feature importances (fold 1):")
                    print(imp.head(15).to_string(index=False))

                del X_train, X_val, X_test, y_train, y_val, model
                gc.collect()
        except (RuntimeError, MemoryError) as e:
            # Hard guard: a rung that OOMs mid-loop is skipped-with-log, never a crashed run.
            print(f"      [{tag}] RUNG ABORTED after {len(fold_scores)}/{CFG.N_FOLDS} folds: "
                  f"{type(e).__name__}: {e} — no partial rung is ever used for selection.")
            del oof_probs, test_probs
            gc.collect()
            rungs[max_bin] = {'skipped': True,
                              'reason': f"{type(e).__name__} during fit",
                              'est_gb': est_gb, 'mem_used': mem_used,
                              'mem_total': mem_total, 'fold_scores': fold_scores,
                              'best_iters': best_iters, 'prep_times': prep_times,
                              'train_times': train_times,
                              'oof': None, 'test': None, 'oof_cv': None,
                              'mem_end_gb': None}
            continue

        oof_cv = auc_score(y.values, oof_probs)
        used_end, total_end = gpu_memory_mib(f"{tag} end")
        print(f"\n   RUNG max_bin={max_bin} — OOF CV (AUC): {oof_cv:.5f} | "
              f"Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
        print(f"      Fold scores: {[f'{s:.5f}' for s in fold_scores]} | Best iters: {best_iters}")
        print(f"      Prep s/fold: {[f'{p:.0f}s' for p in prep_times]} | "
              f"Train s/fold: {[f'{p:.0f}s' for p in train_times]}")
        print(f"      Rung wall time: {(time.time() - rung_start)/60:.1f} min | "
              f"GPU at rung end: used={('%.0f MiB' % used_end) if used_end is not None else 'n/a'} "
              f"total={('%.0f MiB' % total_end) if total_end is not None else 'n/a'}")

        rungs[max_bin] = {'skipped': False, 'reason': '', 'est_gb': est_gb,
                          'mem_used': mem_used, 'mem_total': mem_total,
                          'fold_scores': fold_scores, 'best_iters': best_iters,
                          'prep_times': prep_times, 'train_times': train_times,
                          'oof': oof_probs.copy(), 'test': test_probs.copy(),
                          'oof_cv': oof_cv,
                          'mem_end_gb': (used_end / 1024.0) if used_end is not None else None}

    # Rung selection = CV-side half of the gate (DeLong + seed-2027 offline); never a combination.
    # Baseline rung = the parent's own value (XGB_PARAMS['max_bin'] = 256).
    base_bin = int(XGB_PARAMS['max_bin'])
    completed = {b: r for b, r in rungs.items() if not r['skipped']}
    baseline = completed.get(base_bin)   # parent-equivalent rung (256)
    print("\n   Rung selection (ladder control, runbook §4; never a combination):")
    for b, r in rungs.items():
        if r['skipped']:
            print(f"      max_bin={b:>5}: SKIPPED ({r['reason']})")
            continue
        if b == base_bin:
            print(f"      max_bin={b:>5}: OOF CV {r['oof_cv']:.5f} — baseline rung "
                  f"(parent V2's own value)")
            continue
        if baseline is None:
            print(f"      max_bin={b:>5}: OOF CV {r['oof_cv']:.5f} — baseline rung "
                  f"{base_bin} unavailable (skipped), no comparison possible")
            continue
        folds_up = sum(1 for s, b0 in zip(r['fold_scores'], baseline['fold_scores'])
                       if s > b0)
        d_cv = r['oof_cv'] - baseline['oof_cv']
        passes_cv_half = (d_cv > 0.00005) and (folds_up >= 4)
        print(f"      max_bin={b:>5}: OOF CV {r['oof_cv']:.5f} | dCV vs {base_bin} "
              f"{d_cv:+.6f} | folds improved {folds_up}/{CFG.N_FOLDS} | "
              f"CV half of gate: {'PASS' if passes_cv_half else 'FAIL'}")

    if baseline is None:
        record_bin = min(completed) if completed else None
        print(f"      Baseline rung {base_bin} unavailable -> rung-of-record falls "
              f"back to the lowest completed rung: {record_bin}")
    else:
        record_bin = base_bin
        best_bin = max(completed, key=lambda b: completed[b]['oof_cv'])
        if best_bin != base_bin:
            folds_up = sum(1 for s, b0 in
                           zip(completed[best_bin]['fold_scores'],
                               baseline['fold_scores']) if s > b0)
            if (completed[best_bin]['oof_cv'] - baseline['oof_cv'] > 0.00005) \
                    and folds_up >= 4:
                record_bin = best_bin
                print(f"      CV-side candidate max_bin={best_bin} passes dCV + {folds_up}/5 folds "
                      f"-> rung-of-record PROVISIONALLY (still needs offline DeLong dAUC > +0.00005, "
                      f"z > 3 vs v2_oof + reproduction at RANDOM_SEED={CFG.SEED_REPRO}).")
            else:
                print(f"      CV-side candidate max_bin={best_bin} fails the CV half -> "
                      f"rung-of-record stays at parent value {base_bin} (a tie/null, not an improvement).")

    if record_bin is None:
        raise RuntimeError("Every rung was skipped — no OOF to write. "
                           "Log as a failed version, do not fall back to another model.")

    oof_probs = rungs[record_bin]['oof']
    test_probs = rungs[record_bin]['test']
    oof_cv = rungs[record_bin]['oof_cv']
    fold_scores = rungs[record_bin]['fold_scores']
    best_iters = rungs[record_bin]['best_iters']
    prep_times = rungs[record_bin]['prep_times']
    train_times = rungs[record_bin]['train_times']
    print(f"\n   Rung-of-record max_bin={record_bin}: OOF CV (AUC) {oof_cv:.5f}, "
          f"fold AUCs {[f'{s:.5f}' for s in fold_scores]}, best iters {best_iters}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS — every rung's OOF + the rung-of-record as v14_oof
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    for b, r in rungs.items():
        if r['skipped']:
            print(f"   [SKIP] no OOF for max_bin={b} ({r['reason']})")
            continue
        p = f"{out_dir}/{stem}_maxbin{b}_oof.csv"
        pd.DataFrame({'id': train_id, 'pred': r['oof']}).to_csv(p, index=False)
        print(f"   [SAVED] {p} (id, pred) — rung {b}")

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred) "
          f"= rung-of-record max_bin={record_bin}")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv "
          f"= rung-of-record max_bin={record_bin}")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    for b, r in rungs.items():
        if r['skipped']:
            print(f"   max_bin={b:>5}: SKIPPED — {r['reason']} | est {r['est_gb']:.2f} GB vs 16 GB device "
                  f"| folds completed {len(r['fold_scores'])}/{CFG.N_FOLDS}")
            continue
        mark = "  <- rung-of-record" if b == record_bin else ""
        print(f"   max_bin={b:>5}: OOF CV {r['oof_cv']:.5f} | folds {[f'{s:.5f}' for s in r['fold_scores']]}"
              f" | best iters {r['best_iters']}"
              f" | prep {np.mean(r['prep_times']):.0f}s + fit {np.mean(r['train_times']):.0f}s/fold"
              f" | GPU at end {('%.2f GB' % r['mem_end_gb']) if r['mem_end_gb'] is not None else 'n/a'}{mark}")
    if baseline is not None:
        print(f"dCV vs baseline rung {base_bin}: {oof_cv - baseline['oof_cv']:+.6f}")
    else:
        print(f"dCV vs baseline rung {base_bin}: not computable (baseline skipped)")

    total_time_min = (time.time() - t0_all) / 60
    print(f"Total: {total_time_min:.1f} min")
