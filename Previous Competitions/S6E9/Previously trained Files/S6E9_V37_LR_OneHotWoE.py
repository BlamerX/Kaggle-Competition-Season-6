"""
S6E9 V37 - Logistic regression on a one-hot + WoE design (CPU)
A different function class rather than another GBM: winning stacks keep linear members
because they err differently, and on low-cardinality synthetic data a one-hot/WoE
design is competitive with trees. Sparse CSR design: one-hot of the categoricals and
of the concern x subsidy x range-anxiety cells, ~900 per-fold quantile income bins,
exact-lattice one-hot for commute/age/cars, and 8 smoothed WoE single terms. Every
label-dependent statistic is fitted on the fold's training rows only and unseen levels
become all-zero rows, so per-fold width is constant.
LogisticRegression(l2, C=0.3, lbfgs), class_weight=None as AUC is rank-based.
Outputs: /kaggle/working/oof_v37.csv (id, pred), sub_v37.csv (id, Will_Buy_EV)
Estimate: under 30 min.
"""

import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print("Device: CPU ONLY (no GPU, no cuDF)")

# 1. CONFIG
class CFG:
    VERSION_NAME = "v37"
    EXP_ID = "S6E9_V37_LR_OneHotWoE"
    DEVICE = "CPU"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'

    N_FOLDS = 10
    RANDOM_SEED = 42

    N_INCOME_BINS   = 900
    N_CHARGING_BINS = 24
    WOE_SMOOTH = 25.0

    OH_BLOCK_COLS = [
        'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
        'Subsidy_Available', 'Range_Anxiety_Level', 'ECL_str',
    ]
    OH_KEY_COLS = ['key_ecl_sub', 'key_ecl_ra', 'key_ecl_sub_ra']
    OH_EXACT_COLS = ['commute_tenth', 'age_int', 'cars_int']

    RAW_CATS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
                'Subsidy_Available', 'Range_Anxiety_Level']
    RAW_NUMS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level']

# 2. SEED EVERYTHING
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

LR_PARAMS = {
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'class_weight': None,
    'random_state': CFG.RANDOM_SEED,
}
C_GRID = [0.3, 1.0, 3.0, 10.0]

# 3. METRIC
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

def map_target_to_binary(df):
    """S6E9 stores the target as Yes/No strings: map to 1/0 in place when the
    column is not already numeric (train and original file only — test has no
    target)."""
    target2idx = {'No': 0, 'Yes': 1}
    if not pd.api.types.is_numeric_dtype(df[CFG.TARGET]):
        df[CFG.TARGET] = (df[CFG.TARGET].astype(str).str.strip()
                          .str.title().map(target2idx))
    return df

def fill_medians(df, medians):
    """Fill the known-NaN raw columns from a medians dict. Which rows are
    fitted depends on the CALLER — in this file medians are always labelled-pool
    (comp train + original) only, never test."""
    for c, v in medians.items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
    return df

def derive_key_frames(df):
    """
    Create the target-free code columns the static one-hot blocks consume.
    - ECL_str: the 1-5 concern level as a string category ('-1' = missing)
    - cell keys: ECL x Subsidy, ECL x Range-Anxiety, ECL x Subsidy x RA
    - commute_tenth / age_int / cars_int: exact lattice integer codes
    NaN in commute/age/cars is median-filled by the caller BEFORE this runs.
    """
    df = df.copy()
    ecl = df['Environmental_Concern_Level'].fillna(-1).round().astype(int).astype(str)
    df['ECL_str'] = ecl
    df['key_ecl_sub'] = ecl + '_' + df['Subsidy_Available'].astype(str)
    df['key_ecl_ra'] = ecl + '_' + df['Range_Anxiety_Level'].astype(str)
    df['key_ecl_sub_ra'] = ecl + '_' + df['Subsidy_Available'].astype(str) + '_' + df['Range_Anxiety_Level'].astype(str)
    df['commute_tenth'] = np.rint(df['Daily_Commute_km'].astype('float64') * 10.0).astype(str)
    df['age_int'] = np.rint(df['Age'].astype('float64')).astype(int).astype(str)
    df['cars_int'] = np.rint(df['Number_of_Cars_Owned'].astype('float64')).astype(int).astype(str)
    return df

def build_vocab(codes_list):
    """
    Fixed category vocabulary (sorted) from a list of code Series/arrays.
    Built ONCE from the labelled train+original pool only — never test — so any
    test-only value is 'unseen' and becomes an all-zero row via reindex. This
    is what keeps the matrix width constant across folds and splits: the
    vocabulary, not the data, decides the columns of each block.
    """
    seen = set()
    for c in codes_list:
        seen.update(pd.unique(c))
    return np.array(sorted(map(str, seen)), dtype=object)

def onehot_block(codes, vocab, offset):
    """
    Reindex-then-encode: pd.Categorical(codes, categories=vocab).codes maps every
    value to its FIXED vocabulary slot and gives -1 to unseen values, which are
    simply omitted from the COO index — the row stays all-zero, never NaN. The
    block always has len(vocab) columns, whichever rows go in.
    """
    cat_codes = pd.Categorical(pd.Series(codes).astype(str), categories=vocab).codes
    cat_codes = np.asarray(cat_codes)
    ok = np.flatnonzero(cat_codes >= 0)
    n_cols = len(vocab)
    return sp.coo_matrix(
        (np.ones(len(ok), dtype='float64'),
         (ok.astype('int64'), (cat_codes[ok] + offset).astype('int64'))),
        shape=(len(codes), n_cols)).tocsr()

class FoldQuantileBinner:
    """
    Fixed-width quantile binning whose EDGES are refitted every fold on the
    outer-training rows only (bin edges are label-adjacent statistics too, and
    refitting them per fold is what makes the income block leak-free). The bin
    COUNT is a constructor constant, so the one-hot block built on top of the
    codes has provably constant width across folds even if the edges move.
    """
    def __init__(self, n_bins):
        self.n_bins = int(n_bins)
        self.edges = None

    def fit(self, x):
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        edges = np.quantile(np.asarray(x, dtype='float64'), qs)
        self.edges = np.unique(edges)
        return self

    def transform(self, x):
        idx = np.searchsorted(self.edges, np.asarray(x, dtype='float64'), side='right')
        return np.clip(idx, 0, self.n_bins - 1).astype('int32')

def fit_woe_table(bin_idx, y, smooth):
    """
    Smoothed (shrunk) weight of evidence per bin, fitted on the fold-training
    rows only: woe_k = ln(pos_k/neg_k) with `smooth` pseudo-rows of the fold
    prior added, then centred on the fold-wide log-odds so the global reference
    is 0 and a thin or unseen bin decays toward 0 instead of exploding. This is
    the scoring-card rule that a bin earns its own parameter only when it has
    the counts — the formal version of our 926-bins-beat-13,214-values result.
    """
    bin_idx = np.asarray(bin_idx)
    y = np.asarray(y)
    p0 = float(y.mean())
    df = pd.DataFrame({'k': bin_idx, 'y': y})
    g = df.groupby('k')['y'].agg(['sum', 'count'])
    pos_s = g['sum'].values + smooth * p0
    neg_s = (g['count'].values - g['sum'].values) + smooth * (1.0 - p0)
    woe = np.log(pos_s) - np.log(neg_s)
    woe -= (np.log(p0) - np.log(1.0 - p0))
    return dict(zip(g.index.values.astype('int64'), woe.astype('float64')))

def apply_woe(bin_idx, table, default=0.0):
    """Map fold-fitted WoE onto any rows; unseen bins get the shrunk default 0."""
    return pd.Series(np.asarray(bin_idx)).map(table).fillna(default).to_numpy(dtype='float64')

def woe_codes_simple(frame, col, scale=1.0):
    """Lattice integer codes for the WoE keys: scale first, then round, so the
    0.1 commute step becomes an integer key without float-repr fragility."""
    return np.rint(frame[col].to_numpy(dtype='float64') * scale).astype('int64')

def align_original_schema(orig_df):
    """Align original dataset schema with competition train (drops Buyer_ID)."""
    return orig_df[CFG.RAW_CATS + CFG.RAW_NUMS + [CFG.TARGET]].copy()

def coef_table(design_names, coef):
    """
    Fold-1 coefficient table: the design column names paired with the fitted
    LogisticRegression coef_, sorted by |coef| descending — the house
    importance print for a linear model. For logistic regression the largest
    |coef| columns are the vocabulary cells the fit leans on hardest.
    """
    return (pd.DataFrame({'column': design_names,
                          'coef': coef,
                          '|coef|': np.abs(coef)})
            .sort_values('|coef|', ascending=False))

def fit_fold_bins(train, orig_aligned, tr_idx, otr_idx):
    """
    Refit the three quantile binners (income, charging near home, charging near
    work) on the fold's outer-training rows only. The fitted numeric vectors
    are returned alongside: they are exactly the rows the WoE tables are keyed
    by, so every fold statistic is provably 1:1 with y_fit.
    """
    inc_bin = FoldQuantileBinner(CFG.N_INCOME_BINS)
    ch_bin = FoldQuantileBinner(CFG.N_CHARGING_BINS)
    cw_bin = FoldQuantileBinner(CFG.N_CHARGING_BINS)

    inc_fit = np.concatenate([train['Annual_Income_USD'].to_numpy()[tr_idx],
                              orig_aligned['Annual_Income_USD'].to_numpy()[otr_idx]])
    ch_fit = np.concatenate([train['Charging_Stations_Near_Home'].to_numpy()[tr_idx],
                             orig_aligned['Charging_Stations_Near_Home'].to_numpy()[otr_idx]])
    cw_fit = np.concatenate([train['Charging_Stations_Near_Work'].to_numpy()[tr_idx],
                             orig_aligned['Charging_Stations_Near_Work'].to_numpy()[otr_idx]])
    inc_bin.fit(inc_fit)
    ch_bin.fit(ch_fit)
    cw_bin.fit(cw_fit)

    return inc_bin, ch_bin, cw_bin, inc_fit, ch_fit, cw_fit

def build_bin_blocks(inc_bin, ch_bin, cw_bin, train, test, orig_aligned,
                     tr_idx, va_idx, otr_idx):
    """
    The three bin one-hot blocks for the fold's train / validation / test row
    sets, from the fold-fitted binners. Each block is exactly nb columns wide
    (900 / 24 / 24) whichever rows go in; codes outside this fold's edge
    support clip into the first/last bin, so the widths never move.
    """
    specs = ((inc_bin, 'Annual_Income_USD', CFG.N_INCOME_BINS),
             (ch_bin, 'Charging_Stations_Near_Home', CFG.N_CHARGING_BINS),
             (cw_bin, 'Charging_Stations_Near_Work', CFG.N_CHARGING_BINS))

    def one_set(frames):
        mats = []
        for binner, col, nb in specs:
            codes = np.concatenate([binner.transform(f[col].to_numpy(dtype='float64'))
                                    for f in frames])
            mats.append(onehot_block(pd.Series(codes).astype(str),
                                     np.array([str(b) for b in range(nb)], dtype=object), 0))
        return mats

    tr_bins = one_set([train.iloc[tr_idx], orig_aligned.iloc[otr_idx]])
    va_bins = one_set([train.iloc[va_idx]])
    te_bins = one_set([test])
    return tr_bins, va_bins, te_bins

def fit_fold_woe_tables(inc_bin, ch_bin, cw_bin, inc_fit, ch_fit, cw_fit,
                        train, orig_aligned, tr_idx, otr_idx, y_fit):
    """
    The 8 smoothed WoE tables of the design, fitted on the fold's
    outer-training rows only. Keys: the fold-binned income and charging counts,
    the commute 0.1 lattice, exact age / cars integers, the rounded ECL level
    and the binary subsidy flag.
    """
    fold_pool = pd.concat([train.iloc[tr_idx], orig_aligned.iloc[otr_idx]], ignore_index=True)
    return {
        'income': fit_woe_table(inc_bin.transform(inc_fit), y_fit, CFG.WOE_SMOOTH),
        'commute': fit_woe_table(woe_codes_simple(fold_pool, 'Daily_Commute_km', 10.0),
                                 y_fit, CFG.WOE_SMOOTH),
        'age': fit_woe_table(woe_codes_simple(fold_pool, 'Age', 1.0), y_fit, CFG.WOE_SMOOTH),
        'cars': fit_woe_table(woe_codes_simple(fold_pool, 'Number_of_Cars_Owned', 1.0),
                              y_fit, CFG.WOE_SMOOTH),
        'charg_home': fit_woe_table(ch_bin.transform(ch_fit), y_fit, CFG.WOE_SMOOTH),
        'charg_work': fit_woe_table(cw_bin.transform(cw_fit), y_fit, CFG.WOE_SMOOTH),
        'ecl': fit_woe_table(fold_pool['Environmental_Concern_Level'].to_numpy(dtype='float64')
                             .round().astype('int64'), y_fit, CFG.WOE_SMOOTH),
        'subsidy': fit_woe_table((fold_pool['Subsidy_Available'].to_numpy() == 'Yes')
                                 .astype('int64'), y_fit, CFG.WOE_SMOOTH),
    }

def apply_woe_terms(inc_bin, ch_bin, cw_bin, woe_tables, frame, frame_or=None):
    """The 8 dense smoothed-WoE single terms for any row set: train+orig,
    validation or test, all reading the SAME fold-fitted tables — an unseen key
    maps to 0 by the centring and the fillna default, never to NaN."""
    frames = [frame] + ([frame_or] if frame_or is not None else [])
    cols = [
        apply_woe(np.concatenate([inc_bin.transform(f['Annual_Income_USD'].to_numpy(dtype='float64')) for f in frames]),
                  woe_tables['income']),
        apply_woe(np.concatenate([woe_codes_simple(f, 'Daily_Commute_km', 10.0) for f in frames]),
                  woe_tables['commute']),
        apply_woe(np.concatenate([woe_codes_simple(f, 'Age', 1.0) for f in frames]),
                  woe_tables['age']),
        apply_woe(np.concatenate([woe_codes_simple(f, 'Number_of_Cars_Owned', 1.0) for f in frames]),
                  woe_tables['cars']),
        apply_woe(np.concatenate([ch_bin.transform(f['Charging_Stations_Near_Home'].to_numpy(dtype='float64')) for f in frames]),
                  woe_tables['charg_home']),
        apply_woe(np.concatenate([cw_bin.transform(f['Charging_Stations_Near_Work'].to_numpy(dtype='float64')) for f in frames]),
                  woe_tables['charg_work']),
        apply_woe(np.concatenate([f['Environmental_Concern_Level'].to_numpy(dtype='float64').round().astype('int64') for f in frames]),
                  woe_tables['ecl']),
        apply_woe(np.concatenate([(f['Subsidy_Available'].to_numpy() == 'Yes').astype('int64') for f in frames]),
                  woe_tables['subsidy']),
    ]
    return np.column_stack(cols)

def build_fold_design(tr_idx, va_idx, otr_idx, train, test, orig_aligned,
                      y_fit, STATIC_TRAIN, STATIC_TEST, STATIC_ORIG):
    """
    Assemble the three per-fold CSR matrices (train / validation / test) of the
    one-hot + WoE design: static labelled-pool-vocabulary blocks (sliced), the
    three fold-binned one-hot blocks (fixed widths), and the 8 dense smoothed
    WoE columns. The dense stacks join the sparse ones via sp.csr_matrix(...) —
    lbfgs reads C-ordered CSR float64, the documented fast sparse path.

    The constant-width assertion below is correctness, not diagnostics:
    pool-only vocabularies and fixed bin counts make the width constant by
    construction, and the assert proves the construction held for THIS fold's
    row sets.
    """
    inc_bin, ch_bin, cw_bin, inc_fit, ch_fit, cw_fit = fit_fold_bins(
        train, orig_aligned, tr_idx, otr_idx)

    tr_bins, va_bins, te_bins = build_bin_blocks(
        inc_bin, ch_bin, cw_bin, train, test, orig_aligned, tr_idx, va_idx, otr_idx)

    woe_tables = fit_fold_woe_tables(inc_bin, ch_bin, cw_bin, inc_fit, ch_fit, cw_fit,
                                     train, orig_aligned, tr_idx, otr_idx, y_fit)

    w_tr = apply_woe_terms(inc_bin, ch_bin, cw_bin, woe_tables,
                           train.iloc[tr_idx], orig_aligned.iloc[otr_idx])
    w_va = apply_woe_terms(inc_bin, ch_bin, cw_bin, woe_tables, train.iloc[va_idx])
    w_te = apply_woe_terms(inc_bin, ch_bin, cw_bin, woe_tables, test)

    st_tr = sp.vstack([STATIC_TRAIN[tr_idx], STATIC_ORIG[otr_idx]]).tocsr()
    X_tr = sp.hstack([st_tr] + tr_bins + [sp.csr_matrix(w_tr)]).tocsr()
    X_va = sp.hstack([STATIC_TRAIN[va_idx]] + va_bins +
                     [sp.csr_matrix(w_va)]).tocsr()
    X_te = sp.hstack([STATIC_TEST] + te_bins +
                     [sp.csr_matrix(w_te)]).tocsr()

    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1] == DESIGN_WIDTH, \
        "per-fold matrix width changed — fold-safe vocabulary alignment BROKEN"

    return X_tr, X_va, X_te

# 5. MAIN EXECUTION
if __name__ == "__main__":
    t0_all = time.time()

    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIG_PATH)

    train = map_target_to_binary(train)
    orig = map_target_to_binary(orig)

    train_id = train['id'].copy()
    test_id = test['id'].copy()
    y_orig = orig[CFG.TARGET].copy().reset_index(drop=True)

    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig).reset_index(drop=True)

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")
    print("\n   Class Distribution (train):")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100 * count / len(train):.1f}%)")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")
    print(f"   Pos rate (orig):  {y_orig.mean():.4f}")
    print(f"   Orig share of the training matrix: "
          f"{len(orig_aligned) / (len(train) + len(orig_aligned)):.4%}")
    print(f"   Rows held out per fold: {len(train) // CFG.N_FOLDS:,}")

    print(f"   Categorical columns ({len(CFG.RAW_CATS)}): {CFG.RAW_CATS}")
    print(f"   Numerical columns ({len(CFG.RAW_NUMS)}): {CFG.RAW_NUMS}")

    med_df = pd.concat([train[CFG.RAW_NUMS], orig_aligned[CFG.RAW_NUMS]], ignore_index=True)
    POOL_MEDIANS = med_df.median().to_dict()
    print(f"   Pool medians (labelled rows only): {POOL_MEDIANS}")

    for df_ in (train, test, orig_aligned):
        fill_medians(df_, POOL_MEDIANS)

    y_np = train[CFG.TARGET].values.astype('int64')
    n_train = len(train)

    print("\n[2/5] Building static one-hot design (sparse CSR)...")

    train_k = derive_key_frames(train)
    test_k = derive_key_frames(test)
    orig_k = derive_key_frames(orig_aligned)

    offset = 0
    n_cats_cols = n_keys_cols = n_exact_cols = 0
    blocks_tr, blocks_te, blocks_or = [], [], []
    design_names = []
    for col in CFG.OH_BLOCK_COLS + CFG.OH_KEY_COLS + CFG.OH_EXACT_COLS:
        vocab = build_vocab([train_k[col].astype(str), orig_k[col].astype(str)])
        blocks_tr.append(onehot_block(train_k[col].astype(str), vocab, 0))
        blocks_te.append(onehot_block(test_k[col].astype(str), vocab, 0))
        blocks_or.append(onehot_block(orig_k[col].astype(str), vocab, 0))
        design_names += [f"oh_{col}={v}" for v in vocab]
        offset += len(vocab)
        if col in CFG.OH_BLOCK_COLS:
            n_cats_cols += len(vocab)
        elif col in CFG.OH_KEY_COLS:
            n_keys_cols += len(vocab)
        else:
            n_exact_cols += len(vocab)
        print(f"   Vocabulary {col}: {len(vocab)} levels")
    print(f"   Static one-hot width: {offset} cols "
          f"(cats {n_cats_cols} + cell-keys {n_keys_cols} + exact lattices {n_exact_cols};")
    print(f"   vocabularies labelled-pool-only, reindex maps unseen -> all-zero row)")

    STATIC_TRAIN = sp.hstack(blocks_tr).tocsr()
    STATIC_TEST = sp.hstack(blocks_te).tocsr()
    STATIC_ORIG = sp.hstack(blocks_or).tocsr()
    del blocks_tr, blocks_te, blocks_or, train_k, test_k, orig_k
    gc.collect()

    pool = sp.vstack([STATIC_TRAIN, STATIC_ORIG]).tocsr()
    colsum = np.asarray(pool.sum(axis=0)).ravel()
    n_pool = pool.shape[0]
    keep = np.flatnonzero((colsum > 0) & (colsum < n_pool))
    STATIC_TRAIN = STATIC_TRAIN[:, keep]
    STATIC_TEST = STATIC_TEST[:, keep]
    STATIC_ORIG = STATIC_ORIG[:, keep]
    design_names = [design_names[i] for i in keep]
    pool = None
    gc.collect()

    design_names += [f"oh_incomebin_{b:04d}" for b in range(CFG.N_INCOME_BINS)]
    design_names += [f"oh_chomabin_{b:02d}" for b in range(CFG.N_CHARGING_BINS)]
    design_names += [f"oh_cworkbin_{b:02d}" for b in range(CFG.N_CHARGING_BINS)]
    design_names += [f"woe_{f}" for f in ['income', 'commute', 'age', 'cars',
                                          'charg_home', 'charg_work', 'ecl', 'subsidy']]
    DESIGN_WIDTH = len(design_names)
    print(f"   Design width: {DESIGN_WIDTH} columns (asserted constant per fold in [3/5])")

    print(f"\n[3/5] Training LogisticRegression ({CFG.N_FOLDS}-Fold CV, per-fold orig "
          f"concat, all label-dependent stats fold-training-only)...")

    print(f"   LR_PARAMS: {LR_PARAMS} | C grid {C_GRID}")

    oof_by_c = {c: np.zeros(n_train, dtype='float64') for c in C_GRID}
    test_by_c = {c: np.zeros(len(test), dtype='float64') for c in C_GRID}
    fold_aucs = {c: [] for c in C_GRID}
    n_iters = {c: [] for c in C_GRID}

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    for fold, ((tr_idx, va_idx), (otr_idx, _ova)) in enumerate(zip(kf.split(train), orig_splits)):
        fold_start = time.time()
        print(f"\n   Fold {fold + 1}/{CFG.N_FOLDS}:")

        y_fit = np.concatenate([y_np[tr_idx], y_orig.values[otr_idx]])

        X_tr, X_va, X_te = build_fold_design(
            tr_idx, va_idx, otr_idx, train, test, orig_aligned, y_fit,
            STATIC_TRAIN, STATIC_TEST, STATIC_ORIG)

        first = None
        for c_val in C_GRID:
            lr = LogisticRegression(C=c_val, **LR_PARAMS)
            lr.fit(X_tr, y_fit)
            p_va = lr.predict_proba(X_va)[:, 1]
            oof_by_c[c_val][va_idx] = p_va
            test_by_c[c_val] += lr.predict_proba(X_te)[:, 1] / CFG.N_FOLDS
            fold_auc = auc_score(y_np[va_idx], p_va)
            fold_aucs[c_val].append(fold_auc)
            n_iters[c_val].append(int(lr.n_iter_[0]))
            if first is None:
                first = lr.coef_[0].copy()
            del lr

        print(f"      AUC by C: " + " | ".join(f"C={c}:{auc_score(y_np[va_idx], oof_by_c[c][va_idx]):.5f}"
                                              for c in C_GRID) +
              f" | Time: {(time.time() - fold_start) / 60:.1f} min | "
              f"total {(time.time() - t0_all) / 60:.1f} min")

        if fold == 0:
            print(f"\n      Top-15 coefficients (fold 1, C={C_GRID[0]}, largest |coef|):")
            print(coef_table(design_names, first).head(15).to_string(index=False))
            print()

        del X_tr, X_va, X_te, first
        gc.collect()

    cv_by_c = {c: auc_score(y_np, oof_by_c[c]) for c in C_GRID}
    best_c = max(cv_by_c, key=cv_by_c.get)
    oof_preds, test_preds = oof_by_c[best_c], test_by_c[best_c]
    cv_score = cv_by_c[best_c]
    fold_scores = fold_aucs[best_c]
    print(f"\n   {'=' * 70}")
    for c in C_GRID:
        print(f"   C={c:<5} OOF {cv_by_c[c]:.6f}  fold mean {np.mean(fold_aucs[c]):.6f} "
              f"+/- {np.std(fold_aucs[c]):.6f}  n_iter {min(n_iters[c])}-{max(n_iters[c])}")
    print(f"   chosen C={best_c}: {CFG.N_FOLDS}-Fold OOF CV (AUC) {cv_score:.6f}")

    print("\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_preds})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)  — probabilities")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_preds})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv (id, {CFG.TARGET})  — "
          f"probabilities, range {test_preds.min():.4f} .. {test_preds.max():.4f}")

    print(f"\n{'=' * 80}")
    print(f"[5/5] V37 RESULTS — Logistic Regression, One-Hot + Fold-Safe WoE Sparse Design (CPU)")
    print(f"{'=' * 80}")
    print(f"OOF CV (AUC): {cv_score:.6f}  at C={best_c}")
    print(f"Fold mean: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print(f"Design width: {DESIGN_WIDTH} sparse CSR columns "
          f"({CFG.N_INCOME_BINS} income bins + {CFG.N_CHARGING_BINS * 2} charging bins + 8 WoE terms)")
    print(f"n_iter per fold: min {min(n_iters[best_c])} | max {max(n_iters[best_c])}")
    print(f"Test predictions: min {test_preds.min():.6f} | max {test_preds.max():.6f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
