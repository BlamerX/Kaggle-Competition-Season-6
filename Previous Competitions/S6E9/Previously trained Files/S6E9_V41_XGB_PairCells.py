"""
S6E9 V41 - V40 plus the 15 categorical pair cells (key + lift + novelty), GPU XGBoost

Strategy:
    V40 is the only version since V20 to add honest CV rather than rearrange it
    (+0.000037 over V30, LB 0.94646), and the mechanism was addressable cells: income
    resolutions the exact///100///1000 keys could not express. The same audit of the 66
    target-encoded keys shows 13 of the 15 categorical PAIRS have never been encoded at
    all - only two bigrams exist - while the pair/triple cell family owns every model
    (TE_lift_trigram_cat_100 at 5.2x the #2 feature). This version encodes that gap.

References:
    V30 0.946171. V40 0.946208 (the base: same config plus the window/ladder block).
    V20's artifact features and V19's lift block are the last gains of this type.

V41 Change from V40:
    Additive only, nothing removed. For each of the 15 pairs (i<j) of the six categorical
    originals: a string cell key run through the triple target encoder (45 columns), a
    pair-level generator lift = freq_pool/freq_original (label-free, the same estimator as
    lift_trigram), and a pair-level novelty flag for cells absent from the original 10k.
    Pair keys join TARGET_ENCODE_COLS; pair lift/novelty join the base features.

Parameters:
    XGB lr 0.01, depth 3, max_leaves 8, lossguide, colsample 0.85, subsample 0.740,
    mcw 4.532, alpha 0.752, lambda 0.619, gamma 1.0, max_bin 1024, device cuda,
    8000 rounds / ES 400, KFold(10, shuffle, rs=42), TE auto/10/100 at cv=10.
    Outputs: oof_v41.csv (id, pred), sub_v41.csv (id, Will_Buy_EV). Estimate ~60-85 min.

Dataset Structure:
    train.csv 668,665 / test.csv 286,571 / original 10,000 rows concat'd per fold under its
    own KFold(10, rs=42); No/Yes -> 0/1. Pair cells are built on train+test+orig together so
    the pool frequency uses the full 955,236-row pool, as every other lift feature does.
"""

import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"xgboost version: {xgb.__version__}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# 1. CONFIG
class CFG:
    VERSION_NAME = "v41"
    EXP_ID = "S6E9_V41_XGB_PairCells"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'

    N_FOLDS = 10
    RANDOM_SEED = 42

    NUM_ROUND = 8000
    ES_ROUNDS = 400

    LIFT_CATS = ['Environmental_Concern_Level', 'Range_Anxiety_Level', 'Subsidy_Available',
                 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible']

# View-A encoding-block constants (module level, used inside the fold loop)
WIN_INC  = [2, 5, 10, 25, 50, 200]
WIN_KM   = [1, 3, 10]
WIN_GRP  = [5, 25]
WIN_PRIOR = 10.0
TE_SMOOTHS = [('auto', 'auto'), (10.0, '10'), (100.0, '100')]

# 2. SEED EVERYTHING
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'device': 'cuda',
    'seed': CFG.RANDOM_SEED,
    'nthread': -1,
    'verbosity': 0,
    'learning_rate': 0.01,
    'max_depth': 3,
    'max_leaves': 8,
    'grow_policy': 'lossguide',
    'min_child_weight': 4.532387806880492,
    'subsample': 0.7400402414525654,
    'colsample_bytree': 0.85,
    'alpha': 0.7523885021652775,
    'reg_lambda': 0.6189949691705282,
    'gamma': 1.0,
    'max_bin': 1024,
}

# 3. METRIC
def auc_score(y_true, y_probs):
    """ROC AUC for binary classification."""
    return roc_auc_score(y_true, y_probs)

# 4. FEATURE ENGINEERING
def add_digit_features(df, num_cols):
    """Digit features from a single int64 rescale - the FIXED version (V19 onwards)."""
    df = df.copy()

    for c in num_cols:
        scaled = np.rint(df[c].fillna(0).astype('float64') * 1e4).astype('int64')
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = ((scaled // np.int64(10 ** (k + 4))) % 10).astype('int8')

    return df

def add_engineered_features(df):
    """EDA-validated engineered features: 4 interactions, 3 logs, 4 hard-edge flags."""
    df = df.copy()

    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')

    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

    df['_ECL_x_Subsidy'] = (
        df['Environmental_Concern_Level'] * df['_Subsidy_bin']
    ).astype('float32')
    df['_ECL_x_RangeAnxiety'] = (
        df['Environmental_Concern_Level'] * (3 - df['_RA_code'])
    ).astype('float32')
    df['_Income_x_Subsidy'] = (
        df['Annual_Income_USD'] * df['_Subsidy_bin']
    ).astype('float32')
    df['_Charging_Total'] = (
        df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    ).astype('float32')

    df['_log_Income'] = np.log1p(df['Annual_Income_USD'].clip(lower=0)).astype('float32')
    df['_log_Commute'] = np.log1p(df['Daily_Commute_km'].clip(lower=0)).astype('float32')
    df['_log_Charging_Total'] = np.log1p(df['_Charging_Total'].clip(lower=0)).astype('float32')

    df['_high_income'] = (df['Annual_Income_USD'] > 170537).astype('int8')
    df['_range_anxiety_high'] = (df['Range_Anxiety_Level'] == 'High').astype('int8')
    df['_ecl_max'] = (df['Environmental_Concern_Level'] == 5).astype('int8')
    df['_ev_recipe'] = (
        (df['Environmental_Concern_Level'] == 5) &
        (df['Range_Anxiety_Level'] == 'Low')
    ).astype('int8')

    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])

    return df

def add_synthetic_artifact_flags(df):
    """CTGAN anomaly flags mapped by starkhushi / Aryan Kaisth (V10 set)."""
    df = df.copy()
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')
    df['is_dead_zone'] = ((df['Annual_Income_USD'] >= 38000.0) &
                          (df['Annual_Income_USD'] <= 42000.0)).astype('int8')
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
    return df

def add_structural_flags(df):
    """Exact structural bounds decoded from the generator recipe (buy bound, dead zone)."""
    df = df.copy()
    df['_below_buy_bound'] = (df['Annual_Income_USD'] < 41667).astype('int8')
    df['_dead_zone_exact'] = ((df['Annual_Income_USD'] >= 31004) &
                              (df['Annual_Income_USD'] <= 41970)).astype('int8')
    return df

def add_smooth_keys(df):
    """Markus's multi-scale 'Smooth Keys' (binned numerics as strings)."""
    df = df.copy()
    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)
    return df

def add_lift_features(train_df, test_df, orig_df):
    """Generator-lift features, lift = freq_pool / freq_orig; never touches y."""
    print("   Adding generator-lift features (pool freq / orig freq, target-free)...")
    new_cols = []

    def _lift(keys_tr, keys_te, keys_og, col_name):
        pool = pd.concat([keys_tr, keys_te], ignore_index=True)
        comp_freq = pool.value_counts(normalize=True)
        orig_freq = keys_og.value_counts(normalize=True)
        for df_, keys in ((train_df, keys_tr), (test_df, keys_te), (orig_df, keys_og)):
            raw = keys.map(comp_freq).fillna(0.0) / keys.map(orig_freq)
            df_[col_name] = raw.fillna(0.0).astype('float32')
        new_cols.append(col_name)

    _lift(train_df['income_exact_int'], test_df['income_exact_int'],
          orig_df['income_exact_int'], 'lift_inc_exact')
    _lift(train_df['income100_floor'], test_df['income100_floor'],
          orig_df['income100_floor'], 'lift_income100')
    _lift(train_df['commute_integer'], test_df['commute_integer'],
          orig_df['commute_integer'], 'lift_commute')
    for c in CFG.LIFT_CATS:
        _lift(train_df[c].astype(str), test_df[c].astype(str),
              orig_df[c].astype(str), f'lift_{c}')

    for df_ in (train_df, test_df, orig_df):
        df_['_tri_key'] = (df_['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
                           + '_' + df_['Subsidy_Available'].astype(str)
                           + '_' + df_['Range_Anxiety_Level'].astype(str))
    _lift(train_df['_tri_key'], test_df['_tri_key'], orig_df['_tri_key'], 'lift_trigram')

    orig_inc_set = set(orig_df['income_exact_int'].unique())
    train_df['novel_inc'] = (~train_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    test_df['novel_inc']  = (~test_df['income_exact_int'].isin(orig_inc_set)).astype('int8')
    orig_df['novel_inc']  = np.int8(0)
    new_cols.append('novel_inc')

    for df_ in (train_df, test_df, orig_df):
        df_.drop(columns=['_tri_key'], errors='ignore', inplace=True)

    print(f"      Added {len(new_cols)} lift/novelty columns: {new_cols}")
    return train_df, test_df, orig_df, new_cols

def add_pair_cells(train_df, test_df, orig_df, cat_cols):
    """All 15 categorical pair cells: TE'd key + pool/orig frequency lift + novelty flag."""
    print("   Adding PAIR-CELL keys, pair lift and pair novelty (15 categorical pairs)...")
    keys, extra = [], []

    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            name = f'pair_{a}__{b}'
            keys.append(name)
            for df_ in (train_df, test_df, orig_df):
                df_[name] = df_[a].astype(str) + '_' + df_[b].astype(str)

            pool = pd.concat([train_df[name], test_df[name]], ignore_index=True)
            comp_freq = pool.value_counts(normalize=True)
            orig_freq = orig_df[name].value_counts(normalize=True)
            orig_set = set(orig_freq.index)
            for df_ in (train_df, test_df):
                k = df_[name]
                df_[f'lift_{name}'] = (k.map(comp_freq).fillna(0.0) / k.map(orig_freq)).fillna(0.0).astype('float32')
                df_[f'novel_{name}'] = (~k.isin(orig_set)).astype('int8')
            orig_df[f'lift_{name}'] = np.float32(0.0)
            orig_df[f'novel_{name}'] = np.int8(0)
            extra += [f'lift_{name}', f'novel_{name}']

    print(f"      Added {len(keys)} pair keys (triple-TE'd) and {len(extra)} pair lift/novelty columns")
    return train_df, test_df, orig_df, keys, extra

def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
    """Map the original dataset's per-value target means onto train and test."""
    orig_global_mean = orig_df[target].mean()
    orig_stats = {}
    for col in cat_cols + num_cols:
        if col in orig_df.columns:
            orig_stats[col] = orig_df.groupby(col, observed=False)[target].mean()

    for col, stats in orig_stats.items():
        map_name = f"{col}_org_mean"
        train_df[map_name] = train_df[col].map(stats).fillna(orig_global_mean).astype('float32')
        test_df[map_name]  = test_df[col].map(stats).fillna(orig_global_mean).astype('float32')

    return train_df, test_df

def add_numeric_as_string(df, num_cols):
    """Numeric-as-string columns so micro-values get their own TE category."""
    df = df.copy()
    num_to_cat_cols = []
    for col in num_cols:
        if col not in df.columns:
            continue
        cat_name = f"{col}_cat"
        df[cat_name] = df[col].fillna('NaN').astype(str)
        num_to_cat_cols.append(cat_name)
    return df, num_to_cat_cols

def add_frequency_encoding(train_df, test_df, cols_to_encode):
    """Frequency encoding computed on the train+test pool (no target used)."""
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df

def add_targeted_bigrams(train_df, test_df):
    """The two missing bigrams V10 found (V10 = 0.94636 LB): ECL_bin x RA, ECL_bin x Subsidy."""
    print("   Adding TARGETED bigram interactions (ECL_bin x RA, ECL_bin x Subsidy)...")

    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    bigram_name_1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bigram_name_1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bigram_name_1]  = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)

    bigram_name_2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bigram_name_2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bigram_name_2]  = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)

    bigram_cols = [bigram_name_1, bigram_name_2]
    return train_df, test_df, bigram_cols

def add_selective_groupby(train_df, test_df):
    """Selective groupby deviations (V10's finding); pool stats, never the target."""
    print("   Adding SELECTIVE groupby deviation features (ECL_bin, income_bin groups)...")

    if 'ECL_bin' not in train_df.columns:
        for df in [train_df, test_df]:
            df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)

    for df in [train_df, test_df]:
        df['income_bin'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=5, labels=['1', '2', '3', '4', '5']).astype(str)

    combined = pd.concat([train_df, test_df], axis=0)
    group_cols = ['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level']
    agg_num_cols = ['Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']

    new_features = []
    for group_col in group_cols:
        for num_col in agg_num_cols:
            if group_col == num_col or group_col not in combined.columns:
                continue
            grp_mean = combined.groupby(group_col, observed=False)[num_col].mean()
            grp_std  = combined.groupby(group_col, observed=False)[num_col].std().fillna(0)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            train_df[dev_name] = ((train_df[num_col] - train_df[group_col].map(grp_mean)) /
                                  (train_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            test_df[dev_name]  = ((test_df[num_col] - test_df[group_col].map(grp_mean)) /
                                  (test_df[group_col].map(grp_std) + 1e-6)).astype('float32')
            new_features.append(dev_name)

    print(f"      Added {len(new_features)} SELECTIVE groupby deviation features")
    return train_df, test_df

def drop_redundant_features(train_df, test_df, target):
    """Drop constant columns and columns perfectly correlated (|r| = 1.0)."""
    eval_cols = [c for c in train_df.columns
                 if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]
    to_drop_corr = []
    if len(eval_cols) > 1:
        corr_matrix = train_df[eval_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
        del corr_matrix, upper_tri
        gc.collect()

    to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + \
                    [c for c in test_df.columns if test_df[c].nunique() == 1]

    DROP = set(to_drop_corr).union(set(to_drop_const))
    DROP = [c for c in DROP if c not in ['id', target]]

    if len(DROP) > 0:
        print(f"   Dropping {len(DROP)} redundant/constant features")
        train_df = train_df.drop(columns=DROP, errors='ignore')
        test_df  = test_df.drop(columns=DROP, errors='ignore')
    else:
        print("   No redundant features found.")

    return train_df, test_df, DROP

def align_original_schema(orig_df):
    """Align original dataset schema with competition train (drops Buyer_ID)."""
    keep_cols = [
        'Age', 'Annual_Income_USD', 'Daily_Commute_km',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level',
        'Gender', 'City_Type', 'Current_Car_Type',
        'Home_Charging_Possible', 'Subsidy_Available',
        'Range_Anxiety_Level',
    ]
    return orig_df[keep_cols].copy()

# 4b. VIEW-A ENCODING BLOCK (V40)
LADDER_COLS = ['ladder_inc10', 'ladder_inc50', 'ladder_inc500', 'ladder_inc5000',
               'ladder_km5', 'ladder_km50']

def add_quantisation_ladder(df):
    """View-A quantisation keys as strings, TE'd at all three smoothings."""
    inc = np.rint(df['Annual_Income_USD'].astype('float64')).astype('int64')
    km = np.rint(df['Daily_Commute_km'].astype('float64')).astype('int64')
    for name, v, d in [('inc10', inc, 10), ('inc50', inc, 50), ('inc500', inc, 500),
                       ('inc5000', inc, 5000), ('km5', km, 5), ('km50', km, 50)]:
        df[f'ladder_{name}'] = (v // d).astype(str)
    return df

def window_rates(v_fit, y_fit, v_query, widths, prior, gm):
    """Centred-window smoothed target rate: histogram + prefix-sum range scan."""
    lo, hi = int(min(v_fit.min(), v_query.min())), int(max(v_fit.max(), v_query.max()))
    n = hi - lo + 1
    cnt = np.bincount(v_fit - lo, minlength=n).astype(np.float64)
    s   = np.bincount(v_fit - lo, weights=y_fit, minlength=n)
    ccnt = np.concatenate([[0.0], np.cumsum(cnt)]); cs = np.concatenate([[0.0], np.cumsum(s)])
    q = v_query - lo; out = []
    for w in widths:
        a = np.clip(q - w, 0, n); b = np.clip(q + w + 1, 0, n)
        out.append((((cs[b]-cs[a]) + prior*gm) / ((ccnt[b]-ccnt[a]) + prior)).astype("float32"))
    return np.stack(out, 1)

def group_window_scan(g_a, inc_a, y_a, g_q, inc_q, gm):
    """Income windows restricted to the City_Type x Current_Car_Type group."""
    out = np.full((len(inc_q), len(WIN_GRP)), gm, np.float32)
    for g in np.unique(g_q):
        ma, mq = g_a == g, g_q == g
        if ma.sum(): out[mq] = window_rates(inc_a[ma], y_a[ma], inc_q[mq], WIN_GRP, WIN_PRIOR, gm)
    return out

def crossfit_windows(key_fit, y_fit, key_val, key_test, widths, prior, gm, prefix):
    """One window family: inner StratifiedKFold(5) on fit rows, full stats for val/test."""
    fit_cols = np.zeros((len(key_fit), len(widths)), np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.RANDOM_SEED)
    for tr_i, q_i in skf.split(key_fit, y_fit):
        fit_cols[q_i] = window_rates(key_fit[tr_i], y_fit[tr_i], key_fit[q_i], widths, prior, gm)
    val_cols = window_rates(key_fit, y_fit, key_val, widths, prior, gm)
    test_cols = window_rates(key_fit, y_fit, key_test, widths, prior, gm)
    names = [f'win_{prefix}_{w}' for w in widths]
    return {nm: (fit_cols[:, j], val_cols[:, j], test_cols[:, j]) for j, nm in enumerate(names)}

def crossfit_group_windows(g_fit, inc_fit, y_fit, g_val, inc_val, g_test, inc_test, gm):
    """Group-restricted income windows under the same cross-fitting discipline."""
    fit_cols = np.zeros((len(inc_fit), len(WIN_GRP)), np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.RANDOM_SEED)
    for tr_i, q_i in skf.split(g_fit, y_fit):
        fit_cols[q_i] = group_window_scan(g_fit[tr_i], inc_fit[tr_i], y_fit[tr_i], g_fit[q_i], inc_fit[q_i], gm)
    val_cols = group_window_scan(g_fit, inc_fit, y_fit, g_val, inc_val, gm)
    test_cols = group_window_scan(g_fit, inc_fit, y_fit, g_test, inc_test, gm)
    names = [f'win_grp_inc_{w}' for w in WIN_GRP]
    return {nm: (fit_cols[:, j], val_cols[:, j], test_cols[:, j]) for j, nm in enumerate(names)}

# 5. MAIN EXECUTION
if __name__ == "__main__":
    t0_all = time.time()

    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    if not pd.api.types.is_numeric_dtype(train[CFG.TARGET]):
        train[CFG.TARGET] = (train[CFG.TARGET].astype(str).str.strip()
                             .str.title().map(target2idx))
    if not pd.api.types.is_numeric_dtype(orig[CFG.TARGET]):
        orig[CFG.TARGET] = (orig[CFG.TARGET].astype(str).str.strip()
                            .str.title().map(target2idx))

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy().reset_index(drop=True)

    med_inc, med_km = train['Annual_Income_USD'].median(), train['Daily_Commute_km'].median()
    def int_key(df, col, scale, med):
        """Integer scan key: raw dollars (scale 1) or tenths of a km (scale 10)."""
        return np.rint(df[col].fillna(med).astype('float64') * scale).astype('int64').values
    def group_key(df):
        return df['City_Type'].astype(str) + '_' + df['Current_Car_Type'].astype(str)

    inc_tr, inc_te, inc_or = (int_key(d, 'Annual_Income_USD', 1.0, med_inc) for d in (train, test, orig))
    km_tr, km_te, km_or = (int_key(d, 'Daily_Commute_km', 10.0, med_km) for d in (train, test, orig))
    g_codes = pd.factorize(pd.concat([group_key(train), group_key(test), group_key(orig)],
                                     ignore_index=True))[0].astype('int32')
    grp_tr, grp_te, grp_or = g_codes[:len(train)], g_codes[len(train):len(train)+len(test)], g_codes[len(train)+len(test):]

    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig).reset_index(drop=True)

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape:  {test.shape}")
    print(f"   Orig shape:  {orig_aligned.shape}")

    CATS = [c for c in test.columns if train[c].dtype == object or train[c].dtype.name == 'str']
    NUMS = [c for c in test.columns if c not in CATS]

    print(f"   Categorical columns ({len(CATS)}): {CATS}")
    print(f"   Numerical columns ({len(NUMS)}): {NUMS}")

    print(f"\n   Target mapping: {target2idx}")
    print("\n   Class Distribution (train):")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")
    print(f"   Pos rate (train): {train[CFG.TARGET].mean():.4f}")
    print(f"   Pos rate (orig):  {y_orig.mean():.4f}")
    print(f"   Orig share of the training matrix: {len(orig_aligned)/(len(train)+len(orig_aligned)):.4%}")
    print(f"   Rows held out per fold: {len(train)//CFG.N_FOLDS:,} "
          f"(V19's 5-fold held out {len(train)//5:,})")
    print(f"   Missing in orig: {int(orig_aligned.isna().sum().sum())} (XGBoost handles NaN natively)")

    print("\n[2/5] Feature Engineering (V34 pipeline + view-A encoding block)...")

    print("   Adding FIXED digit features...")
    train = add_digit_features(train, NUMS)
    test  = add_digit_features(test,  NUMS)
    orig_aligned = add_digit_features(orig_aligned, NUMS)

    print("   Adding engineered interactions + flags...")
    train = add_engineered_features(train)
    test  = add_engineered_features(test)
    orig_aligned = add_engineered_features(orig_aligned)

    print("   Adding synthetic-artifact flags...")
    train = add_synthetic_artifact_flags(train)
    test  = add_synthetic_artifact_flags(test)
    orig_aligned = add_synthetic_artifact_flags(orig_aligned)

    print("   Adding STRUCTURAL flags (buy bound 41,667 + exact dead zone)...")
    train = add_structural_flags(train)
    test  = add_structural_flags(test)
    orig_aligned = add_structural_flags(orig_aligned)

    print("   Adding multi-scale smooth keys + view-A quantisation ladder...")
    train = add_smooth_keys(add_quantisation_ladder(train))
    test  = add_smooth_keys(add_quantisation_ladder(test))
    orig_aligned = add_smooth_keys(add_quantisation_ladder(orig_aligned))

    print("   Adding original dataset target means...")
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean
            orig_aligned[org_mean_name] = orig_aligned[org_mean_name].astype('float32')

    train, test, orig_aligned, lift_cols = add_lift_features(train, test, orig_aligned)
    train, test, orig_aligned, pair_keys, pair_extra = add_pair_cells(train, test, orig_aligned, CATS)

    print("   Converting numerics to string categories...")
    all_numeric_cols = [c for c in train.columns
                        if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c])
                        and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
                          '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    all_numeric_to_str = all_numeric_cols + engineered_numeric

    train, train_num_cat_cols = add_numeric_as_string(train, all_numeric_to_str)
    test, _ = add_numeric_as_string(test, all_numeric_to_str)
    orig_aligned, _ = add_numeric_as_string(orig_aligned, all_numeric_to_str)
    print(f"   Created {len(train_num_cat_cols)} numeric-as-string columns")

    print("   Adding frequency encoding on all cat + num-as-string cols...")
    freq_target_cols = CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ]
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns:
            orig_aligned[f"{col}_fe"] = 0.0

    train, test, bigram_cols = add_targeted_bigrams(train, test)
    orig_aligned['ECL_bin'] = orig_aligned['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bigram_name in bigram_cols:
        if 'ECL_bin_x_RangeAnxiety' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'ECL_bin_x_Subsidy' in bigram_name:
            orig_aligned[bigram_name] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test = add_selective_groupby(train, test)
    orig_aligned['income_bin'] = pd.cut(
        orig_aligned['Annual_Income_USD'].fillna(orig_aligned['Annual_Income_USD'].median()),
        bins=5, labels=['1', '2', '3', '4', '5']).astype(str)
    combined_for_grp = pd.concat([
        train[['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level',
               'Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']],
        test[['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level',
              'Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']]],
        axis=0)
    group_cols = ['ECL_bin', 'income_bin', 'Subsidy_Available', 'Range_Anxiety_Level']
    agg_num_cols = ['Annual_Income_USD', 'Environmental_Concern_Level', 'Daily_Commute_km']
    for group_col in group_cols:
        for num_col in agg_num_cols:
            if group_col == num_col or group_col not in combined_for_grp.columns:
                continue
            grp_mean = combined_for_grp.groupby(group_col, observed=False)[num_col].mean()
            grp_std  = combined_for_grp.groupby(group_col, observed=False)[num_col].std().fillna(0)
            dev_name = f'grp_{group_col}_{num_col}_dev'
            orig_aligned[dev_name] = ((orig_aligned[num_col] - orig_aligned[group_col].map(grp_mean)) /
                                      (orig_aligned[group_col].map(grp_std) + 1e-6)).astype('float32')

    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin', 'income_bin'], errors='ignore', inplace=True)

    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + [
        'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'
    ] + bigram_cols + LADDER_COLS + pair_keys) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"   View-A encodings per fold: {len(WIN_INC)} income-window + {len(WIN_KM)} commute-window + "
          f"{len(WIN_GRP)} group-window cols, {len(LADDER_COLS) * len(TE_SMOOTHS)} ladder TE cols = "
          f"{len(WIN_INC) + len(WIN_KM) + len(WIN_GRP) + len(LADDER_COLS) * len(TE_SMOOTHS)} new cols")
    print(f"   Pair cells: {len(pair_keys)} keys -> {len(pair_keys) * len(TE_SMOOTHS)} TE cols, "
          f"plus {len(pair_extra)} pair lift/novelty base cols")

    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, GPU, orig concat + Triple TE + windows)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()
    y_np   = y.values

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    best_iters = []
    fold_scores = []
    n_te = 0
    cols_used = 0

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        inc_a = np.concatenate([inc_tr[train_idx], inc_or[or_train_idx]])
        km_a  = np.concatenate([km_tr[train_idx], km_or[or_train_idx]])
        grp_a = np.concatenate([grp_tr[train_idx], grp_or[or_train_idx]])
        y_fit = y_train.to_numpy(dtype=np.float64); gm = float(y_fit.mean())
        window_blocks = {}
        window_blocks.update(crossfit_windows(inc_a, y_fit, inc_tr[val_idx], inc_te, WIN_INC, WIN_PRIOR, gm, 'inc'))
        window_blocks.update(crossfit_windows(km_a, y_fit, km_tr[val_idx], km_te, WIN_KM, WIN_PRIOR, gm, 'km'))
        window_blocks.update(crossfit_group_windows(grp_a, inc_a, y_fit, grp_tr[val_idx], inc_tr[val_idx], grp_te, inc_te, gm))
        for nm, (c_fit, c_val, c_test) in window_blocks.items():
            X_train[nm], X_val[nm], X_test_fold[nm] = c_fit, c_val, c_test

        te_feature_names = []
        for smooth_val, smooth_name in TE_SMOOTHS:
            te = TargetEncoder(target_type='binary', smooth=smooth_val,
                               cv=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
            X_train_enc = te.fit_transform(X_train[TARGET_ENCODE_COLS], y_train).astype('float32')
            X_val_enc   = te.transform(X_val[TARGET_ENCODE_COLS]).astype('float32')
            X_test_enc  = te.transform(X_test_fold[TARGET_ENCODE_COLS]).astype('float32')

            for i, col in enumerate(TARGET_ENCODE_COLS):
                te_name = f"TE_{col}_{smooth_name}"
                X_train[te_name] = X_train_enc[:, i]
                X_val[te_name]   = X_val_enc[:, i]
                X_test_fold[te_name] = X_test_enc[:, i]
                if te_name not in te_feature_names:
                    te_feature_names.append(te_name)

        n_te = len(te_feature_names)

        cols_to_drop_after_te = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop_after_te, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop_after_te, errors='ignore')

        string_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        if string_cols:
            X_train = X_train.drop(columns=string_cols)
            X_val   = X_val.drop(columns=string_cols)
            X_test_fold = X_test_fold.drop(columns=string_cols)

        cols_fold = list(X_train.columns)
        cols_used = len(cols_fold)
        if fold == 0:
            print(f"      Feature count: {len(cols_fold)} "
                  f"({n_te} Triple TE of {len(TARGET_ENCODE_COLS)} cols, "
                  f"{len(window_blocks)} window cols) | "
                  f"train rows {len(train_idx):,} comp + {len(or_train_idx):,} orig")

        dtrain = xgb.DMatrix(X_train[cols_fold], label=y_train, feature_names=cols_fold)
        dval   = xgb.DMatrix(X_val[cols_fold], label=y_val, feature_names=cols_fold)
        dtest  = xgb.DMatrix(X_test_fold[cols_fold], feature_names=cols_fold)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=CFG.NUM_ROUND,
                            evals=[(dval, 'valid')], verbose_eval=False,
                            callbacks=[xgb.callback.EarlyStopping(rounds=CFG.ES_ROUNDS,
                                                                  save_best=True)])
        best_iter = booster.num_boosted_rounds()

        val_probs = booster.predict(dval)
        oof_probs[val_idx] = val_probs
        test_probs += booster.predict(dtest) / CFG.N_FOLDS

        best_iters.append(best_iter)
        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        print(f"      AUC: {fold_auc:.5f} | BestIter {best_iter} | "
              f"Time {(time.time()-fold_start)/60:.1f} min | Total {(time.time()-t0_all)/60:.1f} min")

        if fold == 0:
            try:
                imp = booster.get_score(importance_type='gain')
                top15 = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
                print("      Fold-1 top-15 gain importances:")
                for _name, _score in top15:
                    print(f"        {_name:<46s} {_score:.1f}")
            except Exception as _e:
                print(f"      (importance diagnostics skipped: {_e})")

        del dtrain, dval, dtest, booster, X_train, X_val, X_test_fold
        gc.collect()

    cv_score = auc_score(y_np, oof_probs)
    print(f"\n   {'='*70}")
    print(f"   {CFG.N_FOLDS}-Fold OOF CV (AUC): {cv_score:.6f}")
    print(f"   Fold mean {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print(f"   Best iterations: {best_iters}")

    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)

    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/oof_{CFG.VERSION_NAME}.csv (id, pred)  — probabilities")

    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] {out_dir}/sub_{CFG.VERSION_NAME}.csv  — probabilities, "
          f"range {test_probs.min():.4f} .. {test_probs.max():.4f}")

    print(f"\n{'='*80}")
    print(f"[5/5] V41 RESULTS — Single Model, {CFG.N_FOLDS}-Fold XGBoost (GPU, view-A + pair cells)")
    print(f"{'='*80}")
    try:
        print(f"OOF CV (AUC): {cv_score:.6f}")
        print(f"Fold mean {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
        print(f"Best iterations: {best_iters}")
        print(f"Columns used: {cols_used} ({len(FEATURES)} base + {n_te} Triple TE + {len(window_blocks)} window cols)")
        print(f"Test predictions: min {test_probs.min():.6f} | max {test_probs.max():.6f}")
        print("\nReference points:")
        print(f"  V40 0.946208, identical config WITHOUT the pair cells -> "
              f"the pair-cell effect: {cv_score - 0.946208:+.6f}")
        print(f"  V30 0.946171, before any of the new encodings -> "
              f"cumulative encoding effect: {cv_score - 0.946171:+.6f}")
        print(f"  V31 0.946224, best honest single before this -> delta: {cv_score - 0.946224:+.6f}")
        print("GATE: +0.00005 over V40 is the noise floor; below it the pair-cell axis is")
        print("      closed and no further feature work is scheduled.")
    except Exception as _e:
        print(f"   (results diagnostics skipped: {_e}) — artefacts were already saved")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*80}")
