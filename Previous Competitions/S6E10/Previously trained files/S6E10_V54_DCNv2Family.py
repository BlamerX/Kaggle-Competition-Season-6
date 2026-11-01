"""
S6E10 V54 - DCN-V2 on V18's frame: the estimator family is the axis, and the net is OUR OWN code - GPU (cuda)
Axis: ONE change vs V18 - the fitted model becomes DCN-V2 (low-rank MoE cross network + parallel deep MLP, Hadamard
      crosses). V18's 289-column frame (teacher, twins, route profile, 13 aux-EV, per-fold TE/freq/rarity) and its
      frozen StratifiedKFold(5, shuffle, seed 42) driver are untouched, including the per-fold vocabularies and the
      fit-fold-only median fill.
Model provenance: the architecture is OUR OWN hand-written implementation, copied from S6E9 V13:115-211 (itself
      verified down the S6E4 V39 -> S6E2 V31 lineage), with its hyperparameters unchanged: embedding_dim 16,
      num_cross_layers 4, low_rank 64, num_experts 4, dnn_hidden [256,128], dropout 0.2, AdamW lr 1e-3,
      weight_decay 1e-5, batch 4096, inference chunk 8192, max 50 epochs, early-stopping patience 10 on val AUC
      with best-checkpoint restore. Deliberate deviations, both declared: (i) StratifiedKFold, not the KFold V13
      used, because the house partition is frozen and every other arm is scored under it; (ii) pos_weight is
      computed from the FIT FOLD, not from all training rows as V13 did, because the campaign takes every
      label-derived statistic from the fit fold only.
Why: the audit of this episode's 52 arms found no DCN run, and the V51 appendix's own conclusion is that only a
      different FAMILY can move the board. Prior-episode evidence, honestly read: S6E9 V13's DCN-V2 ran only 14.4
      min yet scored 0.94490 OOF, BELOW that episode's TabM (0.94585) and its XGB control (0.946076), and its
      ledger says it "did not outperform trees or TabM"; S6E4 V39 called it that episode's "Best NN" (0.96986).
      So the expectation is a cheap decorrelated LEG, not a solo winner - which is what the composition arm
      (V55) measures, not this arm's CV.
Parent: Previously trained files/Archieve/S6E10_V18_RealMLPFusionEV.py (its CV/LB are in Memory/public_scores.md).
Standalone (owner law): no saved OOF is a feature or a target; v18_oof enters only offline, after the run, for the gate.
Gate (pre-declared): as a single model, CV >= 0.961337 (V18 + one 0.00005 ruler) AND paired DeLong z > 3 vs v18_oof
      AND >=4/5 folds improved, then reproduce at SEED_REPRO 2027. In-run checks, each calibrated to fire: the
      architecture is read back from the CONSTRUCTED net's structure (embedding count, cross-layer count, output
      width) before fold 1 finishes; a learning floor aborts fold 1 if its val AUC < 0.90 (a degenerate net or a
      misaligned label, not a result); and a fold-1 wall-time canary at 45 min aborts before the cost is times five.
V55 composition rule, fixed before this runs: ship the equal-weight logit-mean of v27+v36+v44+v54 only if v54's solo
      CV >= 0.9600 AND that 4-set beats V51's recorded CV; otherwise the trio stands. Correlation with the members is
      reported as mechanism, never used as the admission test. (The blend arm that held this number before was V54;
      the owner reassigned V54 to DCN, so the composition moves to V55 with its gate unchanged.)
Expected / Cost: 14.4 min for 5 folds at ~80 columns in S6E9; this frame is 289 columns, so budget ~30-60 min on one
      T4 and let the canary decide. Losing V18 solo is the likely outcome and does not close the leg's value.
Outputs: v54_oof.csv (id, pred) + v54_sub.csv (id, satisfaction) to /kaggle/working - exactly two writes.
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
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
torch.set_float32_matmul_precision('high')

print(f"scikit-learn version: {sklearn_version}")
print(f"PyTorch version: {torch.__version__} | device: "
      f"{'cuda' if torch.cuda.is_available() else 'CPU (not the T4 this arm was budgeted for)'}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = 54
    VERSION_NAME = "V54_DCNv2Family"
    EXP_ID = "S6E10_V54_DCNv2Family"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e10/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e10/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv"

    TARGET = 'satisfaction'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027        # reproduction run changes only RANDOM_SEED (42 -> 2027)

    # V11's builder switch, inherited unchanged: True = predictors fitted on the
    # overlap-cleaned original (~1 min), False = 5-fold cross-fit over ~1M pooled rows.
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
    """V18 seeded numpy and random only, because PyTabKit takes its own RNG from random_state. A
    hand-written net draws its initial weights from torch, so torch has to be seeded too or the
    SEED_REPRO reproduction would be reproducing a different random initialisation."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# The hand-written DCN-V2, copied from OUR OWN S6E9 V13:76-91 / S6E4 V39 lineage with every value
# unchanged. These are the family's numbers, not this episode's RealMLP tuning: there is no
# 'epochs'/'lr' key inherited from REALMLP_PARAMS because a PyTorch loop has no PyTabKit dict.
# Borrowed-recipe note the ledger keeps honest: pos_weight is a class weight V18's PyTabKit path
# does not carry, so this arm differs from the parent in the loss as well as the architecture, and
# the difference is the family - it is not silently inherited, and it is computed from the FIT FOLD.
DCN_CFG = {
    'embedding_dim': 16,        # per-categorical embedding width (capped by the class at min(16, max(8, card//2)))
    'num_cross_layers': 4,
    'low_rank': 64,
    'num_experts': 4,
    'dnn_hidden': [256, 128],
    'dropout': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 4096,
    'inference_batch': 8192,    # S6E9 V7 lesson: chunked inference, never one giant predict
    'max_epochs': 50,
    'es_patience': 10,
    'learning_floor': 0.90,     # fold-1 val AUC below this means a broken net, not a losing model
    'fold_ceiling_min': 45.0,   # S6E9 V13 took 14.4 min for 5 folds at ~80 columns; 289 columns is ~3.6x
}


# =============================================================================
# 4b. THE DCN-V2 NET - copied from S6E9 V13:115-211 (verified down the S6E4 V39 / S6E2 V31 lineage)
# =============================================================================
class CrossLayer(nn.Module):
    """DCN-V2 cross layer with low-rank MoE: x_{l+1} = x0 * (W_l x_l + b_l) + x_l, W = sum_k gate_k V_k U_k."""
    def __init__(self, input_dim, low_rank, num_experts):
        super().__init__()
        self.input_dim, self.low_rank, self.num_experts = input_dim, low_rank, num_experts
        self.gates = nn.Linear(input_dim, num_experts, bias=False)
        self.U = nn.ModuleList([nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_experts)])
        self.V = nn.ModuleList([nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_experts)])
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, x0):
        gates = F.softmax(self.gates(x), dim=1)
        expert_stack = torch.stack([self.V[k](self.U[k](x)) for k in range(self.num_experts)], dim=1)
        w = torch.einsum('be,bei->bi', gates, expert_stack)
        return x0 * (w + self.bias) + x


class LowRankCrossNetwork(nn.Module):
    """Stacked cross layers, each receiving the original input x0."""
    def __init__(self, input_dim, num_layers, low_rank, num_experts):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim, low_rank, num_experts)
                                     for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x


class DCNv2(nn.Module):
    """Deep & Cross Network V2, binary classification - the architecture our earlier episodes ran."""
    def __init__(self, n_num, cat_cards, embedding_dim=16, num_cross_layers=4,
                 low_rank=64, num_experts=4, dnn_hidden=[256, 128], dropout=0.2):
        super().__init__()
        self.n_num, self.cat_cards = n_num, list(cat_cards)
        self.num_cross_layers, self.low_rank, self.num_experts = num_cross_layers, low_rank, num_experts

        self.cat_embeddings = nn.ModuleList()
        total_cat_dim = 0
        for card in cat_cards:
            emb_dim = min(embedding_dim, max(8, card // 2))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            total_cat_dim += emb_dim
        self.total_cat_dim = total_cat_dim
        input_dim = n_num + total_cat_dim
        self.input_dim = input_dim

        self.cross_network = LowRankCrossNetwork(input_dim, num_cross_layers, low_rank, num_experts)

        dnn_layers, prev_dim = [], input_dim
        for h in dnn_hidden:
            dnn_layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.deep_network = nn.Sequential(*dnn_layers)
        self.output = nn.Linear(prev_dim + input_dim, 1)

    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat([x_num] + cat_embs, dim=1)
        combined = torch.cat([self.cross_network(x), self.deep_network(x)], dim=1)
        return self.output(combined).squeeze(-1)


def chunked_predict(model, x_num, x_cat, batch_size, device):
    """Predict in chunks to avoid OOM on large datasets (S6E9 V7 lesson)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            logits = model(x_num[i:i + batch_size].to(device), x_cat[i:i + batch_size].to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def integer_encode(train_df, val_df, test_df, cat_columns):
    """Integer-encode the per-fold categorical vocabularies V18 already built: unseen levels (and NaN)
    become code 0, so an embedding row is always reserved for them and no fold can crash on a level the
    fit fold never carried. Cardinalities come from the FIT FOLD only."""
    for col in cat_columns:
        mapping = {v: i + 1 for i, v in enumerate(sorted(train_df[col].astype(str).unique()))}
        for d in (train_df, val_df, test_df):
            d[col] = d[col].astype(str).map(mapping).fillna(0).astype('int64')
    cards = [int(train_df[col].max()) + 1 for col in cat_columns]
    return train_df, val_df, test_df, cards


def dcn_report(tag, model, n_num, n_cat, rows_fit, rows_val, rows_test):
    """ABSOLUTE readings only: what the constructed net really is, and the device counters.
    Replaces V18's estimate_expanded_matrix/net_memory_report pair, which measured PyTabKit's
    one-hot/embedding transform - a tensor this hand-written net never builds, so a guard on it
    could only fire, or stay silent, for the wrong reason."""
    gib = 1024 ** 3
    n_params = sum(p.numel() for p in model.parameters())
    lines = [
        f"{tag}: architecture read back from the constructed net - {len(model.cat_embeddings)} categorical "
        f"embeddings over cards {model.cat_cards} (+{model.total_cat_dim} cat dims), input_dim "
        f"{model.input_dim} = {n_num} numeric + {model.total_cat_dim} embedded, "
        f"{len(model.cross_network.layers)} cross layers at rank {model.low_rank} x "
        f"{model.num_experts} experts, output in_features {model.output.in_features}",
        f"{tag}: parameters {n_params / 1e6:.2f}M | rows fit/val/test {rows_fit:,}/{rows_val:,}/{rows_test:,} "
        f"| batch {DCN_CFG['batch_size']}, inference chunk {DCN_CFG['inference_batch']}",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(
            f"{tag}: device {props.name} total {props.total_memory / gib:.2f} GB "
            f"| allocated {torch.cuda.memory_allocated(0) / gib:.2f} GB "
            f"| reserved {torch.cuda.memory_reserved(0) / gib:.2f} GB "
            f"| peak reserved {torch.cuda.max_memory_reserved(0) / gib:.2f} GB")
    else:
        lines.append(f"{tag}: CUDA UNAVAILABLE - this run is not on the T4 the arm was budgeted for")
    for line in lines:
        print(f"      {line}")
    return n_params


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

# Inherited from V11 unchanged: the 13 EV predictors. multi:softprob over the rating levels.
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
# 6. FEATURE ENGINEERING (fold-independent part) - byte-identical to V12
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
# 6b. INHERITED FROM V11 - THE 13 LABEL-FREE AUX-EV COLUMNS (this version's axis)
# =============================================================================
def add_aux_ev_features(train_df, test_df, orig_clean):
    """
    V11's axis, inherited unchanged: add 13 `aux_ev_<rating>` columns, EV = sum_k k * P_hat(rating=k),
    where each P_hat comes from an XGB predicting that rating from the OTHER 20 ALL_FEATURES columns
    (never `satisfaction`). They are label-free feature producers, not a second competing model: only
    the net predicts satisfaction and nothing is blended at inference.

    LEAKAGE GUARD: the aux stage is fitted ONCE globally, never per outer fold, and no row's EV may
    come from a model that saw that row. DEFAULT (CFG.AUX_EV_FIT_ON_ORIGINAL=True, run-of-record)
    fits only on the overlap-cleaned original - prepare_original has dropped every original row equal
    to a competition row, so no competition row is in any fit set.
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
        """Contiguous 0-based codes + the ascending rating levels actually present. The level vector
        is load-bearing: `Baggage handling` is 1-5 with NO 0, so EV must use the real k."""
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
# 6c. INHERITED FROM V12 - THE NET'S INPUT REPRESENTATION (twins + teacher-as-logit)
# =============================================================================
def add_numeric_categorical_twins(train_df, test_df, cols):
    """
    V12's axis, carried unchanged. `cols` is the 17 NUMERIC raw predictors (ALL_FEATURES minus
    CATS) - the derived numeric columns, the EVs included, deliberately get NO twin: they are
    continuous estimates, not levels, and factorising one of them would hand PyTabKit hundreds of
    thousands of singleton categories. NO numeric column is dropped.
    RNG-free by construction: pd.factorize over the concatenated train+test pool uses no RNG and no
    labels; NaN becomes code -1, i.e. missing becomes a level of its own. Both frames share one code
    space, so no fold can meet an unseen twin level.
    """
    twin_cols = []
    n_train = len(train_df)
    pool = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
    for c in cols:
        codes = pd.factorize(pool[c])[0]
        levels = np.unique(codes)
        tr = pd.Categorical(codes[:n_train], categories=levels)
        te = pd.Categorical(codes[n_train:], categories=levels)
        if int(np.asarray(tr.isna()).sum() + np.asarray(te.isna()).sum()):
            raise ValueError(f"twin of {c} produced an out-of-vocabulary level")
        name = f"{c}_cat_"
        train_df[name] = tr
        test_df[name] = te
        twin_cols.append(name)
        # V18 printed how PyTabKit's REALMLP transform would expand each twin (one-hot above
        # max_one_hot_cat_size, else a fixed-width embedding). This net does neither: every categorical
        # column - the four raw ones and all 17 twins - is integer-coded per fold and embedded at
        # min(16, max(8, card//2)) by the DCNv2 constructor, so the level count is all that is printed.
        print(f"      twin {name}: {len(levels):,} levels")
    return train_df, test_df, twin_cols


def teacher_to_logit(df):
    """V12's axis (b), unchanged: logit of the teacher probability, clip 1e-6 (V3's house value)
    because the teacher saturates at exactly 0/1 and an inf would poison robust-scaling."""
    p = np.clip(df['_og_teacher'].to_numpy(dtype='float64'), 1e-6, 1.0 - 1e-6)
    df['_og_teacher'] = np.log(p / (1.0 - p)).astype('float32')
    return df


# =============================================================================
# 7. PER-FOLD FEATURE BUILDER (curated TE + frequency/rarity + native cats)
# =============================================================================
def feature_name(column):
    return column.lower().replace(' ', '_').replace('/', '_').replace('-', '_')


def make_key_frames(fit, frames):
    """Int-factorized key columns for TE + frequency. Pair code = code_a * n_cat_b + code_b using
    vocabularies fitted on the fit fold only."""
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
    """PyTabKit raises on NaN in continuous columns; V2 relied on XGBoost routing them. Medians come
    from the FIT FOLD only (no val/test statistics). Inherited from V12 unchanged."""
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X_fit.columns if c not in cat_cols]
    med = X_fit[num].median(numeric_only=True).astype('float32')
    out = []
    for f in (X_fit, X_val, X_test):
        g = f.copy()
        g[num] = g[num].fillna(med)
        g[num] = g[num].astype('float32')
        for c in cat_cols:
            cats = g[c].cat.categories
            g[c] = g[c].astype('object').fillna(cats[0]).astype(pd.CategoricalDtype(cats))
        left = g.isna().sum()
        left = left[left > 0]
        if len(left):
            raise ValueError(f"NaN survived fill: {dict(left)}")
        out.append(g)
    return out


def build_fold_features(X_fit, X_val, X_test, y_fit, fold):
    """Curated per-fold TE (smooth=20, inner cv=5), frequency+rarity, native categorical
    vocabularies from the fit fold. Returns (3 frames, n_added)."""
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
            f[c] = f[c].astype('object').fillna(cats[0]).astype(dt)

    n_added = len(frames[0].columns) - len(X_fit.columns)
    return frames[0], frames[1], frames[2], n_added

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    MAX_EPOCH = DCN_CFG['max_epochs']

    t0_all = time.time()
    print("="*80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED}")
    print(f"Axis: the fitted family only - DCN-V2 (cross={DCN_CFG['num_cross_layers']}, "
          f"rank={DCN_CFG['low_rank']}, experts={DCN_CFG['num_experts']}, "
          f"dnn={DCN_CFG['dnn_hidden']}, emb_dim={DCN_CFG['embedding_dim']}) on V18's frame "
          f"at max_epochs={MAX_EPOCH} / patience {DCN_CFG['es_patience']}")
    print(f"Model provenance: our own S6E9 V13:115-211 implementation (S6E4 V39 -> S6E2 V31 lineage), "
          f"hyperparameters unchanged; deviations are the Stratified partition and a fit-fold pos_weight")
    print(f"Parent: Previously trained files/Archieve/S6E10_V18_RealMLPFusionEV.py "
          f"| gate vs v18_oof, offline after the run")
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

    # ---- V18 AXIS: the 13 aux-EV columns, added here - AFTER the teacher and BEFORE the float32
    # cast and the twin pass, exactly where V11 put them, so the fold driver / TE / recipe stay
    # byte-identical and the EVs can never be mistaken for twin sources.
    print("   Adding V18 axis: 13 label-free aux_ev columns...")
    t0_ev = time.time()
    train, test = add_aux_ev_features(train, test, orig_clean)
    ev_build_s = time.time() - t0_ev
    EV_COLS = [c for c in test.columns if c.startswith('aux_ev_')]
    assert len(EV_COLS) == 13, f"expected 13 aux_ev columns, got {len(EV_COLS)}"
    print(f"      {ev_build_s:.0f}s EV build | +{len(EV_COLS)} columns on the net frame")

    for c in test.columns:
        if c not in CATS and pd.api.types.is_numeric_dtype(test[c]):
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    # ---- INHERITED FROM V12 (a): twins of the 17 numeric RAW predictors only. EV columns are not
    # in ALL_FEATURES, so they are structurally excluded from TWIN_SOURCE - no EV twin exists here.
    TWIN_SOURCE = [c for c in ALL_FEATURES if c not in CATS]
    train, test, twin_cols = add_numeric_categorical_twins(train, test, TWIN_SOURCE)
    print(f"   V12 axis (a) inherited: +{len(twin_cols)} twins (source {len(TWIN_SOURCE)} cols) "
          f"-> frame {len(test.columns)} cols")

    # ---- INHERITED FROM V12 (b): teacher as logit.
    train = teacher_to_logit(train)
    test = teacher_to_logit(test)
    FEATURES = list(test.columns)

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold curated TE + native cats, DCN-V2 in a torch loop)
    # =========================================================================
    print(f"\n[3/5] Training DCN-V2 (max_epochs={MAX_EPOCH}, patience={DCN_CFG['es_patience']}, "
          f"{CFG.N_FOLDS}-Fold CV)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []
    best_iters  = []
    prep_times  = []
    fit_times   = []
    n_params    = []
    frame_cols  = []
    cat_counts  = []

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        prep_start = time.time()
        X_train, X_val, X_test, n_added = build_fold_features(
            X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), test_X.copy(),
            y.iloc[train_idx], fold)
        X_train, X_val, X_test = fill_continuous_nan(X_train, X_val, X_test)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)
        print(f"\n      Fold {fold+1}/{CFG.N_FOLDS} | Train: {X_train.shape} | Val: {X_val.shape} | "
              f"Per-fold cols added: {n_added} | Prep: {prep_s:.0f}s")

        # The column split comes from the dtypes build_fold_features actually produced - not from a
        # remembered list - so a frame change cannot silently move a column between the two paths.
        cat_cols = [c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)]
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        if fold == 0:
            # POSITIVE CONTROL on the split: the frame is expected to carry the 4 raw categoricals plus
            # one twin per numeric raw predictor. Zero or a wrong count means the net is reading the frame
            # differently than this arm claims, which is exactly the K8 class of defect.
            expected_cats = len(CATS) + len(twin_cols)
            print(f"      fold-1 column split: {len(num_cols)} numeric (incl. {len(EV_COLS)} EV) + "
                  f"{len(cat_cols)} categorical (expected {expected_cats} = {len(CATS)} raw + "
                  f"{len(twin_cols)} twins) -> frame {X_train.shape[1]} cols")
            if len(cat_cols) != expected_cats:
                raise AssertionError(f"expected {expected_cats} categorical columns, found {len(cat_cols)} - "
                                     f"the net would embed a different set than this arm describes")
            if len(num_cols) + len(cat_cols) != X_train.shape[1]:
                raise AssertionError("numeric + categorical split does not cover the frame")
        cat_counts.append(len(cat_cols))
        frame_cols.append(int(X_train.shape[1]))

        X_train, X_val, X_test, cat_cards = integer_encode(X_train, X_val, X_test, cat_cols)
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].astype('float32'))
        X_val[num_cols] = scaler.transform(X_val[num_cols].astype('float32'))
        X_test[num_cols] = scaler.transform(X_test[num_cols].astype('float32'))

        x_num_tr = torch.tensor(X_train[num_cols].to_numpy(np.float32))
        x_cat_tr = torch.tensor(X_train[cat_cols].to_numpy(np.int64))
        x_num_va = torch.tensor(X_val[num_cols].to_numpy(np.float32))
        x_cat_va = torch.tensor(X_val[cat_cols].to_numpy(np.int64))
        x_num_te = torch.tensor(X_test[num_cols].to_numpy(np.float32))
        x_cat_te = torch.tensor(X_test[cat_cols].to_numpy(np.int64))
        y_tr_t = torch.tensor(y_train.to_numpy(np.float32))
        del X_train, X_val, X_test
        gc.collect()

        torch.manual_seed(CFG.RANDOM_SEED + fold)
        model = DCNv2(n_num=len(num_cols), cat_cards=cat_cards, **{k: DCN_CFG[k] for k in
                        ('embedding_dim', 'num_cross_layers', 'low_rank', 'num_experts',
                         'dnn_hidden', 'dropout')}).to(DEVICE)

        # ARCHITECTURE READ-BACK from the constructed net, before a single epoch runs. This is the check
        # the borrowed implementation never had (grep: no get_params/assertion in any prior DCN or TabM
        # arm), and it is what proves the net is reading both halves of the frame.
        structural = {
            'embeddings == cat columns': len(model.cat_embeddings) == len(cat_cols),
            'cross layers == declared': len(model.cross_network.layers) == DCN_CFG['num_cross_layers'],
            'input_dim == num + embedded': model.input_dim == len(num_cols) + model.total_cat_dim,
            'output width == deep tail + input': model.output.in_features == DCN_CFG['dnn_hidden'][-1] + model.input_dim,
            'every card >= 2 (code 0 reserved for unseen)': all(c >= 2 for c in cat_cards),
        }
        if fold == 0:
            print(f"      structural read-back: {structural}")
        _bad = [k for k, ok in structural.items() if not ok]
        if _bad:
            raise AssertionError(f"fold {fold+1} built a net that fails {_bad} - stopping before the fits")
        n_params.append(dcn_report(f"pre-fit fold {fold+1}", model, len(num_cols), len(cat_cols),
                                   len(x_num_tr), len(x_num_va), len(x_num_te)))

        optimizer = torch.optim.AdamW(model.parameters(), lr=DCN_CFG['lr'],
                                      weight_decay=DCN_CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)
        # FIT-FOLD class weight (V13 computed this on all training rows; the campaign takes every
        # label-derived statistic from the fit fold only).
        n_pos = float(y_train.sum())
        pos_weight = torch.tensor([(len(y_train) - n_pos) / max(n_pos, 1.0)],
                                  dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loader = DataLoader(TensorDataset(x_num_tr, x_cat_tr, y_tr_t),
                            batch_size=DCN_CFG['batch_size'], shuffle=True)
        if fold == 0:
            print(f"      pos_weight (fit fold) {float(pos_weight):.4f} | steps/epoch "
                  f"{len(loader)} | lr {DCN_CFG['lr']} wd {DCN_CFG['weight_decay']}")

        best_auc, best_epoch, patience, best_state = 0.0, 0, 0, None
        t_fit = time.time()
        epoch_trace = []
        for epoch in range(1, MAX_EPOCH + 1):
            model.train()
            for x_n, x_c, y_b in loader:
                x_n, x_c, y_b = x_n.to(DEVICE), x_c.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x_n, x_c), y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

            vp = np.nan_to_num(chunked_predict(model, x_num_va, x_cat_va,
                                               DCN_CFG['inference_batch'], DEVICE), nan=0.5)
            va = auc_score(y_val.to_numpy(), vp)
            epoch_trace.append(va)
            if va > best_auc:
                best_auc, patience, best_epoch = va, 0, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= DCN_CFG['es_patience']:
                    break
        fit_times.append(time.time() - t_fit)

        # LEARNING FLOOR: a net that never beats 0.90 is broken (misaligned labels, NaN, a dead
        # encoder), not outscored. Report the trace so a null verdict can be told from a bad setup.
        if best_auc < DCN_CFG['learning_floor']:
            print(f"      epoch trace: {[f'{e:.5f}' for e in epoch_trace]}")
            raise AssertionError(f"fold {fold+1} best val AUC {best_auc:.5f} is under the "
                                 f"{DCN_CFG['learning_floor']} learning floor - this is a broken run, "
                                 f"not a losing model")

        # BUDGET CANARY on fold 1 only, before the cost is multiplied by five (S6E9 V29's mechanism).
        fold_min = fit_times[-1] / 60
        if fold == 0:
            print(f"      canary: fold 1 trained in {fold_min:.1f} min -> projected 5-fold "
                  f"{fold_min * CFG.N_FOLDS:.0f} min (ceiling {DCN_CFG['fold_ceiling_min']:.0f} min/fold)")
            if fold_min > DCN_CFG['fold_ceiling_min']:
                raise RuntimeError(f"V54 aborts after fold 1: {fold_min:.1f} min/fold exceeds the declared "
                                   f"ceiling. Do not shrink the frame mid-run - re-run at a declared smaller "
                                   f"DCN_CFG (cross layers / dnn_hidden) and record which one produced the number.")

        model.load_state_dict(best_state)
        model.to(DEVICE)
        oof_probs[val_idx] = np.nan_to_num(chunked_predict(model, x_num_va, x_cat_va,
                                            DCN_CFG['inference_batch'], DEVICE), nan=0.5)
        test_probs += np.nan_to_num(chunked_predict(model, x_num_te, x_cat_te,
                                    DCN_CFG['inference_batch'], DEVICE), nan=0.5) / CFG.N_FOLDS
        fold_scores.append(best_auc)
        best_iters.append(best_epoch)

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {best_auc:.5f} | best epoch {best_epoch} of {MAX_EPOCH} "
              f"(ceiling) | fit {fit_times[-1]:.0f}s | Total: {elapsed:.1f}min")

        del model, optimizer, scheduler, criterion, best_state, loader, epoch_trace
        del x_num_tr, x_cat_tr, x_num_va, x_cat_va, x_num_te, x_cat_te, y_tr_t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_probs = np.clip(oof_probs, 0.0, 1.0)      # each fold wrote its own slice, no averaging left to do
    test_probs = np.clip(test_probs, 0.0, 1.0)    # already averaged by /CFG.N_FOLDS per fold
    if not (np.isfinite(oof_probs).all() and np.isfinite(test_probs).all()):
        raise AssertionError("non-finite probability in the shipped vectors")
    oof_cv = auc_score(y.values, oof_probs)

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"v{CFG.VERSION_NUMBER}"

    pd.DataFrame({'id': train_id, 'pred': oof_probs}).to_csv(
        f"{out_dir}/{stem}_oof.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_oof.csv (id, pred)")

    pd.DataFrame({'id': test_id, CFG.TARGET: test_probs}).to_csv(
        f"{out_dir}/{stem}_sub.csv", index=False)
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, satisfaction)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Axis: the fitted family - DCN-V2 (cross={DCN_CFG['num_cross_layers']}, "
          f"rank={DCN_CFG['low_rank']}, experts={DCN_CFG['num_experts']}, dnn={DCN_CFG['dnn_hidden']}) "
          f"on V18's frame | the structural read-back and fold-1 AUC trace above are the proof of what ran")
    print(f"   OOF CV (AUC): {oof_cv:.5f} | Fold AUC mean +/- std: "
          f"{np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"   Fold AUCs: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Convergence: best epochs {best_iters} of ceiling {MAX_EPOCH} "
          f"(patience {DCN_CFG['es_patience']}; an arm that used every epoch was not converged)")
    print(f"   Parameters per fold: {[f'{p/1e6:.2f}M' for p in n_params]} | "
          f"cat columns embedded {cat_counts} | frame cols handed to the encoder {frame_cols}")
    print(f"   V54's own bar (pre-declared): CV >= 0.961337 AND paired DeLong z > 3 vs v18_oof AND >=4/5 "
          f"folds improved, then reproduce at seed {CFG.SEED_REPRO}")
    print(f"   V55 rule (fixed before this ran): ship mean(v27,v36,v44,v54) only if V54 solo >= 0.9600 "
          f"AND the 4-set beats V51's recorded CV; the trio stands otherwise")
    print(f"   The CV->LB offset of a blend containing a NEW family is UNMEASURED (V44/V45 lesson): "
          f"V51's gap does not transfer here")
    print(f"   EV build {ev_build_s:.0f}s | Timing: prep {np.mean(prep_times):.0f}s + "
          f"train {np.mean(fit_times):.0f}s per fold (mean) | total {total_time_min:.1f} min")
