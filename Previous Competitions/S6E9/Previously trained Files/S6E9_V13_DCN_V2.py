"""
S6E9 V13 - DCN-V2 (Deep & Cross Network V2, PyTorch/GPU)
================================================================================
Strategy: DCN-V2 with V6 Feature Selection + V10/V12 Proven Features

Reference: S6E4 V39 DCN-V2 — "Best NN" on S6E2 (0.95366 LB)
- Real DCN-V2 architecture (verified: Cross layer + Low-rank MoE + parallel Deep)
- Google Research (WWW 2021): "DCN V2: Improved Deep & Cross Network"

Why DCN-V2 for S6E9:
- S6E2 V31: 0.95366 — "Best NN" (beat TabM 0.95383, FT-Transformer 0.95370)
- DCN-V2 uses EXPLICIT cross interactions (Hadamard product) — perfect for
  S6E9's rule-based signal (Sub×ECL×RA 3-way interaction)
- Different from TabM (MLP+BatchEnsemble) and FT-Transformer (self-attention)
- Cross layer: x_{l+1} = x0 * (W * x + b) + x — learns feature crosses explicitly

Lessons Applied:
1. V6 lesson: Feature selection for NN (~80 features, not 344)
2. V7 lesson: Batched inference to avoid OOM
3. V10 lesson: Include trigram_Sub_ECL_RA + bigram_income_band_x_Subsidy
4. V11 lesson: NO _recipe_residual (cannibalizes)
5. V12 lesson: sklearn TargetEncoder (fast, not manual inner K-fold)

Architecture:
  Categoricals -> Embedding(dim=16)
  Numericals -> StandardScaler
  Concatenate -> LowRankCrossNetwork(x0, W=U*V, MoE gating) || Deep MLP -> Dense(1)

  Cross layer: x_{l+1} = x0 * (W_l * x_l + b_l) + x_l  (Hadamard product)
  Low-rank MoE: W = sum_k(gate_k * V_k(U_k(x))), rank=64, experts=4

Device: GPU | Est. Time: ~20-30 min
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v13"
    EXP_ID = "S6E9_V13_DCN_V2"
    DEVICE = DEVICE
    N_FOLDS = 5
    RANDOM_SEED = 42
    TARGET = 'Will_Buy_EV'

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    # DCN-V2 hyperparams (from S6E4 V39)
    EMBEDDING_DIM = 16
    NUM_CROSS_LAYERS = 4
    LOW_RANK = 64
    NUM_EXPERTS = 4
    DNN_HIDDEN = [256, 128]
    DROPOUT = 0.2

    # Training
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 4096
    INFERENCE_BATCH = 8192
    MAX_EPOCHS = 50
    ES_PATIENCE = 10

# =============================================================================
# 3. SEED
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 5. DCN-V2 MODEL (Verified real implementation from S6E4 V39)
# =============================================================================
class CrossLayer(nn.Module):
    """DCN-V2 cross layer with low-rank MoE."""
    def __init__(self, input_dim, low_rank, num_experts):
        super().__init__()
        self.gates = nn.Linear(input_dim, num_experts, bias=False)
        self.U = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_experts)
        ])
        self.V = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_experts)
        ])
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, x0):
        gates = F.softmax(self.gates(x), dim=1)
        expert_outs = [self.V[k](self.U[k](x)) for k in range(len(self.U))]
        expert_stack = torch.stack(expert_outs, dim=1)
        w = torch.einsum('be,bei->bi', gates, expert_stack)
        return x0 * (w + self.bias) + x  # Hadamard product + residual


class LowRankCrossNetwork(nn.Module):
    """Stacked cross layers, each receiving the original input x0."""
    def __init__(self, input_dim, num_layers, low_rank, num_experts):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayer(input_dim, low_rank, num_experts) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x


class DCNv2(nn.Module):
    """Deep & Cross Network V2 — binary classification version."""
    def __init__(self, n_num, cat_cards, embedding_dim=16, num_cross_layers=4,
                 low_rank=64, num_experts=4, dnn_hidden=[256, 128], dropout=0.2):
        super().__init__()

        self.cat_embeddings = nn.ModuleList()
        total_cat_dim = 0
        for card in cat_cards:
            emb_dim = min(embedding_dim, max(8, card // 2))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            total_cat_dim += emb_dim

        input_dim = n_num + total_cat_dim

        self.cross_network = LowRankCrossNetwork(
            input_dim, num_cross_layers, low_rank, num_experts)

        dnn_layers = []
        prev_dim = input_dim
        for h in dnn_hidden:
            dnn_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.deep_network = nn.Sequential(*dnn_layers)

        # Binary: single output logit
        self.output = nn.Linear(prev_dim + input_dim, 1)

    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_emb = torch.cat(cat_embs, dim=1)
        x = torch.cat([x_num, cat_emb], dim=1)

        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output(combined)
        return logits.squeeze(-1)


# =============================================================================
# 6. CHUNKED INFERENCE (V7 lesson: avoid OOM)
# =============================================================================
def chunked_predict(model, x_num, x_cat, batch_size, device):
    """Predict in chunks to avoid OOM on large datasets."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            logits = model(
                x_num[i:i+batch_size].to(device),
                x_cat[i:i+batch_size].to(device)
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# =============================================================================
# 7. FEATURE ENGINEERING (V3 base + V10/V12 proven + V6 selection)
# =============================================================================
def add_digit_features(df, num_cols):
    df = df.copy()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c].fillna(0) // (10**k) % 10).astype('int8')
    return df


def add_engineered_features(df):
    df = df.copy()
    for col in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    ra_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['_RA_code'] = df['Range_Anxiety_Level'].map(ra_map).fillna(0).astype('int8')
    df['_Subsidy_bin'] = (df['Subsidy_Available'] == 'Yes').astype('int8')

    df['_ECL_x_Subsidy'] = (df['Environmental_Concern_Level'] * df['_Subsidy_bin']).astype('float32')
    df['_ECL_x_RangeAnxiety'] = (df['Environmental_Concern_Level'] * (3 - df['_RA_code'])).astype('float32')
    df['_Income_x_Subsidy'] = (df['Annual_Income_USD'] * df['_Subsidy_bin']).astype('float32')
    df['_Charging_Total'] = (df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']).astype('float32')
    df['_log_Income'] = np.log1p(df['Annual_Income_USD'].clip(lower=0)).astype('float32')
    df['_log_Commute'] = np.log1p(df['Daily_Commute_km'].clip(lower=0)).astype('float32')
    df['_log_Charging_Total'] = np.log1p(df['_Charging_Total'].clip(lower=0)).astype('float32')
    df['_high_income'] = (df['Annual_Income_USD'] > 170537).astype('int8')
    df['_range_anxiety_high'] = (df['Range_Anxiety_Level'] == 'High').astype('int8')
    df['_ecl_max'] = (df['Environmental_Concern_Level'] == 5).astype('int8')
    df['_ev_recipe'] = ((df['Environmental_Concern_Level'] == 5) & (df['Range_Anxiety_Level'] == 'Low')).astype('int8')
    df = df.drop(columns=['_RA_code', '_Subsidy_bin'])
    return df


def add_synthetic_artifact_flags(df):
    df = df.copy()
    df['is_30k_spike'] = (df['Annual_Income_USD'] == 30000.0).astype('int8')
    df['is_millionaire_cliff'] = (df['Annual_Income_USD'] >= 170537.0).astype('int8')
    df['is_dead_zone'] = ((df['Annual_Income_USD'] >= 38000.0) & (df['Annual_Income_USD'] <= 42000.0)).astype('int8')
    df['is_env_hater'] = (df['Environmental_Concern_Level'] == 1).astype('int8')
    return df


def add_smooth_keys(df):
    df = df.copy()
    df['income_exact_int'] = np.floor(df['Annual_Income_USD']).astype(str)
    df['income100_floor']  = np.floor(df['Annual_Income_USD'] / 100.0).astype(str)
    df['income1000_floor'] = np.floor(df['Annual_Income_USD'] / 1000.0).astype(str)
    df['commute_integer']  = np.floor(df['Daily_Commute_km']).astype(str)
    return df


def add_original_target_means(train_df, test_df, orig_df, cat_cols, num_cols, target):
    orig_global_mean = orig_df[target].mean()
    for col in cat_cols + num_cols:
        if col in orig_df.columns:
            stats = orig_df.groupby(col, observed=False)[target].mean()
            train_df[f"{col}_org_mean"] = train_df[col].map(stats).fillna(orig_global_mean).astype('float32')
            test_df[f"{col}_org_mean"] = test_df[col].map(stats).fillna(orig_global_mean).astype('float32')
    return train_df, test_df


def add_numeric_as_string(df, num_cols):
    df = df.copy()
    num_to_cat_cols = []
    for col in num_cols:
        if col not in df.columns: continue
        cat_name = f"{col}_cat"
        df[cat_name] = df[col].fillna('NaN').astype(str)
        num_to_cat_cols.append(cat_name)
    return df, num_to_cat_cols


def add_frequency_encoding(train_df, test_df, cols_to_encode):
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"]  = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df


def drop_redundant_features(train_df, test_df, target):
    eval_cols = [c for c in train_df.columns
                 if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]
    to_drop_corr = []
    if len(eval_cols) > 1:
        corr_matrix = train_df[eval_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
        del corr_matrix, upper_tri; gc.collect()
    to_drop_const = [c for c in train_df.columns if train_df[c].nunique() == 1] + \
                    [c for c in test_df.columns if test_df[c].nunique() == 1]
    DROP = set(to_drop_corr).union(set(to_drop_const))
    DROP = [c for c in DROP if c not in ['id', target]]
    if len(DROP) > 0:
        print(f"   Dropping {len(DROP)} redundant/constant features")
        train_df = train_df.drop(columns=DROP, errors='ignore')
        test_df  = test_df.drop(columns=DROP, errors='ignore')
    return train_df, test_df, DROP


def add_targeted_bigrams(train_df, test_df):
    print("   Adding TARGETED bigram interactions...")
    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    bigram_name_1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bigram_name_1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bigram_name_1] = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)
    bigram_name_2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bigram_name_2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bigram_name_2] = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)
    return train_df, test_df, [bigram_name_1, bigram_name_2]


def add_proven_features(train_df, test_df, orig_df):
    """V10/V12 proven: trigram + income-band bigram."""
    print("   Adding 2 PROVEN features (trigram + income-band bigram)...")
    trigram_name = 'trigram_Sub_ECL_RA'
    for df in [train_df, test_df, orig_df]:
        df[trigram_name] = (df['Subsidy_Available'].astype(str) + '_' +
            df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            df['Range_Anxiety_Level'].astype(str))
    bigram_inc_sub = 'bigram_income_band_x_Subsidy'
    income_bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
    income_labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
    for df in [train_df, test_df, orig_df]:
        df['_inc_band'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=income_bins, labels=income_labels).astype(str)
        df[bigram_inc_sub] = df['_inc_band'] + '_' + df['Subsidy_Available'].astype(str)
        df.drop(columns=['_inc_band'], inplace=True)
    return train_df, test_df, orig_df, [trigram_name, bigram_inc_sub]


def select_features_for_nn(train_df, test_df, cat_cols, num_cols):
    """V6 lesson: Select ~80 features for NN (not 344)."""
    print("   Selecting features for DCN-V2 NN (V6 evidence-based)...")
    keep = set()
    keep.update(num_cols); keep.update(cat_cols)
    engineered = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy',
                  '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    keep.update([c for c in engineered if c in train_df.columns])
    flags = ['_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe']
    keep.update([c for c in flags if c in train_df.columns])
    magic = ['is_30k_spike', 'is_millionaire_cliff', 'is_dead_zone', 'is_env_hater']
    keep.update([c for c in magic if c in train_df.columns])
    smooth = ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    keep.update([c for c in smooth if c in train_df.columns])
    org_means = [c for c in train_df.columns if c.endswith('_org_mean')]
    keep.update(org_means)
    top_fe = ['Subsidy_Available_fe', 'Environmental_Concern_Level_cat_fe',
              '_Income_x_Subsidy_cat_fe', 'Annual_Income_USD_cat_fe',
              'income100_floor_fe', 'income1000_floor_fe', 'Gender_fe', 'City_Type_fe',
              'Current_Car_Type_fe', 'Home_Charging_Possible_fe', 'Range_Anxiety_Level_fe',
              '_ECL_x_Subsidy_cat_fe', '_ECL_x_RangeAnxiety_cat_fe',
              '_log_Income_cat_fe', '_log_Commute_cat_fe', '_Charging_Total_cat_fe',
              'Age_cat_fe', 'Daily_Commute_km_cat_fe', 'Number_of_Cars_Owned_cat_fe',
              'Charging_Stations_Near_Home_cat_fe', 'Charging_Stations_Near_Work_cat_fe',
              'income_exact_int_fe', 'commute_integer_fe']
    keep.update([c for c in top_fe if c in train_df.columns])
    top_te = ['TE_income100_floor_auto', 'TE_income1000_floor_auto',
              'TE_income_exact_int_auto', 'TE_commute_integer_auto',
              'TE__Income_x_Subsidy_cat_auto', 'TE__log_Income_cat_auto',
              'TE__ECL_x_Subsidy_cat_auto', 'TE__ECL_x_RangeAnxiety_cat_auto',
              'TE__log_Commute_cat_auto', 'TE__log_Charging_Total_cat_auto',
              'TE__Charging_Total_cat_auto', 'TE_Annual_Income_USD_cat_auto',
              'TE_Age_cat_auto', 'TE_Environmental_Concern_Level_cat_auto',
              'TE_Daily_Commute_km_cat_auto', 'TE_Number_of_Cars_Owned_cat_auto',
              'TE_Charging_Stations_Near_Home_cat_auto', 'TE_Charging_Stations_Near_Work_cat_auto']
    keep.update([c for c in top_te if c in train_df.columns])
    keep = [c for c in keep if c in train_df.columns and c in test_df.columns]
    print(f"      Keeping {len(keep)} features for DCN-V2")
    return keep


def align_original_schema(orig_df):
    keep_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                 'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
                 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    return orig_df[keep_cols].copy()


def integer_encode(train_df, val_df, test_df, cat_columns):
    """Integer encode categoricals for embeddings."""
    mappings = {}
    for col in cat_columns:
        unique_vals = sorted(train_df[col].astype(str).unique())
        mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
        mappings[col] = mapping
        train_df[col] = train_df[col].astype(str).map(mapping).fillna(0).astype(np.int32)
        val_df[col] = val_df[col].astype(str).map(mapping).fillna(0).astype(np.int32)
        test_df[col] = test_df[col].astype(str).map(mapping).fillna(0).astype(np.int32)
    cardinalities = [train_df[col].max() + 1 for col in cat_columns]
    return train_df, val_df, test_df, cardinalities


# =============================================================================
# 8. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: DCN-V2 (cross={CFG.NUM_CROSS_LAYERS}, rank={CFG.LOW_RANK}, experts={CFG.NUM_EXPERTS})")
    print(f"FE: V3 + V10/V12 proven + V6 feature selection (~80 features)")
    print("="*80)

    # [1/5] LOAD
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)
    orig[CFG.TARGET] = orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    print(f"   Train: {train.shape} | Pos rate: {train[CFG.TARGET].mean():.4f}")

    # [2/5] FE
    print("\n[2/5] Feature Engineering...")
    train = add_digit_features(train, NUMS); test = add_digit_features(test, NUMS); orig_aligned = add_digit_features(orig_aligned, NUMS)
    train = add_engineered_features(train); test = add_engineered_features(test); orig_aligned = add_engineered_features(orig_aligned)
    train = add_synthetic_artifact_flags(train); test = add_synthetic_artifact_flags(test); orig_aligned = add_synthetic_artifact_flags(orig_aligned)
    train = add_smooth_keys(train); test = add_smooth_keys(test); orig_aligned = add_smooth_keys(orig_aligned)
    train, test = add_original_target_means(train, test, orig, CATS, NUMS, CFG.TARGET)
    orig_global_mean = orig[CFG.TARGET].mean()
    for col in CATS + NUMS:
        org_mean_name = f"{col}_org_mean"
        if org_mean_name in train.columns and col in orig_aligned.columns:
            orig_stats = orig.groupby(col, observed=False)[CFG.TARGET].mean()
            orig_aligned[org_mean_name] = orig_aligned[col].map(orig_stats).fillna(orig_global_mean).astype('float32')
        elif org_mean_name in train.columns:
            orig_aligned[org_mean_name] = orig_global_mean

    all_numeric_cols = [c for c in train.columns if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c]) and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy', '_Charging_Total', '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    train, train_num_cat_cols = add_numeric_as_string(train, all_numeric_cols + engineered_numeric)
    test, _ = add_numeric_as_string(test, all_numeric_cols + engineered_numeric)
    orig_aligned, _ = add_numeric_as_string(orig_aligned, all_numeric_cols + engineered_numeric)

    freq_target_cols = CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns: orig_aligned[f"{col}_fe"] = 0.0

    # TE (auto only — V6 lesson)
    TE_COLS = [c for c in (CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']) if c in train.columns]
    te = TargetEncoder(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=42)
    train_enc = te.fit_transform(train[TE_COLS], train[CFG.TARGET]).astype('float32')
    test_enc  = te.transform(test[TE_COLS]).astype('float32')
    for i, col in enumerate(TE_COLS):
        train[f"TE_{col}_auto"] = train_enc[:, i]
        test[f"TE_{col}_auto"]  = test_enc[:, i]

    # V10/V12 proven features
    train, test, bigram_cols = add_targeted_bigrams(train, test)
    for df in [orig_aligned]: df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bn in bigram_cols:
        if 'RangeAnxiety' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'Subsidy' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test, orig_aligned, proven_cols = add_proven_features(train, test, orig_aligned)

    # TE on bigrams + proven cols
    for col in bigram_cols + proven_cols:
        if col in train.columns:
            te2 = TargetEncoder(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=42)
            train[f"TE_{col}"] = te2.fit_transform(train[[col]], train[CFG.TARGET]).ravel().astype('float32')
            test[f"TE_{col}"] = te2.transform(test[[col]]).ravel().astype('float32')

    # Drop helper columns
    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin'], errors='ignore', inplace=True)

    # Feature selection
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # V6 lesson: Select features for NN
    selected = select_features_for_nn(train, test, CATS, NUMS)
    train = train[selected + [CFG.TARGET]]
    test  = test[selected]

    # Identify cat vs num for DCN-V2
    cat_features = [c for c in selected if train[c].dtype == 'object' or str(train[c].dtype) == 'string'
                    or train[c].dtype == 'int8' or train[c].dtype == 'int32']
    num_features = [c for c in selected if c not in cat_features]

    print(f"\n   Selected: {len(selected)} features ({len(cat_features)} cat, {len(num_features)} num)")

    # [3/5] TRAINING
    print(f"\n[3/5] Training DCN-V2 ({CFG.N_FOLDS}-Fold CV)...")
    X = train.drop(columns=[CFG.TARGET]); y = train[CFG.TARGET]; test_X = test.copy()
    oof_probs = np.zeros(len(y)); test_probs = np.zeros(len(test_X)); fold_scores = []

    # Class weight
    pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()], dtype=torch.float32, device=DEVICE)

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_tr, X_va = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        X_te = test_X.copy()

        # Integer encode cats per-fold
        X_tr, X_va, X_te, cat_cards = integer_encode(X_tr, X_va, X_te, cat_features)

        # StandardScaler on numericals
        scaler = StandardScaler()
        X_tr[num_features] = scaler.fit_transform(X_tr[num_features])
        X_va[num_features] = scaler.transform(X_va[num_features])
        X_te[num_features] = scaler.transform(X_te[num_features])

        # Convert to tensors
        x_num_tr = torch.tensor(X_tr[num_features].values, dtype=torch.float32)
        x_cat_tr = torch.tensor(X_tr[cat_features].values, dtype=torch.long)
        x_num_va = torch.tensor(X_va[num_features].values, dtype=torch.float32)
        x_cat_va = torch.tensor(X_va[cat_features].values, dtype=torch.long)
        x_num_te = torch.tensor(X_te[num_features].values, dtype=torch.float32)
        x_cat_te = torch.tensor(X_te[cat_features].values, dtype=torch.long)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)

        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)

        # Build model
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        model = DCNv2(
            n_num=len(num_features), cat_cards=cat_cards,
            embedding_dim=CFG.EMBEDDING_DIM, num_cross_layers=CFG.NUM_CROSS_LAYERS,
            low_rank=CFG.LOW_RANK, num_experts=CFG.NUM_EXPERTS,
            dnn_hidden=CFG.DNN_HIDDEN, dropout=CFG.DROPOUT,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        if fold == 0:
            print(f"      DCN-V2 params: {n_params/1e6:.1f}M")
            print(f"      Cat features: {len(cat_features)}, Num features: {len(num_features)}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.MAX_EPOCHS)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = 0.0; patience = 0; best_state = None
        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            model.train()
            for x_n, x_c, y_b in train_loader:
                x_n, x_c, y_b = x_n.to(DEVICE), x_c.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_n, x_c)
                loss = criterion(logits, y_b)
                loss.backward(); optimizer.step()
            scheduler.step()

            # Validation (chunked — V7 lesson)
            val_probs = chunked_predict(model, x_num_va, x_cat_va, CFG.INFERENCE_BATCH, DEVICE)
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            val_auc = auc_score(y_va.values, val_probs)

            if val_auc > best_auc:
                best_auc = val_auc; patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= CFG.ES_PATIENCE:
                    print(f"      [ES@{epoch}]", end="")
                    break

        # Restore best
        model.load_state_dict(best_state); model.to(DEVICE)

        # OOF + test predictions (chunked)
        oof_probs[val_idx] = chunked_predict(model, x_num_va, x_cat_va, CFG.INFERENCE_BATCH, DEVICE)
        test_probs += chunked_predict(model, x_num_te, x_cat_te, CFG.INFERENCE_BATCH, DEVICE) / CFG.N_FOLDS

        oof_probs[val_idx] = np.nan_to_num(oof_probs[val_idx], nan=0.5)
        fold_scores.append(best_auc)

        fold_time = time.time() - fold_start; elapsed = (time.time() - t0) / 60
        print(f" AUC={best_auc:.5f} | Time={fold_time:.0f}s | Total={elapsed:.1f}min")

        del model, optimizer, scheduler, criterion, best_state, x_num_tr, x_cat_tr, x_num_va, x_cat_va, x_num_te, x_cat_te, y_tr_t, train_ds, train_loader
        gc.collect(); torch.cuda.empty_cache()

    oof_cv = auc_score(y.values, oof_probs)
    print(f"\n   OOF CV (AUC): {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"   Mean +/- std: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # [4/5] SAVE
    print(f"\n[4/5] Saving outputs...")
    out_dir = "/kaggle/working"; os.makedirs(out_dir, exist_ok=True)
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_probs})
    oof_df.to_csv(f"{out_dir}/oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_probs})
    sub_df.to_csv(f"{out_dir}/sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv + sub_{CFG.VERSION_NAME}.csv")

    # [5/5] RESULTS
    print(f"\n{'='*80}")
    print(f"V13 RESULTS — DCN-V2 ({DEVICE})")
    print(f"{'='*80}")
    print(f"Model: DCN-V2 (cross={CFG.NUM_CROSS_LAYERS}, rank={CFG.LOW_RANK}, experts={CFG.NUM_EXPERTS})")
    print(f"Features: {len(selected)} ({len(cat_features)} cat + {len(num_features)} num)")
    print(f"FE: V3 + V10/V12 proven + V6 selection")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
