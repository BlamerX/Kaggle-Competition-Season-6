"""
S6E9 V17 - RealMLP (PyTorch GPU)
================================================================================
Strategy: RealMLP architecture (PBLD embedding + ensemble + EMA) + V14 FE pipeline

Why V17 = RealMLP:
- Trees (XGB/LGBM/CatBoost) all saturate at ~0.946 OOF — proven 7 times
- RealMLP has fundamentally different inductive bias:
  * PBLD embedding for numerics (Periodic Basis with Learned Decay)
  * Neural ensemble inside one network (n_ens=8)
  * Continuous representation (vs tree's discrete splits)
- Different errors from trees = either a win alone, or gold for ensemble

Architecture (from public ps-s6-e9-realmlp-pytorch notebook, adapted):
- PBLDEmbedding: numerical features → periodic basis → learned decay
- CategoricalFeatureLayer: onehot ≤4, embedding for the rest
- 3 hidden layers of 256 with residual blocks
- 8 ensemble members inside one network (n_ens=8)
- SiLU activation, learnable residual strength
- EMA weights (decay 0.997875)
- Label smoothing (eps=0.04, cosine schedule)
- Per-parameter-group LRs (5 groups)
- AdamW with flat_anneal LR schedule

Improvements over public notebook:
- V14 full FE pipeline (digit features, smooth keys, bigrams, trigrams, freq, org means)
- Per-fold orig concat (V14 style — 10k extra training rows per fold)
- Triple TE on key cols (smooth auto, 10, 100) — V14 style
- 3 epochs (vs public's 2)
- KFold(5, shuffle=True, rs=42) per our golden rules

Device: GPU (cuda) | Est. Time: ~25-35 min
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import gc
import math
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: GPU ({'cuda' if torch.cuda.is_available() else 'cpu'})")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v17"
    EXP_ID = "S6E9_V17_RealMLP"
    DEVICE = "GPU"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'
    N_FOLDS = 5
    RANDOM_SEED = 42

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed_everything(CFG.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 4. REALMLP ARCHITECTURE (from public notebook, unchanged)
# =============================================================================
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """Applies configurable sequence: median_center, robust_scale, smooth_clip, l2_normalize."""
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_ens, dim, dropout, activation=nn.SiLU):
        super().__init__()
        self.linear = NTPLinear(n_ens=n_ens, in_features=dim, out_features=dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.ones(n_ens, dim) * 0.1)

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.act(x)
        x = self.drop(x)
        return residual + x * self.res_scale.unsqueeze(0)


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay embedding for numerical features."""
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.empty(n_ens, n_features, hidden_dim))
        nn.init.normal_(self.w1, mean=0.0, std=freq_scale / math.sqrt(hidden_dim))
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim, onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(n_ens=n_ens, n_features=n_numerical, hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"], freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = cfg["activation"]
        self._dropout_modules = []
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        in_dim = total_dim
        first_linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=hidden_dims[0])
        self.first_linear = first_linear
        layers.extend([first_linear, act()])
        in_dim = hidden_dims[0]
        for hdim in hidden_dims[1:]:
            if in_dim != hdim:
                layers.extend([NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=hdim), act()])
                in_dim = hdim
            block = ResidualBlock(n_ens=n_ens, dim=hdim, dropout=cfg["dropout"], activation=act)
            self._dropout_modules.append(block.drop)
            layers.append(block)
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)
        with torch.no_grad():
            self.output_layer.weight.mul_(0.1)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return x


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR = p["lr"]; WD = p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD, "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR, "weight_decay": WD, "group": "other_w"},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"], "group": "bias"},
    ]


def binary_bce_loss(y_true, logits, ls=0.0, pos_weight=None):
    if ls > 0.0:
        y_true = y_true * (1.0 - ls) + 0.5 * ls
    if pos_weight is None:
        loss = (1.0 - y_true) * logits + F.softplus(-logits)
    else:
        loss = (1.0 - y_true) * logits + (1.0 + (pos_weight - 1.0) * y_true) * F.softplus(-logits)
    return loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    """Sklearn-compatible wrapper around RealMLP."""
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train); y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        pos_weight = torch.tensor(weights_np[1], dtype=torch.float32, device=dev)

        self.model_ = RealMLP(output_dim=1, cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p).to(dev)
        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.float32, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]; train_bs = p["train_bs"]; eval_bs = p["eval_bs"]
        epochs = p["epochs"]; lr_sched = p["lr_sched"]; flat_ratio = p["flat_ratio"]
        ema_decay = p["ema_decay"]
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf; best_epoch = 0
        best_val_probs = None; best_state = None; ema_state = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start:start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = binary_bce_loss(ytt[idx_batch].repeat_interleave(n_ens), y_pred.reshape(-1), ls=ls_val, pos_weight=None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        model_state = self.model_.state_dict()
                        for key, value in model_state.items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)
            np.random.shuffle(train_order)

            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs_pos = np.concatenate([
                    torch.sigmoid(self.model_(Xvn[s:s + eval_bs], Xvc[s:s + eval_bs])).mean(dim=1).squeeze(-1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
                val_probs = np.stack([1.0 - val_probs_pos, val_probs_pos], axis=1)

            val_pred = val_probs[:, 1]
            epoch_score = roc_auc_score(y_v, val_pred)
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score; best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}
            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)
            if verbose >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  score = {epoch_score:.5f}  best = {best_score:.5f}" + (" ✓" if improved else ""))
            if p["use_early_stopping"]:
                patience = best_epoch * p["early_stopping_multiplicative_patience"] + p["early_stopping_additive_patience"]
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  → best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            probs_pos = np.concatenate([
                torch.sigmoid(self.model_(Xn[s:s + eval_bs], Xc[s:s + eval_bs])).mean(dim=1).squeeze(-1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)


# RealMLP configuration (from public notebook, tuned for S6E9)
CONFIG = {
    "n_ens": 8, "embed_dim": 6, "onehot_thresh": 4,
    "hidden_dims": [256, 256, 256], "dropout": 0.05, "p_drop_sched": "expm4t",
    "activation": nn.SiLU, "add_front_scale": True,
    "pbld_hidden_dim": 20, "pbld_out_dim": 5, "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU, "pbld_lr_factor": 0.093,
    "lr": 0.01, "mom": 0.9, "sq_mom": 0.99, "lr_sched": "flat_anneal", "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.2, "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0, "lr_bias_mult": 0.1,
    "weight_decay": 0.013, "wd_scale_mult": 0.1, "wd_bias_mult": 0.5,
    "ema_decay": 0.997875, "grad_clip": 1.0,
    "ls_eps": 0.04, "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale", "smooth_clip"],
    "epochs": 3,  # V17: 3 epochs (vs public's 2 — more training)
    "train_bs": 256, "eval_bs": 10240,
    "verbosity": 1,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "device": "cuda", "random_state": 42,
}

# =============================================================================
# 5. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 6. FEATURE ENGINEERING — V14 base pipeline (proven)
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

def add_frequency_encoding(train_df, test_df, cols_to_encode):
    combined = pd.concat([train_df[cols_to_encode], test_df[cols_to_encode]], axis=0)
    for col in cols_to_encode:
        freq_mapping = combined[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_fe"] = train_df[col].map(freq_mapping).astype('float32').fillna(0.0)
        test_df[f"{col}_fe"] = test_df[col].map(freq_mapping).astype('float32').fillna(0.0)
    return train_df, test_df

def drop_redundant_features(train_df, test_df, target):
    eval_cols = [c for c in train_df.columns if c not in ['id', target] and pd.api.types.is_numeric_dtype(train_df[c])]
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
    print("   Adding TARGETED bigram interactions (V10 proven)...")
    for df in [train_df, test_df]:
        df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    bn1 = 'bigram_ECL_bin_x_RangeAnxiety'
    train_df[bn1] = train_df['ECL_bin'] + '_' + train_df['Range_Anxiety_Level'].astype(str)
    test_df[bn1] = test_df['ECL_bin'] + '_' + test_df['Range_Anxiety_Level'].astype(str)
    bn2 = 'bigram_ECL_bin_x_Subsidy'
    train_df[bn2] = train_df['ECL_bin'] + '_' + train_df['Subsidy_Available'].astype(str)
    test_df[bn2] = test_df['ECL_bin'] + '_' + test_df['Subsidy_Available'].astype(str)
    return train_df, test_df, [bn1, bn2]

def add_proven_features(train_df, test_df, orig_df):
    print("   Adding 2 PROVEN features (trigram + income-band bigram)...")
    tn = 'trigram_Sub_ECL_RA'
    for df in [train_df, test_df, orig_df]:
        df[tn] = (df['Subsidy_Available'].astype(str) + '_' +
            df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str) + '_' +
            df['Range_Anxiety_Level'].astype(str))
    bis = 'bigram_income_band_x_Subsidy'
    bins = [0, 31004, 42000, 70000, 100000, 150000, 170537, 999999]
    labels = ['lt31k', '31-42k', '42-70k', '70-100k', '100-150k', '150-170k', 'gt170k']
    for df in [train_df, test_df, orig_df]:
        df['_inc_band'] = pd.cut(df['Annual_Income_USD'].fillna(df['Annual_Income_USD'].median()),
                                  bins=bins, labels=labels).astype(str)
        df[bis] = df['_inc_band'] + '_' + df['Subsidy_Available'].astype(str)
        df.drop(columns=['_inc_band'], inplace=True)
    return train_df, test_df, orig_df, [tn, bis]

def align_original_schema(orig_df):
    keep = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
            'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
            'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    return orig_df[keep].copy()

# =============================================================================
# 7. CATEGORICAL ENCODING (factorize to int64 for RealMLP)
# =============================================================================
def factorize_categoricals(train_df, val_df, test_df, cat_cols):
    """Factorize string categoricals to int64 codes for RealMLP.
    Unseen values in val/test get code -1 (clipped to 0 later by RealMLP)."""
    for col in cat_cols:
        # Get all unique values from train
        uniques = train_df[col].astype(str).unique()
        code_map = {v: i for i, v in enumerate(uniques)}
        train_df[col] = train_df[col].astype(str).map(code_map).fillna(-1).astype('int64')
        val_df[col] = val_df[col].astype(str).map(code_map).fillna(-1).astype('int64')
        test_df[col] = test_df[col].astype(str).map(code_map).fillna(-1).astype('int64')
    return train_df, val_df, test_df

# =============================================================================
# 8. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} (RealMLP) | Folds: {CFG.N_FOLDS}")
    print(f"Architecture: PBLD embedding + 8 ensemble + 3 hidden×256 + EMA + label smoothing")
    print(f"FE: V14 full pipeline (digits, smooth keys, bigrams, trigrams, freq, org means)")
    print(f"Improvements over public: V14 FE + orig concat + triple TE + 3 epochs")
    print("="*80)

    # [1/5] LOAD
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    target2idx = {'No': 0, 'Yes': 1}
    train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)
    orig[CFG.TARGET]  = orig[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    orig_aligned = align_original_schema(orig)

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    print(f"   Train: {train.shape} | Pos rate: {train[CFG.TARGET].mean():.4f}")

    # [2/5] FEATURE ENGINEERING (V14 base pipeline)
    print("\n[2/5] Feature Engineering (V14 base)...")
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

    # Numeric-as-string features (for TE)
    all_numeric_cols = [c for c in train.columns if c != CFG.TARGET and pd.api.types.is_numeric_dtype(train[c]) and not c.startswith('_') and not c.startswith('is_')]
    engineered_numeric = ['_ECL_x_Subsidy', '_ECL_x_RangeAnxiety', '_Income_x_Subsidy', '_Charging_Total',
                          '_log_Income', '_log_Commute', '_log_Charging_Total']
    engineered_numeric = [c for c in engineered_numeric if c in train.columns]
    # Convert numerics to string for TE columns
    for col in all_numeric_cols + engineered_numeric:
        cat_name = f"{col}_cat"
        train[cat_name] = train[col].fillna('NaN').astype(str)
        test[cat_name] = test[col].fillna('NaN').astype(str)
        orig_aligned[cat_name] = orig_aligned[col].fillna('NaN').astype(str) if col in orig_aligned.columns else 'NaN'

    train_num_cat_cols = [f"{c}_cat" for c in all_numeric_cols + engineered_numeric]

    freq_target_cols = CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
    freq_target_cols = [c for c in freq_target_cols if c in train.columns and c in test.columns]
    train, test = add_frequency_encoding(train, test, freq_target_cols)
    for col in freq_target_cols:
        if f"{col}_fe" in train.columns: orig_aligned[f"{col}_fe"] = 0.0

    train, test, bigram_cols = add_targeted_bigrams(train, test)
    for df in [orig_aligned]: df['ECL_bin'] = df['Environmental_Concern_Level'].fillna(3).astype(int).astype(str)
    for bn in bigram_cols:
        if 'RangeAnxiety' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Range_Anxiety_Level'].astype(str)
        elif 'Subsidy' in bn: orig_aligned[bn] = orig_aligned['ECL_bin'] + '_' + orig_aligned['Subsidy_Available'].astype(str)

    train, test, orig_aligned, proven_cols = add_proven_features(train, test, orig_aligned)

    for df in [train, test, orig_aligned]:
        df.drop(columns=['ECL_bin'], errors='ignore', inplace=True)

    # Feature selection
    print("   Feature selection (drop constants + perfectly-correlated)...")
    train, test, dropped = drop_redundant_features(train, test, CFG.TARGET)
    orig_aligned = orig_aligned.drop(columns=[c for c in dropped if c in orig_aligned.columns], errors='ignore')

    # Define TE columns
    TARGET_ENCODE_COLS = [c for c in (CATS + train_num_cat_cols + ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
                                       + bigram_cols + proven_cols) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']
    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   TE columns: {len(TARGET_ENCODE_COLS)}")

    # [3/5] TRAINING
    print(f"\n[3/5] Training RealMLP ({CFG.N_FOLDS}-Fold CV, orig concat + Triple TE)...")

    X      = train.drop([CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs  = np.zeros(len(y))
    test_probs = np.zeros(len(test_X))
    fold_scores = []

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    kf_orig = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    orig_splits = list(kf_orig.split(orig_aligned, y_orig))

    t0 = time.time()
    for fold, ((train_idx, val_idx), (or_train_idx, or_val_idx)) in enumerate(
            zip(kf.split(X), orig_splits)):

        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}:")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Per-fold: concat competition train + original
        orig_tr = orig_aligned.iloc[or_train_idx].copy()
        y_orig_tr = y_orig.iloc[or_train_idx].copy()
        X_train = pd.concat([X_train, orig_tr], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_orig_tr], axis=0).reset_index(drop=True)
        X_test_fold = test_X.copy()

        print(f"      Train (comp+orig): {X_train.shape} | Val: {X_val.shape} | Test: {X_test_fold.shape}")

        # ---- Triple Target Encoding (auto, 10, 100) ----
        te_cols_active = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
        for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
            te = TargetEncoder(target_type='binary', smooth=smooth_val, cv=CFG.N_FOLDS, shuffle=True, random_state=42)
            X_train_enc = te.fit_transform(X_train[te_cols_active], y_train).astype('float32')
            X_val_enc   = te.transform(X_val[te_cols_active]).astype('float32')
            X_test_enc  = te.transform(X_test_fold[te_cols_active]).astype('float32')
            for i, col in enumerate(te_cols_active):
                te_name = f"TE_{col}_{smooth_name}"
                X_train[te_name] = X_train_enc[:, i]
                X_val[te_name]   = X_val_enc[:, i]
                X_test_fold[te_name] = X_test_enc[:, i]

        # Drop string columns (RealMLP takes int64 cats + float32 nums)
        cols_to_drop = [c for c in te_cols_active if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_val   = X_val.drop(columns=cols_to_drop, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=cols_to_drop, errors='ignore')
        string_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or str(X_train[c].dtype) == 'string']
        X_train = X_train.drop(columns=string_cols, errors='ignore')
        X_val   = X_val.drop(columns=string_cols, errors='ignore')
        X_test_fold = X_test_fold.drop(columns=string_cols, errors='ignore')

        # Fill NaNs
        X_train = X_train.fillna(0)
        X_val   = X_val.fillna(0)
        X_test_fold = X_test_fold.fillna(0)

        # Identify cat cols (digit features, flags, bin features — small int cardinality)
        # RealMLP treats these as categoricals (onehot if ≤4 unique, else embedding)
        cat_col_names = []
        for c in X_train.columns:
            if c in CATS: cat_col_names.append(c); continue
            if 'digit' in c or c.startswith('is_') or c in ['_high_income', '_range_anxiety_high', '_ecl_max', '_ev_recipe']:
                cat_col_names.append(c); continue
            # Small-cardinality int features → cat
            if X_train[c].dtype in ['int8', 'int16', 'int32', 'int64']:
                if X_train[c].nunique() <= 50:
                    cat_col_names.append(c)

        # Factorize cat cols to int64
        X_train, X_val, X_test_fold = factorize_categoricals(X_train, X_val, X_test_fold, cat_col_names)

        if fold == 0:
            print(f"      Final feature count: {len(X_train.columns)} ({len(cat_col_names)} cat, {len(X_train.columns) - len(cat_col_names)} num)")

        # Train RealMLP
        model = RealMLP_TD_Classifier(**CONFIG)
        model.fit(X_train, y_train.values, X_val, y_val.values, cat_col_names=cat_col_names, X_test=X_test_fold)

        val_probs = model.best_val_probs_[:, 1]
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1] / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)

        elapsed = (time.time() - t0) / 60
        print(f"      AUC: {fold_auc:.5f} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test_fold, y_train, y_val, model
        torch.cuda.empty_cache()
        gc.collect()

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
    print(f"V17 RESULTS — RealMLP ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Architecture: PBLD + 8 ensemble + 3×256 + EMA + label smoothing")
    print(f"FE: V14 full pipeline (digits, smooth keys, bigrams, trigrams, freq, org means)")
    print(f"Improvements: V14 FE + orig concat + triple TE + 3 epochs")
    print(f"V14 XGB baseline: 0.946+ OOF → V17 RealMLP: {oof_cv:.5f} OOF")
    print(f"OOF CV (AUC): {oof_cv:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)