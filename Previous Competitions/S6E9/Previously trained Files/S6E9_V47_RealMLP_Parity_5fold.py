"""
S6E9 V47 — RealMLP 8-member, V40's 341-column matrix, 5-fold (the plan's 5-fold roster)

Strategy:
    V49 (its 10-fold parent, now logged at OOF 0.945940 / LB 0.94616) closed the neural
    family fairly at the house 10-fold ruler: the published view-G RealMLP recipe on V40's
    matrix reached −0.000268 vs V40's 0.946208, below the 0.946150 leg gate the script
    carried. V47 is the geometry rerun that same roster requires: V49's exact architecture
    (PBLD periodic-bias numeric embeddings, entity embeddings for categoricals with
    one-hot below 4 uniques, 3×256 SiLU 8-way net, EMA 0.997875, label smoothing 0.04 on a
    cosine schedule, AdamW lr 0.01 / wd 0.013, front scaling layer, 2 epochs, batch 256)
    on V40's exact 341-column view-A matrix under the 5-fold house ruler instead of 10.
    Single model only: one net per fold, no ensemble across folds, no multi-seed.

References:
    V49 0.945940 / LB 0.94616 (this architecture at 10 folds, the parent). V40 0.946208 /
    LB 0.94646 (this matrix under XGBoost, the family control). V17 0.94582-era (this
    architecture on the older V14 matrix, 5 folds, 3 epochs). Published view G 0.946139 /
    0.946182 (the claim V49 failed to reproduce).

V47 Change from V49:
    Exactly one variable: N_FOLDS = 10 -> 5, with the TargetEncoder's inner cv following
    (it is keyed off CFG.N_FOLDS in the parent, so it moves with it: cv=10 -> cv=5).
    Feature engineering, the window/ladder block, the triple-TE smoothings auto/10/100,
    the per-fold original-row concat under KFold(5, rs=42), the net config and the 2-epoch
    schedule are all V49's, so V49 minus V47 is the fold-geometry effect on this net.
    Each model trains on 80% of the labels (V49's 10-fold models saw 90%), which is the
    direction V30->V28 measured at +0.000097 for the trees.

Parameters:
    As V49. A fold-0 input-shape canary prints the categorical/numeric split and
    parameter count before training and raises on any mismatch; a fold-0 time projection
    (elapsed + per-fold rate x remaining folds) aborts rather than being killed mid-session.
    Outputs: oof_v47.csv (id, pred), sub_v47.csv (id, Will_Buy_EV). Estimate ~55-75 min GPU
    (V49 measured 10.3-10.9 min/fold at 10 folds; 5 folds is ~half the booster time).
    GATE: solo 5-fold OOF >= 0.94615 (the leg parity bar) to survive; below it the net
    family is closed at the 5-fold ruler too.

Dataset Structure:
    train.csv 668,665 / test.csv 286,571 / original 10,000 rows concat'd per fold under its
    own KFold(5, rs=42); No/Yes -> 0/1.
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
print(f"PyTorch version: {torch.__version__}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# 1. CONFIG
class CFG:
    VERSION_NAME = "v47"
    EXP_ID = "S6E9_V47_RealMLP_Parity_5fold"

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e9/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e9/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety/EV_Adoption_and_Range_Anxiety_Dataset.csv"

    TARGET = 'Will_Buy_EV'

    N_FOLDS = 5
    RANDOM_SEED = 42

    TIME_CAP_MIN = 130

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

NET_CONFIG = {
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
    "epochs": 2,
    "train_bs": 256, "eval_bs": 10240,
    "verbosity": 1,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "device": "cuda", "random_state": CFG.RANDOM_SEED,
}


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
        self.params = {**NET_CONFIG, **kwargs}

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
    ] + bigram_cols + LADDER_COLS) if c in train.columns]

    FEATURES = [c for c in test.columns if c != 'id']

    print(f"\n   Total features: {len(FEATURES)}")
    print(f"   Columns to Triple-TE: {len(TARGET_ENCODE_COLS)}")
    print(f"     - Generator-lift / novelty cols: {len([c for c in lift_cols if c in FEATURES])}")
    print(f"     - Targeted bigram cols: {len([c for c in bigram_cols if c in TARGET_ENCODE_COLS])}")
    print(f"   View-A encodings per fold: {len(WIN_INC)} income-window + {len(WIN_KM)} commute-window + "
          f"{len(WIN_GRP)} group-window cols, {len(LADDER_COLS) * len(TE_SMOOTHS)} ladder TE cols = "
          f"{len(WIN_INC) + len(WIN_KM) + len(WIN_GRP) + len(LADDER_COLS) * len(TE_SMOOTHS)} new cols")

    print(f"\n   Net self-check before any training...")
    try:
        _ = torch.zeros(8, 4, device='cpu'); torch.cuda.empty_cache()
        print(f"   torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
        assert torch.cuda.is_available(), "no GPU - a RealMLP run on CPU would overrun the cap"
    except AssertionError as _e:
        raise SystemExit(f"ABORT: {_e}")
    assert callable(getattr(RealMLP_TD_Classifier, 'fit', None)), "net class did not import cleanly"

    print(f"\n[3/5] Training RealMLP ({CFG.N_FOLDS}-Fold CV, GPU, orig concat + Triple TE + windows, view-G config)...")

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

        X_train = X_train.fillna(0); X_val = X_val.fillna(0); X_test_fold = X_test_fold.fillna(0)
        cat_col_names = [c for c in cols_fold if c in CATS]
        cat_col_names += [c for c in cols_fold if c not in cat_col_names and (
            'digit' in c or c.startswith('is_') or c.startswith('novel')
            or (X_train[c].dtype.kind in 'iib' and X_train[c].nunique() <= 50))]
        X_train, X_val, X_test_fold = factorize_categoricals(X_train, X_val, X_test_fold, cat_col_names)
        num_col_names = [c for c in cols_fold if c not in cat_col_names]

        model = RealMLP_TD_Classifier(**NET_CONFIG)
        model.fit(X_train[cols_fold], y_train.values, X_val[cols_fold], y_val.values,
                  cat_col_names=cat_col_names, X_test=X_test_fold[cols_fold])
        best_iter = NET_CONFIG['epochs']

        val_probs = np.asarray(model.best_val_probs_[:, 1], dtype='float64')
        oof_probs[val_idx] = val_probs
        test_probs += np.asarray(model.predict_proba(X_test_fold[cols_fold])[:, 1],
                                 dtype='float64') / CFG.N_FOLDS

        best_iters.append(best_iter)
        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        print(f"      AUC: {fold_auc:.5f} | BestIter {best_iter} | "
              f"Time {(time.time()-fold_start)/60:.1f} min | Total {(time.time()-t0_all)/60:.1f} min")

        if fold == 0:
            print(f"      Net input shapes: train {X_train[cols_fold].shape} | "
                  f"val {X_val[cols_fold].shape} | test {X_test_fold[cols_fold].shape}")
            print(f"      Split: {len(cat_col_names)} categorical (factorised int64) + "
                  f"{len(num_col_names)} numeric (scaled) = {len(cols_fold)}")
            print(f"      Parameters: {sum(q.numel() for q in model.model_.parameters()):,} | "
                  f"device {next(model.model_.parameters()).device} | "
                  f"epochs {NET_CONFIG['epochs']} | n_ens {NET_CONFIG['n_ens']}")

        if fold == 0:
            _fold_min = (time.time() - fold_start) / 60.0
            _projected = (time.time() - t0_all) / 60.0 + _fold_min * (CFG.N_FOLDS - 1)
            print(f"      fold 1 took {_fold_min:.1f} min; projected total {_projected:.1f} min "
                  f"(cap {CFG.TIME_CAP_MIN} min)")
            if _projected > CFG.TIME_CAP_MIN:
                raise SystemExit(f"ABORT after fold 1: projected {_projected:.0f} min exceeds the "
                                 f"{CFG.TIME_CAP_MIN} min cap. Rerun with a smaller n_ens; do not "
                                 f"silently change the ruler.")

        del model, X_train, X_val, X_test_fold
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
    print(f"[5/5] V47 RESULTS — Single Model, {CFG.N_FOLDS}-Fold RealMLP (GPU, view-G config, V40 matrix)")
    print(f"{'='*80}")
    try:
        print(f"OOF CV (AUC): {cv_score:.6f}")
        print(f"Fold mean {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
        print(f"Per-fold epochs: {best_iters}")
        print(f"Columns used: {cols_used} ({len(FEATURES)} base + {n_te} Triple TE + {len(window_blocks)} window cols)")
        print(f"Test predictions: min {test_probs.min():.6f} | max {test_probs.max():.6f}")
        print("\nReference points (is this family at parity at the 5-fold ruler, and is it a leg?):")
        print(f"  V49 0.945940 = this architecture at 10 folds -> geometry effect: {cv_score - 0.945940:+.6f}")
        print(f"  V40 0.946208 = this matrix under XGBoost (10-fold) -> family effect: {cv_score - 0.946208:+.6f} (geometry caveat: 5-fold OOFs read ~+0.0001 above the 10-fold OOF of the same model)")
        print("GATE: 5-fold OOF >= 0.94615 (the leg parity bar) to survive; below it the net family is closed at the 5-fold ruler too.")
    except Exception as _e:
        print(f"   (results diagnostics skipped: {_e}) — artefacts were already saved")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*80}")