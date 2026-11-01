"""
S6E11 V{N} - {Name}: {the one change, in a clause} - {CPU | GPU (cuda)}
GOAL: {the open question this arm closes - one line}
Axis: {the ONE thing changed vs the parent; what stays frozen - one line}
Parent: {path to the archived script it clones}
Gate: {pre-declared numeric adoption test + the ruler it is judged against}
Cost: {expected minutes, device, and the losing outcome that is live}
Outputs: v{N}_oof.csv (id, pred) + v{N}_sub.csv (id, is_fraud) to /kaggle/working - exactly two writes.
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
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# Optional libs: install-then-import, never assume the accelerator image has them.
# import torch
# try:
#     from pytabkit import TabM_D_Classifier
# except ImportError:
#     import subprocess, sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])
#     from pytabkit import TabM_D_Classifier

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NUMBER = None          # FILL - the guard in section 8 refuses to run a template
    VERSION_NAME = "V{_}_Template"  # FILL - must match the filename stem
    EXP_ID = "S6E11_V{_}_Template"  # FILL - the ledger's join key
    PARENT = "{V#_ParentName}"      # FILL - the archived script this arm clones

    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e11/train.csv"
    TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e11/test.csv"
    ORIG_PATH  = "/kaggle/input/<owner>/<dataset>/credit_card_fraud_2026.csv"   # FILL - confirm the attached slug; drop the line if this arm reads no original data

    TARGET = 'is_fraud'
    N_FOLDS = 5
    RANDOM_SEED = 42
    SEED_REPRO = 2027              # the house reproduction seed - a gain that does not repeat is a tie

    BUILD = "2026-11-{dd}-r1"      # FILL - build stamp, printed in the head banner

    FEATURES = []                  # FILL - base feature list after DROP
    DROP = ['id', TARGET]

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.cuda.manual_seed_all(seed)


seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# ONE dict per arm, and the axis is the single key that differs from the parent.
# Copy kwargs from our own archived scripts, never from documentation memory:
#   XGBoost    S6E10_V1, S6E10_V8 (leaf-wise), S6E10_V20 (soft labels)
#   LightGBM   S6E10_V15, S6E10_V19, S6E9_V46
#   CatBoost   S6E10_V4, S6E10_V16, S6E10_V42, S6E9_V8   (task_type GPU; eval_metric stays Logloss)
#   RealMLP    S6E10_V5-V7, V17, V18, V44, V48, V54, S6E9_V47
#   TabM       S6E10_V53, S6E9_V6, S6E9_V29               (convergence is early stopping)
#   kNN        S6E10_V38, S6E9_V15
XGB_PARAMS = dict(
    n_estimators=6000,
    max_leaves=64,
    learning_rate=0.04,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.85,
    reg_lambda=5.0,
    reg_alpha=0.05,
    tree_method='hist',
    device='cuda',
    eval_metric='aucpr',
    early_stopping_rounds=150,
    enable_categorical=True,
    random_state=CFG.RANDOM_SEED,
    n_jobs=-1,
)

# Imbalance lever - one per arm, never combined with another axis in the same run.
# scale_pos_weight=(1 - pos_rate) / pos_rate  # measured on the FIT rows of each fold, not on the frame
# scale_pos_weight=1.0                        # the parent's value, for a control run

# =============================================================================
# 5. METRIC - ROC AUC
# =============================================================================
def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def pr_auc_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
# Target-free work only. Anything that reads y belongs in section 7.
def build_features(df):
    out = df.copy()
    # FILL: velocity / time-of-day / mismatch / merchant-risk transforms, all computed
    #       per row from its own columns (or from the pooled train+test frame with no label).
    return out

# =============================================================================
# 7. PER-FOLD FEATURE BUILDER
# =============================================================================
def per_fold_features(X_fit, y_fit, X_val, X_test, base):
    """Refit every label-consuming producer on fit rows only, then map it onto val and test.

    Returns (X_fit_aug, X_val_aug, X_test_aug, fitted_encoder). A vector built here from
    the whole train frame leaks the validation labels and makes the fold CV meaningless.
    """
    # FILL: TargetEncoder(smooth=20, cv=5) fitted on (X_fit, y_fit) only; frequency and
    #       rarity maps from fit rows; any target-mean / EV column from fit rows.
    return X_fit, X_val, X_test, None

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    if CFG.VERSION_NUMBER is None:
        raise SystemExit("TEMPLATE: fill CFG.VERSION_NUMBER / VERSION_NAME / EXP_ID before running.")

    t0_all = time.time()
    print("=" * 80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: GPU (cuda) | "
          f"Folds: {CFG.N_FOLDS} | Seed: {CFG.RANDOM_SEED} | Build: {CFG.BUILD}")
    print(f"Axis: {CFG.VERSION_NAME} - restate THIS arm's axis, not the parent's")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    y = train[CFG.TARGET].astype(np.int8)
    train_id = train['id'].to_numpy()
    test_id = test['id'].to_numpy()
    print(f"   Train: {train.shape} | Test: {test.shape} | Positive rate: {y.mean():.6f} "
          f"({int(y.sum()):,} of {len(y):,})")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print("\n[2/5] Feature Engineering...")
    train_fe = build_features(train)
    test_fe = build_features(test)
    FEATURES = [c for c in train_fe.columns if c not in CFG.DROP]
    # All numeric matrices float32 - the columnar dtype discipline every tree arm assumes.
    num_cols = [c for c in FEATURES if train_fe[c].dtype.kind in 'if']
    train_fe[num_cols] = train_fe[num_cols].astype(np.float32)
    test_fe[num_cols] = test_fe[num_cols].astype(np.float32)
    print(f"   {len(FEATURES)} columns fed to the model")

    # Pre-flight, before any fold fits: build the object and read every config key back.
    probe = xgb.XGBClassifier(**XGB_PARAMS)
    missing = [k for k in XGB_PARAMS if k not in probe.get_params()]
    if missing:
        raise SystemExit(f"Pre-flight: constructor dropped {missing}")
    print(f"   Pre-flight: {len(XGB_PARAMS)} params all read back off the built object")

    # =========================================================================
    # [3/5] TRAINING (5-Fold CV, per-fold encoders)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, per-fold TE)...")
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    oof_probs = np.zeros(len(train_fe), dtype=np.float64)
    test_probs = np.zeros(len(test_fe), dtype=np.float64)
    fold_scores, fold_pr, best_iters, prep_times, fit_times = [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y), start=1):
        fold_start = time.time()
        print(f"\n   Fold {fold}/{CFG.N_FOLDS}")

        X_fit = train_fe.iloc[tr_idx].reset_index(drop=True)
        y_fit = y.iloc[tr_idx].reset_index(drop=True)
        X_val = train_fe.iloc[va_idx].reset_index(drop=True)
        y_val = y.iloc[va_idx].reset_index(drop=True)

        prep_start = time.time()
        X_fit, X_val, X_test_fold, _enc = per_fold_features(
            X_fit, y_fit, X_val, test_fe, FEATURES)
        prep_s = time.time() - prep_start
        prep_times.append(prep_s)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_fit, y_fit,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        fit_s = time.time() - fold_start - prep_s
        fit_times.append(fit_s)

        val_probs = model.predict_proba(X_val)[:, 1].astype(np.float64)
        oof_probs[va_idx] = val_probs
        test_probs += model.predict_proba(X_test_fold)[:, 1].astype(np.float64) / CFG.N_FOLDS

        fold_auc = auc_score(y_val.values, val_probs)
        fold_scores.append(fold_auc)
        fold_pr.append(pr_auc_score(y_val.values, val_probs))
        best_iter = (getattr(model, 'best_iteration', None)
                     if getattr(model, 'best_iteration', None) is not None
                     else XGB_PARAMS['n_estimators'])
        best_iters.append(best_iter)

        # Read-back AFTER the fit: construction accepts things the core rejects or ignores.
        kept = model.get_params()
        for k in ('max_leaves', 'learning_rate', 'subsample'):
            assert kept[k] == XGB_PARAMS[k], f"fit dropped {k}: {kept[k]} != {XGB_PARAMS[k]}"

        fold_s = time.time() - fold_start
        elapsed = (time.time() - t0_all) / 60
        print(f"      ROC AUC: {fold_auc:.6f} | PR-AUC: {fold_pr[-1]:.6f} | BestIter: {best_iter} | "
              f"Time: {fold_s:.0f}s (prep {prep_s:.0f}s) | Total: {elapsed:.1f}min")

        if fold == 1:
            canary_min = (time.time() - fold_start) / 60
            CEILING_MIN = 75          # FILL - declared ceiling per fold, before the run
            if canary_min > CEILING_MIN:
                raise SystemExit(
                    f"Wall-time canary: fold 1 took {canary_min:.1f} min > {CEILING_MIN}. "
                    f"Re-run at the fallback config - never change it mid-run.")
            print(f"      Wall-time canary: {canary_min:.1f} min/fold (ceiling {CEILING_MIN})")
            imp = (pd.DataFrame({'feature': X_fit.columns,
                                 'importance': model.feature_importances_})
                   .sort_values('importance', ascending=False))
            print("      Top-15 feature importances (fold 1):")
            print(imp.head(15).to_string(index=False))

        del X_fit, X_val, X_test_fold, y_fit, y_val, model
        gc.collect()
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

    oof_cv = auc_score(y.values, oof_probs)
    oof_pr = pr_auc_score(y.values, oof_probs)
    print(f"\n   OOF ROC AUC: {oof_cv:.6f} | OOF PR-AUC: {oof_pr:.6f}")
    print(f"   Fold ROC AUCs: {[f'{s:.6f}' for s in fold_scores]}")
    print(f"   Fold mean +/- std: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print(f"   Pooled minus fold-mean: {oof_cv - np.mean(fold_scores):+.6f}")
    print(f"   Best iters: {best_iters}")

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
    print(f"   [SAVED] {out_dir}/{stem}_sub.csv (id, is_fraud)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
    print(f"   Axis: restate THIS arm's axis in one clause")
    print(f"   OOF ROC AUC: {oof_cv:.6f} | OOF PR-AUC: {oof_pr:.6f}")
    print(f"   Fold ROC AUCs: {[f'{s:.6f}' for s in fold_scores]}")
    print(f"   Fold mean +/- std: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print(f"   Fold PR-AUCs: {[f'{s:.6f}' for s in fold_pr]}")
    print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold (mean) "
          f"| total {total_time_min:.1f} min")
    print(f"   Saved: {stem}_oof.csv + {stem}_sub.csv")
