"""
S6E9 V18 - Hill Climber Ensemble (CPU)
================================================================================
Strategy: Forward-stepwise hill climbing over all available OOF predictions

After 17 single-model versions all hitting ~0.946 OOF ceiling, this is the
final move. Every recent tabular playground winner used ensembles — we have
5 diverse OOFs ready:
  - V3 (LGBM): ~0.94601 OOF
  - V8 (CatBoost Ordered): ~0.94583 OOF
  - V10 (LGBM + targeted): ~0.94597 OOF
  - V14 (XGB depth=3 + pseudo-labels): ~0.94608 OOF
  - V17 (RealMLP): ~0.94582 OOF

Same score, DIFFERENT errors. Hill climber finds optimal combination.

Algorithm:
1. Start with best single model (highest OOF AUC)
2. Try adding each remaining model with various weights (0.01 to 1.00)
3. Keep the addition that maximizes OOF AUC
4. Repeat until no improvement
5. Apply same weights to test predictions

Also computes for comparison:
- Simple average (all weights equal)
- Rank average (rank each model's predictions, average ranks)
- Logit-space average (Anhadm's S6E9 method — average in logit space)
- Ridge meta-learner (RidgeCV on OOFs)

Expected: +0.001 to +0.003 LB over best single model

Device: CPU | Est. Time: ~2-5 min
Golden Rules: KFold(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print("S6E9 V18 - Hill Climber Ensemble (CPU)")
t0 = time.time()

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v18"
    EXP_ID = "S6E9_V18_HillClimber"
    DEVICE = "CPU"

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "Dataset"
    PREVIOUS_DIR = PROJECT_ROOT / "Previously trained Files"

    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    ORIG_PATH = DATA_DIR / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    OOF_DIR = PREVIOUS_DIR / "oof"
    SUB_DIR = PREVIOUS_DIR / "sub"

    TARGET = 'Will_Buy_EV'
    RANDOM_SEED = 42

    # Hill climber params
    # 0.05 keeps the local CPU run practical. Set S6E9_HILL_WEIGHT_STEP=0.01
    # to reproduce the original Kaggle grid exactly.
    WEIGHT_STEP = float(os.environ.get("S6E9_HILL_WEIGHT_STEP", "0.05"))
    MAX_WEIGHT = 1.0
    MIN_IMPROVEMENT = 1e-6  # Minimum AUC improvement to accept a new model

    OUT_DIR = PROJECT_ROOT / "Outputs"

# =============================================================================
# 3. SEED
# =============================================================================
np.random.seed(CFG.RANDOM_SEED)

# =============================================================================
# 4. METRIC
# =============================================================================
def auc_score(y_true, y_probs):
    return roc_auc_score(y_true, y_probs)

# =============================================================================
# 5. LOAD TRAIN DATA (for target labels)
# =============================================================================
print("\n[1/4] Loading train data for target labels...")
train = pd.read_csv(CFG.TRAIN_PATH)
target2idx = {'No': 0, 'Yes': 1}
train[CFG.TARGET] = train[CFG.TARGET].astype(str).str.strip().str.title().map(target2idx)
y = train[CFG.TARGET].values
train_id = train['id'].values
print(f"   Train: {len(y):,} rows | Pos rate: {y.mean():.4f}")

# Load test data (for IDs)
test = pd.read_csv(CFG.TEST_PATH)
test_id = test['id'].values
print(f"   Test: {len(test):,} rows")

# =============================================================================
# 6. LOAD ALL AVAILABLE OOFs
# =============================================================================
print("\n[2/4] Loading OOF predictions...")

# Try to load OOFs from V1 through V17
oof_dfs = {}
sub_dfs = {}
for v in range(1, 18):
    oof_path = CFG.OOF_DIR / f"oof_v{v}.csv"
    sub_path = CFG.SUB_DIR / f"sub_v{v}.csv"
    if os.path.exists(oof_path) and os.path.exists(sub_path):
        try:
            oof_df = pd.read_csv(oof_path)
            sub_df = pd.read_csv(sub_path)
            # Find prediction column (not 'id')
            oof_pred_col = [c for c in oof_df.columns if c != 'id'][0]
            sub_pred_col = [c for c in sub_df.columns if c != 'id'][0]
            oof_dfs[f"V{v}"] = oof_df[oof_pred_col].values
            sub_dfs[f"V{v}"] = sub_df[sub_pred_col].values
            # Verify alignment with train_id
            if not np.array_equal(oof_df['id'].values, train_id):
                print(f"   ⚠ V{v} OOF id mismatch — skipping")
                del oof_dfs[f"V{v}"]; del sub_dfs[f"V{v}"]
                continue
            if not np.array_equal(sub_df['id'].values, test_id):
                print(f"   ⚠ V{v} sub id mismatch — skipping")
                del oof_dfs[f"V{v}"]; del sub_dfs[f"V{v}"]
                continue
            oof_auc = auc_score(y, oof_dfs[f"V{v}"])
            print(f"   V{v}: OOF AUC = {oof_auc:.5f} | OOF shape: {oof_dfs[f'V{v}'].shape}")
        except Exception as e:
            print(f"   V{v}: Failed to load — {e}")

if len(oof_dfs) < 2:
    print(f"\n❌ Need at least 2 OOFs for ensemble. Found {len(oof_dfs)}.")
    print(f"   Available OOFs: {list(oof_dfs.keys())}")
    raise SystemExit("Insufficient OOFs for ensemble")

print(f"\n   Loaded {len(oof_dfs)} OOFs: {list(oof_dfs.keys())}")

# Build OOF and test matrices
model_names = list(oof_dfs.keys())
n_models = len(model_names)
OOF = np.column_stack([oof_dfs[m] for m in model_names])  # (n_train, n_models)
TEST = np.column_stack([sub_dfs[m] for m in model_names])  # (n_test, n_models)
print(f"   OOF matrix: {OOF.shape} | TEST matrix: {TEST.shape}")

# =============================================================================
# 7. ENSEMBLE METHODS
# =============================================================================
print(f"\n[3/4] Computing ensemble methods...")

results = {}

# --- Method 1: Simple Average ---
simple_avg_oof = OOF.mean(axis=1)
simple_avg_test = TEST.mean(axis=1)
simple_avg_auc = auc_score(y, simple_avg_oof)
results['simple_avg'] = {
    'auc': simple_avg_auc,
    'oof': simple_avg_oof,
    'test': simple_avg_test,
    'weights': np.ones(n_models) / n_models,
}
print(f"   Simple average:        OOF AUC = {simple_avg_auc:.5f}")

# --- Method 2: Rank Average ---
# Convert each model's predictions to ranks, then average ranks
def rank_average(preds):
    """Convert predictions to ranks (0-1 range)."""
    return pd.Series(preds).rank().values / len(preds)

OOF_RANKS = np.column_stack([rank_average(OOF[:, i]) for i in range(n_models)])
TEST_RANKS = np.column_stack([rank_average(TEST[:, i]) for i in range(n_models)])
rank_avg_oof = OOF_RANKS.mean(axis=1)
rank_avg_test = TEST_RANKS.mean(axis=1)
rank_avg_auc = auc_score(y, rank_avg_oof)
results['rank_avg'] = {
    'auc': rank_avg_auc,
    'oof': rank_avg_oof,
    'test': rank_avg_test,
    'weights': np.ones(n_models) / n_models,
}
print(f"   Rank average:          OOF AUC = {rank_avg_auc:.5f}")

# --- Method 3: Logit-Space Average (Anhadm's method) ---
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

OOF_LOGITS = np.column_stack([logit(OOF[:, i]) for i in range(n_models)])
TEST_LOGITS = np.column_stack([logit(TEST[:, i]) for i in range(n_models)])
logit_avg_oof = sigmoid(OOF_LOGITS.mean(axis=1))
logit_avg_test = sigmoid(TEST_LOGITS.mean(axis=1))
logit_avg_auc = auc_score(y, logit_avg_oof)
results['logit_avg'] = {
    'auc': logit_avg_auc,
    'oof': logit_avg_oof,
    'test': logit_avg_test,
    'weights': np.ones(n_models) / n_models,
}
print(f"   Logit-space average:   OOF AUC = {logit_avg_auc:.5f}")

# --- Method 4: Ridge Meta-Learner ---
# Use RidgeCV on OOF logits (5-fold internal CV to find best alpha)
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(OOF_LOGITS, y)
ridge_oof = sigmoid(ridge.predict(OOF_LOGITS))
ridge_test = sigmoid(ridge.predict(TEST_LOGITS))
ridge_auc = auc_score(y, ridge_oof)
results['ridge_meta'] = {
    'auc': ridge_auc,
    'oof': ridge_oof,
    'test': ridge_test,
    'weights': ridge.coef_,
}
print(f"   Ridge meta-learner:    OOF AUC = {ridge_auc:.5f} (alpha={ridge.alpha_})")

# --- Method 5: Forward-Stepwise Hill Climber (main method) ---
print(f"\n   Hill climber (forward-stepwise):")
def hill_climb(oof_matrix, y_true, weight_step=0.01, min_improvement=1e-6):
    """
    Forward-stepwise hill climbing.
    1. Start with best single model
    2. Try adding each remaining model with weights from 0 to 1
    3. Keep the addition that maximizes AUC
    4. Repeat until no improvement
    Returns: weights array (n_models,)
    """
    n_models = oof_matrix.shape[1]
    weights = np.zeros(n_models)

    # Start with best single model
    best_aucs = [auc_score(y_true, oof_matrix[:, i]) for i in range(n_models)]
    best_model = np.argmax(best_aucs)
    weights[best_model] = 1.0
    current_pred = oof_matrix[:, best_model].copy()
    current_auc = best_aucs[best_model]
    print(f"      Start: {model_names[best_model]} (AUC={current_auc:.5f})")

    iteration = 0
    while True:
        iteration += 1
        best_improvement = 0
        best_add_model = -1
        best_add_weight = 0
        best_new_auc = current_auc

        # Try adding each model not yet at weight 1
        for m in range(n_models):
            if weights[m] >= 1.0 - 1e-9:
                continue  # Already at max weight

            # Try weights from weight_step to 1.0
            # We mix: new_pred = (1-w) * current_pred + w * model_m_pred
            # But also need to renormalize if other weights exist
            # Simpler: try adding w to model m, then renormalize
            for w in np.arange(weight_step, 1.0 + weight_step, weight_step):
                trial_weights = weights.copy()
                trial_weights[m] += w
                # Renormalize
                if trial_weights.sum() > 0:
                    trial_weights = trial_weights / trial_weights.sum()
                trial_pred = oof_matrix @ trial_weights
                trial_auc = auc_score(y_true, trial_pred)
                improvement = trial_auc - current_auc
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_add_model = m
                    best_add_weight = w
                    best_new_auc = trial_auc

        if best_improvement < min_improvement or best_add_model == -1:
            print(f"      Iteration {iteration}: No improvement. Stopping.")
            break

        # Apply best addition
        weights[best_add_model] += best_add_weight
        weights = weights / weights.sum()  # Renormalize
        current_pred = oof_matrix @ weights
        current_auc = best_new_auc
        print(f"      Iteration {iteration}: +{model_names[best_add_model]} (w={best_add_weight:.2f}) → AUC={current_auc:.5f}")

    return weights, current_auc

hill_weights, hill_auc = hill_climb(OOF, y, weight_step=CFG.WEIGHT_STEP, min_improvement=CFG.MIN_IMPROVEMENT)
hill_oof = OOF @ hill_weights
hill_test = TEST @ hill_weights
results['hill_climber'] = {
    'auc': hill_auc,
    'oof': hill_oof,
    'test': hill_test,
    'weights': hill_weights,
}
print(f"   Hill climber final:    OOF AUC = {hill_auc:.5f}")
print(f"   Weights: ", end="")
for m, w in zip(model_names, hill_weights):
    if w > 0.001:
        print(f"{m}={w:.3f} ", end="")
print()

# --- Method 6: Hill Climber in Logit Space ---
print(f"\n   Hill climber (logit space):")
def hill_climb_logit(oof_logits, y_true, test_logits, weight_step=0.01, min_improvement=1e-6):
    """Hill climber in logit space — predictions are sigmoid(weighted logit avg)."""
    n_models = oof_logits.shape[1]
    weights = np.zeros(n_models)

    best_aucs = [auc_score(y_true, sigmoid(oof_logits[:, i])) for i in range(n_models)]
    best_model = np.argmax(best_aucs)
    weights[best_model] = 1.0
    current_pred = sigmoid(oof_logits[:, best_model])
    current_auc = best_aucs[best_model]
    print(f"      Start: {model_names[best_model]} (AUC={current_auc:.5f})")

    iteration = 0
    while True:
        iteration += 1
        best_improvement = 0
        best_add_model = -1
        best_add_weight = 0
        best_new_auc = current_auc

        for m in range(n_models):
            if weights[m] >= 1.0 - 1e-9:
                continue
            for w in np.arange(weight_step, 1.0 + weight_step, weight_step):
                trial_weights = weights.copy()
                trial_weights[m] += w
                if trial_weights.sum() > 0:
                    trial_weights = trial_weights / trial_weights.sum()
                trial_pred = sigmoid(oof_logits @ trial_weights)
                trial_auc = auc_score(y_true, trial_pred)
                improvement = trial_auc - current_auc
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_add_model = m
                    best_add_weight = w
                    best_new_auc = trial_auc

        if best_improvement < min_improvement or best_add_model == -1:
            print(f"      Iteration {iteration}: No improvement. Stopping.")
            break

        weights[best_add_model] += best_add_weight
        weights = weights / weights.sum()
        current_pred = sigmoid(oof_logits @ weights)
        current_auc = best_new_auc
        print(f"      Iteration {iteration}: +{model_names[best_add_model]} (w={best_add_weight:.2f}) → AUC={current_auc:.5f}")

    return weights, current_auc

hill_logit_weights, hill_logit_auc = hill_climb_logit(OOF_LOGITS, y, TEST_LOGITS, weight_step=CFG.WEIGHT_STEP, min_improvement=CFG.MIN_IMPROVEMENT)
hill_logit_oof = sigmoid(OOF_LOGITS @ hill_logit_weights)
hill_logit_test = sigmoid(TEST_LOGITS @ hill_logit_weights)
results['hill_climber_logit'] = {
    'auc': hill_logit_auc,
    'oof': hill_logit_oof,
    'test': hill_logit_test,
    'weights': hill_logit_weights,
}
print(f"   Hill climber logit:    OOF AUC = {hill_logit_auc:.5f}")
print(f"   Weights: ", end="")
for m, w in zip(model_names, hill_logit_weights):
    if w > 0.001:
        print(f"{m}={w:.3f} ", end="")
print()

# =============================================================================
# 8. SELECT BEST METHOD
# =============================================================================
print(f"\n[4/4] Selecting best ensemble method...")

print(f"\n{'='*60}")
print(f"ENSEMBLE RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Method':<25} {'OOF AUC':<12} {'Δ vs Best Single':<18}")
print(f"{'-'*60}")

# Best single model
best_single_auc = max([auc_score(y, OOF[:, i]) for i in range(n_models)])
best_single_idx = np.argmax([auc_score(y, OOF[:, i]) for i in range(n_models)])
print(f"{'Best Single (' + model_names[best_single_idx] + ')':<25} {best_single_auc:<12.5f} {'(baseline)':<18}")

for method_name, res in results.items():
    delta = res['auc'] - best_single_auc
    print(f"{method_name:<25} {res['auc']:<12.5f} {delta:+.5f}{'  ★' if delta > 0 else ''}")

# Select best
best_method = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest method: {best_method[0]} (OOF AUC = {best_method[1]['auc']:.5f})")
print(f"Best single: {model_names[best_single_idx]} (OOF AUC = {best_single_auc:.5f})")
print(f"Improvement: {best_method[1]['auc'] - best_single_auc:+.5f}")

# =============================================================================
# 9. SAVE BEST ENSEMBLE
# =============================================================================
print(f"\nSaving outputs...")
os.makedirs(CFG.OUT_DIR, exist_ok=True)

# Save best ensemble OOF and submission
oof_df = pd.DataFrame({'id': train_id, 'pred': best_method[1]['oof']})
oof_df.to_csv(f"{CFG.OUT_DIR}/oof_{CFG.VERSION_NAME}.csv", index=False)
sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: best_method[1]['test']})
sub_df.to_csv(f"{CFG.OUT_DIR}/sub_{CFG.VERSION_NAME}.csv", index=False)
print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv + sub_{CFG.VERSION_NAME}.csv")
print(f"   Method: {best_method[0]}")

# Save all ensemble OOFs for downstream use
for method_name, res in results.items():
    pd.DataFrame({'id': train_id, 'pred': res['oof']}).to_csv(
        f"{CFG.OUT_DIR}/oof_{CFG.VERSION_NAME}_{method_name}.csv", index=False)
    pd.DataFrame({'id': test_id, CFG.TARGET: res['test']}).to_csv(
        f"{CFG.OUT_DIR}/sub_{CFG.VERSION_NAME}_{method_name}.csv", index=False)
print(f"   [SAVED] All method OOFs and subs for analysis")

# =============================================================================
# 10. FINAL RESULTS
# =============================================================================
print(f"\n{'='*80}")
print(f"V18 RESULTS — Hill Climber Ensemble")
print(f"{'='*80}")
print(f"Models ensembled: {len(model_names)}")
for m, single_auc in zip(model_names, [auc_score(y, OOF[:, i]) for i in range(n_models)]):
    print(f"  {m}: OOF AUC = {single_auc:.5f}")
print(f"\nBest single model: {model_names[best_single_idx]} ({best_single_auc:.5f})")
print(f"Best ensemble:     {best_method[0]} ({best_method[1]['auc']:.5f})")
print(f"Improvement:       {best_method[1]['auc'] - best_single_auc:+.5f}")
print(f"\nBest ensemble weights:")
for m, w in zip(model_names, best_method[1]['weights']):
    if w > 0.001:
        print(f"  {m}: {w:.4f}")
print(f"\nExpected LB: ~{best_method[1]['auc']:.4f} (typically matches OOF for ensemble)")
print(f"Best single LB was: 0.94636 (V10)")
print(f"\nTotal time: {time.time() - t0:.1f}s")
print("="*80)
