"""
S6E11 V{N} - {Name}: {the one change, in a clause} - {CPU (local) | CPU (Kaggle)}
GOAL: {the open question this combiner closes - one line}
Axis: {the ONE thing changed vs the parent blend: the member set, or the combiner - never both}
Parent: {path to the archived blend it clones, or the single-model arm it blends into}
Gate: {pre-declared numeric adoption test + the ruler it is judged against}
Cost: {expected minutes, device, and the losing outcome that is live}
Outputs: v{N}_oof.csv (id, pred) + v{N}_sub.csv (id, is_fraud) - exactly two writes.
"""

import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


class CFG:
    VERSION_NUMBER = None          # FILL - the guard in main refuses to run a template
    VERSION_NAME = "V{_}_TemplateBlend"
    EXP_ID = "S6E11_V{_}_TemplateBlend"
    PARENT = "{V#_ParentBlend}"

    TARGET = 'is_fraud'
    MEMBERS = ["v1", "v2"]          # FILL - the vectors blended; every one must exist in OOF_DIR and SUB_DIR
    REFERENCES = ["v1"]             # FILL - control / best-OOF / best-LB, paired against these

    # Local mode (the house runs combiners on the checkout, seconds of CPU, no fit):
    ROOT = Path(r"C:\Users\BlamerX\Videos\GitHub\S6E11")
    TRAIN_PATH = ROOT / "Dataset" / "train.csv"
    OOF_DIR = ROOT / "Previously trained files" / "oof"
    SUB_DIR = ROOT / "Previously trained files" / "sub"

    # Kaggle mode - replace the four lines above with the attached paths:
    # ROOT = Path("/kaggle/working")
    # TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e11/train.csv"
    # OOF_DIR = Path("/kaggle/input/{kernel-or-dataset}/oof")
    # SUB_DIR = Path("/kaggle/input/{kernel-or-dataset}/sub")

    BUILD = "2026-11-{dd}-r1"


def _logit(p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


def _inv_logit(z):
    return 1 / (1 + np.exp(-z))


def _sernels(y, p):
    p1v, p0 = p[y == 1], np.sort(p[y == 0])
    lo = np.searchsorted(p0, p1v, side="left")
    hi = np.searchsorted(p0, p1v, side="right")
    v10 = (lo + hi) / 2.0 / len(p0)
    p1s = np.sort(p[y == 1])
    p0v = p[y == 0]
    lo = np.searchsorted(p1s, p0v, side="left")
    hi = np.searchsorted(p1s, p0v, side="right")
    v01 = 1.0 - (lo + hi) / 2.0 / len(p1v)
    return v10, v01


def _delong(y, p1, p2):
    """Paired DeLong: the AUC difference and its z. Both vectors score the same rows."""
    v10_1, v01_1 = _sernels(y, p1)
    v10_2, v01_2 = _sernels(y, p2)
    s10 = np.cov(v10_1, v10_2, ddof=1)
    s01 = np.cov(v01_1, v01_2, ddof=1)
    var = ((s10[0, 0] - 2 * s10[0, 1] + s10[1, 1]) / len(v10_1)
           + (s01[0, 0] - 2 * s10[0, 1] + s10[1, 1]) / len(v01_1))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    return d, d / np.sqrt(var)


if __name__ == "__main__":
    if CFG.VERSION_NUMBER is None:
        raise SystemExit("TEMPLATE: fill CFG.VERSION_NUMBER / VERSION_NAME / EXP_ID before running.")

    t0 = time.time()
    print("=" * 80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: CPU | "
          f"Members: {CFG.MEMBERS} | Build: {CFG.BUILD}")
    print(f"Axis: restate THIS blend's axis - the member set or the combiner, not the parent's")
    print("=" * 80)

    # =========================================================================
    # [1/5] LABELS
    # =========================================================================
    y_df = pd.read_csv(CFG.TRAIN_PATH, usecols=["id", CFG.TARGET]).sort_values("id")
    y = y_df[CFG.TARGET].to_numpy(np.int8)
    ids = y_df["id"].to_numpy()
    print(f"[1/5] labels: n={len(y):,}, positive rate {y.mean():.6f}")

    # =========================================================================
    # [2/5] MEMBER OOF VECTORS
    # =========================================================================
    print("[2/5] member OOF vectors:")
    oof = {}
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.OOF_DIR / f"{v}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{v} id misaligned"
        oof[v] = df["pred"].to_numpy(np.float64)
        print(f"   {v}: ROC AUC {roc_auc_score(y, oof[v]):.6f} | "
              f"PR-AUC {average_precision_score(y, oof[v]):.6f}")
    names = list(CFG.MEMBERS)
    L = {v: _logit(oof[v]) for v in names}
    C = np.corrcoef(np.stack([L[v] for v in names]))
    print("   member mutual logit-corrs: " + " | ".join(
        f"{names[i]}~{names[j]} {C[i, j]:.6f}"
        for i in range(len(names)) for j in range(i + 1, len(names))))

    # =========================================================================
    # [3/5] BLEND + IN-RUN GATE
    # =========================================================================
    print("[3/5] blend + in-run gate:")
    oof_probs = _inv_logit(np.mean([L[v] for v in names], axis=0))
    oof_cv = roc_auc_score(y, oof_probs)
    oof_pr = average_precision_score(y, oof_probs)
    print(f"   BLENDED OOF: ROC AUC {oof_cv:.6f} | PR-AUC {oof_pr:.6f} "
          f"(best member {max(roc_auc_score(y, oof[v]) for v in names):.6f})")
    for ref in CFG.REFERENCES:
        df = pd.read_csv(CFG.OOF_DIR / f"{ref}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{ref} id misaligned"
        d, z = _delong(y, oof_probs, df["pred"].to_numpy(np.float64))
        print(f"   vs {ref}: dROC {d:+.6f}, paired DeLong z {z:+.2f}")

    # =========================================================================
    # [4/5] MEMBER SUB VECTORS + WRITE
    # =========================================================================
    print("[4/5] member sub vectors:")
    subs, ids_sub = {}, None
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.SUB_DIR / f"{v}_sub.csv").sort_values("id")
        if ids_sub is None:
            ids_sub = df["id"].to_numpy()
        assert np.array_equal(df["id"].to_numpy(), ids_sub), f"{v} sub id misaligned"
        subs[v] = df[CFG.TARGET].to_numpy(np.float64)
        print(f"   {v}: loaded {len(subs[v]):,} rows")
    test_probs = _inv_logit(np.mean([_logit(subs[v]) for v in names], axis=0))

    stem = f"v{CFG.VERSION_NUMBER}"
    for d in (CFG.OOF_DIR, CFG.SUB_DIR):          # the archive dirs; first blend of an episode creates them
        os.makedirs(d, exist_ok=True)

    pd.DataFrame({"id": ids, "pred": oof_probs}).to_csv(
        CFG.OOF_DIR / f"{stem}_oof.csv", index=False)
    print(f"   [SAVED] {CFG.OOF_DIR / (stem + '_oof.csv')} (id, pred)")

    pd.DataFrame({"id": ids_sub, CFG.TARGET: test_probs}).to_csv(
        CFG.SUB_DIR / f"{stem}_sub.csv", index=False)
    print(f"   [SAVED] {CFG.SUB_DIR / (stem + '_sub.csv')} (id, is_fraud)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: CPU")
    print(f"   Axis: equal-weight logit-mean of {names}")
    print(f"   OOF ROC AUC: {oof_cv:.6f} | OOF PR-AUC: {oof_pr:.6f}")
    print(f"   Timing: total {time.time() - t0:.1f}s")
    print(f"   Saved: {stem}_oof.csv + {stem}_sub.csv")
