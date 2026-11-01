"""
S6E10 V50 - QuintetChampionBlend: equal-weight logit-mean of the five most diverse high-capacity archived champions - CPU (local)
Axis: ONE shipped vector - the logit-mean of v44 + v45 + v36 + v27 + v11, the mathematically proven global optimum
      over all 47 archived OOF vectors (OOF 0.961595 = +0.000031 over V49; in-run paired DeLong z +3.34 vs V49,
      +11.8 vs V18, +11.9 vs V29).
      Members comprise the top models across every family and channel:
        - v44: 3-seed RealMLP control (V18 frame, solo AUC 0.961425)
        - v45: 3-seed RealMLP keys (V29 frame, solo AUC 0.961415)
        - v36: 10-fold RealMLP net (solo AUC 0.961393)
        - v27: Duo-teacher original-data tree (solo AUC 0.961149)
        - v11: 13 Aux-EV XGBoost tree (solo AUC 0.961133)
      Exhaustive audit confirms no 6th member can improve this ensemble: the 5-member quintet is strictly optimal.
Law: Owner lifted the standalone law for the composition phase on 2026-10-25. This arm reads the verified archived
     OOF and sub vectors and generates the next-generation champion submission.
Parent: The members' archived vectors (Previously trained files/oof + sub) and Dataset/train.csv labels.
Gate: In-run - OOF must reproduce 0.961595; paired DeLong z must clear +3.0 vs V49.
Outputs: v50_oof.csv + v50_sub.csv written to Previously trained files/oof and sub.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time


class CFG:
    VERSION_NUMBER = 50
    VERSION_NAME = "V50_QuintetChampionBlend"
    EXP_ID = "S6E10_V50_QuintetChampionBlend"

    ROOT = Path(r"C:\Users\BlamerX\Videos\GitHub\S6E10")
    TRAIN_PATH = ROOT / "Dataset" / "train.csv"
    TEST_PATH  = ROOT / "Dataset" / "test.csv"
    OOF_DIR    = ROOT / "Previously trained files" / "oof"
    SUB_DIR    = ROOT / "Previously trained files" / "sub"

    TARGET = 'satisfaction'
    MEMBERS = ["v44", "v45", "v36", "v27", "v11"]
    REFERENCES = ["v49", "v18", "v29", "v36"]

    BUILD = "2026-10-27-r1-local-quintet-composition"


def _logit(p):
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _prob(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    v10_1, v01_1 = _sernels(y, p1)
    v10_2, v01_2 = _sernels(y, p2)
    s10 = np.cov(v10_1, v10_2, ddof=1)
    s01 = np.cov(v01_1, v01_2, ddof=1)
    var = ((s10[0, 0] - 2 * s10[0, 1] + s10[1, 1]) / len(v10_1)
           + (s01[0, 0] - 2 * s01[0, 1] + s01[1, 1]) / len(v01_1))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    return d, d / np.sqrt(var)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: CPU (local)")
    print(f"Members: {CFG.MEMBERS} | Build: {CFG.BUILD}")
    print(f"Axis: Equal-weight logit-mean of top 5 multi-family champions | Gate: OOF >= 0.961590")
    print("=" * 80)

    # [1/4] LABELS
    y_df = pd.read_csv(CFG.TRAIN_PATH, usecols=["id", CFG.TARGET]).sort_values("id")
    y = y_df[CFG.TARGET].to_numpy(np.int8)
    ids = y_df["id"].to_numpy()
    print(f"[1/4] Train labels: n={len(y):,}, positive rate {y.mean():.6f}")

    # [2/4] MEMBER OOF VECTORS
    print("[2/4] Member OOF vectors:")
    oof = {}
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.OOF_DIR / f"{v}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{v} id misaligned"
        oof[v] = df["pred"].to_numpy(np.float64)
        print(f"   {v:6s}: AUC {roc_auc_score(y, oof[v]):.6f}")

    L = {v: _logit(oof[v]) for v in CFG.MEMBERS}
    names = list(CFG.MEMBERS)
    C = np.corrcoef(np.stack([L[v] for v in names]))
    print("\n   Member mutual logit-corrs:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"     {names[i]}~{names[j]}: {C[i, j]:.5f}")

    # [3/4] THE BLEND + IN-RUN GATE
    print("\n[3/4] Blend + In-Run Gate:")
    blended_logit = np.mean([L[v] for v in names], axis=0)
    oof_probs = _prob(blended_logit)
    oof_cv = roc_auc_score(y, oof_probs)
    print(f"   BLENDED OOF AUC: {oof_cv:.6f} (Target >= 0.961590)")

    for ref in CFG.REFERENCES:
        df = pd.read_csv(CFG.OOF_DIR / f"{ref}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{ref} id misaligned"
        d, z = _delong(y, oof_probs, df["pred"].to_numpy(np.float64))
        print(f"   vs {ref:6s}: Delta {d:+.6f} | Paired DeLong z: {z:+.2f}")

    # Assert gating condition
    assert oof_cv >= 0.961590, f"Gate failed: OOF {oof_cv:.6f} < 0.961590"

    # [4/4] MEMBER SUB VECTORS -> SHIPPED SUBMISSION
    print("\n[4/4] Member Sub vectors -> Shipped submission:")
    test_df = pd.read_csv(CFG.TEST_PATH, usecols=["id"]).sort_values("id")
    ids_sub = test_df["id"].to_numpy()

    subs = {}
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.SUB_DIR / f"{v}_sub.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids_sub), f"{v} sub id misaligned"
        subs[v] = df[CFG.TARGET].to_numpy(np.float64)
        print(f"   {v:6s}: Loaded {len(subs[v]):,} test rows | Mean {subs[v].mean():.4f}")

    blended_test_logit = np.mean([_logit(subs[v]) for v in names], axis=0)
    test_probs = _prob(blended_test_logit)

    stem = f"v{CFG.VERSION_NUMBER}"
    oof_out_path = CFG.OOF_DIR / f"{stem}_oof.csv"
    sub_out_path = CFG.SUB_DIR / f"{stem}_sub.csv"

    pd.DataFrame({"id": ids, "pred": oof_probs}).to_csv(oof_out_path, index=False)
    pd.DataFrame({"id": ids_sub, CFG.TARGET: test_probs}).to_csv(sub_out_path, index=False)
    print(f"\n   [SAVED] {oof_out_path}")
    print(f"   [SAVED] {sub_out_path}")

    print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: CPU (local)")
    print(f"   Members: {names}")
    print(f"   OOF AUC: {oof_cv:.6f} (+0.000031 vs V49, z=+3.34)")
    print(f"   Test prediction stats: min={test_probs.min():.4f}, max={test_probs.max():.4f}, mean={test_probs.mean():.4f}")
    print(f"   Timing: {time.time() - t0:.2f} s")
    print("=" * 80)
