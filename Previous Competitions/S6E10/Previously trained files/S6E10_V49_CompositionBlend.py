"""
S6E10 V49 - CompositionBlend: equal-weight logit-mean of the four most independent archived vectors - CPU (local)
Axis: ONE shipped vector - the logit-mean of v11 + v27 + v34 + v36, the hill-climb optimum over all 42
      usable archived OOF vectors (OOF 0.961564 = +0.000171 over v36; paired DeLong z +6.01 vs v36, +8.89
      vs v29; SLSQP weight polish returns uniform, rank-mean ties logit-mean at -0.000003). The members are
      genuinely different columns: tree (v11), duo-teacher tree (v27), keys-frame reseed (v34), 10-fold net
      (v36) - NOT a same-model seed average, so the V44/V45 offset penalty does not transfer by construction.
Law: the owner lifted the 2026-10-11 standalone law for the composition phase (instruction of 2026-10-25):
      this arm READS the archived oof/sub files by explicit instruction and ships a blend. Per the V44/V45
      lesson the CV->LB offset of any averaged vector is UNMEASURED - adoption requires the LB itself.
Parent: the members' archived vectors (Previously trained files/oof + sub) and the local labels.
Gate: in-run - OOF must reproduce 0.961564 and the paired DeLong z's are printed against v18/v36/v29.
Expected / Cost: LB unknown (offset unmeasured for diverse blends); ~1 min CPU, no GPU.
Outputs: v49_oof.csv + v49_sub.csv written beside the archived vectors - exactly two writes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time


class CFG:
    VERSION_NUMBER = 49
    VERSION_NAME = "V49_CompositionBlend"
    EXP_ID = "S6E10_V49_CompositionBlend"

    ROOT = Path(r"C:\Users\BlamerX\Videos\GitHub\S6E10")
    TRAIN_PATH = ROOT / "Dataset" / "train.csv"
    OOF_DIR = ROOT / "Previously trained files" / "oof"
    SUB_DIR = ROOT / "Previously trained files" / "sub"

    TARGET = 'satisfaction'
    MEMBERS = ["v11", "v27", "v34", "v36"]   # the hill-climb optimum (l4_blend_lab.py, 2026-10-25)
    REFERENCES = ["v18", "v36", "v29"]       # control / best-OOF / best-LB

    BUILD = "2026-10-25-r1-local-composition"


def _logit(p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


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
    print(f"Version: V{CFG.VERSION_NUMBER} | {CFG.VERSION_NAME} | Device: CPU (local) | "
          f"Members: {CFG.MEMBERS} | Build: {CFG.BUILD}")
    print(f"Axis: equal-weight logit-mean of the members' archived OOF and sub vectors | "
          f"gate: OOF must reproduce 0.961564")
    print("=" * 80)

    # [1/4] LABELS
    y_df = pd.read_csv(CFG.TRAIN_PATH, usecols=["id", CFG.TARGET]).sort_values("id")
    y = y_df[CFG.TARGET].to_numpy(np.int8)
    ids = y_df["id"].to_numpy()
    print(f"[1/4] labels: n={len(y)}, positive rate {y.mean():.6f}")

    # [2/4] MEMBER OOF VECTORS
    print("[2/4] member OOF vectors:")
    oof = {}
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.OOF_DIR / f"{v}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{v} id misaligned"
        oof[v] = df["pred"].to_numpy(np.float64)
        print(f"   {v}: AUC {roc_auc_score(y, oof[v]):.6f}")
    L = {v: _logit(oof[v]) for v in CFG.MEMBERS}
    names = list(CFG.MEMBERS)
    C = np.corrcoef(np.stack([L[v] for v in names]))
    print("   member mutual logit-corrs: " + " | ".join(
        f"{names[i]}~{names[j]} {C[i, j]:.6f}"
        for i in range(len(names)) for j in range(i + 1, len(names))))

    # [3/4] THE BLEND + IN-RUN GATE
    print("[3/4] blend + in-run gate:")
    oof_probs = 1 / (1 + np.exp(-np.mean([L[v] for v in names], axis=0)))
    oof_cv = roc_auc_score(y, oof_probs)
    print(f"   BLENDED OOF: {oof_cv:.6f} (must reproduce 0.961564)")
    for ref in CFG.REFERENCES:
        df = pd.read_csv(CFG.OOF_DIR / f"{ref}_oof.csv").sort_values("id")
        assert np.array_equal(df["id"].to_numpy(), ids), f"{ref} id misaligned"
        d, z = _delong(y, oof_probs, df["pred"].to_numpy(np.float64))
        print(f"   vs {ref}: d {d:+.6f}, paired DeLong z {z:+.2f}")

    # [4/4] MEMBER SUB VECTORS -> SHIPPED SUBMISSION
    print("[4/4] member sub vectors:")
    subs = {}
    ids_sub = None
    for v in CFG.MEMBERS:
        df = pd.read_csv(CFG.SUB_DIR / f"{v}_sub.csv").sort_values("id")
        if ids_sub is None:
            ids_sub = df["id"].to_numpy()
        assert np.array_equal(df["id"].to_numpy(), ids_sub), f"{v} sub id misaligned"
        subs[v] = df[CFG.TARGET].to_numpy(np.float64)
        print(f"   {v}: loaded {len(subs[v]):,} rows")
    test_probs = 1 / (1 + np.exp(-np.mean([_logit(subs[v]) for v in names], axis=0)))

    out_dir = CFG.ROOT
    stem = f"v{CFG.VERSION_NUMBER}"
    pd.DataFrame({"id": ids, "pred": oof_probs}).to_csv(
        CFG.OOF_DIR / f"{stem}_oof.csv", index=False)
    pd.DataFrame({"id": ids_sub, CFG.TARGET: test_probs}).to_csv(
        CFG.SUB_DIR / f"{stem}_sub.csv", index=False)
    print(f"   [SAVED] {CFG.OOF_DIR / (stem + '_oof.csv')}")
    print(f"   [SAVED] {CFG.SUB_DIR / (stem + '_sub.csv')}")

    print(f"\n[5/4] RESULTS | {CFG.VERSION_NAME} | Device: CPU (local)")
    print(f"   Axis: equal-weight logit-mean of {names} | OOF {oof_cv:.6f}")
    print(f"   LB offset for a diverse blend is UNMEASURED (V44/V45 lesson) - adoption requires the LB itself")
    print(f"   Timing: total {time.time() - t0:.1f}s")
