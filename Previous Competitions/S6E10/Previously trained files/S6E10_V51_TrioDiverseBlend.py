"""
S6E10 V51 - TrioDiverseBlend: three members instead of five, to find out whether the CV->LB offset is set by leg count - CPU (local)
Axis: ONE change vs V50 - the shipped vector is the equal-weight logit-mean of **v27 + v36 + v44** (a duo-teacher tree,
      the 10-fold net, a 3-seed net: the highest-CV 3-member set over all 14 single-arm vectors we own, CV 0.961585,
      0.00001 BELOW V50's quintet). Frame, models and weights are untouched; only the number of averaged legs changes.
Why: V49 (4 legs) and V50 (5 legs) both sit at gap -0.00048, while single-model nets sit at -0.00018 (V29) / -0.00023
      (V18 frame). Two hypotheses fit those four points and no experiment yet separates them:
        (H1) the offset is set by the FAMILY MIX  -> V51 lands ~0.96110, and V50 stays the submission;
        (H2) it grows with leg count              -> V51 lands ~0.96126-0.96128 and fewer legs beat more CV.
      The two predict board scores 0.00018 apart (3 rulers), so ONE submission decides it, and it decides whether the
      remaining days are spent adding diverse members or adding legs.
Parent: V50_QuintetChampionBlend (CV 0.961595 / LB 0.96112, the standing best). No model is fitted and no saved OOF is
      trained on - the owner's 2026-10-25 composition lift covers averaging archived vectors into one shipped blend.
Gate (pre-declared): CV within 0.00003 of V50's (it is, at -0.000010); paired DeLong vs v50_oof reported by the same
      full-row bootstrap used to validate V50 (z is expected INSIGNIFICANT here - this arm is not a CV claim);
      the measurement that matters is the LB, and it is recorded whether it wins or loses.
Outputs: v51_oof.csv + v51_sub.csv - exactly two writes, both labelled V51.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MEMBERS = ["v27", "v36", "v44"]
VERSION = "V51"
NAME = "TrioDiverseBlend"
EPS = 1e-6
BOOT_REPS = 120
V50_CV, V50_LB = 0.961595, 0.96112
OFFSETS = {"H1 family-mix (frozen)": 0.00048, "H2 leg-count (shrinking)": 0.00030}

BASE = Path("Previously trained files")
OOF_DIR, SUB_DIR = BASE / "oof", BASE / "sub"
TRAIN_PATH = Path("Dataset/train.csv")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _prob(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_oof(v):
    df = pd.read_csv(OOF_DIR / f"{v}_oof.csv")
    return df.sort_values("id").reset_index(drop=True)


def load_sub(v):
    df = pd.read_csv(SUB_DIR / f"{v}_sub.csv")
    return df.sort_values("id").reset_index(drop=True)


def boot_paired_z(p_blend, p_ref, y, reps=BOOT_REPS, seed=7):
    """Nonparametric paired bootstrap of the AUC difference - the statistic validated against this campaign's in-run
    DeLong (recorded +8.76 vs bootstrap +8.09), so the two are comparable rather than contradictory."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if not (10 < yb.sum() < len(yb) - 10):
            diffs[r] = np.nan
            continue
        diffs[r] = roc_auc_score(yb, p_blend[idx]) - roc_auc_score(yb, p_ref[idx])
    diffs = diffs[~np.isnan(diffs)]
    d = roc_auc_score(y, p_blend) - roc_auc_score(y, p_ref)
    return d, (d / diffs.std() if diffs.std() > 0 else 0.0), len(diffs)


def main():
    t0 = time.time()
    print("=" * 80)
    print(f"Version: {VERSION} | {NAME} | Device: CPU (local, no model fitted)")
    print(f"Axis: {len(MEMBERS)} members instead of V50's 5 -> {' + '.join(MEMBERS)} | parent V50 "
          f"(CV {V50_CV:.6f} / LB {V50_LB:.5f})")
    print(f"Question: is the CV->LB offset set by family mix or by leg count? Predictions differ by "
          f"{abs(OFFSETS['H1 family-mix (frozen)'] - OFFSETS['H2 leg-count (shrinking)']):.5f} = "
          f"{abs(0.00048 - 0.00030) / 0.000057:.1f} rulers")
    print("=" * 80)

    train = pd.read_csv(TRAIN_PATH, usecols=["id", "satisfaction"])
    y = (train["satisfaction"].astype(str).str.lower().isin(["true", "1", "1.0"])).to_numpy().astype(int)
    ids = train["id"].to_numpy()
    print(f"\n[1/4] Train labels: n={len(y):,} | positive rate {y.mean():.6f}")

    oof = {}
    for v in MEMBERS:
        d = load_oof(v)
        assert np.array_equal(d["id"].to_numpy(), ids), f"{v}_oof ids do not align row-for-row with train.csv"
        oof[v] = d["pred"].to_numpy()
        print(f"   {v:>4} loaded | solo AUC {roc_auc_score(y, oof[v]):.6f}")

    blend_z = np.mean([_logit(oof[v]) for v in MEMBERS], axis=0)
    blend = _prob(blend_z)
    cv = roc_auc_score(y, blend)
    print(f"\n[2/4] BLENDED OOF AUC {cv:.6f} | vs V50 {cv - V50_CV:+.6f}")

    folds = [roc_auc_score(y[te], blend[te])
             for _, te in StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)]
    print("   folds (house partition): " + " ".join(f"{f:.5f}" for f in folds)
          + f" | mean {np.mean(folds):.5f} | pooled - mean {cv - np.mean(folds):+.5f}")
    print("   the partition is verified, not assumed: fold-scoring v18/v29 under it reproduces their recorded arrays")

    ref = load_oof("v50")
    assert np.array_equal(ref["id"].to_numpy(), ids), "v50_oof ids differ"
    d, z, used = boot_paired_z(blend, ref["pred"].to_numpy(), y)
    print(f"\n[3/4] paired bootstrap vs v50_oof: dAUC {d:+.6f} | z {z:+.2f} ({used} reps)")
    print("   an INSIGNIFICANT z is the expected outcome - this arm is an offset experiment, not a CV claim.")
    for label, off in OFFSETS.items():
        print(f"   predicted LB under {label}, offset -{off:.5f}: {cv - off:.5f} (V50 actual {V50_LB:.5f})")

    subs = {}
    for v in MEMBERS:
        s = load_sub(v)
        assert len(s) == 299844, f"{v}_sub has {len(s)} rows"
        subs[v] = s
    out_z = np.mean([_logit(subs[v]["satisfaction"].to_numpy()) for v in MEMBERS], axis=0)
    out_p = _prob(out_z)

    oof_out = pd.DataFrame({"id": ids, "pred": blend})
    sub_out = pd.DataFrame({"id": subs[MEMBERS[0]]["id"].to_numpy(), "satisfaction": out_p})
    assert oof_out["pred"].between(0, 1).all() and sub_out["satisfaction"].between(0, 1).all()
    oof_out.to_csv(OOF_DIR / "v51_oof.csv", index=False)
    sub_out.to_csv(SUB_DIR / "v51_sub.csv", index=False)

    print(f"\n[4/4] SAVED v51_oof.csv + v51_sub.csv (exactly two writes) | total {time.time() - t0:.1f}s")
    print(f"   ship {VERSION}: submit it once and record the gap. H1 -> ~0.96110 (add diverse members); "
          f"H2 -> ~0.96126 (add fewer legs). Both answers change the plan.")


if __name__ == "__main__":
    main()
