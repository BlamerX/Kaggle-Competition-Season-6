"""
S6E9 V39 - Equal-weight ensemble over the elite leg band: the combiner with no weights.

Strategy:
    V38's fold-sealed L2-logistic stack reached 0.946333 honest CV and returned
    0.94639 on the public board, identical to V30's score for +0.000162 of CV. The
    mechanism we cannot rule out with CV is a stack/test geometry mismatch: a leg's OOF
    row comes from one fold model while its test row comes from the average of ten, so
    fitted weights are partly paid to cancel noise that is absent at test time. Equal
    weighting has no weights to fail to transfer. This version builds that ensemble under
    a declared rule, ships the winner of three weight-free combiners, and is meant to be
    submitted as a daily so the public gap measures the question CV cannot.

References:
    V38 - combiner shoot-out, sealed 0.946333 / LB 0.94639 (gap +0.00006); equal-weight
          over the 25-leg plateau pool gave 0.946223, over the strong eight 0.946259.
    V30 - KFold(10, shuffle, random_state=42), OOF 0.946171, LB 0.94639, gap +0.00022.
    V31 - additive base_margin prior, OOF 0.946224, best honest single model we own.
    V34 - LightGBM parity, OOF 0.946156. V36 - 20-fold geometry, OOF 0.946199.
    S6E6 26th place writeup: repeated-CV noise floor 0.000036-0.000048, so only changes
    of >= ~0.00005 are outside measurement error.

V39 Change from V38:
    Same inputs, opposite combiner class: nothing is fitted. The leg pool comes from a
    rule declared before measuring - every version whose solo OOF is within 0.00010 of the
    best model in the archive, i.e. statistically indistinguishable from it - excluding
    lineage duplicates and V38 itself, which is a stack and would otherwise be averaging a
    model made from its own inputs. Band sensitivity is printed, but the declared band is
    what ships.

Parameters:
    ELITE_BAND=0.00010 (declared), SENSITIVITY_BANDS=[0.00005, 0.00010, 0.00020, 0.00050],
    LINEAGE_DUPLICATES=[18, 23, 29], SELF_EXCLUDED=38. No CV geometry is used because
    nothing is fitted: the OOF average is its own honest estimate.

Dataset Structure:
    Inputs: Dataset/train.csv and test.csv for ids and labels, plus every oof_v*.csv and
    sub_v*.csv in the archive. Outputs: oof_v39.csv (id, pred) and sub_v39.csv
    (id, Will_Buy_EV) from the winning weight-free combiner only.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class CFG:
    VERSION_NAME = "v39"
    TARGET = "Will_Buy_EV"
    DATA_DIR = r"C:\Users\adars\Videos\Github\S6E9\Dataset"
    OOF_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\oof"
    SUB_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\sub"
    OUT_DIR = r"C:\Users\adars\Videos\Github\S6E9\Outputs"
    ELITE_BAND = 0.00010
    SENSITIVITY_BANDS = [0.00005, 0.00010, 0.00020, 0.00050]
    LINEAGE_DUPLICATES = [18, 23, 29]
    SELF_EXCLUDED = 38
    ORDER = ["rank_mean", "logit_mean", "prob_mean"]


def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def pct_rank(x):
    return pd.Series(x).rank(method="average").to_numpy(dtype="float64") / len(x)


def to_logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.clip(np.log(p / (1.0 - p)), -30.0, 30.0)


def from_logit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def load_archive(directory, prefix):
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.startswith(prefix) or not fn.endswith(".csv"):
            continue
        out[int(fn[len(prefix):-4])] = pd.read_csv(os.path.join(directory, fn))
    return out


def combine(name, P, R, L, pool):
    if name == "rank_mean":
        return np.mean([R[v] for v in pool], axis=0)
    if name == "logit_mean":
        return from_logit(np.mean([L[v] for v in pool], axis=0))
    if name == "prob_mean":
        return np.mean([P[v] for v in pool], axis=0)
    raise KeyError(name)


if __name__ == "__main__":
    t0_all = time.time()
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    print("[1/5] Loading labels and the saved prediction archive...")
    train = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
    y = pd.to_numeric(train[CFG.TARGET], errors="coerce")
    if y.isna().all():
        y = train[CFG.TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1})
    y = y.to_numpy(dtype="int8")
    n = len(y)
    test_ids = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"), usecols=["id"])["id"].to_numpy()
    print(f"   train rows: {n:,} | positive rate: {y.mean():.4f} | test rows: {len(test_ids):,}")

    oof_df = load_archive(CFG.OOF_DIR, "oof_v")
    sub_df = load_archive(CFG.SUB_DIR, "sub_v")
    paired = sorted(set(oof_df) & set(sub_df))
    for v in paired:
        assert len(oof_df[v]) == n and (oof_df[v]["id"].to_numpy() == train["id"].to_numpy()).all(), v
        assert len(sub_df[v]) == len(test_ids) and (sub_df[v]["id"].to_numpy() == test_ids).all(), v
    P = {v: oof_df[v].iloc[:, 1].to_numpy(dtype="float64") for v in paired}
    S = {v: sub_df[v].iloc[:, 1].to_numpy(dtype="float64") for v in paired}
    R = {v: pct_rank(P[v]) for v in paired}
    L = {v: to_logit(P[v]) for v in paired}
    RT = {v: pct_rank(S[v]) for v in paired}
    LT = {v: to_logit(S[v]) for v in paired}
    aucs = {v: auc_score(y, P[v]) for v in paired}
    print(f"   paired OOF/test vectors: {len(paired)} (all id-aligned)")

    print("\n[2/5] Leg pool by declared rule...")
    eligible = [v for v in paired if v not in CFG.LINEAGE_DUPLICATES and v != CFG.SELF_EXCLUDED]
    best_v = max(eligible, key=aucs.get)
    best = aucs[best_v]
    pool = sorted([v for v in eligible if aucs[v] >= best - CFG.ELITE_BAND])
    assert len(pool) >= 2, f"elite band produced {len(pool)} legs, cannot ensemble"
    print(f"   rule: solo OOF >= best single model ({best_v} {best:.6f}) - {CFG.ELITE_BAND}")
    print(f"   elite pool: {len(pool)} legs " + " ".join(f"V{v} {aucs[v]:.6f}" for v in pool))
    for band in CFG.SENSITIVITY_BANDS:
        p = sorted([v for v in eligible if aucs[v] >= best - band])
        print(f"   sensitivity band {band}: {len(p):2d} legs -> rank_mean "
              f"{auc_score(y, combine('rank_mean', P, R, L, p)):.6f}")

    print("\n[3/5] Weight-free combiners...")
    scores = {}
    for name in CFG.ORDER:
        scores[name] = auc_score(y, combine(name, P, R, L, pool))
        print(f"   {name:10s} OOF {scores[name]:.6f}")
    best_name = max(CFG.ORDER, key=lambda nm: (scores[nm], -CFG.ORDER.index(nm)))
    top = max(scores.values())
    winner = [nm for nm in CFG.ORDER if top - scores[nm] <= 0.00002][0]
    print(f"   best {best_name} {top:.6f} | within 0.00002: "
          f"{[nm for nm in CFG.ORDER if top - scores[nm] <= 0.00002]} | ships {winner}")

    print("\n[4/5] Saving outputs...")
    oof_v39 = combine(winner, P, R, L, pool)
    sub_v39 = combine(winner, S, RT, LT, pool)
    oof_path = os.path.join(CFG.OUT_DIR, f"oof_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": train["id"].to_numpy(), "pred": oof_v39}).to_csv(oof_path, index=False)
    print(f"   [SAVED] {oof_path} (id, pred)")
    sub_path = os.path.join(CFG.OUT_DIR, f"sub_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": test_ids, CFG.TARGET: np.clip(sub_v39, 0.0, 1.0)}).to_csv(sub_path, index=False)
    print(f"   [SAVED] {sub_path} (id, {CFG.TARGET}) - {len(pool)} legs, {winner}")

    print(f"\n{'=' * 80}")
    print(f"[5/5] V39 RESULTS - equal-weight {winner} over {len(pool)} elite legs")
    print(f"{'=' * 80}")
    try:
        cv = scores[winner]
        print(f"   OOF CV (AUC): {cv:.6f}")
        print(f"   vs V30 0.946171: {cv - 0.946171:+.6f}")
        print(f"   vs V31 0.946224: {cv - 0.946224:+.6f}")
        print(f"   vs V38 0.946333: {cv - 0.946333:+.6f} (weighted stack, LB 0.94639)")
        rho = [float(np.corrcoef(R[a], R[b])[0, 1]) for i, a in enumerate(pool) for b in pool[i + 1:]]
        print(f"   mean pairwise rank rho between legs: {np.mean(rho):.5f} | min {np.min(rho):.5f}")
        shipped = pct_rank(sub_v39)
        for ref, label in [(30, "V30"), (38, "V38 stack"), (23, "V23 best LB")]:
            if ref in RT:
                rt = pct_rank(S[ref])
                print(f"   test rank shift vs {label}: {np.abs(shipped - rt).mean()*100:.3f}% "
                      f"of positions | spearman {np.corrcoef(shipped, rt)[0, 1]:.5f}")
        print(f"   test predictions: min {sub_v39.min():.6f} | max {sub_v39.max():.6f}")
        print("   nothing is fitted in this version: sealed == in-sample == the numbers above")
    except Exception as exc:
        print(f"   [DIAGNOSTICS FAILED AFTER SAVE] {type(exc).__name__}: {exc}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
