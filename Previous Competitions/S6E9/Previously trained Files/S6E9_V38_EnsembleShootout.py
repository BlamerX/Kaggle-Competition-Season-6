"""
S6E9 V38 - Ensemble combiner shootout: five combiners, fold-sealed, one winner.

Strategy:
    No model is trained. Every out-of-fold and test prediction vector this project has
    saved is loaded, a declared eligibility rule filters the pool, and five combiners are
    compared under one protocol: each combiner's weights and leg set are fitted on nine
    folds and applied to the tenth, so no combiner is scored on rows it fitted. The sealed
    OOF AUC is the combiner's honest score and its in-sample AUC is printed beside it,
    because the gap between the two is the evidence about which combiner can be trusted.
    Highest sealed AUC wins; everything within 2e-05 of the best resolves to the simplest
    combiner by declared order. Only the winner's oof_v38.csv and sub_v38.csv are written.

References:
    V18 - the first hill climber in this repo: OOF 0.94623 fitted directly on the training
          rows, LB 0.94635. That optimism gap is why every arm here is sealed.
    V30 - house ruler KFold(10, shuffle, random_state=42), OOF 0.946171, LB 0.94639.
    V31 - additive base_margin prior, OOF 0.946224, best honest single model we own.
    V34 - LightGBM published config with bagging live, OOF 0.946156.
    V36 - KFold(20, seed 7), OOF 0.946199.
    Measured on these vectors before this version: fold-sealed greedy over the 10-fold-42
    legs +0.000121 over V30 (DeLong z = +8.83), equal-weight over the strong eight
    +0.000088, all 35 archive OOFs +0.000046 - pool width and combiner sophistication trade
    against each other, so both pools are declared rather than hand-picked.

V38 Change from V37:
    Different version class: inference only, local CPU. V37 trained a logistic model on a
    sparse design; V38 trains nothing and decides between rank mean, logit mean,
    non-negative-least-squares logit blend, L2-logistic stack and greedy forward selection
    on measured evidence instead of preference. Paths point at the locally saved oof/sub
    archive, so this is the one version that never needs a Kaggle session.

Parameters:
    N_FOLDS=10, RANDOM_SEED=42 (sealing geometry only), PLATEAU_BAND=0.0005,
    WIDE_BAND=0.0015, DUPLICATE_RHO=0.99995, GREEDY_STEPS=6, STACK_C=0.3, TIE_BAND=0.00002.
    Lineage duplicates V18/V23/V29 are excluded by rule. Runtime: local CPU, minutes.

Dataset Structure:
    Inputs: Dataset/train.csv for labels and id order, Dataset/test.csv for test ids, plus
    every oof_v*.csv (id, pred) and sub_v*.csv (id, Will_Buy_EV) in the archive.
    Outputs: oof_v38.csv (id, pred) and sub_v38.csv (id, Will_Buy_EV) from the winner only.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class CFG:
    VERSION_NAME = "v38"
    TARGET = "Will_Buy_EV"
    DATA_DIR = r"C:\Users\adars\Videos\Github\S6E9\Dataset"
    OOF_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\oof"
    SUB_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\sub"
    OUT_DIR = r"C:\Users\adars\Videos\Github\S6E9\Outputs"
    N_FOLDS = 10
    RANDOM_SEED = 42
    PLATEAU_BAND = 0.0005
    WIDE_BAND = 0.0015
    DUPLICATE_RHO = 0.99995
    GREEDY_STEPS = 6
    STACK_C = 0.3
    TIE_BAND = 0.00002
    LINEAGE_DUPLICATES = [18, 23, 29]
    ENSEMBLE_LEGS = [38, 39]
    ORDER = ["equal_rank", "mean_logit", "nnls_logit", "stack_logit", "greedy_logit"]
    POOLS = ["plateau", "wide"]


def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def pct_rank(x):
    return pd.Series(x).rank(method="average").to_numpy(dtype="float64") / len(x)


def to_logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.clip(np.log(p / (1.0 - p)), -30.0, 30.0)


def load_archive(directory, prefix):
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.startswith(prefix) or not fn.endswith(".csv"):
            continue
        num = int(fn[len(prefix):-4])
        df = pd.read_csv(os.path.join(directory, fn))
        out[num] = df.iloc[:, 1].to_numpy(dtype="float64")
    return out


def merge_duplicates(pool, rank_tr, aucs):
    kept, dropped = list(pool), []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a not in kept or b not in kept:
                continue
            rho = float(np.corrcoef(rank_tr[a], rank_tr[b])[0, 1])
            if rho > CFG.DUPLICATE_RHO:
                drop = a if aucs[a] < aucs[b] else b
                kept.remove(drop)
                dropped.append((drop, rho))
    return kept, dropped


def fit_combiner(name, R_tr, L_tr, y_tr):
    """Return (blend_fn, chosen_idx) fitted on the supplied rows only."""
    if name == "equal_rank":
        return (lambda R_va, L_va: R_va.mean(axis=0)), None
    if name == "mean_logit":
        return (lambda R_va, L_va: L_va.mean(axis=0)), None
    if name == "nnls_logit":
        w, _ = nnls(L_tr.T, y_tr.astype("float64"))
        if w.sum() <= 0:
            w = np.ones(L_tr.shape[0]) / L_tr.shape[0]
        w = w / w.sum()
        return (lambda R_va, L_va: L_va.T.dot(w)), None
    if name == "stack_logit":
        model = LogisticRegression(C=CFG.STACK_C, solver="lbfgs", max_iter=200)
        model.fit(L_tr.T, y_tr)
        return (lambda R_va, L_va: model.predict_proba(L_va.T)[:, 1]), None
    if name == "greedy_logit":
        solo = [auc_score(y_tr, L_tr[j]) for j in range(L_tr.shape[0])]
        chosen = [int(np.argmax(solo))]
        for _ in range(CFG.GREEDY_STEPS - 1):
            cands = [j for j in range(L_tr.shape[0]) if j not in chosen]
            if not cands:
                break
            scores = [auc_score(y_tr, np.mean([L_tr[c] for c in chosen + [j]], axis=0))
                      for j in cands]
            chosen.append(cands[int(np.argmax(scores))])
        return (lambda R_va, L_va: np.mean([L_va[c] for c in chosen], axis=0)), chosen
    raise KeyError(name)


def sealed_score(name, pool, rank_tr, logit_tr, y, fold):
    out = np.zeros(len(y))
    picks = []
    for f in range(CFG.N_FOLDS):
        tr, va = fold != f, fold == f
        blend, chosen = fit_combiner(
            name, np.vstack([rank_tr[v][tr] for v in pool]),
            np.vstack([logit_tr[v][tr] for v in pool]), y[tr])
        if chosen is not None:
            picks.append([pool[c] for c in chosen])
        out[va] = blend(np.vstack([rank_tr[v][va] for v in pool]),
                        np.vstack([logit_tr[v][va] for v in pool]))
    return out, picks


def insample_blend(name, pool, rank, logit, y):
    blend, _ = fit_combiner(name, np.vstack([rank[v] for v in pool]),
                            np.vstack([logit[v] for v in pool]), y)
    return blend(np.vstack([rank[v] for v in pool]), np.vstack([logit[v] for v in pool]))


def test_blend(name, pool, rank_tr, logit_tr, rank_te, logit_te, y):
    """Fit the combiner on train rows only, then apply it to the test vectors."""
    blend, _ = fit_combiner(name, np.vstack([rank_tr[v] for v in pool]),
                            np.vstack([logit_tr[v] for v in pool]), y)
    return blend(np.vstack([rank_te[v] for v in pool]), np.vstack([logit_te[v] for v in pool]))


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
    print(f"   train rows: {n:,} | positive rate: {y.mean():.4f}")

    oof = load_archive(CFG.OOF_DIR, "oof_v")
    sub = load_archive(CFG.SUB_DIR, "sub_v")
    paired = sorted(set(oof) & set(sub))
    print(f"   OOF vectors: {len(oof)} | test vectors: {len(sub)} | paired: {len(paired)}")

    print("\n[2/5] Eligibility rule...")
    aucs_all = {v: auc_score(y, oof[v]) for v in oof}
    keep = [v for v in paired
            if v not in CFG.LINEAGE_DUPLICATES and v not in CFG.ENSEMBLE_LEGS
            and v != int(CFG.VERSION_NAME[1:])]
    best = max(aucs_all[v] for v in keep)
    best_v = max(keep, key=aucs_all.get)
    fold = np.zeros(n, dtype="int8")
    for i, (_, va) in enumerate(KFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                      random_state=CFG.RANDOM_SEED).split(np.arange(n))):
        fold[va] = i

    rank_tr, logit_tr, rank_te, logit_te = {}, {}, {}, {}
    for v in keep:
        rank_tr[v] = pct_rank(oof[v])
        logit_tr[v] = to_logit(oof[v])
        rank_te[v] = pct_rank(sub[v])
        logit_te[v] = to_logit(sub[v])

    print(f"   best single OOF in the archive: V{best_v} {best:.6f}")
    print(f"   excluded by lineage rule: {sorted(set(paired) - set(keep))}")

    pools = {}
    for band, pname in [(CFG.PLATEAU_BAND, "plateau"), (CFG.WIDE_BAND, "wide")]:
        pools[pname] = sorted([v for v in keep if aucs_all[v] >= best - band])
        print(f"   {pname} pool (>= best - {band}): {pools[pname]}")
    print(f"   excluded by band rule: {sorted(set(keep) - set(pools['wide']))}")

    merged_pools = {}
    for pname, pool in pools.items():
        merged, dropped = merge_duplicates(pool, rank_tr, aucs_all)
        merged_pools[pname] = merged
        for d, rho in dropped:
            print(f"   duplicate merge [{pname}]: V{d} dropped (rho = {rho:.6f})")
        print(f"   {pname} pool after merge: {len(merged)} legs {merged}")

    print("\n[3/5] Fold-sealed combiner shootout...")
    results = {}
    for pname in CFG.POOLS:
        pool = merged_pools[pname]
        for name in CFG.ORDER:
            t1 = time.time()
            sealed, picks = sealed_score(name, pool, rank_tr, logit_tr, y, fold)
            ins = insample_blend(name, pool, rank_tr, logit_tr, y)
            s_auc, i_auc = auc_score(y, sealed), auc_score(y, ins)
            results[(pname, name)] = (s_auc, i_auc, sealed, picks)
            print(f"   {pname:8s} x {name:12s} n={len(pool):2d} | sealed {s_auc:.6f} | "
                  f"in-sample {i_auc:.6f} | optimism {i_auc - s_auc:+.6f} | "
                  f"{time.time() - t1:.0f}s")

    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
    best_sc = ranked[0][1][0]
    near = [(pn, nm, s) for (pn, nm), (s, _, _, _) in results.items()
            if best_sc - s <= CFG.TIE_BAND]
    win_pn, win_nm, win_sc = sorted(near, key=lambda t: (CFG.ORDER.index(t[1]),
                                                         t[0] != "plateau"))[0]
    print(f"\n   best sealed {best_sc:.6f} | {len(near)} cells inside the "
          f"{CFG.TIE_BAND:.5f} tie band | simplest declared order wins")
    print(f"   WINNER: {win_pn} pool x {win_nm} | sealed OOF {win_sc:.6f}")

    print("\n[4/5] Saving outputs...")
    pool = merged_pools[win_pn]
    oof_v38 = results[(win_pn, win_nm)][2]
    sub_v38 = test_blend(win_nm, pool, rank_tr, logit_tr, rank_te, logit_te, y)
    oof_path = os.path.join(CFG.OUT_DIR, f"oof_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": train["id"].to_numpy(), "pred": oof_v38}).to_csv(oof_path, index=False)
    print(f"   [SAVED] {oof_path} (id, pred) - sealed OOF of the winner")
    test_ids = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"), usecols=["id"])["id"].to_numpy()
    sub_path = os.path.join(CFG.OUT_DIR, f"sub_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": test_ids, CFG.TARGET: np.clip(sub_v38, 0.0, 1.0)}).to_csv(sub_path, index=False)
    print(f"   [SAVED] {sub_path} (id, {CFG.TARGET}) - {len(pool)} legs, combiner {win_nm}")

    print(f"\n{'=' * 80}")
    print(f"[5/5] V38 RESULTS - winner {win_nm} on the {win_pn} pool ({len(pool)} legs)")
    print(f"{'=' * 80}")
    try:
        for v in pool:
            print(f"   leg V{v:<3d} solo OOF {aucs_all[v]:.6f} | rho vs V30 "
                  f"{float(np.corrcoef(rank_tr[v], rank_tr[30])[0, 1]):.5f}")
        rho_legs = [float(np.corrcoef(rank_tr[a], rank_tr[b])[0, 1])
                    for i, a in enumerate(pool) for b in pool[i + 1:]]
        print(f"\n   OOF CV (AUC): {win_sc:.6f}")
        print(f"   vs V30 0.946171: {win_sc - 0.946171:+.6f}")
        print(f"   vs V31 0.946224: {win_sc - 0.946224:+.6f}")
        print(f"   in-sample (unsealed) score of this combiner: "
              f"{results[(win_pn, win_nm)][1]:.6f}")
        print(f"   mean pairwise rank rho between legs: {np.mean(rho_legs):.5f}")
        picks = results[(win_pn, win_nm)][3]
        if picks:
            print(f"   leg-set stability across the 10 folds: "
                  f"{len({tuple(p) for p in picks})} distinct sets")
            for f, p in enumerate(picks):
                print(f"      fold {f + 1:>2d} picks: {['V%d' % v for v in p]}")
        print(f"   test predictions: min {sub_v38.min():.6f} | max {sub_v38.max():.6f}")
        print("\n   full shootout ranking:")
        for (pn, nm), (s, i, _, _) in ranked:
            print(f"      {pn:8s} x {nm:12s} sealed {s:.6f} | in-sample {i:.6f} "
                  f"| optimism {i - s:+.6f}")
    except Exception as exc:
        print(f"   [DIAGNOSTICS FAILED AFTER SAVE] {type(exc).__name__}: {exc}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
