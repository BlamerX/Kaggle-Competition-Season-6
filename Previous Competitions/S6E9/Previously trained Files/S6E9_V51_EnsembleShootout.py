"""
S6E9 V51 - Combiner shootout over the whole saved-prediction archive: five combiners, three
           declared pools, all fold-sealed, one winner, one oof and one sub.

Strategy:
    Nothing is trained. Every out-of-fold and test vector this project has saved is loaded, three
    eligibility rules filter it, and five combiners are compared under one protocol: each
    combiner's weights and leg set are fitted on nine folds and applied to the tenth, so no
    combiner is scored on rows it fitted. The sealed OOF AUC is the honest score; the in-sample
    number is printed beside it because the gap between them is the evidence about which
    combiner transfers. Highest sealed wins; anything inside the tie band resolves to the
    simplest combiner by declared order, since a fitted combiner's advantage that cannot be
    resolved is not worth its degrees of freedom.

Combiners (declared order, simplest first - the tie-break order IS the preference order):
    equal_rank   mean percentile rank, nothing fitted
    mean_logit   mean logit, nothing fitted
    nnls_logit   non-negative-least-squares weights on logits
    stack_logit  L2-logistic stack on logits (C 0.3)
    greedy_logit forward-stepwise LEG SELECTION, 6 steps - "let the code decide which models go
                 in". It is kept, but only on the two diversity pools and only fold-sealed, so
                 the legs it picks on nine folds are scored on the tenth and its optimism is
                 measured rather than assumed. V18 ran the same climber fitted in-sample
                 (OOF 0.94623 -> LB 0.94635) and the V38 reprice sealed it at 0.946356 with zero
                 optimism, between equal-rank's 0.946259 and the stack's 0.946373.

Pools (declared bands below the archive's best single model; ensembles and lineage duplicates
excluded from the reference and the pool):
    elite / plateau / wide          >= best - 0.00010 / 0.00050 / 0.00150, duplicates merged at
                                     rho > 0.99995 (V38's discipline: removes only bit-level twins)
    elite_div / plateau_div         same parity bands, but merged at rho > 0.999 - the DIVERSITY
                                     arm handled by rule instead of by hand, so what survives is
                                     the decorrelated subset of legs that already reach parity
    Two measured cautions, stated before the run because they are our own numbers, not theory:
    "a diverse leg helps" has failed here twice - the most decorrelated constructible leg
    (logistic + target encoding, solo 0.93815, rho 0.974 vs V44) contributed -0.000001 to the
    stack and -0.000176 to the averager, and andrewleal70's independent falsification found the
    MLP (rho 0.975) earned weight ~0. The reason is that every learner on this matrix reads the
    same encoded conditionals, so real diversity requires parity, which is why the *_div pools
    stay inside the parity band instead of reaching down for weird models.

References:
    V18  hill climber fitted directly on training rows: OOF 0.94623 optimistic, LB 0.94635 - the
         reason every arm here is sealed.
    V38  L2-logit stack sealed 0.946333, LB 0.94639: best honest OOF we produced, and the board
         paid 0.00000 for +0.000162 of CV. Shipped weights had 12 of 25 negative.
    V39  equal-weight elite band sealed 0.946285, LB 0.94640 - beat V38's stack on the board
         while carrying 0.000048 LESS CV.
    V44  best honest single model we own: OOF 0.946252, LB 0.94638.
    External corroboration measured this week: georgymamarin's 11-submission ledger found every
    combiner worth +0.000063 +/- 0.000014 in CV and -1e-5/+1e-5 on the board, while feature gains
    transferred. Combiner gains on this board are private-score arguments, not public ones.

Parameters:
    N_FOLDS=10, RANDOM_SEED=42 (sealing geometry only), STACK_C=0.3, TIE_BAND=0.00002,
    GREEDY_STEPS=5, GREEDY_CANDIDATES=12. Pool bands 0.00010 / 0.00050 / 0.00150 with rank-
    correlation merge at 0.99995 (strict) or 0.999 (the *_div pools). Lineage duplicates
    V18/V23/V29 and prior ensembles V38/V39 are excluded by rule. Runtime: local CPU, minutes.

Outputs:
    oof_v51.csv (id, pred) and sub_v51.csv (id, Will_Buy_EV) from the winning cell only, written
    to the repo Outputs folder so the committed V38/V39 artefacts stay exactly as submitted.
    Inputs: Dataset/train.csv, Dataset/test.csv, and every oof_v*.csv / sub_v*.csv in the archive.
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
    VERSION_NAME = "v51"
    TARGET = "Will_Buy_EV"
    DATA_DIR = r"C:\Users\adars\Videos\Github\S6E9\Dataset"
    OOF_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\oof"
    SUB_DIR = r"C:\Users\adars\Videos\Github\S6E9\Previously trained Files\sub"
    OUT_DIR = r"C:\Users\adars\Videos\Github\S6E9\Outputs"

    N_FOLDS = 10
    RANDOM_SEED = 42
    STACK_C = 0.3
    TIE_BAND = 0.00002
    # Greedy is the slow arm, so it is bounded three ways: it only runs on the two *_div pools
    # (which are already collapsed to their decorrelated parity subset), it takes fewer steps,
    # and at each step it only considers the top-K remaining legs by solo AUC. The cap is a
    # measured claim, not a convenience: a leg soloing near 0.9450 was negative to the stack at
    # every rho it can actually attain, so the candidates greedy would ever pick are few.
    GREEDY_STEPS = 5
    GREEDY_CANDIDATES = 12

    LINEAGE_DUPLICATES = {"18", "23", "29"}
    ENSEMBLE_LEGS = {"38", "39"}
    BASELINE_SINGLE = ("44", 0.946252)

    # (pool name, band below the best single, rank-correlation merge threshold, allow leg selection)
    # "*_div" pools are the DIVERSITY ask handled by rule rather than by hand: legs that are
    # near-duplicates of a kept leg (rho > 0.999) are merged away, so what survives is the
    # decorrelated subset of the same parity band. STRICT pools keep the V38 discipline (0.99995),
    # which only removes true bit-level duplicates.
    POOLS = [("elite",       0.00010, 0.99995, False),
             ("plateau",     0.00050, 0.99995, False),
             ("wide",        0.00150, 0.99995, False),
             ("elite_div",   0.00010, 0.99900, True),
             ("plateau_div", 0.00050, 0.99900, True)]
    ORDER = ["equal_rank", "mean_logit", "nnls_logit", "stack_logit", "greedy_logit"]


def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def pct_rank(x):
    return pd.Series(x).rank(method="average").to_numpy(dtype="float64") / len(x)


def to_logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.clip(np.log(p / (1.0 - p)), -30.0, 30.0)


def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype="float64")
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype="float64")
    out[order] = ranks
    return out


def paired_z(y_true, s_a, s_b):
    """DeLong z for AUC(a) - AUC(b) when both are predictions on the same rows (Sun & Xu 2014)."""
    y = np.asarray(y_true).astype("int8")
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    M = np.vstack([s_a, s_b])
    X = np.asarray(M[:, pos], dtype="float64")
    Y = np.asarray(M[:, neg], dtype="float64")
    v10, v01 = np.empty((2, m)), np.empty((2, n))
    for r in range(2):
        tz = _midrank(np.concatenate([X[r], Y[r]]))
        v10[r] = (tz[:m] - _midrank(X[r])) / n
        v01[r] = (m - (tz[m:] - _midrank(Y[r]))) / m
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    z = float("inf") if se == 0.0 else (v10[0].mean() - v10[1].mean()) / se
    return v10[0].mean(), v10[1].mean(), se, z


def load_archive(directory, prefix):
    """Keyed by filename stem so plain versions and arm vectors coexist."""
    out = {}
    for fn in sorted(os.listdir(directory)):
        if fn.startswith(prefix) and fn.endswith(".csv"):
            stem = fn[len(prefix):-4]
            out[stem] = pd.read_csv(os.path.join(directory, fn)).iloc[:, 1].to_numpy("float64")
    return out


def merge_duplicates(pool, rank_tr, aucs, rho_threshold):
    """Collapse legs whose OOF ranks are indistinguishable, keeping the stronger solo."""
    kept, dropped = list(pool), []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a in kept and b in kept:
                rho = float(np.corrcoef(rank_tr[a], rank_tr[b])[0, 1])
                if rho > rho_threshold:
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
            # only the top-K remaining legs by solo AUC are candidates, and they are tried in
            # that order so the first strict improvement is also the strongest available
            cands = [j for j in sorted([k for k in range(L_tr.shape[0]) if k not in chosen],
                                       key=lambda k: -solo[k])[:CFG.GREEDY_CANDIDATES]]
            if not cands:
                break
            improved, best_auc = None, None
            for j in cands:
                a = auc_score(y_tr, np.mean([L_tr[c] for c in chosen + [j]], axis=0))
                if best_auc is None or a > best_auc:
                    best_auc, improved = a, j
            if improved is None:
                break
            chosen.append(improved)
        return (lambda R_va, L_va: np.mean([L_va[c] for c in chosen], axis=0)), chosen
    raise KeyError(name)


def sealed_score(name, pool, rank_tr, logit_tr, y, fold):
    """Apply the combiner to each fold using weights AND leg set fitted on the other nine only."""
    out = np.zeros(len(y))
    picks = []
    for f in range(CFG.N_FOLDS):
        tr, va = fold != f, fold == f
        blend, chosen = fit_combiner(name,
                                     np.vstack([rank_tr[v][tr] for v in pool]),
                                     np.vstack([logit_tr[v][tr] for v in pool]), y[tr])
        if chosen is not None:
            picks.append([pool[c] for c in chosen])
        out[va] = blend(np.vstack([rank_tr[v][va] for v in pool]),
                        np.vstack([logit_tr[v][va] for v in pool]))
    return out, picks


def insample_blend(name, pool, rank_tr, logit_tr, y):
    blend, _ = fit_combiner(name, np.vstack([rank_tr[v] for v in pool]),
                            np.vstack([logit_tr[v] for v in pool]), y)
    return blend(np.vstack([rank_tr[v] for v in pool]), np.vstack([logit_tr[v] for v in pool]))


def test_blend(name, pool, rank_tr, logit_tr, rank_te, logit_te, y):
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

    print("\n[2/5] Eligibility rules...")
    aucs_all = {v: auc_score(y, oof[v]) for v in oof}
    keep = [v for v in paired
            if v != CFG.VERSION_NAME[1:]
            and v not in CFG.LINEAGE_DUPLICATES and v not in CFG.ENSEMBLE_LEGS]
    base_v, base_auc = CFG.BASELINE_SINGLE
    best_v = max(keep, key=aucs_all.get)
    best = aucs_all[best_v]
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

    print(f"   best single OOF eligible: V{best_v} {best:.6f} "
          f"(declared baseline V{base_v} {base_auc:.6f})")
    print(f"   excluded by lineage/ensemble rule: "
          f"{sorted(set(paired) - set(keep))}")

    merged_pools, greedy_pools = {}, set()
    for pname, band, rho_thresh, allow_greedy in CFG.POOLS:
        pool = sorted([v for v in keep if aucs_all[v] >= best - band])
        merged, dropped = merge_duplicates(pool, rank_tr, aucs_all, rho_thresh)
        merged_pools[pname] = merged
        if allow_greedy:
            greedy_pools.add(pname)
        note = ("; merged " + ",".join(f"V{d}" for d, _ in dropped)) if dropped else ""
        print(f"   {pname:12s} (>= best - {band}, merge rho {rho_thresh}): "
              f"{len(merged)}/{len(pool)} legs{note}")
        print(f"       {merged}")

    print("\n[3/5] Fold-sealed combiner shootout...")
    results = {}
    for pname, _, _, allow_greedy in CFG.POOLS:
        pool = merged_pools[pname]
        combiners = CFG.ORDER if allow_greedy else CFG.ORDER[:-1]
        for name in combiners:
            t1 = time.time()
            sealed, picks = sealed_score(name, pool, rank_tr, logit_tr, y, fold)
            ins = insample_blend(name, pool, rank_tr, logit_tr, y)
            s_auc, i_auc = auc_score(y, sealed), auc_score(y, ins)
            results[(pname, name)] = (s_auc, i_auc, sealed, picks)
            print(f"   {pname:12s} x {name:12s} n={len(pool):2d} | sealed {s_auc:.6f} | "
                  f"in-sample {i_auc:.6f} | optimism {i_auc - s_auc:+.6f} | {time.time() - t1:.0f}s")

    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
    best_sc = ranked[0][1][0]
    near = [(pn, nm, s) for (pn, nm), (s, _, _, _) in results.items()
            if best_sc - s <= CFG.TIE_BAND]
    # Declared tie-break, in order: simplest combiner, then narrowest pool (fewest legs), then
    # the highest sealed score. Without the pool term the winner of a tie between same-combiner
    # cells fell out of dict insertion order, which is not a rule - found by auditing this run.
    win_pn, win_nm, win_sc = sorted(
        near, key=lambda t: (CFG.ORDER.index(t[1]), len(merged_pools[t[0]]), -t[2]))[0]
    print(f"\n   best sealed {best_sc:.6f} | {len(near)} cells inside the "
          f"{CFG.TIE_BAND:.5f} tie band | simplest declared order wins")
    print(f"   WINNER: {win_pn} pool x {win_nm} | sealed OOF {win_sc:.6f}")

    print("\n[4/5] Saving outputs...")
    pool = merged_pools[win_pn]
    oof_v51 = results[(win_pn, win_nm)][2]
    sub_v51 = test_blend(win_nm, pool, rank_tr, logit_tr, rank_te, logit_te, y)

    oof_path = os.path.join(CFG.OUT_DIR, f"oof_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": train["id"].to_numpy(), "pred": oof_v51}).to_csv(oof_path, index=False)
    print(f"   [SAVED] {oof_path} (id, pred) - sealed OOF of the winner")
    test_ids = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"), usecols=["id"])["id"].to_numpy()
    sub_path = os.path.join(CFG.OUT_DIR, f"sub_{CFG.VERSION_NAME}.csv")
    pd.DataFrame({"id": test_ids, CFG.TARGET: np.clip(sub_v51, 0.0, 1.0)}).to_csv(sub_path, index=False)
    print(f"   [SAVED] {sub_path} (id, {CFG.TARGET}) - {len(pool)} legs, combiner {win_nm}")

    print(f"\n{'=' * 80}")
    print(f"[5/5] V51 RESULTS - winner {win_nm} on the {win_pn} pool ({len(pool)} legs)")
    print(f"{'=' * 80}")
    try:
        for v in pool:
            print(f"   leg V{v:<4s} solo OOF {aucs_all[v]:.6f} | rho vs V{base_v} "
                  f"{float(np.corrcoef(rank_tr[v], rank_tr[base_v])[0, 1]):.5f}")
        rho_legs = [float(np.corrcoef(rank_tr[a], rank_tr[b])[0, 1])
                    for i, a in enumerate(pool) for b in pool[i + 1:]]

        a_w, a_b, se, z = paired_z(y, oof_v51, oof[base_v])
        print(f"\n   OOF CV (AUC): {win_sc:.6f}")
        print(f"   paired DeLong vs V{base_v}'s saved OOF: {a_w - a_b:+.6f} | SE {se:.6f} | z {z:+.2f}")
        print(f"   in-sample (unsealed) score of this combiner: "
              f"{results[(win_pn, win_nm)][1]:.6f} | optimism "
              f"{results[(win_pn, win_nm)][1] - win_sc:+.6f}")
        print(f"   mean pairwise rank rho between legs: {np.mean(rho_legs):.5f}")
        picks = results[(win_pn, win_nm)][3]
        if picks:
            print(f"   leg-set stability across the 10 folds (selection arm): "
                  f"{len({tuple(p) for p in picks})} distinct leg sets chosen")
            for f, p in enumerate(picks):
                print(f"      fold {f + 1:>2d} picks: {', '.join('V' + v for v in p)}")
        print(f"   vs best single {best:.6f} (V{best_v}): {win_sc - best:+.6f}")
        print(f"   test predictions: min {sub_v51.min():.6f} | max {sub_v51.max():.6f}")
        print("\n   full shootout ranking:")
        for (pn, nm), (s, i, _, _) in ranked:
            print(f"      {pn:13s} x {nm:12s} sealed {s:.6f} | in-sample {i:.6f} "
                  f"| optimism {i - s:+.6f}")
        print("\n   Read before shipping: combiner gains on this board have converted at ~0 to the")
        print("   public slice (V38 +0.000162 CV -> 0.00000 LB). A fitted combiner's in-sample")
        print("   advantage is not a transferable one; the sealed column is the only honest number.")
    except Exception as exc:
        print(f"   [DIAGNOSTICS FAILED AFTER SAVE] {type(exc).__name__}: {exc}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
