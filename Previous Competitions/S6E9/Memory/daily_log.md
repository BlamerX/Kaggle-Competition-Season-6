# S6E9 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first), with the format block first and dated entries written below it
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---
# Format

The format block stays first; dated entries are written below it.

### DD-MM-YYYY

- **Goal**: 
- **Experiments**:
- **Timing**:
- **Key Learning**:
- **Status**: 
---


### 21-09-2026
- **Goal**: Answer the user's question — *why did the recent versions score worse and the old ones better, and can we go back to a single model built from FE and training alone* — then build that model
- **Experiments**:
  - Audited all 29 saved OOF/sub pairs against their logged LB: recomputed every OOF from file (all matched their logs to <1e-6), measured each submission's mean rank distance to our own consensus test ranking, and correlated both with LB
  - **V30: the plain single model** — 774 lines, no arms, no harness, no blend, no refit, no prior, no DeLong. V19's FE verified **byte-identical by sha256**, V22's config verified identical by dict diff, training call switched to the booster API because the sklearn wrapper alone shifts the test ranking ~1,600 positions. One change: **10 folds instead of 5**
  - Found and corrected a defect of mine from the V29 save block while answering the question (`sigmoid` applied to probabilities → `oof_v29/sub_v29` are monotone-squashed, ranks/AUC/LB unaffected but the values are not calibrated)
- **Timing**: V30 45.3 min (my 25–30 min estimate was wrong: 10 folds cost 4.2× a 5-fold run's wall time, not 2×, because each fold also trains on 12.5% more rows); audit ~1 min of local file reads
- **Key Learning**:
  - **The reason is measurement, not modelling.** Across the ten versions sharing this matrix, honest OOF spans **0.000107** while LB spans **0.000130** — the board *adds* spread rather than resolving quality — and **Spearman(OOF, LB) = −0.619**: our best OOF ever (V25, 0.946094) has the lowest LB of the group (0.94628), and V19 has the *lowest* OOF and is joint second-best
  - **What does predict the board is divergence from our own consensus ranking: Spearman −0.770** over ten versions (−0.667 before V29/V30). Ordered by divergence: V22 869→.94640, V28/V29 885→.94640, V20 917→.94639, V23 1,073→.94641, **V30 1,111→.94639**, V27 1,140→.94638, V26 1,462→.94637, V19 1,922→.94639, V25 2,400→.94628. Every post-V23 technique (ExtraTrees split bias, logistic prior as base_margin, cross keys, a second family, unconstrained bins, a neural view, and now a fold-count change) moved the test ranking, and the public 20% billed the variance while the models' expected AUC was unchanged
  - **V30's own result: the fold count is real but about two-thirds of the advertised size** — OOF +0.000097 (external claim +0.00015) — and the LB is −0.00002, exactly its divergence band. Independent corroboration that the *model* improved: our LB−OOF gap collapsed from a stable +0.00032/+0.00035 to **+0.00022**, because a 10-fold OOF sits closer to the ensemble-of-ten that actually predicts the test set
  - **Decision the user's instinct earned:** the ablation-harness era is over for shipping. Going forward a version is either a plain single model or an experiment that never becomes a submission. V23 (0.94641) stays the best *score*; V30 is the best *model*. The two differ by one noise unit, and only the model is worth optimising
  - **Fold count is now spent**: 20 folds would cost ~90 min for a fraction of the remaining gain (3→20 splits measured +0.00015 in total), so there is no second bite here
- **Status**: ✅ Root cause identified with numbers and a −0.770 predictor; V30 shipped the only remaining measured training gain and is our best model despite a board-neutral score

### 21-09-2026
- **Goal**: Settle the two questions the morning's plan revision left open — is bin resolution really the +0.002 lever the discussions claimed, and can a non-tree family stand near the incumbent on our matrix?
- **Experiments**:
  - V28 rerun at **`max_bin` 1024 → 16384**, i.e. unconstrained (income holds 14,667 distinct pool values), in-run V22 control, single split, with a trees-level integrity guard
  - V29 **TabM dual view**: XGBoost control + pytabkit TabM (k=4, PWL embeddings, 25 epochs) fed the top 120 of 312 columns by *that fold's own* control gain (nested, leak-free), fold-0 time canary at 20 min, and one cross-fitted linear blend weight — the first neural model ever run on the V19+ artifact matrix
  - Offline reproducibility audit across the three saved copies of V22's estimator (V22, V28 control, V29 control), OOF and test
- **Timing**: V28 66.5 min (control 10.7 vs raised arm 45.3 = 4.2×); V29 32.4 min (control 10.1, TabM 12.7, FE ~2, overhead ~7)
- **Key Learning**:
  - **Resolution axis closed by construction.** 16× the bins moved the OOF **−0.000002 at z = −0.40**. The guard proved the parameter reached the model — income-derived columns used 6,456 distinct thresholds at 1024 bins vs **7,972** at 16384 (`TE_Annual_Income_USD_cat_auto` 318 → 538 cuts) — so this is a real null, not a silently ignored setting. The two arms correlate 0.99992, so bin resolution is not a diversity source either. **Keep 1024 permanently**; the community's +0.00201 belonged to a 0.94172 baseline without our digit + exact-value TE stack
  - **Neural family question answered: nets do not exploit this matrix better than trees.** TabM reached OOF **0.94564 (−0.00043)** and ranked **0.9951–0.9964** with the GBM — weak *and* correlated, the worst quadrant for a blend. It even fell short of V6's 0.94585 on the older 83-feature set. Expected blend value ≈ +0.00005, so **no rerun was justified** once the blend number was lost to a bug
  - **Two self-inflicted bugs, one instructive.** (1) `TABM_PARAMS` moved inside `class CFG` inherited a `CFG.RANDOM_SEED` reference — a class name is not bound inside its own body, so the module died at import. (2) `paired_z` returns 4 values and two fresh call sites unpacked 3 — the *same* arity bug fixed a day earlier, which survived because my "unit test" called the helper with a correct 4-name unpack instead of executing the script's real line. **Testing a paraphrase of a call site is not testing the call site** — grep the call sites of multi-return helpers. The V28-style guard held both times: outputs survived, the run degraded instead of dying
  - **Why the last six versions "got worse" — it was not quality.** Across the eight versions sharing this matrix, OOF spans 0.000107 while LB spans 0.000130, and **Spearman(OOF, LB) = −0.619**: our best OOF ever (V25, 0.946094) has the *lowest* LB of the group (0.94628) and the lowest OOF (V19) is joint second-best. What does track the board is **divergence from our own consensus test ranking** (Spearman −0.667) — the four least divergent submissions are the four best-scoring (838–1,061 mean rank positions) and the most divergent (V25, 2,343) is last. All of these models share the same expected AUC, so disturbing the board buys variance with no bias reduction and the public slice charges for it. **Rule: among equal-OOF candidates submit the least divergent model, and stop reading sub-0.00013 board moves as verdicts** — which is also the vindication of going back to one plain FE+training model
  - **V29 save-path bug found while auditing that claim:** the save block applied `_sigmoid` unconditionally, so `oof_v29.csv`/`sub_v29.csv` hold sigmoid(probability) (log range 0.5000–0.7310). Ranks, AUC and the LB are unaffected, but **compare those files by rank only** — and note this is what made V28-vs-V29 look like "identical ranks, values differing at 1e-7"
  - **Reproducibility, measured:** V28's and V29's controls rank **identically** (Spearman 1.0000000, mean |rank difference| 0.0 of 668,665 OOF rows and of 286,571 test rows), so the whole pipeline is deterministic across Kaggle sessions including the GPU. V22 — same config via the sklearn wrapper rather than the booster API — differs by a mean 1,600 OOF rank positions and 532 test positions **yet scored the identical LB 0.94640**. That puts an empirical floor under the noise debate: rank perturbations of ~0.2% of positions cannot be confirmed on the public board at all
- **Status**: ⚠️ Two axes closed by measurement (resolution, neural family), one submission-equivalent confirmation of determinism, and the plan now rests on the mechanical fold-count gain

### 21-09-2026
- **Goal**: Answer "is the 4-5 arm × 5 fold × 2 stage harness still worth it, and can features built from scratch — not the public artefact ones — add anything?", then run the one experiment that survived that scrutiny
- **Experiments**:
  - Protocol review: ~700 min of compute since V22 (V22 116.8 + V23 23.4 + V24 13.7 + V25 90.2 + V26 387.6 + V27 69.6) bought a net **−0.00003 LB**, because the harness was built to resolve +0.00002 effects that turned out to be noise
  - From-scratch offline screens against `oof_v23` (allowed offline; never read inside a script): exact row identity pool↔original (0 train / 1 test twin), `id`/row-order structure (AUC(id,y) = 0.49999), single-column lift on the 4 numerics that never had one, **all 91 pairwise lifts**, **201 continuous arithmetic derivatives**, missingness (comp train/test have zero NaNs), GBM fitted on the original 10k (its own CV on orig rows 0.9024), and an **oracle** cell-recalibration bound
  - Blend screen over all 26 saved OOFs with **cross-fitted weights** (choose w on 4/5 of the rows, score the held-out 1/5)
  - Four parallel investigations: S6E9 discussion verification at source, cross-episode winning-writeup mining, raw-data resolution forensics, and an audit of all 27 scripts for parameters never varied
  - V28: `max_bin` 1024 → 4096 (30.6 min), then → 16384 unconstrained (66.5 min), in-run V22 control, paired DeLong, one split, no Stage B
- **Timing**: V28 first run 30.6 min, second run 66.5 min (a1 alone 45.3 min = 4.2× the control's 10.7); local screens ~25 min total at zero GPU cost
- **Key Learning**:
  - **Every from-scratch feature channel is dead**: single-column lift on the uncovered numerics +0.000000 each, all 91 pairwise lifts +0.000000 each, best of 201 arithmetic derivatives +0.000002, orig-only GBM +0.000003, identity and row-order nil. **And the whole "known distribution" family is closed with an oracle bound**: shrinking logits toward *true* cell centres costs −0.0003 even at the coarsest usable key (30 cells) — the model's per-row ranking strictly dominates cell-level truth, independently confirmed by thread 740775 (per-cell isotonic −0.0003, Platt −0.0011)
  - **The max_bin headline I built a plan on did not survive source checking.** The +0.00201 is *one lever reported twice* (digits and the raised cap read +0.00201 each; both together +0.00238, so digits add +0.00036) — and we have carried the fixed digit block since V19, so it was already banked. The best published single model on this board (Tilii's XGBoost, CV 0.94623 / LB 0.94640) runs `max_bin: 1024`, and community-measured hyper-parameter search tops out at +0.00027
  - **V28 settled it directly: 16× the bins moved the OOF −0.000002 at z = −0.40.** The integrity guard is what makes that conclusive — the trees report 6,456 distinct income thresholds at 1024 bins vs **7,972 at 16384**, so the parameter took effect and the null is real. Resolution is closed by construction; keep 1024
  - **Two unbanked gains did emerge, and both are structural not representational:** 10-fold instead of 5-fold is worth **+0.00015** (measured by two independent sources; never tried here, and every fold currently discards 20% of the labels), and a *decorrelated second view* is worth **+0.000135 at z = 9.1** cross-fitted (v23 ⊕ v3; the partner ranking follows decorrelation × quality, with v4 the least correlated at ρ = 0.9854). Two bin settings, by contrast, correlate 0.99992 — representation knobs are not a diversity source
  - **Why V18's blend failed, identified at source:** on this board a non-linear combiner measured CV +0.0005 → **LB −0.0004** (≈8σ) while a plain linear one held CV 0.94275 → LB 0.94278. So any blend here is a single cross-fitted linear weight — never a hill-climber or ridge/tree stacker
  - **Gate revised from per-version to cumulative.** No remaining lever individually reaches the +0.00034 the target needs, so a per-version materiality gate would block everything: accumulate independently-measured orthogonal gains (fold count, one decorrelated view, the V23 refit blend), each judged by its own z > 3, and submit **once** from the combined build
  - Also retired by measurement rather than by assumption: the additive/depth-1 leg (best additive model 0.94251, best depth-1 GBM 0.94142 — 0.0035–0.0046 below us), monotone constraints (−0.00024), CatBoost's deficit as a family verdict (its `max_bin` may have been silently ignored in V24 — it wants `border_count`)
  - Reproducibility datapoint with real value: V28's control re-ran V22's estimator on V22's matrix and returned **LB 0.94640, identical to the last digit**
  - Process failure owned: a `paired_z` unpacking bug (it returns 4 values) threw *after* training but *before* saving and cost 31 min of GPU artifacts. Outputs now save before any post-hoc diagnostic, and diagnostics are wrapped so commentary can never destroy the product
- **Status**: ⚠️ Partial — resolution axis closed by a clean null, ~700 min of micro-ablation retired, two structural gains (10-fold, decorrelated linear blend) now queued as V29/V30

### 19-09-2026
- **Goal**: Combine three queued ideas (V21's crosses, `base_margin`, gain pruning) in ONE version without repeating V21's bundling mistake
- **Experiments**:
  - V25: factorial ablation — one build per fold, in-run control a0 = V22's winning config, four single-factor arms: a1 + six cross keys as pure addition, a2 + per-row `base_margin` from a logistic fitted on the original 10k per fold (clip ±5), a3 + in-run gain pruning to 94 features via a 600-tree lr-0.05 probe, a4 all three
  - Cross block appended last and excluded from the redundancy scan so a0's matrix is byte-identical to V22's (312 base-block + 54 cross-block columns)
  - Booster API (`xgb.DMatrix` + `xgb.train`) because the sklearn wrapper cannot pass a per-row base_margin; in-run paired DeLong of every arm vs a0
  - Stage A rs=42 → a2 0.94609, a0 0.94607, a3 0.94603, a4 0.94602, a1 0.94601; DeLong vs a0: a1 -0.00006 z=-3.98, a3 -0.00004 z=-2.77, a4 -0.00005, a2 +0.00002 z=+1.29
  - Stage B rs=7 → a2 0.94611 (z=+2.45) declared winner; submission averaged both splits
  - Post-mortem on the saved oof/sub CSVs (allowed offline; not inside a version script): AUC/LB gap table for v19-v25, paired DeLong across versions, test rank divergence, per-decile local AUC, prior clipping fractions
- **Timing**: 90.2 min total (Stage A 53.2 min across 5 folds, Stage B 33.8 min) — cheaper than V22's 116.8 min with more arms
- **Key Learning**:
  - a0 reproduced V22 to 6e-6 (0.94607 vs 0.946076): the shared-build ablation harness is trustworthy and now the preferred way to test several ideas at once
  - **The feature axis is closed.** V21's six cross keys, tested alone for the first time, REGRESS at z = -3.98; gain pruning to 94 also regresses (z = -2.77). V21's failure was not only the removals — the crosses themselves add nothing
  - LB 0.94628 / -0.00012 despite the best OOF we have ever had (0.94609). Diagnosis: no code error (clean submission file; prior clipped 20.82% of train vs 20.72% of test, all on the -5 side; OOF and test prediction distributions match to 4 decimals). The a2 gain is +0.000018 at z = +1.21 — a **tie** — and it lives entirely in the bottom three prediction deciles, which carry 174/116,779 positives = **0.15% of the AUC pair mass**, while reshuffling 31% of the test board (vs 16% for V23)
  - Public LB does not order our submissions at the 0.0001 scale: V19's OOF is 0.000107 worse than V25's yet its LB is 0.00011 better. OOF→LB gap jumped from a stable +0.00032/+0.00035 to +0.00019
  - **Process fix: a submission slot requires z > 3 on honest OOF, or an inference-side variant of V23. Ties stay CV experiments.** Next submission reverts to the V23 line (LB 0.94641)
  - Memory-rule clarification from the user: reading saved OOF/sub files for offline diagnosis and idea-screening is allowed; the single-model rule only forbids reading them *inside* a version script
- **Status**: ⚠️ Partial (best OOF, rejected submission, ablation design adopted)

### 20-09-2026
- **Goal**: Run the two queued structural probes (V26 ExtraTrees bias on CPU, V27 original-row weight + pairwise loss on GPU) in parallel, then answer why recent submissions keep landing *below* V23 even though OOF keeps improving
- **Experiments**:
  - V26 (CPU LightGBM, 387.6 min): a0 = V19 params control, a1 = + `extra_trees`, a2 = + lookup capacity, a3 = + wide/weak/many → **a1 +0.00007 z=+3.58 and a3 +0.00010 z=+5.22 (rs=42), a3 +0.00008 z=+4.11 (rs=7): first arms ever to clear our z > 3 gate, on both splits**
  - V27 (GPU XGBoost, 69.6 min): a1 **drop the original 10k rows = +0.00002, z=+2.66**, a2 up-weight ×10 = -0.00003, a3 `rank:pairwise` = **-0.000329, z=-42.56**, a4 pairwise + 10x replication = -0.00069; Stage B correctly skipped, a0 reproduced V20 to 1e-5
  - Cross-version analysis on the saved oof/sub CSVs of v19-v27: paired DeLong, test-board rank divergence, OOF→LB regression, per-decile attribution, identical-OOF pair comparison
  - LB attribution caveat: the two scores arrived in one message without labels; assigned by run order (V26 0.94637, V27 0.94638). Both are below V23's 0.94641 and both had OOF ≈ 0.94608, so the conclusion is unchanged if they are swapped
- **Timing**: V26 387.6 min CPU (Stage A 206.3 + Stage B 177.9; a3 needs 6,000-8,200 trees per fold) + V27 69.6 min GPU, run concurrently on separate instances
- **Key Learning**:
  - **The main reason our scores look "lowered": we are now measuring leaderboard noise, not regressions.** Across the nine v19-v27 submissions the honest OOF spread is 0.000161 while the LB spread is 0.00026 — *the LB adds variance rather than resolving our ordering* — and the rank correlation between them is **Spearman 0.071** (Pearson 0.634 comes only from the clearly-worse v21/v24 outliers). Residual sd of LB about the OOF regression line is **0.000066**, i.e. **our unit of progress (±0.00002) is about one third of the public-slice scatter**
  - Direct evidence: three pairs with *identical* OOF differ on LB purely through test-side reordering — v20 vs v23 (ΔOOF 0.000000, ΔLB -0.00002, 892 mean rank diff), v22 vs v27 (0.000000, +0.00002, 1,725), v26 vs v27 (+0.000009, -0.00001, 2,154). Even V23's celebrated +0.00002 refit gain is exactly one noise unit
  - So v25 (0.94628), v26 (0.94637) and v27 (0.94638) are not three regressions; they are three draws. Nothing in the files was broken — every version's OOF recomputed from its CSV matches its log to <1e-6 and each sub correlates ≥0.9995 with V23's
  - **The real finding: ExtraTrees split randomisation closed the LightGBM→XGBoost gap.** LightGBM was 0.00007 behind XGBoost with greedy splits; with random thresholds + shallow/wide/weak it reaches 0.946085, statistically level with V22/V27's 0.946076 and V25's 0.946094. The gap was a *split-bias* gap, not an information gap — and one config of one grid gained +0.00010, so the tuning surface there is genuinely unexplored (a3 was chosen a priori, not searched)
  - My lookup-memorisation mechanism was wrong and the opposite happened: fully-grown random trees (a2) lost badly (-0.00023). Random thresholds help as *regularisation of a shallow model*, not as value memorisation
  - **Two structural assumptions retired**: the original 10k rows are not earning their concat (dropping them ties at +0.00002 with 1.47% of the training matrix removed, up-weighting them is negative), and the pairwise-AUC theory is dead (-0.000329, z=-42.6, wildly unstable across folds). Pointwise logloss was already near-optimal for ranking here
  - Diagnostic worth noting even though we will not use it: an equal-logit blend of the V26 and V27 winners scores 0.946118 on OOF, +0.000033 over the better member (test predictions correlate 0.9997), consistent with V18's ≤0.000066 combiner ceiling. Single-model rule stands
  - **Decision for the remaining days**: stop treating ±0.00002 LB moves as signal; judge versions on OOF only, keep V23's recipe as the submission line, and if one more experiment is run make it ExtraTrees-direction tuning (V28), which is the only lever with a reproducible +0.00010 behind it
- **Status**: ✅ Good (first gate-cleared gain since V19) + ⚠️ noise-floor finding that reframes every recent version

### 19-09-2026
- **Goal**: Test a second model family on V20's artifact matrix
- **Experiments**:
  - V24: GPU CatBoost `depth=6`, lr=0.03, `l2_leaf_reg=3.0`, `min_data_in_leaf=20`, Bernoulli 0.8, `random_strength=1.0`, `max_bin=1024`, od_wait=500, `use_best_model=True`; features/params/CV otherwise identical to V20
  - Best iterations 1243-1549 per fold (vs XGBoost's 3375-4984), fold times 126-140 s
- **Timing**: 13.7 min total — fastest full run yet (V20 20.6 min, V19 33.7 min)
- **Key Learning**:
  - OOF 0.94593 and LB 0.94615: rejected, -0.00013 OOF / -0.00024 LB vs V20; CatBoost stayed behind XGBoost and LightGBM on identical features, so the family ordering is settled and the axis is closed
  - The feature story replicated: `TE_lift_trigram_cat_10` (8.20%) and `TE_lift_trigram_cat_auto` (8.18%) took the top two slots, plus `TE_lift_Subsidy_Available_cat_auto` at #7 — the lift trigram is the dominant signal in every family, not an XGBoost-specific artefact
  - Cheap to run (~1.5k iterations), so it is useful as a fast diagnostic harness, not as a scorer
- **Status**: ❌ Failed

### 19-09-2026
- **Goal**: Buy score on the inference side instead of the feature side — give the test predictions a model that saw 100% of the labels
- **Experiments**:
  - V23: V20's CV loop unchanged (same 378 features, same depth=4 lossguide recipe, same KFold rs=42), then `[3b]` refits one model on train + original (678,665 rows x 312 cols) at `n_estimators` = mean best iteration (4112), no early stopping
  - `test_probs = 0.50 x fold-average + 0.50 x refit`; OOF left purely out-of-fold so the CV stays honest and comparable
  - Instrumented the blend: Pearson / Spearman / mean |rank difference| between refit and fold average on the 286,571 test rows
- **Timing**: 23.4 min total (CV 17.0 min + refit 2.2 min + loading/FE); folds 200 s, 219 s, 190 s, 189 s, 223 s
- **Key Learning**:
  - LB 0.94641 (new best, +0.00002 over V20/V19's 0.94639) with OOF 0.94606 unchanged by construction — the first gain we have taken from inference rather than features, and it cost 2.2 minutes of compute
  - The refit barely moved the ranking: Pearson 0.99959, Spearman 0.99946, mean |rank diff| 1,806 of 286,571 rows. So the +25% labels is worth a little, and the small disagreement that survives is exactly where the remaining inference-side variance sits (blend weight, iteration count, repeated-split averaging)
  - Fold importances matched V20 within rounding (`TE_lift_trigram_cat_auto` 0.14488), confirming the deterministic CV path was not perturbed
- **Status**: 🏆 Best

### 19-09-2026
- **Goal**: Tune the estimator on our own matrix — V20 inherited najiama's parameters, which were tuned on his feature set, not ours
- **Experiments**:
  - V22: two-stage search, 6 configs sharing one fold-matrix build; Stage A on the canonical KFold(rs=42), top-3 re-run on KFold(rs=7), winner by two-split mean
  - c0 = V20 params (control), c1 = depth 3 / 8 leaves / gamma 1.0 / colsample 0.85, c2 = deeper + heavy leaf reg, c3 = sparse cols + high gamma, c4 = low min_child_weight, c5 = tiny lr near-depthless lossguide
  - Stage A OOF: c1 0.94608, c0 0.94606, c4 0.94605, c2 0.94593, c3 0.94576, c5 0.94546; rs=7: c1 0.94608, c0 0.94607, c4 0.94605
  - Submission = winner's test predictions averaged over both splits; OOF = rs=42 winner
- **Timing**: 116.8 min total (Stage A 75.9 min, Stage B 37.9 min)
- **Key Learning**:
  - LB 0.94640 and OOF 0.94608 — new best single-model OOF, +0.00002 over the in-run control on both splits; the control reproduced V20's 0.94606 exactly, which validates the harness
  - Direction is the useful part: deeper trees and heavier leaf regularisation lose (-0.00013), tiny learning rate loses badly (-0.00060), while depth 3 with 3x the columns and ~5.5k trees wins narrowly — this matrix wants wide, weak, many learners
  - c1 re-concentrated gain onto the artifact crosses (lift-trigram auto+10 = 0.419 of fold-1 gain vs 0.253 for c0) and pushed raw `Environmental_Concern_Level` to #3: the inverse of V21's collapse, confirming a shallow model needs both a wide pool and the pre-computed crosses
- **Status**: ✅ Good

### 19-09-2026
- **Goal**: Reallocate V20's feature budget from dead weight into explicit conditional crosses (Simpson-reversal keys from the discussion threads)
- **Experiments**:
  - V21: V20 model unchanged; added 6 cross keys (charging-total x Home_Charging, City x Home_Charging, ECL x Home_Charging, ECL x City, income-band x ECL x Subsidy, ECL x Subsidy x commute with 5.0 km isolated), each Triple-TE'd plus its own target-free lift column; removed the digit block and 4 income-anomaly flags
  - Fixed bin edges in CFG so train/test/orig bin identically; 159 base + 189 TE = 348 features (V20 had 378)
  - Paired DeLong against V20/V19/V10/V14 to judge the change
- **Timing**: 16.4 min total (186 s, 181 s, 169 s, 176 s, 183 s) — fastest full run yet
- **Key Learning**:
  - OOF 0.94597 and LB 0.94629: REJECTED, delta -0.00009 vs V20 at z = -5.01 (significant regression, and LB agreed)
  - The addition was right and the subtraction was wrong: `cx_inc_x_ecl_x_sub` plus its lift took five of the top eight gain slots, so the crosses do carry signal
  - Removing the digits and flags collapsed the gain distribution — `_ECL_x_Subsidy` grabbed 0.5899 of total gain (5.4x its V20 share) and the lift-trigram family fell from ~35% combined to ~6.7%; only 27 columns were dropped as redundant versus 149 in V20
  - Process lesson: bundling an addition with a removal destroyed attribution. One change per version, and a shallow lossguide model needs a wide candidate pool so no single feature monopolises splits
- **Status**: ❌ Failed

### 19-09-2026
- **Goal**: Isolate the model family by running V19's exact feature matrix through the reference XGBoost depth=4 lossguide recipe
- **Experiments**:
  - V20: GPU XGBoost depth=4, lossguide, max_leaves=16, gamma=3.673, min_child_weight=4.532, subsample=0.740, colsample=0.570, alpha=0.752, lambda=0.619, lr=0.01, ES=500; converged 3375-4984 trees
  - Features reused byte-for-byte from V19: 180 base + 198 Triple TE = 378; same KFold(5, shuffle=True, rs=42) split and original-data concat
  - Offline paired DeLong of every new OOF against V19/V14/V10/V3 to separate real gains from CV noise
  - Research pass: read the Simpson's-paradox thread (Charging x HomeCharging and City x HomeCharging reversals), the original-dataset logistic thread (4-feature LR beats XGB/LGB/TabPFN on the 10k source; refit gives inc 2.292 / ecl 1.078 / sub 3.385 / ra_med -1.669 / ra_high -2.960, AUC 0.93766 on comp), the replication-aware Newton boosting thread, and the digit/TE ablation thread
  - Screened 7 target-free propensity features and Simpson crosses against V19's OOF: no linearly-accessible residual signal (but that screen cannot rule out tree-partition value)
- **Timing**: 20.6 min total (203 s, 218 s, 207 s, 203 s, 233 s) — 40% faster than V19's 33.7 min
- **Key Learning**:
  - OOF 0.94606 vs V19 0.94599: Δ +0.00007 at z = +4.76 — our first statistically significant single-model gain; LB stayed at exactly 0.94639
  - Public LB (20% of test) cannot resolve +0.00007, so OOF significance and LB movement are now decoupled; V14's OOF remains statistically tied (z = -0.83)
  - `TE_lift_trigram_cat_auto` is the #1 feature by gain under depth-4 XGBoost (0.1449) vs #4 under LightGBM — shallow models need the pre-computed artifact crosses more
  - Structural flags and digit features earned almost no gain in XGBoost, independently matching the reference notebook's decision to prune them
  - Rank 1 is 0.94675 and Deotte is 0.94672 on only 3 submissions, so roughly +0.0003 of real generalizing signal is still available above us
- **Status**: ✅ Good

### 19-09-2026
- **Goal**: Convert the verified generator-artifact findings into a new single model (V19) on the proven V10 LightGBM pipeline
- **Experiments**:
  - V19: CPU LightGBM 5-fold CV, original-data concatenation, Triple TE, V10 params unchanged
  - UPGRADE 1: Fixed the digit-extraction bug (`np.rint(col*1e4)` integer divmods) present in all 18 prior versions
  - UPGRADE 2: 12 target-free generator-lift features (pool ÷ original frequency) on exact income, 100-dollar band, integer commute, 6 categoricals, ECL×Subsidy×Anxiety trigram, plus `novel_inc`
  - UPGRADE 3: Structural flags `_below_buy_bound` (41,667) and `_dead_zone_exact` (31,004-41,970)
  - A paired DeLong gate against the saved V10 OOF was written, then removed before running to keep V19 a self-contained single model
  - 180 base + 198 Triple TE = 378 features after dropping 149 redundant/constant columns
- **Timing**: 33.7 min total; fold times were 328 s, 408 s, 348 s, 356 s, and 368 s
- **Key Learning**:
  - OOF AUC 0.94599 and LB Score 0.94639 — a new best, +0.00003 over V10 and the first real gain since V10
  - `TE_lift_trigram_cat_auto` ranked #4 in fold-1 importance (531), so the frequency-shaping artifact carries signal the model could not reach before
  - Fold std stayed at 0.00069 and the OOF gain was +0.00002, so the improvement is genuine but at the edge of the CV noise floor; the LB gain is the stronger evidence
- **Status**: 🏆 Best

### 19-09-2026
- **Goal**: Test forward-stepwise hill climbing over all 17 saved OOF predictions, then verify discussion findings and audit scripts for errors
- **Experiments**:
  - V18: CPU hill climber (weight step 0.05) over V1–V17 OOFs, compared against simple/rank/logit averages and RidgeCV meta-learner
  - Selected ensemble: V14=0.35, V3=0.28, V11=0.15, V6=0.13, V2=0.05, V8=0.05; hill OOF 0.94623 (optimistic — weights fit OOF directly), other methods ≤0.94601
  - Dataset verification: confirmed all discussion claims (30k spike, dead zone, ECL/commute rules, tenths corruption 89.63%, recipe AUC 0.9377)
  - New artifact findings: income lift buckets non-monotone (under-produced buy 23.19% vs over-produced ~17%), novel-income rows buy 19.56%, ECL=3 cells out-perform recipe formula by ~22–31%, `min income 41,667` structural zero bound, adversarial validation AUC 0.501 (no train/test shift)
  - Script audit: digit-extraction bug (`// (10**k)` with negative k) present in all 17 versions; TE verified leak-free; OOF integrity confirmed
- **Timing**: ~10 min for subsampled hill-climb recomputation; full-data grid search infeasible in reasonable time
- **Key Learning**:
  - LB finished at 0.94635, 0.00001 below V10's 0.94636 — ensembling inside a 0.996+ correlated pool is exhausted
  - The +0.00015 OOF hill-climb gain was weight-fitting noise, confirming the honest-combiner ceiling from the discussions
  - Next lever is generator-artifact features (lift/novelty/structural flags) plus fixing the digit bug, not new combiners
- **Status**: ⚠️ Partial

### 11-09-2026
- **Goal**: Test RealMLP with the V14 full feature pipeline, original-data concatenation, Triple TE, and short three-epoch training
- **Experiments**:
  - V17: GPU RealMLP 5-fold CV with PBLD embeddings and an 8-model ensemble
  - Architecture: three hidden layers of 256 units, EMA, label smoothing, and 3 epochs
  - Used 159 engineered features expanded to 293 model inputs: 124 categorical and 169 numerical
- **Timing**: 97.3 min total; fold times were 1052 s, 1105 s, 1123 s, 1195 s, and 1240 s
- **Key Learning**:
  - OOF AUC reached 0.94582 and LB Score reached 0.94612, tying V9
  - RealMLP remained below V14’s 0.94630 LB despite using the full feature pipeline and original-data concatenation
  - The model was substantially slower than the strongest tree-based baselines
- **Status**: ✅ Good

### 11-09-2026
- **Goal**: Test five forensic-targeted features on the V14 depth-3 XGBoost baseline without pseudo-labels
- **Experiments**:
  - V16: GPU XGBoost depth 3, 5-fold CV, based on V14’s proven feature pipeline
  - Added subsidy × home-charging bigram, recipe × subsidy/income interaction, and ECL-specific buyer-centroid distances
  - Distance features were computed per fold to avoid leakage
  - Retained 164 final features, with 2 of 2 forensic features surviving selection and 5 forensic signals reported in importance
- **Timing**: 28.6 min total; fold times were 283 s, 355 s, 333 s, 282 s, and 334 s
- **Key Learning**:
  - OOF AUC reached 0.94595 and LB Score reached 0.94625, below V14’s 0.94630 LB
  - The forensic features contributed small fold-1 importance, led by `TE_bigram_Sub_HomeCharging`
  - No pseudo-labels were used, isolating the impact of the forensic feature additions
- **Status**: ✅ Good

### 10-09-2026
- **Goal**: Test exact-match lookup plus CPU KDTree KNN as a non-tree baseline
- **Experiments**:
  - V15: 5-fold CPU KNN with k=10 fallback and exact-match lookup
  - Encoded 24 features for KDTree distance search
  - Checked train/test key overlap and train-key uniqueness before modeling
- **Timing**: 19.5 min total; fold times were 164 s, 172 s, 166 s, 164 s, and 146 s
- **Key Learning**:
  - OOF AUC was 0.91359 and LB Score was 0.91256, with a negative -0.00103 LB–OOF gap
  - Exact matches were 0% for validation and test; every train key was unique, confirming no deterministic lookup shortcut
  - KNN was substantially weaker than the feature-engineered tree and neural baselines
- **Status**: ❌ Failed

### 09-09-2026
- **Goal**: Test high-confidence pseudo-labeling using V10 teacher predictions with the proven V12 depth-3 XGBoost student
- **Experiments**:
  - V14: GPU XGBoost depth 3 with 5-fold CV, original-data concatenation, and pseudo-labeled test rows
  - Teacher: V10 predictions; pseudo-label thresholds `p>=0.98` and `p<=0.02`
  - Added 150,859 pseudo-labeled rows: 999 buyers and 149,860 non-buyers, each at half weight
  - Used V10/V12 proven features with 159 final columns and 97 TE columns
- **Timing**: 35.4 min total; fold times were 404 s, 405 s, 365 s, 345 s, and 464 s
- **Key Learning**:
  - OOF AUC reached 0.94608 and LB Score reached 0.94630, ranking third overall
  - Pseudo-labeling improved V12’s LB from 0.94629 to 0.94630, but the gain was only 0.00001
  - `_ECL_x_Subsidy`, `TE_trigram_Sub_ECL_RA`, and `_ev_recipe` dominated fold-1 importance
- **Status**: ✅ Good

### 09-09-2026
- **Goal**: Test DCN-V2 deep cross networks with the compact evidence-based feature set used by V6
- **Experiments**:
  - V13: CUDA DCN-V2 5-fold CV with cross layers, low-rank factorization, and 4 experts
  - V3 features plus V10/V12 proven interactions and V6 evidence-based selection
  - Reduced the input to 79 features: 18 categorical and 61 numerical
- **Timing**: 14.4 min total; fold times were 127 s, 124 s, 123 s, 124 s, and 121 s
- **Key Learning**:
  - OOF AUC reached 0.94490 and LB Score reached 0.94568, tying V7 but with a larger +0.00078 LB–OOF gap
  - DCN-V2’s cross architecture did not match the tree-based or TabM models on this feature set
  - The compact 79-feature representation remained computationally efficient, but predictive performance was below the current Top 5
- **Status**: ✅ Good

### 09-09-2026
- **Goal**: Test Deotte-style shallow XGBoost with all discussion-derived synthetic patterns and inner K-fold target encoding
- **Experiments**:
  - V12: GPU XGBoost depth 3, 5-fold CV, per-fold competition + original-data concatenation
  - V3 full FE plus V10/V11 proven targeted features and 67-column inner K-fold TE
  - Implemented all 10 discussion patterns, excluding the recipe score after V11 showed cannibalization
  - Trained with depth 3, learning rate 0.0025, and 50,000 maximum estimators
- **Timing**: 29.7 min total; fold times were 292 s, 346 s, 383 s, 275 s, and 351 s
- **Key Learning**:
  - OOF AUC reached 0.94598 and LB Score reached 0.94629, tying V11 for third overall
  - `TE_trigram_Sub_ECL_RA` was the strongest discussion-derived feature at 18.28% importance
  - The shallow Deotte-style model remained competitive but did not exceed V10’s 0.94636 or V3’s 0.94634
- **Status**: ✅ Good

### 09-09-2026
- **Goal**: Add deep-analysis features to the corrected V10 targeted-feature LightGBM pipeline
- **Experiments**:
  - V11: CPU LightGBM 5-fold CV with V10 targeted bigrams and selective groupby deviations
  - Added six deep-analysis features: buy/no-buy flags, income-ending pattern, recipe residual, trigram interaction, and income-band × subsidy bigram
  - Expanded to 173 base features, 67 TE source columns, 201 Triple TE features, and 374 total features
- **Timing**: 30.3 min total; fold times were 316 s, 345 s, 328 s, 326 s, and 345 s
- **Key Learning**:
  - OOF AUC reached 0.94599 and LB Score reached 0.94629, ranking third overall
  - `_recipe_residual` was the strongest fold-1 feature at 996 importance, followed by trigram target encodings
  - The new features slightly underperformed V10’s 0.94636 LB by 0.00007, despite improving over V3’s feature recipe in some fold signals
- **Status**: ✅ Success

### 09-09-2026
- **Goal**: Retest V10 with targeted bigrams and selective groupby deviations while preserving the V3 344-feature baseline
- **Experiments**:
  - V10 Modified: CPU LightGBM 5-fold CV with per-fold competition + original-data concatenation
  - Added missing targeted bigrams: `ECL_bin × RangeAnxiety` and `ECL_bin × Subsidy`
  - Added 12 selective groupby deviation features using ECL_bin and income_bin groups
  - Kept the feature expansion compact: 168 base features plus 195 Triple TE features, 363 total
- **Timing**: 36.5 min total; fold times were 354 s, 374 s, 448 s, 474 s, and 380 s
- **Key Learning**:
  - OOF AUC reached 0.94597 and LB Score reached 0.94636, a new best and +0.00002 over V3
  - The targeted feature set improved substantially over the first V10 attempt, with targeted bigram TE and groupby income deviations among the strongest new signals
  - `grp_income_bin_Annual_Income_USD_dev` and `grp_ECL_bin_Annual_Income_USD_dev` were the strongest new features
- **Status**: 🏆 Best

### 09-09-2026
- **Goal**: Test XGBoost with V3’s complete 344-feature FE pipeline using V2’s proven weak-regularization parameters
- **Experiments**:
  - V9 Baseline: GPU XGBoost 5-fold CV with per-fold competition + original-data concatenation
  - Full V3 feature pipeline: Triple TE, smooth keys, magic flags, original means, and frequency features
  - V2-style parameters with `lr=0.005`, depth 7, weak regularization, and `max_bin=1024`
  - 344 final features compared with V2’s 84-feature representation
- **Timing**: 21.4 min total; fold times were 222 s, 234 s, 234 s, 239 s, and 241 s
- **Key Learning**:
  - OOF AUC reached 0.94584 and LB Score reached 0.94612, improving V2 LB by 0.00043
  - Full FE plus proven XGBoost parameters entered second place overall, only 0.00022 below V3
  - `_ECL_x_Subsidy_cat` and `_ECL_x_Subsidy` dominated fold-1 importance, together accounting for most of the reported signal
- **Status**: ✅ Success

### 09-09-2026
- **Goal**: Test CatBoost Ordered boosting against the V4 Plain CatBoost baseline using the same Triple TE feature space
- **Experiments**:
  - V8 Baseline: GPU CatBoost Ordered 5-fold CV with per-fold competition + original-data concatenation
  - Triple TE with auto, 10, and 100 smoothing on 63 categorical-like columns
  - Native CatBoost categorical features, multi-scale smooth keys, synthetic-artifact flags, and feature selection
  - Ordered boosting with `lr=0.015`, depth 5, `l2_leaf_reg=5.0`, Bernoulli bootstrap, and balanced weighting
- **Timing**: 19.6 min total; fold times were 194 s, 232 s, 224 s, 206 s, and 202 s
- **Key Learning**:
  - OOF AUC reached 0.94583 and LB Score reached 0.94603, improving V4 LB by 0.00013
  - Ordered boosting improved CatBoost over V4 Plain and entered the leaderboard Top 5, but remained 0.00031 below V3
  - `_ECL_x_Subsidy` remained the strongest fold-1 feature at 15.32%, followed by target-encoded subsidy/environment interactions
- **Status**: ✅ Success

### 09-09-2026
- **Goal**: Compare FT-Transformer self-attention against V6 TabM and the tree-based models using the same evidence-based feature selection
- **Experiments**:
  - V7 Baseline: FT-Transformer 5-fold KFold CV with per-fold competition + original-data concatenation
  - Evidence-based selection from V5 LogisticRegression and V3 LightGBM, retaining 79 of 344 raw features
  - Auto-smoothed target encoding only, with 35 numerical and 34 low-cardinality categorical features
  - FT-Transformer with `d_block=128`, 2 blocks, 8 heads, AdamW, cosine annealing, and BCEWithLogitsLoss
- **Timing**: 159.1 min total; fold times were 1630 s, 2569 s, 1005 s, 1890 s, and 2201 s
- **Key Learning**:
  - OOF AUC reached 0.94545 and LB Score reached 0.94568, with a close +0.00023 LB–OOF gap
  - Self-attention did not surpass V6 TabM or the strongest tree models; LB was 0.00038 below V6 and 0.00066 below V3
  - The 79-feature compact representation remained viable, but FT-Transformer training was substantially slower than V6
- **Status**: ✅ Success

### 08-09-2026
- **Goal**: Test TabM with evidence-based feature selection derived from V5 LogisticRegression and V3 LightGBM importance
- **Experiments**:
  - V6 Baseline: TabM 5-fold KFold CV with per-fold competition + original-data concatenation
  - Feature selection from 344 raw features down to 83 selected NN features
  - Per-fold target encoding from 14 source columns, with TabM k=16, d_block=256, and PWL embeddings
  - Retained engineered interactions, flags, smooth keys, original target means, frequency features, and selected TE features
- **Timing**: 43.9 min total; fold times were 395 s, 402 s, 446 s, 424 s, and 466 s
- **Key Learning**:
  - OOF AUC reached 0.94585 and LB Score reached 0.94606, improving V5 LB by 0.00128
  - The evidence-based 83-feature subset preserved strong signal while dropping most digit, categorical, and redundant TE features
  - TabM was competitive with CatBoost and surpassed V4/V5, but remained 0.00028 below the V3 LightGBM best
- **Status**: ✅ Success

### 08-09-2026
- **Goal**: Evaluate a CPU LogisticRegression baseline with standardized Triple Target Encoding features
- **Experiments**:
  - V5 Baseline: LogisticRegression 5-fold KFold CV with per-fold competition + original-data concatenation
  - Triple TE with auto, 10, and 100 smoothing on 63 categorical-like columns
  - Per-fold StandardScaler for numeric-only logistic regression convergence
  - Frequency encoding, multi-scale engineered features, original target means, and feature selection
- **Timing**: 16.0 min total; fold times were 158 s, 166 s, 175 s, 166 s, and 170 s
- **Key Learning**:
  - OOF AUC reached 0.94468 and LB Score reached 0.94478, with a close +0.00010 LB–OOF gap
  - The strongest fold-1 absolute coefficients were `Subsidy_Available_fe`, `Environmental_Concern_Level_org_mean`, and `Environmental_Concern_Level_cat_fe`
  - StandardScaler enabled reliable convergence in 227–249 iterations, but linear modeling did not match the tree-based baselines
- **Status**: ✅ Success

### 08-09-2026
- **Goal**: Compare GPU CatBoost with native categorical handling against the V3 CPU LightGBM baseline
- **Experiments**:
  - V4 Baseline: CatBoost GPU 5-fold CV with per-fold competition + original-data concatenation
  - Triple TE with auto, 10, and 100 smoothing on 63 categorical-like columns
  - Native CatBoost handling for 10 string categorical columns
  - Multi-scale income/commute smooth keys, synthetic-artifact flags, frequency encoding, and feature selection
- **Timing**: 35.0 min total; fold times were 406 s, 434 s, 436 s, 319 s, and 389 s
- **Key Learning**:
  - OOF AUC reached 0.94581 and LB Score reached 0.94590, with a close +0.00009 LB–OOF gap
  - `_ECL_x_Subsidy` was the strongest fold-1 feature at 10.97%, followed by multiple target-encoded subsidy interactions
  - Native CatBoost categorical handling did not surpass V3 LightGBM; LB was 0.00044 lower
- **Status**: ✅ Success

### 08-09-2026
- **Goal**: Test a CPU-only LightGBM baseline with broader Triple Target Encoding and multi-scale smooth keys
- **Experiments**:
  - V3 Baseline: LightGBM 5-fold CV with per-fold competition + original-data concatenation
  - Triple TE with auto, 10, and 100 smoothing on 6 categorical, 53 numeric-as-string, and 4 smooth-key columns
  - Multi-scale income/commute smooth keys and synthetic-artifact flags
  - Frequency encoding, feature selection, and 155 retained base features
- **Timing**: 30.2 min total; fold times were 346 s, 328 s, 356 s, 320 s, and 324 s
- **Key Learning**:
  - OOF AUC reached 0.94601 and LB Score reached 0.94634, improving the prior LB by 0.00065
  - The strongest fold-1 signals were `TE_income100_floor_auto`, `TE__log_Income_cat_auto`, and `TE_income100_floor_10`
  - Multi-scale smoothing and numeric-as-string encodings were valuable; LightGBM trained successfully on CPU with 344 final features
- **Status**: 🏆 Best

### 08-09-2026
- **Goal**: Improve the V1 XGBoost baseline with Triple Target Encoding and broader categorical representations
- **Experiments**:
  - V2 Baseline: XGBoost 5-fold CV with Triple TE on 13 core columns
  - Triple TE smoothing levels: auto, 10, 100
  - True frequency encoding on all columns
  - Label encoding of 6 categorical columns
  - Feature selection: dropped 76 constant columns and 10 perfectly correlated columns
  - Recipe score added using cdeotte's buy_score formula
- **Timing**: 6.0 min total; fold times were 67 s, 68 s, 76 s, 63 s, and 68 s
- **Key Learning**: 
  - OOF AUC improved to 0.94560 and LB Score improved to 0.94569
  - Strongest fold-1 signals were `LE_Subsidy_Available`, `_recipe_score`, and `_ECL_x_Subsidy`
  - Triple TE added a small but real gain over the V1 baseline
  - Final feature count reached 123 total features
- **Status**: 🏆 Best

### 08-09-2026
- **Goal**: XGBoost baseline with GPU acceleration using cuDF
- **Experiments**:
  - V1 Baseline: XGBoost 5-fold CV with cuDF acceleration
  - Feature engineering: digit features, engineered interactions/flags, frequency encoding
  - 43 base + TE columns: 6 categorical, 19 digit, 4 flags, 14 numeric
- **Timing**: 2.1 min total; fold times were 21 s, 20 s, 22 s, 21 s, and 20 s
- **Key Learning**: 
  - GPU-accelerated XGBoost achieved OOF AUC of 0.94535
  - LB Score: 0.94559
  - Class imbalance (82.5%/17.5%) handled with inverse-frequency sample weights
  - 37 constant features dropped, frequency encoding applied to 29 categorical columns
- **Status**: 🏆 Best
