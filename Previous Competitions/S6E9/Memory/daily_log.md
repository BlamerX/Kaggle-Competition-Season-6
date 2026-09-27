# S6E9 Daily Log

> **⚠️ RULES:**
>
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

### 28-09-2026 (V51 ensemble — LB 0.94646 confirmed)

- **Goal**: Spend the freed V51 slot on the one gain research said was still measurable — a combiner over our own saved vectors — and settle combiner and pool by rule rather than preference. (The number's first life, a monotone-constraint test, measured −0.000035 at z −2.64 against V44, was never submitted, and its script was deleted.)
- **Experiments**:
  - `S6E9_V51_EnsembleShootout.py`: 5 combiners (equal-rank, mean-logit, NNLS, L2-logistic stack, greedy climber) × 5 declared pools (elite/plateau/wide at −1e-4/−5e-4/−1.5e-3 from the best single, plus two ρ > 0.999 diversity variants), all **fold-sealed** — weights and leg sets fitted on 9 folds, scored on the 10th. Nothing trained.
  - Winner **stack over 26 legs: sealed OOF 0.946372 = +0.000120 over V44, paired DeLong z +7.34** (SE 1.6e-5) → submitted, **LB 0.94646**, tying V40 as best held with a gap of +0.00009 against V40's +0.00025.
  - Weight-free result recorded alongside: 6-leg elite-diversity equal-rank **0.946325, optimism exactly 0.000000**. Hill climber measured, not skipped: 0.946352 on the same pool (+9.2e-5 over equal-rank there, −2.0e-5 under the stack, optimism +4e-6).
  - Tie-break defect caught and fixed mid-run: a 3-way stack tie was being resolved by dict insertion order; the rule is now simplest combiner → narrowest pool → highest sealed, and it moved the winner from plateau (35 legs) to plateau_div (26).
- **Timing**: 11.6 min local CPU (load 1.1 / rules 0.6 / 25 sealed cells 9.4 / save 0.5); no GPU, no Kaggle session
- **Key Learning**:
  - Best honest CV the project has produced, and the first artifact where our best CV, best LB and a tight gap coincide. About 60% of the gain (+0.000073) survives with zero fitted parameters; the rest is weights we cannot verify — **11 of 26 negative, Σ|w| 2.82** — the same transfer defect V38 shipped and the board paid 0.00000 for.
  - **Diversity by rule beat diversity by hope:** tightening the duplicate merge to ρ > 0.999 *raised* equal-rank (0.946318 → 0.946325) by deleting near-twin legs, and ρ > 0.99995 auto-dropped V45 as V40's bit-identical twin. No curation involved.
  - Stack across four pools spans 3.3e-5 — inside our own resolution — so the declared tie-break, not the score, chose the winner.
  - Verification: 18 independent checks pass (AUC re-derived by a second implementation and by brute-force pair counting; DeLong against a 250-rep bootstrap, ratio 1.05; sealed == plain for every no-fit combiner; the winner's oof and sub reproduced from raw leg files). One residual fragility logged: the stack's output depends on leg-name string order at 5e-9, four orders below anything we act on.
  - Finals recommendation: **slot 1 = V51, slot 2 = V44**, so one slot does not depend on the combiner transferring.
- **Status**: ✅ V51 LB 0.94646 — best CV (0.946372) and tied-best score held | ✅ 18/18 checks pass | ⏭️ Finals selection pending, due 09-30 23:59 UTC

---

### 26-09-2026 (V49)

- **Goal**: Log V49's completed Kaggle run and its LB.
- **Experiments**:
  - V49 RealMLP 8-member on V40's 341-col matrix, published view-G settings, 10-fold house ruler, GPU 107.1 min → OOF 0.945940 / LB 0.94616; all folds best-epoch 2 (2-epoch schedule signature, not truncation)
- **Timing**: 107.1 min GPU (folds 10.3–10.9 each; fold-1 projection 107.5 vs 130-min cap, no time truncation)
- **Key Learning**:
  - Neural family closed fairly: 0.945940 is 2.7e-4 behind V40's 0.946208 on identical features; 10-fold ladder now XGB 0.946252 · LGBM 0.946197 · CatBoost 0.945987 · **RealMLP 0.945940** · TabM 0.94564 · HistGB 0.945482 — six families, one plateau, XGB on top
  - Reproduction delta, second data point: published view-G 0.946139/0.946182 sit 0.000199–0.000242 above V49's 0.945940, like view A's 0.946281 sat 0.000090 above our 2×2 (V43); andrewleal70's falsification (MLP blend weight ≈ 0) predicted this
  - The plan's 5-fold roster was built in-tree the same day: V47 (V49 at N_FOLDS 10→5, last open family at the 5-fold ruler), V48 (V44's composition — the two effects that ever measured positive — at 5 folds, highest-expectation run), V50 (V46's ExtraTrees null re-priced, cheap CPU insurance); all compile clean, carry in-run parent controls, 5-fold house ruler
- **Status**: ❌ V49 below its own leg gate (0.945940 < 0.946150) — neural family closed fairly | ✅ LB 0.94616 logged | ✅ V47/V48/V50 in-tree, ready for Kaggle

---
### 26-09-2026 (5-fold roster)

- **Goal**: Run and log the plan's three 5-fold roster models; confirm LBs.
- **Experiments**:
  - V48 XGB composition (V40 encodings + V31 `base_margin` prior), GPU 19.9 min → OOF 0.946161 / LB 0.94636; passes the 0.94615 leg gate by +0.000011; backbone-only 0.938311 (V31/V44 reproduce to 4 dp)
  - V50 LGBM ExtraTrees, CPU 46.5 min → OOF 0.946085 / LB 0.94635; null confirmed geometry-robust (exactly equals V26's 5-fold a3, 0.946085, to the last digit)
  - V47 RealMLP 8-member, GPU 51.4 min → OOF 0.945866 / LB 0.94608; −0.000284 below the gate; all folds best-epoch 2 (2-epoch schedule's signature, not truncation)
- **Timing**: V48 19.9 min · V50 46.5 min · V47 51.4 min (fold-1 projection 50.9 vs the 130-min cap, no time truncation)
- **Key Learning**:
  - 5-fold roster complete: XGB 0.946161 (V48) · LGBM 0.946085 (V50) · RealMLP 0.945866 (V47) — 10-fold ladder ordering reproduces; ceiling is representation-bound, not learner-bound, at either ruler
  - 5→10 geometry step measured on three models: −0.000091 (V44→V48), −0.000112 (V46→V50), −0.000074 (V49→V47): 5-fold OOFs read ~0.00008–0.00011 *below* their 10-fold counterparts, so the "+0.0001 caveat" in the plan docstrings was the wrong sign
  - Fold 2 is the weak block at 5 folds (0.94493–0.94526 across all three families), not fold 4 — the 10-fold fold-4 deficit is the same phenomenon under a different row partition
  - Ship candidate from the roster: V48 — the only run that cleared its gate
- **Status**: ✅ V48 passes the 5-fold leg gate (0.946161 ≥ 0.94615) | ❌ V50/V47 below the gate — LGBM and RealMLP closed at the 5-fold ruler

---

### 25-09-2026

- **Goal**: After the user asked why recent versions scored worse and to do proper research before the final week (two finals left, closes 30-09): audit V33–V37, mine the field's writeups, scrap the two queued architecture versions, build a defensible finals plan.
- **Experiments**:
  - V33–V37 audit: 4 implementation defects logged — V33 9/10 folds hit the 8,000-round cap (−0.001204 is a floor), V32's `rsm=0.4` inexecutable on catboost 1.2.10 GPU (verdict came from the Bernoulli fallback), V35 ran with sklearn's hidden ~10% ES holdout, V36 moved seed and K together. None changes any ranking
  - Research waves: (1) discussion 742927 = re-uploaded generator script, no model; its label law scores 0.908475 on the original 10k, 0.937691 on the pool, fold-sealed Δ over V30 = −0.000001 → absorbed, closed. (2) najiama's Honest Model Directory = index of others' CV/LB; honest field 0.9460–0.94634. (3) megayak's four-view ensemble read in source: 0.946077–0.946281 per view, weight 0.3/0.3/0.2/0.2 vs equal 0.946337
  - V38/V39 queues scrapped (mis-read slice + below-noise-floor upside); one durable number: 80 of 312 columns carry the model to within 0.00038
  - V38 combinator shoot-out, local 18.6 min: 5 combiners × 2 pools fold-sealed → L2-logit stack, 25 legs, sealed 0.946333 / LB 0.94639; NNLS dead (0.945826); shipped weights 12/25 negative at ρ 0.9946 — the stack's transfer defect is CV-invisible
  - V39 equal-weight rank mean over the elite pool (V30/V31/V34/V36), local 0.6 min → OOF 0.946285 / LB 0.94640: beat V38's stack on the board with 0.000048 less CV; its first run caught its own rule bug (band reference was V38's stack)
  - V40 run on Kaggle GPU: OOF 0.946208 / **LB 0.94646, new best score held** (+0.000037 encoding effect, below the +0.00005 gate)
  - Reissued V41/V42/V43 on Kaggle: pair cells 0.946190/0.94644 (−0.000018 vs V40), density ratio 0.946192/0.94645 (−0.000016), view-A 2×2 0.946191/0.94640 (−0.000017) — three nulls of the same sign; feature axis closed by measurement
  - V44: V40 encodings + V31 prior composed → OOF 0.946252 / LB 0.94638, predicted-if-additive 0.946261 (shortfall 9e-6): both effects real and independent
  - V45: full-data refit on V40, submission-side → OOF bit-identical to V40 (diff exactly 0.0) / LB 0.94643; refit effect measured twice, +2e-5/−3e-5 = neutral
  - V46: V43 + ExtraTrees → 0.946197 / 0.94637, +0.000006 over V43: V26's +0.00010 reattributed to its shallow/wide config
  - V47 (neural leg, fresh file) and V48 (CatBoost CPU, fresh file) written for the fairness roster; V48 died twice on Kaggle API issues, control probe added; V49 canary arithmetic bug fixed
- **Timing**: V38 18.6 min local · V39 0.6 min · V40 46.2 min GPU · V41 63.2 · V42 58.9 · V43 70.7 CPU · V44 43.7 · V45 51.0 · V46 55.8 CPU
- **Key Learning**:
  - Board ranks inverse to honest CV on the 09-25 batch: V40 (worst CV of the trio) → best LB; V41–V43 within 2e-6 in CV drew 5e-5 apart on the board. No LB statement below ~0.0001 is a verdict
  - V44's additivity test (to 9e-6) rescues V40: failed extensions of a mechanism say nothing about the mechanism itself
  - Refit is public-LB-neutral (+2e-5/−3e-5 in two directions); V45's bit-identical OOF (3rd determinism confirmation) is what makes that null mean anything
  - V46 composes out the last tuning lever: a mechanism that clears the gate in one config and vanishes in another is a config effect wearing a mechanism's clothes
  - Family fairness: LightGBM now fair (−5.5e-5); CatBoost/HistGB verdicts still rest on mis-runs
- **Status**: ❌ V41/V42/V43 all fail the gate (feature axis closed by three nulls) | 🏆 V40 new best held score 0.94646 | 🏆 V44 best honest single CV 0.946252 | ⚖️ V45 refit neutral | ❌ V46 ExtraTrees null | ⏭️ V47/V48 in-tree, ready for Kaggle

---

### 24-09-2026

- **Goal**: Mine the field's winning writeups for how the top thinks, then run the V31–V37 single-model roster in house style.
- **Experiments**:
  - Writeup mining over the 548-link index + non-Playground classics; ~200 readable writeups, rest marked NOT FOUND
  - V31 additive `base_margin` → OOF 0.946224 / LB 0.94630 — best honest CV of the day
  - V32 CatBoost on valid params (`border_count` + published recipe) → 0.945987 / 0.94609
  - V33 zero-TE + numerics-as-codes, `max_bin 8192` → 0.944967 / 0.94481 (floor: 9/10 folds at the cap)
  - V34 LightGBM published config, bagging live → 0.946156 / 0.94638 = XGBoost parity
  - V35 HistGB first run → 0.945482 / 0.94572 (pessimistic: hidden 10% ES holdout)
  - V36 `KFold(20, seed=7)` → 0.946199 / 0.94637 (+0.000028, 2.1× cost)
  - V37 one-hot + fold-safe WoE logistic → 0.943220 / 0.94335
- **Timing**: V31 37.8 + V32 37.5 GPU; V33 94.7 · V34 68.1 · V35 116.7 · V36 95.3 · V37 5.0 min
- **Key Learning**:
  - Original 10k's law is purely additive (depth-1 0.90676 beats all deeper); pool is more predictable than the real law, so all margin over 0.907 is generator artefact; fresh GBM on V30's residual R² = −0.000085, AUC-optimal blend weight 0.000 — feature search finished
  - Config audit found three outcome-deciding defects: V24's `max_bin` for CatBoost (ignored), LightGBM `subsample` without `subsample_freq` (bagging never ran), S6's biggest win was someone's float64/float32 bug — verify a config does what it says
  - V31 overturned a negative 200k-row proxy (−0.0020): a proxy tests implementation, never the idea
  - Batch verdict: one model in seven beat V30, by 0.000053 (V31); plateau 0.9462 ± 0.0001 for anything tree-shaped — the ceiling is representation-bound, not learner-bound
  - V33's −0.001204 (TE deletion) prices target encoding at six times any other axis we measured
  - Leaked-generator-law claim (thread 742927) verified in 3 lines and absorbed: 0.908475 original / 0.937691 pool / Δ over V30 sealed −0.000001
- **Status**: ✅ V31 best honest OOF 0.946224 | ✅ V34 LGBM parity | ⚠️ V36 fold lever spent | ❌ V32/V35/V37 families closed | ❌ V33 no-TE −0.001204 prices TE

---

### 21-09-2026

- **Goal**: Answer why recent versions scored worse than old ones and whether we can go back to a single FE+training model; build that model.
- **Experiments**:
  - Audited all 29 saved OOF/sub pairs against logged LB: every OOF recomputed to <1e-6; measured rank divergence to our own consensus test ranking
  - V30 plain single model: V19 FE verified byte-identical by hash, V22 config, **10 folds instead of 5**
  - Fixed my V29 save-block bug (`sigmoid` applied to probabilities — values squashed, ranks/AUC/LB unaffected)
- **Timing**: V30 45.3 min GPU (10 folds cost 4.2× a 5-fold run, not 2×); audit ~1 min local
- **Key Learning**:
  - Board adds spread, doesn't resolve it: OOF span 0.000107 vs LB span 0.000130, Spearman(OOF, LB) = −0.619; what predicts the board is **divergence from our own consensus ranking, Spearman −0.770**
  - V30: OOF +0.000097 (two-thirds of the external +0.00015 claim), LB −0.00002; LB−OOF gap collapsed to +0.00022
  - Decision: ablation-harness era over for shipping — a version is either a plain single model or never a submission; V23 stays best score, V30 is best model
  - Fold count is spent (20 folds ≈ +0.000028 at 2.1× cost)
- **Status**: ✅ V30 = best model, board-neutral score; divergence rule now the standing decision instrument

---

### 21-09-2026

- **Goal**: Settle two open questions — is `max_bin` the +0.002 lever, and can a non-tree family stand near the incumbent on our matrix?
- **Experiments**:
  - V28 `max_bin` 1024 → 16384 unconstrained, in-run V22 control, trees-level integrity guard
  - V29 TabM dual view: XGBoost control + nested top-120-column TabM, one cross-fitted blend weight
  - Reproducibility audit across the three saved copies of V22's estimator
- **Timing**: V28 66.5 min (4.2× the control); V29 32.4 min (TabM 12.7 of it)
- **Key Learning**:
  - Resolution axis closed by construction: 16× bins moved OOF −0.000002 (z=−0.40); guard proved the cap took effect (income thresholds 6,456 → 7,972) — a real null, not a no-op
  - Nets do not exploit this matrix better than trees: TabM 0.94564 (−0.00043), ranking 0.9951–0.9964 with the GBM — weak and correlated, the worst blend quadrant
  - Pipeline is deterministic across Kaggle sessions (V28/V29 controls rank identically); a ~0.2% rank perturbation is board-invisible
  - Two self-inflicted reporting bugs (class-body `RANDOM_SEED` reference, `paired_z` 4-vs-3 unpack) held by guards; outputs survived both
- **Status**: ⚠️ Two axes closed by measurement; plan rests on the mechanical fold-count gain (V30)

---

### 21-09-2026

- **Goal**: Decide whether the 4–5 arm × 5 fold × 2 stage harness is still worth it, and whether from-scratch features can add anything; run the one survivor.
- **Experiments**:
  - ~700 min of compute since V22 bought a net −0.00003 LB: the harness chased +0.00002 effects that were noise
  - Offline screens vs `oof_v23`: exact row identity, `id` order, 91 pairwise lifts, 201 arithmetic derivatives, oracle cell-recalibration bound — all nil
  - Blend screen over 26 saved OOFs with cross-fitted weights
  - V28 `max_bin` 4096 (30.6 min) then 16384 (66.5 min) with in-run V22 control
- **Timing**: V28 two runs; local screens ~25 min, zero GPU
- **Key Learning**:
  - Every from-scratch feature channel is dead: single-column lifts +0.000000, all 91 pairwise +0.000000, best derivative +0.000002; the "known distribution" family closed with an oracle bound (shrinking logits to true cell centres costs −0.0003)
  - The community's +0.00201 bin-resolution claim was one lever reported twice, and digits were already banked since V19; V28's real null confirms it
  - Two structural gains emerged: 10-fold over 5-fold = +0.00015, one decorrelated view = +0.000135 (z=9.1) cross-fitted — both structural, not representational
  - Non-linear combiners paid CV +0.0005 → LB −0.0004 (≈8σ) last episode: any blend here is one cross-fitted linear weight
  - Gate revised from per-version to cumulative: accumulate orthogonal z>3 gains, submit once
- **Status**: ⚠️ Resolution axis closed; 700 min of micro-ablation retired; 10-fold + decorrelated view queued as V29/V30

---

### 20-09-2026

- **Goal**: Run the two queued structural probes (V26 ExtraTrees on CPU, V27 original-row weight + pairwise loss on GPU) in parallel, then answer why recent submissions keep landing below V23.
- **Experiments**:
  - V26 (CPU 387.6 min): a0 V19 control, a1 +`extra_trees`, a2 + lookup capacity, a3 + wide/weak/many → a1 +0.00007 z=3.58, a3 +0.00010 z=5.22 (rs=42) / +0.00008 z=4.11 (rs=7): first arms ever to clear z>3 on both splits
  - V27 (GPU 69.6 min): a1 drop original 10k = +0.00002 z=2.66, a2 up-weight ×10 = −0.00003, a3 `rank:pairwise` = −0.000329 z=−42.56, a4 pairwise + 10× = −0.00069
  - Cross-version analysis on v19–v27 OOF/sub files
- **Timing**: V26 387.6 CPU + V27 69.6 GPU, concurrent
- **Key Learning**:
  - We are measuring board noise, not regressions: OOF spread 0.000161 vs LB spread 0.00026, Spearman 0.071; residual sd 0.000066 ≈ one third of our progress unit
  - Three pairs with identical OOF differ on LB purely through test reordering (v20/v23, v22/v27, v26/v27) — V23's +0.00002 refit gain is exactly one noise unit
  - ExtraTrees closed the LightGBM→XGBoost gap: it is a split-bias gap, not an information gap; random thresholds work as shallow-model regularisation, not memorisation
  - Original 10k rows do not earn their concat; pairwise-AUC theory is dead
  - Decision: judge versions on OOF only; next experiment = ExtraTrees-direction tuning
- **Status**: ✅ First gate-cleared gain since V19 + ⚠️ noise-floor finding that reframes every recent version

---
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

---

### 19-09-2026

- **Goal**: Buy score on the inference side — give test a model that saw 100% of the labels.
- **Experiments**:
  - V23: V20's CV path untouched; `[3b]` refits one model on all 678,665 rows at mean best iteration (4112), no early stopping
  - `test_probs = 0.50 fold-average + 0.50 refit`; OOF left purely out-of-fold
- **Timing**: 23.4 min total (refit 2.2 min = 8% of the run)
- **Key Learning**:
  - LB 0.94641, new best (+0.00002 over V20/V19), OOF unchanged by construction: first gain taken from inference, not features
  - Refit barely moves the ranking (mean |rank diff| 1,806/286,571) — the remaining inference-side variance lives in the small disagreement
- **Status**: 🏆 Best

---

### 19-09-2026

- **Goal**: Tune the estimator on our own matrix — V20 inherited parameters tuned on someone else's feature set.
- **Experiments**:
  - V22: two-stage search, 6 configs sharing one fold-matrix build; Stage A KFold(rs=42), top-3 on KFold(rs=7)
  - c1 = depth 3 / 8 leaves / gamma 1.0 / colsample 0.85
- **Timing**: 116.8 min total
- **Key Learning**:
  - LB 0.94640 / OOF 0.94608 — new best single-model OOF, +0.00002 over the in-run control on both splits (control reproduced V20 exactly, validating the harness)
  - This matrix wants wide, weak, many learners: deeper/heavier reg and tiny-lr arms lose (−0.00013 / −0.00060)
- **Status**: ✅ Good

---

### 19-09-2026

- **Goal**: Reallocate V20's feature budget from dead weight into explicit conditional crosses.
- **Experiments**:
  - V21: V20 model unchanged; +6 cross keys (each Triple-TE'd + target-free lift), removed digit block + 4 anomaly flags
- **Timing**: 16.4 min total — fastest full run at the time
- **Key Learning**:
  - OOF 0.94597 / LB 0.94629: REJECTED, −0.00009 vs V20 at z=−5.01
  - The addition was right, the subtraction wrong: removing the space narrowed let `_ECL_x_Subsidy` seize 5.4× its share
  - Lesson: one change per version; a shallow lossguide model needs a wide candidate pool
- **Status**: ❌ Failed

---

### 19-09-2026

- **Goal**: Isolate the model family by running V19's exact matrix through the reference XGBoost depth-4 recipe.
- **Experiments**:
  - V20: GPU XGBoost depth 4 lossguide (V19's exact 378-feature matrix, byte-identical, same KFold(5, rs=42))
  - Research pass: Simpson-paradox thread, original-data logistic thread, replication-aware Newton boosting, digit/TE ablation
- **Timing**: 20.6 min total — 40% faster than V19
- **Key Learning**:
  - OOF 0.94606 vs V19 0.94599: Δ +0.00007 at z=+4.76 — first statistically significant single-model gain; LB held at exactly 0.94639
  - Public LB cannot resolve +0.00007: OOF significance and LB movement are now decoupled
  - `TE_lift_trigram_cat_auto` is #1 under depth-4 XGBoost (0.1449): shallow models need the pre-computed artifact crosses more
- **Status**: ✅ Good

---

### 19-09-2026

- **Goal**: Convert the verified generator-artifact findings into V19 on the proven V10 LightGBM pipeline.
- **Experiments**:
  - V19: CPU LightGBM 5-fold, original-data concat, Triple TE, V10 params
  - Fixed the digit-extraction bug (`np.rint(col*1e4)` divmods) present in all 18 prior versions
  - +12 target-free generator-lift/novelty features; +2 structural flags (buy bound 41,667, exact dead zone)
- **Timing**: 33.7 min total
- **Key Learning**:
  - OOF 0.94599 / LB 0.94639 — new best, +0.00003 over V10, first real gain since V10
  - `TE_lift_trigram_cat_auto` at fold-1 rank #4: the frequency-shaping artefact now carries signal the model could not reach before
- **Status**: 🏆 Best

---

### 11-09-2026

- **Goal**: Test RealMLP with the V14 full feature pipeline and short three-epoch training.
- **Experiments**:
  - V17: GPU RealMLP 5-fold, PBLD embeddings, 8-model ensemble, EMA + label smoothing
  - 159 engineered features → 293 model inputs (124 categorical, 169 numerical)
- **Timing**: 97.3 min total
- **Key Learning**:
  - OOF 0.94582 / LB 0.94612, tying V9; RealMLP stayed below V14's 0.94630 despite the full pipeline
  - Substantially slower than the tree baselines — no rerun justified
- **Status**: ✅ Good

---

### 11-09-2026

- **Goal**: Test five forensic-targeted features on the V14 depth-3 XGBoost baseline without pseudo-labels.
- **Experiments**:
  - V16: GPU XGBoost depth 3, 5-fold, V14's pipeline
  - Added subsidy × home-charging bigram, recipe × subsidy/income interaction, ECL-specific buyer-centroid distances (per-fold, leak-free)
  - 164 final features; 2 of 5 forensic signals survived selection
- **Timing**: 28.6 min total
- **Key Learning**:
  - OOF 0.94595 / LB 0.94625, below V14's 0.94630; forensic features earned small fold-1 importance, led by `TE_bigram_Sub_HomeCharging`
- **Status**: ✅ Good

---

### 10-09-2026

- **Goal**: Test exact-match lookup plus CPU KDTree KNN as a non-tree baseline.
- **Experiments**:
  - V15: 5-fold CPU KNN, k=10 KDTree fallback + exact-match lookup
  - 24 encoded features for KDTree; key overlap and uniqueness checked before modeling
- **Timing**: 19.5 min total
- **Key Learning**:
  - OOF 0.91359 / LB 0.91256, a −0.00103 LB–OOF gap (overfit/underfit)
  - Exact matches 0% val/test, every train key unique: no deterministic lookup shortcut in the data
  - KNN far below the engineered tree/neural baselines
- **Status**: ❌ Failed

---

### 09-09-2026

- **Goal**: Test high-confidence pseudo-labeling with the V10 teacher and the proven V12 depth-3 XGBoost student.
- **Experiments**:
  - V14: GPU XGBoost depth 3, 5-fold, original-data concat
  - Pseudo-labels `p>=0.98` / `p<=0.02` → 150,859 half-weight rows (999 buyers, 149,860 non-buyers)
- **Timing**: 35.4 min total
- **Key Learning**:
  - OOF 0.94608 / LB 0.94630, third overall; pseudo-labeling improved V12's LB by only 0.00001
  - `_ECL_x_Subsidy`, `TE_trigram_Sub_ECL_RA`, `_ev_recipe` dominated fold-1 importance
- **Status**: ✅ Good

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---
