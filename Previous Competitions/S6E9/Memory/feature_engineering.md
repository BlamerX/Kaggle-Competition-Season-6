# S6E9 Feature Engineering Notes

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first), with the format block first and feature rows written below it; entries order by version number where that does not conflict with same-day batch run order
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format

The format block stays first; feature entries are written below it.

| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|

---

## Version 51 — Confirmed LB 0.94646 (2026-09-28) ✅ Best honest CV in project history (0.946372); the input is saved predictions, and the new content is the pool rule

No features and no training: five combiners over the archive's saved vectors, all fold-sealed. The engineering content is what decides *which* legs go in.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Declared pool bands** | legs with solo OOF ≥ best *single* − 0.00010 (elite) / − 0.00050 (plateau) / − 0.00150 (wide); ensembles and lineage twins (V18/V23/V29, V38/V39) excluded from the reference and the pool | 12 / 36 / 42 legs before merging; best single V44 0.946252 | Pool width and combiner sophistication trade off, so both are declared rather than picked — the wide pool's equal-rank (0.946270) is *below* the elite pool's (0.946318) | ✅ Used |
| **ρ > 0.999 diversity merge** | collapse legs whose OOF ranks correlate above 0.999, keep the stronger solo | elite 12 → **6 legs** (34/36/40/43/44/46); equal-rank **rose** 0.946318 → 0.946325 | **Diversity filtering by rule helps instead of hurting**: dropping near-twin legs removes duplicate votes, not signal | ✅ Used |
| **ρ > 0.99995 lineage merge** | same rule at the V38 threshold | V45 auto-dropped as V40's bit-identical twin (12 → 11 legs) | The rule catches duplicates without anyone curating the pool | ✅ Used |
| **Combiner comparison, fold-sealed** | weights and leg sets fitted on 9 folds, scored on the 10th; tie band 2e-05 → simplest combiner, then narrowest pool | stack 0.946352 / 0.946372 / 0.946382 / 0.946385 across pools; greedy 0.946352; equal-rank 0.946325; NNLS dead at 0.945826 | Stack spread over four pools is 3.3e-5 — **inside our own resolution**, so the tie-break rule chose the winner, not the score | ✅ Used |
| **Winner: L2-logistic stack (C 0.3), 26 legs** | mean of fitted logits, weights from `LogisticRegression` on leg logits | sealed **0.946372**, **+0.000120 over V44 at paired DeLong z +7.34** (SE 1.6e-5); LB 0.94646 | Biggest honest CV gain we have measured. Gap +0.00009, vs V40's +0.00025 for the same score | ✅ Used |
| **Weight-transfer defect, still present** | negative coefficients on legs that agree out-of-sample | **11 of 26 weights negative, Σ\|w\| 2.82**, V44 = 0.352, mean pairwise leg ρ 0.99405, optimism +2.5e-5 | Fold-sealing cannot see this — it is exactly the mechanism behind V38's +0.000162 CV → 0.00000 LB. The hedge is the weight-free 6-leg equal-rank at **0.946325, optimism exactly 0** | ⚠️ Note |
| **Hill climber (forward-stepwise, 5 steps, top-12 candidates, `_div` pools only)** | leg *set* selected on the training folds, equal-weighted | plateau_div **0.946352** (+4e-6 optimism), elite_div 0.946326 | Beats equal-rank on the same pool by +9.2e-5, loses to the stack by 2.0e-5, ties the elite stack — measured, not assumed; V18's in-sample version and V25's promoted tie are the cautionary cases | 🔬 Research (measured) |
| Cost | local CPU, nothing trained, no Kaggle session | **11.6 min** | Repricing the pool after every new model is the cheapest available CV gain and it is now the best CV we hold | ✅ Used |

## Version 50 — Confirmed LB 0.94635 (2026-09-26) ❌ 5-fold geometry roster completed: XGB passes, LGBM and RealMLP fail the 0.94615 leg gate

No new columns: the plan's three 5-fold roster models (V47/V48/V50) all run on V40's exact 341-column matrix (114 base + 216 triple TE of 72 keys + 11 window cols), so the entry records the ruler's outcome, not a feature table.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **V48: V44's composition at 5 folds** (V40's 341-col encodings + V31's 31-col additive `base_margin` prior, inner cv 10→5 to follow the outer geometry) | 1 XGBoost per fold, no ensemble, no refit; gate ≥ 0.94615 | OOF **0.946161 = passes the gate by +0.000011**; geometry step **−0.000091 vs V44's 10-fold 0.946252**; backbone-only OOF reproduces V31/V44's 0.938310 to four decimals (0.938311), trees add +0.007850; BestIter 3,935–5,407/8,000, nothing truncated | **The plan's highest-expectation run is its best model, and the only one that cleared its gate** — if a single model ships from the 5-fold roster it is V48. The trigram-lift mega-feature is absent from fold-1's top 15 (led instead by `TE_income_exact_int_10` 425.9): the prior is doing its job at 5 folds too | ✅ Used (roster winner) |
| **V50: V46's ExtraTrees LightGBM at 5 folds** | same 341-col matrix, `extra_trees=True` + `split_histogram_sampling=True` (printed before fold 1) | OOF **0.946085 = −0.000112 vs V46's 10-fold 0.946197**, below the gate by 0.000065; **lands exactly on V26's 5-fold winner a3 (0.946085, to the last digit)** | The ExtraTrees null is **geometry-robust**: V26's +0.00010 was the shallow/wide config it rode in on, not the random thresholds, and V43 already had the config. LightGBM stays ~5.5e-5 under XGBoost at the 5-fold ruler too | ❌ Removed (null confirmed) |
| **V47: V49's RealMLP 8-member at 5 folds** | PBLD + entity embeddings, 3×256 SiLU, 2 epochs, 4,967,793 params (87 categorical + 254 numeric = 341) | OOF **0.945866 = −0.000284 below the gate**; 5→10 geometry step **−0.000074 vs V49's 0.945940** — smallest in the roster | The neural family is closed at the 5-fold ruler too, 2.95e-4 behind the matrix's XGBoost. The 2-epoch net is less sensitive to train-row fraction than the boosters; the open test (3 epochs at 5 folds) has a negative prior (V17's 3-epoch run gained nothing) | ❌ Removed (gate failed) |
| **The 5→10 geometry step, now measured on three models** | V44→V48 −0.000091 · V46→V50 −0.000112 · V49→V47 −0.000074 | All in the **−0.000074…−0.000112 band**: the 5-fold OOF of a 10-fold model reads ~0.00008–0.00011 *below*, not +0.0001 above, the 10-fold OOF | The stale "+0.0001 geometry caveat" printed in the V47/V48/V50 docstrings was the wrong sign; the measurements correct it. Neither ruler is "better" — they are different rulers, and the 5-fold models train on 80% of the labels, not 90% | 🔬 Research (verdict) |
| **Fold 2 is the 5-fold weak block, not fold 4** | per-fold low points: V48 0.94526 · V50 0.94512 · V47 0.94493 | All three families show fold 2 as the low point at 5 folds, where all six model classes showed fold 4 at 10 | The split's weakest block is always the weakest fold, and no model class fixes it (V44's prior improved fold 4 at 10 folds: 0.94493 vs 0.94488 — the only mechanism that ever moved a weak block) | 🔬 Research |
| **5-fold ladder, complete** | V48 XGB **0.946161** · V50 LGBM **0.946085** · V47 RealMLP **0.945866** | Ordering matches the 10-fold ladder (XGB 0.946252 · LGBM 0.946197 · CatBoost 0.945987 · RealMLP 0.945940 · TabM 0.94564 · HistGB 0.945482); the net's deficit from the top is *larger* at 5 folds | **The ceiling is representation-bound, not learner-bound, at either ruler** — measured now on both rulers | 🔬 Research (verdict) |
| **Board behaviour** | V48 **0.94636** · V50 **0.94635** · V47 **0.94608** | Gaps +0.00021 / +0.00047 / +0.00021: the ladder order held on the board in reverse (weakest OOF, lowest LB), and V50's +0.00047 is the batch's widest — the 5-fold geometry's known cost | No new divergence-rule data: all three are inside the band the 5-fold models always draw | ⚖️ No Improvement |

| **V48 ship candidate (5-fold roster winner)** | the only roster run to clear its gate, and the composition of the only two effects that ever measured positive | if one single model ships from the 5-fold roster it is V48; V45's refit mechanic, if used on the final, belongs on V44 (10-fold best single), not on V40 | see training_logs V48 | ✅ Used (decision) |

---

## Version 49 — Confirmed LB 0.94616 (2026-09-26) ❌ The neural family closed fairly: −0.000268 vs the matrix's XGBoost, below the 0.94615 leg gate

No new columns: V40's exact 341-column matrix (114 base + 216 triple TE + 11 window) under a different learner. The entry records the learner and its gate outcome, not a feature table.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Published view-G RealMLP recipe on our matrix** | PBLD periodic-bias numeric embeddings (hidden 20, out 5, freq 5.0, PReLU), entity embeddings for the 87 categoricals with one-hot below 4 uniques, 3×256 SiLU 8-member net, EMA 0.997875, label smoothing 0.04 on a cosine schedule, AdamW lr 0.01 / wd 0.013, **2 epochs**, batch 256; NaNs in the 543 original rows zero-filled after scaling (a net has no native NaN path, unlike trees) | OOF **0.945940** vs V40's 0.946208 on the identical 341 columns = **−0.000268**; fold mean 0.946038 ± 0.000787, all ten folds best-epoch 2 (the schedule's signature, not truncation: fold-1 projection 107.5 min vs the 130-min cap) | V29's TabM 0.94564 was a mis-framed run (nested 120-column subset, our params). V49 is the fair version — the published neural recipe on our own matrix — and it lands 2.7e-4 behind the matrix's XGBoost | ❌ Removed (family closed fairly) |
| **The leg gate the script carried** | declared in the docstring: solo OOF ≥ 0.94615 to be an ensemble leg, ≥ 0.946252 to challenge the best single | 0.945940 sits **0.000210 below the 0.94615 gate** | It joins the pool as a body at ρ ≈ 0.99x with the tree legs, not as a contribution. andrewleal70's falsification predicted this exactly: the MLP is the one genuinely uncorrelated build (ρ 0.975 vs trees) and its optimal blend weight measured 0.03 ≈ zero — low correlation only pays if the model also catches signal the trees miss, and it does not | ❌ Removed (gate failed) |
| **Reproduction delta vs the published view-G numbers** | published CV 0.946139 (single-seed) / 0.946182 (3 seeds) on the same episode | our run: **−0.000199 / −0.000242** | Second "leader margin is not in the recipe" data point: view A's 0.946281 sat ≈0.000090 above our 2×2 (V43), and view G's now sits 2–3× that above our fair neural number. The field's *published* frontier keeps landing above our reproducible frontier; our honest numbers (V44 single 0.946252, sealed 0.946333–0.946373) match cdeotte's measured frontier (single 0.94627, ensemble 0.94639) — we are at it, not behind it | 🔬 Research (closed) |
| **Fold 4** | 0.94469 | **Seventh model class on the same weak row block** (XGB, LGBM, CatBoost, TabM, HistGB, RealMLP, and V37's plain logistic) | Only V44's additive prior has ever improved fold 4 (0.94493 vs V30's 0.94488 on the same rows); the deficit is a *fitting* deficit on that block, not irreducible data hardness | 🔬 Research |
| **Ladder at 10 folds, now complete** | XGB 0.946252 (V44) · LGBM 0.946197 (V46) · CatBoost 0.945987 (V32) · **RealMLP 0.945940 (V49, fair)** · TabM 0.94564 (V29) · HistGB 0.945482 (V35) | Six families, one plateau | Every learner we have ever run sits within 3.1e-4 of the top, and the top is XGBoost. **The ceiling is representation-bound, not learner-bound, and now measured fairly across all six families.** | 🔬 Research (verdict) |
| **Cost** | 107.1 min GPU, folds 10.3–10.9 min each | ~2× the 50 min the docstring estimated | An 8-member × 5,053,473-parameter net on 341 columns at 2 epochs is a fair price for the last open family question; no further net runs are justified at this matrix | ✅ Used (tooling) |

## Version 46 — Confirmed LB 0.94637 (2026-09-25) ❌ ExtraTrees splits are a null: +0.000006 over V43, and V26's +0.00010 gets reattributed

No new columns. V43's exact 341-column LightGBM with `extra_trees=True` and `split_histogram_sampling=True`, so the delta is the split mechanism alone.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **ExtraTrees threshold randomisation on our best feature block** | every split threshold drawn uniformly inside its histogram bin instead of taking the bin's loss-minimising point; nothing else changed | **+0.000006 over V43** (0.946197 vs 0.946191); BestIter 1,606–3,138 of 20,000, nothing truncated | The only two-split-reproduced tuning gain we ever measured (V26: +0.00010 at z=5.22 on rs=42, +0.00008 at z=4.11 on rs=7) **does not compose with the encodings** — an eighth of the noise floor | ❌ Removed (not adopted) |
| **What V26's gain actually was** | V26 ran extra_trees *together with* the shallow/wide direction (depth 3, 8 leaves, min_child 50, colsample 0.85, lr 0.01, 6-8k trees); V43 already had that shape | +0.00010 with both → +0.000006 with the mechanism alone | **A mechanism that clears the gate inside one configuration and vanishes inside another is a configuration effect wearing a mechanism's clothes.** This is the cleanest disproof procedure we have for a tuning claim: compose it, don't replicate it | 🔬 Research (closed) |
| **Split-mechanism guard** | `assert` both parameters in the dict + a printed line before fold 1 | printed `extra_trees=True split_histogram_sampling=True` | LightGBM silently ignores unknown/vetoed parameters; without this, a no-op run would have been logged as a real null | ✅ Used (tooling) |
| **Importance profile under randomised thresholds** | fold-1 gain, LightGBM units | `TE_lift_trigram_cat_200` **621,657** with 15/15 top slots TE or interaction; `_ECL_x_Subsidy` 557,018, `_ev_recipe` 195,770 | Seventh learner, same owner: the generator-artifact cross. Randomised thresholds redistribute gain *inside* the interaction family, they do not create a new one | ⚠️ No Improvement |

## Version 45 — Confirmed LB 0.94643 (2026-09-25) ⚖️ The full-data refit at inference: public effect measured twice, +2e-5 and −3e-5, i.e. neutral

No new features and no new columns: V40's 341-column matrix, the same 341 in the refit (guarded, printed, matched), plus one extra booster trained on 100% of the labelled rows.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Full-data refit blended at inference** | one booster on all 668,665 comp + 10,000 original rows, `num_boost_round = round(mean(best_iters)) = 5,224`, no early stopping, TE refit on that frame, windows re-cross-fitted inside it; `sub = 0.50 fold-average + 0.50 refit` | OOF **bit-identical to V40 (max diff exactly 0.0)**, LB **0.94643 vs V40's 0.94646 → −0.00003** | With V23's **+0.00002** this is two opposite-signed draws on the same mechanic: **the refit is invisible on the public slice.** Variance reduction on the private 80% is a theory argument, and this board cannot test it | ⚖️ No Improvement (measured neutral) |
| **Refit-vs-fold-average agreement** | Spearman between the refit's test vector and the 10-fold average's | **0.999708**; mean \|rank shift\| **1,342.5** of 286,571 for the refit alone, **666.6 (0.23%)** for the shipped 50/50 blend | Quantifies what the whole operation changes: a quarter of one percent of test order. Useful calibration for the divergence rule — **0.23% of ranks ≈ 3e-5 of LB** | 🔬 Research |
| **Column-set guard on the refit path** | assert the refit matrix carries the same 341 names as the CV matrices (the one way this version could be silently wrong: a mismatch makes it a *different* model, not a *bigger* one) | matched, 341 = 341, no warning printed | A refit is only a variance-reduction argument if it is the same model trained on more rows | ✅ Used (audit) |
| **Determinism, third confirmation** | V45 vs V40 fold AUCs and BestIters | **all ten folds identical to the logged digit** (0.94637/5112 … 0.94614/6108) | Third independent proof this pipeline is bit-deterministic across Kaggle sessions (after V28's and V29's controls). **It is what licenses reading −0.00003 as the refit alone** rather than as run-to-run noise | ✅ Used (method) |
| **Cost of the mechanic** | refit stage vs total | **4.2 min of 51.0 = 8%** | Cheap enough to fold into a final without spending a version on it | ✅ Used |

## Version 44 — Confirmed LB 0.94638 (2026-09-25) 🏆 Best honest single-model CV we have built (0.946252); the composition of the only two effects that ever measured positive

No new columns: V40's 341-column matrix byte-for-byte, plus **V31's additive backbone fed in as `base_margin`**. The engineering content is the demonstration that a prior and a representation are different axes.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Additive `base_margin` prior on V40's matrix** | LogisticRegression (C 0.5, lbfgs, max_iter 3000) on 31 additive cols — the 7 numerics + `_log_*` terms + `income_exact_int/100/1000` + `commute_integer` + one-hot of the 6 categoricals, standardised, clipped to [1e-6, 1−1e-6], logit-transformed; per-fold fit on fold-train rows, cross-fitted on an inner `KFold(5,42)` so training rows get out-of-fold margins | **+0.000044 over V40**, +0.000028 over V31, +0.000081 over V30; misses our +0.00005 gate by 6e-6 | **The two effects add to within 9e-6** (predicted 0.946261 vs measured 0.946252), which proves both are real and independent — the one thing no amount of solo-CV measurement could establish | ✅ Used |
| **Where the trees' work went** | fold-1 gain profile vs V40's | V40: `TE_lift_trigram_cat_100` **9,642.9**, 5.2× the #2. V44: max **251.0** (`TE_income_exact_int_10`), flat, with `Range_Anxiety_Level_fe` 227.3 and `Daily_Commute_km_org_mean_cat_fe` 146.3 in the top 15 | **A working prior dissolves the mega-feature.** Given the additive law for free, the trees stop needing one giant trigram-lift column to reconstruct it and spread the work over raw frequencies — the clearest causal read on our #1 feature we have ever got | 🔬 Research |
| **Rounds consumed by the prior** | BestIter distribution | **3,312–5,709** vs V40's 4,091–6,420; total wall time **fell** to 43.7 min from 46.2 | The prior is doing the arithmetic the first ~1,000 trees used to do. Cheaper *and* better — the signature of a real gain rather than a lucky one | ✅ Used |
| **Fold 4** | the block that has been weakest across six model classes | **0.94493 vs V30's 0.94488 on identical rows** — first improvement on record | Suggests fold 4's deficit was a *fitting* deficit (the additive law being re-derived badly on that row block), not irreducibly hard data | 🔬 Research |
| **Convergence and stability** | 8,000 rounds / ES 400; fold SD | No fold within 1.7× of the cap; **fold SD 0.000779**, tightest of the V41–V44 batch (0.000792 / 0.000794) | The gain is spread across folds rather than carried by one draw | ✅ Used (audit) |
| **Local pre-run audit of the spliced block** | FE replayed with `xgb.train` stubbed; assert one shared column set across all DMatrix objects, `base_margin` present, finite, length-matched, non-constant | Kaggle printed 341 columns and margins in (−13.8, +3.4) as predicted | Third version where this check converted a would-be 45-minute failure into a clean run | ✅ Used (tooling) |

## Version 43 — Confirmed LB 0.94640 (2026-09-25) ❌ The missing 2×2 corner: view-A optimiser on view-A encodings

No new features. V40's exact 341-column block re-fit under megayak's own LightGBM recipe (lr 0.02, 32 leaves, depth 5, min_child 10, bagging 0.8/1, colsample 0.3, α 0.071, λ 2.0, max_bin 1024) with their triple-TE smoothings **auto/20/200 at cv=5** and `StratifiedKFold(10, shuffle, rs=42)`, so the cell is comparable to both parents.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **View-A smoothings (auto/20/200, cv=5)** | replacing our house auto/10/100 at cv=10 on the identical key set | part of a cell worth **−0.000017** vs V40 | Their smoothing choice is not a feature gain; V34 already measured their optimiser alone at −0.000015 vs V30, and here it is −0.000017 with the encodings present — the same number twice is the most convincing form a null can take | ❌ Removed (not adopted) |
| **Encodings × learner, the full 2×2** | V30 (ours/ours) 0.946171 · V40 (theirs/ours) 0.946208 · V34 (theirs/ours, our features) 0.946156 · V43 (theirs/theirs) 0.946191 | encodings **+0.000035**, optimiser **−0.000017**, both **+0.000020** | The effects are close to additive and both tiny; nothing in the pairing rescues the other. **Every cell remains ≈0.000090 below the published 0.946281** | 🔬 Research (closed) |
| **Usage of the new columns on LightGBM** | fold-1 gain split | `win_inc_2` **120,354**, `TE_ladder_inc10_auto` **116,497**, `win_inc_10` 57,699, `win_inc_5` 50,050 | The window/ladder family is *more* heavily used by LightGBM than by XGBoost and the score is *lower*. **Usage is not accuracy** — the cleanest statement this project has about importances | ⚠️ No Improvement |
| **Convergence** | 20,000 rounds, ES 500 | BestIter 1,093–2,318 — no cap touched | Kills the "V34 was undertrained" defence for LightGBM on this matrix | ✅ Used (audit) |
| **LightGBM ceiling** | fold-mean 0.946205 vs V26 0.946085, V34 0.946156, V19 0.94599 | best LightGBM fold-mean we have produced, still under XGBoost's 0.946217 | Family gap is ≈4e-5 on identical features — measured, not asserted | ⚠️ No Improvement |

## Version 42 — Confirmed LB 0.94645 (2026-09-25) ❌ Continuous generator density ratio (pool vs original kNN)

V40's config and block unchanged, plus `log(p_pool / p_original)` over the 7 numerics. 359 fitted columns.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **kNN density ratio, multi-scale** | `log(d_orig/d_pool)` at k ∈ {5,25,100}; pool tree fit on a seeded 150,000-row subsample of train+test, original tree on all 10,000 pool-source rows; 7 numerics standardised on the pool | 21 columns; **none in fold-1's top 15** | −0.000016 vs V40. Unlike V40's windows (which the trees used immediately), the trees ignored these entirely — the model never asked for a continuous synthesis score | ❌ Removed |
| **`dens_bin` through the triple TE** | 20-quantile bin of the k=25 ratio, encoded as a string key at all three smoothings | 6 TE columns; not in the top 15 | Discretising the ratio did not rescue it. The information is already in `freq_*`/`lift_*`, which are exactly count-based density ratios | ❌ Removed |
| **Label-free pool usage** | density is computed without `y`, so all 955,236 pool rows are legitimately queryable (same licence the transductive frequency encoding already uses) | FE stage **2.5 min** total, not the feared hour | The cost argument against kNN features was wrong and is now retired as a reason to skip an idea | ✅ Used (method) |
| **Both kNN estimators, side by side** | kNN label lookup (earlier probe, −0.000004) and kNN density ratio (−0.000016) | two nulls, two different quantities | **The generator's fingerprint is already fully carried by count features.** A neighbourhood estimator of it adds no information the frequencies don't have | ❌ Failed |

## Version 41 — Confirmed LB 0.94644 (2026-09-25) ❌ All 15 categorical pair cells (key + lift + novelty)

V40's config and block unchanged, plus every pairwise categorical cell. 461 fitted columns (341 + 45 pair keys × 3 smoothings + 15 pair lift + 15 pair novelty + filter churn).

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Pair-cell target encoding** | for each of the C(6,2)=15 pairs of categorical originals, `A__B` as a string key through the triple TE. Only `City×Car` and `ECL×Subsidy` had been encoded before | 45 TE columns; **not one in fold-1's top 15** | −0.000018 vs V40. This was the channel V40's success most directly implied — 13 of 15 pairs untouched — and filling it bought nothing | ❌ Removed |
| **Pair-level generator lift** | `freq_pool(A,B) / freq_orig(A,B)`, 0 when the cell is absent from the original | 15 columns; not in the top 15 | The trigram lift still owns the signal (`TE_lift_trigram_cat_100` **11,150**, `_10` 7,622). Pairwise lift is a lower-order projection of the same fingerprint | ❌ Removed |
| **Pair-level novelty flag** | `1 if the pair cell never appears in the original 10k` | 15 columns; not in the top 15 | Novelty is already a single bit on the trigram; duplicating it at pair resolution adds representation, not information | ❌ Removed |
| **Column-count side effects** | pair keys are string features, so the numeric-as-string block grew 95 → 125; the redundancy filter then dropped **194** columns instead of 149 | Runtime stayed at 63.2 min | The filter absorbed the 60 new columns; "more features = slower" is not automatic on this pipeline | 🔬 Research |
| **The inference being punished** | I derived a *gap* from a *diff of feature lists* (their matrix has pair cells, ours didn't ⇒ pair cells are the missing 9e-5) | Cost 63 min to answer "no" | **Absence of a feature is not evidence of a gap.** A missing column matters only if the learner asks for it once it is there; importances never did | ❌ Failed (method) |

## Version 40 — Confirmed LB 0.94646 (2026-09-25) 🏆 New best score; the last feature idea, measured at +0.000037 and closed

V30's GPU XGBoost, house ruler, house TE settings — **only the encoding block is new**. 341 fitted columns = 114 base + 216 TE (72 keys × 3 smoothings) + 11 window rates.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Centred-window target rates, income** | for each row, `Σy + prior·gm / Σcount + prior` over incomes within ±w dollars, w ∈ {2,5,10,25,50,200}, bincount + prefix-sum range scan, `prior=10`, cross-fitted on inner `SKFold(5,42)` of fold-train rows | 6 columns; `win_inc_2` earns **378.5** fold-1 gain (top 15) | Whole model moves **+0.000037** (V40 vs V30, everything else identical) — a legitimate representation that is *used* and worth less than our +0.00005 noise gate. Neighbourhood smoothing of exact-income TE adds what the exact key missed: nothing measurable | ⚠️ No Improvement |
| **Centred-window target rates, commute** | same estimator on `round(km×10)` (tenths of a km), w ∈ {1,3,10} | 3 columns; none in fold-1's top 15 | Commute is already carried by `commute_integer`, its TE and the ECL×RangeAnxiety interaction. A third resolution of the same axis is redundant | ⚠️ No Improvement |
| **Group-restricted windows** | income windows at w ∈ {5,25} computed **inside `City_Type × Current_Car_Type`** groups (factorised over pooled train+test), gm pre-fill for empty groups | 2 columns; not in the top 15 | Distinct from our 3-way joint-cell TE null because it is a *continuous-neighbourhood* rate inside a cell, not a cell mean — and it still adds nothing | ⚠️ No Improvement |
| **Quantisation ladder** | `inc//10, //50, //500, //5000`, `km//5, //50` as string keys through the triple TE, kept **alongside** exact/100/1000 | 18 TE columns; `TE_ladder_inc10_10` **458.4**, `TE_ladder_inc10_auto` 303.4, `TE_ladder_inc50_auto` 246.4 in fold-1's top 15 | The most-used new family, and the reason the aggregate effect isn't zero — yet 3 keys in the top 15 still buy 3.7e-5 overall. Mid-resolution ladders duplicate both the exact key above them and the //100 key below | ⚠️ No Improvement |
| **Design choice: encodings on our model, not their clone** | the first draft cloned view A (CPU LightGBM, their smoothings/cv=5, `StratifiedKFold`); rebuilt as V30's config + house TE so exactly one variable moves | V40 − V30 **is** the encoding effect; V34 already showed their optimiser is a tie (0.946156) | Without this, +0.000037 would have been confounded with learner, smoothing and ruler changes at once. Same lesson as V38's slice error: replicate a rival by changing one thing | ✅ Used (method) |
| **Convergence check** | `NUM_ROUND=8000`, ES 400 | BestIter 4,091–6,420 across 10 folds — no fold truncated | Unlike V33, this figure is a converged measurement, so "+0.000037" is a real floor *and* ceiling, not an undertrained artefact | ✅ Used (audit) |
| **Local pre-run audit of the encoding block** | replay FE with `xgb.train` stubbed; assert 11 window + 18 ladder columns, 341 total, zero NaN/inf, window values ∈ [0,1] | Kaggle reproduced the counts exactly (341 = 312 + 29) | Two earlier versions died on wrong assumptions about the column set; this check costs a local minute and retires that failure class | ✅ Used (tooling) |
| **view A's 0.946281 not reproduced** | their encodings + our best model = 0.946208 | −0.000073 vs the published claim | With V32 (their CatBoost at our params: 0.945987) and V34 (their optimiser: 0.946156), every candidate source of their edge except their own data/ruler has now been tested. Nothing left to copy from them | ❌ Removed (route closed) |
| `TE_lift_trigram_cat_100` | unchanged generator-lift trigram TE | **9,642.9**, 5.2× the #2 feature | Fifth version and second learner where the trigram owns the model; 29 extra encoding columns did not move the hierarchy | ✅ Used |

## Version 39 — Confirmed LB 0.94640 (2026-09-25) ✅ Second-best LB we hold, from a combiner with **zero** fitted degrees of freedom

No features and no fitting: equal-weight rank mean over four saved prediction vectors. The engineering content of this version is the **eligibility rule** and what band width costs.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Elite leg pool by declared rule** | every version whose solo OOF ≥ best *single* model − 0.00010; ensembles excluded from the reference and the pool; lineage duplicates (V18 blend, V23 = V20's OOF, V29 = V22 control) excluded; V38 excluded because it is built from these legs | 4 legs: V30 0.946171 · V31 0.946224 · V34 0.946156 · V36 0.946199; mean pairwise rank ρ **0.99797** (min 0.99728) | "Statistically indistinguishable from our best model" is the narrowest defensible leg-pool definition, and an equal-weight averager needs exactly that — see the dilution row | ✅ Used |
| **Equal-weight rank mean** | `mean(percentile_rank(leg))` — no weights, no selection, nothing fitted | OOF **0.946285** = +0.000114 over V30, +0.000061 over V31; **LB 0.94640** | ~70% of the ensemble effect is reachable with zero fitted freedom, so most of what V38's stack earned was averaging, not meta-learning | ✅ Used |
| Weight-free alternatives | `logit_mean` (sigmoid of mean logit), `prob_mean` (mean probability) | 0.946286 / 0.946286 vs rank mean 0.946285 | All three inside the 0.00002 tie band, so the pre-declared order shipped rank_mean rather than the data deciding — **averaging space is worth 1e-6 and is not a lever** | ✅ Equivalent |
| **Band sensitivity (printed, not selected on)** | same combiner at 5e-05 / **1e-04** / 2e-04 / 5e-04 | 2 legs 0.946304 · 4 **0.946285** · 11 0.946238 · 25 0.946223 | **Dilution is monotone in pool width** (~3e-6 per 1e-4 of widening): the quantitative reason equal weight needs a narrow rule while a stacker tolerates a wide pool — it can down-weight what an averager must include | 🔬 Research (measured) |
| `rank_mean` output scale | shipped values are percentile ranks (0.000023 … 0.999998), not calibrated probabilities | AUC-identical, monotone | Valid for an AUC metric; if a probability-looking file is wanted, `prob_mean` is 1e-6 better and order-equivalent | ⚠️ Note |
| Self-referential rule bug (caught by running it) | the band reference was the archive's best OOF — by then **V38's stack**, made of the legs being selected | elite pool came up **empty**; mean of nothing → NaN | Any eligibility rule over a growing archive must exclude prior ensembles from its own reference or it turns circular. Fixed: reference = best *single* model | ✅ Used (lesson) |
| Cost | local CPU, no training, no Kaggle session | **0.6 min** | Two of the three best scores we now hold cost under a minute of compute — the open question is the marginal value of another trained model, not its marginal cost | ✅ Used |

## Version 38 — Confirmed LB 0.94639 (2026-09-25) ✅ Best honest OOF (0.946333); a new *input class*: saved predictions, not features

No feature engineering at all — this version's inputs are the archive's own prediction vectors. The relevant "feature table" is therefore the leg pool and what the combiner did with it.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| Leg pool by **declared rule**, not hand-picking | 36 paired `oof_v*`/`sub_v*` vectors; plateau = solo OOF ≥ best − 0.0005 (25 legs), wide = ≥ best − 0.0015 (31 legs); lineage duplicates excluded by rule (V23's OOF *is* V20's, V29's saved model is the V22 control, V18 is itself a blend of V1–V17); auto-merge any pair at rank ρ > 0.99995 | Band rule excluded V5/V15/V37; lineage rule excluded V23/V29; **the duplicate merge fired zero times** | The zero-merge result is independent confirmation of the archive finding: once true duplicates are gone, no two of our models are near-identical (mean pairwise ρ 0.9946), so averaging can in principle help — the ceiling is not "all models are the same" | ✅ Used |
| **Fold-sealed combiner protocol** | Fit weights/leg-set on 9 folds of `KFold(10, shuffle, 42)`, apply to the 10th; print the in-sample score of the same arm alongside | Stack sealed **0.946333** / in-sample 0.946354 (optimism +0.000021); greedy 0.946312 / 0.946316; equal weight identical by construction | This is the protocol V18 violated (hill climber fitted directly on training rows: OOF 0.94623, LB 0.94635, no gain) and it is what makes the +0.000162-over-V30 number credible | ✅ Used |
| **L2-logistic stack on logits** (winner) | `LogisticRegression(C=0.3, lbfgs)` over 25 leg logits, refit on all train rows for the test blend | **+0.000162 over V30, +0.000109 over V31** sealed; LB 0.94639 = V30's exact LB | Largest honest gain since the 10-fold ruler, and the board paid nothing for it. Shipped as candidate #1 for the final, with the caveat below | ✅ Used |
| **Weighted-stack transfer defect** (new finding, invisible to CV) | A leg's OOF row comes from **one** fold model; its test row from the **average of ten** — so the meta-learner is fitted on inputs noisier than the ones it will face | Fitted weights: **12/25 negative**, sum \|w\| **2.592**, max \|w\| 0.362, V16 **−0.2789**, V12 −0.1403, V32 −0.1151, V21 −0.1016 at ρ = 0.9946 | The stack is being paid to cancel leg-specific noise that 10-fold averaging has already removed at test time. Fold-sealing cannot detect this — it is not label leakage, it is train/test geometry mismatch | 🔬 Research (unresolved) |
| Shrinkage sweep as the test of that defect | Same arm, C = 1.0 / 0.3 / 0.1 / 0.03 / 0.01, plus non-negative-clipped renormalised variants | sealed 0.946333 / 0.946333 / 0.946334 / 0.946335 / 0.946334; clipped non-negative 0.946301 / 0.946302 | **Our CV cannot adjudicate the defect**: 100× shrinkage moves nothing and removing all negative weights costs only 3e-5. Hence the decision moves to a cheap public experiment (submit the equal-weight variant) instead of a number we can compute locally | ⚠️ Inconclusive |
| Equal-weight averaging | Mean of percentile ranks / mean of logits over the same pools | 25 legs → 0.946223 (≈ V31 alone); 31 legs → 0.946234; the earlier 8-strong-leg set → 0.946259 | **Pool width punishes equal weight specifically** — averaging in legs 0.0003 behind is not neutral, it is a 3.6e-5 cost. Immune to the transfer defect, which makes it the conservative final candidate | ✅ Used (fallback) |
| NNLS logit blend | Non-negative least squares of leg logits against 0/1 labels | **0.945826** on both pools — below every leg except V13/V33 | Wrong objective for a rank metric: least squares to binary labels is not AUC. Closed, do not revisit | ❌ Removed |
| Greedy forward selection | Stepwise leg addition to 6 legs, equal weight within the chosen set, fold-sealed | 0.946312 sealed, optimism **+0.000004**, but **416 s** (plateau) and **510 s** (wide) per rerun | Best-behaved arm and still 2e-5 behind the stack — leg *selection* is not where the gain is. Kept as an audit tool, not a ship candidate | ⚠️ No Improvement |
| Divergence of the shipped test ranking | mean \|rank shift\| across the 286,571 test rows vs our own submissions | vs V30 **0.658%**, vs V31 0.634%, vs V23 0.709%, vs V34 0.801% (Spearman 0.99930 / 0.99946 / 0.99922 / 0.99898) | A low-divergence submission that still returned only 0.94639 — so record the limit of our own predictor: divergence-vs-consensus ranking LB at **−0.770 as a group trend**, and it does **not** license per-submission inference | 🔬 Research |
| Cost profile | Local CPU, no training, no Kaggle session | **18.6 min** for all 10 arms; the winning arm alone is 33 s | The shoot-out is now a standing tool: after every new version, rerun it and judge the version by whether the pool's sealed score rises, not by whether its solo CV beats V30 | ✅ Used |

## Version 37 — Confirmed LB 0.94335 (2026-09-24) ❌ Linear ceiling pinned: one-hot + WoE loses to target encoding

**A deliberately different representation**: no V19 FE pipeline at all. The design is 2,076 sparse CSR columns built per fold, with every label-dependent statistic fitted on that fold's training rows only (composition pool + its own original rows), and `assert X.shape[1] == DESIGN_WIDTH` guarding column alignment across folds.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| Static one-hot lattice (1,120 cols) | 6 originals + `ECL_str` (22 levels) • 4 cell keys `key_ecl_sub`/`key_ecl_ra`/`key_ecl_sub_ra` (55) • exact integer lattices `commute_tenth` 994, `age_int` 45, `cars_int` 4 (1,043). Vocabularies built on **labelled pool rows only**; `reindex` maps unseen test values to an all-zero row | Design fits in memory as CSR; fold-1 top coefficients are `oh_Range_Anxiety_Level=High` **−1.7728**, `=Low` **+1.1165** | The lattice reproduces the generator's *deterministic* cells exactly where they matter (Range=High is a hard −1.77 logit), and still scores 0.0030 below the GBM — the artefact value is in the *conditional*, not the cell identity | ✅ Used (diagnostic) |
| Quantile-binned high-cardinality numerics (948 cols) | 900 income bins + 48 charging bins, quantile edges from labelled pool rows, per-fold | Never individually important; `oh_incomebin_0565` is the first bin-level term to appear in the top 15 (−0.469) | Binning income at 900 levels instead of using its exact value is the *deliberate* weakness of this view — it is the control that isolates what TE contributes | ⚠️ No Improvement |
| 8 smoothed **WoE** terms (target-dependent, fold-safe) | `woe = ln(P(1\|cell)/P(0\|cell))` with smoothing, for income / commute / age / concern / subsidy / charging / car-type / city-type, mapped into the design as continuous columns | `woe_income` **+1.1143**, `woe_subsidy` **+1.0254**, `woe_commute` **+0.9795**, `woe_ecl` **+0.9385** — four of the six largest coefficients | This is the closest we have read the generator's additive law directly, and it matches the reconstructed coefficients (income/commute/concern/subsidy positive, Range=High strongly negative). It also explains V31's gain: an additive prior *is* present in the data, it just cannot be extended by trees alone | 🔬 Research (measured) |
| One-hot/WoE vs **target encoding** for a linear model | Same folds, same rows, `LogisticRegression` on 2,076 one-hot+WoE columns (V37) vs V5's TE-bearing linear run | OOF **0.943220** vs V5 **0.944677** = **−0.001457**; vs V30's GBM 0.946171 = −0.002951 | **Target encoding is the better linear representation**, because it hands the model the estimated conditional for an exact high-cardinality cell, which a fixed lattice cannot express at stable variance. The 0.002951 linear-vs-GBM gap is the price of the interaction/artefact structure only trees exploit | ❌ Removed (route closed) |
| `C` grid {0.3, 1.0, 3.0, 10.0} per fold, one design build shared | lbfgs, l2, `max_iter=1000`, n_iter 44–71 | OOF by C: **0.943220 / 0.943193 / 0.943165 / 0.943168** — the tightest penalty wins and loosening hurts monotonically | My prior (that a 2,076-column design was over-penalised) is **wrong**: the design is over-*parameterised* for this label, so more freedom only adds variance. Recorded as a hypothesis corrected by measurement | ❌ Removed |
| Cost profile | 4 C values × 10 folds on CPU | **5.0 min total**, 0.4–0.5 min per fold — cheapest run in the series by ~4× | A linear leg is nearly free to keep in the toolkit for sanity checks; it just cannot compete here | ✅ Used (tooling) |

---

## Version 36 — Confirmed LB 0.94637 (2026-09-24) ✅ Fold geometry measured twice; lever spent

FE region **byte-identical to V30** (180 base + 198 Triple TE = 312 columns). Two constants change: `N_FOLDS = 20` (which also drives `TargetEncoder(cv=20)` and the per-fold original-row concat, 9,500 of 10,000 orig rows in-train) and the split seed, `KFold(20, shuffle, random_state=7)`.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| Fold count 10 → 20 (validation geometry, not features) | Each model trains on 95% of labels (635,231 comp + 9,500 orig) and validates on 33,433 rows | OOF **0.946199** vs V30's 0.946171 = **+0.000028** at 2.1× the wall time (95.3 vs 45.3 min) | The geometry curve is now measured at both steps and it is **concave**: 5→10 = +0.000097, 10→20 = +0.000028. The lever is spent; no further fold count is worth buying | ✅ Used (verdict) |
| Second CV schema | Different seed as well as different K | Fold SD **0.001352** (V30: 0.000783), per-fold AUC 0.94346–0.94901, BestIter 2,927–7,943 — the widest spread in the archive | Half the validation rows per fold makes the mean a noisier estimate, and because the seed changed too, **+0.000028 cannot be separated from split luck**. That ambiguity is exactly why V22 used two-split confirmation; not worth 2× again for a number this small | ⚠️ Confounded |
| TE block re-fitted at `cv=20` | 66 keys × (auto/10/100) = 198 columns, now averaged over 19 out-of-fold segments | Fold-1 leaders: `TE_lift_trigram_cat_auto` 6,210, `TE_lift_trigram_cat_10` 5,127, then raw **`Environmental_Concern_Level` 2,620** and `_ev_recipe` 1,292; `TE_bigram_ECL_bin_x_RangeAnxiety_100` 1,205 rises into the top 5 | The trigram lift still leads, but a 20-fold TE is smoother and the raw generator columns gain relative share — consistent with V29/V32's finding that the ranking of features is stable while their weights move with the fitting geometry | ✅ Used (diagnostic) |

---

## Version 35 — Confirmed LB 0.94572 (2026-09-24) ❌ HistGradientBoosting family closed

FE region **byte-identical to V30**, plus one representation change native to this learner: the declared categoricals are kept as columns alongside their TE copies, so the fitted matrix is **320 columns** (180 base + 198 TE, 8 declared categorical).

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| `HistGradientBoostingClassifier` (a family never run here) | `lr .02, max_iter 6000, max_leaf_nodes 31, min_samples_leaf 40, l2_regularization 1.0, max_bins 255, early_stopping=True`, `categorical_features` passed as **integer column indices** | OOF **0.945482** vs V30 0.946171 = **−0.000689**; per-fold 0.94417–0.94732; `n_iter_` 547–984 | Fourth distinct learner on this exact matrix (XGB, LGBM, CatBoost, TabM, now HGB) and none of them beats XGBoost. **The ceiling is representation-bound, not learner-bound** — that sentence was a hypothesis before this batch and is a measurement now | ❌ Removed (family closed) |
| Native categorical handling with cardinality guard | Declare a column categorical only if its distinct count ≤ `max_bins`; sklearn hard-errors otherwise | **8 of 10** declared: the 6 originals + `income1000_floor` + `commute_integer`; 2 rejected for cardinality | The guard converts a guaranteed crash into a logged skip — this was defect #4 found while auditing V19–V30 configs | ✅ Used |
| sklearn's private early-stopping split | `early_stopping=True` holds out a random ~10% **inside** each training fold | Each fit saw ≈ 81% of V30's training rows | **V35's number is pessimistic by construction** — but not enough to matter: V32 ran with no such handicap and still sat at −0.000184, so closing −0.000689 on this correction alone is not credible. No rerun | ⚠️ Confounded |
| Time/value ratio | CPU, 10 folds, 9.7–12.6 min each | **116.7 min** for the batch's second-worst score | Screening lesson kept: a new family must be cheap or the score must be competitive; this was neither | ❌ Removed |

---

## Version 34 — Confirmed LB 0.94638 (2026-09-24) ✅ LightGBM reaches XGBoost parity — the defect was real

FE region **byte-identical to V30** (312 columns). The change is the learner and its config — specifically the first LightGBM run in this repo in which bagging actually executes.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| `subsample_freq=1` paired with `subsample=0.8` | LightGBM only samples rows when **both** are set; V3/V10/V19/V26 set `subsample` alone, so their bagging silently never ran | Published config (`lr .02, num_leaves 32, max_depth 5, min_data_in_leaf 10, bagging .8/1, feature_fraction .3, λ1 .071, λ2 2.03, max_bin 1024, 20k rounds/ES 500`) → OOF **0.946156** vs V30's 0.946171 = **−0.000015** | The latent defect was worth measuring, not just noting: with it fixed, LightGBM **ties** XGBoost on the identical matrix and folds for the first time. Four archived verdicts described a different model than intended | ✅ Used (record corrected) |
| Early stopping vs round cap | `num_boost_round=20000`, `early_stopping_rounds=500` | BestIter **892–1,985**, i.e. no fold anywhere near the cap | Contrast with V33, where 9/10 folds hit the 8,000-round cap: V34's parity figure is a converged measurement, not a truncated one | ✅ Used |
| Importances (fold 1, gain) | Same 312 columns | `TE_lift_trigram_cat_auto` 1,746,919 • `_cat_10` 946,812 • `_cat_100` 712,696 • `_ECL_x_Subsidy` 436,035 • `TE_bigram_ECL_bin_x_Subsidy_auto` 310,273 | The lift **trigram occupies three of the top five slots on a fourth learner configuration**; the artefact signal is not an XGBoost idiosyncrasy and no reweighting of this matrix will beat it | ✅ Used (diagnostic) |
| Cost of parity | CPU, 5.4–8.0 min per fold | 68.1 min vs V30's 45.3 min GPU for the same score, at ~1.3k trees vs XGBoost's 4.5–6.5k | Two families now sit on the same plateau, which means the archive's 0.9944–0.9999 pairwise rank correlation is a property of the **matrix**, not of the learner — and confirms the ensemble ceiling is where we measured it | ⚠️ No Improvement |

---

## Version 33 — Confirmed LB 0.94481 (2026-09-24) ❌ Zero-TE view: the single largest regression of the batch

The representation is the experiment: `USE_TRIPLE_TE = False` removes all **198** target-encoded columns (the 66 source keys stay in the matrix as global integer codes instead of being dropped), and `max_bin 1024 → 8192` is the only parameter change, to compensate for the lost resolution.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| **Target encoding removed entirely** (198 columns) | 180 fitted columns instead of 312; all other FE (digit block, artifact flags, lift/novelty, multi-scale keys, num-as-string, freq, targeted bigrams, groupby deviations) byte-identical to V30 | OOF **0.944967** vs V30 0.946171 = **−0.001204** — the batch's biggest single-factor loss | **The TE block *is* the score.** Every other representation axis (learner family, bin resolution, fold geometry, additive prior) moved OOF by ≤ 0.00019; removing the encoded conditionals costs six times more than any of them. This is the direct answer to V32's puzzle (a published CatBoost at CV 0.94621): the difference was never the learner | ✅ Used (verdict) |
| Numerics-as-string kept as **global factorize codes** (66 columns label-coded) | `LabelEncoder`-style integer codes over train+orig+test pool, so XGBoost histograms them instead of one-hotting | Fold-1 gains: `_ECL_x_Subsidy_cat` **20,273.9**, `_ECL_x_Subsidy` 15,255.8, then `_ev_recipe` 1,986.6 — the top pair is **60×** the third | Without TE the model rediscovers the *same* interaction from raw codes, but at far lower resolution and with no conditional information: same story, 0.0012 worse. Consistent with V21's `_ECL_x_Subsidy` seizure (59% of gain) when the feature space is narrowed | ⚠️ No Improvement |
| `max_bin 1024 → 8192` (the only parameter change) | Compensates exact-income resolution in the absence of `income_exact_int` TEs | Per-fold time **8.8–9.5 min** vs V30's 4.0–4.7 (2×), and the run still finished at **94.7 min** | V28's null (resolution axis closed) reproduces on the no-TE view: doubling bins doubles the time and buys nothing visible | ❌ Removed |
| Round-cap truncation | `NUM_ROUND = 8000`, ES 400 | BestIter **[7997, 8000, 8000, 7997, 8000, 8000, 7999, 8000, 7998, 7982]** — 9 of 10 folds stopped at the cap | **V33's 0.944967 is a floor, not a point estimate.** The honest claim is "no-TE ≤ 0.9450" plus a direction; the exact size of the TE effect is confounded with undertraining. Not rerun at 20k — the sign is already decisive and would not change any decision | ⚠️ Truncated |
| Test-prediction spread | min/max of the submitted probabilities | 0.000006 / 0.999645 vs V30's similar saturation | The no-TE model still separates the deterministic cells hard; the loss is in the graded middle, not at the extremes | 🔬 Research |

---

## Version 32 — Confirmed LB 0.94609 (2026-09-24) ❌ No improvement; CatBoost closed on valid parameters

FE region **byte-identical to V30/V22's** (180 base + 198 Triple TE = 312 fitted columns). The only change is the learner: V24 had passed `max_bin`, which **CatBoost does not have** (`border_count`), so this re-runs the family with a real bin parameter plus the published recipe.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| `border_count=1024` (the fix for V24's dead `max_bin`) + published recipe `iterations=100000, lr=.015, depth=6, l2_leaf_reg=3.0, od_wait=800` | Same matrix, same folds, CatBoost instead of XGBoost | 10-fold OOF **0.945987** vs V30 0.946171 = **−0.000184**; vs V24's 5-fold 0.945933 the gain is only ~+0.00005 where XGBoost got +0.000097 from the same fold change | The V24 verdict was a mis-run, but **re-running it correctly confirms it**: CatBoost is genuinely behind here and converts extra labels into less improvement. So the published CV 0.94621 CatBoost is a different *input representation*, not better tuning — which is V33's hypothesis | ❌ Removed (family closed) |
| `rsm=0.4` (the published column-sampling leg) | Fold-1 probe fits 20 iterations before the real run | probe printed `PRIMARY rejected (TypeError) -> using FALLBACK` | catboost 1.2.10 on GPU will not take `rsm` with any usable bootstrap, so the published config is only partially executable; the probe converts a guaranteed fold-1 crash into a logged substitution | ⚠️ Not executable here |
| CatBoost feature importances (fold 1) | gain-based, same 312 columns | `TE_lift_trigram_cat_10` 7.77, `TE_lift_trigram_cat_auto` 7.28, `TE__Income_x_Subsidy_cat_10` 7.03 | Third learner in a row (V20 XGB 14.49%, V24 CB #1/#2, V32 CB #1/#2) whose top signal is the generator-lift **trigram** — the artefact is the signal, family-independently | ✅ Used (diagnostic) |

---

## Version 31 — Confirmed LB 0.94630 (2026-09-24) ✅ Best honest OOF (0.946224); zero feature changes

**FE region byte-identical to V30/V28/V22's** (180 base + 198 Triple TE = 312 fitted columns, config unchanged). V31 adds no feature to the tree matrix — it adds a *second, additive view of the same columns* used only as a boosting prior.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| Additive backbone as `base_margin` | `LogisticRegression(C=?, l2, lbfgs, class_weight=None)` on 31 columns: 7 numerics + `_log_Income`/`_log_Commute`/`_log_Charging_Total` + 4 smooth keys (`income_exact_int`, `income100_floor`, `income1000_floor`, `commute_integer`) + one-hot of the 6 originals. **No `TE_`, `lift_`, `*_digit*`, `_fe` columns.** Training rows get out-of-fold logits from an inner `KFold(5, rs=42)`; val/test get the outer-train-fit logits. XGBoost gets those as `base_margin`. | OOF **0.946224** vs V30 0.946171 = **+0.000053** (identical folds/split/config); backbone alone **0.938310**; trees 3,298–5,177 vs V30's 4,461–6,554; LB −0.00009 | The prior works, but its size depends on the backbone *reaching the additive ceiling* — a weak backbone displaces tree capacity instead of informing it (offline proxy: −0.0020). Corroborates the depth ladder: additive-only on the pool = 0.9385, so the backbone is exactly at the additive limit | ✅ Used |
| Additive-structure probe (offline, this session) | Depth ladder on each file: single tree and HGB by `max_depth`, original 10k vs pool | Original 10k: **depth-1 HGB best at 0.90676** (2: 0.90637, 3: 0.90362, 5: 0.89702, 8: 0.89293). Pool: 1 → 0.93847, 3 → 0.94007, 8 → 0.93968 | The generator's label law is **additive**; the pool's surplus predictability is artefact, and interactions beyond depth ~3 buy nothing — which is why 30 versions of feature work plateaued | 🔬 Research (measured) |

---

## Version 30 — Confirmed LB 0.94639 (2026-09-21) ✅ Best model built; zero feature changes

The FE section is **byte-identical to V28/V22's** (verified by hashing the pipeline region, not by eye) and the XGBoost config is identical to V22's winner — deliberately, because this version changed exactly one thing: **10 folds instead of 5**, so each model trains on 90% of the labels (601,798 + 9,000 original rows per fold) instead of 80%.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| Fold count (a training change, not a feature change) | `N_FOLDS = 10`, which also drives the per-fold original-row concat and the nested `TargetEncoder(cv=10)` | 10-fold OOF **0.946171** vs 5-fold 0.946074 = **+0.000097** (external claim +0.00015); folds 0.94488–0.94810, SD 0.000799 | The only remaining measured, mechanism-free gain and it is now spent: 20 folds costs ~90 min for a fraction more. Wall time scales 4.2× not 2×, because each fold also trains on more rows | ✅ Used |
| Nested 10-fold TargetEncoder | Same Triple TE (auto/10/100) but fitted inside 10 inner folds rather than 5 | Included in the +0.000097; not separable from the fold count by design | Encoding smoothness is a known 0.0002-scale lever (thread 739354: smoothing 5→200 moves 0.0002) and is now folded in, not swept | ✅ Used |
| OOF→LB gap as a proxy-quality signal | LB minus OOF | **collapsed from a stable +0.00032/+0.00035 to +0.00022** | A 10-fold OOF sits closer to the ensemble-of-ten that actually predicts test, so less optimism remains — evidence the estimate itself improved, independent of the board | 🔬 Research |
| Consensus divergence, recomputed over 10 versions | Spearman(mean \|rank − consensus rank\|, LB) | **−0.770** (−0.667 before V29/V30). V30 divergence 1,111 → LB 0.94639, matching neighbours at 917–1,140 (V20/V23/V27) | Confirms the rule with a tenth data point: the board prices *movement in the test ranking*, not merit. V30 gained +0.000097 of true accuracy and lost 0.00001 of public score | 🔬 Research |
| Script shape | No arms, no harness, no blend, no refit, no prior, no DeLong — 774 lines, one loop | Ran first time, 45.3 min | The plain single-model form is now the template for shipping; the ablation harness is for killing ideas, not for building them | ✅ Used |

## Version 29 — Confirmed LB 0.94640 (2026-09-21) ❌ Neural family adds nothing; axis closed

No feature changes. This version asked whether a neural model can exploit the V19+ artifact stack at all — the one family question never tested on this matrix. Control = V22 winner in-run, landing on 0.94607 vs V22's stored 0.946076; the saved submission is that control, which is why the LB repeats V22's 0.94640.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| TabM on the artifact matrix | pytabkit `TabM_D_Classifier`, k=4, PWL numeric embeddings d=16, d_block 128, n_blocks 2, batch 4096, 25 epochs | OOF **0.94564** vs control 0.94607 (**−0.00043**); per-fold 0.94584/0.94484/0.94684/0.94529/0.94590 | First neural run on this matrix and it loses — but it also *underperformed V6's 0.94585 on the older 83-feature set*, so nets do not exploit the artifact stack better than trees | ❌ Removed |
| Correlation with the GBM | rank correlation per fold between the two views | **0.99514–0.99642** | Weak *and* correlated — the worst quadrant for a blend leg. V6's TabM (better and less correlated, ρ 0.9952) bought +0.000093, so this predicts ≈ +0.00005 | ❌ Removed |
| Nested column budget | per fold, the NN sees only the top 120 of 312 columns by **that fold's own** control gain (534,932 training rows; no validation label involved) | Fold-1 picks: `TE_lift_trigram_cat_auto`, `TE_lift_trigram_cat_10`, `Environmental_Concern_Level`, `_ev_recipe`, then the income/ECL TEs | Leak-free way to cut NN cost ~2.6×; also confirms the gain ranking matches the feature story from V20 onward | 🔬 Research |
| Cost of the family | 12.7 min for five folds vs the GBM's 10.1 min | — | An NN leg is nearly free to add if a *different feature view* ever justifies it; the family itself is not the blocker | 🔬 Research |
| Reproducibility audit | three saved copies of V22's estimator (V22, V28 control, V29 control) | V28 vs V29 rank **identically** (mean rank difference 0.0 rows, OOF and test). V22 (sklearn wrapper, not booster API): mean rank difference 1,600 OOF / 532 test rows — **same LB 0.94640** | The pipeline is reproducible across sessions including the GPU, and a rank perturbation of ~0.2% of positions is invisible on the public board. Caveat: V29's files are `sigmoid(probability)` because of a save-path bug, so the *values* are squashed (0.5000–0.7310) even though ranks/AUC are unaffected — compare these files by rank only | 🔬 Research |
| Board-vs-OOF reversal | Across the eight versions sharing this matrix (V19/20/22/23/25/26/27/28): OOF span 0.000107 vs LB span 0.000130 | **Spearman(OOF, LB) = −0.619**; Spearman(test-board divergence from our consensus, LB) = **−0.667**. Best OOF (V25 0.946094) has the lowest LB (0.94628); lowest OOF (V19 0.945987) is joint 2nd-best on the board (0.94639) | For six versions the public board has not been ranking our quality — it has been punishing how much a change disturbed the test ranking. The four least divergent submissions are the four best-scoring ones. Decision rule that follows: among equal-OOF candidates submit the **least divergent** model, and treat any board move below ~0.00013 as unmeasurable | 🔬 Research |

## Version 28 — Confirmed LB 0.94640 (2026-09-21) ⚠️ CV-only null, resolution axis closed

No feature changes (V22's 180 base + 198 Triple TE = 312 fitted columns, unchanged). This version tested whether the *existing* exact-value features are being throttled by the histogram bin cap, so the entry records resolution diagnostics rather than importances. Control a0 = V22 winner, landing on 0.94607 vs V22's stored 0.946076.

| Item | Definition | Result | Impact | Status |
|------|------------|--------|--------|--------|
| `max_bin` 1024 → 16384 | Unconstrained binning: income holds 14,667 distinct pool values (13,214 train) and is the only column of 13 above the cap; commute, the next largest, is 0.81× | OOF **−0.000002, z = −0.40** (4096 gave −0.00000, z = −0.21) for **4.2× the GPU time** | The axis is closed by construction — there is no higher setting to try | ⚠️ No Improvement |
| Integrity guard (trees, not requests) | Distinct thresholds actually used across all 62 income-derived columns | 1024 bins → **6,456**; 16384 → **7,972**; `TE_Annual_Income_USD_cat_auto` 318 → 538 cuts; `_10` variant 336 → 558; 62/62 vs 61/62 income columns used as splits | Proves the parameter reached the model, so the tie is a real null and not a silently ignored setting. Keep this guard pattern for any representation-level test | 🔬 Research |
| Income columns that dominate splitting | Fold-1 distinct-cut counts | `grp_income_bin_Annual_Income_USD_dev` 339, `TE_Annual_Income_USD_cat_10` 336, `TE__log_Income_cat_100` 334, `TE_Annual_Income_USD_cat_auto` 318 (1,175 splits) | The model is already carving income finely through the TE/groupby stack — which is exactly why the raw bin cap was never the binding constraint | ✅ Used |
| Resolution as a blend source | Equal-logit blend of the 1024 and 16384 arms' OOF | rank corr **0.99992**, blend +0.00000, z = +1.20 | Different bin settings are the same model. Diversity must come from a different *feature view* or family, not from representation knobs | ❌ Removed |
| Estimator fidelity | Same config, same folds, same matrix as V22 | OOF 0.94607 vs 0.946076 **and LB 0.94640 identical to the last digit** | Strong evidence that ±0.00001 LB moves between near-identical models are model difference, not board noise | 🔬 Research |

## Version 27 — Confirmed LB 0.94638 (2026-09-20) ⚠️ Probe, not promoted

No feature changes (V20's 180 base + 198 Triple TE). This version ablated two *structural* assumptions instead. Control a0 = V20's config, landing on 0.94605 vs V20's stored 0.946060.

| Item | Definition | Result vs a0 | Impact | Status |
|------|------------|--------------|--------|--------|
| Original 10k rows in the per-fold concat | 8,000 extra rows per fold = 1.47% of the training matrix | a1 drops them **+0.00002, z = +2.66** | Tie in the *removal* direction: the rows are not earning their place; our original-data gains are the pool/orig **frequency** features, not the rows | ⚠️ No Improvement |
| Original rows at sample weight 10 | `DMatrix(weight=)` on the 8,000 orig rows | a2 **-0.00003, z = -2.30** | Clean-label supervision from the source file does not transfer — do not pursue | ❌ Removed |
| `objective = rank:pairwise` | Random contiguous groups of 64 rows = uniform subsample of the global positive-negative pairs AUC averages over | a3 **-0.000329, z = -42.56** | The pairwise-AUC theory is dead. Fold AUCs swing 0.94420-0.94624 and convergence ranges 416-3,827 trees | ❌ Removed |
| Pairwise + orig rows replicated 10x | Duplication stands in for weight (this build rejects `weight` + `set_group`; equal under a pairwise loss) | a4 **-0.00069, z = -19.89** | Replicating clean-label rows recovers most of the pairwise damage but still loses | ❌ Removed |
| `eval_metric='auc'` on a ranking objective | Accepted by XGBoost 3.2 — early stopping can watch the real metric | — | Useful mechanism knowledge even though the objective failed | 🔬 Research |

## Version 26 — Confirmed LB 0.94637 (2026-09-20) ✅ First gate-cleared gain since V19

No feature changes (V19/V22's 180 base + 198 Triple TE). The change is the tree *split* bias inside LightGBM, so this entry records what the bias did to the feature story. Control a0 = V19's params, landing on 0.94599 vs V19's stored 0.945987.

| Item | Definition | Result vs a0 | Impact | Status |
|------|------------|--------------|--------|--------|
| `extra_trees=True` (+ `split_histogram_sampling`) | Split thresholds sampled from bin boundaries instead of chosen greedily | a1 **+0.00007, z = +3.58** | The bias itself pays — first arm ever to clear z > 3 | ✅ Used |
| extra_trees + wide/weak/many | depth 3, `num_leaves` 8, `min_child_samples` 50, colsample 0.85, lr 0.01, ~6,000-8,200 trees | a3 **+0.00010, z = +5.22** (rs=7 +0.00008, z = +4.11) | Reproduced on both splits — the strongest single measured gain since V19's artifact features | ✅ Used |
| extra_trees + lookup capacity | unlimited depth, 128 leaves, `feature_fraction_bynode` 0.3, lr 0.05 | a2 **-0.00023, z = -8.00** | Falsifies the memorisation mechanism: fully-grown random trees are much worse. Random thresholds work as regularisation of a *shallow* model | ❌ Removed |
| Cross-family level | a3's OOF 0.946085 vs V22 0.946076 / V25 0.946094 | z = +1.60 vs V23 | **Ties XGBoost, does not beat it.** The 0.00007 LightGBM→XGBoost gap was a split-bias gap, not an information gap | ⚠️ No Improvement |
| Cost | 387.6 min CPU; a3 needs 6,000-8,200 trees per fold | — | Expensive to explore, which is why only one config of this direction has been tried | 🔬 Research |

## Version 25 — Confirmed LB 0.94628 (2026-09-19) ⚠️ Best OOF, submission rejected

Factorial ablation on V22's matrix: nothing was removed anywhere, and each arm changed one factor against an in-run control. The cross block was appended last and excluded from the redundancy scan, so the control's matrix is V22's exactly (312 base-block columns; +54 cross-block for a1). Importance is fold-1 XGBoost gain share from the control arm.

| Feature / Item | Formula / Definition | Importance % | Impact | Status |
|----------------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the trigram lift (control arm) | 23.12 gain | #1 again; auto+10 pair = 39.3% of gain in the depth-3 wide-column model | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 16.21 gain | Second view of the same key | ✅ Used |
| `Environmental_Concern_Level` | Raw ECL column | 12.41 gain | #3 under c1-style params — raw signal reaches the top when colsample is wide | ✅ Used |
| six cross keys + their lifts (`cx_chg_x_home`, `cx_city_x_home`, `cx_ecl_x_home`, `cx_ecl_x_city`, `cx_inc_x_ecl_x_sub`, `cx_ecl_x_sub_x_comm`) | Fixed-edge Simpson/recipe crossings, Triple-TE'd + target-free lift each (54 columns) | arm OOF **-0.00006**, z = **-3.98** | **Tested alone for the first time and REJECTED.** V21 was not the removals' fault — the crosses add nothing. The last open feature door | ❌ Removed |
| `base_margin` = logit(LR on original 10k) | Per-fold logistic on 12 raw recipe drivers, clipped ±5, supplied per row through the booster API | arm OOF **+0.00002**, z = **+1.29** (rs=7 +0.00004, z=+2.45) | Only arm with a pulse — and it is a **tie**, below the z>3 gate. Its gain sits in the bottom 3 deciles = 0.15% of the AUC pair mass; LB then fell 0.00012 | ⚠️ No Improvement |
| in-run gain pruning to 94 features | 600-tree lr-0.05 probe on the fold's own training rows, top-94 by gain | arm OOF **-0.00004**, z = -2.77 | The reference recipe's 94-feature budget does NOT transfer to our 378-feature matrix | ❌ Removed |
| all three combined | cross + prior + pruning | arm OOF -0.00005, z = -2.55 | No positive interaction to rescue; the prior's tiny edge survives neither the crosses nor the pruning | ❌ Removed |

### V25 post-mortem diagnostics (offline, on saved oof/sub CSVs)

| Diagnostic | Value | Reading |
|------------|-------|---------|
| Control a0 OOF vs V22's stored OOF | 0.94607 vs 0.946076 | Harness reproduces the previous version to 6e-6 |
| Paired DeLong, v25 vs v22 (668,665 rows) | +0.000018, SE 0.000015, z = +1.21 | Best OOF we have ever had, and still a tie |
| OOF→LB gap | +0.00019 (v25) vs +0.00032…+0.00040 for every trusted model | The outlier is the submission, not the CV |
| Test rank divergence vs v22 | mean abs rank diff 2,725; 31% of rows move >1% of the board | v23: 1,598/16%, v20: 1,495/14%, v21: 2,874/36% (v21 also lost ~0.00011) |
| Prior clipping | 20.82% of train vs 20.72% of test rows, all at -5, 0% at +5 | No train/test extrapolation asymmetry — the arm is not broken |
| Submission integrity | 286,541 unique / 286,571 values, no NaN, range 3.9e-6…0.9996 | No ties, no corruption |
| Per-decile OOF delta (v25 - v22) | deciles 0/1/2: +0.0324 / -0.0151 / -0.0162; deciles 3-9 within ±0.0012 | The whole "gain" is reordering rows where AUC is nearly blind (174 of 116,779 positives) |
| LB vs OOF monotonicity check | v19 OOF 0.000107 worse than v25, LB 0.00011 better | Public LB cannot order our submissions at the 0.0001 scale |

## Version 24 — Confirmed LB 0.94615 (2026-09-19) ❌ Rejected

No new features. V20's exact matrix (180 base + 198 Triple TE) through CatBoost depth=6 to test whether the feature set family-depends. Importance is CatBoost's fold-1 percentage (predictions/gain based, sums to 100 across the full table), so values are not comparable to XGBoost gain shares above.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the ECL × Subsidy × Anxiety pool/orig lift | 8.20 | #1 under CatBoost too — the lift trigram leads in a third family | ✅ Used |
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the same lift | 8.18 | #2, near-tied with the 10-view; together they carry 16.4% | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of the ECL × Subsidy category key | 6.79 | CatBoost's native categorical path for the classic interaction | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the trigram lift | 3.64 | Third lift-trigram view; family total ≈ 20% | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 3.18 | Raw interaction, much weaker here than under XGBoost | ✅ Used |
| `_ECL_x_RangeAnxiety` | `ECL * (3 - RA_code)` | 1.41 | Behavioral interaction | ✅ Used |
| `TE_lift_Subsidy_Available_cat_auto` | Auto-smoothed TE of the subsidy lift | 1.27 | Only single-column lift in the top 8 — the crosses dominate | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy | 1.12 | V10 bigram retained but small | ✅ Used |
| `TE_lift_income100_cat_*` | Triple TE of the income-band lift | 0.46/0.26 | Income lift again absorbed by the income TEs | ⚠️ No Improvement |
| whole model | CatBoost depth=6 vs XGBoost depth=4 on identical features | — | OOF 0.94593 vs 0.94606 (-0.00013) / LB 0.94615: right features, worse estimator | ❌ Removed |

## Version 23 — Confirmed LB 0.94641 (2026-09-19) 🏆 Best

No feature changes at all: the CV path is V20's verbatim, so the fold-1 gain table below is V20's within rounding. The change is on the inference side, which this file records because it is now our only working LB lever.

| Item | Definition | Value | Impact | Status |
|------|------------|-------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the trigram lift (fold 1) | 0.14488 gain | Byte-identical to V20's 0.14488 — proof the CV path was untouched | ✅ Used |
| Full-data refit | 4112 trees (= mean best iteration) on 678,665 rows of train + original | +0.00002 LB | Test blend 0.50 fold-average + 0.50 refit; OOF unaffected, so still honest | ✅ Used |
| Refit-vs-fold-average correlation | Pearson 0.99959 / Spearman 0.99946 | mean \|rank diff\| 1,806 / 286,571 | The two predictors agree to 4th decimal; only ~0.6% of rows change rank materially | ⚠️ Research |
| Refit iteration count | `n_estimators` from the mean best iteration, no early stopping | — | Untested alternatives: per-fold mean-of-ranks blend, higher/lower tree count, blend weight ≠ 0.5 | 🔬 Research |

## Version 22 — Confirmed LB 0.94640 (2026-09-19)

No new features (180 base + 198 Triple TE, identical to V19/V20). The estimator was searched, so this entry records what the tuning winner did to the gain distribution. Importance is XGBoost fold-1 gain share; c0's column is the rs=42 table, c1's is the rs=7 fold-1 table (the harness prints only the first config of each fold).

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the trigram lift | 0.2274 (c1) vs 0.1441 (c0) | Doubles its share when depth drops to 3 and colsample rises to 0.85 | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 0.1918 (c1) vs 0.1080 (c0) | Together with `_auto` it takes 0.419 of fold-1 gain vs 0.253 for V20's params | ✅ Used |
| `Environmental_Concern_Level` | Raw ECL column | 0.1039 (c1) | Enters at #3 only in the wide-column config — more columns per split lets raw signals through | ✅ Used |
| `_ev_recipe` | `ECL == 5 & Range_Anxiety == Low` | 0.0458 (c1) vs 0.0654 (c0) | Stable composite | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety_100` | Smoothing-100 TE of the V10 bigram | 0.0444 (c1) | Promoted by the shallower tree | ✅ Used |
| `_ECL_x_Subsidy` | `ECL * Subsidy_Available` | 0.1084 (c0), absent from c1's top 10 | Loses share to the lift crosses under c1 | ⚠️ No Improvement |
| Estimator settings | c1 = `max_depth=3, max_leaves=8, gamma=1.0, colsample_bytree=0.85`, ~5.5k trees | +0.00002 OOF over the in-run c0 control on both rs=42 and rs=7 | Confirms direction: wide + weak + many beats deep + heavily regularised on this matrix | ✅ Used |

> Caveat: V22's c0 fold-1 gain shares (e.g. `TE_lift_trigram_cat_auto` 0.14410) sit slightly below V20/V23's 0.14488 even though c0 reproduced V20's fold AUCs and best iterations exactly. V22 computed importances from the untruncated booster (it ran to `n_estimators=8000`), V20/V23 slice at `best_iteration`. Metrics are unaffected; treat cross-version gain shares as ±0.001.

### V22 search outcomes (feature set fixed, estimator varied)

| Config | Override | rs=42 OOF | Verdict |
|--------|----------|-----------|---------|
| c1 shallow + wide cols | depth 3, 8 leaves, gamma 1.0, colsample 0.85 | 0.94608 | ✅ Winner (also 0.94608 on rs=7) |
| c0 V20 baseline | none (control) | 0.94606 | ✅ Reproduced V20 exactly |
| c4 low min_child_weight | `min_child_weight` reduced | 0.94605 | ⚠️ No Improvement |
| c2 deeper + heavy leaf reg | deeper + stronger leaf regularisation | 0.94593 | ❌ Worse |
| c3 sparse cols + high gamma | colsample ↓, gamma ↑ | 0.94576 | ❌ Worse |
| c5 tiny lr, near-depthless | lr 0.008, `min_child_weight` 500, lambda 10 | 0.94546 | ❌ Clearly worse |

## Version 21 — Confirmed LB 0.94629 (2026-09-19) ❌ Rejected

Importance is fold-1 XGBoost gain share. V21 added six explicit cross keys (each Triple-TE'd plus a target-free lift column) and removed the digit block and four income-anomaly flags. The experiment regressed (paired DeLong vs V20: -0.00009, z = -5.01), so V20 remains the reference feature set.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 58.99 gain | Pathological monopoly — 5.4x its V20 share after the feature pool narrowed; starving every TE branch | ⚠️ No Improvement |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_10` | Smoothing-10 TE of the income-band x ECL x Subsidy lift | 5.12 gain | Strongest NEW cross; #2 overall — the addition itself worked | ✅ Used |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_auto` | Auto-smoothed TE of the same lift | 4.60 gain | Companion view of the new deepest key | ✅ Used |
| `TE_cx_inc_x_ecl_x_sub_auto` | Auto-smoothed TE of the raw cross key | 4.39 gain | Key without the lift transform | ✅ Used |
| `TE_cx_inc_x_ecl_x_sub_10` | Smoothing-10 TE of the raw cross key | 3.34 gain | Lower-smoothing view | ✅ Used |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_100` | Smoothing-100 TE of the cross lift | 3.17 gain | Stable broad encoding | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the ECL x Subsidy x Anxiety lift | 3.10 gain | Was #1 in V20 at 14.49; displaced by the near-duplicate deeper key | ⚠️ No Improvement |
| `TE_cx_inc_x_ecl_x_sub_100` | Smoothing-100 TE of the raw cross key | 2.14 gain | Third view of the same key | ✅ Used |
| `TE_lift_trigram_cat_10` / `_auto` | Lift-trigram TE variants | 2.13 / 1.21 gain | Combined lift-trigram share fell from ~35% (V20) to ~6.7% | ⚠️ No Improvement |
| `lift_Range_Anxiety_Level` | Pool/orig lift of the anxiety level | 0.87 gain | Rose in rank but small absolute gain | ✅ Used |

### V21 cross keys and removals

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `cx_chg_x_home` + `lift_cx_chg_x_home` | charging-total bins x Home_Charging_Possible (Simpson reversal 9.20% -> 14.70%) | not in top-15 | Built correctly, contributed almost nothing once the gain collapsed onto `_ECL_x_Subsidy` | ⚠️ No Improvement |
| `cx_city_x_home` + lift | City_Type x Home_Charging_Possible (Rural 4.64% vs Urban 13.56%) | not in top-15 | Same as above | ⚠️ No Improvement |
| `cx_ecl_x_home` / `cx_ecl_x_city` + lifts | ECL x Home_Charging, ECL x City_Type | not in top-15 | Composition confounders, absorbed by the raw columns | ⚠️ No Improvement |
| `cx_ecl_x_sub_x_comm` + lift | ECL x Subsidy x commute bins (5.0 km isolated) | not in top-15 | The 5.0 km bin did not register | ⚠️ No Improvement |
| digit block (56 cols in V20) | place-value digits and their `_cat` / `_fe` / TE variants | removed | Removing them did not cause the loss directly but narrowed the pool from 312 to 285 fit columns | ❌ Removed |
| `is_30k_spike` / `is_millionaire_cliff` / `is_dead_zone` / `_below_buy_bound` / `_dead_zone_exact` | income anomaly + structural recipe flags | removed with digits | Same removal batch; `is_env_hater` was kept | ❌ Removed |

## Version 20 — Confirmed LB 0.94639 (2026-09-19)

No new features were engineered in V20; the V19 matrix was reused unchanged and only the model family was swapped to XGBoost depth=4. Importance is the reported fold-1 XGBoost **gain share** (not a split count, so it is not comparable to V19's counts).

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the ECL × Subsidy × Anxiety pool/orig lift | 14.49 gain | #1 feature overall (was #4 under LightGBM) — a depth-4 model depends heavily on the pre-computed artifact cross | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 10.87 gain | Strongest raw interaction, unchanged | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 10.84 gain | Second lift-trigram view | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the trigram lift | 9.31 gain | Stable broad lift encoding | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy | 9.12 gain | V10 bigram still carries signal | ✅ Used |
| `_ev_recipe` | `ECL == 5 & Range_Anxiety == Low` | 6.60 gain | Composite recipe segment | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_auto` | Auto-smoothed TE of ECL bin × subsidy | 4.95 gain | Companion bigram view | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * (3 - RA_code)` | 4.47 gain | Behavioral interaction | ✅ Used |
| `lift_trigram` | Raw pool/orig lift of the trigram key | 2.14 gain | Direct artifact ratio | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 TE of ECL × subsidy categories | 2.05 gain | Categorical interaction view | ✅ Used |

### V20 usage of the V19 artifact set

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `lift_trigram_cat_fe` | Frequency encoding of the trigram lift | 0.19 gain | Marginal next to the TE views | ✅ Used |
| `TE_lift_commute_cat_*` | Triple TE of integer-commute lift | 0.12/0.10 gain | Weak under XGBoost (stronger under LightGBM) | ⚠️ No Improvement |
| `lift_income100_cat_fe` / `lift_inc_exact_cat_fe` | Frequency encoding of the income-band / exact-income lift | 0.10/0.08 gain | Income lift largely absorbed by income TE keys | ⚠️ No Improvement |
| `novel_inc_cat_fe` | Frequency encoding of the novel-income flag | 0.08 gain | Almost unused by the shallow model | ⚠️ No Improvement |
| `_below_buy_bound` / `_dead_zone_exact` | Structural recipe bounds (income < 41,667 / 31,004–41,970) | ~0 gain | Not selected; matches the reference notebook pruning its equivalent flags | ⚠️ No Improvement |
| `{col}_digit{k}` and digit TE variants | Fixed integer-place digits | ≤ 0.08 gain | Near-zero contribution under depth=4 | ⚠️ No Improvement |

## Version 19 — Confirmed LB 0.94639 (2026-09-19) 🏆 Best

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted V19 model. Lift features are target-free: pool (train+test) frequency ÷ original-dataset frequency.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 735 (count) | Strongest fold-1 signal, unchanged from V10 | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 TE of the 100-dollar income floor key | 632 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed TE of income-as-string categories | 601 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the pool/orig frequency lift of ECL × Subsidy × Anxiety | 531 (count) | Strongest NEW artifact feature; encodes the ECL=3 cells that beat the linear recipe by 1.22–1.31× | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 TE of income-as-string categories | 458 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed TE of age-as-string categories | 408 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed TE of log-income categories | 406 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 335 (count) | Second trigram-lift view | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed TE of income × subsidy categories | 333 (count) | Strong affordability interaction | ✅ Used |
| `TE_lift_income100_cat_auto` | Auto-smoothed TE of the 100-dollar income-band lift | 330 (count) | Income over/under-production signal | ✅ Used |
| `grp_income_bin_Annual_Income_USD_dev` | Income deviation from income-bin group mean | 323 (count) | Preserved V10 groupby signal | ✅ Used |
| `TE_income_exact_int_auto` | Auto-smoothed TE of exact integer income | 318 (count) | Captured repeated exact-income structure | ✅ Used |
| `lift_inc_exact_cat_fe` | Frequency encoding of the exact-income lift bucket | 313 (count) | Direct target-free artifact prevalence signal | ✅ Used |

### V19 generator-artifact features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `lift_inc_exact` | pool freq(exact income) ÷ orig freq(exact income) | 230 (count) | Raw lift column; buy rate 23.19% under-produced vs 17.12% heavily over-produced | ✅ Used |
| `TE_lift_commute_cat_auto/10/100` | Triple TE of integer-commute lift | 230/188/240 (count) | Added commute over-production signal | ✅ Used |
| `TE_lift_income100_cat_100` | Smoothing-100 TE of income-band lift | 220 (count) | Stable broad income-lift encoding | ✅ Used |
| `lift_income100_cat_fe` | Frequency encoding of income-band lift | 211 (count) | Secondary artifact prevalence signal | ✅ Used |
| `TE_Annual_Income_USD_digit3_cat_auto` | TE of the fixed thousands-digit feature (was corrupted pre-V19) | 198 (count) | Recovered by the digit fix | ✅ Used |
| `TE_Annual_Income_USD_digit2_cat_auto` | TE of the fixed hundreds-digit feature | 178 (count) | Recovered by the digit fix; hundreds digit ≈ round-income marker | ✅ Used |
| `novel_inc` | Income value absent from the original 10k dataset | — | 2.06% of rows at 19.56% buy vs 17.46% base; low split count | ✅ Used |
| `_below_buy_bound` | `Annual_Income_USD < 41667` (recipe can never reach 5.5) | — | Structural near-zero region | ✅ Used |
| `_dead_zone_exact` | `31004 <= Annual_Income_USD <= 41970` | — | 1,257 train rows, 0 buyers | ✅ Used |
| `lift_Subsidy_Available` / `lift_Range_Anxiety_Level` / `lift_Home_Charging_Possible` / `lift_Gender` / `lift_City_Type` / `lift_Current_Car_Type` / `lift_Environmental_Concern_Level` | Per-value pool ÷ orig frequency ratio per categorical | Not in top-15 | Low-cardinality lifts were near-collinear with the raw categorical and its TE; some variants were removed by redundancy selection (149 features dropped) | ⚠️ No Improvement |

### V19 digit-extraction fix (all 18 prior versions were affected)

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `{col}_digit{k}` | Old: `col // (10**k) % 10` — negative k uses float floor-div and reads IEEE-754 binary representation | 198 (digit3), 178 (digit2) fold-1 | Corrupted 89.63% of `Daily_Commute_km` tenths; `digit-1` was effectively an is-integer flag | ❌ Removed |
| `{col}_digit{k}` (fixed) | `np.rint(col*1e4).astype(int64)` then integer `// 10**(k+4) % 10` | as above | Correct tenths digit spans buy rate 0.160 → 0.184 | ✅ Used |

## Version 18 — Confirmed LB 0.94635 (2026-09-19)

No new features were engineered; V18 combined saved OOF predictions only. This entry records the ensemble representation and its diagnostics.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Hill-climber blend | `0.35·V14 + 0.28·V3 + 0.15·V11 + 0.13·V6 + 0.05·V2 + 0.05·V8` on raw probabilities | N/A | LB 0.94635, -0.00001 vs V10; weights fit OOF directly so the +0.00015 OOF gain was noise | ⚠️ No Improvement |
| Simple/rank/logit averages | Equal-weight mean of probabilities, ranks, and logit-space means | N/A | OOF ≤0.94601, all below best single +0.94608 | ⚠️ No Improvement |
| RidgeCV meta-learner | Ridge on OOF logit features, alpha chosen by 5-fold CV | N/A | OOF 0.94314; stacker broken by single-fold OOF vs averaged-test aggregation mismatch | ❌ Removed |

## Version 17 — Confirmed LB 0.94612 (2026-09-11)

No fold-level feature importances were reported. This entry records the RealMLP architecture and the V14-derived feature representation.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| V14 full feature pipeline | Digits, smooth keys, targeted bigrams/trigrams, frequency features, and original target means | N/A | Supplied the full engineered input representation | ✅ Used |
| Triple target encoding | Auto/10/100-smoothed target encodings across selected categorical-like columns | N/A | Added categorical signal for the neural model | ✅ Used |
| PBLD embeddings | Piecewise-linear/binned numerical embeddings used by RealMLP | N/A | Converted numerical patterns into learnable representations | ✅ Used |
| 8-model ensemble | Eight RealMLP ensemble members averaged for prediction | N/A | Reduced neural-model variance | ✅ Used |
| EMA and label smoothing | Exponential moving average plus smoothed training targets | N/A | Stabilized the short three-epoch training regime | ✅ Used |
| Original-data concatenation | Competition data combined with original data per fold | N/A | Preserved the proven V14 training setup | ✅ Used |

## Version 16 — Confirmed LB 0.94625 (2026-09-11)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V16 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 34.0651 | Strongest overall fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 20.3502 | Strongest interaction encoding | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 9.1921 | Strong categorical interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 8.5334 | Major composite adoption signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 5.9434 | Strong behavioral interaction | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 2.3350 | Strong targeted bigram signal | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 2.1165 | Preserved direct environmental signal | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.2917 | Added income-band signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.1424 | Added affordability interaction | ✅ Used |
| `TE__log_Income_cat` | Target encoding of log-income categories | 1.0560 | Added nonlinear income-shape signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.9369 | Added behavioral categorical interaction | ✅ Used |
| `TE_Annual_Income_USD_cat` | Target encoding of income-as-string categories | 0.9294 | Added exact-income categorical signal | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.7350 | Added compact anxiety prevalence signal | ✅ Used |
| `TE_income_exact_int` | Target encoding of exact integer income | 0.6939 | Captured repeated exact-income structure | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety` | Target encoding of ECL bin × range anxiety | 0.5194 | Added targeted anxiety interaction | ✅ Used |

### V16 forensic-targeted features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_Sub_HomeCharging` | Target encoding of subsidy × home-charging possibility | 0.1375 | Strongest forensic feature, but small overall contribution | ✅ Used |
| `_recipe_sub_income` | Recipe/subsidy/income interaction | 0.0370 | Captured subgroup lift but added limited global signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL3` | Fold-safe distance to buyer centroid within ECL=3 | 0.0366 | Added small geometric subgroup signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL1` | Fold-safe distance to buyer centroid within ECL=1 | 0.0353 | Added small geometric subgroup signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL2` | Fold-safe distance to buyer centroid within ECL=2 | 0.0340 | Added small geometric subgroup signal | ✅ Used |

## Version 15 — Confirmed LB 0.91256 (2026-09-10)

No fold-level feature importances were reported. This entry records the encoded lookup/KNN representation and exact-match diagnostics.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Exact-match key | Encoded combination of the 24 KNN input features used for lookup | N/A | Produced 0% validation/test exact matches; no lookup signal was available | ⚠️ No Improvement |
| KDTree numerical representation | 24 encoded train/test/original features used for Euclidean neighbor search | N/A | Enabled k=10 fallback predictions but produced weak AUC | ✅ Used |
| KNN k=10 fallback | Mean target of the 10 nearest training neighbors | N/A | Main prediction mechanism; underperformed all established baselines | ⚠️ No Improvement |
| Train-key uniqueness check | All 668,665 training keys were unique | N/A | Confirmed the data did not contain a deterministic duplicate-key shortcut | 🔬 Research |
## Version 14 — Confirmed LB 0.94630 (2026-09-10)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V14 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 39.9435 | Strongest fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 18.3191 | Strongest interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 9.2675 | Major composite adoption signal | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 6.5782 | Strong categorical interaction encoding | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 6.3003 | Strong behavioral interaction | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 1.7572 | Preserved direct environmental signal | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 1.3927 | Strong targeted bigram signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.2637 | Added affordability interaction | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.0890 | Added income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.9194 | Added behavioral categorical interaction | ✅ Used |

### Pseudo-labeling configuration

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| High-confidence pseudo-labels | Teacher probability `p>=0.98` or `p<=0.02` | N/A | Added 150,859 test rows at half weight | ✅ Used |
| V10 teacher predictions | Test probabilities from V10 LightGBM | N/A | Supplied pseudo-labels for the V12 XGBoost student | ✅ Used |

## Version 13 — Confirmed LB 0.94568 (2026-09-09)

No fold-level feature importances were reported. This entry records the selected feature architecture and observed model contribution.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| V6 evidence-based 79-feature subset | Features selected from the V3/V10/V12 feature family using V6 evidence | N/A | Compact input representation for DCN-V2 | ✅ Used |
| DCN-V2 cross features | Four cross layers with rank 64 low-rank factorization | N/A | Modeled explicit high-order feature interactions | ✅ Used |
| DCN-V2 mixture-of-experts | Four experts in the deep component | N/A | Added multiple nonlinear interaction pathways | ✅ Used |
| Proven targeted interactions | V10 targeted bigrams plus V12 trigram and income-band × subsidy features | N/A | Preserved the strongest discussion-derived interactions | ✅ Used |
| Categorical feature group | 18 selected categorical features | N/A | Enabled learned categorical representations | ✅ Used |
| Numerical feature group | 61 selected numerical features | N/A | Provided compact continuous inputs | ✅ Used |

## Version 12 — Confirmed LB 0.94629 (2026-09-09)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V12 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 35.3514 | Strongest overall fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 18.2776 | Strongest discussion-derived feature | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 8.3521 | Strong categorical interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 7.5737 | Major composite adoption signal | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 6.1115 | Strong direct environmental signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 5.8006 | Strong secondary behavioral interaction | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 1.5226 | Strong targeted bigram signal | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.2162 | Added income-band signal | ✅ Used |
| `TE__log_Income_cat` | Target encoding of log-income categories | 1.0464 | Added nonlinear income-shape signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.0406 | Added affordability interaction | ✅ Used |
| `TE_income_exact_int` | Target encoding of exact integer income | 0.8521 | Captured repeated exact-income structure | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.8519 | Added behavioral categorical interaction | ✅ Used |
| `TE_Annual_Income_USD_cat` | Target encoding of income-as-string categories | 0.8364 | Added exact-income categorical signal | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.7139 | Added compact anxiety prevalence signal | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety` | Target encoding of ECL bin × range anxiety | 0.5799 | Added targeted anxiety interaction | ✅ Used |

### Proven discussion features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_income_band_x_Subsidy` | Target encoding of income band × subsidy | 0.0607 | Small but retained targeted affordability signal | ✅ Used |

## Version 11 — Confirmed LB 0.94629 (2026-09-09)

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted V11 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_recipe_residual` | `recipe_score - 5.5`, measuring distance from the recipe boundary | 996 (count) | Strongest reported fold-1 feature | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 761 (count) | Strong income-band signal | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 759 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 634 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 606 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 488 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_income100_floor_100` | Smoothing-100 target encoding of the 100-dollar income floor key | 423 (count) | Stable broad income-band encoding | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 404 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 381 (count) | Complementary income-shape view | ✅ Used |
| `TE_Annual_Income_USD_cat_100` | Smoothing-100 target encoding of income-as-string categories | 377 (count) | Stable exact-income encoding | ✅ Used |
| `TE_commute_integer_auto` | Auto-smoothed target encoding of integer commute distance | 362 (count) | Added commute-pattern signal | ✅ Used |
| `TE_income_exact_int_auto` | Auto-smoothed target encoding of exact integer income | 350 (count) | Captured repeated exact-income structure | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 350 (count) | Added income prevalence information | ✅ Used |
| `TE_trigram_Sub_ECL_RA_auto` | Auto-smoothed TE of subsidy × environmental concern × range anxiety | 339 (count) | Strongest new trigram encoding | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 326 (count) | Strong affordability interaction | ✅ Used |

### New V11 deep-analysis features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_trigram_Sub_ECL_RA_10` | Smoothing-10 TE of subsidy × environmental concern × range anxiety | 257 (count) | Complementary trigram signal | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_auto` | Auto-smoothed TE of income band × subsidy | 121 (count) | Targeted affordability interaction | ✅ Used |
| `TE_trigram_Sub_ECL_RA_100` | Smoothing-100 TE of subsidy × environmental concern × range anxiety | 87 (count) | Stable broad trigram encoding | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_100` | Smoothing-100 TE of income band × subsidy | 66 (count) | Stable targeted affordability encoding | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_10` | Smoothing-10 TE of income band × subsidy | 49 (count) | Lower-smoothing targeted affordability encoding | ✅ Used |
| `_never_buy_flag` | `Subsidy_Available=No OR Range_Anxiety_Level=High` | 0 (count) | Retained for analysis despite no fold-1 split importance | ✅ Used |
| `_income_round00` | Indicator that income ends in `00` | 0 (count) | Retained for analysis despite no fold-1 split importance | ✅ Used |

## Version 10 — Confirmed LB 0.94636 (2026-09-09)

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted corrected V10 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 948 (count) | Strongest reported fold-1 signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 877 (count) | Complementary broad income-band signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 747 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 651 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE_income100_floor_100` | Smoothing-100 target encoding of the 100-dollar income floor key | 529 (count) | Stable broad income-band encoding | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 491 (count) | Strong affordability interaction | ✅ Used |
| `grp_income_bin_Annual_Income_USD_dev` | Income deviation from income-bin group mean | 461 (count) | Strongest targeted groupby feature | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 454 (count) | Added income prevalence information | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 453 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_commute_integer_auto` | Auto-smoothed target encoding of integer commute distance | 443 (count) | Added commute-pattern signal | ✅ Used |
| `TE_Annual_Income_USD_cat_100` | Smoothing-100 target encoding of income-as-string categories | 441 (count) | Stable exact-income encoding | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 425 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 415 (count) | Complementary income-shape view | ✅ Used |
| `grp_ECL_bin_Annual_Income_USD_dev` | Income deviation from environmental-concern-bin group mean | 406 (count) | Strong targeted environmental-income groupby feature | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 401 (count) | Lower-smoothing affordability signal | ✅ Used |

### Targeted new features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_ECL_bin_x_Subsidy_auto` | Auto-smoothed TE of ECL bin × subsidy interaction | 348 (count) | Strongest targeted bigram signal | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety_auto` | Auto-smoothed TE of ECL bin × range-anxiety interaction | 317 (count) | Strong targeted behavioral interaction | ✅ Used |
| `grp_Subsidy_Available_Annual_Income_USD_dev` | Income deviation from subsidy-availability group mean | 243 (count) | Added subsidy-income context | ✅ Used |
| `grp_Range_Anxiety_Level_Daily_Commute_km_dev` | Commute deviation from range-anxiety group mean | 215 (count) | Added anxiety-commute context | ✅ Used |
| `grp_Range_Anxiety_Level_Annual_Income_USD_dev` | Income deviation from range-anxiety group mean | 206 (count) | Added anxiety-income context | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy interaction | 201 (count) | Added lower-smoothing targeted bigram signal | ✅ Used |
| `grp_ECL_bin_Daily_Commute_km_dev` | Commute deviation from environmental-concern-bin group mean | 196 (count) | Added environmental-commute context | ✅ Used |
| `grp_income_bin_Daily_Commute_km_dev` | Commute deviation from income-bin group mean | 189 (count) | Added income-commute context | ✅ Used |

## Version 9 — Confirmed LB 0.94612 (2026-09-09)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V9 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy_cat` | Categorical interaction of environmental concern × subsidy | 39.1568 | Strongest fold-1 signal in the full FE XGBoost model | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 36.0873 | Second dominant interaction; together with the categorical form drove most importance | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 4.6745 | Stable broad encoding of the main interaction | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 1.7511 | Retained a useful composite adoption signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 1.4976 | Strong secondary behavioral interaction | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 1.1030 | Preserved direct environmental signal | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 0.8482 | Added interaction prevalence information | ✅ Used |
| `TE_income_exact_int_10` | Smoothing-10 target encoding of exact integer income | 0.6434 | Captured repeated exact-income structure | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 0.5420 | Added lower-smoothing income signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 0.5196 | Stable broad behavioral interaction encoding | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 0.4694 | Added nonlinear income-shape signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 0.4547 | Added lower-smoothing affordability interaction | ✅ Used |
| `Environmental_Concern_Level_cat_fe` | Frequency encoding of environmental concern categories | 0.3458 | Added prevalence information for environmental concern | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.2688 | Added compact anxiety prevalence signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_10` | Smoothing-10 target encoding of environmental concern × range anxiety categories | 0.2551 | Added lower-smoothing behavioral interaction | ✅ Used |

## Version 8 — Confirmed LB 0.94603 (2026-09-09)

Importance is the reported fold-1 CatBoost Ordered feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V8 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 15.319007 | Strongest fold-1 feature; remained dominant under Ordered boosting | ✅ Used |
| `TE__ECL_x_Subsidy_cat_auto` | Auto-smoothed target encoding of environmental concern × subsidy categories | 8.098418 | Strong smoothed version of the main interaction | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 7.654647 | Stable broad interaction encoding | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 5.052358 | Added interaction prevalence information | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 4.635254 | Strong behavioral interaction | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 3.588998 | Complementary affordability signal | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 target encoding of environmental concern × subsidy categories | 3.518334 | Added a lower-smoothing interaction view | ✅ Used |
| `Annual_Income_USD_org_mean` | Original-data target mean for annual income | 3.280062 | Strong original-data income prior | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 2.891573 | Lower-smoothing affordability signal | ✅ Used |
| `_ECL_x_RangeAnxiety_cat_fe` | Frequency encoding of environmental concern × range anxiety categories | 2.696740 | Added prevalence structure to the behavioral interaction | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 2.677775 | Added an income-band signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 2.343336 | Complementary broad income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_auto` | Auto-smoothed target encoding of environmental concern × range anxiety categories | 2.331000 | Complementary behavioral encoding | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 2.191141 | Stable broad behavioral interaction signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 1.621161 | Stable broad affordability encoding | ✅ Used |

## Version 7 — Confirmed LB 0.94568 (2026-09-09)

Importance is the evidence-based selected-feature set derived from V5 LogisticRegression and V3 LightGBM importance. No isolated ablation was run, so impact records observed contribution of the retained feature groups in the submitted V7 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Evidence-based 79-feature subset | Features selected using V5 LR and V3 LightGBM importance from 344 raw features | N/A | Reduced the FT-Transformer input space while retaining established linear and tree-model signals | ✅ Used |
| Auto-smoothed target encoding | Auto-smoothed TE on selected source columns, with 93 TE features generated | N/A | Reduced redundancy for neural-network training while preserving categorical-like signal | ✅ Used |
| Numerical feature group | 35 selected numerical features after feature selection | N/A | Provided continuous inputs for the transformer | ✅ Used |
| Categorical feature group | 34 selected categorical features with cardinality below 20 | N/A | Enabled learned categorical embeddings and self-attention over compact categorical inputs | ✅ Used |
| Engineered interactions and smooth keys | Income, commute, environmental, subsidy, and range-anxiety interactions/bins | N/A | Preserved structured nonlinear signals in the selected representation | ✅ Used |
| Original target means and frequency features | Original-data means plus selected frequency encodings | N/A | Added compact prior and prevalence information | ✅ Used |

## Version 6 — Confirmed LB 0.94606 (2026-09-08)

Importance is the evidence-based selected-feature set derived from V5 LogisticRegression and V3 LightGBM importance. No isolated ablation was run, so impact records observed contribution of the retained feature groups in the submitted V6 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Evidence-based 83-feature subset | Features selected using V5 LR and V3 LightGBM importance from 344 raw features | N/A | Reduced the neural-network input space while retaining strong linear and tree-model signals | ✅ Used |
| Per-fold target encoding | Target encoding from 14 source columns computed within each fold | N/A | Provided leakage-controlled categorical-like signal for TabM | ✅ Used |
| Original target means | Per-feature means from the original dataset concatenated per fold | N/A | Preserved signal from the original data source | ✅ Used |
| Frequency features | Top frequency-encoded features retained during selection | N/A | Added compact prevalence information without the full raw categorical expansion | ✅ Used |
| Engineered interactions and smooth keys | Income, commute, environmental, subsidy, and range-anxiety interactions/bins | N/A | Retained structured nonlinear signals in the compact feature set | ✅ Used |
| Hard-edge and synthetic-artifact flags | 30k spike, millionaire cliff, dead zone, env-hater, and related flags | N/A | Preserved known synthetic-data boundary signals | ✅ Used |

## Version 5 — Confirmed LB 0.94478 (2026-09-08)

Importance is the reported fold-1 LogisticRegression absolute coefficient. No isolated ablation was run, so impact records observed model contribution in the submitted V5 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `Subsidy_Available_fe` | Frequency encoding of `Subsidy_Available` | 1.954700 (|coefficient|) | Strongest fold-1 linear signal | ✅ Used |
| `Environmental_Concern_Level_org_mean` | Original-data target mean for environmental concern level | 1.831096 (|coefficient|) | Strong direct original-data signal | ✅ Used |
| `Environmental_Concern_Level_cat_fe` | Frequency encoding of environmental concern level as a category | 1.538157 (|coefficient|) | Strong categorical prevalence signal | ✅ Used |
| `_Income_x_Subsidy_cat_fe` | Frequency encoding of income × subsidy interaction categories | 0.934229 (|coefficient|) | Main linear affordability interaction | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 0.604567 (|coefficient|) | Captured exact-income prevalence | ✅ Used |
| `TE_income1000_floor_100` | Smoothing-100 target encoding of the 1000-dollar income floor key | 0.592623 (|coefficient|) | Stable broad income-band signal | ✅ Used |
| `_ecl_max` | Maximum-based environmental concern summary | 0.400672 (|coefficient|) | Added a compact threshold-style environmental signal | ✅ Used |
| `City_Type_org_mean` | Original-data target mean for city type | 0.373705 (|coefficient|) | Added direct city-type prior information | ✅ Used |
| `TE_income1000_floor_10` | Smoothing-10 target encoding of the 1000-dollar income floor key | 0.366862 (|coefficient|) | Added lower-smoothing income-band signal | ✅ Used |
| `Annual_Income_USD_org_mean` | Original-data target mean for annual income | 0.336444 (|coefficient|) | Added original-data income prior | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 0.306922 (|coefficient|) | Added interaction prevalence information | ✅ Used |
| `income100_floor_fe` | Frequency encoding of the 100-dollar income floor key | 0.300946 (|coefficient|) | Added income-band prevalence | ✅ Used |
| `income1000_floor_fe` | Frequency encoding of the 1000-dollar income floor key | 0.241697 (|coefficient|) | Added coarse income prevalence | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 0.238883 (|coefficient|) | Added smoothed affordability signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 0.235881 (|coefficient|) | Added nonlinear income-shape signal | ✅ Used |

## Version 4 — Confirmed LB 0.94590 (2026-09-08)

Importance is the reported fold-1 CatBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V4 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 10.971282 | Strongest fold-1 feature and main subsidy/environment interaction | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 5.836506 | Strong target-encoded affordability interaction | ✅ Used |
| `TE__ECL_x_Subsidy_cat_auto` | Auto-smoothed target encoding of environmental concern × subsidy categories | 5.583682 | Added a smoothed view of the strongest interaction | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 5.565833 | Captured interaction prevalence structure | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 5.285489 | Stable broad interaction encoding | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 target encoding of environmental concern × subsidy categories | 5.117647 | Added a lower-smoothing interaction signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 4.525729 | Complementary affordability signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 3.907854 | Stable heavily smoothed affordability signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 3.596592 | Strong behavioral interaction | ✅ Used |
| `_ECL_x_RangeAnxiety_cat_fe` | Frequency encoding of environmental concern × range anxiety categories | 3.095117 | Added prevalence information to the interaction | ✅ Used |
| `Range_Anxiety_Level` | Native CatBoost categorical range-anxiety feature | 2.520769 | Retained direct categorical behavior signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 2.432908 | Added a broad income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_auto` | Auto-smoothed target encoding of environmental concern × range anxiety categories | 2.158419 | Complementary behavioral interaction encoding | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 2.059766 | Added a lower-smoothing income-band view | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 1.820202 | Stable broad behavioral interaction signal | ✅ Used |

## Version 3 — Confirmed LB 0.94634 (2026-09-08)

Importance is the reported fold-1 LightGBM split-importance count. No isolated ablation was run, so impact records observed model contribution in the submitted V3 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_auto` | Triple target encoding of the 100-dollar income floor key with auto smoothing | 949 (count) | Strongest reported fold-1 signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of the log-income categorical representation | 799 (count) | Major income-shape signal | ✅ Used |
| `TE_income100_floor_10` | Target encoding of the 100-dollar income floor key with smoothing 10 | 790 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_income100_floor_100` | Target encoding of the 100-dollar income floor key with smoothing 100 | 652 (count) | Complementary heavily smoothed income signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categorical interaction | 639 (count) | Strong affordability-by-subsidy interaction | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of the log-income categorical representation | 588 (count) | Added a second income-shape view | ✅ Used |
| `Annual_Income_USD` | Raw annual income | 579 (count) | Retained direct continuous income signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 562 (count) | Strong behavioral interaction | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 561 (count) | Captured repeated exact-income structure | ✅ Used |
| `_Income_x_Subsidy` | `Annual_Income_USD * Subsidy_Available` interaction | 527 (count) | Useful affordability interaction | ✅ Used |
| `TE__log_Income_cat_100` | Smoothing-100 target encoding of the log-income categorical representation | 506 (count) | Stable broad income signal | ✅ Used |
| `income100_floor_fe` | Frequency encoding of the 100-dollar income floor key | 492 (count) | Added prevalence information for income bands | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 489 (count) | Added lower-smoothing exact-income signal | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 477 (count) | Captured age-specific response structure | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 467 (count) | Stable affordability interaction encoding | ✅ Used |

## Version 2 — Confirmed LB 0.94569 (2026-09-08)

Importance is the reported fold-1 XGBoost importance. No isolated ablation was run, so impact records observed model contribution in the submitted V2 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `LE_Subsidy_Available` | Label-encoded `Subsidy_Available` | 26.33 | Strongest fold-1 signal in the Triple TE model | ✅ Used |
| `_recipe_score` | cdeotte-style buy score / recipe score | 25.41 | Major composite signal that ranked just below the subsidy label feature | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 16.85 | Stayed highly predictive even with Triple TE added | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 3.46 | Smaller but still meaningful interaction feature | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 1.51 | Continued to add signal in the richer feature space | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern numeric feature | 1.24 | Retained direct signal after encoding and interactions | ✅ Used |
| `_ECL_x_Subsidy_freq` | Frequency-encoded version of `_ECL_x_Subsidy` | 1.09 | Added a compact frequency-based interaction signal | ✅ Used |
| `TE_Annual_Income_USD_10` | Target encoding of `Annual_Income_USD` with 10 smoothing | 0.77 | Benefited from the lower-smoothing TE view on income | ✅ Used |
| `LE_Range_Anxiety_Level_freq` | Frequency encoding of `Range_Anxiety_Level` | 0.44 | Provided a small but useful categorical frequency signal | ✅ Used |

## Version 1 — Confirmed LB 0.94559 (2026-09-08)

Importance is the reported fold-1 XGBoost importance. No isolated ablation was run, so impact records observed model contribution in the submitted V1 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 37.50 | Strongest fold-1 signal; captured the main subsidy-by-environment preference pattern | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 8.13 | High-value composite feature that ranked near the top in fold-1 importance | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 7.92 | Strong interaction signal tied to concern versus anxiety behavior | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern numeric feature | 5.47 | Direct signal that remained useful after interaction features were added | ✅ Used |
| `Environmental_Concern_Level_digit0` | Digit-derived feature from `Environmental_Concern_Level` | 1.89 | Captured local numeric structure not fully represented by the raw value | ✅ Used |
| `Annual_Income_USD_10q_bin` | 10-quantile bin of `Annual_Income_USD` | 1.13 | Helped bucket income into a non-linear range effect | ✅ Used |
| `_Income_x_Subsidy` | `Annual_Income_USD * Subsidy_Available` interaction | 0.90 | Added a smaller but useful affordability-by-subsidy interaction | ✅ Used |
| `_ecl_max` | Max-based feature derived from environmental concern bins | 0.73 | Provided a compact threshold-style summary of the ECL signal | ✅ Used |
| `Environmental_Concern_Level_5q_bin` | 5-quantile bin of `Environmental_Concern_Level` | 0.41 | Mild incremental gain from coarse quantization | ✅ Used |
