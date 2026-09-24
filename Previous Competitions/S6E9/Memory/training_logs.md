# S6E9 Training Logs

> **⚠️ RULES:**
>
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT or Delete** previous entries after submission
> 3. **ORDER** by Version Number (Highest on top), with the format block first and new version entries written below it
> 4. **Include timing** breakdown for each version
> 5. **Include all per-fold** results when available

---

## Required Format

The format block stays first; version entries are written below it.

```markdown
### Version [N] ([Description]) - YYYY-MM-DD

**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Device**: CPU/GPU
**Result**: **±X.XXXXX LB**

**Timing:**
| Stage | Time |
|-------|------|
| Total | X.X min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Strategy:** [Brief description]
**File:** `filename.py`

**Key Learning:**

> [Takeaway]

**Status:**
```

---

### Version 30 (Plain single model: V19 FE + 10-Fold XGBoost) - 2026-09-21

**Score**: **0.94639 LB** / 0.946171 OOF (10-fold) (Gap: +0.00022)
**Device**: GPU (cuda)
**Result**: **+0.000097 OOF vs V28's 5-fold figure / −0.00002 LB**

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~2.3 min |
| 10 folds × 4.0–4.7 min | 42.9 min |
| Total | **45.3 min** (I estimated 25–30 and was wrong: doubling folds costs 4.2× wall time, not 2×, because each model also trains on 12.5% more rows) |

**Fold Scores (10-fold, 66,866 validation rows each):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean ± SD |
|----|----|----|----|----|----|----|----|----|-----|-----------|
| 0.94640 | 0.94617 | 0.94570 | 0.94488 | 0.94810 | 0.94637 | 0.94572 | 0.94565 | 0.94663 | 0.94621 | 0.946182 ± 0.000799 |

**Strategy:** The plainest script in the series, deliberately: one matrix, one estimator, one loop — **no arms, no ablation harness, no blending, no full-data refit, no prior, no cross-fitting, no DeLong**. The only difference from V22's winner is `N_FOLDS = 10`, which makes every model train on 90% of the labels instead of 80% (601,798 + 9,000 orig rows per fold vs 534,932 + 8,000). Verified before shipping rather than asserted: the feature-engineering section is **byte-identical to V28's** (matching sha256), the XGBoost parameter dict diffs as **identical**, and the training call was switched to the booster API because our own audit showed V22's sklearn-wrapper path ranks the test board ~1,600 positions differently on an identical config — otherwise the fold count would not have been the only variable.
**File:** `S6E9_V30_XGB_10Fold.py`

**Key Learning:**

> **The fold count is real but smaller than advertised, and the deployed model is strictly better for it.** Pooled 10-fold OOF 0.946171 vs the 5-fold 0.946074 = **+0.000097**, about two-thirds of the externally measured +0.00015 — with the caveat that the two numbers are not the same quantity (different validation rows, 12.5% more training rows per model, and a 10-fold nested TargetEncoder). What is not ambiguous: each of the ten models that average into the submission sees 90% of the labels where the five models saw 80%, so this is the best-trained model we have built, and its OOF is the least biased estimate of it we have.
>
> **Evidence that the OOF is now a better proxy: our LB−OOF gap collapsed from a stable +0.00032/+0.00035 to +0.00022.** That shrinkage is not luck — a 10-fold OOF is closer to the ensemble-of-10 that actually predicts the test set, so less of the usual optimism is left on the table.
>
> **The divergence rule predicted this submission to the band.** Recomputed over all ten versions that share this matrix: **Spearman(mean |rank − consensus rank|, LB) = −0.770** (it was −0.667 before V29/V30). V30's divergence is 1,111, and the neighbours at 917–1,140 (V20, V23, V27) score 0.94638–0.94641. In other words: the +0.000097 of true improvement was worth about −0.00001 on the public 20%, because the fold change also moved the test ranking slightly away from our consensus — and the board bills for movement, not for merit. **A better model and a better score are decoupled at this altitude; only the model is worth optimising.**
>
> **Cost lesson for the fold-count lever, since it is the one we will ship:** 10 folds ≈ 45 min, and the same logic implies 20 folds ≈ 90 min for a fraction of the remaining gain (3→20 splits measured +0.00015 total), so fold count is now spent, not underexplored.

**Status:** ✅ Best model we have built; board-neutral by design. V23 (0.94641) remains the best *score*, V30 the best *estimate* — the two differ by one noise unit.

### Version 29 (TabM + in-run XGBoost control, dual view) - 2026-09-21

**Score**: **0.94640 LB** / 0.94607 OOF (Gap: +0.00033) — the saved model is the **control**, so this repeats V22's score exactly
**Device**: GPU (cuda), PyTorch 2.10.0+cu128, pytabkit auto-installed
**Result**: **TabM OOF 0.94564 (−0.00043 vs the control)** | blend not measured (reporting bug)

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~2.0 min |
| View A — XGBoost control, 5 folds | 10.1 min |
| View B — TabM, 5 folds | **12.7 min (2.0–3.1 min per fold)** |
| Total | **32.4 min** |

**Fold Scores (AUC, XGBoost control / TabM, rank correlation between them):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94613 / 0.94584 / 0.99619 | 0.94515 / 0.94484 / 0.99642 | 0.94717 / 0.94684 / 0.99570 | 0.94563 / 0.94529 / 0.99578 | 0.94636 / 0.94590 / 0.99514 | 0.94609 / 0.94574 |

**Strategy:** Answer the one family question never tested here — **no neural model has ever run on the V19+ artifact matrix** (V6 used 83 pre-V19 features, V7/V13 likewise, V17 ran on the corrupted digit block). TabM was chosen because it is the only non-tree family that was simultaneously strong and cheap here (V6: OOF 0.94585 in 43.9 min) and the best non-tree blend leg measured offline (+0.000093, z = +7.5). Both views trained in one script (single-model rule), with the NN fed per fold only the **top 120 columns by that fold's own freshly trained control gain** — nested, leak-free, and a ~2.6× budget cut — under a fold-0 time canary, and blended with **one linear weight chosen cross-fitted** because a non-linear combiner on this board measured CV +0.0005 against LB −0.0004.
**File:** `S6E9_V29_TabM_DualView.py`

**Key Learning:**

> **The neural family is not the missing piece.** TabM on our matrix reached OOF 0.94564 — 0.00043 behind the control and *behind* V6's 0.94585 on the older, smaller feature set — while its predictions ranked **0.9951–0.9964** with the GBM's. Weak and correlated is the worst combination for a blend: V6's leg, which was better *and* less correlated, bought +0.000093, so this one predicts ≈ +0.00005 — half of a submission gate we can already measure. That is why the lost blend number is not worth a rerun. The useful nugget: **TabM is cheap** (12.7 min for five folds vs the GBM's 10.1), so if a genuinely different *feature view* is ever wanted, an NN is nearly free to add.
>
> **A save-path bug of mine, harmless to the score but it invalidates one conclusion I drew.** V29's log printed `test prediction range: 0.5000 .. 0.7310` — that is `sigmoid(probability)`, because the save block applies `_sigmoid` unconditionally while the no-blend branch already holds probabilities. AUC and rank are invariant, so the OOF (0.946074) and LB (0.94640) are unaffected, but **`oof_v29.csv`/`sub_v29.csv` are monotone-squashed, not calibrated — never feed them to anything that reads values rather than ranks.** It also means "V28 and V29 agree" is true of their *ranks*, not their values.
>
> **Reproducibility, on ranks.** Of three saved copies of V22's estimator, V28's and V29's controls rank **identically** (mean |rank difference| 0.0 rows, OOF and test) — the pipeline is reproducible across sessions including the GPU. V22 itself, run through the sklearn wrapper instead of the booster API, ranks differently by a mean 1,600 OOF and 532 test positions **and still scored the identical LB 0.94640**: a ~0.2% rank perturbation is invisible to the board.
>
> **The real reason recent versions look worse — measured across the eight versions that share this matrix (V19, V20, V22, V23, V25, V26, V27, V28).** Their OOF spans only **0.000107** while their LB spans **0.000130** — the board adds spread rather than resolving quality — and **Spearman(OOF, LB) across them is −0.619: the best-OOF models score *worse* publicly.** V25 has the highest OOF we have ever produced (0.946094) and the lowest LB of the eight (0.94628); V19 has the *lowest* OOF (0.945987) and is joint second-best on LB. What does predict the board is **divergence from our own consensus ranking**: Spearman(mean |rank − consensus rank|, LB) = **−0.667**, and the four best-scoring submissions are precisely the four least divergent (838–1,061 positions) while the two most divergent (V25 at 2,343, V19 at 1,813) sit at the bottom and middle. Interpretation: every one of these models has essentially the same expected AUC, so a change that reshuffles the test board buys variance without bias reduction, and the public slice punishes the variance. **The board has not been measuring our quality for six versions; it has been measuring how much we disturbed the ranking.**
>
> **Process failure owned precisely:** the blend block raised `ValueError: too many values to unpack` because two call sites unpacked `paired_z`'s 4-tuple into 3 names — the identical bug fixed in V28 the day before. The earlier "unit test" caught nothing because it called `paired_z` with a *correct* 4-name unpack instead of executing the script's real line: **testing a paraphrase of the call site is not testing the call site.** The guard held — outputs saved, control result intact, verdict printed — but a guarded crash that silently downgrades the deliverable still cost the run's whole purpose. Fix: grep every call site of a multi-return helper, not the helper.

**Status:** ❌ Failed to add value (family axis closed as a source of gain) — ✅ guard worked, zero artifacts lost, control reproduced V22's LB exactly

### Version 28 (XGBoost Bin Resolution: max_bin 1024 → 16384) - 2026-09-21

**Score**: **0.94640 LB** / 0.94607 OOF (Gap: +0.00033) — CV-only probe, deliberately not promoted
**Device**: GPU (cuda)
**Result**: **−0.000002 OOF vs the in-run control (z = −0.40, tie)**

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~1.5 min |
| a0 control (max_bin 1024), 5 folds | 10.7 min |
| a1 raised (max_bin 16384), 5 folds | 45.3 min (**4.2× the control**) |
| Total | **66.5 min** |

**Fold Scores (winner a0):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94613 | 0.94515 | 0.94717 | 0.94563 | 0.94636 | 0.94609 |

a1's fold AUCs: 0.94612 / 0.94515 / 0.94716 / 0.94564 / 0.94636 — identical to five decimals on two folds.

**Strategy:** Change exactly one parameter on the real 312-column pipeline, with an in-run control that must reproduce V22 (it did: 0.94607 vs stored 0.946076, the sixth consecutive validation of the shared-build harness). Income holds 14,667 distinct values across the pool and is the *only* column of 13 above a 1024-bin cap, so 16384 is not "more bins" — it is unconstrained binning, the last point on the axis. A first run of this version had tested 4096 (delta −0.00000, z = −0.21) and the rerun went to full resolution so the answer could not be "you stopped short".
**File:** `S6E9_V28_XGB_BinResolution.py`

**Key Learning:**

> **The resolution axis is closed by construction, and the community's +0.00201 was ours years ago.** 16× the bins moved nothing. An integrity guard settled the "was the parameter even applied?" question from the fitted trees themselves: income-derived columns used 6,456 distinct thresholds at max_bin 1024 vs **7,972 at 16384** (e.g. `TE_Annual_Income_USD_cat_auto` 318 → 538 cuts), and the arms stopped at different tree counts on every fold — so the tie is a real null, not a silently ignored setting. That matches every additive proxy measured offline (per-value income lookup stacking: +0.000004…+0.000010; raw income significantly *worse* at z = −3.2; the model's per-bin income tracking already at the split-half reliability ceiling). The thread's +0.00201 was raised on a 0.94172 baseline that lacked both the fixed digit block and the exact-value TE stack — **we banked that lever in V19**, which is also why the raised cap and the digits measured as the *same* +0.00201 there. Two secondary findings: the two bin settings' predictions correlate 0.99992, so bin resolution is not a diversity source for blending (blend +0.00000, z = +1.20); and reproducing V22's estimator reproduced its LB **to the last digit** (0.94640), which is the cleanest evidence yet that LB differences of ±0.00001-0.00002 between near-identical models are model difference, not board randomness. Also: 4.2× cost for zero gain retires any further sweep of this knob — keep 1024 permanently. Process note: the first run's 31 min of GPU work was lost because a `paired_z` unpacking bug threw *after* training but *before* saving; outputs are now written before any post-hoc diagnostic, and the diagnostics are guarded.

**Status:** ⚠️ CV-only null by design — axis closed, no submission value, `max_bin` stays 1024

### Version 27 (XGBoost Structural Ablation: original-row weight + pairwise loss) - 2026-09-20

**Score**: **0.94638 LB** / 0.94608 OOF (Gap: +0.00030)
**Device**: GPU (cuda)
**Result**: **-0.00003 LB vs V23 — probe only, not promoted**

**Timing:**
| Stage | Time |
|-------|------|
| Stage A fold 1 (5 arms) | 13.9 min |
| Stage A fold 2 | 13.2 min |
| Stage A fold 3 | 10.7 min |
| Stage A fold 4 | 11.2 min |
| Stage A fold 5 | 17.7 min (split total 66.7 min) |
| Stage B | skipped (no arm cleared z > 3) |
| Total | 69.6 min |

**Fold Scores (winner a1, rs=42):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94613 | 0.94514 | 0.94719 | 0.94563 | 0.94635 | 0.94609 |

**Strategy:** Third use of the ablation harness, aimed at the two structural assumptions every version since V1 has shared. a0 = V20/V23's exact config as the control. a1 removes the original 10k rows from the per-fold concat. a2 keeps them at sample weight 10. a3 swaps `binary:logistic` for `rank:pairwise` (rows shuffled into contiguous random groups of 64, so within-group pairs are a uniform subsample of the global positive-negative pairs AUC averages over) and early-stops on `auc`, which XGBoost accepts for ranking objectives. a4 = a3 with the original rows replicated 10x, because this XGBoost build rejects `weight` together with `set_group` and duplication is the exact pairwise-loss equivalent of up-weighting.
**File:** `S6E9_V27_XGB_StructuralAblation.py`

**Key Learning:**
> a0 landed on 0.94605 against V20's stored 0.946060, so the control held again. Two clean answers. **(1) The original 10k rows are not earning their concat: dropping them gives +0.00002 at z = +2.66** — a tie, but a tie in the *right* direction, with 1.47% of the training matrix removed and no loss; up-weighting them ×10 goes the other way (-0.00003, z = -2.30). Every original-data claim we have made since V1 is really about the pool/original *frequency* features, not about training on those rows. **(2) The pairwise-loss theory is dead: `rank:pairwise` loses 0.000329 at z = -42.56**, and it is unstable across folds (0.94420 to 0.94624, converging anywhere between 416 and 3,827 trees). Replicating the original rows *inside* the ranking objective recovers most of that (-0.00069) but still loses badly. Pointwise logloss is already close to optimal for the ranking here — the model's calibration is its ranking, and a surrogate that ignores absolute probabilities throws away real signal. Winner a1's OOF 0.94608 equals V22's best and its gap to LB (+0.00030) sits back inside the normal band, unlike V25's.

**Status:** ⚠️ Partial (probe answered both questions; no submission candidate)

---

### Version 26 (LightGBM ExtraTrees-Bias Probe) - 2026-09-20

**Score**: **0.94637 LB** / 0.94608 OOF (Gap: +0.00029)
**Device**: CPU (LightGBM 4.6)
**Result**: **-0.00004 LB vs V23 — first arm ever to clear our own z > 3 gate**

**Timing:**
| Stage | Time |
|-------|------|
| Stage A fold 1 (4 arms) | 39.2 min |
| Stage A fold 2 | 43.2 min |
| Stage A fold 3 | 39.4 min |
| Stage A fold 4 | 40.7 min |
| Stage A fold 5 | 43.7 min (split total 206.3 min) |
| Stage B (3 arms, rs=7) | 177.9 min |
| Total | 387.6 min (6.5 h) |

**Fold Scores (winner a3, rs=42):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94611 | 0.94517 | 0.94721 | 0.94564 | 0.94637 | 0.94610 |

**Strategy:** Test the one inductive bias never used on this matrix: randomised split thresholds. Rationale — the pool is 668k rows rejection-sampled from 10k, so exact input values recur thousands of times and the label is close to a lookup on them; a greedy depth-3/4 tree can only approximate that with shared thresholds, while random thresholds over 1,024 bins in fully-grown trees can memorise value boundaries instead. a0 = V19's LightGBM params verbatim as the control, a1 = a0 + `extra_trees=True` (+ `split_histogram_sampling`, which is how LightGBM 4.x implements the bias), a2 = a1 + RF-style lookup capacity (unlimited depth, 128 leaves, `feature_fraction_bynode` 0.3, lr 0.05), a3 = a1 + V22's winning "wide, weak, many" direction (depth 3, 8 leaves, `min_child_samples` 50, colsample 0.85, lr 0.01). Stage B ran *because* arms cleared the gate.
**File:** `S6E9_V26_LGBM_ExtraTreesProbe.py`

**Key Learning:**
> **First gate-cleared improvement since the artifact features.** a1 (ExtraTrees bias alone) beat the in-run greedy control by +0.00007 at z = +3.58, and a3 (bias + shallow/wide/weak) by **+0.00010 at z = +5.22 on rs=42 and +0.00008 at z = +4.11 on rs=7** — reproduced on both splits, which nothing has done since V19. The lookup arm failed hard (a2 -0.00023, z = -8.00), so random thresholds help as *regularisation of a shallow model*, not as memorisation: my stated mechanism was wrong even though the direction paid. a0 also reproduced V19 (0.94599 vs stored 0.945987). **The catch: this beat LightGBM, not XGBoost.** a3's 0.946085 only ties V22/V27's 0.946076-0.946078 (z = +1.60 vs V23) and sits below V25's 0.946094. What it really says is that the 0.00007 LightGBM→XGBoost gap was a *split-bias* gap, not an information gap — once LightGBM is allowed random thresholds it catches up. That makes the ET direction the most promising unexplored tuning surface we have (one config of one grid already gained +0.00010), at a brutal cost: 6.5 h of CPU because a3 needs 6,000-8,200 trees per fold.

**Status:** ✅ Good (best reproducible gain since V19; LB did not move, see the noise-floor note in daily_log 20-09-2026)

---

### Version 25 (XGBoost Factorial Ablation: cross keys / recipe prior / gain pruning) - 2026-09-19

**Score**: **0.94628 LB** / 0.94609 OOF (Gap: +0.00019 — our best OOF, and a broken OOF→LB relation)
**Device**: GPU (cuda)
**Result**: **-0.00012 LB vs V22 — submission reverted to the V23 line**

**Timing:**
| Stage | Time |
|-------|------|
| Stage A fold 1 (5 arms) | 9.7 min |
| Stage A fold 2 | 10.9 min |
| Stage A fold 3 | 10.2 min |
| Stage A fold 4 | 10.7 min |
| Stage A fold 5 | 11.6 min (split total 53.2 min) |
| Stage B (3 arms, rs=7) | 33.8 min |
| Total | 90.2 min |

**Fold Scores (winner a2, rs=42):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94612 | 0.94512 | 0.94721 | 0.94570 | 0.94635 | 0.94610 |

**Strategy:** Instead of bundling three ideas into one model, run them as a **factorial ablation** in one script: one feature build per fold shared by all arms, an in-run control equal to V22's winning config, and four arms that each change exactly one factor. a1 = V21's six cross keys as a PURE addition (312 base-block + 54 cross-block columns; the cross block is appended last and excluded from the redundancy scan, so a0's matrix is V22's); a2 = per-row `base_margin` from a logistic fitted on the original 10k rows only, per fold, clipped to ±5; a3 = in-run gain pruning to 94 features via a 600-tree lr-0.05 probe on that fold's own training rows; a4 = all three. Stage A on rs=42, top-3 + control re-run on rs=7, winner by two-split mean, submission averaged over both splits. Every arm judged against the control by a paired DeLong computed from this run's own OOF vectors. 90.2 min, faster than V22 despite more arms.
**File:** `S6E9_V25_XGB_FactorialAblation.py`

**Key Learning:**
> The harness worked and the science is the useful part. **a0 reproduced V22 exactly** (0.94607 vs V22's stored 0.946076, 6e-6 apart), which validates the shared-build design. Ranking: a2 0.94609, a0 0.94607, a3 0.94603, a4 0.94602, a1 0.94601. In-run paired DeLong vs control: **a1 -0.00006 at z=-3.98 (reject)**, a3 -0.00004 at z=-2.77, a4 -0.00005, **a2 +0.00002 at z=+1.29** (rs=7: +0.00004 at z=+2.45). So V21's crosses are now **cleanly acquitted of the blame and rejected on their own merits** — the last open feature door is shut — and the reference recipe's 94-feature pruning does not transfer to our matrix. Only the recipe prior shows a pulse, and only at tie strength.
>
> Then the discipline test: **I promoted a tie to a submission and it cost 0.00012 LB.** Offline forensics on the saved predictions show this was not a code error. (1) The submission file is clean: 286,541 unique of 286,571 values, no NaN, range 3.9e-6 to 0.9996. (2) The prior is *symmetric* between train and test — clipped on 20.82% of train vs 20.72% of test rows, entirely on the -5 side (0% at +5), and V25's OOF vs test prediction distributions match to four decimals (logit mean -3.757 vs -3.738, sd 3.243 vs 3.223), so there is no test-side extrapolation failure. (3) The real asymmetry is *magnitude of change*: mean |rank difference| vs V22's submission is 2,725 rows with 31% of the board moving more than 1% of ranks, versus 1,598/16% for V23 and 1,495/14% for V20 — a tie-sized gain that reshuffled twice as much as V23's blend, and V21 (the other version that dropped ~0.00011) has an almost identical profile at 2,874/36%. (4) Where the +0.000018 came from is the most diagnostic fact: bucketed by V22's prediction decile, a2's movement is +0.0324 / -0.0151 / -0.0162 in the bottom three deciles and flat to ±0.0012 above them — and those deciles hold 174 of 116,779 positives, i.e. **0.15% of the AUC's pair mass**. The prior buys nothing where the metric lives; it reorders where the metric is blind. (5) Finally, public LB cannot order our submissions at this scale: V19's OOF is 0.000107 *worse* than V25's yet its LB is 0.00011 *better*. **New rule adopted: a submission slot only for an arm that clears z > 3 on OOF, or for an inference-side variant of V23; ties stay CV experiments.**

**Status:** ⚠️ Partial (best OOF 0.94609; LB regressed because a tie was submitted)

---

### Version 24 (CatBoost depth=6 on the V19/V20 Artifact Matrix) - 2026-09-19

**Score**: **0.94615 LB** / 0.94593 OOF (Gap: +0.00022)
**Device**: GPU (CatBoost, task_type=GPU)
**Result**: **-0.00024 LB vs V20 — REJECTED**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 134 s |
| Fold 2 | 126 s |
| Fold 3 | 132 s |
| Fold 4 | 135 s |
| Fold 5 | 140 s |
| Total | 13.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94592 | 0.94497 | 0.94707 | 0.94552 | 0.94623 | 0.94594 |

**Strategy:** Third isolation test — swap the estimator family only. V19's artifact matrix (fixed digits, 12 lift/novelty columns, 2 structural flags, V10 bigrams + groupby deviations, 180 base + 198 Triple TE) trained with CatBoost `depth=6`, `lr=0.03`, `l2_leaf_reg=3.0`, `min_data_in_leaf=20`, Bernoulli bootstrap 0.8, `random_strength=1.0`, `max_bin=1024`, Iter/500 early stop, `use_best_model=True`. Best iterations 1243-1549. Same KFold(5, shuffle=True, rs=42) and per-fold original-data concat.
**File:** `S6E9_V24_CatBoost_Artifacts.py`

**Key Learning:**
> OOF 0.94593 vs V20's 0.94606 (-0.00013) and LB 0.94615 vs 0.94639 — CatBoost is still behind XGBoost on this matrix even after the artifact features, continuing the V4/V8 ordering. It confirms the family axis is closed: XGBoost depth-4 > LightGBM > CatBoost > TabM/RealMLP on identical features. The feature story replicated exactly though — `TE_lift_trigram_cat_10` (8.20) and `_auto` (8.18) took the top two importance slots, so the lift trigram is the dominant signal in every family, not a LightGBM/XGBoost artifact. Fastest full run to date (13.7 min vs V20's 20.6 min), and CatBoost's shrinkage stopped at ~1.5k iterations against XGBoost's ~4k, so it is a cheap candidate generator, not a score lever.

**Status:** ❌ Failed

---

### Version 23 (XGBoost depth=4 + Full-Data Refit Test Blend) - 2026-09-19

**Score**: **0.94641 LB** / 0.94606 OOF (Gap: +0.00035)
**Device**: GPU (cuda)
**Result**: **+0.00002 LB vs V20 — NEW BEST LB**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 200 s |
| Fold 2 | 219 s |
| Fold 3 | 190 s |
| Fold 4 | 189 s |
| Fold 5 | 223 s |
| CV subtotal | 17.0 min |
| Full-data refit | 2.2 min |
| Total | 23.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94612 | 0.94513 | 0.94715 | 0.94564 | 0.94634 | 0.94608 |

**Strategy:** V20's CV loop byte-for-byte (same features, same params, same rs=42 split), then a `[3b]` inference stage: refit one model on 100% of train + original (678,665 x 312 vs the folds' 544,932) with `n_estimators` fixed at the mean best iteration (4112; iterations [4230, 4501, 3375, 3470, 4984]) and no early-stopping set, then blend `test_probs = 0.50 x fold-average + 0.50 x refit`. OOF remains purely out-of-fold, so the reported CV is the V20 number by construction.
**File:** `S6E9_V23_XGB_FullDataRefit.py`

**Key Learning:**
> +0.00002 LB with the OOF untouched — the first movement we have produced from the inference side, and it came with a diagnostic that says the refit is only mildly decorrelated from the fold average (Pearson 0.99959, Spearman 0.99946, mean |rank difference| 1,806 of 286,571 rows). Two things follow: 25% more labels per tree is worth something even at fixed depth, and the surviving 0.0004 of disagreement between the two predictors is where inference-side variance lives, so weight and iteration-count choices on that axis are still open. Fold importances matched V20 within rounding (`TE_lift_trigram_cat_auto` 0.14488), confirming the CV path really was unchanged.

**Status:** 🏆 Best

---

### Version 22 (XGBoost Two-Stage Estimator Search on Our Matrix) - 2026-09-19

**Score**: **0.94640 LB** / 0.94608 OOF (Gap: +0.00032)
**Device**: GPU (cuda)
**Result**: **+0.00001 LB vs V20 (new best OOF for a single model)**

**Timing:**
| Stage | Time |
|-------|------|
| Stage A (6 configs x 5 folds, rs=42) | 75.9 min |
| Stage B (3 configs x 5 folds, rs=7) | 37.9 min |
| Total | 116.8 min |

**Fold Scores (winner c1, rs=42):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94611 | 0.94517 | 0.94717 | 0.94563 | 0.94637 | 0.94609 |

**Strategy:** V20 used a stranger's tuned recipe, so the estimator was never tuned on our own matrix. Six configs sharing one build-per-fold (c0 = V20 control, c1 depth 3 / 8 leaves / gamma 1.0 / colsample 0.85, c2 deeper + heavy leaf reg, c3 sparse cols + high gamma, c4 low min_child_weight, c5 tiny lr near-depthless lossguide), Stage A on the canonical rs=42 split, top-3 re-run on rs=7, winner by two-split mean. Submission averaged the winner's test predictions across both splits; OOF is the rs=42 winner.

**Key Learning:**
> Stage A ranking: c1 0.94608, c0 0.94606, c4 0.94605, c2 0.94593, c3 0.94576, c5 0.94546. The winner was confirmed on a second split (rs=7: c1 0.94608 / c0 0.94607 / c4 0.94605), so the +0.00002 over the in-run control is real but tiny — and the in-run control reproduced V20's OOF exactly, which validates the harness. The useful negative results are directional: deeper/heavier-regularised trees and tiny learning rates are clearly worse (-0.00013 / -0.00060), while depth 3 with 3x the columns and 5,000-6,600 trees is marginally better, i.e. this matrix wants wide, weak, many learners. c1 also redistributed gain (in its rs=7 fold-1 table the lift-trigram auto+10 pair alone took 0.419 of gain versus 0.252 for c0 on rs=42, and raw `Environmental_Concern_Level` entered at #3), which is the V21 lesson reversed: a shallower, column-rich model leans even harder on the pre-computed artifact crosses.

**Status:** ✅ Good

---

### Version 21 (XGBoost + Explicit Cross Keys, digits/flags removed) - 2026-09-19

**Score**: **0.94629 LB** / 0.94597 OOF (Gap: +0.00032)
**Device**: GPU (cuda)
**Result**: **-0.00010 LB vs V20 — REJECTED by the accept gate**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 186 s |
| Fold 2 | 181 s |
| Fold 3 | 169 s |
| Fold 4 | 176 s |
| Fold 5 | 183 s |
| Total | 16.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94607 | 0.94505 | 0.94710 | 0.94548 | 0.94624 | 0.94599 |

**Strategy:** V20's model unchanged, feature budget reallocated. Added six explicit cross keys (charging-total x Home_Charging, City x Home_Charging, ECL x Home_Charging, ECL x City, income-band x ECL x Subsidy, ECL x Subsidy x commute with the 5.0 km cluster isolated), each Triple-TE'd like the V10 bigrams plus its own target-free generator-lift column. Removed the whole digit block and four income-anomaly flags that had earned <= 0.08% gain in V20. Fixed bin edges in CFG so train/test/orig bin identically. 159 base + 189 TE = 348 features (V20: 180 + 198 = 378).
**File:** `S6E9_V21_XGB_CrossKeys.py`

**Key Learning:**
> Paired DeLong vs V20: delta -0.00009, z = -5.01 — a significant regression, and LB agreed (0.94629 vs 0.94639). The addition worked and the subtraction broke it: `cx_inc_x_ecl_x_sub` and its lift took five of the top eight gain slots, proving the crosses carry real signal. But dropping the digit block and flags shrank the space (only 27 columns dropped as redundant vs 149 in V20) and the gain distribution collapsed — `_ECL_x_Subsidy` seized 0.5899 of total gain (5.4x its V20 share) while the lift-trigram family that had been #1 in V20 fell from ~35% combined to ~6.7%. Bundling an addition with a removal made the failure unattributable beyond that; lesson: one change per version, and a shallow lossguide model needs a wide candidate pool to keep any single feature from monopolising splits.

**Status:** ❌ Failed

---

### Version 20 (XGBoost depth=4 lossguide + V19 Artifact Features) - 2026-09-19

**Score**: **0.94639 LB** / 0.94606 OOF (Gap: +0.00033)
**Device**: GPU (cuda)
**Result**: **±0.00000 LB vs V19 (new best OOF for a single model)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 20.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94612 | 0.94513 | 0.94715 | 0.94564 | 0.94634 | 0.94608 |

**Strategy:** One variable changed from V19 — model family. V19's feature matrix is reused byte-for-byte (fixed integer digit extraction, 12 target-free generator-lift/novelty columns, 2 structural flags, V10 bigrams + groupby deviations, Triple TE on 66 columns = 180 base + 198 TE), trained with najiama's tuned XGBoost recipe: `max_depth=4`, `grow_policy=lossguide`, `max_leaves=16`, `gamma=3.673`, `min_child_weight=4.532`, `subsample=0.740`, `colsample_bytree=0.570`, `alpha=0.752`, `lambda=0.619`, `lr=0.01`, ES=500, converging at 3375-4984 trees. 5-fold `KFold(shuffle=True, rs=42)` kept identical to V19 so the OOF stays comparable.
**File:** `S6E9_V20_XGB_ArtifactFeatures.py`

**Key Learning:**

> Gave our first statistically significant OOF gain: paired DeLong vs V19 Δ +0.00007 at SE 0.00002 (z = +4.76), and vs V10 Δ +0.00009 (z = +5.25); statistically tied with V14's long-standing best OOF (Δ -0.00002, z = -0.83). LB still landed on 0.94639 exactly — public LB covers 20% of test and cannot resolve a 0.00007 change, so OOF significance and LB movement are now decoupled. `TE_lift_trigram_cat_auto` became the single most important feature by gain (0.1449) versus #4 under LightGBM, confirming that a shallower model leans harder on pre-computed artifact crosses. Structural flags and digits were near-unused here (flat, tiny gains), matching the reference notebook's decision to prune them. Test predictions correlate 0.99982 with V19's, and their equal-weight average scores 0.94606 — no better than V20 alone. Also 40% faster than V19 (20.6 min vs 33.7 min).

**Status:** ✅ Good

---

### Version 19 (LightGBM + Generator-Artifact Features) - 2026-09-19

**Score**: **0.94639 LB** / 0.94599 OOF (Gap: +0.00040)
**Device**: CPU (LightGBM)
**Result**: **+0.00003 LB over V10 — new best**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94603 | 0.94507 | 0.94710 | 0.94555 | 0.94627 | 0.94600 |

**Strategy:** V10 Modified pipeline plus three self-contained upgrades: (1) digit extraction fixed with integer divmods on `round(col*1e4)` — the `// (10**k)` negative-k float floor-div bug had corrupted 89.63% of commute tenths in all 18 prior versions; (2) 12 target-free generator-lift features (train+test pool freq ÷ original freq) on exact income, 100-dollar income band, integer commute, 6 categoricals, and the ECL×Subsidy×Anxiety trigram, plus a `novel_inc` flag; (3) two structural flags from the decoded recipe (`_below_buy_bound` at 41,667, `_dead_zone_exact` 31,004-41,970). 180 base + 198 Triple TE = 378 features, 5-fold, original-data concatenation, V10 LightGBM params unchanged.
**File:** `S6E9_V19_LGBM_ArtifactFeatures.py`

**Key Learning:**

> First real gain in 9 versions. `TE_lift_trigram_cat_auto` entered at #4 overall (531) with 11 of the top-15 artifact features being lift/novelty columns, confirming the generator's frequency shaping leaks target signal that no model had access to before. The OOF moved only +0.00002 (0.94599 vs 0.94597) while LB moved +0.00003 — fold std held at 0.00069, so the win is small but consistent across both metrics.

**Status:** 🏆 Best

---

### Version 18 (Hill Climber Ensemble) - 2026-09-19

**Score**: **0.94635 LB** / 0.94623 OOF (Gap: +0.00012)
**Device**: CPU (hill climbing over saved OOFs)
**Result**: **-0.00001 LB vs V10 best**

**Timing:**
| Stage | Time |
|-------|------|
| Total | ~10 min (300k-subsampled search; full-data grid search is much slower) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| N/A | N/A | N/A | N/A | N/A | 0.94623 |

**Strategy:** Forward-stepwise hill climbing (weight step 0.05) over all 17 saved OOF predictions, plus simple/rank/logit averages and a Ridge meta-learner for comparison. Selected weights: V14=0.35, V3=0.28, V11=0.15, V6=0.13, V2=0.05, V8=0.05. OOF was fit directly by the hill climber, so the 0.94623 OOF is optimistically biased.
**File:** `S6E9_V18_HillClimber.py`

**Key Learning:**

> Hill climbing pushed OOF to 0.94623 (best single was 0.94608) but LB only reached 0.94635 — 0.00001 below V10's 0.94636. Members correlate 0.996–0.999, so honest combination gains are ≤0.000066; the OOF gain was weight-fitting noise, not real signal. Ensembling within this model pool is exhausted.

**Status:** ⚠️ Partial

---

### Version 15 (KNN Exact-Match + KDTree) - 2026-09-10

**Score**: **0.91256 LB** / 0.91359 OOF (Gap: -0.00103)
**Device**: CPU (KNN with KDTree)
**Result**: **±0.00086 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.91462 | 0.91293 | 0.91338 | 0.91248 | 0.91456 | 0.91359 |

**Strategy:** 5-fold CPU KNN using a 24-feature encoded representation, exact-match lookup when available, and k=10 KDTree neighbor averaging as fallback. Validation and test exact-match rates were both 0%.
**File:** `S6E9_V15_KNN.py`

**Key Learning:**

> KNN achieved only 0.91256 LB and 0.91359 OOF. With no exact matches and a -0.00103 LB–OOF gap, this approach was not competitive for the dataset.

**Status:** ❌ Failed

---

### Version 17 (RealMLP) - 2026-09-11

**Score**: **0.94612 LB** / 0.94582 OOF (Gap: +0.00030)
**Device**: CUDA (RealMLP/PyTorch)
**Result**: **±0.00070 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 97.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94591 | 0.94490 | 0.94699 | 0.94553 | 0.94624 | 0.94591 |

**Strategy:** 5-fold CUDA RealMLP using PBLD embeddings, an 8-model ensemble, three hidden layers of 256 units, EMA, label smoothing, and three epochs. The model used the V14 full feature pipeline, original-data concatenation, Triple TE, and 293 expanded model inputs.
**File:** `S6E9_V17_RealMLP.py`

**Key Learning:**

> RealMLP tied V9 at 0.94612 LB but required 97.3 minutes and remained below V14’s 0.94630. The full engineered representation was viable, but the neural approach did not beat the tree models.

**Status:** ✅ Good

---

### Version 16 (XGBoost Forensic-Targeted Features) - 2026-09-11

**Score**: **0.94625 LB** / 0.94595 OOF (Gap: +0.00030)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00068 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94597 | 0.94505 | 0.94709 | 0.94553 | 0.94619 | 0.94597 |

**Strategy:** 5-fold GPU XGBoost depth 3 based on V14 without pseudo-labels, adding five forensic-targeted features: subsidy × home charging, recipe × subsidy/income, and fold-safe buyer-centroid distances within ECL subgroups. Final training used 164 features.
**File:** `S6E9_V16_Forensic_Targeted.py`

**Key Learning:**

> V16 reached 0.94625 LB and 0.94595 OOF, below V14’s 0.94630 LB. The strongest forensic signal was `TE_bigram_Sub_HomeCharging`, but all forensic features had small overall fold-1 importance.

**Status:** ✅ Good

---

### Version 14 (Pseudo-Labeling XGBoost Depth 3) - 2026-09-10

**Score**: **0.94630 LB** / 0.94608 OOF (Gap: +0.00022)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00068 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94609 | 0.94521 | 0.94721 | 0.94560 | 0.94633 | 0.94609 |

**Strategy:** 5-fold GPU XGBoost depth 3 using V10 teacher predictions to add 150,859 high-confidence pseudo-labeled test rows: 999 positive and 149,860 negative labels at weight 0.5. The student used V12-style features, original-data concatenation, and 159 final columns.
**File:** `S6E9_V14_PseudoLabel_XGB.py`

**Key Learning:**

> Pseudo-labeling improved the V12 LB from 0.94629 to 0.94630 and reached third place overall, but the gain was only +0.00001. The LB–OOF gap was +0.00022.

**Status:** ✅ Good

---

### Version 13 (DCN-V2) - 2026-09-09

**Score**: **0.94568 LB** / 0.94490 OOF (Gap: +0.00078)
**Device**: CUDA (PyTorch)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94503 | 0.94415 | 0.94612 | 0.94469 | 0.94523 | 0.94504 |

**Strategy:** 5-fold CUDA DCN-V2 with 4 cross layers, rank 64, 4 experts, and the V6 evidence-based 79-feature selection. Features combined V3 inputs with V10/V12 proven targeted interactions; the final input contained 18 categorical and 61 numerical features.
**File:** `S6E9_V13_DCN_V2.py`

**Key Learning:**

> V13 tied V7 at 0.94568 LB but had a much larger +0.00078 LB–OOF gap. The DCN-V2 cross architecture was fast but did not outperform the tree-based or TabM approaches.

**Status:** ✅ Good

---

### Version 12 (XGBoost Depth 3 Deotte-style) - 2026-09-09

**Score**: **0.94629 LB** / 0.94598 OOF (Gap: +0.00031)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00069 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94600 | 0.94507 | 0.94713 | 0.94554 | 0.94625 | 0.94600 |

**Strategy:** 5-fold stratified GPU XGBoost with depth 3, `lr=0.0025`, up to 50,000 estimators, V3 full FE, V10/V11 proven targeted features, and inner K-fold target encoding on 67 columns. All 10 discussion patterns were implemented except the recipe score, which was excluded after V11’s cannibalization finding. Final feature count was 237.
**File:** `S6E9_V12_XGB_Depth3_Deotte.py`

**Key Learning:**

> V12 reached 0.94629 LB, tying V11 for third overall. The 3-way `Subsidy × ECL × Range Anxiety` target encoding was the strongest discussion-derived feature, but the shallow model remained below V10 and V3.

**Status:** ✅ Good

---

### Version 11 (LightGBM Deep Analysis Features) - 2026-09-09

**Score**: **0.94629 LB** / 0.94599 OOF (Gap: +0.00030)
**Device**: CPU only (LightGBM)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94601 | 0.94516 | 0.94705 | 0.94555 | 0.94626 | 0.94601 |

**Strategy:** 5-fold stratified CPU LightGBM extending corrected V10 with six deep-analysis features: `_never_buy_flag`, `_always_buy_flag`, `_income_round00`, `trigram_Sub_ECL_RA`, `_recipe_residual`, and `bigram_income_band_x_Subsidy`. The model used 173 base features plus 201 Triple TE features, 374 total.
**File:** `S6E9_V11_LGBM_DeepAnalysis.py`

**Key Learning:**

> V11 reached 0.94629 LB and third place overall. `_recipe_residual` was the strongest fold-1 signal, while trigram target encodings added useful interaction structure; V11 remained 0.00007 below V10.

**Status:** ✅ Good

---

### Version 10 (LightGBM Targeted Features) - 2026-09-09

**Score**: **0.94636 LB** / 0.94597 OOF (Gap: +0.00039)
**Device**: CPU only (LightGBM)
**Result**: **±0.00067 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 36.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94601 | 0.94506 | 0.94705 | 0.94554 | 0.94627 | 0.94598 |

**Strategy:** 5-fold stratified CPU LightGBM based on V3, preserving the core feature set while adding the missing `ECL_bin × RangeAnxiety` and `ECL_bin × Subsidy` bigrams plus 12 selective groupby deviation features. The final model used 168 base features and 195 Triple TE features, 363 total.
**File:** `S6E9_V10_LGBM_TargetedFeatures.py`

**Key Learning:**

> The corrected targeted-feature V10 reached 0.94636 LB, a new best and +0.00002 above V3. Targeted bigram encodings and income/ECL-bin groupby deviations provided useful signal without the larger 479-feature expansion.

**Status:** 🏆 Best

---

### Version 9 (XGBoost Full Feature Engineering) - 2026-09-09

**Score**: **0.94612 LB** / 0.94584 OOF (Gap: +0.00028)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 21.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94583 | 0.94496 | 0.94689 | 0.94546 | 0.94614 | 0.94586 |

**Strategy:** 5-fold stratified GPU XGBoost using V3’s full 344-feature pipeline with Triple TE, smooth keys, magic flags, original target means, and frequency features, combined with V2’s proven weak-regularization parameters: `lr=0.005`, depth 7, and `max_bin=1024`.
**File:** `S6E9_V9_XGB_FullFE.py`

**Key Learning:**

> V9 improved V2 from 0.94569 to 0.94612 LB and reached second place overall. The categorical and raw environmental-concern × subsidy interactions dominated fold-1 importance, while the LB–OOF gap remained a reasonable +0.00028.

**Status:** ✅ Good

---

### Version 8 (CatBoost Ordered Boosting) - 2026-09-09

**Score**: **0.94603 LB** / 0.94583 OOF (Gap: +0.00020)
**Device**: GPU (CatBoost)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94579 | 0.94493 | 0.94691 | 0.94550 | 0.94606 | 0.94584 |

**Strategy:** 5-fold stratified GPU CatBoost with `boosting_type='Ordered'`, original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, native CatBoost categorical features, multi-scale smooth keys, synthetic-artifact flags, and feature selection. Hyperparameters used `lr=0.015`, depth 5, `l2_leaf_reg=5.0`, Bernoulli bootstrap, and balanced weighting.
**File:** `S6E9_V8_CatBoost_Ordered.py`

**Key Learning:**

> Ordered boosting improved the CatBoost LB from V4’s 0.94590 to 0.94603 and entered the leaderboard Top 5. `_ECL_x_Subsidy` remained the strongest fold-1 feature at 15.32%, but V8 stayed 0.00031 below V3.

**Status:** ✅ Good

---

### Version 7 (FT-Transformer Full Features) - 2026-09-09

**Score**: **0.94568 LB** / 0.94545 OOF (Gap: +0.00023)
**Device**: CUDA (FT-Transformer)
**Result**: **±0.00062 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 159.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94562 | 0.94474 | 0.94653 | 0.94507 | 0.94574 | 0.94554 |

**Strategy:** 5-fold KFold CUDA FT-Transformer with the same evidence-based feature selection used in V6, original-data concatenation per fold, per-fold auto-smoothed target encoding, and 79 selected features consisting of 35 numerical and 34 low-cardinality categorical inputs. The model used `d_block=128`, `n_blocks=2`, and `heads=8`.
**File:** `S6E9_V7_FTTransformer_FullFeatures.py`

**Key Learning:**

> FT-Transformer reached 0.94568 LB with a close +0.00023 LB–OOF gap, but did not surpass V6 TabM or V3 LightGBM. Training took 159.1 minutes, making it substantially slower than the competing baselines.

**Status:** ✅ Good

---

### Version 6 (TabM Feature Selection) - 2026-09-08

**Score**: **0.94606 LB** / 0.94585 OOF (Gap: +0.00021)
**Device**: CUDA (TabM)
**Result**: **±0.00067 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 43.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94592 | 0.94500 | 0.94700 | 0.94547 | 0.94613 | 0.94590 |

**Strategy:** 5-fold KFold CUDA TabM with evidence-based feature selection from V5 LogisticRegression and V3 LightGBM importance, original-data concatenation per fold, per-fold target encoding from 14 source columns, and 83 selected numeric features from 344 raw features. TabM used `tabm_k=16`, `d_block=256`, and PWL embeddings.
**File:** `S6E9_V6_TabM_FeatureSelection.py`

**Key Learning:**

> The compact 83-feature evidence-based subset achieved 0.94606 LB and surpassed V4/V5, but remained 0.00028 below the V3 LightGBM best. The model’s LB–OOF gap was +0.00021.

**Status:** ✅ Good

---

### Version 5 (LogisticRegression CPU Triple TE) - 2026-09-08

**Score**: **0.94478 LB** / 0.94468 OOF (Gap: +0.00010)
**Device**: CPU only (no GPU/cuDF)
**Result**: **±0.00064 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94474 | 0.94378 | 0.94570 | 0.94431 | 0.94488 | 0.94468 |

**Strategy:** 5-fold KFold CPU LogisticRegression with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, per-fold StandardScaler, frequency encoding, engineered features, and feature selection. Final model used 155 base features plus 189 Triple TE features, with string columns removed before training.
**File:** `Previously trained Files/Archieve/S6E9_V5_LR_Baseline.py`

**Key Learning:**

> StandardScaler enabled consistent convergence in 227–249 iterations and the LB–OOF gap was only +0.00010, but the linear baseline remained below the tree-based models. `Subsidy_Available_fe` was the strongest fold-1 absolute coefficient.

**Status:** ✅ Good

---

### Version 4 (CatBoost GPU Native Categoricals) - 2026-09-08

**Score**: **0.94590 LB** / 0.94581 OOF (Gap: +0.00009)
**Device**: GPU (CatBoost)
**Result**: **±0.00061 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94488 | 0.94547 | 0.94668 | 0.94609 | 0.94597 | 0.94582 |

**Strategy:** 5-fold stratified GPU CatBoost with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, native CatBoost handling for 10 string categorical features, multi-scale income/commute smooth keys, numeric-as-string columns, frequency encoding, synthetic-artifact flags, and feature selection. Final model used 155 base features plus 189 Triple TE features.
**File:** `Previously trained Files/Archieve/S6E9_V4_CatBoost_Baseline.py`

**Key Learning:**

> Native CatBoost categorical handling produced a close LB–OOF gap (+0.00009), but the 0.94590 LB remained 0.00044 below V3. `_ECL_x_Subsidy` was the strongest fold-1 feature at 10.97%, followed by target-encoded subsidy interactions.

**Status:** ✅ Good

---

### Version 3 (LightGBM CPU Triple TE) - 2026-09-08

**Score**: **0.94634 LB** / 0.94601 OOF (Gap: +0.00033)
**Device**: CPU only (no GPU/cuDF)
**Result**: **±0.00061 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.2 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94505 | 0.94579 | 0.94691 | 0.94625 | 0.94609 | 0.94601 |

**Strategy:** 5-fold stratified CPU LightGBM with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, multi-scale income/commute smooth keys, numeric-as-string columns, frequency encoding, synthetic-artifact flags, and feature selection. Final model used 155 base features plus 189 Triple TE features.
**File:** `Previously trained Files/Archieve/S6E9_V3_LGBM_Baseline.py`

**Key Learning:**

> V3 improved the LB to 0.94634, a +0.00065 gain over V2. Income floor/log-income target encodings dominated fold-1 importance, while CPU LightGBM reached the strongest score so far despite the longer training time.

**Status:** 🏆 Best

---

### Version 2 (XGBoost Triple TE) - 2026-09-08

**Score**: **0.94569 LB** / 0.94560 OOF (Gap: +0.00009)
**Device**: GPU (CUDA)
**Result**: **±0.00060 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94468 | 0.94526 | 0.94647 | 0.94581 | 0.94582 | 0.94561 |

**Strategy:** 5-fold stratified CUDA XGBoost with original-data concatenation per fold after dropping `Buyer_ID`, triple target encoding on 13 core columns with auto/10/100 smoothing, true frequency encoding on all columns, label-encoded categorical features, and feature selection that removed 76 constant columns and 10 perfectly correlated columns.
**File:** `Previously trained Files/Archieve/S6E9_V2_XGB_TripleTE.py`

**Key Learning:**

> Triple TE lifted the score slightly over V1. The strongest fold-1 signals were `LE_Subsidy_Available` (26.33%), `_recipe_score` (25.41%), and `_ECL_x_Subsidy` (16.85%).

**Status:** 🏆 Best

---

### Version 1 (XGBoost Baseline) - 2026-09-08

**Score**: **0.94559 LB** / 0.94535 OOF (Gap: +0.00024)
**Device**: GPU (CUDA)
**Result**: **±0.00062 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94439 | 0.94509 | 0.94629 | 0.94558 | 0.94544 | 0.94535 |

**Strategy:** 5-fold stratified CUDA XGBoost with inverse-frequency sample weights, 19 non-constant digit features, engineered interactions and hard-edge flags, frequency encoding, leakage-safe per-fold target encoding, and original-data concatenation per fold after dropping `Buyer_ID`.
**File:** `Previously trained Files/Archieve/S6E9_V1_XGB_Baseline.py`

**Key Learning:**

> OOF and LB align closely (+0.00024). `_ECL_x_Subsidy` was the strongest fold-1 feature (37.50% importance), followed by `_ev_recipe` (8.13%) and `_ECL_x_RangeAnxiety` (7.92%).

**Status:** 🏆 Best

---
