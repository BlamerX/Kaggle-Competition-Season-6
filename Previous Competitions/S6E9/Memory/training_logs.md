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
