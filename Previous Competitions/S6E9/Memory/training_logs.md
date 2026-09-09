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

### Version 13 (DCN-V2) - 2026-09-09
**Score**: **0.94568 LB** / 0.94490 OOF (Gap: +0.00078)
**Device**: CUDA (PyTorch)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 2.1 min |
| Fold 2 | 2.1 min |
| Fold 3 | 2.1 min |
| Fold 4 | 2.1 min |
| Fold 5 | 2.0 min |
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
| Fold 1 | 4.9 min |
| Fold 2 | 5.8 min |
| Fold 3 | 6.4 min |
| Fold 4 | 4.6 min |
| Fold 5 | 5.9 min |
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
| Fold 1 | 5.3 min |
| Fold 2 | 5.8 min |
| Fold 3 | 5.5 min |
| Fold 4 | 5.4 min |
| Fold 5 | 5.8 min |
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
