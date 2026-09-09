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
