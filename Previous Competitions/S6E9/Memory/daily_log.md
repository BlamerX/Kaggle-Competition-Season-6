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
