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
