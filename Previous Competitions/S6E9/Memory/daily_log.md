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


### 19-09-2026
- **Goal**: Test a second model family on V20's artifact matrix — the family axis was only probed once before (V19→V20, LightGBM→XGBoost)
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
