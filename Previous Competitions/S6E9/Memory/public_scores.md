# S6E9 Public Leaderboard Scores

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category, with the format block first and version rows written below it
> 4. **ORDER** by LB Score (Highest on Top)
> 5. **Include:** OOF, LB, Gap, Training Time
> 6. **CATEGORIZE:** Per the headers below (Two-Stage Models, Multiseed are considered as Single models).
> 7. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit

---

## 📝 Score Logging Format

The format block stays first; score entries are written below it.

| Version | Date  | LB Score | OOF Score | Gap    | Time   | Script    | OOF File  | Sub File  | Notes |
| ------- | ----- | -------- | --------- | ------ | ------ | --------- | --------- | --------- | ----- |
| V#      | MM-DD | X.XXXXX  | X.XXXXX   | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

### Leaderboard Scores Top 5

| Version | Date  | LB Score | OOF Score | Gap      | Time     | Script                         | OOF File         | Sub File         | Notes                                                                                                                                                                                  |
| ------- | ----- | -------- | --------- | -------- | -------- | ------------------------------ | ---------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V23     | 09-19 | 0.94641  | 0.94606   | +0.00035 | 23.4 min | `S6E9_V23_XGB_FullDataRefit.py` | `oof/oof_v23.csv` | `sub/sub_v23.csv` | 🏆 Best; V20's CV path untouched so the OOF is identical by construction; the submission is 0.50 fold-average + 0.50 full-data refit (4112 trees on 100% of train + original). +0.00002 LB purely from the inference side. |
| V22     | 09-19 | 0.94640  | 0.94608   | +0.00032 | 116.8 min | `S6E9_V22_XGB_Tuning.py` | `oof/oof_v22.csv` | `sub/sub_v22.csv` | 🏆 Best OOF; two-stage estimator search on our own matrix — 6 configs on rs=42, top-3 confirmed on rs=7. Winner c1 (`max_depth=3, max_leaves=8, gamma=1.0, colsample_bytree=0.85`) beat the in-run V20 control c0 on both splits (+0.00002 / +0.00001). |
| V20     | 09-19 | 0.94639  | 0.94606   | +0.00033 | 20.6 min | `S6E9_V20_XGB_ArtifactFeatures.py` | `oof/oof_v20.csv` | `sub/sub_v20.csv` | 🏆 Best (tied LB); GPU XGBoost depth=4 lossguide on V19's exact feature matrix. First statistically significant OOF gain over V19 (paired z=+4.76) with no LB movement. |
| V19     | 09-19 | 0.94639  | 0.94599   | +0.00040 | 33.7 min | `S6E9_V19_LGBM_ArtifactFeatures.py` | `oof/oof_v19.csv` | `sub/sub_v19.csv` | 🏆 Best; CPU LightGBM with the V10 pipeline plus fixed digit extraction, 12 target-free generator-lift/novelty features, and 2 structural flags; 378 final features. |
| V10     | 09-09 | 0.94636  | 0.94597   | +0.00039 | 36.5 min | `S6E9_V10_LGBM_TargetedFeatures.py` | `oof/oof_v10.csv` | `sub/sub_v10.csv` | 🏆 Best; CPU LightGBM with targeted ECL-bin bigrams, selective groupby deviations, V3 full FE base, and 363 final features. |
| V18     | 09-19 | 0.94635  | 0.94623   | +0.00012 | ~10 min | `S6E9_V18_HillClimber.py` | `Outputs/oof_v18.csv` | `Outputs/sub_v18.csv` | ✅ Good; CPU forward-stepwise hill climber over V1–V17 OOFs (V14/V3/V11/V6/V2/V8 weights). OOF fit directly, so optimistic; LB tied V10 within noise. |
| V3      | 09-08 | 0.94634  | 0.94601   | +0.00033 | 30.2 min | `S6E9_V3_LGBM_Baseline.py`     | `oof/oof_v3.csv` | `sub/sub_v3.csv` | 🏆 Best; CPU LightGBM with original-data concatenation, triple target encoding, multi-scale smooth keys, numeric-as-string columns, synthetic-artifact flags, and feature selection.   |
| V14     | 09-10 | 0.94630  | 0.94608   | +0.00022 | 35.4 min | `S6E9_V14_PseudoLabel_XGB.py` | `oof/oof_v14.csv` | `sub/sub_v14.csv` | ✅ Good; GPU XGBoost depth 3 with V10 teacher pseudo-labels, 150,859 half-weight pseudo-labeled rows, and V12-style features. |
| V21     | 09-19 | 0.94629  | 0.94597   | +0.00032 | 16.4 min | `S6E9_V21_XGB_CrossKeys.py` | `oof/oof_v21.csv` | `sub/sub_v21.csv` | ❌ Failed; V20 plus 6 explicit cross keys but with the digit block and 4 anomaly flags removed. The removals narrowed the space and `_ECL_x_Subsidy` seized 59% of gain, starving the lift-trigram family. Paired DeLong vs V20: -0.00009, z=-5.01. |
| V12     | 09-09 | 0.94629  | 0.94598   | +0.00031 | 29.7 min | `S6E9_V12_XGB_Depth3_Deotte.py` | `oof/oof_v12.csv` | `sub/sub_v12.csv` | ✅ Good; GPU XGBoost depth 3 with Deotte-style parameters, inner K-fold TE, all discussion patterns, and 237 final features. |
| V11     | 09-09 | 0.94629  | 0.94599   | +0.00030 | 30.3 min | `S6E9_V11_LGBM_DeepAnalysis.py` | `oof/oof_v11.csv` | `sub/sub_v11.csv` | ✅ Good; CPU LightGBM with six deep-analysis features, targeted bigrams, selective groupby deviations, and 374 final features. |

---

## Single Models

| Version | Date  | LB Score | OOF Score | Gap      | Time     | Script                         | OOF File         | Sub File         | Notes                                                                                                                                                                                  |
| ------- | ----- | -------- | --------- | -------- | -------- | ------------------------------ | ---------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V23     | 09-19 | 0.94641  | 0.94606   | +0.00035 | 23.4 min | `S6E9_V23_XGB_FullDataRefit.py` | `oof/oof_v23.csv` | `sub/sub_v23.csv` | 🏆 Best; V20's CV path untouched so the OOF is identical by construction; the submission is 0.50 fold-average + 0.50 full-data refit (4112 trees on 100% of train + original). +0.00002 LB purely from the inference side. |
| V22     | 09-19 | 0.94640  | 0.94608   | +0.00032 | 116.8 min | `S6E9_V22_XGB_Tuning.py` | `oof/oof_v22.csv` | `sub/sub_v22.csv` | 🏆 Best OOF; two-stage estimator search on our own matrix — 6 configs on rs=42, top-3 confirmed on rs=7. Winner c1 (`max_depth=3, max_leaves=8, gamma=1.0, colsample_bytree=0.85`) beat the in-run V20 control c0 on both splits (+0.00002 / +0.00001). |
| V20     | 09-19 | 0.94639  | 0.94606   | +0.00033 | 20.6 min | `S6E9_V20_XGB_ArtifactFeatures.py` | `oof/oof_v20.csv` | `sub/sub_v20.csv` | 🏆 Best (tied LB); GPU XGBoost depth=4 lossguide on V19's exact feature matrix. First statistically significant OOF gain over V19 (paired z=+4.76) with no LB movement. |
| V19     | 09-19 | 0.94639  | 0.94599   | +0.00040 | 33.7 min | `S6E9_V19_LGBM_ArtifactFeatures.py` | `oof/oof_v19.csv` | `sub/sub_v19.csv` | 🏆 Best; CPU LightGBM with the V10 pipeline plus fixed digit extraction, 12 target-free generator-lift/novelty features, and 2 structural flags; 378 final features. |
| V10     | 09-09 | 0.94636  | 0.94597   | +0.00039 | 36.5 min | `S6E9_V10_LGBM_TargetedFeatures.py` | `oof/oof_v10.csv` | `sub/sub_v10.csv` | 🏆 Best; CPU LightGBM with targeted ECL-bin bigrams, selective groupby deviations, V3 full FE base, and 363 final features. |
| V18     | 09-19 | 0.94635  | 0.94623   | +0.00012 | ~10 min | `S6E9_V18_HillClimber.py` | `Outputs/oof_v18.csv` | `Outputs/sub_v18.csv` | ✅ Good; CPU forward-stepwise hill climber over V1–V17 OOFs (V14/V3/V11/V6/V2/V8 weights). OOF fit directly, so optimistic; LB tied V10 within noise. |
| V3      | 09-08 | 0.94634  | 0.94601   | +0.00033 | 30.2 min | `S6E9_V3_LGBM_Baseline.py`     | `oof/oof_v3.csv` | `sub/sub_v3.csv` | 🏆 Best; CPU LightGBM with original-data concatenation, triple target encoding, multi-scale smooth keys, numeric-as-string columns, synthetic-artifact flags, and feature selection.   |
| V14     | 09-10 | 0.94630  | 0.94608   | +0.00022 | 35.4 min | `S6E9_V14_PseudoLabel_XGB.py` | `oof/oof_v14.csv` | `sub/sub_v14.csv` | ✅ Good; GPU XGBoost depth 3 with V10 teacher pseudo-labels, 150,859 half-weight pseudo-labeled rows, and V12-style features. |
| V21     | 09-19 | 0.94629  | 0.94597   | +0.00032 | 16.4 min | `S6E9_V21_XGB_CrossKeys.py` | `oof/oof_v21.csv` | `sub/sub_v21.csv` | ❌ Failed; V20 plus 6 explicit cross keys but with the digit block and 4 anomaly flags removed. The removals narrowed the space and `_ECL_x_Subsidy` seized 59% of gain, starving the lift-trigram family. Paired DeLong vs V20: -0.00009, z=-5.01. |
| V12     | 09-09 | 0.94629  | 0.94598   | +0.00031 | 29.7 min | `S6E9_V12_XGB_Depth3_Deotte.py` | `oof/oof_v12.csv` | `sub/sub_v12.csv` | ✅ Good; GPU XGBoost depth 3 with Deotte-style parameters, inner K-fold TE, all discussion patterns, and 237 final features. |
| V11     | 09-09 | 0.94629  | 0.94599   | +0.00030 | 30.3 min | `S6E9_V11_LGBM_DeepAnalysis.py` | `oof/oof_v11.csv` | `sub/sub_v11.csv` | ✅ Good; CPU LightGBM with six deep-analysis features, targeted bigrams, selective groupby deviations, and 374 final features. |
| V16     | 09-11 | 0.94625  | 0.94595   | +0.00030 | 28.6 min | `S6E9_V16_Forensic_Targeted.py` | `oof/oof_v16.csv` | `sub/sub_v16.csv` | ✅ Good; GPU XGBoost depth 3 with five forensic-targeted features, V14 proven features, and no pseudo-labels. |
| V24     | 09-19 | 0.94615  | 0.94593   | +0.00022 | 13.7 min | `S6E9_V24_CatBoost_Artifacts.py` | `oof/oof_v24.csv` | `sub/sub_v24.csv` | ❌ Failed; V20's exact feature matrix through CatBoost depth=6 GPU (lr=0.03, l2_leaf_reg=3.0, Bernoulli 0.8, od_wait=500). OOF -0.00013 vs V20 and LB -0.00024; fastest run yet but the family is inferior here. |
| V17     | 09-11 | 0.94612  | 0.94582   | +0.00030 | 97.3 min | `S6E9_V17_RealMLP.py` | `oof/oof_v17.csv` | `sub/sub_v17.csv` | ✅ Good; GPU RealMLP with PBLD embeddings, 8-model ensemble, EMA, label smoothing, and V14 full features. |
| V9      | 09-09 | 0.94612  | 0.94584   | +0.00028 | 21.4 min | `S6E9_V9_XGB_FullFE.py` | `oof/oof_v9.csv` | `sub/sub_v9.csv` | ✅ Good; GPU XGBoost using V3’s full 344-feature FE pipeline with V2’s proven parameters. |
| V6      | 09-08 | 0.94606  | 0.94585   | +0.00021 | 43.9 min | `S6E9_V6_TabM_FeatureSelection.py` | `oof/oof_v6.csv` | `sub/sub_v6.csv` | ✅ Good; CUDA TabM with evidence-based selection from V5 LR and V3 LightGBM importance, per-fold target encoding, and 83 selected features. |
| V8      | 09-09 | 0.94603  | 0.94583   | +0.00020 | 19.6 min | `S6E9_V8_CatBoost_Ordered.py` | `oof/oof_v8.csv` | `sub/sub_v8.csv` | ✅ Good; GPU CatBoost Ordered boosting with native categorical features, original-data concatenation, triple target encoding, and 344 final features. |
| V4      | 09-08 | 0.94590  | 0.94581   | +0.00009 | 35.0 min | `S6E9_V4_CatBoost_Baseline.py` | `oof/oof_v4.csv` | `sub/sub_v4.csv` | ✅ Good; GPU CatBoost with native categorical features, original-data concatenation, triple target encoding, multi-scale smooth keys, synthetic-artifact flags, and feature selection. |
| V2      | 09-08 | 0.94569  | 0.94560   | +0.00009 | 6.0 min  | `S6E9_V2_XGB_TripleTE.py`      | `oof/oof_v2.csv` | `sub/sub_v2.csv` | 🏆 Best; CUDA XGBoost with triple target encoding, true frequency encoding on all columns, label encoding, and feature selection.                                                      |
| V13     | 09-09 | 0.94568  | 0.94490   | +0.00078 | 14.4 min | `S6E9_V13_DCN_V2.py` | `oof/oof_v13.csv` | `sub/sub_v13.csv` | ✅ Good; CUDA DCN-V2 with 79 selected features, 4 cross layers, rank 64, and 4 experts. |
| V7      | 09-09 | 0.94568  | 0.94545   | +0.00023 | 159.1 min | `S6E9_V7_FTTransformer_FullFeatures.py` | `oof/oof_v7.csv` | `sub/sub_v7.csv` | ✅ Good; CUDA FT-Transformer with 79 evidence-selected features, auto-smoothed TE, 35 numerical features, and 34 low-cardinality categorical features. |
| V1      | 09-08 | 0.94559  | 0.94535   | +0.00024 | 2.1 min  | `S6E9_V1_XGB_Baseline.py`      | `oof/oof_v1.csv` | `sub/sub_v1.csv` | 🏆 Best; CUDA XGBoost baseline with original data, target encoding, and engineered features.                                                                                           |
| V5      | 09-08 | 0.94478  | 0.94468   | +0.00010 | 16.0 min | `S6E9_V5_LR_Baseline.py`       | `oof/oof_v5.csv` | `sub/sub_v5.csv` | ✅ Good; CPU LogisticRegression with per-fold StandardScaler, original-data concatenation, triple target encoding, KFold CV, and feature selection. |
| V15     | 09-10 | 0.91256  | 0.91359   | -0.00103 | 19.5 min | `S6E9_V15_KNN.py` | `oof/oof_v15.csv` | `sub/sub_v15.csv` | ❌ Failed/Overfit; CPU KNN exact-match lookup had 0% matches and k=10 KDTree fallback severely underperformed. |
