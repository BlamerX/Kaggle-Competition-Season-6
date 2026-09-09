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
| V10     | 09-09 | 0.94636  | 0.94597   | +0.00039 | 36.5 min | `S6E9_V10_LGBM_TargetedFeatures.py` | `oof/oof_v10.csv` | `sub/sub_v10.csv` | 🏆 Best; CPU LightGBM with targeted ECL-bin bigrams, selective groupby deviations, V3 full FE base, and 363 final features. |
| V3      | 09-08 | 0.94634  | 0.94601   | +0.00033 | 30.2 min | `S6E9_V3_LGBM_Baseline.py`     | `oof/oof_v3.csv` | `sub/sub_v3.csv` | 🏆 Best; CPU LightGBM with original-data concatenation, triple target encoding, multi-scale smooth keys, numeric-as-string columns, synthetic-artifact flags, and feature selection.   |
| V12     | 09-09 | 0.94629  | 0.94598   | +0.00031 | 29.7 min | `S6E9_V12_XGB_Depth3_Deotte.py` | `oof/oof_v12.csv` | `sub/sub_v12.csv` | ✅ Good; GPU XGBoost depth 3 with Deotte-style parameters, inner K-fold TE, all discussion patterns, and 237 final features. |
| V11     | 09-09 | 0.94629  | 0.94599   | +0.00030 | 30.3 min | `S6E9_V11_LGBM_DeepAnalysis.py` | `oof/oof_v11.csv` | `sub/sub_v11.csv` | ✅ Good; CPU LightGBM with six deep-analysis features, targeted bigrams, selective groupby deviations, and 374 final features. |
| V9      | 09-09 | 0.94612  | 0.94584   | +0.00028 | 21.4 min | `S6E9_V9_XGB_FullFE.py` | `oof/oof_v9.csv` | `sub/sub_v9.csv` | ✅ Good; GPU XGBoost using V3’s full 344-feature FE pipeline with V2’s proven parameters. |

---

## Single Models

| Version | Date  | LB Score | OOF Score | Gap      | Time     | Script                         | OOF File         | Sub File         | Notes                                                                                                                                                                                  |
| ------- | ----- | -------- | --------- | -------- | -------- | ------------------------------ | ---------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V10     | 09-09 | 0.94636  | 0.94597   | +0.00039 | 36.5 min | `S6E9_V10_LGBM_TargetedFeatures.py` | `oof/oof_v10.csv` | `sub/sub_v10.csv` | 🏆 Best; CPU LightGBM with targeted ECL-bin bigrams, selective groupby deviations, V3 full FE base, and 363 final features. |
| V3      | 09-08 | 0.94634  | 0.94601   | +0.00033 | 30.2 min | `S6E9_V3_LGBM_Baseline.py`     | `oof/oof_v3.csv` | `sub/sub_v3.csv` | 🏆 Best; CPU LightGBM with original-data concatenation, triple target encoding, multi-scale smooth keys, numeric-as-string columns, synthetic-artifact flags, and feature selection.   |
| V12     | 09-09 | 0.94629  | 0.94598   | +0.00031 | 29.7 min | `S6E9_V12_XGB_Depth3_Deotte.py` | `oof/oof_v12.csv` | `sub/sub_v12.csv` | ✅ Good; GPU XGBoost depth 3 with Deotte-style parameters, inner K-fold TE, all discussion patterns, and 237 final features. |
| V11     | 09-09 | 0.94629  | 0.94599   | +0.00030 | 30.3 min | `S6E9_V11_LGBM_DeepAnalysis.py` | `oof/oof_v11.csv` | `sub/sub_v11.csv` | ✅ Good; CPU LightGBM with six deep-analysis features, targeted bigrams, selective groupby deviations, and 374 final features. |
| V9      | 09-09 | 0.94612  | 0.94584   | +0.00028 | 21.4 min | `S6E9_V9_XGB_FullFE.py` | `oof/oof_v9.csv` | `sub/sub_v9.csv` | ✅ Good; GPU XGBoost using V3’s full 344-feature FE pipeline with V2’s proven parameters. |
| V6      | 09-08 | 0.94606  | 0.94585   | +0.00021 | 43.9 min | `S6E9_V6_TabM_FeatureSelection.py` | `oof/oof_v6.csv` | `sub/sub_v6.csv` | ✅ Good; CUDA TabM with evidence-based selection from V5 LR and V3 LightGBM importance, per-fold target encoding, and 83 selected features. |
| V8      | 09-09 | 0.94603  | 0.94583   | +0.00020 | 19.6 min | `S6E9_V8_CatBoost_Ordered.py` | `oof/oof_v8.csv` | `sub/sub_v8.csv` | ✅ Good; GPU CatBoost Ordered boosting with native categorical features, original-data concatenation, triple target encoding, and 344 final features. |
| V4      | 09-08 | 0.94590  | 0.94581   | +0.00009 | 35.0 min | `S6E9_V4_CatBoost_Baseline.py` | `oof/oof_v4.csv` | `sub/sub_v4.csv` | ✅ Good; GPU CatBoost with native categorical features, original-data concatenation, triple target encoding, multi-scale smooth keys, synthetic-artifact flags, and feature selection. |
| V2      | 09-08 | 0.94569  | 0.94560   | +0.00009 | 6.0 min  | `S6E9_V2_XGB_TripleTE.py`      | `oof/oof_v2.csv` | `sub/sub_v2.csv` | 🏆 Best; CUDA XGBoost with triple target encoding, true frequency encoding on all columns, label encoding, and feature selection.                                                      |
| V13     | 09-09 | 0.94568  | 0.94490   | +0.00078 | 14.4 min | `S6E9_V13_DCN_V2.py` | `oof/oof_v13.csv` | `sub/sub_v13.csv` | ✅ Good; CUDA DCN-V2 with 79 selected features, 4 cross layers, rank 64, and 4 experts. |
| V7      | 09-09 | 0.94568  | 0.94545   | +0.00023 | 159.1 min | `S6E9_V7_FTTransformer_FullFeatures.py` | `oof/oof_v7.csv` | `sub/sub_v7.csv` | ✅ Good; CUDA FT-Transformer with 79 evidence-selected features, auto-smoothed TE, 35 numerical features, and 34 low-cardinality categorical features. |
| V1      | 09-08 | 0.94559  | 0.94535   | +0.00024 | 2.1 min  | `S6E9_V1_XGB_Baseline.py`      | `oof/oof_v1.csv` | `sub/sub_v1.csv` | 🏆 Best; CUDA XGBoost baseline with original data, target encoding, and engineered features.                                                                                           |
| V5      | 09-08 | 0.94478  | 0.94468   | +0.00010 | 16.0 min | `S6E9_V5_LR_Baseline.py`       | `oof/oof_v5.csv` | `sub/sub_v5.csv` | ✅ Good; CPU LogisticRegression with per-fold StandardScaler, original-data concatenation, triple target encoding, KFold CV, and feature selection. |
