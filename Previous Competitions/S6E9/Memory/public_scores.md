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

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V# | MM-DD | X.XXXXX | X.XXXXX | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

## Single Models

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V2 | 09-08 | 0.94569 | 0.94560 | +0.00009 | 6.0 min | `S6E9_V2_XGB_TripleTE.py` | `oof/oof_v2.csv` | `sub/sub_v2.csv` | 🏆 Best; CUDA XGBoost with triple target encoding, true frequency encoding on all columns, label encoding, and feature selection. |
| V1 | 09-08 | 0.94559 | 0.94535 | +0.00024 | 2.1 min | `S6E9_V1_XGB_Baseline.py` | `oof/oof_v1.csv` | `sub/sub_v1.csv` | 🏆 Best; CUDA XGBoost baseline with original data, target encoding, and engineered features. |
