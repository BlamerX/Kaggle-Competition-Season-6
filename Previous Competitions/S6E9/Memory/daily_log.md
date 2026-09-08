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
