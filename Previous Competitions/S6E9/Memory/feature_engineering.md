# S6E9 Feature Engineering Notes

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first), with the format block first and feature rows written below it
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format

The format block stays first; feature entries are written below it.

| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|

---

## Version 2 — Confirmed LB 0.94569 (2026-09-08)

Importance is the reported fold-1 XGBoost importance. No isolated ablation was run, so impact records observed model contribution in the submitted V2 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `LE_Subsidy_Available` | Label-encoded `Subsidy_Available` | 26.33 | Strongest fold-1 signal in the Triple TE model | ✅ Used |
| `_recipe_score` | cdeotte-style buy score / recipe score | 25.41 | Major composite signal that ranked just below the subsidy label feature | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 16.85 | Stayed highly predictive even with Triple TE added | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 3.46 | Smaller but still meaningful interaction feature | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 1.51 | Continued to add signal in the richer feature space | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern numeric feature | 1.24 | Retained direct signal after encoding and interactions | ✅ Used |
| `_ECL_x_Subsidy_freq` | Frequency-encoded version of `_ECL_x_Subsidy` | 1.09 | Added a compact frequency-based interaction signal | ✅ Used |
| `TE_Annual_Income_USD_10` | Target encoding of `Annual_Income_USD` with 10 smoothing | 0.77 | Benefited from the lower-smoothing TE view on income | ✅ Used |
| `LE_Range_Anxiety_Level_freq` | Frequency encoding of `Range_Anxiety_Level` | 0.44 | Provided a small but useful categorical frequency signal | ✅ Used |

## Version 1 — Confirmed LB 0.94559 (2026-09-08)

Importance is the reported fold-1 XGBoost importance. No isolated ablation was run, so impact records observed model contribution in the submitted V1 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 37.50 | Strongest fold-1 signal; captured the main subsidy-by-environment preference pattern | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 8.13 | High-value composite feature that ranked near the top in fold-1 importance | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 7.92 | Strong interaction signal tied to concern versus anxiety behavior | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern numeric feature | 5.47 | Direct signal that remained useful after interaction features were added | ✅ Used |
| `Environmental_Concern_Level_digit0` | Digit-derived feature from `Environmental_Concern_Level` | 1.89 | Captured local numeric structure not fully represented by the raw value | ✅ Used |
| `Annual_Income_USD_10q_bin` | 10-quantile bin of `Annual_Income_USD` | 1.13 | Helped bucket income into a non-linear range effect | ✅ Used |
| `_Income_x_Subsidy` | `Annual_Income_USD * Subsidy_Available` interaction | 0.90 | Added a smaller but useful affordability-by-subsidy interaction | ✅ Used |
| `_ecl_max` | Max-based feature derived from environmental concern bins | 0.73 | Provided a compact threshold-style summary of the ECL signal | ✅ Used |
| `Environmental_Concern_Level_5q_bin` | 5-quantile bin of `Environmental_Concern_Level` | 0.41 | Mild incremental gain from coarse quantization | ✅ Used |
