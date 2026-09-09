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

## Version 7 — Confirmed LB 0.94568 (2026-09-09)

Importance is the evidence-based selected-feature set derived from V5 LogisticRegression and V3 LightGBM importance. No isolated ablation was run, so impact records observed contribution of the retained feature groups in the submitted V7 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Evidence-based 79-feature subset | Features selected using V5 LR and V3 LightGBM importance from 344 raw features | N/A | Reduced the FT-Transformer input space while retaining established linear and tree-model signals | ✅ Used |
| Auto-smoothed target encoding | Auto-smoothed TE on selected source columns, with 93 TE features generated | N/A | Reduced redundancy for neural-network training while preserving categorical-like signal | ✅ Used |
| Numerical feature group | 35 selected numerical features after feature selection | N/A | Provided continuous inputs for the transformer | ✅ Used |
| Categorical feature group | 34 selected categorical features with cardinality below 20 | N/A | Enabled learned categorical embeddings and self-attention over compact categorical inputs | ✅ Used |
| Engineered interactions and smooth keys | Income, commute, environmental, subsidy, and range-anxiety interactions/bins | N/A | Preserved structured nonlinear signals in the selected representation | ✅ Used |
| Original target means and frequency features | Original-data means plus selected frequency encodings | N/A | Added compact prior and prevalence information | ✅ Used |

## Version 6 — Confirmed LB 0.94606 (2026-09-08)

Importance is the evidence-based selected-feature set derived from V5 LogisticRegression and V3 LightGBM importance. No isolated ablation was run, so impact records observed contribution of the retained feature groups in the submitted V6 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Evidence-based 83-feature subset | Features selected using V5 LR and V3 LightGBM importance from 344 raw features | N/A | Reduced the neural-network input space while retaining strong linear and tree-model signals | ✅ Used |
| Per-fold target encoding | Target encoding from 14 source columns computed within each fold | N/A | Provided leakage-controlled categorical-like signal for TabM | ✅ Used |
| Original target means | Per-feature means from the original dataset concatenated per fold | N/A | Preserved signal from the original data source | ✅ Used |
| Frequency features | Top frequency-encoded features retained during selection | N/A | Added compact prevalence information without the full raw categorical expansion | ✅ Used |
| Engineered interactions and smooth keys | Income, commute, environmental, subsidy, and range-anxiety interactions/bins | N/A | Retained structured nonlinear signals in the compact feature set | ✅ Used |
| Hard-edge and synthetic-artifact flags | 30k spike, millionaire cliff, dead zone, env-hater, and related flags | N/A | Preserved known synthetic-data boundary signals | ✅ Used |

## Version 5 — Confirmed LB 0.94478 (2026-09-08)

Importance is the reported fold-1 LogisticRegression absolute coefficient. No isolated ablation was run, so impact records observed model contribution in the submitted V5 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `Subsidy_Available_fe` | Frequency encoding of `Subsidy_Available` | 1.954700 (|coefficient|) | Strongest fold-1 linear signal | ✅ Used |
| `Environmental_Concern_Level_org_mean` | Original-data target mean for environmental concern level | 1.831096 (|coefficient|) | Strong direct original-data signal | ✅ Used |
| `Environmental_Concern_Level_cat_fe` | Frequency encoding of environmental concern level as a category | 1.538157 (|coefficient|) | Strong categorical prevalence signal | ✅ Used |
| `_Income_x_Subsidy_cat_fe` | Frequency encoding of income × subsidy interaction categories | 0.934229 (|coefficient|) | Main linear affordability interaction | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 0.604567 (|coefficient|) | Captured exact-income prevalence | ✅ Used |
| `TE_income1000_floor_100` | Smoothing-100 target encoding of the 1000-dollar income floor key | 0.592623 (|coefficient|) | Stable broad income-band signal | ✅ Used |
| `_ecl_max` | Maximum-based environmental concern summary | 0.400672 (|coefficient|) | Added a compact threshold-style environmental signal | ✅ Used |
| `City_Type_org_mean` | Original-data target mean for city type | 0.373705 (|coefficient|) | Added direct city-type prior information | ✅ Used |
| `TE_income1000_floor_10` | Smoothing-10 target encoding of the 1000-dollar income floor key | 0.366862 (|coefficient|) | Added lower-smoothing income-band signal | ✅ Used |
| `Annual_Income_USD_org_mean` | Original-data target mean for annual income | 0.336444 (|coefficient|) | Added original-data income prior | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 0.306922 (|coefficient|) | Added interaction prevalence information | ✅ Used |
| `income100_floor_fe` | Frequency encoding of the 100-dollar income floor key | 0.300946 (|coefficient|) | Added income-band prevalence | ✅ Used |
| `income1000_floor_fe` | Frequency encoding of the 1000-dollar income floor key | 0.241697 (|coefficient|) | Added coarse income prevalence | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 0.238883 (|coefficient|) | Added smoothed affordability signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 0.235881 (|coefficient|) | Added nonlinear income-shape signal | ✅ Used |

## Version 4 — Confirmed LB 0.94590 (2026-09-08)

Importance is the reported fold-1 CatBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V4 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 10.971282 | Strongest fold-1 feature and main subsidy/environment interaction | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 5.836506 | Strong target-encoded affordability interaction | ✅ Used |
| `TE__ECL_x_Subsidy_cat_auto` | Auto-smoothed target encoding of environmental concern × subsidy categories | 5.583682 | Added a smoothed view of the strongest interaction | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 5.565833 | Captured interaction prevalence structure | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 5.285489 | Stable broad interaction encoding | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 target encoding of environmental concern × subsidy categories | 5.117647 | Added a lower-smoothing interaction signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 4.525729 | Complementary affordability signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 3.907854 | Stable heavily smoothed affordability signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 3.596592 | Strong behavioral interaction | ✅ Used |
| `_ECL_x_RangeAnxiety_cat_fe` | Frequency encoding of environmental concern × range anxiety categories | 3.095117 | Added prevalence information to the interaction | ✅ Used |
| `Range_Anxiety_Level` | Native CatBoost categorical range-anxiety feature | 2.520769 | Retained direct categorical behavior signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 2.432908 | Added a broad income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_auto` | Auto-smoothed target encoding of environmental concern × range anxiety categories | 2.158419 | Complementary behavioral interaction encoding | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 2.059766 | Added a lower-smoothing income-band view | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 1.820202 | Stable broad behavioral interaction signal | ✅ Used |

## Version 3 — Confirmed LB 0.94634 (2026-09-08)

Importance is the reported fold-1 LightGBM split-importance count. No isolated ablation was run, so impact records observed model contribution in the submitted V3 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_auto` | Triple target encoding of the 100-dollar income floor key with auto smoothing | 949 (count) | Strongest reported fold-1 signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of the log-income categorical representation | 799 (count) | Major income-shape signal | ✅ Used |
| `TE_income100_floor_10` | Target encoding of the 100-dollar income floor key with smoothing 10 | 790 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_income100_floor_100` | Target encoding of the 100-dollar income floor key with smoothing 100 | 652 (count) | Complementary heavily smoothed income signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categorical interaction | 639 (count) | Strong affordability-by-subsidy interaction | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of the log-income categorical representation | 588 (count) | Added a second income-shape view | ✅ Used |
| `Annual_Income_USD` | Raw annual income | 579 (count) | Retained direct continuous income signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 562 (count) | Strong behavioral interaction | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 561 (count) | Captured repeated exact-income structure | ✅ Used |
| `_Income_x_Subsidy` | `Annual_Income_USD * Subsidy_Available` interaction | 527 (count) | Useful affordability interaction | ✅ Used |
| `TE__log_Income_cat_100` | Smoothing-100 target encoding of the log-income categorical representation | 506 (count) | Stable broad income signal | ✅ Used |
| `income100_floor_fe` | Frequency encoding of the 100-dollar income floor key | 492 (count) | Added prevalence information for income bands | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 489 (count) | Added lower-smoothing exact-income signal | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 477 (count) | Captured age-specific response structure | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 467 (count) | Stable affordability interaction encoding | ✅ Used |

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
