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

## Version 24 — Confirmed LB 0.94615 (2026-09-19) ❌ Rejected

No new features. V20's exact matrix (180 base + 198 Triple TE) through CatBoost depth=6 to test whether the feature set family-depends. Importance is CatBoost's fold-1 percentage (predictions/gain based, sums to 100 across the full table), so values are not comparable to XGBoost gain shares above.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the ECL × Subsidy × Anxiety pool/orig lift | 8.20 | #1 under CatBoost too — the lift trigram leads in a third family | ✅ Used |
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the same lift | 8.18 | #2, near-tied with the 10-view; together they carry 16.4% | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of the ECL × Subsidy category key | 6.79 | CatBoost's native categorical path for the classic interaction | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the trigram lift | 3.64 | Third lift-trigram view; family total ≈ 20% | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 3.18 | Raw interaction, much weaker here than under XGBoost | ✅ Used |
| `_ECL_x_RangeAnxiety` | `ECL * (3 - RA_code)` | 1.41 | Behavioral interaction | ✅ Used |
| `TE_lift_Subsidy_Available_cat_auto` | Auto-smoothed TE of the subsidy lift | 1.27 | Only single-column lift in the top 8 — the crosses dominate | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy | 1.12 | V10 bigram retained but small | ✅ Used |
| `TE_lift_income100_cat_*` | Triple TE of the income-band lift | 0.46/0.26 | Income lift again absorbed by the income TEs | ⚠️ No Improvement |
| whole model | CatBoost depth=6 vs XGBoost depth=4 on identical features | — | OOF 0.94593 vs 0.94606 (-0.00013) / LB 0.94615: right features, worse estimator | ❌ Removed |

## Version 23 — Confirmed LB 0.94641 (2026-09-19) 🏆 Best

No feature changes at all: the CV path is V20's verbatim, so the fold-1 gain table below is V20's within rounding. The change is on the inference side, which this file records because it is now our only working LB lever.

| Item | Definition | Value | Impact | Status |
|------|------------|-------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the trigram lift (fold 1) | 0.14488 gain | Byte-identical to V20's 0.14488 — proof the CV path was untouched | ✅ Used |
| Full-data refit | 4112 trees (= mean best iteration) on 678,665 rows of train + original | +0.00002 LB | Test blend 0.50 fold-average + 0.50 refit; OOF unaffected, so still honest | ✅ Used |
| Refit-vs-fold-average correlation | Pearson 0.99959 / Spearman 0.99946 | mean \|rank diff\| 1,806 / 286,571 | The two predictors agree to 4th decimal; only ~0.6% of rows change rank materially | ⚠️ Research |
| Refit iteration count | `n_estimators` from the mean best iteration, no early stopping | — | Untested alternatives: per-fold mean-of-ranks blend, higher/lower tree count, blend weight ≠ 0.5 | 🔬 Research |

## Version 22 — Confirmed LB 0.94640 (2026-09-19)

No new features (180 base + 198 Triple TE, identical to V19/V20). The estimator was searched, so this entry records what the tuning winner did to the gain distribution. Importance is XGBoost fold-1 gain share; c0's column is the rs=42 table, c1's is the rs=7 fold-1 table (the harness prints only the first config of each fold).

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the trigram lift | 0.2274 (c1) vs 0.1441 (c0) | Doubles its share when depth drops to 3 and colsample rises to 0.85 | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 0.1918 (c1) vs 0.1080 (c0) | Together with `_auto` it takes 0.419 of fold-1 gain vs 0.253 for V20's params | ✅ Used |
| `Environmental_Concern_Level` | Raw ECL column | 0.1039 (c1) | Enters at #3 only in the wide-column config — more columns per split lets raw signals through | ✅ Used |
| `_ev_recipe` | `ECL == 5 & Range_Anxiety == Low` | 0.0458 (c1) vs 0.0654 (c0) | Stable composite | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety_100` | Smoothing-100 TE of the V10 bigram | 0.0444 (c1) | Promoted by the shallower tree | ✅ Used |
| `_ECL_x_Subsidy` | `ECL * Subsidy_Available` | 0.1084 (c0), absent from c1's top 10 | Loses share to the lift crosses under c1 | ⚠️ No Improvement |
| Estimator settings | c1 = `max_depth=3, max_leaves=8, gamma=1.0, colsample_bytree=0.85`, ~5.5k trees | +0.00002 OOF over the in-run c0 control on both rs=42 and rs=7 | Confirms direction: wide + weak + many beats deep + heavily regularised on this matrix | ✅ Used |

> Caveat: V22's c0 fold-1 gain shares (e.g. `TE_lift_trigram_cat_auto` 0.14410) sit slightly below V20/V23's 0.14488 even though c0 reproduced V20's fold AUCs and best iterations exactly. V22 computed importances from the untruncated booster (it ran to `n_estimators=8000`), V20/V23 slice at `best_iteration`. Metrics are unaffected; treat cross-version gain shares as ±0.001.

### V22 search outcomes (feature set fixed, estimator varied)

| Config | Override | rs=42 OOF | Verdict |
|--------|----------|-----------|---------|
| c1 shallow + wide cols | depth 3, 8 leaves, gamma 1.0, colsample 0.85 | 0.94608 | ✅ Winner (also 0.94608 on rs=7) |
| c0 V20 baseline | none (control) | 0.94606 | ✅ Reproduced V20 exactly |
| c4 low min_child_weight | `min_child_weight` reduced | 0.94605 | ⚠️ No Improvement |
| c2 deeper + heavy leaf reg | deeper + stronger leaf regularisation | 0.94593 | ❌ Worse |
| c3 sparse cols + high gamma | colsample ↓, gamma ↑ | 0.94576 | ❌ Worse |
| c5 tiny lr, near-depthless | lr 0.008, `min_child_weight` 500, lambda 10 | 0.94546 | ❌ Clearly worse |

## Version 21 — Confirmed LB 0.94629 (2026-09-19) ❌ Rejected

Importance is fold-1 XGBoost gain share. V21 added six explicit cross keys (each Triple-TE'd plus a target-free lift column) and removed the digit block and four income-anomaly flags. The experiment regressed (paired DeLong vs V20: -0.00009, z = -5.01), so V20 remains the reference feature set.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 58.99 gain | Pathological monopoly — 5.4x its V20 share after the feature pool narrowed; starving every TE branch | ⚠️ No Improvement |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_10` | Smoothing-10 TE of the income-band x ECL x Subsidy lift | 5.12 gain | Strongest NEW cross; #2 overall — the addition itself worked | ✅ Used |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_auto` | Auto-smoothed TE of the same lift | 4.60 gain | Companion view of the new deepest key | ✅ Used |
| `TE_cx_inc_x_ecl_x_sub_auto` | Auto-smoothed TE of the raw cross key | 4.39 gain | Key without the lift transform | ✅ Used |
| `TE_cx_inc_x_ecl_x_sub_10` | Smoothing-10 TE of the raw cross key | 3.34 gain | Lower-smoothing view | ✅ Used |
| `TE_lift_cx_inc_x_ecl_x_sub_cat_100` | Smoothing-100 TE of the cross lift | 3.17 gain | Stable broad encoding | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the ECL x Subsidy x Anxiety lift | 3.10 gain | Was #1 in V20 at 14.49; displaced by the near-duplicate deeper key | ⚠️ No Improvement |
| `TE_cx_inc_x_ecl_x_sub_100` | Smoothing-100 TE of the raw cross key | 2.14 gain | Third view of the same key | ✅ Used |
| `TE_lift_trigram_cat_10` / `_auto` | Lift-trigram TE variants | 2.13 / 1.21 gain | Combined lift-trigram share fell from ~35% (V20) to ~6.7% | ⚠️ No Improvement |
| `lift_Range_Anxiety_Level` | Pool/orig lift of the anxiety level | 0.87 gain | Rose in rank but small absolute gain | ✅ Used |

### V21 cross keys and removals

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `cx_chg_x_home` + `lift_cx_chg_x_home` | charging-total bins x Home_Charging_Possible (Simpson reversal 9.20% -> 14.70%) | not in top-15 | Built correctly, contributed almost nothing once the gain collapsed onto `_ECL_x_Subsidy` | ⚠️ No Improvement |
| `cx_city_x_home` + lift | City_Type x Home_Charging_Possible (Rural 4.64% vs Urban 13.56%) | not in top-15 | Same as above | ⚠️ No Improvement |
| `cx_ecl_x_home` / `cx_ecl_x_city` + lifts | ECL x Home_Charging, ECL x City_Type | not in top-15 | Composition confounders, absorbed by the raw columns | ⚠️ No Improvement |
| `cx_ecl_x_sub_x_comm` + lift | ECL x Subsidy x commute bins (5.0 km isolated) | not in top-15 | The 5.0 km bin did not register | ⚠️ No Improvement |
| digit block (56 cols in V20) | place-value digits and their `_cat` / `_fe` / TE variants | removed | Removing them did not cause the loss directly but narrowed the pool from 312 to 285 fit columns | ❌ Removed |
| `is_30k_spike` / `is_millionaire_cliff` / `is_dead_zone` / `_below_buy_bound` / `_dead_zone_exact` | income anomaly + structural recipe flags | removed with digits | Same removal batch; `is_env_hater` was kept | ❌ Removed |

## Version 20 — Confirmed LB 0.94639 (2026-09-19)

No new features were engineered in V20; the V19 matrix was reused unchanged and only the model family was swapped to XGBoost depth=4. Importance is the reported fold-1 XGBoost **gain share** (not a split count, so it is not comparable to V19's counts).

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the ECL × Subsidy × Anxiety pool/orig lift | 14.49 gain | #1 feature overall (was #4 under LightGBM) — a depth-4 model depends heavily on the pre-computed artifact cross | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` | 10.87 gain | Strongest raw interaction, unchanged | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 10.84 gain | Second lift-trigram view | ✅ Used |
| `TE_lift_trigram_cat_100` | Smoothing-100 TE of the trigram lift | 9.31 gain | Stable broad lift encoding | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy | 9.12 gain | V10 bigram still carries signal | ✅ Used |
| `_ev_recipe` | `ECL == 5 & Range_Anxiety == Low` | 6.60 gain | Composite recipe segment | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_auto` | Auto-smoothed TE of ECL bin × subsidy | 4.95 gain | Companion bigram view | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * (3 - RA_code)` | 4.47 gain | Behavioral interaction | ✅ Used |
| `lift_trigram` | Raw pool/orig lift of the trigram key | 2.14 gain | Direct artifact ratio | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 TE of ECL × subsidy categories | 2.05 gain | Categorical interaction view | ✅ Used |

### V20 usage of the V19 artifact set

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `lift_trigram_cat_fe` | Frequency encoding of the trigram lift | 0.19 gain | Marginal next to the TE views | ✅ Used |
| `TE_lift_commute_cat_*` | Triple TE of integer-commute lift | 0.12/0.10 gain | Weak under XGBoost (stronger under LightGBM) | ⚠️ No Improvement |
| `lift_income100_cat_fe` / `lift_inc_exact_cat_fe` | Frequency encoding of the income-band / exact-income lift | 0.10/0.08 gain | Income lift largely absorbed by income TE keys | ⚠️ No Improvement |
| `novel_inc_cat_fe` | Frequency encoding of the novel-income flag | 0.08 gain | Almost unused by the shallow model | ⚠️ No Improvement |
| `_below_buy_bound` / `_dead_zone_exact` | Structural recipe bounds (income < 41,667 / 31,004–41,970) | ~0 gain | Not selected; matches the reference notebook pruning its equivalent flags | ⚠️ No Improvement |
| `{col}_digit{k}` and digit TE variants | Fixed integer-place digits | ≤ 0.08 gain | Near-zero contribution under depth=4 | ⚠️ No Improvement |

## Version 19 — Confirmed LB 0.94639 (2026-09-19) 🏆 Best

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted V19 model. Lift features are target-free: pool (train+test) frequency ÷ original-dataset frequency.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 735 (count) | Strongest fold-1 signal, unchanged from V10 | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 TE of the 100-dollar income floor key | 632 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed TE of income-as-string categories | 601 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_lift_trigram_cat_auto` | Auto-smoothed TE of the pool/orig frequency lift of ECL × Subsidy × Anxiety | 531 (count) | Strongest NEW artifact feature; encodes the ECL=3 cells that beat the linear recipe by 1.22–1.31× | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 TE of income-as-string categories | 458 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed TE of age-as-string categories | 408 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed TE of log-income categories | 406 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_lift_trigram_cat_10` | Smoothing-10 TE of the trigram lift | 335 (count) | Second trigram-lift view | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed TE of income × subsidy categories | 333 (count) | Strong affordability interaction | ✅ Used |
| `TE_lift_income100_cat_auto` | Auto-smoothed TE of the 100-dollar income-band lift | 330 (count) | Income over/under-production signal | ✅ Used |
| `grp_income_bin_Annual_Income_USD_dev` | Income deviation from income-bin group mean | 323 (count) | Preserved V10 groupby signal | ✅ Used |
| `TE_income_exact_int_auto` | Auto-smoothed TE of exact integer income | 318 (count) | Captured repeated exact-income structure | ✅ Used |
| `lift_inc_exact_cat_fe` | Frequency encoding of the exact-income lift bucket | 313 (count) | Direct target-free artifact prevalence signal | ✅ Used |

### V19 generator-artifact features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `lift_inc_exact` | pool freq(exact income) ÷ orig freq(exact income) | 230 (count) | Raw lift column; buy rate 23.19% under-produced vs 17.12% heavily over-produced | ✅ Used |
| `TE_lift_commute_cat_auto/10/100` | Triple TE of integer-commute lift | 230/188/240 (count) | Added commute over-production signal | ✅ Used |
| `TE_lift_income100_cat_100` | Smoothing-100 TE of income-band lift | 220 (count) | Stable broad income-lift encoding | ✅ Used |
| `lift_income100_cat_fe` | Frequency encoding of income-band lift | 211 (count) | Secondary artifact prevalence signal | ✅ Used |
| `TE_Annual_Income_USD_digit3_cat_auto` | TE of the fixed thousands-digit feature (was corrupted pre-V19) | 198 (count) | Recovered by the digit fix | ✅ Used |
| `TE_Annual_Income_USD_digit2_cat_auto` | TE of the fixed hundreds-digit feature | 178 (count) | Recovered by the digit fix; hundreds digit ≈ round-income marker | ✅ Used |
| `novel_inc` | Income value absent from the original 10k dataset | — | 2.06% of rows at 19.56% buy vs 17.46% base; low split count | ✅ Used |
| `_below_buy_bound` | `Annual_Income_USD < 41667` (recipe can never reach 5.5) | — | Structural near-zero region | ✅ Used |
| `_dead_zone_exact` | `31004 <= Annual_Income_USD <= 41970` | — | 1,257 train rows, 0 buyers | ✅ Used |
| `lift_Subsidy_Available` / `lift_Range_Anxiety_Level` / `lift_Home_Charging_Possible` / `lift_Gender` / `lift_City_Type` / `lift_Current_Car_Type` / `lift_Environmental_Concern_Level` | Per-value pool ÷ orig frequency ratio per categorical | Not in top-15 | Low-cardinality lifts were near-collinear with the raw categorical and its TE; some variants were removed by redundancy selection (149 features dropped) | ⚠️ No Improvement |

### V19 digit-extraction fix (all 18 prior versions were affected)

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `{col}_digit{k}` | Old: `col // (10**k) % 10` — negative k uses float floor-div and reads IEEE-754 binary representation | 198 (digit3), 178 (digit2) fold-1 | Corrupted 89.63% of `Daily_Commute_km` tenths; `digit-1` was effectively an is-integer flag | ❌ Removed |
| `{col}_digit{k}` (fixed) | `np.rint(col*1e4).astype(int64)` then integer `// 10**(k+4) % 10` | as above | Correct tenths digit spans buy rate 0.160 → 0.184 | ✅ Used |

## Version 18 — Confirmed LB 0.94635 (2026-09-19)

No new features were engineered; V18 combined saved OOF predictions only. This entry records the ensemble representation and its diagnostics.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Hill-climber blend | `0.35·V14 + 0.28·V3 + 0.15·V11 + 0.13·V6 + 0.05·V2 + 0.05·V8` on raw probabilities | N/A | LB 0.94635, -0.00001 vs V10; weights fit OOF directly so the +0.00015 OOF gain was noise | ⚠️ No Improvement |
| Simple/rank/logit averages | Equal-weight mean of probabilities, ranks, and logit-space means | N/A | OOF ≤0.94601, all below best single +0.94608 | ⚠️ No Improvement |
| RidgeCV meta-learner | Ridge on OOF logit features, alpha chosen by 5-fold CV | N/A | OOF 0.94314; stacker broken by single-fold OOF vs averaged-test aggregation mismatch | ❌ Removed |

## Version 15 — Confirmed LB 0.91256 (2026-09-10)

No fold-level feature importances were reported. This entry records the encoded lookup/KNN representation and exact-match diagnostics.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| Exact-match key | Encoded combination of the 24 KNN input features used for lookup | N/A | Produced 0% validation/test exact matches; no lookup signal was available | ⚠️ No Improvement |
| KDTree numerical representation | 24 encoded train/test/original features used for Euclidean neighbor search | N/A | Enabled k=10 fallback predictions but produced weak AUC | ✅ Used |
| KNN k=10 fallback | Mean target of the 10 nearest training neighbors | N/A | Main prediction mechanism; underperformed all established baselines | ⚠️ No Improvement |
| Train-key uniqueness check | All 668,665 training keys were unique | N/A | Confirmed the data did not contain a deterministic duplicate-key shortcut | 🔬 Research |

## Version 17 — Confirmed LB 0.94612 (2026-09-11)

No fold-level feature importances were reported. This entry records the RealMLP architecture and the V14-derived feature representation.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| V14 full feature pipeline | Digits, smooth keys, targeted bigrams/trigrams, frequency features, and original target means | N/A | Supplied the full engineered input representation | ✅ Used |
| Triple target encoding | Auto/10/100-smoothed target encodings across selected categorical-like columns | N/A | Added categorical signal for the neural model | ✅ Used |
| PBLD embeddings | Piecewise-linear/binned numerical embeddings used by RealMLP | N/A | Converted numerical patterns into learnable representations | ✅ Used |
| 8-model ensemble | Eight RealMLP ensemble members averaged for prediction | N/A | Reduced neural-model variance | ✅ Used |
| EMA and label smoothing | Exponential moving average plus smoothed training targets | N/A | Stabilized the short three-epoch training regime | ✅ Used |
| Original-data concatenation | Competition data combined with original data per fold | N/A | Preserved the proven V14 training setup | ✅ Used |

## Version 16 — Confirmed LB 0.94625 (2026-09-11)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V16 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 34.0651 | Strongest overall fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 20.3502 | Strongest interaction encoding | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 9.1921 | Strong categorical interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 8.5334 | Major composite adoption signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 5.9434 | Strong behavioral interaction | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 2.3350 | Strong targeted bigram signal | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 2.1165 | Preserved direct environmental signal | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.2917 | Added income-band signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.1424 | Added affordability interaction | ✅ Used |
| `TE__log_Income_cat` | Target encoding of log-income categories | 1.0560 | Added nonlinear income-shape signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.9369 | Added behavioral categorical interaction | ✅ Used |
| `TE_Annual_Income_USD_cat` | Target encoding of income-as-string categories | 0.9294 | Added exact-income categorical signal | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.7350 | Added compact anxiety prevalence signal | ✅ Used |
| `TE_income_exact_int` | Target encoding of exact integer income | 0.6939 | Captured repeated exact-income structure | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety` | Target encoding of ECL bin × range anxiety | 0.5194 | Added targeted anxiety interaction | ✅ Used |

### V16 forensic-targeted features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_Sub_HomeCharging` | Target encoding of subsidy × home-charging possibility | 0.1375 | Strongest forensic feature, but small overall contribution | ✅ Used |
| `_recipe_sub_income` | Recipe/subsidy/income interaction | 0.0370 | Captured subgroup lift but added limited global signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL3` | Fold-safe distance to buyer centroid within ECL=3 | 0.0366 | Added small geometric subgroup signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL1` | Fold-safe distance to buyer centroid within ECL=1 | 0.0353 | Added small geometric subgroup signal | ✅ Used |
| `_dist_to_buyer_centroid_ECL2` | Fold-safe distance to buyer centroid within ECL=2 | 0.0340 | Added small geometric subgroup signal | ✅ Used |

## Version 14 — Confirmed LB 0.94630 (2026-09-10)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V14 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 39.9435 | Strongest fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 18.3191 | Strongest interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 9.2675 | Major composite adoption signal | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 6.5782 | Strong categorical interaction encoding | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 6.3003 | Strong behavioral interaction | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 1.7572 | Preserved direct environmental signal | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 1.3927 | Strong targeted bigram signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.2637 | Added affordability interaction | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.0890 | Added income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.9194 | Added behavioral categorical interaction | ✅ Used |

### Pseudo-labeling configuration

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| High-confidence pseudo-labels | Teacher probability `p>=0.98` or `p<=0.02` | N/A | Added 150,859 test rows at half weight | ✅ Used |
| V10 teacher predictions | Test probabilities from V10 LightGBM | N/A | Supplied pseudo-labels for the V12 XGBoost student | ✅ Used |

## Version 13 — Confirmed LB 0.94568 (2026-09-09)

No fold-level feature importances were reported. This entry records the selected feature architecture and observed model contribution.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| V6 evidence-based 79-feature subset | Features selected from the V3/V10/V12 feature family using V6 evidence | N/A | Compact input representation for DCN-V2 | ✅ Used |
| DCN-V2 cross features | Four cross layers with rank 64 low-rank factorization | N/A | Modeled explicit high-order feature interactions | ✅ Used |
| DCN-V2 mixture-of-experts | Four experts in the deep component | N/A | Added multiple nonlinear interaction pathways | ✅ Used |
| Proven targeted interactions | V10 targeted bigrams plus V12 trigram and income-band × subsidy features | N/A | Preserved the strongest discussion-derived interactions | ✅ Used |
| Categorical feature group | 18 selected categorical features | N/A | Enabled learned categorical representations | ✅ Used |
| Numerical feature group | 61 selected numerical features | N/A | Provided compact continuous inputs | ✅ Used |

## Version 12 — Confirmed LB 0.94629 (2026-09-09)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V12 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 35.3514 | Strongest overall fold-1 signal | ✅ Used |
| `TE_trigram_Sub_ECL_RA` | Target encoding of subsidy × environmental concern × range anxiety | 18.2776 | Strongest discussion-derived feature | ✅ Used |
| `TE__ECL_x_Subsidy_cat` | Target encoding of environmental concern × subsidy categories | 8.3521 | Strong categorical interaction encoding | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 7.5737 | Major composite adoption signal | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 6.1115 | Strong direct environmental signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 5.8006 | Strong secondary behavioral interaction | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy` | Target encoding of ECL bin × subsidy | 1.5226 | Strong targeted bigram signal | ✅ Used |
| `TE_income100_floor` | Target encoding of the 100-dollar income floor key | 1.2162 | Added income-band signal | ✅ Used |
| `TE__log_Income_cat` | Target encoding of log-income categories | 1.0464 | Added nonlinear income-shape signal | ✅ Used |
| `TE__Income_x_Subsidy_cat` | Target encoding of income × subsidy categories | 1.0406 | Added affordability interaction | ✅ Used |
| `TE_income_exact_int` | Target encoding of exact integer income | 0.8521 | Captured repeated exact-income structure | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat` | Target encoding of environmental concern × range anxiety categories | 0.8519 | Added behavioral categorical interaction | ✅ Used |
| `TE_Annual_Income_USD_cat` | Target encoding of income-as-string categories | 0.8364 | Added exact-income categorical signal | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.7139 | Added compact anxiety prevalence signal | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety` | Target encoding of ECL bin × range anxiety | 0.5799 | Added targeted anxiety interaction | ✅ Used |

### Proven discussion features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_income_band_x_Subsidy` | Target encoding of income band × subsidy | 0.0607 | Small but retained targeted affordability signal | ✅ Used |

## Version 11 — Confirmed LB 0.94629 (2026-09-09)

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted V11 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_recipe_residual` | `recipe_score - 5.5`, measuring distance from the recipe boundary | 996 (count) | Strongest reported fold-1 feature | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 761 (count) | Strong income-band signal | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 759 (count) | Complementary lower-smoothing income signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 634 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 606 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 488 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_income100_floor_100` | Smoothing-100 target encoding of the 100-dollar income floor key | 423 (count) | Stable broad income-band encoding | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 404 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 381 (count) | Complementary income-shape view | ✅ Used |
| `TE_Annual_Income_USD_cat_100` | Smoothing-100 target encoding of income-as-string categories | 377 (count) | Stable exact-income encoding | ✅ Used |
| `TE_commute_integer_auto` | Auto-smoothed target encoding of integer commute distance | 362 (count) | Added commute-pattern signal | ✅ Used |
| `TE_income_exact_int_auto` | Auto-smoothed target encoding of exact integer income | 350 (count) | Captured repeated exact-income structure | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 350 (count) | Added income prevalence information | ✅ Used |
| `TE_trigram_Sub_ECL_RA_auto` | Auto-smoothed TE of subsidy × environmental concern × range anxiety | 339 (count) | Strongest new trigram encoding | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 326 (count) | Strong affordability interaction | ✅ Used |

### New V11 deep-analysis features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_trigram_Sub_ECL_RA_10` | Smoothing-10 TE of subsidy × environmental concern × range anxiety | 257 (count) | Complementary trigram signal | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_auto` | Auto-smoothed TE of income band × subsidy | 121 (count) | Targeted affordability interaction | ✅ Used |
| `TE_trigram_Sub_ECL_RA_100` | Smoothing-100 TE of subsidy × environmental concern × range anxiety | 87 (count) | Stable broad trigram encoding | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_100` | Smoothing-100 TE of income band × subsidy | 66 (count) | Stable targeted affordability encoding | ✅ Used |
| `TE_bigram_income_band_x_Subsidy_10` | Smoothing-10 TE of income band × subsidy | 49 (count) | Lower-smoothing targeted affordability encoding | ✅ Used |
| `_never_buy_flag` | `Subsidy_Available=No OR Range_Anxiety_Level=High` | 0 (count) | Retained for analysis despite no fold-1 split importance | ✅ Used |
| `_income_round00` | Indicator that income ends in `00` | 0 (count) | Retained for analysis despite no fold-1 split importance | ✅ Used |

## Version 10 — Confirmed LB 0.94636 (2026-09-09)

Importance is the reported fold-1 LightGBM importance. No isolated ablation was run, so impact records observed model contribution in the submitted corrected V10 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 948 (count) | Strongest reported fold-1 signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 877 (count) | Complementary broad income-band signal | ✅ Used |
| `TE_Annual_Income_USD_cat_auto` | Auto-smoothed target encoding of income-as-string categories | 747 (count) | Strong exact-income categorical signal | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 651 (count) | Added lower-smoothing income signal | ✅ Used |
| `TE_income100_floor_100` | Smoothing-100 target encoding of the 100-dollar income floor key | 529 (count) | Stable broad income-band encoding | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 491 (count) | Strong affordability interaction | ✅ Used |
| `grp_income_bin_Annual_Income_USD_dev` | Income deviation from income-bin group mean | 461 (count) | Strongest targeted groupby feature | ✅ Used |
| `Annual_Income_USD_cat_fe` | Frequency encoding of income-as-string categories | 454 (count) | Added income prevalence information | ✅ Used |
| `TE__log_Income_cat_auto` | Auto-smoothed target encoding of log-income categories | 453 (count) | Nonlinear income-shape signal | ✅ Used |
| `TE_commute_integer_auto` | Auto-smoothed target encoding of integer commute distance | 443 (count) | Added commute-pattern signal | ✅ Used |
| `TE_Annual_Income_USD_cat_100` | Smoothing-100 target encoding of income-as-string categories | 441 (count) | Stable exact-income encoding | ✅ Used |
| `TE_Age_cat_auto` | Auto-smoothed target encoding of age-as-string categories | 425 (count) | Added age-specific structure | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 415 (count) | Complementary income-shape view | ✅ Used |
| `grp_ECL_bin_Annual_Income_USD_dev` | Income deviation from environmental-concern-bin group mean | 406 (count) | Strong targeted environmental-income groupby feature | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 401 (count) | Lower-smoothing affordability signal | ✅ Used |

### Targeted new features

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `TE_bigram_ECL_bin_x_Subsidy_auto` | Auto-smoothed TE of ECL bin × subsidy interaction | 348 (count) | Strongest targeted bigram signal | ✅ Used |
| `TE_bigram_ECL_bin_x_RangeAnxiety_auto` | Auto-smoothed TE of ECL bin × range-anxiety interaction | 317 (count) | Strong targeted behavioral interaction | ✅ Used |
| `grp_Subsidy_Available_Annual_Income_USD_dev` | Income deviation from subsidy-availability group mean | 243 (count) | Added subsidy-income context | ✅ Used |
| `grp_Range_Anxiety_Level_Daily_Commute_km_dev` | Commute deviation from range-anxiety group mean | 215 (count) | Added anxiety-commute context | ✅ Used |
| `grp_Range_Anxiety_Level_Annual_Income_USD_dev` | Income deviation from range-anxiety group mean | 206 (count) | Added anxiety-income context | ✅ Used |
| `TE_bigram_ECL_bin_x_Subsidy_10` | Smoothing-10 TE of ECL bin × subsidy interaction | 201 (count) | Added lower-smoothing targeted bigram signal | ✅ Used |
| `grp_ECL_bin_Daily_Commute_km_dev` | Commute deviation from environmental-concern-bin group mean | 196 (count) | Added environmental-commute context | ✅ Used |
| `grp_income_bin_Daily_Commute_km_dev` | Commute deviation from income-bin group mean | 189 (count) | Added income-commute context | ✅ Used |

## Version 9 — Confirmed LB 0.94612 (2026-09-09)

Importance is the reported fold-1 XGBoost feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V9 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy_cat` | Categorical interaction of environmental concern × subsidy | 39.1568 | Strongest fold-1 signal in the full FE XGBoost model | ✅ Used |
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 36.0873 | Second dominant interaction; together with the categorical form drove most importance | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 4.6745 | Stable broad encoding of the main interaction | ✅ Used |
| `_ev_recipe` | Engineered EV recipe feature | 1.7511 | Retained a useful composite adoption signal | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 1.4976 | Strong secondary behavioral interaction | ✅ Used |
| `Environmental_Concern_Level` | Raw environmental concern level | 1.1030 | Preserved direct environmental signal | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 0.8482 | Added interaction prevalence information | ✅ Used |
| `TE_income_exact_int_10` | Smoothing-10 target encoding of exact integer income | 0.6434 | Captured repeated exact-income structure | ✅ Used |
| `TE_Annual_Income_USD_cat_10` | Smoothing-10 target encoding of income-as-string categories | 0.5420 | Added lower-smoothing income signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 0.5196 | Stable broad behavioral interaction encoding | ✅ Used |
| `TE__log_Income_cat_10` | Smoothing-10 target encoding of log-income categories | 0.4694 | Added nonlinear income-shape signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 0.4547 | Added lower-smoothing affordability interaction | ✅ Used |
| `Environmental_Concern_Level_cat_fe` | Frequency encoding of environmental concern categories | 0.3458 | Added prevalence information for environmental concern | ✅ Used |
| `Range_Anxiety_Level_fe` | Frequency encoding of range anxiety level | 0.2688 | Added compact anxiety prevalence signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_10` | Smoothing-10 target encoding of environmental concern × range anxiety categories | 0.2551 | Added lower-smoothing behavioral interaction | ✅ Used |

## Version 8 — Confirmed LB 0.94603 (2026-09-09)

Importance is the reported fold-1 CatBoost Ordered feature importance. No isolated ablation was run, so impact records observed model contribution in the submitted V8 model.

| Feature | Formula / Definition | Importance % | Impact | Status |
|---------|----------------------|--------------|--------|--------|
| `_ECL_x_Subsidy` | `Environmental_Concern_Level * Subsidy_Available` interaction | 15.319007 | Strongest fold-1 feature; remained dominant under Ordered boosting | ✅ Used |
| `TE__ECL_x_Subsidy_cat_auto` | Auto-smoothed target encoding of environmental concern × subsidy categories | 8.098418 | Strong smoothed version of the main interaction | ✅ Used |
| `TE__ECL_x_Subsidy_cat_100` | Smoothing-100 target encoding of environmental concern × subsidy categories | 7.654647 | Stable broad interaction encoding | ✅ Used |
| `_ECL_x_Subsidy_cat_fe` | Frequency encoding of environmental concern × subsidy categories | 5.052358 | Added interaction prevalence information | ✅ Used |
| `_ECL_x_RangeAnxiety` | `Environmental_Concern_Level * Range_Anxiety_Level` interaction | 4.635254 | Strong behavioral interaction | ✅ Used |
| `TE__Income_x_Subsidy_cat_auto` | Auto-smoothed target encoding of income × subsidy categories | 3.588998 | Complementary affordability signal | ✅ Used |
| `TE__ECL_x_Subsidy_cat_10` | Smoothing-10 target encoding of environmental concern × subsidy categories | 3.518334 | Added a lower-smoothing interaction view | ✅ Used |
| `Annual_Income_USD_org_mean` | Original-data target mean for annual income | 3.280062 | Strong original-data income prior | ✅ Used |
| `TE__Income_x_Subsidy_cat_10` | Smoothing-10 target encoding of income × subsidy categories | 2.891573 | Lower-smoothing affordability signal | ✅ Used |
| `_ECL_x_RangeAnxiety_cat_fe` | Frequency encoding of environmental concern × range anxiety categories | 2.696740 | Added prevalence structure to the behavioral interaction | ✅ Used |
| `TE_income100_floor_10` | Smoothing-10 target encoding of the 100-dollar income floor key | 2.677775 | Added an income-band signal | ✅ Used |
| `TE_income100_floor_auto` | Auto-smoothed target encoding of the 100-dollar income floor key | 2.343336 | Complementary broad income-band signal | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_auto` | Auto-smoothed target encoding of environmental concern × range anxiety categories | 2.331000 | Complementary behavioral encoding | ✅ Used |
| `TE__ECL_x_RangeAnxiety_cat_100` | Smoothing-100 target encoding of environmental concern × range anxiety categories | 2.191141 | Stable broad behavioral interaction signal | ✅ Used |
| `TE__Income_x_Subsidy_cat_100` | Smoothing-100 target encoding of income × subsidy categories | 1.621161 | Stable broad affordability encoding | ✅ Used |

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
