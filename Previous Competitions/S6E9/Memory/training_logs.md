# S6E9 Training Logs

> **⚠️ RULES:**
>
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT or Delete** previous entries after submission
> 3. **ORDER** by Version Number (Highest on top), with the format block first and new version entries written below it
> 4. **Include timing** breakdown for each version
> 5. **Include all per-fold** results when available

---

## Required Format

The format block stays first; version entries are written below it, highest version number first.

```markdown
### Version [N] ([Description]) - YYYY-MM-DD

**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Device**: CPU/GPU
**Result**: **±X.XXXXX LB**

**Timing:**
| Stage | Time |
|-------|------|
| Total | X.X min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Strategy:** [Brief description]
**File:** `Previously trained Files/Archieve/filename.py`

**Key Learning:**

> [Takeaway]

**Status:**
```

### Version 51 (Ensemble combiner shootout — stack over 26 saved vectors) - 2026-09-28

**Score**: **0.94646 LB** / 0.946372 OOF (sealed) (Gap: +0.00009)
**Device**: local CPU only — inference-only version, nothing trained, no Kaggle session
**Result**: **+0.00000 LB** (ties V40's 0.94646, our best held) / **+0.000120 OOF vs V44**, paired DeLong **z +7.34**, SE 1.6e-5

**Timing:**
| Stage | Time |
|-------|------|
| Load 49 OOF + 49 sub vectors | 1.1 min |
| Eligibility rules + pool construction | 0.6 min |
| 5 combiners × 5 pools, all fold-sealed (10 fits each) | 9.4 min |
| Save + diagnostics | 0.5 min |
| **Total** | **11.6 min** |

**Sealed results (no per-fold print exists for a combiner; each cell is a full 10-fold sealed vector):**
| pool (legs) | equal_rank | mean_logit | nnls | stack | greedy |
|-------------|-----------|-----------|------|-------|--------|
| elite (11) | 0.946318 | 0.946319 | 0.946208 | 0.946352 | — |
| elite_div (6) | 0.946325 | 0.946325 | 0.946208 | 0.946344 | 0.946326 |
| plateau (35) | 0.946266 | 0.946266 | 0.945826 | 0.946382 | — |
| plateau_div (26) | 0.946260 | 0.946261 | 0.945826 | **0.946372** ← winner | 0.946352 |
| wide (41) | 0.946270 | 0.946271 | 0.945826 | 0.946385 | — |

**Strategy:** Train nothing. Load every saved out-of-fold and test vector, filter the pool by declared bands around the best single model (−0.00010 elite / −0.00050 plateau / −0.00150 wide, plus two ρ > 0.999 diversity variants of the narrow bands), and compare five combiners under one protocol: weights and leg sets fitted on nine folds, scored on the tenth. Highest sealed wins; ties inside 2e-05 resolve to the simplest combiner, then the narrowest pool. Lineage twins (V18/V23/V29) and prior ensembles (V38/V39) excluded from the reference and the pool by rule.
**File:** `Previously trained Files/Archieve/S6E9_V51_EnsembleShootout.py` — outputs `oof/oof_v51.csv`, `sub/sub_v51.csv`

**Key Learning:**

> **Best honest CV this project has ever produced (0.946372), and the first artifact where our best CV, a tied-best LB and a tight gap (+0.00009 against V40's +0.00025 for the same score) are the same file.** Three things came out of it. (1) The stack's gain is only half weight-free: the equal-rank answer over the six decorrelated elite legs is 0.946325 with optimism exactly 0.000000, so of the +0.000120, +0.000073 survives with no fitted parameters at all. (2) The diversity filter helped rather than hurt — collapsing the elite pool from 12 legs to 6 at ρ > 0.999 *raised* equal-rank from 0.946318 to 0.946325, and the rule auto-dropped V45 as V40's bit-identical twin with no hand-picking. (3) The winner still carries V38's defect: **11 of 26 weights negative, Σ|w| 2.82**, on legs averaging ρ 0.99405 — and fold-sealing is structurally blind to that, which is exactly how V38's stack came out −0.00001 at equal LB. The hill climber ran and lost: plateau_div greedy 0.946352, +9.2e-5 over equal-rank on the same pool, 2.0e-5 under the stack, optimism +4e-6 (capped at 5 steps / top-12 candidates for speed, so it searched less than the stack).
>
> **Verified independently, because a number nobody re-derived is a guess:** AUC recomputed by a second implementation across all 49 legs (max diff 2.2e-16) and against a brute-force positive/negative pair count (exact); DeLong SE checked against a 250-rep bootstrap (ratio 1.05); every no-fit combiner's sealed vector is bit-identical to its plain blend; the winner's OOF and sub both reproduce from the raw leg files; and 18 checks in total pass. One real fragility surfaced: the saved vector depends on the arbitrary string order of leg names at the 5e-9 level (recomputing in the module's own order matches to 1.1e-16), i.e. four orders below anything we act on.

**Status:** ✅ **Best CV in project history and tied-best LB held — recommended as finals slot 1; V44 (best single, CV 0.946252) recommended as slot 2 so one slot does not depend on the combiner transferring.**

### Version 50 (LightGBM ExtraTrees re-priced at the 5-fold ruler) - 2026-09-26

**Score**: **0.94635 LB** / 0.946085 OOF (5-fold) (Gap: +0.00047)
**Device**: CPU (LightGBM 4.6.0)
**Result**: **-0.000112 OOF vs V46** (10-fold) — below the 0.94615 leg gate by 0.000065

**Timing:**
| Stage | Time |
|-------|------|
| Total | 46.5 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.94615/2031 | 0.94512/1642 | 0.94722/2033 | 0.94565/1649 | 0.94637/2185 | 0.946104 ± 0.000707 |

**Strategy:** V46's exact build (megayak's LightGBM optimiser + `extra_trees=True`/`split_histogram_sampling=True` on V40's 341-column view-A matrix) with `N_FOLDS` 10 → 5 as the only variable, so the ExtraTrees null is read on the 5-fold ruler the plan uses.
**File:** `Previously trained Files/Archieve/S6E9_V50_LGBM_ViewA_5fold.py`

**Key Learning:**

> The null is geometry-robust, and it lands on V26's 5-fold winner a3 at **0.946085 to the last digit** — the same shallow/wide config, now on the richer matrix, scores the same. V26's celebrated +0.00010 was the configuration it rode in on, not randomised thresholds, and V43 already had that configuration.

**Status:** ❌ Failed (fold 2 is the weak block at 5 folds, not fold 4; LightGBM stays ~5.5e-5 under XGBoost)

---

### Version 49 (RealMLP on V40's matrix — the neural family's fair run) - 2026-09-26

**Score**: **0.94616 LB** / 0.945940 OOF (10-fold) (Gap: +0.00024)
**Device**: GPU (cuda), PyTorch 2.10.0+cu128
**Result**: **-0.000268 OOF vs V40** on identical features — below the 0.94615 leg gate by 0.000210

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~2.6 min |
| 10 folds | 10.3–10.9 min each |
| Total | 107.1 min |

**Fold Scores (AUC | best epoch):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94620/2 | 0.94597/2 | 0.94563/2 | 0.94469/2 | 0.94798/2 | 0.94616/2 | 0.94575/2 | 0.94562/2 | 0.94641/2 | 0.94596/2 | 0.946038 ± 0.000787 |

**Strategy:** The published view-G RealMLP recipe (PBLD periodic numeric embeddings, entity embeddings for 87 categoricals, 3×256 SiLU × 8 members, EMA 0.997875, label smoothing 0.04, AdamW lr 0.01 / wd 0.013, 2 epochs, batch 256) on **V40's exact 341-column matrix** and the house `KFold(10, shuffle, 42)` — the fair run the archive never gave the neural family.
**File:** `Previously trained Files/Archieve/S6E9_V49_RealMLP_Parity.py`

**Key Learning:**

> **0.945940 does not reproduce the published 0.946139 / 0.946182** (−0.000199 / −0.000242) and sits 2.7e-4 behind this matrix's XGBoost, so the neural family is closed on the merits rather than on a mis-framed run (V29's TabM was a nested 120-of-312 column subset). All ten folds peaked at epoch 2 — the two-epoch schedule's signature, not truncation — and fold 4 (0.94469) is the seventh model class to bottom out on the same row block.

**Status:** ❌ Failed (six families, one plateau, XGBoost on top)

---

### Version 48 (V44's composition at the 5-fold ruler) - 2026-09-26

**Score**: **0.94636 LB** / 0.946161 OOF (5-fold) (Gap: +0.00021)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **+0.000011 above the 0.94615 leg gate**, −0.000091 vs V44's 10-fold 0.946252

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.9 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.94618/4382 | 0.94526/4263 | 0.94723/3935 | 0.94582/4109 | 0.94637/5407 | 0.946170 ± 0.000650 |

**Strategy:** V44's build — V40's 341-column view-A matrix with V31's 31-column additive LogisticRegression fed as `base_margin` — run at 5 folds instead of 10, with the TargetEncoder's inner cv following the outer geometry (cv 10 → 5). The two effects that ever measured positive against V30, composed at the ruler the plan is judged on.
**File:** `Previously trained Files/Archieve/S6E9_V48_XGB_PriorAndEncodings_5fold.py`

**Key Learning:**

> The only run of the roster to clear its gate, and the mechanism is visible at 5 folds: backbone-only OOF **0.938311** reproduces V31/V44's 0.938310 to four decimals, trees add +0.007850, BestIter 3,935–5,407 of 8,000 (nothing truncated), and the `lift_trigram` mega-feature drops out of fold-1's top 15 — given the additive law for free, the trees stop needing one giant column to rebuild it.

**Status:** ✅ Success (best 5-fold single we own; the roster's ship candidate)

---

### Version 47 (RealMLP at the 5-fold ruler) - 2026-09-26

**Score**: **0.94608 LB** / 0.945866 OOF (5-fold) (Gap: +0.00021)
**Device**: GPU (cuda), PyTorch 2.10.0+cu128
**Result**: **-0.000284 vs the 0.94615 leg gate**, −0.000074 vs V49's 10-fold 0.945940

**Timing:**
| Stage | Time |
|-------|------|
| Total | 51.4 min |

**Fold Scores (AUC | best epoch):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.94591/2 | 0.94493/2 | 0.94699/2 | 0.94555/2 | 0.94613/2 | 0.945903 ± 0.000680 |

**Strategy:** V49's exact architecture and schedule on V40's exact 341-column matrix, with fold geometry as the only variable (10 → 5). The question was whether a parity non-tree single exists on our matrix at the 5-fold ruler.
**File:** `Previously trained Files/Archieve/S6E9_V47_RealMLP_Parity_5fold.py`

**Key Learning:**

> It does not: the net is **0.000295** behind V48's XGBoost on the same columns, and the 5-fold ladder (XGB 0.946161 · LGBM 0.946085 · RealMLP 0.945866) reproduces the 10-fold ordering. The net's geometry step (−0.000074) is the smallest in the roster — a 2-epoch schedule is less sensitive to train-row fraction than a 4–5k-tree booster — which is the one reading that keeps "undertrained, not outclassed" testable; V17's 3-epoch run gained nothing, so the prior is negative.

**Status:** ❌ Failed (neural family closed at both rulers)

---

### Version 46 (LightGBM + ExtraTrees split thresholds on V40's matrix) - 2026-09-25

**Score**: **0.94637 LB** / 0.946197 OOF (10-fold) (Gap: +0.00017)
**Device**: CPU (LightGBM 4.6.0)
**Result**: **+0.000006 OOF vs V43** — the ExtraTrees effect, vs the +0.00010 V26 reported

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~3.4 min |
| 10 folds | 4.6–7.1 min |
| Total | 55.8 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94647/2302 | 0.94618/2001 | 0.94571/1714 | 0.94484/2556 | 0.94816/1758 | 0.94646/3138 | 0.94585/1606 | 0.94571/1990 | 0.94662/1984 | 0.94612/1963 | 0.946213 ± 0.000814 |

**Strategy:** V43 unchanged except `extra_trees=True` + `split_histogram_sampling=True` (every threshold a uniform draw inside its bin instead of the loss-minimising point), on V40's 341-column matrix at 10 folds. The last uncomposed gate-cleared gain in the archive; a split-mechanism line is printed before fold 1 so an ignored parameter cannot masquerade as a null.
**File:** `Previously trained Files/Archieve/S6E9_V46_LGBM_ExtraTrees.py`

**Key Learning:**

> Composed with the encodings, the mechanism that passed two splits inside V26's configuration (z = +5.22 / +4.11) is worth six millionths. **V26's +0.00010 was the shallow/wide config it rode in on, not randomised thresholds** — the durable method that came out of it: compose a tuning claim with your best config; a gain that survives is a mechanism, one that vanishes was a config. BestIter 1,606–3,138 of 20,000, so this is converged, not truncated.

**Status:** ❌ Failed (LightGBM now measured fairly, ~5.5e-5 under XGBoost)

---

### Version 45 (V40's CV path with a full-data refit blended into the submission only) - 2026-09-25

**Score**: **0.94643 LB** / 0.946208 OOF (10-fold) (Gap: +0.00022)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **-0.00003 LB vs V40** (V23's precedent for the same construction was +0.00002)

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE + 10 folds | 46.7 min |
| [3b] refit (5,224 trees on 678,665 rows) | 4.2 min |
| Total | 51.0 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94637/5112 | 0.94625/6420 | 0.94577/4791 | 0.94488/4673 | 0.94811/5424 | 0.94645/5830 | 0.94586/4091 | 0.94573/5260 | 0.94661/4536 | 0.94614/6108 | 0.946217 ± 0.000787 |

**Strategy:** V23's inference-side construction applied to V40: refit one booster on 100% of the labelled rows and average it 0.50/0.50 with the fold-mean **for the submission only**, leaving the CV path untouched. The base was deliberately V40 rather than V44 so that every difference from V40 is the refit and nothing else.
**File:** `Previously trained Files/Archieve/S6E9_V45_XGB_FullDataRefit.py`

**Key Learning:**

> The determinism check passed at the strongest level: V45's OOF equals V40's to **exactly 0.0** across all 668,665 rows, with every fold AUC and BestIter matching, so the refit's ranking perturbation is fully attributable — a mean 666.6 positions of 286,571 (0.23% of test order) at rank ρ 0.999929, which bought −0.00003. The refit column set matched the CV set (341 = 341), the one way this version could have been silently wrong.

**Status:** ⚖️ Partial (full-data refit is public-LB-neutral, measured twice in opposite directions; mechanics for the final, not a model)

---

### Version 44 (V40's encodings + V31's additive base_margin prior, composed) - 2026-09-25

**Score**: **0.94638 LB** / **0.946252 OOF** (10-fold) (Gap: +0.00013) — best honest single-model CV in project history
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **+0.000044 OOF vs V40**, −0.000008 vs the 0.946261 predicted if the two effects are additive

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~2.6 min |
| 10 folds | 3.7–4.7 min |
| Total | 43.7 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94642/4165 | 0.94626/3978 | 0.94589/4305 | 0.94493/3434 | 0.94817/3919 | 0.94636/5709 | 0.94595/3312 | 0.94577/4178 | 0.94664/4622 | 0.94620/4724 | 0.946260 ± 0.000779 |

**Strategy:** Exactly two changes have ever measured positive against V30 — V31's additive logistic backbone as `base_margin` (+0.000053) and V40's window/ladder encodings (+0.000037) — and they had never been in one model. This is their composition, so the number either confirms two independent effects add, or exposes one effect counted twice.
**File:** `Previously trained Files/Archieve/S6E9_V44_XGB_PriorAndEncodings.py`

**Key Learning:**

> Additivity confirmed to 9e-6, so **V40's small gain was real and was not overfitting**, and V44 is the best honest single we have built (+0.000081 over V30 — the largest single-model gain since V19). Three internal signs agree: the backbone alone reproduces its own 0.938310 with the trees adding +0.007942, fold 4 improved for the first time in six model classes (0.94493 vs V30's 0.94488 on identical rows), fold SD 0.000779 was the tightest of the batch, and BestIter *fell* to 3,312–5,709 while the run got cheaper. The board paid −0.00008 for it.

**Status:** ✅ Success (best single CV; held LB record stays V40's 0.94646 — a draw, not a ranking)

---

### Version 43 (The completed view-A 2×2 — their optimiser and smoothings on their encodings) - 2026-09-25

**Score**: **0.94640 LB** / 0.946191 OOF (10-fold) (Gap: +0.00021)
**Device**: CPU (LightGBM 4.6.0)
**Result**: **-0.000017 OOF vs V40**, −0.000090 vs the 0.946281 claim it was built to reproduce

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~3.4 min |
| 10 folds | 5.9–8.4 min |
| Total | 70.7 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94644/1233 | 0.94614/2318 | 0.94578/1601 | 0.94491/1498 | 0.94810/1279 | 0.94641/1742 | 0.94584/1093 | 0.94571/1182 | 0.94660/1316 | 0.94613/1655 | 0.946205 ± 0.000778 |

**Strategy:** V34 tested megayak's optimiser on our encodings (0.946156) and V40 tested their encodings on our optimiser (0.946208); this is the fourth cell — their LightGBM params and their TE smoothings (auto/20/200) applied to V40's window/ladder block, on our `KFold(10, shuffle, 42)` so it stays comparable to both parents.
**File:** `Previously trained Files/Archieve/S6E9_V43_LGBM_ViewAFull.py`

**Key Learning:**

> The 2×2 says their published best single is not reachable from this matrix: +0.000035 for the encodings, −0.000017 for the optimiser, +0.000020 for both, still 0.000090 short of 0.946281. Their encodings are heavily used here (`win_inc_2` 120,354, `TE_ladder_inc10_auto` 116,497) and the metric does not move — **usage is not accuracy**, and leader-replication is closed.

**Status:** ❌ Failed

---

### Version 42 (V40 + continuous generator density ratio, pool vs original kNN) - 2026-09-25

**Score**: **0.94645 LB** / 0.946192 OOF (10-fold) (Gap: +0.00025)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **-0.000016 OOF vs V40**

**Timing:**
| Stage | Time |
|-------|------|
| FE (incl. 955,236 kNN queries) | ~2.5 min |
| 10 folds | 4.7–6.1 min |
| Total | 58.9 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94635/4484 | 0.94625/7395 | 0.94577/4863 | 0.94483/4125 | 0.94808/5093 | 0.94644/5823 | 0.94585/4040 | 0.94569/5080 | 0.94662/4851 | 0.94612/5184 | 0.946200 ± 0.000792 |

**Strategy:** The continuous analogue of the lift features: `log(p_pool(x)/p_original(x))` over the 7 numerics from kNN distances at k = 5/25/100 (pool tree on a seeded 150k subsample, original tree on all 10k rows), plus a 20-quantile bin of the middle scale through the triple TE. Label-free, so pool rows including test are legitimately usable.
**File:** `Previously trained Files/Archieve/S6E9_V42_XGB_DensityRatio.py`

**Key Learning:**

> A smoother version of the same ratio adds nothing once the discrete versions exist: −0.000016, and no density column reached fold-1's top 15 — the trees never asked for them. With the earlier kNN label-lookup null (−0.000004) **both kNN estimators are now null: the generator's fingerprint is already fully carried by count-based features.** The feared cost never materialised — 955k queries fit a 2.5-minute FE stage.

**Status:** ❌ Failed

---

### Version 41 (V40 + all 15 categorical pair cells: key + lift + novelty) - 2026-09-25

**Score**: **0.94644 LB** / 0.946190 OOF (10-fold) (Gap: +0.00025)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **-0.000018 OOF vs V40**

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~3.5 min |
| 10 folds | 5.1–7.2 min |
| Total | 63.2 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94633/4262 | 0.94623/7472 | 0.94576/4874 | 0.94485/4923 | 0.94811/5455 | 0.94644/5274 | 0.94582/4320 | 0.94570/4153 | 0.94663/5086 | 0.94613/5160 | 0.946201 ± 0.000794 |

**Strategy:** V40's success pointed at an unencoded surface: of the 15 pairs among the six categorical originals, only two had ever been encoded. Adds per pair a string cell key through the triple TE (45 cols), a pair-level generator lift (15) and a pair novelty flag (15) — 461 columns vs V40's 341, additive only.
**File:** `Previously trained Files/Archieve/S6E9_V41_XGB_PairCells.py`

**Key Learning:**

> **Absence of a feature is not evidence of a gap.** 60 new columns and 108 new TE inputs are worth −0.000018, the hierarchy does not move (`TE_lift_trigram_cat_100` 11,150 still leads) and not one pair column enters the top 15 — those pairs lack representation because they carry no extra information about the label. Third feature-side null in a day (V41 −1.8e-5, V42 −1.6e-5, V43's whole block +3.5e-5) closes feature engineering by measurement.

**Status:** ❌ Failed

---

### Version 40 (V30's GPU XGBoost + megayak's view-A encoding block) - 2026-09-25

**Score**: **0.94646 LB** / 0.946208 OOF (10-fold) (Gap: +0.00025) — best LB the project has held
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **+0.000037 OOF vs V30** = the encoding effect; −0.000073 vs the 0.946281 view-A claim

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE | ~2.7 min |
| 10 folds | 3.8–5.2 min each |
| Total | 46.2 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94637/5112 | 0.94625/6420 | 0.94577/4791 | 0.94488/4673 | 0.94811/5424 | 0.94645/5830 | 0.94586/4091 | 0.94573/5260 | 0.94661/4536 | 0.94614/6108 | 0.946217 ± 0.000787 |

**Strategy:** One variable — the input encoding. V30's exact GPU XGBoost config and house ruler plus megayak's three untested encoding families only: centred-window target rates (income at radii 2/5/10/25/50/200 dollars, commute at 1/3/10 tenths-of-km, prior 10 pseudo-counts, inner `StratifiedKFold(5, 42)` cross-fit), the same windows inside `City_Type × Current_Car_Type`, and the `//10 //50 //500 //5000` ladder through the triple TE alongside the existing keys. 341 columns = 114 base + 216 TE + 11 window.
**File:** `Previously trained Files/Archieve/S6E9_V40_XGB_ViewAEncodings.py`

**Key Learning:**

> The best score we have posted came from the smallest effect we ever measured: +0.000037 is below the +0.00005 noise gate even though the new columns are used (`TE_ladder_inc10_10`, `win_inc_2`, both ladder `auto` TEs in fold-1's top 15) and every fold converged (BestIter 4,091–6,420 of 8,000). Read with V39 (CV 0.946285 → 0.94640) and V38 (CV 0.946333 → 0.94639), the three invert the CV ranking, so the board is not resolving anything at this altitude.

**Status:** 🏆 Best (LB) | encoding effect fails its own gate → no further feature work

---

### Version 39 (Equal-weight ensemble over the elite leg band) - 2026-09-25

**Score**: **0.94640 LB** / 0.946285 OOF (Gap: +0.000115)
**Device**: local CPU, inference-only (nothing fitted)
**Result**: **+0.000114 OOF vs V30**, and +0.00001 LB over V38's stack while carrying 0.000048 less CV

**Timing:**
| Stage | Time |
|-------|------|
| Load + id-assert 37 paired OOF/test vectors | ~30 s |
| Eligibility rule + 3 combiners + band sensitivity | ~5 s |
| Total | 0.6 min |

**Legs (declared rule, not curation):** solo OOF ≥ best single − 0.00010, ensembles excluded → V30 0.946171 · V31 0.946224 · V34 0.946156 · V36 0.946199; mean pairwise rank ρ 0.99797. `rank_mean` **0.946285** | `logit_mean` 0.946286 | `prob_mean` 0.946286 (all inside the tie band, pre-declared order shipped rank_mean).

**Strategy:** V38's stack got the best honest CV in history and returned V30's exact LB, and the one mechanism CV could not exclude is the stack/test geometry mismatch (a leg's OOF row comes from one fold model, its test row from the average of ten). Equal weighting has no weights to fail to transfer, so this ships exactly that and goes in as a daily to let the public gap answer what CV cannot.
**File:** `Previously trained Files/Archieve/S6E9_V39_EqualWeightElite.py` | **Outputs:** `oof/oof_v39.csv`, `sub/sub_v39.csv`

**Key Learning:**

> ~70% of the ensemble effect (+0.000114 of V38's +0.000162) is combiner-independent, and the fitted meta-learner's remaining ~5e-5 is the same size as moving the eligibility band one notch — so on this dataset a fitted combiner and a plain average are **not resolvable at the resolution we can measure**, and the option with fewer fitted degrees of freedom ships. Dilution is monotone in pool width: 2 legs 0.946304 → 4 → 0.946285 → 11 → 0.946238 → 25 → 0.946223.

**Status:** ✅ Good (default final candidate; V38 keeps the record honest CV)

---

### Version 38 (Ensemble combiner shoot-out — five combiners, fold-sealed) - 2026-09-25

**Score**: **0.94639 LB** / 0.946333 sealed OOF (Gap: +0.00006)
**Device**: local CPU only — first version that never touches Kaggle
**Result**: **+0.000162 OOF vs V30**, and 0.00000 LB movement

**Timing:**
| Stage | Time |
|-------|------|
| Load 36 OOF + 36 test vectors, transform | ~1 min |
| plateau pool × 5 arms | 7 / 7 / 10 / 33 / 416 s |
| wide pool × 5 arms | 8 / 7 / 14 / 41 / 510 s |
| Total | 18.6 min |

**Sealed OOF by combiner (plateau 25 legs / wide 31 legs):** equal rank 0.946223 / 0.946234 · equal logit 0.946224 / 0.946236 · NNLS **0.945826** / 0.945826 · **L2-logistic stack 0.946333** / 0.946335 · greedy 0.946312 / 0.946312. Winner by the declared tie rule (cells inside 0.00002 → simplest first): `plateau × stack_logit`.

**Strategy:** Inference-only. Load every `oof_v*`/`sub_v*` pair, apply a **declared** eligibility rule instead of hand-picking legs (plateau ≥ best − 0.0005, wide ≥ best − 0.0015, lineage duplicates V18/V23/V29 excluded by rule, ρ > 0.99995 auto-merged), then score five combiners under one protocol: weights and leg sets fitted on nine folds and applied to the tenth, with every arm's in-sample score printed beside its sealed score.
**File:** `Previously trained Files/Archieve/S6E9_V38_EnsembleShootout.py` | **Outputs:** `oof/oof_v38.csv`, `sub/sub_v38.csv`

**Key Learning:**

> A weighted stack carries a transfer defect fold-sealing cannot see: leg OOF rows come from one fold model, leg test rows from the average of ten, so part of the CV advantage pays for noise that does not exist at test time — the fitted weights show it directly (12 of 25 negative, Σ|w| 2.592, V16 −0.279, among legs correlating at ρ 0.9946). Our CV cannot adjudicate it (C from 1.0 to 0.01 moves sealed AUC by 2e-6), which is why the equal-weight variant went in as a daily. NNLS is dead (0.945826) and greedy cost 8.5 min to score 2e-5 behind the stack.

**Status:** ✅ Good (best honest OOF in project history; candidate for a final slot)

---

### Version 37 (Logistic Regression on a sparse one-hot + fold-safe WoE design) - 2026-09-24

**Score**: **0.94335 LB** / 0.943220 OOF (10-fold, C=0.3) (Gap: +0.00013)
**Device**: CPU only, scikit-learn 1.6.1
**Result**: **-0.002951 OOF vs V30**, −0.001457 vs V5's 0.944677 (our best linear leg)

**Timing:**
| Stage | Time |
|-------|------|
| Load + static one-hot vocabulary | ~0.6 min |
| 10 folds × 4 LogisticRegression fits (shared design) | 0.4–0.5 min per fold |
| Total | 5.0 min |

**Fold Scores (AUC at C=0.3):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94336 | 0.94329 | 0.94259 | 0.94163 | 0.94554 | 0.94324 | 0.94292 | 0.94280 | 0.94352 | 0.94339 | 0.943229 ± 0.000933 |

**Strategy:** The one version in the batch that changes the representation instead of the learner — no V19 pipeline at all. A 2,076-column sparse CSR design (1,120 static one-hot levels, exact integer lattices, 900 income + 48 charging quantile bins, 8 smoothed weight-of-evidence terms) into `LogisticRegression(lbfgs, l2)`, every label-dependent statistic fitted on that fold's training rows only, with `assert X.shape[1] == 2076` guarding alignment. C grid {0.3, 1.0, 3.0, 10.0} reused one build per fold.
**File:** `Previously trained Files/Archieve/S6E9_V37_LR_OneHotWoE.py`

**Key Learning:**

> For a linear model on this data **target encoding beats one-hot + WoE by 0.0015**: a fixed lattice can say *which* cell a row is in but not *how often that cell buys*, while TE hands the model that conditional already shrunk. The C grid also corrected my prior — the tightest C won and loosening degraded monotonically (0.943220 → 0.943168). The fold-1 coefficients remain the cleanest reading of the generator's additive law we own (Range=High −1.7728, woe_income +1.1143, woe_subsidy +1.0254).

**Status:** ❌ Failed (cheapest run in the series; kept as the linear-ceiling leg)

---

### Version 36 (Second CV geometry: 20-fold, seed 7) - 2026-09-24

**Score**: **0.94637 LB** / 0.946199 OOF (20-fold) (Gap: +0.00017)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **+0.000028 OOF vs V30** — the fold lever measured twice and found spent

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE (byte-identical to V30) | ~2.6 min |
| 20 folds | 3.7–5.7 min each |
| Total | 95.3 min (2.1× V30 for +0.000028) |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| 0.94736/4273 | 0.94788/4130 | 0.94470/2927 | 0.94604/7943 | 0.94601/4481 | 0.94758/6694 | 0.94528/5504 | 0.94611/5183 | 0.94699/7054 | 0.94461/6598 | 0.94708/5295 | 0.94471/5173 | 0.94764/5951 | 0.94346/4909 | 0.94670/5752 | 0.94901/4168 | 0.94543/5183 | 0.94465/4507 | 0.94605/3546 | 0.94682/5656 | 0.946205 ± 0.001352 |

**Strategy:** Isolate the validation geometry. `N_FOLDS = 20` with `KFold(20, shuffle, random_state=7)` is the only change from V30 — same FE region byte-identical, same config, same early stopping; the count propagates into `TargetEncoder(cv=20)` and into the per-fold original-row concat.
**File:** `Previously trained Files/Archieve/S6E9_V36_XGB_SecondSchema.py`

**Key Learning:**

> The fold curve is concave and now measured at both steps: 5→10 bought +0.000097, 10→20 buys +0.000028 at 2.1× the cost. Because the seed moved with K, "20 folds" and "seed 7 is friendlier" are confounded and the design cannot separate them; the added noise is visible in the run itself (fold SD 0.001352 vs V30's 0.000783, BestIter 2,927–7,943). Geometry axis closed, V30's 10-fold split stays the house ruler.

**Status:** ⚠️ Partial

---

### Version 35 (HistGradientBoosting, native categoricals, first run of the family) - 2026-09-24

**Score**: **0.94572 LB** / 0.945482 OOF (10-fold) (Gap: +0.00024)
**Device**: CPU only, scikit-learn 1.6.1
**Result**: **-0.000689 OOF vs V30** on the same folds and matrix

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE (byte-identical) | ~2.8 min |
| 10 folds | 9.7–12.6 min each |
| Total | 116.7 min |

**Fold Scores (AUC | `n_iter_`):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94563/984 | 0.94537/679 | 0.94513/663 | 0.94417/851 | 0.94732/706 | 0.94555/761 | 0.94502/700 | 0.94509/547 | 0.94614/701 | 0.94549/739 | 0.945492 ± 0.000779 |

**Strategy:** A learner family never run in this competition's history, on the unchanged V30 matrix: `lr 0.02, max_iter 6000, max_leaf_nodes 31, min_samples_leaf 40, l2 1.0, max_bins 255, early_stopping=True`, with categoricals declared **as column indices** (sklearn rejects a bool mask and hard-errors on any declared column whose cardinality exceeds `max_bins`, so 8 of 10 candidates are declared). 320 fitted columns.
**File:** `Previously trained Files/Archieve/S6E9_V35_HistGB.py`

**Key Learning:**

> Fourth distinct learner on this exact matrix and none beats XGBoost (0.946171 / 0.946156 / 0.945987 / 0.94564 / 0.945482) — the ceiling is **representation-bound, not learner-bound**. The logged number is pessimistic because `early_stopping=True` makes sklearn carve its own ~10% holdout inside each training fold, but V32 ran unhandicapped and still lost 0.000184, so no rerun was justified; read it as a floor, not the family's fair score.

**Status:** ❌ Failed (worst time/value ratio in the series; screening rule: a new family must be cheap or competitive)

---

### Version 34 (LightGBM with the published config, bagging actually enabled) - 2026-09-24

**Score**: **0.94638 LB** / 0.946156 OOF (10-fold) (Gap: +0.00022)
**Device**: CPU (LightGBM 4.6.0)
**Result**: **-0.000015 OOF vs V30** — parity, the family's first equal-fold match of XGBoost

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE (byte-identical) | ~2.6 min |
| 10 folds | 5.4–8.0 min each |
| Total | 68.1 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94647/1367 | 0.94615/1733 | 0.94573/1217 | 0.94485/1543 | 0.94807/1160 | 0.94635/1473 | 0.94572/1077 | 0.94561/892 | 0.94660/1334 | 0.94616/1985 | 0.946170 ± 0.000798 |

**Strategy:** Re-run LightGBM with the published winning configuration and — the reason the version exists — with bagging **live**: every LightGBM version in this repo (V3, V10, V19, V26) set `subsample=0.8` while leaving `subsample_freq` at 0, and LightGBM only samples rows when both are set. Config `lr .02, num_leaves 32, max_depth 5, min_data_in_leaf 10, bagging .8/1, feature_fraction .3, λ1 .071, λ2 2.03, max_bin 1024`, 20,000 rounds / ES 500, predictions from the frame with pinned `feature_name`.
**File:** `Previously trained Files/Archieve/S6E9_V34_LGBM_PublishedConfig.py`

**Key Learning:**

> **Parameter names are not documentation.** With bagging genuinely executing, LightGBM ties XGBoost to 1.5e-5, so V19's and V26's family verdicts were produced by a model silently ignoring one of its two stated regularisers — and the tie is the useful outcome: the two families sit on one plateau. BestIter 892–1,985 of 20,000 means this parity figure is converged, and `TE_lift_trigram_cat_auto/10/100` lead again on a learner whose tree-growth machinery differs from the three that led with it before.

**Status:** ✅ Good (kept as the record that LightGBM was never the bottleneck)

---

### Version 33 (XGBoost with zero target encoding, numerics as categorical codes) - 2026-09-24

**Score**: **0.94481 LB** / 0.944967 OOF (10-fold) (Gap: -0.00016)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **-0.001204 OOF vs V30** — the largest single-factor regression in the batch, and a floor

**Timing:**
| Stage | Time |
|-------|------|
| Load + FE with the TE block switched off | ~2.4 min |
| 10 folds at `max_bin=8192` | 8.8–9.5 min each (2× V30) |
| Total | 94.7 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94522/7997 | 0.94483/8000 | 0.94459/8000 | 0.94356/7997 | 0.94679/8000 | 0.94502/8000 | 0.94479/7999 | 0.94464/8000 | 0.94531/7998 | 0.94503/7982 | 0.944978 ± 0.000762 |

**Strategy:** Change only the input representation, to test V32's leftover hypothesis that our encoded columns crowd the learner out. `USE_TRIPLE_TE = False` removes all 198 encoded columns while the 66 source keys stay as integer codes from a global factorize over train+orig+test; the one accompanying parameter change is `max_bin 1024 → 8192`, to buy back resolution. 180 fitted columns instead of 312.
**File:** `Previously trained Files/Archieve/S6E9_V33_XGB_NoTE_Views.py`

**Key Learning:**

> **The TE block *is* the score, priced at −0.001204** — six times what any learner, geometry, bin-resolution or additive-prior change has moved. Read it as a floor, not a point estimate: nine of ten folds terminated on the 8,000-round cap, so the exact magnitude is confounded with undertraining and no decision changes at either value. The model rediscovers the same structure without the encodings at lower resolution (`_ECL_x_Subsidy_cat` 20,273.9 leading 60× the third), and the board compressed a real 0.0012 regression into −0.00016.

**Status:** ❌ Failed (prices target encoding; truncation warning for any future zero-TE run)

---

### Version 32 (CatBoost with valid parameters, published tuned recipe) - 2026-09-24

**Score**: **0.94609 LB** / 0.945987 OOF (10-fold) (Gap: +0.00010)
**Device**: GPU, catboost 1.2.10
**Result**: **-0.000184 OOF vs V30** on the same folds, matrix and FE

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE (byte-identical) | ~2.6 min |
| Config probe (fold 1, 5,000 rows, 20 iterations) | seconds |
| 10 folds | 2.8–4.2 min each |
| Total | 37.5 min |

**Fold Scores (AUC | BestIter):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94617/3466 | 0.94584/5297 | 0.94563/2360 | 0.94455/3926 | 0.94798/5127 | 0.94627/5821 | 0.94559/3682 | 0.94546/3192 | 0.94644/3285 | 0.94605/3937 | 0.945997 ± 0.000832 |

**Strategy:** Re-run CatBoost properly. V24 passed `'max_bin': 1024`, a parameter **CatBoost does not have** (`border_count` is the name), so the family verdict of 09-19 came from a run with a silently ignored knob — while a published tuned CatBoost in this episode claims CV 0.94621. Learner change only: `iterations 100000, lr 0.015, depth 6, l2_leaf_reg 3.0, border_count 1024, od_type Iter, od_wait 800`, with a fold-1 probe to settle the `rsm` bootstrap conflict.
**File:** `Previously trained Files/Archieve/S6E9_V32_CatBoost_Tuned.py`

**Key Learning:**

> The family verdict survives the parameter fix, so it was never a parameter problem: 0.945987 against XGBoost's 0.946171 on the identical matrix, and CatBoost converts the extra training data into less than XGBoost does (+0.00005 vs +0.000097 from 5→10 folds). The probe earned its keep — `rsm` is rejected by catboost 1.2.10 on GPU, so the logged number is the **Bernoulli/subsample=0.4 fallback**, one degree of freedom short of the published recipe. `lift_trigram` TEs lead again: artefact dependence is family-independent.

**Status:** ❌ Failed (family closed on valid parameters; keep the probe pattern)

---

### Version 31 (Additive logistic backbone as XGBoost `base_margin`) - 2026-09-24

**Score**: **0.94630 LB** / 0.946224 OOF (10-fold) (Gap: +0.00008)
**Device**: GPU (cuda), XGBoost 3.2.0
**Result**: **+0.000053 OOF vs V30** at identical folds, matrix and booster config

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE (byte-identical) | ~2.6 min |
| Backbone (31-col LogisticRegression, inner KFold(5) per fold) | included per fold |
| 10 folds | 3.2–3.9 min each |
| Total | 37.8 min |

**Fold Scores (AUC | trees):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.94643/3773 | 0.94616/3985 | 0.94584/4105 | 0.94493/3588 | 0.94815/5177 | 0.94632/4518 | 0.94584/3298 | 0.94576/4157 | 0.94665/4880 | 0.94621/4725 | 0.946229 ± 0.000783 |

**Strategy:** Change the formulation, not the features. A purely additive `LogisticRegression` on 31 columns (7 numerics + 3 log1p + 4 smooth-key ordinals + one-hot of the 6 originals; no TE/lift/digit/`_fe`) is fitted per fold — out-of-fold for training rows via an inner `KFold(5)`, full-fit for validation and test — and its logits go to XGBoost as `base_margin`, so every tree must earn gain over the additive fit. Motivation measured the same day: on the original 10k rows a depth-1 boosted model is the *best* configuration (0.90676), i.e. the generator's label law is additive.
**File:** `Previously trained Files/Archieve/S6E9_V31_XGB_AdditiveResidual.py`

**Key Learning:**

> **An offline proxy may test the implementation, not the idea.** A same-day reduced-feature proxy said this formulation *costs* −0.0020, so V31 ranked near last; on the real matrix it gained +0.000053. The reversal's mechanism: the proxy's backbone scored 0.9375 and displaced tree capacity, this one reaches **0.938310** against the measured additive ceiling for these features (HistGB depth-1 0.93847) and so informs instead of competing. BestIter fell to 3,298–5,177 because the residual model spends its rounds only on interaction and artefact structure.

**Status:** ✅ Good (best honest CV until V44; the board billed it −0.00009, divergence rule again)

---

### Version 30 (Plain single model: V19 FE + 10-fold XGBoost) - 2026-09-21

**Score**: **0.94639 LB** / 0.946171 OOF (10-fold) (Gap: +0.00022)
**Device**: GPU (cuda)
**Result**: **+0.000097 OOF vs V28's 5-fold figure**, -0.00002 LB

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~2.3 min |
| 10 folds × 4.0–4.7 min | 42.9 min |
| Total | 45.3 min |

**Fold Scores (10-fold, 66,866 validation rows each):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean ± SD |
|----|----|----|----|----|----|----|----|----|-----|-----------|
| 0.94640 | 0.94617 | 0.94570 | 0.94488 | 0.94810 | 0.94637 | 0.94572 | 0.94565 | 0.94663 | 0.94621 | 0.946182 ± 0.000799 |

**Strategy:** The plainest script in the series, deliberately: one matrix, one estimator, one loop — no arms, no harness, no blend, no refit, no prior. The only difference from V22's winner is `N_FOLDS = 10`, so every model trains on 90% of the labels instead of 80%. Verified before shipping: FE section byte-identical to V28 by sha256, parameter dict identical, and the training call switched to the booster API because V22's sklearn-wrapper path ranks the test board ~1,600 positions differently on the same config.
**File:** `Previously trained Files/Archieve/S6E9_V30_XGB_10Fold.py`

**Key Learning:**

> The fold count is real but smaller than advertised (pooled 0.946171 against the 5-fold 0.946074, +0.000097 where the externally measured step is +0.00015), and the two numbers are not the same quantity — different validation rows, 12.5% more training rows per model, a 10-fold nested TargetEncoder. What is not ambiguous: the LB−OOF gap collapsed from a stable +0.00032/+0.00035 to **+0.00022**, because a 10-fold OOF is closer to the ensemble-of-ten that actually predicts test. The −0.00002 LB is exactly where the divergence rule places it (divergence 1,111, neighbours at 917–1,140 score 0.94638–0.94641).

**Status:** ✅ Good (best model built; V23 keeps the best score, V30 the best estimate)

---

### Version 29 (TabM + in-run XGBoost control, dual view) - 2026-09-21

**Score**: **0.94640 LB** / 0.94607 OOF (Gap: +0.00033) — the saved model is the control, so this repeats V22's score
**Device**: GPU (cuda), PyTorch 2.10.0+cu128
**Result**: **TabM OOF 0.94564 (-0.00043 vs the control)**; blend number lost to a reporting bug

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~2.0 min |
| View A — XGBoost control, 5 folds | 10.1 min |
| View B — TabM, 5 folds | 12.7 min |
| Total | 32.4 min |

**Fold Scores (XGBoost control / TabM / rank ρ between them):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|----|
| 0.94613/0.94584/0.99619 | 0.94515/0.94484/0.99642 | 0.94717/0.94684/0.99570 | 0.94563/0.94529/0.99578 | 0.94636/0.94590/0.99514 | 0.94609/0.94574 |

**Strategy:** Answer the one family question never tested — no neural model had run on the V19+ artefact matrix (V6 used 83 pre-V19 features, V17 ran on the corrupted digit block). Both views train in one script under the single-model rule, the NN seeing per fold only the top 120 columns by that fold's own freshly trained control gain, blended with one cross-fitted linear weight.
**File:** `Previously trained Files/Archieve/S6E9_V29_TabM_DualView.py`

**Key Learning:**

> **Weak and correlated is the worst combination for a blend:** TabM reached 0.94564 — behind the control *and* behind V6's 0.94585 on the older, smaller feature set — while ranking 0.9951–0.9964 with the GBM, so it predicts ≈ +0.00005, below the gate a rerun would justify. Two things survive: TabM is cheap (12.7 min for five folds), and of the three saved copies of V22's estimator, V28's and V29's controls rank **identically** — the pipeline is reproducible across sessions including the GPU.

**Status:** ❌ Failed (family closed as a source of gain). **Warning:** `oof_v29.csv`/`sub_v29.csv` are saved through an unconditional sigmoid — monotone, so AUC and rank hold, but never feed them to anything that reads values.

---

### Version 28 (XGBoost bin resolution: max_bin 1024 → 16384) - 2026-09-21

**Score**: **0.94640 LB** / 0.94607 OOF (Gap: +0.00033) — CV-only probe, deliberately not promoted
**Device**: GPU (cuda)
**Result**: **-0.000002 OOF vs the in-run control** (z = -0.40, tie)

**Timing:**
| Stage | Time |
|-------|------|
| Load + full FE | ~1.5 min |
| a0 control (max_bin 1024), 5 folds | 10.7 min |
| a1 raised (max_bin 16384), 5 folds | 45.3 min (4.2× control) |
| Total | 66.5 min |

**Fold Scores (winner a0):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|----|
| 0.94613 | 0.94515 | 0.94717 | 0.94563 | 0.94636 | 0.94609 |

**Strategy:** Change exactly one parameter on the real 312-column pipeline with an in-run control that must reproduce V22 (it did). Income holds 14,667 distinct values and is the only column of 13 above a 1024-bin cap, so 16384 is unconstrained binning — the last point on the axis, chosen after a first run tested 4096 (−0.00000, z = −0.21).
**File:** `Previously trained Files/Archieve/S6E9_V28_XGB_BinResolution.py`

**Key Learning:**

> The resolution axis is closed **by construction**: an integrity guard read the answer off the fitted trees (income-derived columns used 6,456 distinct thresholds at 1024 bins vs 7,972 at 16384), so the tie is a real null and not a silently ignored setting. That matches every offline proxy — the community's +0.00201 was raised on a 0.94172 baseline lacking the digit block and exact-value TE stack, i.e. a lever we had already banked in V19. 4.2× cost for zero gain retires the knob: `max_bin` stays 1024.

**Status:** ⚠️ Partial (CV-only null by design)

---

### Version 27 (XGBoost structural ablation: original-row weight + pairwise loss) - 2026-09-20

**Score**: **0.94638 LB** / 0.94608 OOF (Gap: +0.00030)
**Device**: GPU (cuda)
**Result**: **-0.00003 LB vs V23** — probe only, not promoted

**Timing:**
| Stage | Time |
|-------|------|
| Stage A, 5 folds × 5 arms | 66.7 min |
| Stage B | skipped (no arm cleared z > 3) |
| Total | 69.6 min |

**Fold Scores (winner a1, rs=42):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|----|
| 0.94613 | 0.94514 | 0.94719 | 0.94563 | 0.94635 | 0.94609 |

**Strategy:** Third use of the ablation harness, on the two structural assumptions every version since V1 shared. a0 = V20/V23's config as control, a1 drops the original 10k rows from the per-fold concat, a2 keeps them at sample weight 10, a3 swaps to `rank:pairwise` (rows shuffled into contiguous random groups of 64) early-stopping on `auc`, a4 = a3 with the original rows replicated 10× (this build rejects `weight` with `set_group`).
**File:** `Previously trained Files/Archieve/S6E9_V27_XGB_StructuralAblation.py`

**Key Learning:**

> **(1) The original 10k rows are not earning their concat** — dropping them is +0.00002 at z = +2.66, a tie in the right direction with 1.47% of the matrix removed; up-weighting them ×10 goes the other way (−0.00003, z = −2.30). Every original-data claim since V1 is really about the pool/original *frequency* features. **(2) The pairwise-loss theory is dead:** `rank:pairwise` loses 0.000329 at z = −42.56 and is unstable fold to fold — pointwise logloss already carries the ranking, because the model's calibration *is* its ranking.

**Status:** ⚠️ Partial (both questions answered; no submission candidate)

---

### Version 26 (LightGBM ExtraTrees-bias probe) - 2026-09-20

**Score**: **0.94637 LB** / 0.94608 OOF (Gap: +0.00029)
**Device**: CPU (LightGBM 4.6)
**Result**: **-0.00004 LB vs V23** — first arm ever to clear our own z > 3 gate

**Timing:**
| Stage | Time |
|-------|------|
| Stage A, 5 folds × 4 arms | 206.3 min |
| Stage B (3 arms, rs=7) | 177.9 min |
| Total | 387.6 min |

**Fold Scores (winner a3, rs=42):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|----|
| 0.94611 | 0.94517 | 0.94721 | 0.94564 | 0.94637 | 0.94610 |

**Strategy:** Test the one inductive bias never used here — randomised split thresholds — on the reasoning that a pool rejection-sampled from 10k rows makes the label close to a lookup on exact input values. a0 = V19's LightGBM params as control, a1 = a0 + `extra_trees=True` (+ `split_histogram_sampling`, LightGBM 4.x's implementation), a2 = a1 + random-forest lookup capacity (unlimited depth, 128 leaves), a3 = a1 + V22's wide/weak/many direction (depth 3, 8 leaves, min_child 50, colsample 0.85, lr 0.01). Stage B ran *because* arms cleared the gate.
**File:** `Previously trained Files/Archieve/S6E9_V26_LGBM_ExtraTreesProbe.py`

**Key Learning:**

> a1 beat the greedy control +0.00007 (z = +3.58) and a3 **+0.00010 at z = +5.22 on rs=42, +0.00008 at z = +4.11 on rs=7** — the only two-split-reproduced tuning gain since V19 — while the lookup arm failed hard (a2 −0.00023, z = −8.00), so random thresholds help as regularisation of a shallow model, not as memorisation. **The catch, now closed: it beat LightGBM, not XGBoost**, and V46/V50 measured the same mechanism inside the full matrix at +0.000006 and a null, which reattributes the +0.00010 to the shallow/wide config.

**Status:** ⚠️ Partial (gain real inside LightGBM at 6.5 h of CPU; superseded as a mechanism by V46/V50)

---

### Version 25 (XGBoost factorial ablation: cross keys / recipe prior / gain pruning) - 2026-09-19

**Score**: **0.94628 LB** / 0.94609 OOF (Gap: +0.00019) — our best OOF then, and a broken OOF→LB relation
**Device**: GPU (cuda)
**Result**: **-0.00012 LB vs V22** — submission reverted to the V23 line

**Timing:**
| Stage | Time |
|-------|------|
| Stage A, 5 folds × 5 arms (rs=42) | 53.2 min |
| Stage B (3 arms, rs=7) | 33.8 min |
| Total | 90.2 min |

**Fold Scores (winner a2, rs=42):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|----|
| 0.94612 | 0.94512 | 0.94721 | 0.94570 | 0.94635 | 0.94610 |

**Strategy:** Instead of bundling three ideas into one model, run them as a factorial ablation in one script: one feature build per fold shared by all arms, an in-run control equal to V22's winner, and four single-factor arms — a1 V21's six cross keys as a pure addition, a2 per-row `base_margin` from a logistic fitted on the original 10k rows only (clipped ±5), a3 in-run gain pruning to 94 features, a4 all three. Stage A on rs=42, top-3 re-run on rs=7, every arm judged against the control by paired DeLong from this run's own OOF.
**File:** `Previously trained Files/Archieve/S6E9_V25_XGB_FactorialAblation.py`

**Key Learning:**

> a0 reproduced V22 to 6e-6, which validated the harness; the arms then acquitted V21's crosses of blame and rejected them on merit (a1 −0.00006, z = −3.98), refused the gain pruning (a3 −0.00004), and left only the recipe prior at tie strength (a2 +0.00002, z = +1.29; rs=7 +0.00004, z = +2.45). **I promoted that tie to a submission and it cost 0.00012 LB** — forensics on the saved vectors show no code error: the prior moved +0.0324 in the bottom decile and ±0.0012 above it, and those deciles hold 0.15% of the AUC's pair mass, i.e. it reordered where the metric is blind. **Rule adopted: a submission slot only for an arm clearing z > 3, or for an inference-side variant; ties stay CV experiments.**

**Status:** ⚠️ Partial (best OOF to date; LB regressed because a tie was submitted)

---

### Version 24 (CatBoost depth=6 on the V19/V20 Artifact Matrix) - 2026-09-19

**Score**: **0.94615 LB** / 0.94593 OOF (Gap: +0.00022)
**Device**: GPU (CatBoost, task_type=GPU)
**Result**: **-0.00024 LB vs V20 — REJECTED**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 134 s |
| Fold 2 | 126 s |
| Fold 3 | 132 s |
| Fold 4 | 135 s |
| Fold 5 | 140 s |
| Total | 13.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94592 | 0.94497 | 0.94707 | 0.94552 | 0.94623 | 0.94594 |

**Strategy:** Third isolation test — swap the estimator family only. V19's artifact matrix (fixed digits, 12 lift/novelty columns, 2 structural flags, V10 bigrams + groupby deviations, 180 base + 198 Triple TE) trained with CatBoost `depth=6`, `lr=0.03`, `l2_leaf_reg=3.0`, `min_data_in_leaf=20`, Bernoulli bootstrap 0.8, `random_strength=1.0`, `max_bin=1024`, Iter/500 early stop, `use_best_model=True`. Best iterations 1243-1549. Same KFold(5, shuffle=True, rs=42) and per-fold original-data concat.
**File:** `Previously trained Files/Archieve/S6E9_V24_CatBoost_Artifacts.py`

**Key Learning:**
> OOF 0.94593 vs V20's 0.94606 (-0.00013) and LB 0.94615 vs 0.94639 — CatBoost is still behind XGBoost on this matrix even after the artifact features, continuing the V4/V8 ordering. It confirms the family axis is closed: XGBoost depth-4 > LightGBM > CatBoost > TabM/RealMLP on identical features. The feature story replicated exactly though — `TE_lift_trigram_cat_10` (8.20) and `_auto` (8.18) took the top two importance slots, so the lift trigram is the dominant signal in every family, not a LightGBM/XGBoost artifact. Fastest full run to date (13.7 min vs V20's 20.6 min), and CatBoost's shrinkage stopped at ~1.5k iterations against XGBoost's ~4k, so it is a cheap candidate generator, not a score lever.

**Status:** ❌ Failed

---

### Version 23 (XGBoost depth=4 + Full-Data Refit Test Blend) - 2026-09-19

**Score**: **0.94641 LB** / 0.94606 OOF (Gap: +0.00035)
**Device**: GPU (cuda)
**Result**: **+0.00002 LB vs V20 — NEW BEST LB**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 200 s |
| Fold 2 | 219 s |
| Fold 3 | 190 s |
| Fold 4 | 189 s |
| Fold 5 | 223 s |
| CV subtotal | 17.0 min |
| Full-data refit | 2.2 min |
| Total | 23.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94612 | 0.94513 | 0.94715 | 0.94564 | 0.94634 | 0.94608 |

**Strategy:** V20's CV loop byte-for-byte (same features, same params, same rs=42 split), then a `[3b]` inference stage: refit one model on 100% of train + original (678,665 x 312 vs the folds' 544,932) with `n_estimators` fixed at the mean best iteration (4112; iterations [4230, 4501, 3375, 3470, 4984]) and no early-stopping set, then blend `test_probs = 0.50 x fold-average + 0.50 x refit`. OOF remains purely out-of-fold, so the reported CV is the V20 number by construction.
**File:** `Previously trained Files/Archieve/S6E9_V23_XGB_FullDataRefit.py`

**Key Learning:**
> +0.00002 LB with the OOF untouched — the first movement we have produced from the inference side, and it came with a diagnostic that says the refit is only mildly decorrelated from the fold average (Pearson 0.99959, Spearman 0.99946, mean |rank difference| 1,806 of 286,571 rows). Two things follow: 25% more labels per tree is worth something even at fixed depth, and the surviving 0.0004 of disagreement between the two predictors is where inference-side variance lives, so weight and iteration-count choices on that axis are still open. Fold importances matched V20 within rounding (`TE_lift_trigram_cat_auto` 0.14488), confirming the CV path really was unchanged.

**Status:** 🏆 Best

---

### Version 22 (XGBoost Two-Stage Estimator Search on Our Matrix) - 2026-09-19

**Score**: **0.94640 LB** / 0.94608 OOF (Gap: +0.00032)
**Device**: GPU (cuda)
**Result**: **+0.00001 LB vs V20 (new best OOF for a single model)**

**Timing:**
| Stage | Time |
|-------|------|
| Stage A (6 configs x 5 folds, rs=42) | 75.9 min |
| Stage B (3 configs x 5 folds, rs=7) | 37.9 min |
| Total | 116.8 min |

**Fold Scores (winner c1, rs=42):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94611 | 0.94517 | 0.94717 | 0.94563 | 0.94637 | 0.94609 |

**Strategy:** V20 used a stranger's tuned recipe, so the estimator was never tuned on our own matrix. Six configs sharing one build-per-fold (c0 = V20 control, c1 depth 3 / 8 leaves / gamma 1.0 / colsample 0.85, c2 deeper + heavy leaf reg, c3 sparse cols + high gamma, c4 low min_child_weight, c5 tiny lr near-depthless lossguide), Stage A on the canonical rs=42 split, top-3 re-run on rs=7, winner by two-split mean. Submission averaged the winner's test predictions across both splits; OOF is the rs=42 winner.
**File:** `Previously trained Files/Archieve/S6E9_V22_XGB_Tuning.py`

**Key Learning:**
> Stage A ranking: c1 0.94608, c0 0.94606, c4 0.94605, c2 0.94593, c3 0.94576, c5 0.94546. The winner was confirmed on a second split (rs=7: c1 0.94608 / c0 0.94607 / c4 0.94605), so the +0.00002 over the in-run control is real but tiny — and the in-run control reproduced V20's OOF exactly, which validates the harness. The useful negative results are directional: deeper/heavier-regularised trees and tiny learning rates are clearly worse (-0.00013 / -0.00060), while depth 3 with 3x the columns and 5,000-6,600 trees is marginally better, i.e. this matrix wants wide, weak, many learners. c1 also redistributed gain (in its rs=7 fold-1 table the lift-trigram auto+10 pair alone took 0.419 of gain versus 0.252 for c0 on rs=42, and raw `Environmental_Concern_Level` entered at #3), which is the V21 lesson reversed: a shallower, column-rich model leans even harder on the pre-computed artifact crosses.

**Status:** ✅ Good

---

### Version 21 (XGBoost + Explicit Cross Keys, digits/flags removed) - 2026-09-19

**Score**: **0.94629 LB** / 0.94597 OOF (Gap: +0.00032)
**Device**: GPU (cuda)
**Result**: **-0.00010 LB vs V20 — REJECTED by the accept gate**

**Timing:**
| Stage | Time |
|-------|------|
| Fold 1 | 186 s |
| Fold 2 | 181 s |
| Fold 3 | 169 s |
| Fold 4 | 176 s |
| Fold 5 | 183 s |
| Total | 16.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94607 | 0.94505 | 0.94710 | 0.94548 | 0.94624 | 0.94599 |

**Strategy:** V20's model unchanged, feature budget reallocated. Added six explicit cross keys (charging-total x Home_Charging, City x Home_Charging, ECL x Home_Charging, ECL x City, income-band x ECL x Subsidy, ECL x Subsidy x commute with the 5.0 km cluster isolated), each Triple-TE'd like the V10 bigrams plus its own target-free generator-lift column. Removed the whole digit block and four income-anomaly flags that had earned <= 0.08% gain in V20. Fixed bin edges in CFG so train/test/orig bin identically. 159 base + 189 TE = 348 features (V20: 180 + 198 = 378).
**File:** `Previously trained Files/Archieve/S6E9_V21_XGB_CrossKeys.py`

**Key Learning:**
> Paired DeLong vs V20: delta -0.00009, z = -5.01 — a significant regression, and LB agreed (0.94629 vs 0.94639). The addition worked and the subtraction broke it: `cx_inc_x_ecl_x_sub` and its lift took five of the top eight gain slots, proving the crosses carry real signal. But dropping the digit block and flags shrank the space (only 27 columns dropped as redundant vs 149 in V20) and the gain distribution collapsed — `_ECL_x_Subsidy` seized 0.5899 of total gain (5.4x its V20 share) while the lift-trigram family that had been #1 in V20 fell from ~35% combined to ~6.7%. Bundling an addition with a removal made the failure unattributable beyond that; lesson: one change per version, and a shallow lossguide model needs a wide candidate pool to keep any single feature from monopolising splits.

**Status:** ❌ Failed

---

### Version 20 (XGBoost depth=4 lossguide + V19 Artifact Features) - 2026-09-19

**Score**: **0.94639 LB** / 0.94606 OOF (Gap: +0.00033)
**Device**: GPU (cuda)
**Result**: **±0.00000 LB vs V19 (new best OOF for a single model)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 20.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94612 | 0.94513 | 0.94715 | 0.94564 | 0.94634 | 0.94608 |

**Strategy:** One variable changed from V19 — model family. V19's feature matrix is reused byte-for-byte (fixed integer digit extraction, 12 target-free generator-lift/novelty columns, 2 structural flags, V10 bigrams + groupby deviations, Triple TE on 66 columns = 180 base + 198 TE), trained with najiama's tuned XGBoost recipe: `max_depth=4`, `grow_policy=lossguide`, `max_leaves=16`, `gamma=3.673`, `min_child_weight=4.532`, `subsample=0.740`, `colsample_bytree=0.570`, `alpha=0.752`, `lambda=0.619`, `lr=0.01`, ES=500, converging at 3375-4984 trees. 5-fold `KFold(shuffle=True, rs=42)` kept identical to V19 so the OOF stays comparable.
**File:** `Previously trained Files/Archieve/S6E9_V20_XGB_ArtifactFeatures.py`

**Key Learning:**

> Gave our first statistically significant OOF gain: paired DeLong vs V19 Δ +0.00007 at SE 0.00002 (z = +4.76), and vs V10 Δ +0.00009 (z = +5.25); statistically tied with V14's long-standing best OOF (Δ -0.00002, z = -0.83). LB still landed on 0.94639 exactly — public LB covers 20% of test and cannot resolve a 0.00007 change, so OOF significance and LB movement are now decoupled. `TE_lift_trigram_cat_auto` became the single most important feature by gain (0.1449) versus #4 under LightGBM, confirming that a shallower model leans harder on pre-computed artifact crosses. Structural flags and digits were near-unused here (flat, tiny gains), matching the reference notebook's decision to prune them. Test predictions correlate 0.99982 with V19's, and their equal-weight average scores 0.94606 — no better than V20 alone. Also 40% faster than V19 (20.6 min vs 33.7 min).

**Status:** ✅ Good

---

### Version 19 (LightGBM + Generator-Artifact Features) - 2026-09-19

**Score**: **0.94639 LB** / 0.94599 OOF (Gap: +0.00040)
**Device**: CPU (LightGBM)
**Result**: **+0.00003 LB over V10 — new best**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94603 | 0.94507 | 0.94710 | 0.94555 | 0.94627 | 0.94600 |

**Strategy:** V10 Modified pipeline plus three self-contained upgrades: (1) digit extraction fixed with integer divmods on `round(col*1e4)` — the `// (10**k)` negative-k float floor-div bug had corrupted 89.63% of commute tenths in all 18 prior versions; (2) 12 target-free generator-lift features (train+test pool freq ÷ original freq) on exact income, 100-dollar income band, integer commute, 6 categoricals, and the ECL×Subsidy×Anxiety trigram, plus a `novel_inc` flag; (3) two structural flags from the decoded recipe (`_below_buy_bound` at 41,667, `_dead_zone_exact` 31,004-41,970). 180 base + 198 Triple TE = 378 features, 5-fold, original-data concatenation, V10 LightGBM params unchanged.
**File:** `Previously trained Files/Archieve/S6E9_V19_LGBM_ArtifactFeatures.py`

**Key Learning:**

> First real gain in 9 versions. `TE_lift_trigram_cat_auto` entered at #4 overall (531) with 11 of the top-15 artifact features being lift/novelty columns, confirming the generator's frequency shaping leaks target signal that no model had access to before. The OOF moved only +0.00002 (0.94599 vs 0.94597) while LB moved +0.00003 — fold std held at 0.00069, so the win is small but consistent across both metrics.

**Status:** 🏆 Best

---

### Version 18 (Hill Climber Ensemble) - 2026-09-19

**Score**: **0.94635 LB** / 0.94623 OOF (Gap: +0.00012)
**Device**: CPU (hill climbing over saved OOFs)
**Result**: **-0.00001 LB vs V10 best**

**Timing:**
| Stage | Time |
|-------|------|
| Total | ~10 min (300k-subsampled search; full-data grid search is much slower) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| N/A | N/A | N/A | N/A | N/A | 0.94623 |

**Strategy:** Forward-stepwise hill climbing (weight step 0.05) over all 17 saved OOF predictions, plus simple/rank/logit averages and a Ridge meta-learner for comparison. Selected weights: V14=0.35, V3=0.28, V11=0.15, V6=0.13, V2=0.05, V8=0.05. OOF was fit directly by the hill climber, so the 0.94623 OOF is optimistically biased.
**File:** `Previously trained Files/Archieve/S6E9_V18_HillClimber.py`

**Key Learning:**

> Hill climbing pushed OOF to 0.94623 (best single was 0.94608) but LB only reached 0.94635 — 0.00001 below V10's 0.94636. Members correlate 0.996–0.999, so honest combination gains are ≤0.000066; the OOF gain was weight-fitting noise, not real signal. Ensembling within this model pool is exhausted.

**Status:** ⚠️ Partial

---

### Version 17 (RealMLP) - 2026-09-11

**Score**: **0.94612 LB** / 0.94582 OOF (Gap: +0.00030)
**Device**: CUDA (RealMLP/PyTorch)
**Result**: **±0.00070 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 97.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94591 | 0.94490 | 0.94699 | 0.94553 | 0.94624 | 0.94591 |

**Strategy:** 5-fold CUDA RealMLP using PBLD embeddings, an 8-model ensemble, three hidden layers of 256 units, EMA, label smoothing, and three epochs. The model used the V14 full feature pipeline, original-data concatenation, Triple TE, and 293 expanded model inputs.
**File:** `Previously trained Files/Archieve/S6E9_V17_RealMLP.py`

**Key Learning:**

> RealMLP tied V9 at 0.94612 LB but required 97.3 minutes and remained below V14’s 0.94630. The full engineered representation was viable, but the neural approach did not beat the tree models.

**Status:** ✅ Good

---

### Version 16 (XGBoost Forensic-Targeted Features) - 2026-09-11

**Score**: **0.94625 LB** / 0.94595 OOF (Gap: +0.00030)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00068 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94597 | 0.94505 | 0.94709 | 0.94553 | 0.94619 | 0.94597 |

**Strategy:** 5-fold GPU XGBoost depth 3 based on V14 without pseudo-labels, adding five forensic-targeted features: subsidy × home charging, recipe × subsidy/income, and fold-safe buyer-centroid distances within ECL subgroups. Final training used 164 features.
**File:** `Previously trained Files/Archieve/S6E9_V16_Forensic_Targeted.py`

**Key Learning:**

> V16 reached 0.94625 LB and 0.94595 OOF, below V14’s 0.94630 LB. The strongest forensic signal was `TE_bigram_Sub_HomeCharging`, but all forensic features had small overall fold-1 importance.

**Status:** ✅ Good

---

### Version 15 (KNN Exact-Match + KDTree) - 2026-09-10

**Score**: **0.91256 LB** / 0.91359 OOF (Gap: -0.00103)
**Device**: CPU (KNN with KDTree)
**Result**: **±0.00086 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.91462 | 0.91293 | 0.91338 | 0.91248 | 0.91456 | 0.91359 |

**Strategy:** 5-fold CPU KNN using a 24-feature encoded representation, exact-match lookup when available, and k=10 KDTree neighbor averaging as fallback. Validation and test exact-match rates were both 0%.
**File:** `Previously trained Files/Archieve/S6E9_V15_KNN.py`

**Key Learning:**

> KNN achieved only 0.91256 LB and 0.91359 OOF. With no exact matches and a -0.00103 LB–OOF gap, this approach was not competitive for the dataset.

**Status:** ❌ Failed

---
### Version 14 (Pseudo-Labeling XGBoost Depth 3) - 2026-09-10

**Score**: **0.94630 LB** / 0.94608 OOF (Gap: +0.00022)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00068 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94609 | 0.94521 | 0.94721 | 0.94560 | 0.94633 | 0.94609 |

**Strategy:** 5-fold GPU XGBoost depth 3 using V10 teacher predictions to add 150,859 high-confidence pseudo-labeled test rows: 999 positive and 149,860 negative labels at weight 0.5. The student used V12-style features, original-data concatenation, and 159 final columns.
**File:** `Previously trained Files/Archieve/S6E9_V14_PseudoLabel_XGB.py`

**Key Learning:**

> Pseudo-labeling improved the V12 LB from 0.94629 to 0.94630 and reached third place overall, but the gain was only +0.00001. The LB–OOF gap was +0.00022.

**Status:** ✅ Good

---

### Version 13 (DCN-V2) - 2026-09-09

**Score**: **0.94568 LB** / 0.94490 OOF (Gap: +0.00078)
**Device**: CUDA (PyTorch)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94503 | 0.94415 | 0.94612 | 0.94469 | 0.94523 | 0.94504 |

**Strategy:** 5-fold CUDA DCN-V2 with 4 cross layers, rank 64, 4 experts, and the V6 evidence-based 79-feature selection. Features combined V3 inputs with V10/V12 proven targeted interactions; the final input contained 18 categorical and 61 numerical features.
**File:** `Previously trained Files/Archieve/S6E9_V13_DCN_V2.py`

**Key Learning:**

> V13 tied V7 at 0.94568 LB but had a much larger +0.00078 LB–OOF gap. The DCN-V2 cross architecture was fast but did not outperform the tree-based or TabM approaches.

**Status:** ✅ Good

---

### Version 12 (XGBoost Depth 3 Deotte-style) - 2026-09-09

**Score**: **0.94629 LB** / 0.94598 OOF (Gap: +0.00031)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00069 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94600 | 0.94507 | 0.94713 | 0.94554 | 0.94625 | 0.94600 |

**Strategy:** 5-fold stratified GPU XGBoost with depth 3, `lr=0.0025`, up to 50,000 estimators, V3 full FE, V10/V11 proven targeted features, and inner K-fold target encoding on 67 columns. All 10 discussion patterns were implemented except the recipe score, which was excluded after V11’s cannibalization finding. Final feature count was 237.
**File:** `Previously trained Files/Archieve/S6E9_V12_XGB_Depth3_Deotte.py`

**Key Learning:**

> V12 reached 0.94629 LB, tying V11 for third overall. The 3-way `Subsidy × ECL × Range Anxiety` target encoding was the strongest discussion-derived feature, but the shallow model remained below V10 and V3.

**Status:** ✅ Good

---

### Version 11 (LightGBM Deep Analysis Features) - 2026-09-09

**Score**: **0.94629 LB** / 0.94599 OOF (Gap: +0.00030)
**Device**: CPU only (LightGBM)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94601 | 0.94516 | 0.94705 | 0.94555 | 0.94626 | 0.94601 |

**Strategy:** 5-fold stratified CPU LightGBM extending corrected V10 with six deep-analysis features: `_never_buy_flag`, `_always_buy_flag`, `_income_round00`, `trigram_Sub_ECL_RA`, `_recipe_residual`, and `bigram_income_band_x_Subsidy`. The model used 173 base features plus 201 Triple TE features, 374 total.
**File:** `Previously trained Files/Archieve/S6E9_V11_LGBM_DeepAnalysis.py`

**Key Learning:**

> V11 reached 0.94629 LB and third place overall. `_recipe_residual` was the strongest fold-1 signal, while trigram target encodings added useful interaction structure; V11 remained 0.00007 below V10.

**Status:** ✅ Good

---

### Version 10 (LightGBM Targeted Features) - 2026-09-09

**Score**: **0.94636 LB** / 0.94597 OOF (Gap: +0.00039)
**Device**: CPU only (LightGBM)
**Result**: **±0.00067 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 36.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94601 | 0.94506 | 0.94705 | 0.94554 | 0.94627 | 0.94598 |

**Strategy:** 5-fold stratified CPU LightGBM based on V3, preserving the core feature set while adding the missing `ECL_bin × RangeAnxiety` and `ECL_bin × Subsidy` bigrams plus 12 selective groupby deviation features. The final model used 168 base features and 195 Triple TE features, 363 total.
**File:** `Previously trained Files/Archieve/S6E9_V10_LGBM_Groupby_BigramTE.py`

**Key Learning:**

> The corrected targeted-feature V10 reached 0.94636 LB, a new best and +0.00002 above V3. Targeted bigram encodings and income/ECL-bin groupby deviations provided useful signal without the larger 479-feature expansion.

**Status:** 🏆 Best

---

### Version 9 (XGBoost Full Feature Engineering) - 2026-09-09

**Score**: **0.94612 LB** / 0.94584 OOF (Gap: +0.00028)
**Device**: GPU (XGBoost CUDA)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 21.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94583 | 0.94496 | 0.94689 | 0.94546 | 0.94614 | 0.94586 |

**Strategy:** 5-fold stratified GPU XGBoost using V3’s full 344-feature pipeline with Triple TE, smooth keys, magic flags, original target means, and frequency features, combined with V2’s proven weak-regularization parameters: `lr=0.005`, depth 7, and `max_bin=1024`.
**File:** `Previously trained Files/Archieve/S6E9_V9_XGB_FullFE.py`

**Key Learning:**

> V9 improved V2 from 0.94569 to 0.94612 LB and reached second place overall. The categorical and raw environmental-concern × subsidy interactions dominated fold-1 importance, while the LB–OOF gap remained a reasonable +0.00028.

**Status:** ✅ Good

---

### Version 8 (CatBoost Ordered Boosting) - 2026-09-09

**Score**: **0.94603 LB** / 0.94583 OOF (Gap: +0.00020)
**Device**: GPU (CatBoost)
**Result**: **±0.00065 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94579 | 0.94493 | 0.94691 | 0.94550 | 0.94606 | 0.94584 |

**Strategy:** 5-fold stratified GPU CatBoost with `boosting_type='Ordered'`, original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, native CatBoost categorical features, multi-scale smooth keys, synthetic-artifact flags, and feature selection. Hyperparameters used `lr=0.015`, depth 5, `l2_leaf_reg=5.0`, Bernoulli bootstrap, and balanced weighting.
**File:** `Previously trained Files/Archieve/S6E9_V8_CatBoost_Ordered.py`

**Key Learning:**

> Ordered boosting improved the CatBoost LB from V4’s 0.94590 to 0.94603 and entered the leaderboard Top 5. `_ECL_x_Subsidy` remained the strongest fold-1 feature at 15.32%, but V8 stayed 0.00031 below V3.

**Status:** ✅ Good

---

### Version 7 (FT-Transformer Full Features) - 2026-09-09

**Score**: **0.94568 LB** / 0.94545 OOF (Gap: +0.00023)
**Device**: CUDA (FT-Transformer)
**Result**: **±0.00062 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 159.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94562 | 0.94474 | 0.94653 | 0.94507 | 0.94574 | 0.94554 |

**Strategy:** 5-fold KFold CUDA FT-Transformer with the same evidence-based feature selection used in V6, original-data concatenation per fold, per-fold auto-smoothed target encoding, and 79 selected features consisting of 35 numerical and 34 low-cardinality categorical inputs. The model used `d_block=128`, `n_blocks=2`, and `heads=8`.
**File:** `Previously trained Files/Archieve/S6E9_V7_FTTransformer_FullFeatures.py`

**Key Learning:**

> FT-Transformer reached 0.94568 LB with a close +0.00023 LB–OOF gap, but did not surpass V6 TabM or V3 LightGBM. Training took 159.1 minutes, making it substantially slower than the competing baselines.

**Status:** ✅ Good

---

### Version 6 (TabM Feature Selection) - 2026-09-08

**Score**: **0.94606 LB** / 0.94585 OOF (Gap: +0.00021)
**Device**: CUDA (TabM)
**Result**: **±0.00067 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 43.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94592 | 0.94500 | 0.94700 | 0.94547 | 0.94613 | 0.94590 |

**Strategy:** 5-fold KFold CUDA TabM with evidence-based feature selection from V5 LogisticRegression and V3 LightGBM importance, original-data concatenation per fold, per-fold target encoding from 14 source columns, and 83 selected numeric features from 344 raw features. TabM used `tabm_k=16`, `d_block=256`, and PWL embeddings.
**File:** `Previously trained Files/Archieve/S6E9_V6_TabM_FeatureSelection.py`

**Key Learning:**

> The compact 83-feature evidence-based subset achieved 0.94606 LB and surpassed V4/V5, but remained 0.00028 below the V3 LightGBM best. The model’s LB–OOF gap was +0.00021.

**Status:** ✅ Good

---

### Version 5 (LogisticRegression CPU Triple TE) - 2026-09-08

**Score**: **0.94478 LB** / 0.94468 OOF (Gap: +0.00010)
**Device**: CPU only (no GPU/cuDF)
**Result**: **±0.00064 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94474 | 0.94378 | 0.94570 | 0.94431 | 0.94488 | 0.94468 |

**Strategy:** 5-fold KFold CPU LogisticRegression with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, per-fold StandardScaler, frequency encoding, engineered features, and feature selection. Final model used 155 base features plus 189 Triple TE features, with string columns removed before training.
**File:** `Previously trained Files/Archieve/S6E9_V5_LR_Baseline.py`

**Key Learning:**

> StandardScaler enabled consistent convergence in 227–249 iterations and the LB–OOF gap was only +0.00010, but the linear baseline remained below the tree-based models. `Subsidy_Available_fe` was the strongest fold-1 absolute coefficient.

**Status:** ✅ Good

---

### Version 4 (CatBoost GPU Native Categoricals) - 2026-09-08

**Score**: **0.94590 LB** / 0.94581 OOF (Gap: +0.00009)
**Device**: GPU (CatBoost)
**Result**: **±0.00061 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94488 | 0.94547 | 0.94668 | 0.94609 | 0.94597 | 0.94582 |

**Strategy:** 5-fold stratified GPU CatBoost with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, native CatBoost handling for 10 string categorical features, multi-scale income/commute smooth keys, numeric-as-string columns, frequency encoding, synthetic-artifact flags, and feature selection. Final model used 155 base features plus 189 Triple TE features.
**File:** `Previously trained Files/Archieve/S6E9_V4_CatBoost_Baseline.py`

**Key Learning:**

> Native CatBoost categorical handling produced a close LB–OOF gap (+0.00009), but the 0.94590 LB remained 0.00044 below V3. `_ECL_x_Subsidy` was the strongest fold-1 feature at 10.97%, followed by target-encoded subsidy interactions.

**Status:** ✅ Good

---

### Version 3 (LightGBM CPU Triple TE) - 2026-09-08

**Score**: **0.94634 LB** / 0.94601 OOF (Gap: +0.00033)
**Device**: CPU only (no GPU/cuDF)
**Result**: **±0.00061 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.2 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94505 | 0.94579 | 0.94691 | 0.94625 | 0.94609 | 0.94601 |

**Strategy:** 5-fold stratified CPU LightGBM with original-data concatenation per fold, static original target means, triple target encoding with auto/10/100 smoothing on 63 categorical-like columns, multi-scale income/commute smooth keys, numeric-as-string columns, frequency encoding, synthetic-artifact flags, and feature selection. Final model used 155 base features plus 189 Triple TE features.
**File:** `Previously trained Files/Archieve/S6E9_V3_LGBM_Baseline.py`

**Key Learning:**

> V3 improved the LB to 0.94634, a +0.00065 gain over V2. Income floor/log-income target encodings dominated fold-1 importance, while CPU LightGBM reached the strongest score so far despite the longer training time.

**Status:** 🏆 Best

---

### Version 2 (XGBoost Triple TE) - 2026-09-08

**Score**: **0.94569 LB** / 0.94560 OOF (Gap: +0.00009)
**Device**: GPU (CUDA)
**Result**: **±0.00060 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94468 | 0.94526 | 0.94647 | 0.94581 | 0.94582 | 0.94561 |

**Strategy:** 5-fold stratified CUDA XGBoost with original-data concatenation per fold after dropping `Buyer_ID`, triple target encoding on 13 core columns with auto/10/100 smoothing, true frequency encoding on all columns, label-encoded categorical features, and feature selection that removed 76 constant columns and 10 perfectly correlated columns.
**File:** `Previously trained Files/Archieve/S6E9_V2_XGB_TripleTE.py`

**Key Learning:**

> Triple TE lifted the score slightly over V1. The strongest fold-1 signals were `LE_Subsidy_Available` (26.33%), `_recipe_score` (25.41%), and `_ECL_x_Subsidy` (16.85%).

**Status:** 🏆 Best

---

### Version 1 (XGBoost Baseline) - 2026-09-08

**Score**: **0.94559 LB** / 0.94535 OOF (Gap: +0.00024)
**Device**: GPU (CUDA)
**Result**: **±0.00062 OOF**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.94439 | 0.94509 | 0.94629 | 0.94558 | 0.94544 | 0.94535 |

**Strategy:** 5-fold stratified CUDA XGBoost with inverse-frequency sample weights, 19 non-constant digit features, engineered interactions and hard-edge flags, frequency encoding, leakage-safe per-fold target encoding, and original-data concatenation per fold after dropping `Buyer_ID`.
**File:** `Previously trained Files/Archieve/S6E9_V1_XGB_Baseline.py`

**Key Learning:**

> OOF and LB align closely (+0.00024). `_ECL_x_Subsidy` was the strongest fold-1 feature (37.50% importance), followed by `_ev_recipe` (8.13%) and `_ECL_x_RangeAnxiety` (7.92%).

**Status:** 🏆 Best

---
