# S6E10 Training Logs

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

---

### Version 55 (`V18's frame and recipe, with 20% of the overlap-cleaned original appended as training rows`) - 2026-10-28

**Score**: **0.96094 LB** / 0.96117 OOF (Gap: **-0.00023** — V18's own frame constant, reproduced on a ninth occasion)
**Device**: GPU (Tesla T4), scikit-learn 1.6.1, PyTabKit 1.7.3, Lightning `max_epochs=3`
**Result**: **ΔLB -0.00012 vs V18, ΔCV -0.00012 (-2.1 rulers) pooled / -0.00005 on fold means, folds improved 1/5** — the mechanism ran cleanly and lost.

**Fold Scores (vs V18's archived folds):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 0.96126 (-0.00015) | 0.96037 (-0.00004) | 0.96205 (-0.00004) | 0.96147 (**+0.00005**) | 0.96106 (-0.00007) | 0.96124 ± 0.00055 |

Pooled 0.96117 − fold mean 0.96124 = **-0.00007** (within the ordinary single-model band; V48 was the outlier at -0.00029).

**The augmentation demonstrably reached the fit — this is the arm's own instrument, read back off the run:**
`in-sample AUC appended rows 0.99678 | competition fit rows (100k probe) 0.96594 | held-out fold 0.96126`. The
pre-declared reading was "near the fit-row level means they were fitted" — the appended rows sit *above* the
competition fit rows, i.e. the net memorises the originals almost perfectly (0.9968) while the competition rows
reached only 0.9659 in-sample. So the negative delta is not an artifact of the rows being ignored: they were learned,
and learning them did not transfer. `n_epochs` read back as 3 from the fitted object (V53's `_read_held` mechanism),
matching `REALMLP_PARAMS`.

**Data path, measured twice** — the coverage block printed on Kaggle reproduces the local pre-run pass digit-for-digit:
overlap removal 129,880 → 129,859; **R = 103,887 / A = 25,972**; **0 of 25,972 A predictor vectors shared with R**;
unrepresentable A rows 411 (Flight Distance, 1.58%) + 531 (Departure Delay, 2.04%) + 602 (Arrival Delay, 2.32%) =
**1,241 dropped = 4.78%**, below the 10% cap, kept-rows pos rate 0.4344 → 0.4346. Fit frame **584,439 rows**
(559,708 competition + 24,731 appended, **+4.4%**), fit positives 259,019 (0.4432 vs 0.4436 competition-only);
289 frame cols → 374 expanded, 0.81 + 0.42 = 1.23 GB estimate, peak reserved 3.20 GB (V18: 3.10). EV/rating
correlations 0.52–0.98, identical to V18's run of record. Best epochs **[3, 3, 3, 3, 3]** against the ceiling of 3 —
V18's expected signature, so parity holds and starvation is not the explanation.

**Timing**: 19.4 min total (prep 23 s + fit 144 s per fold), inside the 19–25 min estimate. The EV build took
**215 s against V18's 115 s** — expected and declared: leakage-safety fits the 13 aux-EV producers twice, once on the
cleaned original for competition rows and once on R alone for the appended rows.

**Gate**: fails on three of the four pre-declared criteria outright — solo CV 0.96117 is **-0.00023 below the member
strength bar (≥ 0.96140)**, ΔCV is negative rather than ≥ +0.00005, and folds improved 1/5 rather than ≥4/5. The
paired test was then run offline on the archived vectors (temp `gate_v55_calibrated.py`): **dAUC -0.000119 (-2.09 rulers),
full-row bootstrap z -3.63** (60 reps, SE 3.29e-05) — the campaign's binding estimator, reproduced at +3.26 against the
ledger's recorded +3.38 for v50-vs-v49 before being trusted here. The analytic paired DeLong reads z -108 with an SE
30× smaller than the bootstrap's, and is reported only as a floor (same 30× overstatement it gave for v50-vs-v18:
analytic +159.7 against the recorded +10.56). **Leg census, measured rather than deferred to the bar: trio + v55 =
0.961471, i.e. -0.000114 CV and 0/5 folds improved, bootstrap z -6.68** — the appended-rows vector is the *least
damaging* leg tried (v52 -0.000076, v55 -0.000114, v53 -0.000239, v54 -0.000937; logit corr with the trio 0.99841) and
it still lowers the blend. **Not adopted; no board slot; not a leg.**

**Authoring cost, recorded honestly**: this arm needed three Kaggle attempts. Attempt 1 died at line 890 (a threaded
producer still returned the parent's 2-tuple), attempt 2 died on a 2% twin guard I had invented rather than measured,
and the third pass ran only after a synthetic dry run found the teacher being refit twice for competition rows and the
augment frame arriving in the original CSV's column order. The full repair record is in `Memory/daily_log.md`.

**File**: `Previously trained files/Archieve/S6E10_V55_RealRowsAugment.py`

### Version 49 (`equal-weight logit-mean of v11+v27+v34+v36 - the composition phase's first arm`) - 2026-10-25

**Score**: **0.96108 LB** / 0.961564 OOF (Gap: -0.00048)
**Device**: CPU (local - the first fully locally-run arm; reads the archived oof/sub vectors, owner lifted the standalone law for the composition phase)
**Result**: **+0.00002 LB** vs the V18 band top (0.96106) = tied 2nd-highest with V32; -0.00002 vs V29 0.96110

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.11 min (6.9 s, local CPU) |

**Fold Scores:** n/a - composition arm (no folds). Members' solo AUCs: v11 0.961133, v27 0.961149, v34 0.961241, v36 0.961393; mutual logit-corrs 0.9959-0.9988.

**Strategy:** equal-weight logit-mean of the hill-climb optimum over all 42 archived OOF vectors (greedy with replacement; SLSQP returns uniform weights; rank-mean ties logit-mean); test side = the same mean over the archived sub files; in-run paired DeLong gate (z +8.76 vs v18_oof, +6.01 vs v36, +8.89 vs v29).
**File:** `S6E10_V49_CompositionBlend.py` (root)

**Key Learning:**

> The CV→LB offset for a DIVERSE cross-family blend measured **-0.00048** - the largest constant yet (V36 -0.00035, V44 -0.00038, V45 -0.00037): the averaging-arm offset law extends to composition. Own-pool composition is offset-bound (~0.9611 LB ceiling); the route above 0.9611 is the public OOF pools.

**Status:** ⚠️ not adopted; V29 0.96110 remains best LB

### Version 54 (`DCN-V2 on V18's frame - our own hand-written net, family swap`) - 2026-10-28

**Score**: **0.95965 LB** / 0.958693 OOF (Gap: **+0.000957** — the campaign's largest, and the sign is wrong)
**Device**: GPU (Tesla T4), PyTorch 2.11.0+cu128
**Result**: **ΔLB -0.00141 vs V18, ΔCV -0.002594 (-45.5 rulers)** — the weakest arm since V40

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 0.95886 | 0.95782 | 0.95979 | 0.95909 | 0.95846 | 0.95880 ± 0.00066 |

Pooled 0.958693 − fold mean 0.958804 = **-0.000111**.
**Convergence — the real finding**: best epochs **[3, 3, 2, 3, 2] of a 50-epoch ceiling**. The net peaks inside
three epochs (~410 steps at 137 steps/epoch) and then degrades for the full patience window in every fold, so the
family does not exploit this 289-column frame — it is not a budget limit (contrast V53, which hit its ceiling).
**Architecture as built and read back**: 1.18M params, `input_dim 468` = 268 numeric + 200 embedded across **21
categorical embeddings**, 4 cross layers at rank 64 × 4 experts, `output.in_features 596`, fit-fold `pos_weight`
1.2544, cards up to 3,447 (the Flight Distance twin). All five structural checks passed, the column-split control
found exactly 21 categorical columns, and the 0.90 learning floor was cleared.
**Timing**: 18.6 min total (prep 23 s + train 157 s per fold); the 2.8 min/fold canary reading was 16× under its
45 min ceiling, so cost was never the constraint here.
**Vector integrity note**: `v54_oof` ships **110 rows at exactly 0.0 and 86 at exactly 1.0** (0.0280% saturated,
largest tie group 110), and `v54_sub` has 4. At 196 rows of 699,635 the AUC cost is a rounding artifact, not the
cause of the loss — but it is the first time an arm of ours has shipped boundary probabilities, and V55 clips them
to |logit| 13.82 rather than pretending they are interior values.
**Gate**: CV 0.958693 fails its own bar (>= 0.961337), and the arm **also fails the leg precondition it declared
before running** (`solo >= 0.9600`), which is what excluded it from V55 as a candidate leg. Honoured, not re-litigated.
**File**: `Arms/S6E10_V54_DCNv2Family.py`

### Version 53 (`TabM_D on V18's frame, config copied from our own S6E9 V6`) - 2026-10-28

**Score**: **0.96072 LB** / 0.960667 OOF (Gap: **+0.000053** — a positive gap, unlike every RealMLP net here)
**Device**: GPU (Tesla T4), PyTabKit 1.7.3, scikit-learn 1.6.1
**Result**: **ΔLB -0.00034 vs V18, ΔCV -0.000620 (-10.9 rulers)**

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 0.96094 | 0.96013 | 0.96151 | 0.96082 | 0.96029 | 0.96074 ± 0.00049 |

Pooled 0.960667 − fold mean 0.960738 = **-0.000071**.
**Convergence — the caveat that matters**: best epochs **[30, 30, 30, 30, 30] against a 30-epoch ceiling**, i.e.
early stopping never fired in any fold. TabM was still improving when the budget ran out, so this arm's loss is
**budget-limited, not a measurement of the family's ceiling** on our frame.
**Config as fitted, read back twice**: `arch_type tabm-mini-normal`, `tabm_k 16`, `num_emb_type pwl`,
`d_embedding 16`, `d_block 256`, `n_blocks 3`, `batch 1024`, `lr 1e-3`, `weight_decay 1e-3`, `patience 10`,
`n_epochs 30`, `device cuda` — printed identically by the pre-flight probe and the fold-1 fitted object, so the run
demonstrably trained the model it describes (the check V42/K8 demand).
**Frame the net saw**: 289 columns = 268 numeric (incl. 13 aux-EV) + 21 categorical; EV build 121 s; the 17 twins
carry 5-3,521 levels. Peak reserved **8.54 GB of the T4's 14.56 GB**, so the memory read never approached the alert line.
**Timing**: 140.3 min total (prep 21 s + fit **1,598 s per fold**). The canary read 26.1 min/fold against a 75 min
ceiling and did not fire; its 131 min projection came in within 5% of the 137.9 min actually spent — the first
budget instrument of ours that was validated by its own run.
**Cosmetic defect in its own diagnostics**: the variant-discovery line filters lowercase `'abm'`, so it printed
`TabM variants in this build: []` even though `TabM_D_Classifier` imported fine — a false empty list, not an empty
build. It changed nothing (the import and both read-backs are the real evidence) and is recorded so no future
session reads that line as a missing-class alarm.
**Gate**: fails CV >= 0.961337; no seed-2027 reproduction was run (final-run budget). Its leg value was measured by
V55, not assumed.
**File**: `Arms/S6E10_V53_TabMFamily.py`

### Version 52 (`drop the 81 columns K8 measured inert on the shipped net; recipe unchanged`) - recorded 2026-10-27

**Score**: **0.96102 LB** (user supplied) / 0.961295 OOF (earlier full-row recheck; Gap: −0.000275)
**Configuration**: CUDA RealMLP_TD, V18's parameter dictionary unchanged in the earlier AST comparison; the intended axis is feature removal. Runtime environment and fitted parameters were not captured.
**Result**: ΔCV **+0.000008**, paired-bootstrap z **+0.32**, folds improved **2/5**: no demonstrated improvement for this drop set.

**Fold Scores (recomputed from `v52_oof.csv` on the house partition, not copied from stdout):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96139 | 0.96045 | 0.96205 | 0.96150 | 0.96111 | 0.96130 |

Deltas vs V18's recorded folds: −0.00002, +0.00004, −0.00004, +0.00008, −0.00002. Pooled − fold mean **−0.00001**.

**Timing:** not supplied; the attachment was the script, not the execution log.
**Configured removal**: digits 20 + org_stats 22 + tm_single_TE 22 + other 14 + rating_aggregates 3 = **81 of 289 columns**, intended width **208**. The 21 raw inputs stay; the probe-only noise column was never present. Code contains exact-count and embedding-parameter checks, but their runtime output is unavailable.
**Gate**: CV ≥ 0.96140 and ≥4/5 folds both fail. The available z is a **paired-bootstrap diagnostic, not DeLong**; it also misses the z > 3 threshold. No adoption as a single-model improvement; V52's blend contribution remains unverified.
**File:** `Previously trained files/Archieve/S6E10_V52_InertFrameDrop.py` (earlier SHA-1 comparison verified a pure move from `Arms/`).

### Version 51 (`v27 + v36 + v44 — a 3-member blend, to price the leg-count term of the offset`) - 2026-10-25

**Score**: **0.96113 LB** / 0.961585 OOF (Gap: −0.000455, written −0.00046)
**Device**: CPU (local), no model fitted — reads the three archived OOF/sub vectors and averages their logits.
**Result**: **new highest LB, and a clean negative for the hypothesis it tested.** Three legs instead of V50's five cost 0.000010 of CV and bought 0.00001 of LB, so the gap (−0.000455) is the same constant as V49's −0.000484 and V50's −0.000475: **the offset is a property of blending this pool, not of averaging depth.** H2 predicted ≈0.96128; it missed by 0.00015 while H1 missed by 0.00003.

**Members (solo AUC recomputed from the stored vectors):**
| member | what it is | solo OOF AUC |
|--------|-----------|--------------|
| v27 | duo-teacher original-data tree | 0.961149 |
| v36 | 10-fold RealMLP | 0.961393 |
| v44 | 3-seed RealMLP, V18 frame | 0.961425 |
| **trio** | equal-weight logit-mean | **0.961585** |

**Fold Scores (house partition `StratifiedKFold(5, shuffle, rs=42)`, whose identity with the campaign's split is proved by reproducing V18's and V29's recorded arrays):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96173 | 0.96069 | 0.96234 | 0.96174 | 0.96145 | 0.96159 |

Pooled − fold mean **−0.00000**.

**Timing:**
| Stage | Time |
|-------|------|
| Load 3 vectors, blend, score, 120-rep bootstrap | 39.8 s |
| Total | 39.8 s, no GPU |

**Pre-declared gate, read against the run:** CV within 0.00003 of V50 → **PASS** (−0.000010); paired DeLong vs `v50_oof` **expected insignificant** → bootstrap **z −1.25** ✓; **the decisive measurement was the LB**, and the arm's whole purpose was to move the offset question off CV and onto the board.
**Selection provenance:** the 3-member set is the maximum over **every** 2- and 3-subset of the 14 single-arm vectors (exhaustive, not greedy), so the trio is not a lucky draw.
**Adoption stance:** V51 is the best public number; V50 remains an equal claim on the private board (0.00001 of CV between them = 0.2 ruler). Neither is a proven gain over V29's 0.96110 (+0.00003 = 0.5 ruler).
**File:** `Previously trained files/Archieve/S6E10_V51_TrioDiverseBlend.py`

### Version 50 (`equal-weight logit-mean of v44 + v45 + v36 + v27 + v11 - the quintet`) - 2026-10-25

**Score**: **0.96112 LB** / 0.961595 OOF (recomputed from the shipped vector) (Gap: -0.00048)
**Device**: CPU (local), no model fitted - reads the archived OOF/sub vectors. The owner lifted the 2026-10-11 standalone law for composition arms on 2026-10-25, so reading saved vectors is now lawful for this class.
**Result**: **New highest LB, and the first arm to pass the whole campaign gate outright**: ΔCV **+0.00032** vs both V18 and V29 (≥ +0.00001), folds improved **5/5** on the house partition (≥ 4/5), paired DeLong **z +3.38** vs V49 and **+10.56** vs V18 (> 3). The ΔLB over V29 is **+0.00002 = 0.4 of a ruler**, so the *number* is a record and the *board evidence* against V29 is not - which is exactly the case the gate was built for: CV, folds and z decide, and z binds harder than Δ.

**Members (solo AUC recomputed from the stored vectors):**
| member | what it is | solo OOF AUC |
|--------|-----------|--------------|
| v44 | 3-seed RealMLP, V18 frame | 0.961425 |
| v45 | 3-seed RealMLP, V29 keys frame | 0.961415 |
| v36 | 10-fold RealMLP | 0.961393 |
| v27 | duo-teacher original-data tree | 0.961149 |
| v11 | 13 aux-EV XGBoost | 0.961133 |
| **quintet** | equal-weight logit-mean | **0.961595** |

Mutual logit correlations 0.99644–0.99989 (closest pair v44~v45 at 0.99989, i.e. the two seed-nets are nearly one column; the two trees are the diverse ones at 0.99644–0.99653 from everything else).

**Fold Scores (house partition `StratifiedKFold(5, shuffle, rs=42)`):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96173 | 0.96069 | 0.96237 | 0.96174 | 0.96147 | 0.96160 |

Paired against V29's recorded folds: +0.00039, +0.00027, +0.00024, +0.00033, +0.00034 → **5/5 improved**. Pooled − fold mean **+0.00000**, which is the check that makes V48's −0.00029 look like what it was (cross-fitted inputs), not like a blend property.

**Timing:**
| Stage | Time |
|-------|------|
| Load 5 vectors, blend, score | seconds (CPU) |
| Total | no GPU used |

**Verification of this arm's own claims (no stdout was captured, so nothing here is taken on trust):** the shipped `v50_oof.csv` scores 0.961595 and equals a fresh recompute of the logit-mean to 3e-16; the header's z +3.34/+11.8 reproduce by bootstrap at +3.38/+10.56; the "no 6th member can improve it" claim holds within greedy reach (adding v34 gives 0.961584, worse), while "mathematically proven global optimum over all 47 vectors" is not established because no exhaustive search was run. The house partition was confirmed first: scoring v18/v29 fold-wise under it reproduces their recorded arrays digit for digit, and v34 (2027 partition) does not, as expected.
**File:** `Previously trained files/Archieve/S6E10_V50_QuintetChampionBlend.py`
### Version 48 (`three in-run GBDT members' OOF logits as 3 columns for the parent's net`) - 2026-10-25

**Score**: **0.96091 LB** / 0.96070 OOF pooled / 0.96099 fold mean (Gap vs pooled: **+0.00021**)
**Device**: GPU (Tesla T4), pytabkit RealMLP_TD student at `n_epochs=3`, parent's dict byte-for-byte, frame 292 cols (V44's 289 + the 3 stack columns).
**Result**: **-0.00059 CV vs V18 with 0/5 folds improved** — the largest CV loss since V31 — and LB 0.96091, 0.00009 under the parent's band floor. The student finished **-0.00042 below its own best member** (LightGBM 0.96112).

**Members fitted in-run (house-split cross-fit, printed by the arm):**
| member | config source | solo OOF AUC | logit corr vs `_og_teacher` | fit time |
|--------|---------------|--------------|------------------------------|----------|
| XGB | V11's config | 0.96105 | 0.961960 | 219 s |
| LightGBM | V15's config | **0.96112** | 0.962907 | 1,527 s |
| CatBoost | V4's config | 0.96103 | 0.961893 | 429 s |

Mutual member logit corrs: xgb~lgbm 0.998983, xgb~cb 0.997758, lgbm~cb 0.997958 — **one channel, not three** (finding 82's ≥0.999 rule).

**Fold Scores (student):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96111 | 0.96007 | 0.96182 | 0.96109 | 0.96088 | 0.96099 |

Paired against the frozen parent: -0.00030, -0.00034, -0.00027, -0.00033, -0.00025 → **0/5 improved**.

**Timing:**
| Stage | Time |
|-------|------|
| Members (3 fits, house-split cross-fit) | 36.3 min |
| Student (5 folds) | 15.2 min |
| Total | 53.8 min |

**Census (computed from the stored vectors, no labels):** `v48_oof` vs `v18_oof` logit corr **0.996478**, Spearman 0.981499, 3.58% of rows differing >0.05, id-aligned row-for-row. Under the 0.999 collapse threshold, so this is a genuinely different column and it still lost by 10 rulers — independence is necessary, not sufficient.

**Pooled-vs-fold-mean drift: -0.00029** (0.96070 vs 0.96099) where every single-model net arm in the file is within 0.00001. The run's first positive CV→LB gap (+0.00021) is this drift; read against the fold mean the offset is the ordinary -0.00008. Record both numbers for any vector built on cross-fitted columns.

**Gate:** declared `dAUC >= +0.0001 vs v18_oof AND offline DeLong z > 3`, and **deliberately no implied LB** — the V44/V45 lesson applied, so there was no header arithmetic to correct afterwards. CV bar missed by -0.00069. **Not adopted.**
**Law check:** ledger rule 6 counts a two-stage model as a single model; no saved OOF was read (members fitted fresh in-run), so the standalone-single-model law of 2026-10-11 is satisfied. Cross-episode evidence in the header (S4E9/S3E10 cascades, S6E4 +0.00105) is **reported by the authoring session**, not verified here, and this arm is the measurement that it does not transfer.
**File:** `Previously trained files/Archieve/S6E10_V48_StackTrainedNet.py`
### Version 45 (`V29's keys recipe fitted at three seeds, logit-mean shipped`) - 2026-10-25

**Score**: **0.96105 LB** / 0.96142 OOF (blended) (Gap: -0.00037)
**Device**: GPU (Tesla T4), pytabkit RealMLP_TD, `n_epochs=3`, the parent's dict byte-for-byte; 3 legs x 5 folds; frame 301 cols vs V44's 289 (the 4 inherited Stage-1 keys add 12 per-fold columns).
**Result**: **+0.00014 CV vs V29 and −0.00005 LB** — below the standing best and 0.00001 under the V29 frame's own recorded floor (V34 0.96106). Same shape as V44: offset −0.00018 → −0.00037.

**Legs (each 5-fold, pooled OOF printed by the run):**
| Seed | Fold AUCs | Fold mean | OOF | parent reproduced |
|------|-----------|-----------|-----|-------------------|
| 42 | 0.96136, 0.96038, 0.96209, 0.96148, 0.96109 | 0.96128 | 0.96127 | V29 0.961279 (1e-5) |
| 2027 | 0.96082, 0.96123, 0.96180, 0.96101, 0.96144 | 0.96126 | 0.96125 | V34 0.961241 (1e-5) |
| 1997 | 0.96103, 0.96112, 0.96196, 0.96135, 0.96104 | 0.96130 | 0.96129 | — (new seed) |
| **blended** | — | — | **0.96142** | — |

**Timing:**
| Stage | Time |
|-------|------|
| Per leg | 15.4 min |
| Total | 48.8 min |

**Pre-declared gate, read against this run:** blended OOF ≥ 0.96138 → **PASS** (+0.00004); implied LB 0.96120–0.96123 → **actual 0.96105, a −0.00015…−0.00018 miss (2.6–3.2 rulers)**; legs 42/2027 must reproduce V29/V34 digit-for-digit → **both miss by one unit in the 5th decimal** (0.96127 vs 0.96128, 0.96125 vs 0.96124). Offline paired DeLong vs `v29_oof` owed. **Not adopted.**
**Census (computed from the stored vectors, no labels):** `v45_oof` at **logit corr 0.999197** against `v29_oof` (Spearman 0.9956, 1.34% of rows >0.05 apart) — above the 0.999 collapse threshold, so the blend is the parent's ranking with noise removed rather than a new column.

**Note:** V44 and V45 land on the identical blended OOF (0.96142) from two different frames, and their LBs differ by 0.00001 — the two arms are the same experiment on two parents.
**File:** `Previously trained files/Archieve/S6E10_V45_KeysFrameMultiSeed.py`

### Version 44 (`V18's recipe fitted at three seeds, logit-mean shipped`) - 2026-10-25

**Score**: **0.96104 LB** / 0.96142 OOF (blended) (Gap: -0.00038)
**Device**: GPU (Tesla T4), pytabkit RealMLP_TD, `n_epochs=3`, the parent's dict byte-for-byte; 3 legs x 5 folds.
**Result**: **+0.00013 CV vs V18 — the largest CV since V36 — and −0.00002 LB: inside the parent's band.** The shipped vector is the logit-mean of the three legs; the CV gain was paid back as a larger CV→LB offset (−0.00023 → −0.00038), which is the second of three measured instances of that cancellation.

**Legs (each 5-fold, pooled OOF printed by the run):**
| Seed | Fold AUCs | Fold mean | OOF |
|------|-----------|-----------|-----|
| 42 | 0.96135, 0.96042, 0.96204, 0.96145, 0.96108 | 0.96127 | 0.96126 |
| 2027 | 0.96085, 0.96123, 0.96181, 0.96111, 0.96138 | 0.96128 | 0.96127 |
| 1997 | 0.96108, 0.96113, 0.96194, 0.96140, 0.96108 | 0.96133 | 0.96132 |
| **blended** | — | — | **0.96142** |

**Timing:**
| Stage | Time |
|-------|------|
| Per leg | 14.7 min |
| Total | 46.4 min |

**Pre-declared gate, read against this run:** blended OOF ≥ 0.96139 → **PASS** (+0.00003); implied LB 0.96116–0.96119 → **actual 0.96104, a −0.00012…−0.00015 miss (2.1–2.6 rulers)**; seed-42 leg must equal the parent's 0.96129 digit-for-digit → **FAILS at 0.96126** (−0.00003 = 53% of a ruler). Offline paired DeLong vs `v18_oof` owed. Gate says fail-any → tie-maker, so **not adopted**.
**Determinism finding:** same seed, byte-identical recipe, different file moves the net's OOF by ~3e-5, so the record's "control rung reproduces digit-for-digit" is XGBoost-only (K5/V11, V30/V20). Net arms need an in-run paired control or a ±0.00003 tolerance.
**Census (computed from the stored vectors, no labels):** `v44_oof` sits at **logit corr 0.999201** against `v18_oof` (Spearman 0.9957, 1.36% of rows differing by >0.05, id-aligned row-for-row) — above the campaign's own 0.999 collapse threshold (finding 82), so by the census this arm adds 0.000000 to any stack that already holds V18. It also sits at 0.999887 against `v45_oof`: the two arms are one column.

**Strategy:** one axis vs the frozen V18 — `CFG.SEEDS` = [42, 2027, 1997], both the SKF split seed and the net's `random_state`; teacher/EV builders keep their fixed seeds, no saved OOF is read (owner law 2026-10-11; public_scores rule 6 counts multiseed as a single model). Exactly two `to_csv`.
**File:** `Previously trained files/Archieve/S6E10_V44_MultiSeedRealMLP.py`


### Version 43 (`+13 within-respondent relative ratings on V18's frame`) - 2026-10-24

**Score**: **0.96101 LB** / 0.96124 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), pytabkit RealMLP_TD, `n_epochs=3`, the control's dict byte-for-byte.
**Result**: **-0.00005 CV vs V18 (0.96129) with folds improved 1/5** (paired deltas -0.00008, -0.00004, -0.00009, +0.00007, -0.00009) -> **gate fails on magnitude and on the fold criterion, direction negative.** The transform is a re-reference of inputs the net already holds, and V41 had just measured that the net does not represent the row-mean block; the two results are the same statement from two directions. This closes the last un-tried transform family, so **FE is finished on this frame by measurement**.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.8 min |
| — per-fold prep | 23 s |
| — per-fold train | 142 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96133 | 0.96037 | 0.96200 | 0.96149 | 0.96104 | 0.96125 |

**Positive control printed by the arm:** `+13 _rel_<rating> columns | respondents with no rated item: train 0, test 0 | deviation scale: mean +0.0000, sd 1.0866, min -3.692, max +3.615` — the block is live and correctly scaled, and **no row in the whole dataset has all 13 ratings unrated**, which bounds any disengagement mechanism (`_count_0` never reaches 13).
**Log defect (recorded, not edited):** the head print (line 702) and the `[5/5] RESULTS` footer (909) print the parent-inherited text "+13 numeric aux_ev columns on V12's frame" instead of this arm's axis. The clone rewrote `CFG` and the docstring but not those two strings; the axis is correct only in the `[2/5]` feature print (line 764). Noted in the score ledger appendix; future clones rewrite every axis print.
**Strategy:** one axis vs the frozen V18: `_rel_<rating> = r - _mean_excl0` for r > 0 else 0, 13 columns, no label and no cross-row statistic (leak-free by construction), everything else byte-identical. Builder crash caught in local calibration before upload (`np.where` returns an ndarray; `.fillna` does not exist on it).
**File:** `S6E10_V43_IpsativeRatings.py` (root until archived)

### Version 42 (`CatBoost MVS bootstrap, parent's rate held`) - 2026-10-24

**Score**: **0.95965 LB** / 0.95990 OOF (Gap: -0.00025)
**Device**: GPU (CatBoost GPU + cuda teacher).
**Result**: **−0.00098 CV vs parent V4 (0.96088)**; the in-run fold-1 control on the same matrix reads **Bernoulli 0.96100 vs MVS 0.95964, paired d −0.001361 = 23 rulers**, and best iterations rise 1089 → 2932. MVS is closed on our frame with a measured number, replacing the ledger's mislabel (V35 was XGBoost GOSS).

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.0 min |
| — per-fold prep | 9 s |
| — per-fold train | 80–244 s (MVS needs far more rounds) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95964 | 0.95892 | 0.96102 | 0.96026 | 0.95972 | 0.95991 |

**Best iters:** [2932, 1322, 4240, 2161, 1279]
**Strategy:** one key changed from the frozen parent — `bootstrap_type='Bernoulli', subsample=0.8` → `bootstrap_type='MVS', subsample=0.8`; `sampling_frequency` deliberately unset. Post-fit read-back asserts `MVS` + rate 0.8.
**File:** `S6E10_V42_CatBoostMVS.py`

### Version 41 (`net frame minus the tree-centric blocks`) - 2026-10-24

**Score**: **0.96104 LB** / 0.96127 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), RealMLP_TD, `n_epochs=3`.
**Result**: **−0.00002 CV vs parent V18 (0.96129)**, folds 4/5 ahead — a tie. The mechanism finding is the useful part: **the net does not use the digit or rating-aggregate blocks at all**, so the frame is now closed in both directions (V21 subtracted on the tree, V41 on the net).

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.4 min |
| — per-fold prep | 21 s |
| — per-fold train | 132 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96136 | 0.96039 | 0.96205 | 0.96141 | 0.96117 | 0.96128 |

**Positive control printed:** dropped 37 tree-centric columns (32 digit + 17 rating aggregate); kept `_og_teacher`, 13 `aux_ev`, 17 twins, route profile, org stats.
**Strategy:** V18 byte-for-byte except the drop, applied after the parent's own builders so the digit block's rounding of its sources is unchanged.
**File:** `S6E10_V41_NetFrameSubtraction.py`

### Version 40 (`original-data prior as XGBoost base_margin, route self-selected`) - 2026-10-24

**Score**: **0.95196 LB** / 0.95319 OOF (Gap: -0.00123)
**Device**: GPU (cuda), XGBoost hist on V11's frame.
**Result**: **−0.00794 CV vs parent V11 (0.96113)**, with `Best iters: [0, 0, 0, 0, 0]`. This is the campaign's cleanest negative control: the offset was real (pre-flight measured a +0.3865 mean-probability move at fit and predict; the offset alone scores **0.95508** on train) but Kaggle's build names neither `eval_base_margin` nor a DMatrix eval pair, so early stopping judged a validation block scored *without* the prior and stopped the model at round zero. The shipped vector is the prior, not a model.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.7 min (fit 19 s/fold because nothing trains) |
| — per-fold prep | 9 s |
| — EV build | 116 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95308 | 0.95191 | 0.95374 | 0.95367 | 0.95358 | 0.95320 |

**Route ladder (self-decided, printed):** `eval_base_margin` absent → DMatrix eval pair **rejected** ("Not supported type for data.<class 'xgboost.core.DMatrix'>") → `es_margin_blind`, and the arm labelled its own reading a two-change mechanism finding in RESULTS.
**Strategy:** teacher column dropped, its logit passed as the offset; nothing else moved. The base_margin route is closed on this build — not as a tie, as an unusable mechanism.
**File:** `S6E10_V40_PriorAsBaseMargin.py`

### Version 39 (`num_emb_type pbld → plr on V18's frame`) - 2026-10-24

**Score**: **0.96097 LB** / 0.96120 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), pytabkit 1.7.3, `n_epochs=3`.
**Result**: **−0.00009 CV vs parent V18 (0.96129)** — the function class never tried in 39 arms is not the one that helps. The read-back is the second half of the value: with `plr` the inherited `plr_act_name='gelu'` **is** finally live (under the library default `pbld` it was silently overridden to `linear`, finding 79), so both the embedding-family and the activation question close in one run.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.0 min |
| — per-fold prep | 23 s |
| — per-fold train | 135 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96135 | 0.96023 | 0.96201 | 0.96128 | 0.96116 | 0.96121 |

**Read-back after fold 1:** `{'num_emb_type': 'plr', 'plr_act_name': 'gelu', 'plr_hidden_1': 16, 'plr_hidden_2': 8, 'plr_lr_factor': 0.1151, 'plr_sigma': 2.33}`
**File:** `S6E10_V39_NumEmbPlr.py`

### Version 38 (`k-NN label mean over the overlap-cleaned original`) - 2026-10-24

**Score**: **0.96100 LB** / 0.96125 OOF (Gap: -0.00025)
**Device**: GPU (Tesla T4), RealMLP_TD on V18's frame + 1 column, `n_epochs=3`.
**Result**: **−0.00004 CV vs parent V18** — the new estimator of the original-data law is dominated by the model of that law we already ship: `knn_orig_prob` scores **0.94997** solo against `_og_teacher`'s 0.95508, and both sit far under the student. 0 distance-0 neighbours and 0 base-rate fallbacks prove the column was computed, not defaulted.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 68.5 min |
| — k-NN column build | **51 min** (3066 s) |
| — per-fold prep | 23 s |
| — per-fold train | 141 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96145 | 0.96035 | 0.96209 | 0.96136 | 0.96102 | 0.96125 |

**Cost note:** the most expensive per-unit-effect addition the ladder has made — 51 min of build for a column that moved CV −0.00004.
**File:** `S6E10_V38_KnnOrigLabelMean.py`

### Version 37 (`plr_sigma` ladder {8, 32} — the parameter arm on V18's byte-identical frame) - 2026-10-23

**Score**: **0.96107 LB** / 0.96128 OOF (Gap: -0.00021)  [rung of record σ=8; σ=32 read 0.96121 OOF]
**Device**: GPU (Tesla T4, pytabkit 1.7.3), 2 rungs × 5 folds, prep shared per fold.
**Result**: **σ=8 −0.00001 and σ=32 −0.00008 vs parent V18 (0.96129)** — both ties; LB +0.00001 over the parent's band top, also a tie. The stop-loss pair reads: V36 +0.00010, V37 ≈0, so Phase A did not fire its closure clause.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.8 min |
| — per-fold prep | 24 s (shared by both rungs) |
| — per-fold train | 140 s per rung |

**Fold Scores (rung of record σ=8):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96139 | 0.96042 | 0.96207 | 0.96142 | 0.96110 | 0.96128 |

**σ=32 per fold:** 0.96144 / 0.96029 / 0.96193 / 0.96132 / 0.96112 (mean 0.96122) — paired deltas vs σ=8: +0.00005, −0.00013, −0.00014, −0.00010, +0.00002.

**Strategy:** SIGMA_RUNGS = [8, 32]; `REALMLP_PARAMS` carried unchanged from V18 (including `'plr_sigma': 2.33`) and copied per fit with only that key overridden, so every other key is provably the parent's. One fitted model per rung, rungs never combined, exactly two `to_csv` on `v37`.
**File:** `S6E10_V37_PlrSigmaLadder.py`

**Key Learning:**

> **The knob the front says matters was already turned — by the recipe we inherited.** pytabkit's default is 0.1 and goodpjw2008's bundle put 2.33 in it, so Deotte's 0.1 → 32 (+5.5e-4) describes a journey whose first 23× our control has already made; going further costs (σ=32 is −0.00006 against σ=8 on the same splits). Method takeaway: a published single-knob delta is only transferable if the *starting value* is the same, and fetching a library's default dict (finding 75) is what makes that checkable before spending GPU-hours.

### Version 36 (Outer folds 5 → 10 — the protocol arm, one CFG value on V18) - 2026-10-23

**Score**: **0.96104 LB** / 0.96139 OOF (Gap: -0.00035)
**Device**: GPU (Tesla T4, `device='cuda'`, pytabkit 1.7.3, sklearn 1.6.1).
**Result**: **+0.00010 pooled CV vs parent V18 (0.96129), and LB −0.00002 — the largest CV move the campaign has produced, converting to nothing on the board.** Fold arrays are not comparable across fold counts by construction, so the gate is pooled paired DeLong + the 2027 reproduction; the LB tie is the first read.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.8 min |
| — per-fold prep | 22 s |
| — per-fold train | 153 s (V18 at 5 folds: 147 s) |
| — EV build | 115 s |

**Fold Scores (10-fold, seed 42):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.96155 | 0.96161 | 0.96012 | 0.96088 | 0.96191 | 0.96230 | 0.96093 | 0.96214 | 0.96124 | 0.96129 | 0.96140 |

**Strategy:** V18's frame, recipe, budget and seed with `CFG.N_FOLDS` 5 → 10. Every fold model sees 90% of rows instead of 80%, and the shipped prediction averages 10 models instead of 5. Best epoch 3 on all ten folds (ceiling 3), `keys absorbed by **kwargs: none`.
**File:** `S6E10_V36_TenFoldAveraging.py`

**Key Learning:**

> Two things moved, and only one of them is the model. CV rose +0.00010 exactly as the learning-curve slope priced the extra 10% of rows, and `pooled − fold mean` stayed at −0.000004, so the estimator is honest. The board did not move — and the reason is now measured rather than argued: **the CV→LB offset for this protocol is −0.00035, where the 5-fold frame reproduces −0.00023 to 0.000003 across seeds.** The constant belongs to the (frame, protocol) pair, so a fold-count arm's CV is not on the same scale as the arms it is compared against, and its +0.00010 must not be read as a candidate gain. Related: the OOF-vs-test averaging asymmetry in `PLAN_V36_ONWARD.md` §3b — here it went the other way, the CV gain came from the estimator, the shipped mean-of-10 came to 0.96104.

### Version 35 (Gradient-based row sampling — XGB `sampling_method='uniform'` → `'gradient_based'`, one parameter on V11) - 2026-10-17

**Score**: **0.96058 LB** / 0.96101 OOF (Gap: -0.00043)
**Device**: GPU (Tesla T4, `device='cuda'`, `tree_method='hist'`, xgboost 3.2.0).
**Result**: **−0.000123 pooled CV vs parent V11 (0.961133), folds improved 0/5** — every fold delta negative (−0.00011, −0.00008, −0.00013, −0.00012, −0.00013; mean −0.000114) — and **LB −0.00006**. Gate fails on magnitude and on folds. **The plan's highest-expected-value arm is a unanimous loss.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.9 min (V11 was 5.8 min — MVS bought no speed here) |
| — per-fold prep | 9 s |
| — per-fold train | 30–40 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96113 | 0.96013 | 0.96182 | 0.96105 | 0.96099 | 0.96102 |

Best iterations **[450, 742, 554, 474, 372]** — spread 370 rounds, the pre-declared variance read.

**Strategy:** exactly one parameter on V11's frozen frame; `subsample` stays 0.9 and no `max_delta_step` is set (that constraint is multiclass-only, so setting it would have moved two numbers). The Arg-Name Law was proven twice in-run: the constructor round-tripped `sampling_method='gradient_based'`, and fold 1 printed the live `booster params (fold 1): sampling_method='gradient_based'`.
**File:** `Previously trained files/Archieve/S6E10_V35_GradientRowSampling.py`

**Key Learning:**

> **The published +0.12% ROC-AUC, first on 97 of 153 datasets (Ibragimov et al., KDD 2019), does not transfer to a near-ceiling 700k-row frame.** The parameter demonstrably reached the booster, so this is a mechanism null and not a swallowed kwarg; CV fell with unanimous fold agreement; and the stopping point got *less* stable, not more — a 370-round best-iteration spread. That is the mechanism: MVS concentrates sampling on high-gradient rows, which on a frame already at 0.961 removes no variance worth removing while perturbing where early stopping lands.
>
> The same fold-1 print carries the campaign's best single explanation of its own FE plateau: gain importance is **`_og_teacher` 0.4177 + `_tm_pair_online_boarding__type_of_travel` 0.2199 = 64% of the model in two channels.** The original-data teacher and one TE pair already own the ranking, so a third or fourth key channel cannot move it — which is what the three-for-three null transfers measured from the other side, and why the *teacher* channel (not the key channel) is where the remaining published gains live.

**Status:** ❌ MVS / gradient-based row sampling **closed by measurement**; the variance queue loses its largest published external candidate. ✅ The round-trip + live `booster params` proof pattern worked and stays in the protocol.

---

### Version 34 (V29 reproduced at `RANDOM_SEED=2027` — the ruler's second half, completing {V18, V29} × {42, 2027}) - 2026-10-17

**Score**: **0.96106 LB** / 0.96124 OOF (Gap: -0.00018)
**Device**: GPU (Tesla T4), PyTabKit `RealMLP_TD_Classifier`, `n_epochs=3`.
**Result**: vs V33 (the same seed, V18's frame) — **pooled +0.000010, fold mean +0.000014, LB +0.00006**. vs V29 (the same frame, seed 42) — **pooled −0.000040, fold mean −0.000036, LB −0.000040**. **Not a gain claim: this is the ruler's second tick.** The four Stage-1 keys now read **−0.000007 CV at seed 42 and +0.000010 CV at seed 2027** — the sign flips with the seed, on a swing of 0.000017.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.3 min |
| — per-fold prep | 22 s |
| — per-fold fit | 141 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96084 | 0.96126 | 0.96181 | 0.96096 | 0.96138 | 0.96125 |

Best epochs **[3, 3, 3, 3, 3]** at the ceiling. Fold sd **0.000382** (V29's own at seed 42 was 0.000610).

**Strategy:** seed only. Frame (301 cols → +132 per-fold TE cols), budget, recipe and the four key blocks are V29's byte-for-byte; peak reserved 3.19 GB against the 12 GB abort line.
**File:** `Previously trained files/Archieve/S6E10_V34_V29SeedRepro.py`

**Key Learning:**

> Two readings, both arithmetic rather than modelling. **(1) The LB−CV offset reproduces exactly** — −0.000180 at both seeds, against −0.000230 on V18's frame at both seeds — so for this family **the public board is CV translated by a per-frame constant**, and it carries no information CV does not already have. **(2) Fold-level criteria are seed-bound:** a seed reshuffles which rows sit in which fold, so the per-fold difference has sd 0.00059, larger than the fold spread itself. "Folds improved N/5" is a paired statistic **only inside one fixed split**; across seeds it is noise, and V29's 2/5 → 3/5 is that noise on the record. Together they retire V29's standing as a record: its +0.00004 LB edge is a frame offset and a seed tick, not a model gain.

**Status:** ✅ ruler complete; ❌ the TE-key channel is **unmeasurable** rather than merely small, on both architectures and now both seeds. V18 remains the standing control.

---

### Version 33 (V18 reproduced at `RANDOM_SEED=2027` — the campaign's resolution floor, measured on the control itself) - 2026-10-17

**Score**: **0.96100 LB** / 0.96123 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), PyTabKit `RealMLP_TD_Classifier`, `n_epochs=3`.
**Result**: vs V18 — **pooled −0.000057, fold mean −0.000056, LB −0.000060.** Nothing was changed except the split shuffles, so this measures how far a byte-identical model can appear to move on both yardsticks at once.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.8 min |
| — per-fold prep | 22 s |
| — per-fold fit | 148 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96083 | 0.96123 | 0.96185 | 0.96098 | 0.96129 | 0.96124 |

Best epochs **[3, 3, 3, 3, 3]**. Fold sd **0.000391** against V18's 0.000606 — even the spread is seed-made.

**Strategy:** seed only; frame, TE, EV block, recipe, budget and output discipline identical to V18.
**File:** `Previously trained files/Archieve/S6E10_V33_V18SeedRepro.py`

**Key Learning:**

> **The most valuable run of the campaign, and it changed no score.** It prices every claim ever made in this ledger: V12's +0.00016 = 2.8 ruler units, V11's and V18's +0.00012 = 2.1, V27's +0.00002 = 0.35, V32's +0.000023 = 0.4, V29's +0.00004 LB = 0.7. The gate's Δ > +0.00005 bar sits **at** the noise floor rather than above it, and paired DeLong cannot rescue it — z prices the metric's sampling noise over 140k rows, not the training stochasticity, which is why z ≈ 4.9 corresponds to a mere +0.000115. Two consequences adopted in `PLAN_V31_ONWARD.md` §7: the magnitude bar moves to **+0.0001**, and a **seed-2027 reproduction becomes part of the gate** rather than a postscript. Third consequence and the one that reframes the whole record: **V18's "best single" is a band**, CV 0.96123–0.96129 and LB 0.96100–0.96106, for one unchanged model.

**Status:** ✅ the ruler exists; ❌ no control change — and no future arm may be adopted on a delta the ruler can manufacture.

---

### Version 32 (Bagging capacity rung — `n_ens` 8 → 16 on V18's byte-identical frame and 3-epoch budget) - 2026-10-17

**Score**: **0.96108 LB** / 0.96131 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), PyTabKit `RealMLP_TD_Classifier`; internal bagging inside one model object, not an ensemble.
**Result**: **pooled +0.000023 vs V18 (0.961287), fold mean +0.000022, folds improved 5/5** with every delta in [+0.00001, +0.00003], **LB +0.00002**. Gate **FAILS on magnitude** — +0.000023 is 0.4 of the tick V33 measured — while passing the fold criterion unanimously. The campaign's most consistent non-gain.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.1 min (V18 17.6 min; the axis costs ~1.8× the fit) |
| — per-fold prep | 25 s |
| — per-fold fit | 268 s (V18's was 147 s) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96142 | 0.96044 | 0.96210 | 0.96145 | 0.96116 | 0.96131 |

**Pooled OOF minus fold mean: −0.000007** (V18's was −0.000005) — the pre-declared calibration read, which could not fire. Best epochs **[3, 3, 3, 3, 3]**.
**File:** `Previously trained files/Archieve/S6E10_V32_NetBaggingRung.py`

**Key Learning:**

> **Bagging is saturated, and the diminishing-return curve is now measured at both ends:** `n_ens` 1→8 was **+0.000886**, 8→16 is **+0.000023** for double the compute — a ~38× collapse in effect per doubling. The campaign's largest single lever is exhausted, and no further rung is worth buying.
>
> The second reading retires a hypothesis rather than confirming one. `pooled − fold mean` stayed at 5–7e-6 at both capacities, so on this frame at 3 epochs there is **no bagging or epoch offset hiding in the pooled OOF**: the earlier "pooled OOF is epoch-biased" note does not extend to equal-budget arms, and pooled and fold-mean are interchangeable here. Third, the methodological caution matters more than this arm's score: 5/5 folds improved is a sign test at p = 1/32, so the *direction* is believable — but an effect of 0.4 ruler units cannot be adopted, and adopting it would have installed a control that a seed change alone reproduces.

**Status:** ⚠️ second-highest LB in the file on a consistent-but-immaterial gain; ❌ the bagging axis closes; V18 remains the control.

---

### Version 31 (Confident-row exclusion — drop the top q=0.0387 of each fold's training rows by nested |p−y|) - 2026-10-17

**Score**: **0.95907 LB** / 0.95930 OOF (Gap: -0.00023)
**Device**: GPU (Tesla T4), RealMLP_TD student plus an in-run nested XGB scorer (`device='cuda'`, 5 inner folds, seed base 77 — V30's proven teacher pass, byte-for-byte).
**Result**: **pooled −0.001987 vs V18 (0.961287), folds improved 0/5** (deltas −0.00193, −0.00215, −0.00198, −0.00172, −0.00210; mean **−0.001976**), **LB −0.00199**. The largest single-arm loss of the campaign — ~12× the largest channel ever found here — and its pre-declared falsifier returned intact.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 25.6 min (V18's 17.6 plus 96 s/fold of nested scoring) |
| — per-fold prep | 22 s |
| — per-fold nested scorer | 96 s |
| — per-fold fit | 144 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95948 | 0.95826 | 0.96011 | 0.95970 | 0.95903 | 0.95932 |

Nested-scorer val AUC per fold **[0.96052, 0.95933, 0.96103, 0.96008, 0.96001]**; **21,661 rows dropped every fold** (q=0.0387 of 559,708, leaving 538,047); mean |p−y| **0.9470–0.9473** on the dropped slice vs **0.0860–0.0869** on the kept rows.
**File:** `Previously trained files/Archieve/S6E10_V31_ConfidentRowWeights.py`

**Key Learning:**

> **Why removing "noisy" rows costs 0.002 when those rows are 3.87% of the data.** AUC is scored against the **observed** labels, and the rows with |p−y| ≈ 0.95 are, by construction, the ones whose observed label is most surprising to a strong model. Keeping them teaches the net to place them at the extreme their observed label wants, and the metric credits exactly that; deleting them leaves ~5,400 such rows per validation fold to be ordered by a model that never saw anything like them. So 3.87% of the rows carried roughly **50× their share** of the ranking signal. That is a measurement of what those rows *are*, and it agrees with the two prior readings: K1b M2(a) found **0 byte-identical rows** (no label pair provably conflicting) and M2(c) found the calibration gap running the **other** way (ρ̂ = −0.2149, which a flip rate cannot be). Row-noise selection is closed by three independent routes, and §1(a)'s monotone-invariance argument is confirmed rather than tested.
>
> Two secondary readings worth keeping. **(1) The scorer was strong, so the null belongs to the mechanism, not to the ranking** — the nested 5-model average reached val AUC ≈0.9602, near V18's own fold level, and it beat V31's damaged student on 4 of 5 folds. **(2) This run cannot fully separate its own two explanations**, because the file never prints the dropped slice's class composition or the post-drop positive rate: "deleted real signal" and "shifted the training prior" both push the same direction here, and a future selection arm must print both before the guess.

**Status:** ❌ row-noise / confident-learning selection **closed by measurement**; ✅ §1(a) intact; ⚠️ authored with its gate already empty (K1b M2 found no separable conflict) and run as the owner-requested falsifier, which is what made the −0.00199 worth the 25.6 minutes.

---

### Version 30 (Soft-label ladder — target mixing `p=(1−α)y+α·p_nested-teacher`, (α, es) = (.20,150) → (.20,400) → (.70,400)) - 2026-10-17

**Score**: **0.96060 LB** / 0.96110 OOF (Gap: -0.00050) — rung of record is the control rung, V20 as configured
**Device**: GPU (cuda), XGBoost `device='cuda'`; three complete 5-fold rungs in one run, judged on **fold means**, rungs compared and never combined.
**Result**: **fold means 0.96111 / 0.96111 / 0.96111** — a spread of **0.000003** across a 3.5× α range, so nothing moved and the gate never engaged. The pre-declared falsifier fired **every fold of every rung**: Spearman(student, in-run nested teacher) **0.9928–0.9973** against the 0.990 limit ⇒ **COLLAPSE ⇒ reject at any CV.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 39.7 min (rung cumulatives 11.1 / 22.8 / 37.5 min) |
| — per-fold prep | 9 s |
| — per-fold fit | 123 s mean (incl. ~97 s nested teacher xfit) |
| — EV build | 113 s |

**Fold Scores** (rung of record):
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96125 | 0.96019 | 0.96191 | 0.96115 | 0.96106 | 0.96111 |

Best iters [780, 714, 816, 555, 426] on the control rung, [780, 714, 1085, 555, 655] at es=400, and **[1926, 3251, 2232, 2536, 1387] at α=0.70** — 3–5× the rounds to land on the same fold mean.
**Strategy:** V20's confound separated into rungs so the α step and the early-stopping step are individually readable; the teacher is **nested inside each fit fold** (never a saved OOF), evaluated and early-stopped on **hard** `y_val`, and `y_soft` was range-checked against the hard mean every fold (0.4436 vs 0.4436). Rung 3 moves two knobs from the control, which the file itself prints as `step vs control: BOTH - not one axis`.
**File:** `Previously trained files/Archieve/S6E10_V30_SoftLabelLadder.py`

**Key Learning:**

> **A teacher that is as good as its student cannot teach it anything.** The nested teacher's own fold mean was **0.96100** against the student's **0.96111** — 0.00011 apart — so the soft label is a near-copy of the hard one, its rank correlation with the student exceeds 0.99 by construction, and mixing 70% of it into the target changes the objective by nothing measurable while costing 3–5× the rounds. This is the campaign's cleanest closure: the falsifier was written **before** the run precisely because CV could not be trusted to distinguish "no effect" from "no information", and the guard, not the score, is what closed the axis. A second, rarer gift: the control rung **independently re-measured V20's CV** — recovered as 0.961104 from the archived vector, printed as 0.96110 by a fresh run, and reproducing its LB 0.96060 exactly — the first time a recovered number in this ledger was validated rather than trusted.

**Status:** ❌ **Target-mixing axis closed** (α ∈ [0.20, 0.70], es ∈ [150, 400], all collapsed). V31's region-dependent α was scrapped on this evidence, not on a hunch.

---

### Version 29 (Stage-1 TE keys ported to the net — +4 keys / +12 per-fold columns on V18's frame) - 2026-10-17

**Score**: **0.96110 LB** / 0.96128 OOF (Gap: -0.00018) — the highest LB in the campaign, on a CV tie
**Device**: GPU (cuda) — RealMLP_TD via PyTabKit on a Kaggle T4 (16 GB); pre-flight printed `accepts n_epochs: True | keys absorbed by **kwargs: none`.
**Result**: **−0.000007 CV vs parent V18** (pooled 0.96128 vs 0.961287; fold means equal at 0.96129), **folds improved 2/5 with one tie** (−0.00007, +0.00001, +0.00004, −0.00001, 0.00000) ⇒ **gate FAILS on magnitude and on fold consistency**; LB +0.00004 = **0.23σ** of this family's board scatter ⇒ **no control change, and V18 stays the standing control.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.6 min (V18 17.6 min — the 12 extra per-fold columns are free, as 13 EVs were) |
| — EV build | 115 s |
| — per-fold prep | 22–23 s |
| — per-fold fit | 144 s mean |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96134 | 0.96042 | 0.96213 | 0.96141 | 0.96113 | 0.96129 |

**Strategy:** one axis — the four keys Stage 1 screened (V23's two age crossings, V24's unrated count and banded wifi) appended after V18's own key block in `make_key_frames`, so they arrive only through the parent's three channels (`_tm_`, `_freq_`, `_rarity_`): 301 frame cols → 386 expanded, peak reserved 3.19 GB against the 12 GB abort line. Best epochs [3,3,3,3,3] at the 3-epoch ceiling. Thin-cell pre-flight printed as designed: 50 cells/7 thin (0.0002), 96/13 (0.0003), 7/2 (0.0001), 24/0 (0.0000).
**File:** `Previously trained files/Archieve/S6E10_V29_PortStage1KeysToNet.py`

**Key Learning:**

> The arm was authored to test whether a tree null stays null on the net, and it answered: **yes — for the third time running.** Keys scored −0.00005 and −0.00004 on XGBoost (V23/V24) and 0.00000 here; before them the posteriors scored −0.00004 and −0.00001, and the EVs +0.00012 and +0.00012. **Three channels, three transfers, sign and size preserved**, so the campaign's last architectural excuse for re-testing a null is spent: on this dataset the channel, not the consumer, decides. The secondary lesson is about reading boards: this run produced our best-ever LB **and** a CV tie, and the two are 0.23σ apart — had we promoted V29 on its LB we would have moved the control on noise, which is exactly the failure the gate exists to prevent.

**Status:** ❌ Rejected as configured; key channel closed on both architectures; ⚠️ highest LB in the file and explicitly **not** the new control. The seed-2027 reproduction (new **V33**) is now the arm that decides whether 0.96110 vs 0.96106 is a model difference at all.

---

### Version 28 (Categorical EV posteriors on the net — +7 `aux_p_<cat>_<level>` columns beside V18's 13 EVs) - 2026-10-16

**Score**: **0.96105 LB** / 0.96128 OOF (Gap: -0.00023)
**Device**: GPU (cuda) — sklearn 1.6.1, pytabkit 1.7.3, Lightning `max_epochs=3`; pre-flight printed `RealMLP_TD_Classifier accepts n_epochs: True | keys absorbed by **kwargs: none`.
**Result**: **−0.00001 CV vs parent V18 (0.96129), folds improved 2/5** (+0.00003, 0.00000, +0.00001, −0.00006, −0.00001) and **−0.00001 LB** — **gate FAILS on magnitude, and this is a null rather than a damage report**: the arm neither helps the net nor hurts it.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.4 min (V18 17.6 min) |
| — EV build | 119 s (global, original-fitted, inherited from V18) |
| — per-fold prep | 21 s |
| — per-fold fit | 140 s (V18 146 s — 7 more columns are as free as 13 were) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96144 | 0.96041 | 0.96210 | 0.96136 | 0.96112 | 0.96128 |

Best epochs **[3, 3, 3, 3, 3]** against the 3-epoch ceiling, as on every 3-epoch net. Pooled OOF equals the fold mean exactly, so no epoch/bagging offset is in play on either yardstick.
**Strategy:** one axis — V25's posterior builder copied byte-for-byte onto V18's frame: for each of Class, Type of Travel and Customer Type, a label-free XGB predicts that column from the other 20 `ALL_FEATURES` columns on the overlap-cleaned original (129,859 rows) and P(level) enters as its own column. **7 columns** (3+2+2, `Gender` excluded on measurement), added after the teacher and before the float32 cast and the twin pass, so nothing new is factorised into categorical twins. 296 frame cols → 381 expanded (V18: 289 → 374), peak reserved 3.09 GB against the 12 GB abort line.
**File:** `Previously trained files/Archieve/S6E10_V28_EVTargetsFullSet.py`

**Key Learning:**

> The nominal-posterior channel is now measured **inert on both families**: −0.00004 on the tree (V25) and −0.00001 on the net (V28), from the same builder and the same 7 columns. V12 monetised categoricals on the net by *re-representing* the input, not by importing a second model's belief about it, and this arm is the experiment that says so. Two secondary readings: the run also had to be re-authored twice before it scored (an 8-column assert that the data answers as 7, and a head print that read `P_COLS` 84 lines before it was bound), and its CV block was missing when the run was first reported, which is why a 0.96105 LB — second-highest in
the file — sat unrecorded for a session before the block arrived. **A result the record cannot see is not a
result.**

**Status:** ❌ Categorical-posterior channel closed on the net as well as the tree; ⚠️ 2nd-highest LB in the campaign and the closest arm to the control, which is exactly what a null looks like at this resolution.

---

### Version 27 (Duo original teacher — `_og_teacher` = sigmoid(logit-mean) of XGB + RealMLP_TD, both orig-only) - 2026-10-16

**Score**: **0.96076 LB** / 0.96115 OOF (Gap: -0.00039)
**Device**: GPU (cuda) — sklearn 1.6.1, pytabkit 1.7.3; the net member is `RealMLP_TD_Classifier` and the signature guard reported `n_epochs: True | keys absorbed by **kwargs: none`.
**Result**: **+0.00002 CV vs parent V11 (0.96113), folds improved 4/5** (+0.00002, +0.00004, +0.00005, +0.00010, −0.00008) and **+0.00012 LB** — the only Stage-1 arm ahead of its parent on both metrics, and still **below Δ>+0.00005 → gate FAILS on magnitude**, so no z was needed.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.3 min (V11 5.8 min) |
| — per-fold prep | 9 s |
| — per-fold fit | 27–36 s (mean 33 s) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96126 | 0.96025 | 0.96200 | 0.96127 | 0.96104 | 0.96116 |

**Strategy:** one axis — replace the single ektarr-spec XGB teacher with the logit-mean of an XGB and a RealMLP_TD, both fitted on the overlap-cleaned original only and both predicting the competition rows they never saw. `COLLAPSE_LIMIT = 0.990` on Spearman(xgb logit, net logit), measured on the 699,635 train rows (393 NaN filled for the net), because holding original rows out would also have shrunk the parent's teacher fit set — a second axis. Best iters [423, 642, 663, 621, 310].
**File:** `Previously trained files/Archieve/S6E10_V27_DuoTeacherOrig.py`

**Key Learning:**

> The diversity premise held and the gain did not: Spearman **+0.95990** (well under the limit), yet the members' standalone AUCs on never-fitted rows rank **xgb 0.95508 > logit-mean 0.95477 > net 0.95396** — the average sits between its parents and closer to the worse one, buying 0.00002 on the downstream frame. Tree+MLP diversity is real but the net member is the weak leg here, which is V3's tree+tree tie result re-measured with a genuinely diverse pair. This is the one Stage-1 arm worth re-opening only if the net member gets a real training budget.

**Status:** ⚠️ Best new LB of the episode (0.96076, 4th in the file) and the only arm ahead of its parent; **rejected as configured** (Δ below the bar).

---

### Version 26 (Teacher capacity ladder — (8,600) control → (10,1000) → (12,1500), one model per rung) - 2026-10-16

**Score**: **0.96064 LB** / 0.96113 OOF (Gap: -0.00049) — rung of record is the control rung (8,600)
**Device**: GPU (cuda), three complete 5-fold fits in one run, rungs compared and never combined.
**Result**: **0.00000 CV vs V11 on the rung of record**; the two deeper rungs are **−0.000108** (d10/e1000, 0.96103) and **−0.000125** (d12/e1500, 0.96101) with **0/5 folds improved each**. The teacher's own 80/20 original hold-out AUC moves the same way: **0.99510 → 0.99486 → 0.99466**.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.6 min (rung cumulatives 3.9 / 8.1 / 12.3 min) |
| — per-fold prep | 12 s |
| — per-fold fit | 34 s |

**Fold Scores** (rung of record):
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96124 | 0.96021 | 0.96195 | 0.96117 | 0.96112 | 0.96114 |

**Strategy:** the highest-leverage axis the ledger had left — `_og_teacher` carries 0.33–0.41 of fold-1 gain — tested as capacity, with a pre-declared falsifier: if the teacher's held-out AUC is flat across rungs the axis closes regardless of what the arm scores. Best iters [489, 646, 839, 346, 433].
**File:** `Previously trained files/Archieve/S6E10_V26_TeacherCapacityLadder.py`

**Key Learning:**

> Capacity was not the constraint; **over-fitting the original was** — every rung past (8,600) degraded the teacher itself, and the arm followed it down monotonically. Two readings this closes: the teacher axis, and the hope that a stronger prior could be bought with compute. The control rung reproduces V11's five fold AUCs digit-for-digit, which is the in-house proof that this frame is bit-reproducible at fixed seed and therefore that the rung deltas are signal, not run-to-run noise.

**Status:** ❌ **Teacher capacity closed at (8,600).** V27 attacks the same teacher from the diversity side and is the better-supported of the two.

---

### Version 25 (Categorical EV posteriors on the tree — +7 `aux_p_<cat>_<level>` columns) - 2026-10-16

**Score**: **0.96064 LB** / 0.96109 OOF (Gap: -0.00045)
**Device**: GPU (cuda)
**Result**: **−0.00004 CV vs parent V11, folds improved 1/5 → gate FAILS.** +0.00000 LB (tie band).

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.0 min (EV build 113 s, inherited from V11) |
| — per-fold prep | 9 s |
| — per-fold fit | 33 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96118 | 0.96014 | 0.96197 | 0.96113 | 0.96106 | 0.96109 |

**Strategy:** one axis — for each of Class, Type of Travel and Customer Type, a label-free XGB on the other 20 columns predicts that column on the overlap-cleaned original and P(level) enters as its own column: 3+2+2 = **7** columns (the original has no `Group travel` row), frame 272→279 cols. Best iters [660, 663, 557, 438, 423].
**File:** `Previously trained files/Archieve/S6E10_V25_CategoricalEVOnTree.py`

**Key Learning:**

> Two measured constraints came out of this arm and cost two runs to learn. (1) **`multi:softprob` cannot survive a two-class target through the sklearn wrapper** — it died with `num_class = 0` while the 3-level target fitted fine, so a 2-level target takes `objective='binary:logistic'`, which returns the same two-column shape. (2) The first diagnosis blamed the label coding and the census line printed alongside the fix **refuted it**; the diagnostic, not the reasoning, settled it. Substantively: the nominal-posterior channel that V12 monetised on the net is inert on the tree, where the same levels are already splittable.

**Status:** ❌ Nominal-EV channel closed on the XGBoost line (V28 carries it to the net).

---

### Version 24 (Two unrated/wifi-band TE keys — `n_unrated`, `pair_wifi_band__online_boarding`) - 2026-10-16

**Score**: **0.96066 LB** / 0.96109 OOF (Gap: -0.00043)
**Device**: GPU (cuda)
**Result**: **−0.00004 CV vs parent V11, folds improved 1/5 → gate FAILS.** LB +0.00002 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.0 min |
| — per-fold prep | 10 s |
| — per-fold fit | 34 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96114 | 0.96013 | 0.96194 | 0.96118 | 0.96110 | 0.96110 |

**Strategy:** +2 TE keys appended after V11's own key block (`colsample_bytree` draws by index, so inserting mid-block would have moved parent columns onto this arm's ledger): `n_unrated` keys the **already-shipped** `_count_0` rather than duplicating it, and the wifi band uses K1's {0, 1-3, 4, 5} granularity crossed with `Online boarding`. Cells: `n_unrated` 7 (2 thin), pair 24 (0 thin, 0.0000 of fit rows) — the cleanest cells in the screen. Best iters [423, 647, 806, 739, 535].
**File:** `Previously trained files/Archieve/S6E10_V24_RatedNAndWifiBandKeys.py`

**Key Learning:**

> `_rarity_pair_wifi_band__online_boarding` reached **0.0228 of fold-1 gain** (4th feature) and the arm still lost CV: a key can be *used* and still be *worthless*, and thin-cell safety is not a proxy for information. K1's z² screen had rated these two keys 3.6983 and 3.8548 — after V23 that screen's whole top set, it does not survive contact with a trained model.

**Status:** ❌ **The z²-selected key channel closes here**, with V23, on the tree.

---

### Version 23 (Age-band TE keys — `pair_class__age5`, `pair_inflight_wifi_service__age5`) - 2026-10-16

**Score**: **0.96063 LB** / 0.96108 OOF (Gap: -0.00045)
**Device**: GPU (cuda)
**Result**: **−0.00005 CV vs parent V11, folds improved 1/5 → gate FAILS.** LB −0.00001 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.7 min |
| — per-fold prep | 10 s |
| — per-fold fit | 32 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96122 | 0.96014 | 0.96180 | 0.96121 | 0.96108 | 0.96109 |

**Strategy:** +2 int-coded TE keys crossing a 5-year age band (0..17, bound asserted on train *and* test before any fit so the 18-wide multiplier is provably injective) with Class and with `Inflight wifi service`; frame 272 cols. Cells: 50 (7 thin, 0.0002 of fit rows) and 96 (13 thin, 0.0003). Best iters [469, 557, 577, 749, 495].
**File:** `Previously trained files/Archieve/S6E10_V23_AgeBandKeys.py`

**Key Learning:**

> These two keys were the **highest z² in the entire K1 screen** (4.6649 and 4.0301 against a 1.5 authoring bar) — the pre-training statistic that justified reopening FE after V13's zero. It predicted the direction of nothing. Age information is already splittable on this frame, so a re-cut of it into a TE key costs encoding capacity and buys noise in 20 thin cells.

**Status:** ❌ Age-crossing keys closed on the tree.

---

### Version 22 (Sign-corrected rating aggregates — `6 − r` inside the 9 aggregates) - 2026-10-16

**Score**: **0.96063 LB** / 0.96113 OOF (Gap: -0.00050)
**Device**: GPU (cuda)
**Result**: **0.00000 CV vs parent V11, folds improved 1/5 → gate FAILS.** LB −0.00001 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.0 min (EV build 116 s) |
| — per-fold prep | 10 s |
| — per-fold fit | 34 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96118 | 0.96020 | 0.96204 | 0.96117 | 0.96106 | 0.96113 |

**Strategy:** one axis — the published `timehc` orientation treats a rating as "higher = better", but `Inflight wifi service` etc. are 1–5 with **0 as the unrated marker**, so the aggregates are rebuilt on `6 − r` for r > 0 with 0 preserved; the EV block, the twins and the budget stay V11's byte-for-byte. Best iters [393, 644, 770, 407, 465].
**File:** `Previously trained files/Archieve/S6E10_V22_SignCorrectedAggregates.py`

**Key Learning:**

> K1 measured the reversed scale at composite `_mean_excl0` **0.85658**, +0.04393 above the shipped orientation — a real monotone correction to the aggregate's meaning, and **zero** change in ranking. A tree re-splits each raw rating, so a sign fix to a derived summary cannot reach the metric. Together with V21 this closes the aggregate axis in both directions: absent ≡ present ≡ corrected.

**Status:** ❌ Aggregate sign axis closed.

---

### Version 21 (Aggregate ablation — the 10 rating aggregates are not built) - 2026-10-16

**Score**: **0.96058 LB** / 0.96112 OOF (Gap: -0.00054)
**Device**: GPU (cuda)
**Result**: **−0.00001 CV vs parent V11, folds improved 3/5 → gate FAILS.** LB −0.00006 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.6 min |
| — per-fold prep | 9 s |
| — per-fold fit | 32 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96115 | 0.96020 | 0.96196 | 0.96121 | 0.96113 | 0.96113 |

**Strategy:** a subtraction, not an addition — the V11 recipe with the 10 rating aggregates never constructed (guard printed **0 of 10 named aggregates in the frame**, 142 base columns vs V11's 152, per-fold 262 vs 272), so the remaining channels (digits, route profile, org stats, teacher, EV block) are V11's byte-for-byte. Best iters [415, 777, 572, 407, 495].
**File:** `Previously trained files/Archieve/S6E10_V21_AggregateAblation.py`

**Key Learning:**

> The block that has been in the frame since V2 is **inert**: deleting all ten aggregates costs 0.00001 CV and 0.00006 LB, inside the noise V13's byte-identical re-submission bracketed at 0.00007. Once the 13 label-free EV columns arrived (V11, +0.00012), the raw aggregate summaries stopped carrying information the model could not get elsewhere — which is the cheapest closure in the episode and the one that should have been run first.

**Status:** ❌ Aggregates are dead weight; the axis is closed by V21 (remove) and V22 (repair) together.

---

### Version 19 (LightGBM leaf-count regularization — `min_child_samples` axis on V15's frozen config) - 2026-10-11

**Score**: **0.96057 LB** / 0.96103 OOF (Gap: -0.00046)
**Device**: **CPU** (`device_type='cpu'`, `n_jobs=-1`). Library line: **LightGBM 4.6.0**, PIN WARNING vs the recipe's pinned 4.7.0 printed again and recorded. sklearn 1.6.1.
**Result**: **−0.000024 CV vs parent V15** (0.96103 vs 0.96105), folds improved **4/5 but by ≈+0.00001 each** with fold 1 at −0.00015; +0.000017 vs V2 → **fails Δ>+0.00005. REJECTED.** LB −0.00004 vs V15 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 26.1 min CPU (V15 28.2 min) |
| — per-fold prep | 11–12 s |
| — per-fold fit | 258–349 s (mean 294 s) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96113 | 0.96011 | 0.96183 | 0.96107 | 0.96102 | 0.96103 |

Best iterations **[739, 866, 1011, 627, 639]** vs V15's [1068, 1012, 838, 669, 833] — the knob moved the stopping points hard (fold 1 −329, fold 3 +173) and the score not at all.

**Strategy:** one axis — `min_child_samples` default 20 → **40**, every other value in `LGBM_PARAMS` byte-identical to V15 (V2's frame, `num_leaves 63`, `lr 0.02`, 4000 estimators, callback ES 150, `categorical_feature=CATS`). Import-time pre-flight printed `LightGBM constructor accepts min_child_samples: True` before any fit (Arg-Name Law).
**File:** `S6E10_V19_LGBMMinChildSamples.py` (root until archived)

**Key Learning:**

> Doubling the minimum leaf count changed **when** LightGBM stops by up to 329 trees and **what it uses** not at all — the fold-1 importance profile is V15's column for column (`_og_teacher` 2,150 splits, `_tm_fd_exact` 1,847, then the `_route_*_resid/_mean` tail), and CV moved −0.000024. Capacity knobs on this family are **inert for ranking**, which is what a noise-limited target looks like: the ensemble can be made to fit more or fewer trees and it recovers the same ordering either way.

**Status:** ❌ Rejected as control (V12/V18 hold the net side, V2 the tree side); ✅ the LightGBM regularization axis is closed by measurement, and `v19_oof.csv` is a near-duplicate of `v15_oof.csv` — no reason to keep both as separate legs

---

### Version 18 (RealMLP + V11's 13 label-free aux-EV columns — feature axis on V12's twin frame) - 2026-10-11

**Score**: 🏆 **0.96106 LB** / **0.96129 OOF** (Gap: -0.00023) — **the best single of the episode on CV and LB, and above the previously cited best published single (0.96105)**
**Device**: GPU (Tesla T4, PyTabKit 1.7.3, Lightning `max_epochs=3`)
**Result**: **+0.00012 CV vs parent V12 with folds improved 5/5** (deltas +0.00007, +0.00019, +0.00010, +0.00018, +0.00003) and **+0.00028 CV / +0.00055 LB vs V2** — the second arm ever to clear Δ>+0.00005 **and** ≥4/5 folds, and the second outside the 0.0003 tie band on LB. **Paired DeLong z vs `v12_oof`/`v2_oof` and the `SEED_REPRO=2027` rerun are still pending.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 17.6 min (V12 15.2 min) |
| — per-fold prep | 20–22 s |
| — per-fold fit | 146–147 s (mean 146 s — the same as V12's; 13 extra columns are free) |
| — one-time EV build | 115 s (global, original-fitted, never per fold) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96141 | 0.96041 | 0.96209 | 0.96142 | 0.96113 | 0.96129 |

Best epochs **[3, 3, 3, 3, 3]** against a ceiling of 3 — still pinned after V17 proved 3 is the optimum, so this is now the expected signature rather than starvation.

**Strategy:** one axis — the frame gains exactly 13 numeric `aux_ev_<rating>` columns (V11's builder byte-identical: EV = Σ k·P̂(rating=k), each P̂ an XGB predicting that rating from the **other** 20 columns, predictors fitted on the overlap-cleaned original 129,859 rows, never on `satisfaction`), inserted after the teacher and before the float32 loop and the twin pass. **No EV gets a categorical twin** (they are continuous probabilities; 13 more high-cardinality twins would be a second axis and would widen the expanded matrix). `REALMLP_PARAMS` byte-identical to V12 including `n_epochs=3`; pre-flight printed `accepts n_epochs: True | keys absorbed by **kwargs: none`.
**File:** `S6E10_V18_RealMLPFusionEV.py` (root)

**Memory guard (absolute readings, no alert):** 289 frame cols → 374 expanded (all-one-hot worst case 4,316), transformed matrix 0.78 + 0.42 = **1.20 GB** vs the 16 GB device and the 12 GB abort line; peak reserved 3.10 GB.

**Key Learning:**

> **The EV block is a family-independent information channel, and it stacks.** It bought +0.00012 CV with 5/5 folds on XGBoost (V11) and +0.00012 CV with 5/5 folds on the net (V18) — the same magnitude on two architectures that disagree about almost everything else, and the same shape the published RealMLP+EV pair reported (0.961179 → 0.961344). This is the episode's decisive FE result: every **re-cut of columns the model can already split** measured ≈0 (V8, V9, V10, V13, V16, V19), while the two arms that feed in a **prediction the model was not given** (V11's rating EVs, V12's teacher logit + twins) each moved CV ≥4/5 folds. Note also that the EV-vs-raw-rating r screen reproduced V11's values **exactly** (0.9815 entertainment … 0.5218 checkin) under a different parent frame — the builder is deterministic and original-fitted, so that table is a reusable pre-training diagnostic, not a per-run artifact.

**Status:** 🏆 **New best single (CV 0.96129 / LB 0.96106), the new candidate control** — pending paired DeLong z and the seed-2027 rerun, V2 stays the standing control and V12 the previous best; ✅ the EV block is confirmed on both families, so it belongs in every future frame rebuild

---

### Version 17 (RealMLP epoch ladder — `n_epochs` 6/8/10 on V12's byte-identical twin+logit frame) - 2026-10-11

**Score**: **0.96085 LB** / 0.96092 OOF (Gap: -0.00007) — rung of record `n_epochs=10`
**Device**: GPU (Tesla T4, PyTabKit 1.7.3, `use_early_stopping=False` so the budget is the only stop rule)
**Result**: **−0.00025 CV vs parent V12** (0.96092 vs 0.96117) and **folds improved only 1/5**. Rungs: 6 → 0.96089, 8 → 0.96083, 10 → **0.96092**, i.e. the whole ladder sits ~0.00025–0.00034 **below** V12's 3-epoch budget, with no monotone trend. LB 0.96085 = −0.00010 vs V12 (tie band). **REJECTED; the epoch axis is closed at 3.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 86.4 min for 3 rungs × 5 folds (plan estimated 65–85 min) |
| — per-fold prep | 19–22 s (shared by all rungs) |
| — per-fold fit, mean by rung | 6 ep **263 s**, 8 ep **337 s**, 10 ep **411 s** (V12's 3 ep ≈ 154 s — cost is linear in the budget as predicted, the score is not) |

**Fold Scores (rung of record, n_epochs=10):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96125 | 0.96029 | 0.96176 | 0.96115 | 0.96101 | 0.96109 |

Best epochs **[10, 10, 10, 10, 10]** — and `6: 5/5, 8: 5/5` pinned at their own ceilings too; `Trainer.fit stopped: max_epochs=N reached` printed for every one of the 15 fits.

**Strategy:** one axis — `n_epochs` swept over the pre-declared ladder {6, 8, 10}, one fitted model per rung, rungs **never** combined, exactly one rung is the version's output (V8/V14/V16 pattern). Frame, `lr 0.053`, `flat_anneal`, `n_ens 8`, batch 256, ES-off all byte-identical to V12. Import-time signature check `RealMLP_TD_Classifier accepts n_epochs: True` guarded the axis before the first fit.
**File:** `S6E10_V17_RealMLPEpochLadder.py` (root)

**Key Learning:**

> **My "the net is epoch-starved" inference was wrong, and the number I read it from carried no information.** With `use_early_stopping=False`, PyTabKit always restores the **last** epoch, so "best epoch = 3 = the budget in 5/5 folds" was an artifact of the fixed budget, not evidence that training was cut off mid-improvement. Testing the hypothesis cost the truth: 6/8/10 epochs each lose to 3, so V12's budget was already at the optimum. Mechanism (inference, not measured here): every schedule in the recipe is **budget-relative** — `flat_anneal`, `cos_log_15`, `expm4t`, `sqrt_cos` are fractions of total training — so raising `n_epochs` does not extend the run, it stretches all four schedules and lands a different optimization path. Second measurement worth keeping: at 10 epochs the pooled OOF (0.96092) fell **0.00017 below the fold mean** (0.96109), while V12 and V18 show OOF ≈ fold mean. Longer budgets sharpen each fold's scores and hurt the cross-fold calibration that a pooled AUC actually reads — a cheap diagnostic (`OOF − fold mean`) for any future net arm.

**Status:** ❌ Rejected, all three rungs below the parent; ✅ the epoch axis on this recipe/frame is closed at 3 by measurement, and the `OOF − fold-mean` gap is added as a diagnostic; the net's remaining upside is the frame (V18), not the budget

---

### Version 16 (CatBoost — in-run `eval_metric='AUC'` control + 39 rating×context native crosses, 3-rung ladder) - 2026-10-11

**Score**: **0.96048 LB** / 0.96089 OOF (Gap: -0.00041) — rung of record `none`
**Device**: GPU (CatBoost 1.2.10 `task_type='GPU'` on a Kaggle T4; ektarr teacher `device='cuda'`)
**Result**: **both halves rejected.** V16a (metric): `none` = 0.96089 vs V4's recorded 0.96088 → **+0.00001**, i.e. AUC early stopping is worth nothing on this arm. V16b (frame): `add` 0.96095 and `swap` 0.96096 beat `none` on only **3/5 folds** each, so the pre-declared ≥4/5 rule leaves `none` as the run of record. LB −0.00003 vs V4 = tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 63.0 min for 3 rungs × 5 folds (V4: 7.9 min for one model) |
| — per-fold prep | 18–19 s (shared by all rungs) |
| — per-fold fit | mean 728 s across the three models |

**Fold Scores (rung of record `none`, 259 cols):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96095 | 0.96003 | 0.96172 | 0.96101 | 0.96080 | 0.96090 |

Best iterations (none) **[871, 1325, 892, 870, 736]**; add [942, 990, 908, 684, 711]; swap [642, 1139, 999, 767, 454].

**Strategy:** one run, two read-outs, three rungs — `none` drops the 39 crosses (V4's frame under AUC ES, **re-fitted in-run, not loaded**, so the metric delta is standalone-legal); `add` keeps V4's manual `_tm_pair_*` TE family plus the crosses; `swap` drops the 54 `*_pair_*` columns so CatBoost's own ordered target statistics own the interactions. `CB_PARAMS` byte-identical to V4 with `eval_metric` → `'AUC'`, held constant across rungs so the rung delta belongs to the frame alone. Fold-0 pre-flight fit (30 iters, 20k rows, discarded, seed restored) proved the two claims the arm rests on.
**File:** `S6E10_V16_CatBoostCrosses.py` (root)

**Key Learning:**

> **The plan's premise for this arm was only half true.** The probe accepted `eval_metric='AUC'` on `task_type='GPU'`, but every fit printed `Default metric period is 5 because AUC is/are not implemented for GPU` — CatBoost 1.2.10 **runs** the parameter and evaluates it on the **CPU** at period 5. So V16a's "+0.00007 expected from metric alignment" had no mechanism behind it, and measured +0.00001. And the last un-disproved published FE delta died here too: 39 native rating×context crosses bought +0.00006/+0.00007 CV at 3/5 folds, and **not one `_x_*` column reached fold-1's top-15** (`_og_teacher` 18.67%, `_tm_fd_exact` 4.55%, raw `Inflight wifi service` 1.54%, `_tm_pair_wifi__booking` 1.51% are what the model actually uses). The published +0.00029/+0.00034 was measured on a frame **without** our manual pair-TE family and route profile — on this frame the interaction channel was already occupied, so the crosses were redundant, and `swap` (dropping the manual family) still landed within 0.00001 of `add`. Third: a print defect surfaced — with `none` as the run of record, `dCV frame (none vs none)` degenerates to +0.000000; read the frame delta off the rung lines instead.

**Status:** ❌ Both halves rejected (CatBoost stays at V4 as the family member); ✅ the 4.6-vs-GPU-AUC fact and the "occupied interaction channel" reading are recorded so no future arm re-spends 63 GPU-min on either; `v16_rung_*_oof.csv` kept for the audit trail only

---

### Version 15 (LightGBM single — model-family axis on V2's frozen frame, CPU-only) - 2026-10-11

**Score**: **0.96061 LB** / 0.96105 OOF (Gap: -0.00044)
**Device**: **CPU** (LightGBM `device_type='cpu'`, `n_jobs=-1`; no GPU path in the file by owner decision). Library line: **LightGBM 4.6.0 on Kaggle, not the pinned 4.7.0** — PIN WARNING printed and kept as this run's record. sklearn 1.6.1.
**Result**: **+0.00004 CV vs V2, folds improved 4/5** (deltas +0.00011, +0.00006, +0.00004, −0.00006, +0.00003) → **fails the Δ>+0.00005 magnitude criterion by 0.00001**; LB +0.00010 is inside the 0.0003 tie band. **REJECTED as control.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.2 min CPU (V2 3.8 min GPU; plan estimated ~45 min CPU) |
| — per-fold prep | 11–12 s (same CPU prep cost as V2) |
| — per-fold fit | 267–363 s (mean 318 s) — the family, not the device, is what costs the 7.4× |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96128 | 0.96010 | 0.96181 | 0.96106 | 0.96101 | 0.96105 |

Best iterations **[1068, 1012, 838, 669, 833]** out of `n_estimators=4000` — **early stopping genuinely fired in 5/5 folds**, the counter-example to the net arms (V5/V6/V7/V12), which all pinned at their epoch budget.

**Strategy:** one axis — main class `xgb.XGBClassifier(**XGB_PARAMS)` → `lgb.LGBMClassifier(**LGBM_PARAMS)` (`objective binary`, `boosting_type gbdt`, `extra_trees False`, `num_leaves 63` ≈ V2's 2^6−1 capacity, `lr 0.02`, V2's subsample/colsample/reg values). Deliberately not ported: `max_bin 256` (binning is V14's axis) and `min_child_weight 5` (no count-scale analogue). `eval_metric='auc'` + ES window 150 moved to the fit call site.
**File:** `S6E10_V15_LightGBMSingle.py` (root until archived)

**API forensics (the two failures this run cost, both closed by evidence rather than guessing):**
1. First attempt raised `TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_round'` — the singular name does not exist.
2. The import-time pre-flight now printed `LightGBM fit() exposes: callbacks=True | eval_metric=True | **early_stopping_rounds=False**`. So on Kaggle's 4.6.0 build the **callback is the only available early-stopping mechanism** — the plural kwarg is gone too, meaning the "obvious" `early_stopping_rounds=150` fix would have failed as well. Deprecated `verbose=` in `fit()` was also dropped; verbosity now lives on the callback.

**Key Learning:**

> LightGBM reaches V2's ceiling with an almost opposite importance profile: `_og_teacher` still leads by split count (2,526) but `_tm_fd_exact` (2,079) and a long tail of `_route_*_resid` / `_route_*_mean` columns (752–976 each) dominate the rest, whereas XGBoost concentrated gain in the teacher (40%) and a handful of pair-TE columns. **Two families that partition the same information in visibly different ways land 0.00004 apart in CV and 0.00010 apart in LB** — which is what a noise-limited ceiling looks like from the inside, and the strongest cross-family confirmation that the remaining gap is not model-capacity-bound.
> Second: the plan's own CV reference for this family (evgendvorkin's single, CV 0.96111 / LB 0.96069, built on LightGBM **4.7.0**) is reproduced at 0.96105 / 0.96061 on **4.6.0** — the version mismatch is a plausible part of the −0.00006 CV / −0.00008 LB difference, and the library line is recorded so this arm is never compared to his number without that caveat.

**Status:** ❌ Rejected as control (V2 stands; V12 remains the best single and the candidate control); ✅ `v15_oof.csv` retained — it is the only genuinely different-family leg we now own, and its residual structure is the cheapest test of whether a second family would add anything to a combination (not permitted as a version under the single-model rule)

---

### Version 13 (Original-mean Bayesian shrinkage — defect repair inside `add_org_stats`) - 2026-10-11

**Score**: **0.96058 LB** / 0.96101 OOF (Gap: -0.00043); **identical `v13_sub.csv` re-submitted → 0.96051 LB** (Gap -0.00050)
**Device**: GPU (Tesla T4, XGBoost, `device='cuda'`, cuDF off)
**Result**: **+0.00000 CV vs V2** (0.961011 vs 0.961009), folds improved 3/5 (deltas −0.00014, −0.00003, +0.00010, +0.00001, +0.00004) → **REJECTED**, the repair is neutral. LB 0.96058 then 0.96051 on the *same file* = **0.00007 LB noise measured in-house**.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.7 min (V2 3.8 min) |
| — per-fold prep | 9–10 s |
| — per-fold train | 27–37 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96103 | 0.96001 | 0.96187 | 0.96113 | 0.96102 | 0.96101 |

**Strategy:** one axis — replace V2's unshrunk `groupby().mean()` with `(n_k·mean_k + m·b)/(n_k + m)`, m=30, supports and prior `b` taken from the CLEANED original. Runtime diagnostics confirmed the defect it targets: **649 of 1,513 cells moved**, and competition rows mapping to a `_org_mean` of exactly 1.0000/0.0000 went **train 5,999 → 0, test 2,628 → 0** (the plan's `_gate_org_mean` case: 1,455 train / 617 test rows). Also: the plan's `b = 0.434463` re-derived here as **0.434379** (Δ −0.00008417); the recomputed value was used and the discrepancy printed as a warning.
**File:** `S6E10_V13_OrgMeanShrink.py` (root until archived)

**Key Learning:**

> A measured, provably-wrong input can still carry zero ranking value. V13 removed every saturated lookup cell the plan flagged as a concrete defect and CV did not move at all — the tree either ignored those sparse cells or the label noise swamped them. Combined with the same-file 0.00007 LB swing, this is the episode's second direct demonstration that **correctness of a feature is not the same as usefulness of a feature**, and that our own tie-band floor for LB is ~0.0003.

**Status:** ❌ Rejected (V2 stays the tree control); ✅ the two saturated-cell defects are permanently fixed in the archived file for any future frame rebuild; 🔬 LB-noise datum recorded

---

### Version 12 (Net input representation — 17 numeric→categorical twins + `_og_teacher` as logit) - 2026-10-11

**Score**: **0.96095 LB** / 0.96117 OOF (Gap: -0.00022) — **best single of the episode on both CV and LB**
**Device**: GPU (Tesla T4, PyTabKit 1.7.3, Lightning `max_epochs=3`, `device='cuda'`)
**Result**: **+0.00016 CV vs V2 with folds improved 5/5** (deltas +0.00017, +0.00018, +0.00022, +0.00012, +0.00012) and **+0.00028 CV vs parent V7** (5/5: +0.00036, +0.00027, +0.00035, +0.00012, +0.00028); **LB +0.00044 vs V2 — the only arm in the episode outside the 0.0003 tie band.** Clears Δ>+0.00005 and the ≥4/5 fold criterion; **paired DeLong z and the seed-2027 reproduction are still pending.**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 15.2 min (V7 13.8 min — the frame is nearly free) |
| — per-fold prep | 22–24 s |
| — per-fold fit | 154 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96134 | 0.96022 | 0.96199 | 0.96124 | 0.96110 | 0.96118 |

Best epochs **[3, 3, 3, 3, 3]** — still pinned at the budget ceiling, so this is the recipe's floor, not its ceiling.

**Strategy:** V7's `REALMLP_PARAMS` byte-identical; one axis = the net's input frame. 17 numeric predictors keep their numeric column and gain a factorised `_cat_` twin (21 categoricals total, 255 numeric + 21 cat = 276 cols handed to fit, expanded estimate 361, all-one-hot worst case 4,303); `_og_teacher` enters as its logit (clip 1e-6; prob range [0.000002, 1.000000], 0 saturated train cells after clipping).
**File:** `S6E10_V12_NetTwinTeacher.py` (root until archived)

**Key Learning:**

> **This arm was pre-declared as SKIPPED and it turned out to be the best model we have built.** The plan's rule — "the net below V2 twice ⇒ the net is disproved for our frame ⇒ skip V12" — conditioned on the net losing **while fed V2's tree-shaped frame**. The binding constraint was never the architecture, it was the *representation*: the same recipe on the twin+logit frame gained +0.00028 over V7 and jumped past every tree. Method consequence, to be applied to the remaining ladder: **a skip rule must be stated against the arm's own axis, not against its parent's configuration.** Also: teacher-as-logit removes the saturation the net was seeing (0 cells at p≤0/p≥1 after clipping) and is worth re-testing on trees (V16/V17 territory), and with best-epoch = budget in 5/5 folds the next honest net move is more epochs, not a new frame.
> Checkpoint-selection disclosure: PyTabKit restores the best-validation epoch and the CV fold *is* that validation set — the fold's own labels pick the checkpoint, as tree early stopping does. `flat_anneal`+3ep is the schedule-robust region, so this CV is directly comparable to V2's and V7's.

**Status:** 🏆 New best single (CV 0.96117 / LB 0.96095); **candidate control — gate not yet confirmed** (needs paired DeLong z > 3 vs `v2_oof.csv` and the `SEED_REPRO=2027` rerun); `v12_oof.csv` is now the strongest leg available

---

### Version 11 (Aux-EV features — 13 label-free expected-value columns) - 2026-10-11

**Score**: **0.96064 LB** / 0.96113 OOF (Gap: -0.00049)
**Device**: GPU (Tesla T4, XGBoost `device='cuda'`; the 13 EV predictors also fit on GPU)
**Result**: **+0.00012 CV vs V2 with folds improved 5/5** (deltas +0.00007, +0.00017, +0.00018, +0.00005, +0.00014) — **the first tree arm to clear both the Δ and fold criteria**; LB +0.00013 is inside the tie band, and CV sits 0.00004 below V12. Paired DeLong z + seed-2027 reproduction pending.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.8 min (V2 3.8 min) |
| — EV build | 116 s (original-fitted predictors, run-of-record) |
| — per-fold prep | 9–10 s |
| — per-fold train | 28–39 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96124 | 0.96021 | 0.96195 | 0.96117 | 0.96112 | 0.96114 |

**Strategy:** one axis — for each of the 13 rating columns, `aux_ev_<col>` = P̂(rating) predicted from the **other** rating columns on the cleaned original (129,859 rows, exact-overlap removed), label-free, added before the float32 cast so TE/params stay byte-identical to V2. Instability print recorded Pearson r of each EV against its raw rating: `inflight_entertainment` +0.9815 … `checkin_service` +0.5218 (low r = the EV carries independent information, high r = near-copy).
**File:** `S6E10_V11_AuxEVFeatures.py` (root until archived)

**Key Learning:**

> Four independent public measurements of aux-EV (+0.00009…+0.00017) reproduced here at +0.00012 with perfect fold consistency — the plan's "reliable cheap block" judgement was right and this is the only FE arm of the ladder that moved CV. But three `aux_ev_*` columns in the top-15 (`online_boarding` 1.16%, `leg_room_service` 0.73%, `seat_comfort` 0.60%) are worth only ~2.5% combined, and the gain still does not survive to LB. The EV r-values are the cheap diagnostic to keep: they tell us before training which columns can help.

**Status:** ✅ Cleared the CV + fold criteria, LB tie → kept as the strongest tree single and a first-class combination leg; ❌ not adopted as control until DeLong z and the 2027 rerun land

---

### Version 10 (Original-distribution distance features — 9 fold-independent columns) - 2026-10-11

**Score**: **0.96065 LB** / 0.96103 OOF (Gap: -0.00038)
**Device**: GPU (Tesla T4, XGBoost, `device='cuda'`, cuDF off)
**Result**: **+0.00002 CV / +0.00014 LB vs V2** — folds improved 3/5 (fold deltas −0.00001, +0.00003, +0.00006, +0.00003, 0.00000) → **gate FAILS** on the ≥4/5 criterion; paired DeLong vs `v2_oof.csv` still to be run externally. LB move is inside the 0.0003 tie band.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.7 min (V2 3.8 min — the axis is free at prep time) |
| — per-fold prep | 9 s |
| — per-fold train | 29–34 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96116 | 0.96007 | 0.96183 | 0.96115 | 0.96098 | 0.96104 |

**Strategy:** one axis on V2's frozen frame — for `Flight Distance` / `Age` / the rating-mean column, the competition row's class-conditional percentile rank in the clean original (`_od_*_pct0/pct1/pctgap`, 9 columns), midpoint 0.5 on NaN, clamped out-of-support. Base count proven in-run: 138 → 147 = +9 asserted, not hardcoded.
**File:** `S6E10_V10_OrigDistanceFeatures.py` (root until archived)

**Key Learning:**

> The plan expected +0.00005…+0.00035 from the S6E3 analogue and got +0.00002. The importances explain it: **no `_od_*` column reaches fold-1's top 15**, i.e. the tree barely selects the axis at all. S6E3's +0.00036 came from a frame that had no `_og_teacher` and no wide route profile; on V2's frame the original's marginal distribution is already consumed by the org means + teacher channel, exactly as the 2026-10-07 region dive predicted.

**Status:** ❌ Rejected as control (V2 stands); the arm is closed, not re-tuned

---

### Version 9 (TE row-statistics block — 5 `_tmr_*` columns over the `_tm_*` encoding vector) - 2026-10-11

**Score**: **0.96067 LB** / 0.96103 OOF (Gap: -0.00036) — **highest LB of the episode so far**
**Device**: GPU (Tesla T4, XGBoost, `device='cuda'`, cuDF off)
**Result**: **+0.00002 CV / +0.00016 LB vs V2** — folds improved 2/5 (fold deltas +0.00004, −0.00001, +0.00009, −0.00001, 0.00000) → **gate FAILS** on fold consistency despite the LB gain; paired DeLong vs `v2_oof.csv` pending.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.8 min (identical to V2) |
| — per-fold prep | 10 s |
| — per-fold train | 28–36 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96121 | 0.96003 | 0.96186 | 0.96111 | 0.96098 | 0.96104 |

**Strategy:** one axis on V2's frozen frame — per-row `mean/std/min/max/range` (float32, std ddof=1, range = max−min) over the fold's `_tm_*` target-encoded columns; +5 columns, 125 per-fold cols vs V2's 120.
**File:** `S6E10_V9_TErowStats.py` (root until archived)

**Key Learning:**

> The cleanest available test of "the model uses it" vs "it adds information": `_tmr_mean` takes **7.96%** and `_tmr_min` **6.83%** of fold-1 importance — 3rd and 4th overall behind `_og_teacher` (41.5%) and `_wifi_org_mean` (12.1%) — yet CV moves +0.00002 and folds improve only 2/5. **A feature can be heavily used and still add no ranking information**, because it is a deterministic summary of splits the tree already reaches. This is the strongest single-piece evidence for the 2026-10-07 verdict that the FE axis is finished here, and it retires the "dimensionality reduction" argument for the TE block.

**Status:** ⚠️ Best LB, gate-rejected on CV/fold consistency — record as tie, V2 stays control

---

### Version 8 (Leaf-wise XGB growth-policy ladder — `grow_policy='lossguide'`, `max_depth=0`, `max_leaves` 8/16/32) - 2026-10-11

**Score**: **0.96042 LB** / 0.96099 OOF (Gap: -0.00057) — rung-of-record `max_leaves=32`
**Device**: GPU (Tesla T4, XGBoost 3.2.0, `device='cuda'`, cuDF off)
**Result**: **−0.00002 CV / −0.00009 LB vs V2** — folds improved 1/5 (deltas −0.00005, 0.00000, +0.00004, −0.00007, −0.00002) → **REJECTED**. Ladder CV: 8 → 0.96089, 16 → 0.96097, 32 → 0.96099 (+0.00010 end-to-end, monotone).

**Timing:**
| Stage | Time |
|-------|------|
| Total | 11.1 min (3 rungs × 5 folds; V2 3.8 min single) |
| — per-fold prep | 9 s (shared by all rungs) |
| — per-fold train | 27–54 s (falls with more leaves: 40 s @8 → 28 s @32) |

**Fold Scores (rung 32):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96112 | 0.96004 | 0.96181 | 0.96105 | 0.96096 | 0.96099 |

Best iterations [854, 645, 719, 468, 513] — leaf-wise reaches the same ceiling at ~half the rounds of depth-wise (V2: [649, 465, 548, 662, 423]).

**Strategy:** one axis vs V2 (`XGB_PARAMS` only; `LADDER_RETUNE` empty, so `max_leaves` is the single key each rung moves and rungs are never combined). Per-rung OOFs saved as the audit trail: `v8_ladder_{8,16,32}_oof.csv`.
**File:** `S6E10_V8_XGBLeafWise.py` (root until archived)

**Key Learning:**

> S6E5's largest single-model win here (+0.00089 LB) does **not** transfer: that step bundled TE RowStats, and its isolated effect was never measured. On this dataset the depth-wise tree already reaches the leaf count it needs under `es=150`, so leaf-wise buys capacity the label noise cannot fill. Growth policy is closed as an axis — the ladder's own +0.00010 across a 4× leaf range is the noise floor for shape knobs.

**Status:** ❌ Rejected; per-rung OOFs retained as the closure evidence

---

### Version 7 (RealMLP with the proven in-episode recipe — axis `REALMLP_PARAMS` only) - 2026-10-11

**Score**: **0.96049 LB** / 0.96089 OOF (Gap: -0.00040)
**Device**: GPU (Tesla T4, PyTabKit 1.7.3, sklearn 1.6.1, Lightning `max_epochs=3`, `device='cuda'`)
**Result**: **+0.00025 CV vs parent V6** (0.96064→0.96089) but **−0.00012 CV vs V2**, folds improved 0/5 (deltas −0.00019, −0.00009, −0.00013, 0.00000, −0.00016) → **REJECTED as control**; the ship-as-control bar (CV ≥ 0.9614) is not met. LB +0.00004 vs V6, −0.00002 vs V2 = tie.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 13.8 min (plan estimate ~13 min — held) |
| — per-fold prep | 17–19 s |
| — per-fold fit | 143 s (V6 247 s at 6 epochs: half the budget ≈ 58% of the time) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96098 | 0.95995 | 0.96164 | 0.96112 | 0.96082 | 0.96090 |

Best epochs **[3, 3, 3, 3, 3]** — every fold pins the checkpoint at the budget ceiling, with `use_early_stopping=False` by design.

**Strategy:** goodpjw's measured RealMLP recipe (flat_anneal / 3 epochs / ES off / lr 0.053 / embedding_size 5 / `wd_sched='cos_log_15'` / `bias_init_mode='neg-uniform-dynamic-2'` / `p_drop_sched='expm4t'` / `ls_eps=0.01` `sqrt_cos`) on V6's frozen frame, folds, seed and float32 discipline. `ens_av_before_softmax` deliberately left unset (probability averaging = V6's behaviour).
**File:** `S6E10_V7_RealMLPRecipe.py` (root until archived)

**Key Learning:**

> The recipe was a real +0.00025 on the net (V6→V7) and it closed most of V6's gap to V2 (−0.00037 → −0.00012), which vindicates the "wrong schedule, not wrong architecture" re-reading — but the net still does not beat the tree. Crucially **best epoch = 3 in 5/5 folds**: with the budget fixed and ES off, the model is still improving at the cut-off, so the recipe arms (V5 6 ep, V6 6 ep, V7 3 ep) have all been epoch-starved, and none of them has tested the ceiling. Three net runs below V2 now triggers the plan's pre-declared rule: **V12 is SKIPPED, not re-tuned.**
> Checkpoint-selection disclosure (required): PyTabKit restores the best-validation epoch and the CV fold *is* the validation set, so the fold's own labels pick the checkpoint — the same fold-use tree early stopping makes. `flat_anneal`+3 ep is the schedule-robust region, so this CV is directly comparable to V2's.

**Status:** ❌ Rejected as control (V2 stands); ✅ `v7_oof.csv` is now the strongest net leg for any own-model combination

---

### Version 6 (RealMLP Bagged — single knob `n_ens` 1→8 on the frozen V5 file) - 2026-10-08

**Score**: **0.96045 LB** / 0.96064 OOF (Gap: -0.00019)
**Device**: GPU (Tesla T4, PyTabKit 1.7.3)
**Result**: **-0.00037 CV vs V2 (z=-5.90, 0/5 folds) → REJECTED as control**; **+0.000886 vs V5 (z=+13.40, 5/5 folds)** — bagging was the binding constraint, the architecture is not disproved

**Timing:**
| Stage | Time |
|-------|------|
| Total | 22.6 min (V5 15.3, V2 3.8) |
| — per-fold prep | 18–19 s |
| — per-fold fit | 247–249 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96085 | 0.95979 | 0.96161 | 0.96103 | 0.96069 | 0.96080 |

Best epochs [6,6,6,6,6] — `max_epochs=6 reached` again in 5/5: early stopping never fires, so the epoch budget is the next live constraint (not tested here; one-knob rule).

**Strategy:** byte-identical to V5 except `n_ens: 1 → 8` (AST: all 12 functions identical; 56 other changed lines are docstring/CFG/prints). Runtime was 22.6 min, not the ~2 h I estimated — PyTabKit's bags share the data pipeline.
**File:** `S6E10_V6_RealMLPBagged.py` (root until archived)

**Key Learning:**

> The net is now our best combination leg: equal logit-mean **V2+V4+V6 = 0.961201 (+0.000192** over V2 alone), better than V2+V4 (+0.000113), V2+V5 (+0.000112) and V2+V4+V6+V5 (+0.000137 — adding the weak V5 net *dilutes*, the pool-width rule showing up at n=4). RealMLP alone still loses to XGBoost by 0.00037, so with epochs untested and 6,140 NaN cells force-filled, the honest read is "competitive net, not a new control".

**Status:** ❌ Rejected as control (V2 stands); ✅ `v6_oof.csv` becomes the preferred third leg for any own-model combination

---

### Version 5 (RealMLP Single — architecture axis on the frozen V2 matrix) - 2026-10-07

**Score**: **0.96032 LB** / 0.95975 OOF (Gap: +0.00057)
**Device**: GPU (Tesla T4, PyTabKit 1.7.3, sklearn 1.6.1)
**Result**: **-0.00126 CV / -0.00019 LB vs V2** — REJECTED as control. Paired DeLong vs `v2_oof`: dAUC **-0.001258**, SE 8.0e-5, **z=-15.67**, folds improved **0/5** (harness verified: same code reproduces the V2-V1 gate at +0.000397 / z=+7.44). LB delta is inside the 0.0003 tie band; CV decides.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 15.3 min (vs V2 3.8 min) |
| — per-fold prep | 21–22 s |
| — per-fold fit | 154–158 s (fold wall 176–179 s) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96017 | 0.95916 | 0.96098 | 0.96049 | 0.96018 | 0.96019 |

Best epochs: [6, 6, 6, 6, 6] — `Trainer.fit stopped: max_epochs=6 reached`, i.e. early stopping never fired.

**Strategy:** V2 pipeline byte-identical (AST-verified: all 10 FE/teacher functions unchanged; only `build_fold_features` gained a categorical NaN fill), one axis = main model XGB → `RealMLP_TD_Classifier(n_ens=1, n_epochs=6, lr_sched='flat_anneal', lr 0.075, wd 0.0236, hidden [512,256,128], act silu, PLR + label smoothing, tfms one_hot/median_center/robust_scale/smooth_clip/embedding/l2_normalize)`, X_val block passed. Forced second change: PyTabKit rejects NaN in continuous columns, so 6,140 NaN cells were filled with fit-fold medians (0.0014% of the matrix).
**File:** `S6E10_V5_RealMLPSingle.py` (root until archived)

**Key Learning:**

> n_ens=1 + 6 epochs is an undertrained net, not a falsified architecture — every published RealMLP number here (0.961344 goodpjw, 0.961413 our audited library leg, S6E8 1st place) used internal bagging (n_ens 8–32), so the next RealMLP arm must iterate this file in place with n_ens>1 rather than conclude the class is dead. Second finding worth keeping: the CV→LB gap flipped sign (+0.00057 vs V2's −0.00050), consistent with in-fold early stopping biasing the tree control's CV down while an ES-starved net has no such handicap.

**Status:** ❌ Rejected as control — V2 (0.96101 / 0.96051) remains; `v5_oof.csv` archived as a candidate stack leg only

---

### Version 4 (CatBoost Member — V2 package, main model swapped) - 2026-10-02

**Score**: **0.96051 LB** / 0.96088 OOF (Gap: -0.00045)
**Device**: GPU (CatBoost GPU + cuda teacher)
**Result**: **-0.00013 CV / ±0.00000 LB vs V2** — solo no-gain; member value proven

**Timing:**
| Stage | Time |
|-------|------|
| Total | 7.9 min |
| — per-fold prep | 11 s |
| — per-fold CB fit | 69–85 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96094 | 0.95999 | 0.96171 | 0.96099 | 0.96078 | 0.96088 |

BestIters: [946, 1195, 940, 1182, 1141] (early stop on Logloss — GPU build lacks AUC metric).

**Strategy:** V2 control with ONE axis: main model XGB→CatBoost (d8, lr0.05, Bernoulli ss0.8, l2 5). Same frozen folds/seed, same 259-col matrix, same XGB-d8 teacher.
**File:** `S6E10_V4_CatBoostMember.py` (root until archived)

**Key Learning:**

> Two GBDTs on the same matrix correlate at 0.985 (residual corr 0.997) — yet the 0.4-CB logit blend still gives +0.000113 over the best single, confirming stacks earn from ANY decorrelation. CB's importance profile differs structurally: `_tm_fd_exact` #2 (4.15), `_route_wifi_resid` #4 (1.33) — route residuals matter more to CB than to XGB. Bigger stack gains need a different inductive bias (NN member / aux-EV carriers), not a third GBDT twin.

**Status:** ✅ Kept as V7 stack member; V2 remains CV control

---

### Version 3 (XGBoost Duo Teacher — xgb d8 + LightGBM logit-mean) - 2026-10-02

**Score**: **0.96065 LB** / 0.96096 OOF (Gap: -0.00045)
**Device**: GPU (cuda)
**Result**: **±0.00014 LB / -0.00005 CV vs V2** — tie, inside noise

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.2 min (prep 10 s/fold, fit 26–33 s/fold) |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96115 | 0.96002 | 0.96170 | 0.96108 | 0.96091 | 0.96097 |

BestIters: [648, 523, 540, 549, 290].

**Strategy:** V2 with ONE axis: `_og_teacher` = logit-mean of XGB(d8, ektarr spec) + LightGBM(600×64 leaves, ss0.8/cs0.5), both on overlap-cleaned data.csv.
**File:** `S6E10_V3_DuoTeacher.py` (root until archived)

**Key Learning:**

> Tree+tree teacher diversity on the SAME source data adds nothing — both teachers recover the same clean law, and V2's d8 single teacher already saturates that channel (its gain share held at 40.0 vs 40.4 in V2, pair-TE and org-mean shares unchanged). goodpjw's two-teacher +0.00072 was tree+RealMLP: diversity must be across INDUCTIVE BIAS, not across tree libraries. Teacher capacity axis is now closed at single-XGB level.

**Status:** ⚠️ Not adopted — V2 (`v2_oof.csv`) remains CV control; DeLong vs V2 deferred until `v3_oof.csv` archived (expected tie)

### Version 2 (XGBoost Wide Route Profile + Full-Featured Teacher) - 2026-10-02

**Score**: **0.96051 LB** / 0.96101 OOF (Gap: -0.00050)
**Device**: GPU (cuda)
**Result**: **+0.00032 LB / +0.00040 CV vs V1** (all 5 folds improved: +0.00034/+0.00030/+0.00038/+0.00045/+0.00052)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.8 min (vs V1 26.7 min — int-coded pair keys killed the ~290 s/fold string-join prep) |
| — per-fold prep | 10–11 s |
| — per-fold XGBoost fit | 28–35 s |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96117 | 0.96004 | 0.96177 | 0.96112 | 0.96098 | 0.96101 |

BestIters: [785, 648, 530, 405, 482]; fold mean ± std 0.96101 ± 0.00056.

**Strategy:** Four axes changed vs V1, all pre-registered as the V2 package: (1) int-factorized pair keys (prep speed); (2) teacher upgraded to ektarr ORIGINAL_MODEL spec — d8, cs0.5, ss0.8, 600 trees, ALL 21 features with native cats (V1 teacher: d6, 17 numeric feats); (3) wide route profile — per-FD std + value-minus-route-mean residual + 10 categorical shares + count over the pool; (4) org label means for all 21 features (raw group means, ektarr protocol) + Travel×Class×Customer triple mean. 139 base + 120 per-fold = 259 columns.
**File:** `S6E10_V2_XGB_WideProfile.py` (root until archived)

**Key Learning:**

> Teacher capacity paid exactly as predicted: share of fold-1 gain 0.4797→0.4043 while overall CV rose — the stronger teacher let pair TEs (+0.0961 share, ob×travel now #2 at 0.176) and org means (_cust_org_mean #4, _triple_org_mean in top-10) carry more, i.e. the whole orig-prior family got sharper, not the single column. _cust_org_mean entering top-15 confirms ektarr's all-21 org means (V1 had only ratings+FD). CV gain +0.00040 → LB +0.00032: CV-vs-LB tracking holds at this scale; the −0.0005 offset is stable across V1/V2.

**Status:** ✅ ACCEPTED as new control — paired DeLong vs v1_oof: ΔAUC +0.000397, z=+7.44 (gate Δ>+0.00005 ∧ z>3 PASS; bootstrap 0/200 flips)

### Version 1 (XGBoost Baseline — curated TE + native cats + orig teacher) - 2026-10-02

**Score**: **0.96019 LB** / 0.96061 OOF (Gap: -0.00042)
**Device**: GPU (cuda)
**Result**: **official baseline** (dev-run with house params α10/λ10 + blanket TE had scored 0.96009 CV, not logged as a version)

**Timing:**
| Stage | Time |
|-------|------|
| Load | ~2 s |
| Feature engineering (digits, aggregates, route profile, org stats, teacher) | ~20 s |
| CV training (5 folds) | 26.3 min |
| — per-fold feature prep (string-pair keys + nested-cv TE) | 287–292 s (~90% of fold time) |
| — per-fold XGBoost fit | ~20–35 s |
| Save + total | 26.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.96083 | 0.95974 | 0.96139 | 0.96067 | 0.96046 | 0.96062 |

BestIters: [1141, 790, 1001, 742, 538]; fold mean ± std 0.96062 ± 0.00054.

**Strategy:** ektarr-aligned recipe: light params (d6 lr0.04 mcw5 ss0.9 cs0.85 λ5 α0.05 es150), curated per-fold TE (smooth=20 inner-cv5; 17 singletons + 10 pair keys + FD multi-res + age/fd bins), frequency+rarity, native categoricals, pool route profile, orig per-value means, orig-only teacher (d6 numeric-17 in this run); no sample weights; deterministic AND-flag features excluded. 90 base + 86 per-fold = 176 columns.
**File:** `S6E10_V1_XGB_Baseline.py` (root; note — file was later edited in place with the int-key/wide-profile revision; that un-run code is the V2 candidate, the 0.96019 run corresponds to the string-prep/thin-teacher version described in Strategy)

**Key Learning:**

> Teacher-as-feature is the dominant component (0.4797 of fold-1 gain; org means and pair TEs next) — consistent with the measured mechanism that data.csv carries the clean label law under ~4% symmetric flip noise. Fold cost is dominated by per-fold prep (string pair-key joins, ~290 s/fold), not training: BestIter 538→1141 moved fold time 5%. CV−LB gap -0.00042 is inside the public tie-band; decide on CV, audit on LB.

**Status:** ✅ Official baseline logged (v1_oof.csv frozen as paired-DeLong control)

---

## Appendix — magnitude re-reads, 2026-10-12 (`LEDGER`: recomputed from the fold tables in this file)

No entry above is edited (rule 2). This annotates four recorded CV deltas that compare **nets across
different `n_epochs`/`n_ens` on the pooled OOF**, a yardstick the campaign has since ruled out for that use:
with `use_early_stopping=False` the restored checkpoint is chosen by that fold's own labels, so longer
training sharpens each fold in place while breaking the cross-fold comparability a pooled AUC depends on.
Recomputing each pair on the **fold-mean** basis (mean of the five logged fold AUCs):

| Comparison | Recorded (pooled) | Fold-mean basis | Fold tables |
| --- | --- | --- | --- |
| V5→V6 (`n_ens` 1→8) | **+0.000886** | **+0.000598** | 0.960196 → 0.960794 |
| V6→V7 (6→3 epochs) | +0.00025 | +0.000108 | 0.960794 → 0.960902 |
| V12→V17 (3→10 epochs, rung of record) | **−0.00025** | **−0.000086** | 0.961178 → 0.961092 |
| V12→V18 (same 3-epoch budget) | +0.00012 | +0.000114 | 0.961178 → 0.961292 |
| V5 vs V2 (net vs tree) | −0.00126 | −0.000820 | 0.960196 vs 0.961016 |

**Reading.** No verdict changes: V17 stays rejected (negative on both bases, folds improved 1/5), `n_ens` 1→8
stays the largest effect the campaign has measured. What changes is strength. About a third of the +0.000886
headlines the offset, and **64% of V17's −0.00025 is the offset** — so "the net's epoch axis is closed at 3"
should be quoted as an ≈−0.00009, mixed-sign-fold axis, i.e. tie-band-sized, rather than as a large
rejection. The V12→V18 row is the control case: same budget on both sides, both offsets ≈−1e-5, and the two
bases agree to 0.000006, which is what makes the mechanism identifiable rather than asserted.

**Consequence for pending arms:** V26 (`n_ens` 8/16/32) compares a net against itself across bag counts, so
its rung-to-rung deltas must be read on fold means; V23's and V26's "control reproduction distance" lines are
same-config comparisons and are safe on the pooled value. See `PLAN_V21_ONWARD.md` §7 finding 29 and §8.
