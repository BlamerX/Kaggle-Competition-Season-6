# S6E10 Public Leaderboard Scores

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category, with the format block first and version rows written below it
> 4. **ORDER** by LB Score (Highest on Top)
> 5. **Include:** OOF, LB, Gap, Training Time
> 6. **CATEGORIZE:** Per the headers below (Two-Stage Models, Multiseed are considered as Single models).
> 7. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit
> 8. **TOP 5 table holds exactly the five highest LBs** (any category); all other versions live in Single Models or Ensembles.

---

## 📝 Score Logging Format

The format block stays first; score entries are written below it.

| Version | Date  | LB Score | OOF Score | Gap    | Time   | Script    | OOF File  | Sub File  | Notes |
| ------- | ----- | -------- | --------- | ------ | ------ | --------- | --------- | --------- | ----- |
| V#      | MM-DD | X.XXXXX  | X.XXXXX   | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

### Leaderboard Scores Top 5

The five highest LBs of every submission; each also appears in its category table below (Top 5 is a view, not a copy).

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | --------- | --- | ---- | ------ | -------- | -------- | ----- |
| V51 | 10-25 | **0.96113** | 0.961585 | -0.00046 | 39.8 s (CPU, no fit) | `S6E10_V51_TrioDiverseBlend.py` | `v51_oof.csv` | `v51_sub.csv` | New highest LB and the experiment that settled the offset: 3 legs instead of 5, CV 0.000010 lower, board 0.00001 higher, gap -0.000455 vs -0.000475 and -0.00048 - so the CV→LB offset belongs to the blend's family mix, not its leg count. The +0.00003 over V29 is 0.5 of a ruler: a record number, not a proven gain 🏆 |
| V50 | 10-25 | **0.96112** | **0.961595** | -0.00048 | CPU, seconds (no fit) | `S6E10_V50_QuintetChampionBlend.py` | `v50_oof.csv` | `v50_sub.csv` | New highest LB and the first arm to pass the full campaign gate outright: CV +0.00032 vs both parents, folds improved 5/5 on the house partition, bootstrap paired DeLong z +3.38 vs its own predecessor V49 and +10.56 vs V18. The board move over V29 is +0.00002 = 0.4 of a ruler, so this is a new record number, not yet a proven new model 🏆 |
| V29 | 10-17 | **0.96110** | 0.96128 | -0.00018 | 17.6 min | `S6E10_V29_PortStage1KeysToNet.py` | `v29_oof.csv` | `v29_sub.csv` | Highest LB in the file -- **and V33/V34 now show the number is not a gain**: the same four keys read -0.000007 CV at seed 42 and +0.000010 CV at seed 2027, and the LB edge itself flips from +0.00004 to +0.00006 across seeds, both inside one seed tick. A CV tie against its own parent that the ruler explains ⚠️ |
| V49 | 10-25 | **0.96108** | **0.96156** | -0.00048 | 6.9 s (local CPU) | `S6E10_V49_CompositionBlend.py` | `v49_oof.csv` | `v49_sub.csv` | **The Ensembles table's first entry** — equal-weight logit-mean of v11+v27+v34+v36 (the hill-climb optimum over all 42 archived OOFs; members' corrs 0.9959-0.9988). LB **0.96108 ties V32 for 2nd-highest ever**; the measured CV→LB offset **-0.00048** is the largest yet and extends the averaging-arm law to diverse blends — own-pool composition is offset-bound at ~0.9611 LB. V34 (0.96106, tied with V18) leaves the view ⚠️
| V32 | 10-17 | **0.96108** | 0.96131 | -0.00023 | 29.1 min | `S6E10_V32_NetBaggingRung.py` | `v32_oof.csv` | `v32_sub.csv` | `n_ens` 8->16 on V18's byte-identical frame: CV +0.000023 with **5/5 folds improved**, LB +0.00002 -- every fold agrees on the sign and none of it exceeds a third of the seed tick V33 measured. Second-highest LB in the file, gate fails on magnitude, control unchanged ⚠️ |

---

## Ensembles (combiners over saved OOF vectors)

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | --------- | --- | ---- | ------ | -------- | -------- | ----- |
| V51 | 10-25 | **0.96113** | 0.961585 | -0.00046 | 39.8 s (CPU, no fit) | `S6E10_V51_TrioDiverseBlend.py` | `v51_oof.csv` | `v51_sub.csv` | **New highest LB, and the answer to the question this arm was built to ask.** Three members instead of V50's five - **v27 + v36 + v44**, the best 3-subset over all 14 single-arm vectors (exhaustive over every 2- and 3-member set, not greedy) - CV **0.961585**, 0.000010 *below* V50, and it scored **0.96113**, 0.00001 above. The gap is **-0.000455** against V49's -0.00048 and V50's -0.000475, so **the offset does not shrink when legs are removed: H1 (family mix) wins, H2 (leg count) is dead**, and three blends now calibrate the predictor **LB ≈ CV − 0.00046 ± 0.00001**. Pooled minus fold mean -0.00000; paired bootstrap vs `v50_oof` z **-1.25**, insignificant exactly as pre-declared (this arm made no CV claim). Adoption nuance the ledger must keep: V51 is the best *public* number and V50 is the better *private* bet, because the difference between them is 0.00001 of CV - 0.2 of a ruler - and the final board is the other 80% of test. Neither is a proven gain over V29 either: +0.00003 = 0.5 ruler 🏆 |
| V50 | 10-25 | **0.96112** | **0.961595** | -0.00048 | CPU, seconds (no fit) | `S6E10_V50_QuintetChampionBlend.py` | `v50_oof.csv` | `v50_sub.csv` | **New highest LB, and the first arm in this campaign to pass the entire gate outright.** Equal-weight logit-mean of **v44 + v45 + v36 + v27 + v11** (3-seed nets, 3-seed keys net, 10-fold net, duo-teacher tree, aux-EV XGBoost). CV recomputed from the shipped vector = **0.961595** (+0.00032 vs V18, +0.00032 vs V29); per-fold on the house partition 0.96173 / 0.96069 / 0.96237 / 0.96174 / 0.96147 -> **5/5 folds improved vs V29** (+0.00039, +0.00027, +0.00024, +0.00033, +0.00034); paired DeLong by full-row bootstrap **z +3.38 vs v49** and **+10.56 vs v18** (the arm's own +3.34 / +11.8 reproduce to ~12%). **ΔLB vs V29 is only +0.00002 = 0.4 of a ruler, so the record changes and the board evidence does not** - but the CV, fold and z evidence against both parents is decisive, and by this ledger's own law z binds harder than Δ. Members' mutual logit corr 0.99644-0.99989; greedy forward selection over the whole pool lands on exactly this quintet and a 6th member lowers it (0.961584), so 'no 6th member helps' is confirmed within greedy reach while 'mathematically proven global optimum over 47 vectors' is not (no exhaustive search was run) 🏆 |
| V49 | 10-25 | **0.96108** | **0.96156** | -0.00048 | 6.9 s (local CPU) | `S6E10_V49_CompositionBlend.py` | `v49_oof.csv` | `v49_sub.csv` | **The Ensembles table's first entry, and the composition phase's first measured LB**: equal-weight logit-mean of v11+v27+v34+v36 — the hill-climb optimum over all 42 archived OOF vectors (OOF **0.96156 = +0.00028 CV vs V18**, in-run paired DeLong z +8.76; SLSQP returns uniform weights; rank-mean ties logit-mean). Members are genuinely different columns (tree / duo-teacher tree / keys-frame reseed / 10-fold net; mutual corrs 0.9959-0.9988, below finding 82's collapse bar) and the blend's CV gain was real — but LB **0.96108 ties V32 for 2nd-highest ever** (+0.00002 above the V18 band top, -0.00002 vs V29 0.96110) and the measured CV→LB offset is **-0.00048, the largest constant yet** (V36 -0.00035, V44 -0.00038, V45 -0.00037): the averaging-arm offset law extends to DIVERSE cross-family blends, so **own-pool composition is offset-bound — its OOF ceiling 0.96156 maps to LB ~0.9611, and no composition of our own 43 vectors passes 0.9611 on the board**. The route above is the public pools (kratosyan's 119-member stack measured OOF 0.962048 → LB 0.96167 at his own -0.00038; Deotte's 11 TFM OOFs — tabpfn35 solo 0.961145, above our best single; the 0.96213 wall file; Jiatian's 15-fold GPU stack OOF 0.96238). V34 (0.96106, tied with V18) leaves the Top 5 view and remains in its category table ⚠️

---

## Single Models

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | --------- | --- | ---- | ------ | -------- | -------- | ----- |
| V55 | 10-28 | 0.96094 | 0.96117 | **-0.00023** | 19.4 min (prep 23s + fit 144s/fold) | `S6E10_V55_RealRowsAugment.py` | `v55_oof.csv` | `v55_sub.csv` | **The last untried in-house data axis, and it closes on measurement**: 24,731 overlap-cleaned original rows (+4.4% of the fit) appended to each fold's competition rows, V18's 289-column frame, recipe, folds and budget untouched. Leakage-safe by construction — R 103,887 / A 25,972 split, **0 of 25,972 A predictor vectors shared with R**, and every label-consuming producer refit on R alone for the appended rows. **The arm's own instrument proves the rows were used, which is what makes the loss meaningful**: fold-1 in-sample AUC **0.99678 on the appended rows** vs **0.96594 on the competition fit rows** vs 0.96126 held-out — the net memorises the real regime almost perfectly and none of it transfers. **ΔCV -0.00012 pooled (-2.1 rulers) / -0.00005 on fold means, folds improved 1/5, ΔLB -0.00012**; the gap -0.00023 reproduces V18's frame constant. Fails three of its four pre-declared admission criteria outright (solo CV 0.96117 is 0.00023 under the ≥0.96140 member-strength bar), so **no board slot**; the offline paired test on the archived vectors gives **dAUC -0.000119 = -2.09 rulers, full-row bootstrap z -3.63** (the binding estimator, validated at +3.26 against the ledger's recorded +3.38 before use; the analytic paired DeLong's z -108 is reported as a floor only, since it overstates by ~30× on 0.998-correlated pairs). **And the leg question was measured rather than waved away by the bar: trio+v55 = 0.961471, -0.000114 CV, 0/5 folds** — v55 is the least damaging leg ever tried (v52 -0.000076, v55 -0.000114, v53 -0.000239, v54 -0.000937) and it still lowers the blend. Coverage block printed 4.78% of A dropped (1,241 of 25,972 across three columns) — **digit-for-digit what the local pre-run pass predicted**, so the data path is verified twice ❌ |
| V54 | 10-28 | 0.95965 | 0.958693 | **+0.00096** | 18.6 min (prep 23s + train 157s/fold) | `S6E10_V54_DCNv2Family.py` | `v54_oof.csv` | `v54_sub.csv` | **DCN-V2 family swap on V18's untouched 289-column frame** — our own hand-written net (S6E9 V13 implementation, hyperparameters unchanged; Stratified partition and a fit-fold `pos_weight` declared as deviations). **ΔCV -0.002594 (-45.5 rulers) vs V18, ΔLB -0.00141**; folds 0.95886 / 0.95782 / 0.95979 / 0.95909 / 0.95846, 1.18M params, `input_dim 468` (268 numeric + 200 embedded over 21 categorical embeddings). The finding is convergence, not cost: **best epochs [3,3,2,3,2] of a 50 ceiling** — it peaks in three epochs and degrades for the whole patience window in all five folds, so the family cannot exploit this frame, and the run was 16× under its own budget canary. Structural read-back, column-split control and the 0.90 learning floor all passed; `v54_oof` ships 110 exact-0 and 86 exact-1 rows (0.0280% saturated, clipped to \|logit\| 13.82 in composition). Fails its own bar and its own leg precondition (`solo >= 0.9600`), which is what excluded it from V55 ❌ |
| V53 | 10-28 | 0.96072 | 0.960667 | **+0.00005** | 140.3 min (prep 21s + fit 1,598s/fold) | `S6E10_V53_TabMFamily.py` | `v53_oof.csv` | `v53_sub.csv` | **TabM_D family swap on V18's untouched frame**, config copied from our own S6E9 V6 (`tabm-mini-normal`, k 16, pwl emb 16, d_block 256, n_blocks 3, batch 1024, lr 1e-3, wd 1e-3, **n_epochs 30 + patience 10**) rather than from a public kernel. **ΔCV -0.000620 (-10.9 rulers) vs V18, ΔLB -0.00034**; folds 0.96094 / 0.96013 / 0.96151 / 0.96082 / 0.96029; peak reserved 8.54 GB of 14.56 GB; read back twice (pre-flight probe and fitted object) so the run demonstrably trained what it describes. **The caveat: best epochs [30,30,30,30,30] against a 30 ceiling — early stopping never fired**, so this is a budget-limited reading of the family, not its ceiling on our frame. Its fold-1 canary (26.1 min vs a 75 min ceiling) projected 131 min against 137.9 actual, the first budget instrument here validated by its own run. Cosmetic: the variant-discovery print filtered lowercase `'abm'` and so printed an empty list although the class imported fine ❌ |
| V52 | 10-27 | 0.96102 | 0.961295 | -0.000275 | not captured (run log not attached) | `S6E10_V52_InertFrameDrop.py` | `v52_oof.csv` | `v52_sub.csv` | Selected K8-inert families removed: 81 columns, intended width 289 → 208, V18 parameters unchanged. Earlier full-row recheck: **ΔCV +0.000008, paired-bootstrap z +0.32, 2/5 folds improved**; user supplied LB. The single-model CV/fold gate fails; no demonstrated improvement for this drop set. This does not close all feature selection or establish a global score ceiling. V52's blend value is still unverified. Date is the recording date; stdout and runtime are unavailable. |
| V48 | 10-25 | 0.96091 | **0.96070** | **+0.00021** | 53.8 min (37.9 of it member fits) | `S6E10_V48_StackTrainedNet.py` | `v48_oof.csv` | `v48_sub.csv` | **Stack training, and the campaign's first positive gap.** Three in-run GBDT members (XGB V11-config 0.96105 / LightGBM V15-config 0.96112 / CatBoost V4-config 0.96103) cross-fitted on the house split enter the parent's frame as 3 logit columns; one RealMLP student fits the widened frame. **CV 0.96070 = -0.00059 vs V18 (-10.4 rulers) with 0/5 folds improved** (-0.00030, -0.00034, -0.00027, -0.00033, -0.00025), LB 0.96091 and **0.00009 BELOW the V18 band floor**. The student is -0.00042 against its own best member, so the cascade did not even reach its parts. The in-run census explains it: the members are ~one channel (mutual logit corr 0.9978-0.9990) sitting 0.962 from the `_og_teacher` the frame already carries. Gap **+0.00021** is pooled-OOF drift, not luck - pooled 0.96070 vs fold mean 0.96099 is **-0.00029**, an order of magnitude past every single-model net arm (<=0.00001); read against the fold mean the offset is the ordinary -0.00008. Unlike V44/V45 this arm declared **no implied LB** and so was not caught by its own arithmetic; its CV gate failed by -0.00069 ❌ |
| V45 | 10-25 | 0.96105 | **0.96142** | -0.00037 | 48.8 min (3 legs x 5 folds) | `S6E10_V45_KeysFrameMultiSeed.py` | `v45_oof.csv` | `v45_sub.csv` | The same seed-axis on the keys frame: V29 (the standing best, LB 0.96110) byte-identical at seeds [42, 2027, 1997], shipped = logit-mean. Legs 0.96127 / 0.96125 / 0.96129 -> blend **0.96142** = **+0.00014 CV vs V29** (2.5 rulers), clearing its own CV bar (>= 0.96138). LB **0.96105** = **-0.00005 vs its parent and 0.00001 below the V29 frame's own recorded floor (V34 0.96106)**: the offset grew from that frame's -0.00018 to **-0.00037**, and the implied LB the header pre-declared (0.96120-0.96123) missed by 2.6-3.2 rulers. Its same-seed legs reproduce V29/V34 to within 1e-5 at the 5th decimal, so the digit-for-digit control misses by one last-place digit on both legs, not by nothing. V44 and V45 land on the SAME blended OOF from two different frames. 6th on the board (tied with V28 0.96105); not adopted, V29 0.96110 remains the best LB ⚠️ |
| V44 | 10-25 | 0.96104 | **0.96142** | -0.00038 | 46.4 min (3 legs x 5 folds) | `S6E10_V44_MultiSeedRealMLP.py` | `v44_oof.csv` | `v44_sub.csv` | **The campaign's joint-highest OOF ever** (with V45), 8th on the board, tied with V41/V36, and the board did not move: V18's byte-identical recipe fitted at seeds [42, 2027, 1997], shipped as the logit-mean of the three legs. Legs 0.96126 / 0.96127 / 0.96132 -> blend **0.96142** = **+0.00013 CV vs V18** (2.3 rulers), which clears the arm's own pre-declared CV bar (>= 0.96139). LB **0.96104** sits inside V18's band 0.96100-0.96106, so **+0.00013 CV bought -0.00002 LB**: the CV->LB offset grew from the frame's -0.00023 to **-0.00038**, i.e. by more than the CV gain, and the arm's own pre-declared implied LB (0.96116-0.96119) missed by 2.1-2.6 rulers. Its determinism control also fails: the seed-42 leg reads 0.96126 against the parent's recorded 0.96129, so a same-seed net re-run is NOT digit-for-digit (-0.00003 = 53% of a ruler) - the control that held for XGBoost (K5 vs V11, V30 vs V20) is tree-only. Offline paired DeLong vs `v18_oof` still owed. Tie-maker by its own gate -> not adopted ⚠️ |
| V43 | 10-24 | 0.96101 | 0.96124 | -0.00023 | 17.8 min | `S6E10_V43_IpsativeRatings.py` | `v43_oof.csv` | `v43_sub.csv` | The last un-tried transform family: **+13 within-respondent relative ratings** (`_rel_<rating> = r - _mean_excl0`). **CV -0.00005 vs the control with only 1/5 folds improved** (paired: -0.00008, -0.00004, -0.00009, +0.00007, -0.00009) -> gate fails on magnitude AND folds, direction negative. Its own control prints the reason the net cannot use it: deviation sd 1.087, min -3.69, max +3.62 - a re-reference of inputs the net already has, and V41 had just shown the net does not represent the row-mean block at all. Gap -0.00023 reproduces the V18-frame constant for the 6th time ⚠️ |
| V42 | 10-24 | 0.95965 | 0.95990 | -0.00025 | 14.0 min | `S6E10_V42_CatBoostMVS.py` | `v42_oof.csv` | `v42_sub.csv` | CatBoost `bootstrap_type` Bernoulli → **MVS** at the parent's own `subsample=0.8` — the algorithm our ledger had mislabelled as tested (V35 was XGBoost GOSS). Fold-1 in-run control, paired on the same matrix: Bernoulli 0.96100 vs MVS 0.95964 = **−0.001361 = 23 rulers**, with best iterations ballooning to 2932–4240. MVS does not just fail to help here, it slows convergence ❌ |
| V41 | 10-24 | 0.96104 | 0.96127 | -0.00023 | 16.4 min | `S6E10_V41_NetFrameSubtraction.py` | `v41_oof.csv` | `v41_sub.csv` | The subtraction direction nobody had pulled on a net: V18's frame **minus 37 tree-centric columns** (32 digit + 17 rating-aggregate, printed as dropped). CV −0.00002 vs V18 with folds 4/5 ahead — the RealMLP never needed those blocks, so the net's frame is now closed in both directions ⚠️ |
| V40 | 10-24 | 0.95196 | 0.95319 | -0.00123 | 4.7 min | `S6E10_V40_PriorAsBaseMargin.py` | `v40_oof.csv` | `v40_sub.csv` | Prior as `base_margin` instead of a column. **Decisive negative, and it is a mechanism finding**: Kaggle's xgboost names `base_margin` but not `eval_base_margin`, and its `fit` rejects a DMatrix eval pair, so ES was margin-blind → **BestIter 0 in 5/5 folds** and the shipped vector is the prior itself (offset-alone control 0.95508). Route self-reported as `es_margin_blind` ❌ |
| V39 | 10-24 | 0.96097 | 0.96120 | -0.00023 | 17.0 min | `S6E10_V39_NumEmbPlr.py` | `v39_oof.csv` | `v39_sub.csv` | `num_emb_type` 'pbld' → **'plr'** — the function-class switch 38 arms never touched. Read-back after fold 1 confirms it was live and that our inherited `plr_act_name='gelu'` finally applies (it never did under pbld). CV −0.00009 vs V18 → the untried embedding family is not better, and the gelu question dies with it ❌ |
| V38 | 10-24 | 0.96100 | 0.96125 | -0.00025 | 68.5 min | `S6E10_V38_KnnOrigLabelMean.py` | `v38_oof.csv` | `v38_sub.csv` | +1 column `knn_orig_prob` (K=50 label mean over the overlap-cleaned original; solo AUC **0.94997**, 0 distance-0 neighbours, 0 base-rate fallbacks). CV −0.00004 vs V18. The only genuinely new estimator the cross-episode pass produced, and it is dominated by the teacher it imitates (0.95508). 51 min of the 68.5 went to the k-NN build ❌ |
| V36 | 10-23 | 0.96104 | **0.96139** | -0.00035 | 33.8 min | `S6E10_V36_TenFoldAveraging.py` | `v36_oof.csv` | `v36_sub.csv` | **The campaign's largest CV gain, and the board did not move: CV +0.00010, LB inside the parent's own band.** Outer folds 5→10 on V18's byte-identical frame and recipe (one CFG value; the inner TE/EV cross-fits stay at 5 on purpose): **OOF 0.96139 — the highest CV this file has ever carried** — fold mean 0.96140, per-fold 0.96012…0.96230, and `pooled − fold mean` −0.000004 (no drift, the calibration the net arms owe). LB **0.96104** sits inside V18's measured LB band 0.96100–0.96106, i.e. **−0.00002 vs the parent's best point = a tie**, while its CV→LB offset is **−0.00035, not the −0.00023 the 5-fold frame reproduces**: the offset constant is *protocol*-dependent, not only frame-dependent — see the dated appendix. DeLong vs `v18_oof` still pending the vector ⚠️ |
| V37 | 10-23 | 0.96107 | 0.96128 | -0.00021 | 30.8 min | `S6E10_V37_PlrSigmaLadder.py` | `v37_oof.csv` | `v37_sub.csv` | **`plr_sigma` is at its plateau on our frame: both rungs tie the parent.** Ladder {8, 32} on V18 (dict inherited byte-for-byte, key overridden per fit; rung of record pre-declared as the higher CV): σ=8 **0.96128**, σ=32 **0.96121**, parent 2.33 **0.96129** ⇒ **−0.00001 and −0.00008**. The rung pair is paired per fold (same split, same prep): σ=32 − σ=8 = +0.00005, −0.00013, −0.00014, −0.00010, +0.00002 ⇒ mean **−0.00006**, one ruler. LB 0.96107 is +0.00001 over the parent's band top — a tie, and the control does not move. Deotte's published +5.5e-4 was measured moving **0.1 → 32**; we were already 23× of the way there, so his gain was banked in V7's bundle, not left on the table ❌ |
| V32 | 10-17 | **0.96108** | 0.96131 | -0.00023 | 29.1 min | `S6E10_V32_NetBaggingRung.py` | `v32_oof.csv` | `v32_sub.csv` | `n_ens` 8→16 on V18's byte-identical frame and 3-epoch budget (one parameter): **CV +0.000023 pooled, folds improved 5/5 with every delta in [+0.00001, +0.00003], LB +0.00002 — all four yardsticks agree on the sign and none passes the gate**, because +0.000023 is **0.4 of the seed-noise ruler V33 measured (0.000057)**. Second-highest LB in the file, and the read is about the lever, not the record: **bagging is saturated** — 1→8 bought +0.00089, 8→16 buys +0.00002 for double the fit time (268 s vs 147 s per fold). The pre-declared calibration test also fails to fire: `pooled − fold mean` is −0.000007 here vs −0.000005 on V18, so there was never a bagging offset to shrink ⚠️ |
| V34 | 10-17 | **0.96106** | 0.96124 | -0.00018 | 17.3 min | `S6E10_V34_V29SeedRepro.py` | `v34_oof.csv` | `v34_sub.csv` | **Ruler, not a claim: V29 reproduced at `RANDOM_SEED=2027`, the four Stage-1 keys being the only content difference from V33.** Completes {V18, V29} × {42, 2027}. **V29's keys measured a second time and the sign flipped: −0.000007 pooled at seed 42, +0.000010 at seed 2027** (folds 2/5 then 3/5) — a swing of 0.000017, a third of the ruler, so the keys' CV effect is unmeasurable rather than merely small. **Its LB−CV offset reproduced exactly (−0.000180 twice)** ✅ |
| V33 | 10-17 | 0.96100 | 0.96123 | -0.00023 | 17.8 min | `S6E10_V33_V18SeedRepro.py` | `v33_oof.csv` | `v33_sub.csv` | **The most valuable file in the campaign, and it moves nothing: V18 byte-for-byte at `RANDOM_SEED=2027`.** Same model, same frame, same budget, different split shuffles: **pooled CV −0.000057, fold mean −0.000056, LB −0.000060.** That is the resolution floor of both yardsticks, measured on the control itself; per-fold differences run ±0.00056, because a seed reshuffles folds, so fold-level comparisons across seeds mean nothing. Claims can now be priced: V12's +0.00016 = 2.8 ruler units, V11/V18's +0.00012 = 2.1, V27's and V32's +0.00002 = 0.4, V29's +0.00004 LB = 0.7 ✅ |
| V35 | 10-17 | 0.96058 | 0.96101 | -0.00043 | 5.9 min | `S6E10_V35_GradientRowSampling.py` | `v35_oof.csv` | `v35_sub.csv` | **The plan's highest-expected-value arm is a unanimous loss.** XGB `sampling_method='gradient_based'` (the MVS family) on V11 — one parameter, `subsample` 0.9 held: **CV −0.000123 pooled, folds improved 0/5 (every delta negative, −0.00008…−0.00013), LB −0.00006.** Plumbing is proven, not assumed: the in-run `booster params` printed `sampling_method='gradient_based'` and the constructor round-tripped it at xgboost 3.2.0. So the published **+0.12% ROC-AUC on 97 of 153 datasets (Ibragimov et al., KDD 2019) does not transfer** to a 700k-row frame already at 0.961. No speed either (5.9 vs 5.8 min), and the best-iteration spread widened to 370 rounds (372–742): concentrating sampling on confident rows made the stopping point *less* stable. **Closed** ❌ |
| V31 | 10-17 | 0.95907 | 0.95930 | -0.00023 | 25.6 min | `S6E10_V31_ConfidentRowWeights.py` | `v31_oof.csv` | `v31_sub.csv` | **Row-noise selection closes with −0.00199 — twelve times the largest channel ever found here — and the pre-declared falsifier came back intact.** Confident-row exclusion on V18: each fold's training rows ranked by |p−y| from an in-run nested XGB (V30's proven 5-inner-fold pass), top **q=0.0387** removed (21,661 rows/fold), validation block untouched. **Folds 0/5, deltas −0.00172…−0.00215, uniform; CV −0.001987 and LB −0.00199, the two yardsticks agreeing to 0.00001 — the offset law in action.** The mechanism is on the fold lines: dropped rows carried mean |p−y| **0.9473** vs kept **0.0866**, so the arm deleted the rows where data most contradicts a strong model. AUC is scored against the *observed* labels, so those rows were the ranking's cheapest credit: **3.87% of the rows proved worth ~50× their share**, a third independent measurement that the labels are near-clean (after K1b M2(a)=0 duplicates and M2(c)'s ρ̂ = −0.2149) ❌ |

| V29 | 10-17 | **0.96110** | 0.96128 | -0.00018 | 17.6 min | `S6E10_V29_PortStage1KeysToNet.py` | `v29_oof.csv` | `v29_sub.csv` | **Highest LB in the file, and not a record change.** V18's frame + the four Stage-1 keys (301 → 386 expanded cols, +12 per-fold, no twins): **CV −0.000007 vs V18 (0.96128 pooled, fold mean 0.96129), folds improved 2/5 with one tie → gate FAILS on magnitude and fold consistency.** Its +0.00004 LB is **0.23σ** of the net family's board scatter (0.000174) and smaller than the 0.00007 measured on a byte-identical re-submission, so **V18 remains the standing control**. Third two-architecture test and the third transfer: keys −0.00005/−0.00004 on the tree (V23/V24) → 0.00000 on the net, so the "the net would use what the tree ignored" argument is now dead on measured ground ⚠️ |
| V18 | 10-11 | **0.96106** | 0.96129 | -0.00023 | 17.6 min | `S6E10_V18_RealMLPFusionEV.py` | `v18_oof.csv` | `v18_sub.csv` | **NEW BEST SINGLE — supersedes V12.** V11's 13 label-free `aux_ev_*` columns on V12's byte-identical twin+logit frame at `n_epochs=3` (original-fitted predictors, 115 s build, no EV twins): CV **+0.00012** vs parent V12 with **folds improved 5/5** (+0.00007…+0.00019), +0.00028 CV / **+0.00055 LB** vs V2 — outside the 0.0003 tie band, and above the best published single (0.96105). The EV block's +0.00012 is the same on trees and on the net ⇒ family-independent channel 🏆 (DeLong z + `SEED_REPRO=2027` still pending) |
| V28 | 10-16 | 0.96105 | 0.96128 | -0.00023 | 17.4 min | `S6E10_V28_EVTargetsFullSet.py` | `v28_oof.csv` | `v28_sub.csv` | The 7 `aux_p_<cat>_<level>` posteriors beside V18's 13 EVs (parent **V18**, recipe and budget unchanged, 296 frame cols → 381 expanded): **CV −0.00001 vs V18 with folds improved 2/5 → gate FAILS on Δ**, LB −0.00001 = tie band. Fold array [0.96144, 0.96041, 0.96210, 0.96136, 0.96112], pooled OOF **equal to its own fold mean** and Gap −0.00023 identical to the parent's, so nothing about V18's calibration moved ⚠️ **2nd-highest LB in the file, and a genuine null: the nominal-posterior channel is inert on the net (−0.00001) exactly as on the tree (V25 −0.00004) — closed on both families** |
| V12 | 10-11 | 0.96095 | 0.96117 | -0.00022 | 15.2 min | `S6E10_V12_NetTwinTeacher.py` | `v12_oof.csv` | `v12_sub.csv` | **Best single of the episode — supersedes V9's same-day LB claim above.** Net + twin frame (17 numeric→categorical twins, `_og_teacher` as logit; `REALMLP_PARAMS` byte-identical to V7): CV +0.00016 vs V2 with **folds improved 5/5**, +0.00028 vs parent V7, LB +0.00044 = the only arm outside the 0.0003 tie band. Best epoch 3 (the budget) in 5/5 ⇒ still epoch-starved, ceiling is above this ✅ (two lines below: V18 superseded the score, V17 disproved the starvation reading) |
| V17 | 10-11 | 0.96085 | 0.96092 | -0.00007 | 86.4 min | `S6E10_V17_RealMLPEpochLadder.py` | `v17_oof.csv` | `v17_sub.csv` | `n_epochs` ladder 6/8/10, one model per rung, rungs never combined; rung of record 10. **Every rung is below V12's 3-epoch budget** (0.96089 / 0.96083 / 0.96092 vs 0.96117 = **−0.00025**, folds improved 1/5), all 15 fits pinned at their own ceiling ⇒ with ES off, `best_epoch = budget` is an artifact, not starvation ⇒ **the net's epoch axis is closed at 3** ❌ |
| V27 | 10-16 | 0.96076 | 0.96115 | -0.00039 | 6.3 min | `S6E10_V27_DuoTeacherOrig.py` | `v27_oof.csv` | `v27_sub.csv` | Duo original teacher (`_og_teacher` = sigmoid(logit-mean) of XGB + RealMLP_TD, both orig-only) on V11's frame: **CV +0.00002 vs V11 with folds improved 4/5** — the only Stage-1 arm ahead of its parent on CV *and* LB (+0.00012), yet **below the Δ>+0.00005 bar → gate FAILS on magnitude.** Spearman(xgb logit, net logit) **+0.95990** against the 0.990 collapse limit, so the diversity premise held and the *teacher* AUCs on 699,635 never-fitted rows rank xgb 0.95508 > logit-mean 0.95477 > net 0.95396: averaging two weaker members lands between them and buys 0.00002 ⚠️ |
| V9 | 10-11 | 0.96067 | 0.96103 | -0.00036 | 3.8 min | `S6E10_V9_TErowStats.py` | `v9_oof.csv` | `v9_sub.csv` | TE row-stats (parent V2, axis = 5 `_tmr_*` columns): CV +0.00002 / LB +0.00016 vs V2, folds improved 2/5 → gate FAILS on fold consistency; best LB of the episode but inside the 0.0003 tie band ⚠️ |
| V24 | 10-16 | 0.96066 | 0.96109 | -0.00043 | 6.0 min | `S6E10_V24_RatedNAndWifiBandKeys.py` | `v24_oof.csv` | `v24_sub.csv` | +2 TE keys on V11 (`n_unrated` = the shipped `_count_0`, `pair_wifi_band__online_boarding`): **CV −0.00004 vs V11, folds improved 1/5, best cells 24/24 with 0 thin cells → gate FAILS on Δ.** The K1 z² screen rated these keys 3.6983/3.8548, so **a high z² on named cells does not buy CV** ❌ |
| V3 | 10-02 | 0.96065 | 0.96096 | -0.00045 | 4.2 min | `S6E10_V3_DuoTeacher.py` | `v3_oof.csv` | `v3_sub.csv` | Duo teacher tie: tree+tree diversity on the same orig data adds nothing; goodpjw's two-teacher gain was tree+MLP. DeLong vs v2 pending v3_oof archive |
| V10 | 10-11 | 0.96065 | 0.96103 | -0.00038 | 3.7 min | `S6E10_V10_OrigDistanceFeatures.py` | `v10_oof.csv` | `v10_sub.csv` | 9 original-distribution columns (parent V2): CV +0.00002 / LB +0.00014, folds improved 3/5 → gate FAILS; `_od_*` absent from fold-1 top-15 importances → the axis is near-unused by the model ⚠️ |
| V11 | 10-11 | 0.96064 | 0.96113 | -0.00049 | 5.8 min | `S6E10_V11_AuxEVFeatures.py` | `v11_oof.csv` | `v11_sub.csv` | 13 label-free aux-EV columns on the V2 frame (original-fitted predictors, 116 s build): **CV +0.00012 with folds improved 5/5** → clears Δ>+0.00005 and the fold criterion, the first tree arm to do so; LB +0.00013 is inside the tie band and CV sits −0.00004 under V12 → strongest single TREE, awaiting paired DeLong z + seed-2027 ✅ |
| V26 | 10-16 | 0.96064 | 0.96113 | -0.00049 | 14.6 min (3 rungs) | `S6E10_V26_TeacherCapacityLadder.py` | `v26_oof.csv` | `v26_sub.csv` | Teacher capacity ladder (8,600) → (10,1000) → (12,1500), rung of record = the control rung (8,600): **CV 0.96113 / 0.96103 / 0.96101 with deeper rungs at −0.000108 and −0.000125 and 0/5 folds improved**, and the teacher's own 80/20 original hold-out AUC **falling** 0.99510 → 0.99486 → 0.99466 ⇒ more teacher capacity is worse, on the teacher and on the arm ❌ **The teacher-capacity axis closes.** Control-rung folds are V11's five AUCs digit-for-digit, so the ladder is bit-reproducible at fixed seed and the rung deltas are real, not noise |
| V25 | 10-16 | 0.96064 | 0.96109 | -0.00045 | 6.0 min | `S6E10_V25_CategoricalEVOnTree.py` | `v25_oof.csv` | `v25_sub.csv` | +7 label-free `aux_p_<cat>_<level>` posteriors (Class 3 + Type of Travel 2 + Customer Type 2) on V11's frame: **CV −0.00004 vs V11, folds improved 1/5 → gate FAILS.** The nominal channel V12 exploited on the net does not transfer to the tree, where the same information is already splittable ❌. Two-class targets need `objective='binary:logistic'` — `multi:softprob` dies at the sklearn wrapper with `num_class = 0` (measured, cost two runs) |
| V22 | 10-16 | 0.96063 | 0.96113 | -0.00050 | 6.0 min | `S6E10_V22_SignCorrectedAggregates.py` | `v22_oof.csv` | `v22_sub.csv` | The 9 rating-matrix aggregates rebuilt on the sign-corrected scale (`6−r`, 0 kept as the unrated marker) on V11's frame: **CV 0.00000 vs V11, folds improved 1/5 → gate FAILS.** K1 measured the reversed scale at composite `_mean_excl0` 0.85658 (+0.04393 over the shipped orientation) — a real monotone fix with **zero ranking consequence on a tree that can re-split every rating** ❌ |
| V23 | 10-16 | 0.96063 | 0.96108 | -0.00045 | 5.7 min | `S6E10_V23_AgeBandKeys.py` | `v23_oof.csv` | `v23_sub.csv` | +2 age-crossing TE keys (`pair_class__age5`, `pair_inflight_wifi_service__age5`) on V11's frame: **CV −0.00005 vs V11, folds improved 1/5 → gate FAILS**, despite 7 and 13 thin cells (n<50, ≤0.0003 of fit rows) and the highest z² in the whole K1 screen (4.6649) ❌ |
| V15 | 10-11 | 0.96061 | 0.96105 | -0.00044 | 28.2 min (CPU) | `S6E10_V15_LightGBMSingle.py` | `v15_oof.csv` | `v15_sub.csv` | LightGBM family swap on V2's frozen frame (`num_leaves 63`, `lr 0.02`, CPU-only, Kaggle build **4.6.0** not the pinned 4.7.0): CV +0.00004 vs V2 with folds improved **4/5** → **fails the Δ>+0.00005 magnitude bar by 0.00001**; LB +0.00010 = tie band. ES genuinely fired (best iters 1068/1012/838/669/833) ⚠️ |
| V30 | 10-17 | 0.96060 | 0.96110 | -0.00050 | 39.7 min (3 rungs) | `S6E10_V30_SoftLabelLadder.py` | `v30_oof.csv` | `v30_sub.csv` | Soft-label ladder (α, es) = (.20,150) control → (.20,400) → (.70,400), rung of record = the control rung: **fold means 0.96111 / 0.96111 / 0.96111 — a spread of 0.000003 across the whole α range, and the pre-declared falsifier fired on every fold of every rung** (max Spearman(student, in-run nested teacher) 0.9928–0.9973 vs limit 0.99 ⇒ COLLAPSE). The teacher's own fold mean is 0.96100 against the student's 0.96111, so the "teacher" is within 0.00011 of the model it was supposed to teach ⇒ **target mixing carries no information this dataset does not already have — axis closed, and α=0.7 needed 1,387–3,251 rounds to buy nothing** ❌. Control rung reproduced V20's recovered CV (0.961104) to +0.000008 and its LB exactly (0.96060), which independently validates that recovery |
| V13 | 10-11 | 0.96058 | 0.96101 | -0.00043 | 3.7 min | `S6E10_V13_OrgMeanShrink.py` | `v13_oof.csv` | `v13_sub.csv` | Bayesian shrinkage of `_org_mean` (m=30, b recomputed 0.434379): removes every saturated cell (train 5,999→0, test 2,628→0, 649/1,513 cells moved) and CV is **exactly V2's 0.96101** (folds 3/5) → defect was real, ranking value zero ❌ |
| V21 | 10-16 | 0.96058 | 0.96112 | -0.00054 | 5.6 min | `S6E10_V21_AggregateAblation.py` | `v21_oof.csv` | `v21_sub.csv` | The 10 rating aggregates **not built at all** (guard: 0 of 10 present, 142-column base frame vs V11's 152) on V11's frame: **CV −0.00001 vs V11, folds improved 3/5 → gate FAILS.** Deleting the whole block costs 0.00001, so the aggregates V2 shipped have been inert since the EV block arrived; combined with V22 this closes the aggregate axis in both directions (absent ≡ present ≡ sign-corrected) ❌ |
| V19 | 10-11 | 0.96057 | 0.96103 | -0.00046 | 26.1 min (CPU) | `S6E10_V19_LGBMMinChildSamples.py` | `v19_oof.csv` | `v19_sub.csv` | V15 with one knob, `min_child_samples` 20→**40** (Kaggle LightGBM **4.6.0**, PIN WARNING vs pinned 4.7.0 recorded again): **CV −0.000024 vs V15**, folds improved 4/5 but by ≈+0.00001 each (fold 1 −0.00015); best iters moved up to 329 trees (1068→739) with an unchanged importance profile ⇒ the leaf-count knob is **inert for ranking** ❌ |
| V13 | 10-11 | 0.96051 | 0.96101 | -0.00050 | 3.7 min | `S6E10_V13_OrgMeanShrink.py` | `v13_oof.csv` | `v13_sub.csv` | **Re-submission of the identical `v13_sub.csv`** (same CV, same OOF): LB 0.96058 → 0.96051 = a **0.00007 swing on a byte-identical file**. Direct in-house measurement of LB noise; any LB delta below ~0.0003 is unresolvable ⚠️ |
| V2 | 10-02 | 0.96051 | 0.96101 | -0.00050 | 3.8 min | `S6E10_V2_XGB_WideProfile.py` | `v2_oof.csv` | `v2_sub.csv` | CV +0.00040 vs V1 (5/5 folds improved); paired DeLong vs v1_oof: z=+7.44 ACCEPTED — current control |
| V4 | 10-02 | 0.96051 | 0.96088 | -0.00045 | 7.9 min | `S6E10_V4_CatBoostMember.py` | `v4_oof.csv` | `v4_sub.csv` | CatBoost member (V2 package, model swapped, Logloss ES): solo tie-slight-loss; 2-member blend +0.000113 → stack ingredient ✅ |
| V7 | 10-11 | 0.96049 | 0.96089 | -0.00040 | 13.8 min | `S6E10_V7_RealMLPRecipe.py` | `v7_oof.csv` | `v7_sub.csv` | RealMLP proven recipe (flat_anneal / 3 ep / ES off): CV +0.00025 vs parent V6 but −0.00012 vs V2, folds improved 0/5; best epoch = 3 (budget) in 5/5 → ship bar CV ≥ 0.9614 not met ❌ |
| V16 | 10-11 | 0.96048 | 0.96089 | -0.00041 | 63.0 min | `S6E10_V16_CatBoostCrosses.py` | `v16_oof.csv` | `v16_sub.csv` | 3-rung CatBoost ladder in one run: `none` (V4's frame under `eval_metric='AUC'`, in-run control) 0.96089, `add` (+39 rating×context native crosses) 0.96095, `swap` (crosses in, 54 manual `_tm_pair_*` out) 0.96096. **Metric delta +0.00001 vs V4; both cross rungs beat `none` on only 3/5 folds ⇒ `none` is the run of record and both halves are rejected.** Cause printed by the model itself: `Default metric period is 5 because AUC is/are not implemented for GPU` (the kwarg is accepted, the metric runs on CPU), and **no `_x_*` column reaches fold-1's top-15** ❌ |
| V6 | 10-08 | 0.96045 | 0.96064 | -0.00019 | 22.6 min | `S6E10_V6_RealMLPBagged.py` | `v6_oof.csv` | `v6_sub.csv` | CatBoost member (V2 package, model swapped, Logloss ES): solo tie-slight-loss; 2-member blend +0.000113 → stack ingredient ✅ | RealMLP bagged (n_ens=8): loses to V2 by 0.00037 CV but gains +0.000886 over V5 (5/5 folds) → preferred third leg ⚠️ |
| V8 | 10-11 | 0.96042 | 0.96099 | -0.00057 | 11.1 min | `S6E10_V8_XGBLeafWise.py` | `v8_oof.csv` | `v8_sub.csv` | Leaf-wise ladder rung-of-record max_leaves=32 (`grow_policy='lossguide'`, max_depth 6→0): CV −0.00002 / LB −0.00009 vs V2, folds improved 1/5; ladder 8→16→32 = 0.96089→0.96097→0.96099 (+0.00010 across it) ❌ |
| V5 | 10-07 | 0.96032 | 0.95975 | +0.00057 | 15.3 min | `S6E10_V5_RealMLPSingle.py` | `v5_oof.csv` | `v5_sub.csv` | Architecture axis: RealMLP_TD (PyTabKit 1.7.3, n_ens=1, n_epochs=6, lr_sched=flat_anneal) on V2's exact 259-col matrix. CV −0.00126 vs V2 → rejected; ES never fired (best epoch 6 in 5/5 folds) so the net is undertrained, not disproved ❌ |
| V1 | 10-02 | 0.96019 | 0.96061 | -0.00042 | 26.7 min | `S6E10_V1_XGB_Baseline.py` | `v1_oof.csv` | `v1_sub.csv` | Official baseline; dev-run with house params+blanket TE scored 0.96009 CV (not a version). CV-LB gap -0.00042 inside expected band |

---

## Appendix — Gap re-derivation, 2026-10-12 (`LEDGER`: arithmetic over this file's own columns)

Rule 2 forbids editing the rows above, so the corrections live here and the rows stay exactly as written.

The table's sign convention is **Gap = LB − OOF** (V18: 0.96106 − 0.96129 = −0.00023 ✓; V5: 0.96032 − 0.95975
= +0.00057 ✓). Recomputing all 24 version rows (19 Single Models + the 5 Top-5 view copies) on that basis:
**21 match to the printed precision, three do not.**

| Row | Gap recorded | Gap recomputed | Arithmetic |
| --- | --- | --- | --- |
| V3 (both tables) | −0.00045 | **−0.00031** | 0.96065 − 0.96096 |
| V4 (Single Models) | −0.00045 | **−0.00037** | 0.96051 − 0.96088 |

V3 and V4 carry the *identical* recorded value while their LB/OOF pairs differ, which is the signature of a
copied cell rather than two independent slips. Neither number was an endpoint of any band quoted elsewhere —
the CV→LB offset range used in planning (−0.00057…+0.00057 across the ladder, GBDTs ≈ −0.00043, nets ≈ −0.00009)
is unchanged — but any per-version claim about V3's or V4's gap should read the recomputed value.

One non-numeric defect in the same table: the **V6 row has eleven cells where the header has ten** — V4's Notes
text is pasted into its Notes column and V6's real note follows an extra `|`. Read the second half of that row
as V6's note; its scores, files and timing are unaffected.

## Appendix — the standing control moves V2 → V18, 2026-10-13 (`LEDGER`: paired DeLong over the archived OOF vectors)

**Why this is admissible now:** law 9 needs CV *and* a confirmed LB on both sides. V18 has CV 0.961287 and LB
0.96106; V2 has 0.961009 / 0.96051. Both vectors are archived (`v18_oof.csv`, `v2_oof.csv`), so the gate's
first half can be computed — and was, on 2026-10-12, by the K1 battery.

| Pair | ΔAUC | DeLong SE | z | logit corr | folds improved | gate |
| --- | --- | --- | --- | --- | --- | --- |
| V18 vs V12 | **+0.000115** | 2.356e-05 | **+4.87** | 0.99942 | 5/5 | **ACCEPT** |
| V18 vs V2 | **+0.000278** | 5.43e-05 | **+5.12** | 0.99668 | 5/5 | **ACCEPT** |
| V12 vs V2 | +0.000163 | 5.337e-05 | +3.06 | 0.99676 | 5/5 | ACCEPT (the recorded anchor) |
| V11 vs V2 | +0.000124 | 3.487e-05 | +3.56 | 0.99890 | 5/5 | ACCEPT |
| V20 vs V11 | −0.000030 | 2.805e-05 | −1.06 | 0.99931 | 1/5 | reject |
| V20 vs V2 | +0.000094 | 3.376e-05 | **+2.80 < 3** | 0.99900 | 5/5 | **tie / reject** |

Consequences, stated as the arithmetic states them:

1. **V18 is the standing control and the bar is 0.961287.** A tree arm parented on V11 (0.961133) now needs
   **+0.000154** over its own parent merely to equal the control — larger than any single-channel tree move in
   20 versions. That is why the FE arms were prioritised where a residual was *measured*, not guessed.
2. **V20 closes as configured, not as a mechanism.** It improved 5/5 folds against V2 and still failed z, and
   it is 0.000030 *below* its own parent V11 — the α=0.20 dose inside a 150-round early-stopping window. V30
   re-opens exactly that confound as a three-rung ladder with V20's configuration as its control rung.
3. **Two provenance caveats, and they are load-bearing.** (a) The battery's `N_BOOT=120` paired bootstrap
   **executed 0 reps on every pair** (`mean +nan SE nan sign-flips 0/0`), so every z above rests on DeLong's
   analytic variance with no independent cross-check. (b) The gate's *second* half — reproduction at
   `SEED_REPRO=2027` — has not run; **`S6E10_V33_V18SeedRepro.py` is the file that owes it**, and until it
   returns, "V18 is control" is a first-half acceptance only. A v2 plan text also claimed bootstrap
   corroboration for these pairs; that measurement never happened and is not repeated here.
4. **Recovered CVs, labelled as recoveries.** V20 **0.961104** (never captured at run time; this is the pooled
   AUC of its archived vector). **`v14_oof.csv` is bit-identical to `v2_oof.csv`** (same five fold AUCs,
   residual correlation 1.0000), so V14's "0.961009" is a duplicated archive file, not a result — V14's
   `max_bin` axis remains **unmeasured** and its row above stays closure-only.
5. Nothing else in this file changes: the rows stay as written (rule 2), and no new version row is added —
   V21–V33 have neither CV nor LB, so they cannot have one.

## Appendix — the ruler lands, and the board turns out to be CV translated, 2026-10-17 (late) (`LEDGER`: arithmetic over this file's own columns plus the five captured CV blocks)

| arm | frame | seed | pooled CV | fold mean | LB | LB − CV |
|---|---|---|---|---|---|---|
| V18 | V18 frame | 42 | 0.961287 | 0.961292 | 0.96106 | **−0.000227** |
| V33 | V18 frame | 2027 | 0.961230 | 0.961236 | 0.96100 | **−0.000230** |
| V32 | V18 frame | 42 | 0.961310 | 0.961314 | 0.96108 | **−0.000230** |
| V31 | V18 frame | 42 | 0.959300 | 0.959316 | 0.95907 | **−0.000230** |
| V29 | V29 frame | 42 | 0.961280 | 0.961286 | 0.96110 | **−0.000180** |
| V34 | V29 frame | 2027 | 0.961240 | 0.961250 | 0.96106 | **−0.000180** |
| V11 | V11 frame | 42 | 0.961133 | 0.961138 | 0.96064 | −0.000493 |
| V35 | V11 frame | 42 | 0.961010 | 0.961024 | 0.96058 | −0.000430 |

1. **The resolution floor is now measured on the control itself.** V33 is V18 byte-for-byte at
   `RANDOM_SEED=2027`: pooled CV **−0.000057**, fold mean **−0.000056**, LB **−0.000060**. V34 is V29 at 2027:
   pooled **−0.000040**, fold mean **−0.000040**, LB **−0.000040**. So changing *nothing but the shuffle* moves
   both yardsticks by ~0.00005, and it moves them **together** (the CV delta and the LB delta agree to 0.000003 on
   V18 and to 0.000000 on V29). Fold-level comparisons across seeds are meaningless — the per-fold difference has an
   sd of 0.00056/0.00059, as large as the fold spread itself, because the seed reshuffles which rows are in which
   fold, not which model is better.
2. **The gate's magnitude bar sits *at* the noise floor, not above it.** Δ > +0.00005 was written before a ruler
   existed; the ruler says a byte-identical re-run can differ by 0.000057. Priced in ruler units: V12's +0.00016 =
   2.8, V11's and V18's +0.00012 = 2.1, V27's +0.00002 = 0.35, **V32's +0.000023 = 0.4**, V29's +0.00004 LB = 0.7.
   Only three arms in the whole campaign exceed two ruler units, and all three are the frame channels (twins, EVs,
   EVs-on-net). `PLAN_V31_ONWARD.md` §7 now proposes re-basing the bar at +0.0001 with a seed-2027 reproduction.
3. **LB is CV minus a per-frame constant — the board adds no information.** Four arms on V18's frame give
   −0.000227/−0.000230/−0.000230/−0.000230 (a 0.000003 spread, and the 3e-6 outlier is V18's own 6-digit CV
   rounding), and the two V29-frame arms give −0.000180 both. Even V31's −0.00199 damage appeared as −0.001987 on CV
   and −0.00199 on LB. This **revises the recorded net-family board scatter of 0.000174**: that figure pooled arms
   across different frames and epoch budgets. Hold the frame and the budget, and the residual is ~0.00000 while the
   *seed* moves the pair by 0.00005. So "why did LB not improve?" has a one-line answer: **the board is a
   translation of CV, and CV did not improve.**
4. **V29's four keys flip sign between seeds, so their effect is unmeasurable rather than merely small.**
   −0.000007 pooled at seed 42 (folds 2/5) → **+0.000010** at seed 2027 (folds 3/5). A 0.000017 swing on a channel
   whose entire LB advantage was +0.00004 — i.e. V29's standing as "highest LB in the file" is a
   frame-offset-and-seed artifact, not a model gain. V18 remains the standing control.
5. **V32 is the most consistent non-gain ever recorded.** All five folds improved, every delta in
   [+0.00001, +0.00003] (a 5/5 sign test alone would be p = 1/32), pooled +0.000023, LB +0.00002 — directionally
   real, materially nothing. **Bagging is saturated**: `n_ens` 1→8 bought +0.000886, 8→16 buys +0.000023 for double
   the fit time. The arm's pre-declared calibration test also fails to fire, and that is a finding in itself:
   `pooled − fold mean` is −0.000007 at n_ens=16 vs −0.000005 at n_ens=8, so there was no bagging/epoch offset
   hiding in the pooled number on this frame at 3 epochs.
6. **V35 falsifies the plan's biggest external hope.** MVS/`gradient_based` row sampling — published at +0.12%
   ROC-AUC first on 97 of 153 datasets — landed **−0.000123 pooled with 0/5 folds improved** and no speed gain
   (5.9 vs 5.8 min). The parameter is proven to have reached the booster, so this is a mechanism null, not a
   plumbing failure. Its fold-1 importances also explain why feature arms cannot move this frame: `_og_teacher`
   0.4177 + `_tm_pair_online_boarding__type_of_travel` 0.2199 = **64% of the gain in two channels**, and the
   teacher channel is the one V2/V11/V18 have all carried.
7. **V31 closes row-noise selection with a −0.00199, and the mechanism is on the record.** The deleted slice carried
   mean |p−y| 0.9473 against 0.0866 for the kept rows: the arm removed the 3.87% of rows where the data most
   contradicts a strong model, and the metric — which is scored against the **observed** labels — lost 0.002. Those
   rows proved worth ~50× their share, which is a third measurement (after K1b M2(a)=0 duplicates and M2(c)'s
   negative ρ̂) that the labels are near-clean. §1(a)'s invariance argument stands.
8. **What is still unpaid.** Paired DeLong z for V31–V35 was not computed offline, and after today it matters
   less than it did: DeLong prices the *metric's* sampling noise on 140k validation rows, which is why it reports
   z ≈ 4–5 for a +0.00012 CV change, while the *training* stochasticity the ruler measures is 0.00005 — the same
   order. The ruler, not the z, is now the binding constraint, and all five arms fail on magnitude regardless.

## Appendix — V29 and V30 land, the ladder is scrapped, and what "best LB" now means, 2026-10-17 (`LEDGER`)

1. **V29 is the highest LB in this file (0.96110) and does not become the control.** Against its parent V18 it is
   **CV −0.000007** (pooled 0.96128 vs 0.961287; fold means equal at 0.96129) with **folds improved 2/5 and one
   tie**, so it fails the magnitude bar and the fold bar; the +0.00004 LB difference is **0.23σ** of the net
   family's board scatter (residual sd 0.000174, n=7) and smaller than the 0.00007 that a byte-identical
   re-submission of `v13_sub.csv` moved (row above). Reading that +0.00004 as an improvement would be reading
   noise, so **V18 (CV 0.961287 / LB 0.96106) remains the standing control** and V29 is recorded as its third
   two-architecture tie.
2. **Three for three: every channel we tested on both architectures transferred, in sign and in size.** The EVs
   (+0.00012 tree / +0.00012 net), the categorical posteriors (−0.00004 / −0.00001), the Stage-1 keys
   (−0.00005, −0.00004 / 0.00000). This retires the standing excuse for re-running a tree result on the net, and
   it is why the ladder is scrapped rather than extended: on this dataset **architecture is not the variable that
   decides whether a feature channel pays**.
3. **V30 closes target mixing outright.** All three rungs produced the same fold mean (0.96111) and every fold of
   every rung tripped the pre-declared collapse guard (Spearman student–teacher 0.9928–0.9973 against a 0.99
   limit). Its control rung also **independently confirmed V20's recovered CV** — 0.961104 recovered from the
   archived vector, 0.96110 printed by a fresh run of the same configuration, and LB identical at 0.96060 — the
   first time a "recovered" number in this file has been re-measured rather than trusted.
4. **Scrapped before running, by the owner on 2026-10-17:** `V31_RegionAlpha` (its parent channel — α mixing —
   just measured zero across a 3.5× α range, so the arm had no premise left) and `V32_NetBaggingCapacity` (70 GPU-min
   for a three-rung ladder). Both scripts move to `Previously trained files/Archieve/` marked RETIRED, neither is
   deleted. **One correction on the record's own behalf:** scrapping V32 on cost was reasonable, but the reason I
   first wrote — "a family whose whole CV spread is 0.00003" — overstates the evidence. Bagging is the one knob this
   campaign has ever measured a large effect on: **V5 (`n_ens=1`) → V6 (`n_ens=8`) was +0.000886 pooled / +0.000598 on
   the fold-mean basis**. Its premise is unfalsified, so `PLAN_V31_ONWARD.md` re-proposes it as a **single rung**
   (`n_ens` 8→16, ~35 min) rather than a ladder. **V33 keeps its occupant** (V18 at `RANDOM_SEED=2027`, authored and verified, still unrun) and the new plan adds **V34 = V29 at 2027**, so the grid {V18, V29} x {42, 2027} can finally say whether 0.96106 vs 0.96110 is a model difference or the same model under another seed. **V31 and V32 are reused by new arms** (row-noise weights; one bagging rung) because their retired occupants never ran and own no ledger row.


## Appendix — Stage 1 lands, 2026-10-16 (`LEDGER`: Kaggle runs of record, LB confirmed by the owner)


The last clause above is now superseded by measurement, so it is answered here rather than edited there.

1. **Rows added for V21–V27** (CV *and* confirmed LB both exist for each). All six Stage-1 feature arms are
   null-or-negative against their parent V11 and V27 is +0.00002, so **no Stage-1 arm passed its own gate** —
   none reaches Δ>+0.00005, which is why no paired-DeLong z was needed to close them.2. **V28 initially got no row — and then it earned one the same day.** When this appendix was first written its
   LB (0.96105) was reported but the run's CV block had never been captured, so law 9 barred the row. **The CV
   block arrived later on 2026-10-16**: OOF **0.96128**, fold mean 0.96128 ± 0.00055, folds
   [0.96144, 0.96041, 0.96210, 0.96136, 0.96112], best epochs [3,3,3,3,3], 17.4 min. Against its parent V18
   that is **ΔCV −0.00001 with folds improved 2/5 and ΔLB −0.00001**, so it **fails the gate on magnitude and is
   recorded as a null, not a loss** — its own pooled OOF equals its fold mean, so the reading does not depend on
   which yardstick is used. V28's row now sits in Single Models and, at 0.96105, in the Top-5 view.
3. **Top 5 is a view, so it moved twice today.** V27 (0.96076) entered as the fourth-highest and V3 (0.96065)
   left; then V28 (0.96105) entered as the second-highest and V9 (0.96067) left. Neither V3's nor V9's Single
   Models row was touched (rule 2), and the gap appendix above is untouched too — its "V3 (both tables)" label
   describes the state as at 2026-10-12, which is what it measured.

## Appendix — V36/V37 land, and the offset constant turns out to be protocol-dependent, 2026-10-23 (`LEDGER`)

Arithmetic over this file's own columns (Gap = LB − OOF), with the CV deltas measured against the parent's recorded
OOF and the ruler from V33/V34 (±0.00006).

| arm | OOF | LB | Gap | ΔCV vs V18 | ΔLB vs V18 | read |
| --- | --- | -- | --- | --- | --- | --- |
| V36 folds 5→10 | 0.96139 | 0.96104 | −0.00035 | **+0.00010** | −0.00002 | CV gain at the gate bar, LB inside the parent's band 0.96100–0.96106 → tie |
| V37 σ=8 (record) | 0.96128 | 0.96107 | −0.00021 | −0.00001 | +0.00001 | tie on both yardsticks |
| V37 σ=32 | 0.96121 | — | — | −0.00008 | — | one rung worse than the other by 1 ruler, paired per fold |

**Correction to a recorded law, not an extension of it.** The offset law said `LB = CV − a per-frame constant`
(−0.00023 on the V18 frame, −0.00018 on the V29 frame, −0.00043 on the tree frame), reproduced to 0.000003 **across
seeds**. V36 changed the *protocol*, not the frame, and its constant is **−0.00035**. So the constant belongs to the
(frame, protocol) pair: a fold-count arm's CV is not on the same scale as the arms it is compared against, and the
"CV decides, LB is predicted" rule must be applied only within one protocol. This is why V36's +0.00010 — the largest
CV gain of the campaign — is reported as a tie and not as a candidate.

**Top-5 view recomputed** (a view, not a copy — every row also lives in its category table): V29 0.96110, V32
0.96108, **V37 0.96107**, V18 0.96106, V34 0.96106. V28 (0.96105) leaves the view and stays in Single Models. V36
(0.96104) does not enter. No record change: **V29 remains the highest LB, V18 remains the standing control.**

## Appendix — V38–V42 land in one batch, the queue is empty, and the offset constant breaks on a prior-dominated vector, 2026-10-24 (`LEDGER`)

Five arms, one sitting, none adopted. Gap = LB − CV, recomputed from this file's own columns.

| arm | parent | parent CV | arm CV | ΔCV | LB | LB − CV | verdict |
|---|---|---|---|---|---|---|---|
| V38 k-NN column | V18 | 0.96129 | 0.96125 | −0.00004 | 0.96100 | −0.00025 | tie-to-negative |
| V39 `num_emb_type='plr'` | V18 | 0.96129 | 0.96120 | −0.00009 | 0.96097 | −0.00023 | negative |
| V40 prior as `base_margin` | V11 | 0.96113 | 0.95319 | **−0.00794** | 0.95196 | **−0.00123** | mechanism finding |
| V41 net frame minus 37 cols | V18 | 0.96129 | 0.96127 | −0.00002 | 0.96104 | −0.00023 | tie |
| V42 CatBoost MVS | V4 | 0.96088 | 0.95990 | −0.00098 | 0.95965 | −0.00025 | negative, 23-ruler paired |

1. **The standing control is untouched.** Best LB is still V29 0.96110; the V18 frame's band stays 0.96100–0.96106
   and V41's 0.96104 sits inside it. Top-5 needed no change: the five highest LBs are V29 0.96110, V32 0.96108,
   V37 0.96107, V18 0.96106, V34 0.96106 — nothing in this batch reached it.
2. **The −0.00023 constant is a property of a *model*, not only of a frame.** Four arms on the V18 frame reproduce
   it to 0.00002. V40 does not (−0.00123), and the reason is structural: with ES unable to see the offset, its
   boosters stopped at iteration 0 and the shipped vector *is* the original-data teacher's logit, whose own regime
   mismatch (fitted on real rows, scored on synthetic) is what K1b M2(c) measured. So the CV→LB constant can be
   trusted only for vectors that are genuine competition-trained models.
3. **V40's number is a finding, not a slip.** `Best iters: [0, 0, 0, 0, 0]` with the arm printing
   `margin route used: es_margin_blind` is the cleanest demonstration in the campaign that early stopping is the
   referee: the prior entered at fit and predict (pre-flight measured +0.3865 mean-p move), the offset was real
   (solo AUC 0.95508), and a stopping rule blind to it deleted the model. The route ladder worked exactly as designed.
4. **V42 is the first paired same-machine control on the tree side**: −0.001361 on fold 1 with iterations rising
   1089 → 2932. The published MVS +0.0012 is now closed on our frame with a real number, replacing the mislabel
   (V35 measured GOSS).
5. **Cost datum for the queue's design.** V38's k-NN column build cost 51 min of a 68.5 min run for a column that
   moved CV −0.00004. Retrieval-class features are the most expensive thing this ladder has ever added per unit of
   measured effect.

## Appendix — V43 lands and FE is finished on measurement, and two inherited banners lied, 2026-10-24 (`LEDGER`)

**V43 (ipsative ratings): CV 0.96124 / LB 0.96101, gap −0.00023.** Paired against the control on the identical split,
folds improved **1/5** and the pooled CV is **−0.00005** — so the gate fails on magnitude *and* on the fold criterion,
and the sign is negative. Two things come out of it.

1. **The transform family list is now closed by measurement, not by verdict.** The register had crosses (V16),
   frequency/rarity (bucket 3), EVs (V11, banked), TE keys (bucket 3), retrieval (V38), numeric resolution
   (dominated by `max_bin`) and now **relative ratings (V43)**. Every one of them has been run on this frame. The
   2026-10-07 note that read "feature engineering is finished on this dataset" is now supported by an arm rather than
   by an argument, and it explains why the campaign's only two real channels are the two that *add a model's output
   as an input* (`_og_teacher` +0.00016, `aux_ev` +0.00012) rather than re-cut the columns.
2. **A new dataset fact, from V43's own control line:** `respondents with no rated item: train 0, test 0`. There is no
   row in 999,479 (= the recorded 699,635 train + 299,844 test) whose 13 ratings are all 0 — the unrated marker only
   ever appears *within* a respondent, never as a
   whole response. That bounds any "disengaged respondent" mechanism: `_count_0` never reaches 13.

**Two log defects this arm exposed, recorded rather than rewritten** (the file produced a shipped vector, so its text is
now evidence and must not be quietly edited):
- V43's **two runtime axis prints** — the head print (line 702) and the `[5/5] RESULTS` footer (line 909) — describe
  **V12's axis** ("+13 numeric aux_ev columns on V12's frame"), because the clone carried V18's inherited strings and
  only the identity constants plus the docstring were rewritten. Same class of defect I fixed in V40 *before* it ran;
  here the file is already evidence, so it can only be annotated. Any future clone must rewrite the head print and the
  RESULTS footer, not just `CFG` and the docstring.
- The axis print is the only place the deviation block's scale appears, and it is correct (mean 0.0000 by
  construction, sd 1.087, range ±3.7), which is what lets a null be read as "redundant" rather than "broken".

**Standing numbers after this arm:** control band unchanged (V18 CV 0.96129 / LB 0.96100–0.96106), highest LB still
**V29 0.96110**, and the twelve-arm streak of |ΔCV| ≤ 0.00013 continues. Still un-run: **K8** (does the shipping net
read its own frame — decides whether an FS arm exists) and **K7** (TabPFN-3 feasibility, with the GBT baseline computed
in-probe).

## Appendix — three averaging arms, three times the CV was paid back as offset, 2026-10-25 (`LEDGER`)

V44 and V45 are the second and third arms whose shipped vector averages more models than the parent's does, and both
pre-declared their implied LB by re-using the parent frame's CV→LB constant. Both predictions failed by more than two
rulers, in the same direction, and the arithmetic of why is now measured three times:

| Arm | what is averaged inside the vector | ΔCV vs parent | frame offset → arm offset | ΔLB vs parent |
| --- | --- | --- | --- | --- |
| V36 (10-23) | outer folds 5 → 10 | +0.00010 | −0.00023 → −0.00035 | −0.00002 |
| V44 (10-25) | seeds 1 → 3, V18 frame | +0.00013 | −0.00023 → −0.00038 | −0.00002 |
| V45 (10-25) | seeds 1 → 3, V29 frame | +0.00014 | −0.00018 → −0.00037 | −0.00005 |

**Read:** the offset grows by almost exactly the CV that averaging buys (Δoffset −0.00012/−0.00015/−0.00019 against
ΔCV +0.00010/+0.00013/+0.00014), so on the board averaging is currently a zero-sum move. Three instances, two
mechanisms (fold count, seed count) and two frames. The mechanism is NOT resolved by these rows — what is measured is
the cancellation, and it is what the next averaging arm must be priced against.

Three consequences for the record, each stated as a rule and not as opinion:

1. **An arm may not inherit its parent frame's CV→LB constant if it changes the averaging depth.** The offset is a
   property of (frame, protocol, vector-averaging), and two arms in one day proved the header arithmetic that ignored
   the third term optimistic by 2.1–3.2 rulers.
2. **The CV-first gate is biased against nothing so much as it is flattered by averaging.** These two arms carry the
   highest pooled OOF this file has ever seen (0.96142, above V36's 0.96139 and V18's 0.96129) and rank 8th/9th on the
   board. A gate that reads CV 0.96142 as a +2.3-ruler win is measuring something the leaderboard does not price.
3. **`digit-for-digit` reproducibility is tree-only.** Same seed, byte-identical recipe, different file: V44's leg reads
   −0.00003 against V18's recorded OOF and V45's legs read ±0.00001 against V29/V34. The exact reproductions the record
   cites (K5 against V11, V30 against V20) are both XGBoost. For the net family, up to roughly half of the ±0.00006
   "seed tick" is run-to-run jitter, so a cross-file control is not a control: net arms need an **in-run paired**
   control (the V42 fold-1 pattern) or the comparison must be stated as ±0.00003.


**And the census says the two blends are the same column as their parents — computed locally, no labels needed.**
Reading only the OOF vectors the runs wrote (699,635 rows, id-aligned row-for-row in every pair, the standing
runbook check):

| pair | logit corr | Spearman | rows differing > 0.05 |
| --- | --- | --- | --- |
| v44 ↔ v18 | **0.999201** | 0.995655 | 9,511 (1.36%) |
| v45 ↔ v29 | **0.999197** | 0.995598 | 9,386 (1.34%) |
| v44 ↔ v45 | **0.999887** | 0.999282 | 179 (0.03%) |
| v44 ↔ v33 (V18 at seed 2027) | 0.999193 | 0.995617 | 9,669 (1.38%) |
| v45 ↔ v34 (V29 at seed 2027) | 0.999193 | 0.995611 | 9,538 (1.36%) |

Finding 82's own rule is that a candidate column above logit corr 0.999 against an existing one **collapses in a
stack and adds 0.000000**. All five pairs are above it, so the campaign's census predicted V44/V45's board result
before either ran: a three-seed blend is not a new column, it is the parent's ranking with the prediction noise
removed — which is exactly what inflates pooled CV and exactly what the leaderboard does not pay for. The two arms
are also the same vector as *each other* (0.999887, 0.03% of rows differing), which is why two different frames
blend to an identical 0.96142.

**Operationally, this is the cheapest gate upgrade available:** for any arm whose axis is averaging (folds, seeds,
bagging), the census against the parent is computable from already-stored vectors in seconds and costs no GPU minute.
Run it before authoring, not after scoring.


**Still owed:** the offline paired DeLong z of `v44_oof` against `v18_oof` and `v45_oof` against `v29_oof`. Both arms'
gates were written as CV bar ∧ z > 3 ∧ determinism, so on the determinism half each already fails its own pre-declared
condition and neither can be adopted on this row; the z is owed because a blended vector is the first case where the
DeLong covariance across legs could be the thing that decides. The arms' own headers cite pre-authoring vector
measurements (+0.000076 / z 2.07 for v18+v33, +0.000123 / z 3.34 for +v36, +0.000086 / z 4.64 for v29+v34) — those are
**reported by the authoring session**, not recomputed here, and are recorded for provenance only.

## Appendix — V48 closes stack training, and the first positive gap turns out to be pooled-OOF drift, 2026-10-25 (`LEDGER`)

**What was tested.** One axis vs V18: three competition-trained GBDT members fitted *inside the run* (no saved OOF is
read, so the standalone-single-model law is satisfied), their cross-fitted OOF logits entering the net's frame as three
`_stack_*_logit` columns, one student fit, one shipped vector. Shape taken from prior episodes; the arm's own header
also priced it against K3's measured nested-stack ceiling of **+0.00023** over the control.

**What happened.** CV **0.96070** = **−0.00059** against V18 (−10.4 rulers), per-fold −0.00030 / −0.00034 / −0.00027 /
−0.00033 / −0.00025 with **0/5 folds improved**, LB **0.96091** — 0.00009 under the V18 band floor and −0.00019 under
the standing best. The student finished **−0.00042 below its own best member** (LightGBM at 0.96112), so this is not a
blend that failed to beat its parts by a little; it is a student that was damaged by the columns it was given. 67% of
the 53.8-minute run was fitting the members (2,175 s), which is what makes the reading cheap in hindsight and expensive
in GPU minutes.

**Mechanism, from the arm's own printed census.** The three members are mutually correlated at **0.99776 / 0.99898 /
0.99796** in logits — by finding 82's rule that is essentially **one channel, not three** — and each sits ~0.962 from
`_og_teacher`, the prior the frame already carries. So the widened frame added a near-duplicate of an existing channel
and the 3-epoch net spent capacity re-deriving it. The cross-episode cascade claim (S4E9/S3E10 winning outright, S6E4's
+0.00105, reported by the authoring session, never verified here) therefore does **not** transfer to this regime,
which is the third time prior episodes have supplied vocabulary rather than recipes (findings 78–79).

**A new datum, and it is a caution about the ledger's own OOF column.** `v48_oof` against `v18_oof` is at logit corr
**0.996478** — *below* the 0.999 collapse threshold, so this vector is genuinely a different column and still lost by
10 rulers: independence is necessary for a stack gain, not sufficient. Separately, its pooled OOF (0.96070) sits
**−0.00029** under its own fold mean (0.96099), where every single-model net arm in this file is within 0.00001. The
file's first **positive** gap (+0.00021) is that drift, not a CV→LB regime change: read against the fold mean the
offset is −0.00008, ordinary for the frame. Practical rule: **for any vector whose columns were themselves
cross-fitted, quote the fold mean and the pooled value together**, or the gap column will look like an anomaly.

**One design choice that worked.** V48 declared **no implied LB** — the V44/V45 lesson applied. It still failed, but it
failed its own gate instead of failing a prediction it had published, so no header arithmetic had to be corrected after
the fact. Averaging arms (V44/V45) and stacking arms (V48) now all agree: with the offset unmeasured for the shape, the
LB is the only admissible evidence, and the census should have priced this before the 53.8 minutes were spent.

## Appendix — V50 takes the record and the offset law splits in two: diversity converts, variance-removal does not, 2026-10-25 (`LEDGER`)

Six composition/averaging arms now sit side by side, and they do **not** behave the same way:

| arm | what is averaged inside the vector | legs | CV | LB | gap |
|---|---|---|---|---|---|
| V29 | one net, one seed | 1 | 0.96128 | 0.96110 | −0.00018 |
| V36 | outer folds of the **same recipe** | 10 | 0.96139 | 0.96104 | −0.00035 |
| V44 | seeds of the **same recipe**, V18 frame | 3 | 0.96142 | 0.96104 | −0.00038 |
| V45 | seeds of the **same recipe**, V29 frame | 3 | 0.96142 | 0.96105 | −0.00037 |
| V49 | **four different arms**, two families | 4 | 0.96156 | 0.96108 | −0.00048 |
| V50 | **five different arms**, two families + two seed-nets | 5 | 0.961595 | **0.96112** | −0.00048 |

Reading the bottom two rows against the middle three: inside one recipe the gap **widens by almost exactly the CV the
averaging buys**, and LB sits still (V36, V44, V45). Across *different* arms the gap **stopped widening** (−0.00048 at
4 legs and again at 5) and CV converted to the board roughly **1:1** (V49→V50: CV +0.000035 → LB +0.000040). So the law
this ledger has been stating as "averaging is offset-bound" is really two laws:

1. **Variance-removal on a shared ranking pays nothing.** The gain is in the metric, not in the model.
2. **Added diversity pays ~1:1, at a constant offset that must be measured once per family mix.**

That is the arithmetic behind the next five days: with the offset frozen at −0.00048, **a board score of 0.9621 needs
CV ≈ 0.9626**, i.e. +0.0010 above V50 - larger than every single-model gain this campaign has ever produced combined,
and not obtainable from a pool whose members all share one frame, one teacher and one split. Either foreign vectors
enter the pool (public-kernel OOFs, K3's pending public-pool run) or the frame itself changes (K8, if ≥30% of the net's
inputs are inert-or-harmful, a slimmer frame is the one member that would be genuinely decorrelated).

**Three method findings from verifying this arm rather than trusting it.**

1. **The house partition is confirmed by reproduction, not by assumption.** Scoring `v18_oof` and `v29_oof` fold-wise
   under `StratifiedKFold(5, shuffle=True, random_state=42)` gives V18's recorded array
   (0.96141, 0.96041, 0.96209, 0.96142, 0.96113) and V29's (0.96134, 0.96042, 0.96213, 0.96141, 0.96113) **digit for
   digit**, while `v34_oof` (its run used the 2027 partition) does not match - exactly the expected asymmetry, which is
   what makes the 5/5-fold claim above a paired comparison. Note this is scoring determinism: it says nothing against
   the net-family training jitter measured on V44/V45.
2. **The recorded DeLong implementation is corroborated; my re-ports were the error.** A first analytic port (components
   taken in label order instead of sorted-positives-then-sorted-negatives) gave z +0.66 for V50 vs V49; the corrected
   port gave +71.1; a full-row bootstrap over 699,635 rows gives **+3.14/+3.38 across two seeds**, against the arm's own
   claimed +3.34. The same test reproduces V49's recorded **+8.76** vs V18 at bootstrap **+8.09**. So the ledgers' z
   values stand, and DeLong work in this repo is to be quoted from the in-run implementation or a bootstrap, never from
   a hand-ported analytic SE.
3. **V50's pooled-minus-fold-mean drift is +0.00000** (0.961595 vs 0.96160), where V48's was −0.00029. So the drift is
   not a property of "blended vectors" in general - it belongs to vectors whose *inputs* were themselves cross-fitted.
   The V48 caution stays, with its scope corrected.

**One number this record does not have: V50's own run log.** It was computed over the archived vectors and no stdout
was captured, so CV, folds and z above are **recomputed here from `v50_oof.csv` plus `Dataset/train.csv` labels**, and
the LB is the owner's reported 0.96112. Provenance is stated rather than implied: the recomputed OOF equals the shipped
vector's AUC to 3e-16, so nothing was re-derived from memory.

**View maintenance, the one edit to an existing line this turn makes.** The Top-5 table is defined (rule 8) as exactly
the five highest LBs, "a view, not a copy". V50 enters at the top, so V18's *view* line leaves it; V18's entry in the
Single Models table is untouched and unchanged. Nothing else in this file was edited.

**Follow-up the same day, and it changes what the gate means.** The comparison V44/V45 owed (task #36) is now computed
by the same full-row bootstrap: **v44 vs v18 dAUC +0.000138, z +5.96** and **v45 vs v29 +0.000136, z +5.31**. So the
seed-averaged arms cleared Δ ≥ +0.0001 **and** z > 3 on CV, and still landed at LB 0.96104/0.96105 - at or below their
parents. Read against V49/V50, where a smaller CV gain (+0.000035 between them) moved the board +0.000040, the
conclusion is not that those arms were wrong but that **the current gate has no diversity term**: Δ, z and fold count
are all satisfied by removing prediction variance from a ranking everyone in the pool already shares. The gate the next
composition arm should face is the one in the appendix above - census first (distinct from the parent *and* from the
frame's channels), then the frozen-offset arithmetic `LB ≈ CV − 0.00048`, then Δ/z/folds as tie-breakers.

## Appendix — V51 calibrates the offset: three blends, one constant, and the own-pool is finished, 2026-10-25 (`LEDGER`)

V51 was authored to break a tie between two hypotheses that every earlier measurement fit equally well. It did, on its
first and only reading:

| blend | legs | CV | LB | gap |
|---|---|---|---|---|
| V49 (v11+v27+v34+v36) | 4 | 0.961564 | 0.96108 | −0.000484 |
| V50 (v44+v45+v36+v27+v11) | 5 | 0.961595 | 0.96112 | −0.000475 |
| V51 (v27+v36+v44) | 3 | 0.961585 | **0.96113** | **−0.000455** |

**H1 — the offset is a property of *blending our pool*, not of how many legs are in it — is the survivor.** H2 (the
offset shrinks toward a single model's −0.00018/−0.00023 as legs are removed) predicted V51 near 0.96128; it landed at
0.96113, 0.00015 away from that and 0.00003 from H1's 0.96110. With three calibration points the working predictor is
**LB = CV_blend − 0.00046 ± 0.00001**, which is tight enough to be useful and honest enough to state its limit: it was
fitted on three blends of the same eleven underlying models, so it licenses *ranking* candidates within this pool, not
extrapolation to a vector from a foreign pipeline (V40's −0.00123 is the standing counter-example of exactly that
mistake).

**Two conclusions this forces on the remaining days.**

1. **Removing legs does not buy board score, so the only thing that can is adding CV — and within this pool CV is
   capped.** V50's quintet was already the maximum the search could find, V51's trio is 0.00001 below it, and the
   exhaustive 2-to-6-subset sweep over all fourteen single-arm vectors (run alongside this appendix) is the record's
   proof rather than an argument. A pool whose every member shares one frame, one teacher and one split has one
   ranking; averaging it cannot produce a better one.
2. **The two numbers on the board are noise, and the ledger should say so in the same breath as the record.** V51 +0.00003
   over V29 (0.5 ruler) and +0.00001 over V50 (0.2 ruler) are inside what a pure reseed moves. Choosing which vector to
   *finish* on should therefore follow CV under the calibrated offset, not the public order: V50 and V51 are
   indistinguishable, and both dominate every single model we ever fitted.

**What is left is not "more of the same", and the honest list is short.** (a) A **different frame** — the K8 re-run,
which is the only in-house route to a member that is genuinely decorrelated rather than another copy; (b) a
**different data regime** — the original 129,880 real rows used as *training* rows, never tried, and the one thing in
this campaign that would produce an inductive bias the pool does not contain; (c) **foreign vectors** — public-kernel
or foundation-model OOFs, the tier the leaderboard's top sits in (our own recorded kratosyan reading is 0.96205 /
0.96167), which needs the asset listing the owner has to print because nothing here can reach Kaggle's pages. If none of
those is available in the time left, then V51/V50 at ~0.9611 is the endpoint of this frame, and the correct closing
move is to stop rather than to sample the ruler for a fifth time.
**Sweep numbers behind the paragraph above, as measured (not asserted).** Exhaustive equal-weight logit-mean search
over every 2-, 3-, 4- and 5-subset of the 14 single-arm vectors (91 + 364 + 1001 + 2002 blends, each scored on all
699,635 training rows):

| size | best CV | members | shipped arm at that size |
|---|---|---|---|
| 2 | 0.961557 | v27+v44 | — |
| 3 | 0.961585 | v27+v36+v44 | **V51** (exactly this set) |
| 4 | 0.961584 | v11+v27+v36+v44 | V49 used v11+v27+v34+v36 = 0.961564, 0.000020 below the best 4-set |
| 5 | **0.961595** | v11+v27+v36+v44+v45 | **V50** (exactly this set) |

**So V50 is not a lucky pick, it is the optimum of its size class, and no smaller class beats it** — the pool's
composition ceiling is 0.961595, and the calibrated law predicts its board score as 0.961595 − 0.00046 = **0.96113**,
which is what V51 actually scored and V50 came within 0.00001 of. The 6-subset layer (3,003 blends) was still running
when this line was written; nothing above 5 members has been claimed. One honest wrinkle the table exposes: V49 is 2
units of a ruler *below* the best 4-set, so the hill-climb that chose it was greedy — the law survived, the search did not.

## Appendix — V52 result and the limits of the stopping claim, 2026-10-27 (`LEDGER`)

| Quantity | V18 | V52 | Difference / decision |
|---|---|---|---|
| Pooled OOF AUC | 0.961287 | 0.961295 | +0.000008; below CV ≥ 0.96140 admission bar |
| Paired-bootstrap z | — | +0.32 | Diagnostic below 3; not a DeLong estimate |
| Folds improved | — | 2/5 | Below ≥4/5 requirement |
| Fold AUCs | 0.96141 / 0.96041 / 0.96209 / 0.96142 / 0.96113 | 0.96139 / 0.96045 / 0.96205 / 0.96150 / 0.96111 | Mixed signs |
| Public LB | 0.96106 | 0.96102 | −0.00004 |
| LB − pooled OOF | −0.000227 | −0.000275 | −0.000048; not an identical constant |
| Pooled − fold mean | ~0.00000 | −0.00001 | Small fold-pooling drift |

**Provenance:** LB was supplied by the owner. CV/folds/bootstrap were computed in the earlier full-row recheck of
`v52_oof.csv` against the 699,635 training labels, not copied from Kaggle stdout. The attachment was the script; timings,
actual fit widths, environment and guard read-backs were not captured. The code specifies an 81-column drop and 208-column
fit. The clean K8 census's 103 includes those 81, 21 retained raw inputs and one probe-only noise column.

**Decision:** no demonstrated single-model gain for this specific removal. The experiment does not prove that the net
ignores all removed information, that rankings are identical, or that every feature-selection approach is exhausted.
V21 and V41 were already subtraction tests; V52 is not the first removal arm. A small gap change is not proof of a
universal CV→LB law. Best recorded public scores remain V51 **0.96113** and V50 **0.96112**.

**Correction to earlier stopping language:** the completed equal-weight subset sweep establishes the best candidate
only in its enumerated pool and subset sizes. It does not cover arbitrary weights, all archived vectors or new models.
Three closely related blends' offsets are descriptive observations, not a guaranteed conversion or prediction interval.
Repeated OOF selection can overfit; fixed-vector bootstrap z does not account for that selection or training-seed variation.
V52's follow-up blend search timed out, and its replacement was stopped with an invalid baseline setup; neither establishes
V52's blend value. That question remains open, with no new blend score recorded.

**Date provenance:** V50/V51 rows say 10-25, whereas their ledger commits (`db0ef2f`, `e2b9d04`) are dated 10-27.
Commit dates and local CSV modification times establish local recording, not necessarily training or submission time.
Historical entries remain unchanged; V52's 10-27 is explicitly its recording date, not an inferred Kaggle run timestamp.

## Appendix — V53/V54 verdicts, the V55 composition test, and what the gap is actually worth, 2026-10-28 (`LEDGER`)

**V55 shipped nothing, so it has no score row.** The composition arm ran its closed, pre-declared candidate list on
the archived vectors and every candidate failed its gate, so by its own declared behaviour it wrote no OOF and no
submission. The record of that measurement belongs here, not in the tables:

| Candidate leg | Leg solo CV | Blend pooled | Gain vs V51 trio | Folds | DeLong z | max logit-corr | Admitted? |
|---|---|---|---|---|---|---|---|
| +v53 TabM | 0.960667 | 0.961546 | -0.000039 | 0/5 | -2.00 | 0.98184 | no |
| +v54 DCN-V2 | 0.958693 | 0.961379 | -0.000206 | 0/5 | -6.47 | 0.98267 | precondition failed |
| +v52 drop-frame | 0.961295 | 0.961565 | -0.000020 | 0/5 | -2.60 | 0.99918 | no |

Control verified rather than assumed: recomputing `mean(v27,v36,v44)` reproduces `v51_oof` to **9.326e-14** in logit
space. **The campaign's closing finding is that decorrelation is necessary and not sufficient.** The two genuinely
distinct channels (0.982, nowhere near the 0.999 collapse rule) both *lower* the blend, and the damage tracks the
leg's own weakness: 10 rulers under the members costs 0.7 rulers, 45 rulers under costs 3.6 rulers. An equal-weight
leg buys its independence by diluting three strong members into four, so a leg must arrive near member strength to
pay for it. This inverts the S6E9 precedent, where TabM was the best non-tree leg — there the pool was tree-dominated
and the members were weaker.

**What the gaps of V53 and V54 correct in the earlier prose.** The V51 appendix's calibrated predictor was written as
`LB ≈ CV − 0.00046 ± 0.00001`, and this appendix's own 2026-10-27 note already warned that three blends are a
descriptive observation rather than a law. Two new single-model families now show how wide that scatter really is:
V53 (TabM) lands **+0.00005** and V54 (DCN-V2) **+0.00096** — both **positive**, where every RealMLP net in this file
sits at −0.00018 to −0.00043. The correct statement is therefore: *the CV→LB gap is family- and mix-specific, with an
observed spread of about 0.0014 across this campaign's arms, and it cannot be predicted for a vector whose family has
not already been submitted.* It was never a constant, and the ±0.00001 precision on it was an artifact of calibrating
on three closely related blends.

**Withdrawn numbering, added 2026-10-28 after the owner's challenge.** The leg-extension test above was authored as
`V55` and that number was mis-spent: with v53 measured 10 rulers and v54 45 rulers below the members it would join, an
equal-weight fourth leg had no arithmetic path to a **+0.00005** gain, so per the campaign's own rule ("an unmet
precondition becomes a cheap probe, not a version") it should have been a stdout-only `K#`. The script was deleted and
its `### Version 55` block was removed from `training_logs.md` (that ledger admits LB-confirmed arms only); the
measurement above stands in this appendix and in `daily_log.md`, which is the ledger whose rule allows an OOF-only
result. **V55 is a free number again, and no `v55_oof.csv`/`v55_sub.csv` exists or ever did.**

**Nothing in this appendix closes the campaign on a ceiling.** What is now measured and closed is: this frame's
feature-selection rung (V52), the two families we own as final-model estimators (TabM budget-limited, DCN-V2 peaked
at epoch 2-3), and equal-weight extension of the V51 trio by one leg. What remains unmeasured, and was never claimed
otherwise: TabM at a converged epoch budget, a weighted rather than equal composition (a weight search the owner's
lift does not authorise), the original-rows-as-training-rows mechanism, and the public foreign-vector pools.

### Appendix addendum — K9 settles the weighted-composition question, 2026-10-28 (`LEDGER`, no score entry)

The paragraph above left "a weighted rather than equal composition" as unmeasured. `Probes/S6E10_K9_WeightCensus.py`
has now measured it: `(1-w)·logit(trio) + w·logit(leg)` for legs v53/v54/v52 across w ∈ {0.05 … 0.35} — **21 cells on
all 699,635 rows, against the same pre-declared bar (gain ≥ +0.00005 AND ≥4/5 folds AND DeLong z > 3). Zero cells
clear; 4 of 21 are positive at all**, and the best is **+0.000013** for v53 at w = 0.05 (0.26 of the bar, 5/5 folds,
z +2.85 — sign-consistent and far too small to reach the board). The trio control reproduced `v51_oof` again
(1.17e-13).
**The null validates its own instrument**, which a stop verdict requires: each leg's gain declines monotonically
after its peak, with grid spreads of 2.1 rulers (v53) and 8.1 rulers (v54) — a real gain of bar size would have been
visible. v52's curve is flat (0.6 rulers) and would be unresolved on its own, but it is already closed by the
equal-weight measurement above.
**Consequence for the record:** composition of this campaign's own vector pool is now closed **in both forms it can be
done** — equal-weight (the withdrawn arm above, three legs, 0/5 folds each) and weighted (21 cells, none clearing). What remains is outside the
pool: the originals-as-rows arm (authored as V55, awaiting its run) and the public OOF tier, which needs the owner's
Kaggle session. No new number enters any table from K9; it wrote nothing and fitted nothing.
