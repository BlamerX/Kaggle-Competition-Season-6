# S6E10 Feature Engineering Notes

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first), with the format block first and feature rows written below it; entries order by version number where that does not conflict with same-day batch run order
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format

The format block stays first; feature entries are written below it.

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |

---

### V55 — the data-regime axis: real rows appended, no column changed, recorded 2026-10-28

| Feature / axis | Construction | Importance % | Impact | Status |
|---|---|---|---|---|
| No feature change (V55) — the axis is **rows, not columns** | V18's byte-identical 289-column frame, recipe, frozen folds and 3-epoch budget; 24,731 of the 25,972 held-out original rows appended to each fold's fit set (+4.4% of 559,708), with every label-consuming producer refit on the R-only 80% so no appended row informs its own features | n/a (no column was added, removed or re-defined) | **CV 0.96117 / LB 0.96094**: ΔCV -0.00012 pooled, folds improved 1/5, gap -0.00023 (V18's constant) | ❌ The mechanism ran and lost — but it answers an FE question on the way (below) |

**Why a rows-only arm belongs in the FE ledger.** The premise this arm was written to test was V51's appendix claim
that the real-data regime is "the one source of an inductive bias the pool does not contain" — i.e. that the frame was
missing something about the originals. The fold-1 instrument refutes that reading: the net scores the appended original
rows at **in-sample AUC 0.99678**, far above the **0.96594** it reaches on its own competition fit rows, so the shipped
289-column frame represents the real regime *better than it represents the synthetic one*. The lost 2.1 rulers are a
**transfer** failure (memorising a cleaner, 4%-by-count minority of the fit does not help predictions on the competition
distribution), not a representation failure. ⚠️➡️❌ That distinction is the useful residual: it says no additional
column family is waiting to be discovered by mixing in real data, because the real data is already being fit almost
perfectly.

**Also settled here**: the twin-vocabulary question that killed attempt 2 is now measured rather than guessed — 1,241 of
25,972 appended rows (4.78%) carry a value the competition pool's twin codes cannot express (Flight Distance 1.58%,
Departure Delay 2.04%, Arrival Delay 2.32%), and dropping them moves the appended slice's positive rate by +0.0002.

---

### V53–V54 — two new families read the unchanged V18 frame, recorded 2026-10-28

| Feature / axis | Construction | Importance % | Impact | Status |
|---|---|---|---|---|
| No feature change (V53, V54) | Both arms fit V18's byte-identical 289-column frame (268 numeric incl. 13 aux-EV + 21 categorical); the axis is the estimator family, so **no column was added, removed or re-defined** | n/a | V53 TabM **CV 0.960667 / LB 0.96072**; V54 DCN-V2 **CV 0.958693 / LB 0.95965** | ⚠️ No improvement — and no FE verdict either way: these arms cannot speak to the frame |

**The FE lesson this section exists to keep.** The K8 census's inert/harmful verdict is **model-specific and does not
transfer to another family** — the two new nets read the same frame and lost 10.9 and 45.5 rulers respectively, which
says nothing about whether a column helps *them* (block importance was never re-measured on either). A feature verdict
is therefore only ever a verdict about the model that was probed; carrying V52's "81 columns are inert" forward as a
property of the frame would be a category error.

---

### V52 — census-guided removal of 81 columns, recorded 2026-10-27

| Feature / axis | Construction | Importance % | Impact | Status |
|---|---|---|---|---|
| Remove the selected K8-inert families (V52) | digits(20) + org_stats(22) + tm_single_TE(22) + other(14) + rating_aggregates(3): 81 columns, intended fit width 289 → 208; V18 recipe unchanged | n/a (block-permutation selection, not gain importance) | **CV +0.000008 / LB 0.96102**, 2/5 folds improved, paired-bootstrap z +0.32; LB −0.00004 vs V18's recorded 0.96106 | No demonstrated improvement for this drop set; not a proof that all feature selection is exhausted |

The corrected census's 103 columns comprise these 81, 21 retained raw inputs, and one probe-only noise column. The noise column was never part of V52. A small AUC change does not prove identical rankings or establish redundancy as the sole cause. Removing raw inputs after feature construction remains a different, untested operation, not an automatic recommendation. Runtime widths and timing were not supplied; the archived script specifies the assertions.

---

### V48 — stack training enters the frame as columns, and the frame rejects it, 2026-10-25. ❌ **The stacking channel closes on a measurement: the student lost to its own parts.**

| Feature / axis | Construction | Importance % | Impact | Status |
|---|---|---|---|---|
| `_stack_xgb_logit`, `_stack_lgbm_logit`, `_stack_cb_logit` (V48) | each member's house-split cross-fitted OOF logit; test side = mean of the member's 5 fold models | n/a (net arm; no gain importances) | **CV −0.00059 / LB 0.96091, 0/5 folds improved**; student −0.00042 vs its best member (LightGBM 0.96112); members mutually correlated 0.9978–0.9990 and 0.962 from `_og_teacher` | ❌ Closed: three GBDT members are **one** near-duplicate channel, and the frame already carries that channel's better copy |

**Why this closes a door rather than narrowing one.** The shape was the campaign's last untested route to the K3-measured
nested-stack ceiling (+0.00023 over the control): fit the parts in-run, hand their logits to the shipped model as
columns, ship one student. The in-run census had already predicted the null and cost nothing to check — mutual member
correlation above the 0.999 collapse bar means the "three" columns are one, and ~0.962 against `_og_teacher` means even
that one is a worse copy of a column the frame has held since V1. The measured result is worse than redundant, though:
the student is **below every member it was trained on**, so the cost is not just a wasted channel but capacity spent
re-deriving one at a 3-epoch budget. Distillation-as-target closed the same way in V30 (the student never beat its
teacher); V48 closes the *feature*-as-distillation route, and together they cover both directions a trained model's
output could enter this frame.

**Ledger caution this arm produced.** Its pooled OOF sits −0.00029 under its own fold mean — every single-model net arm
in this file is within 0.00001 — so the +0.00021 CV→LB gap is a drift artefact, not an offset regime change. Vectors
built on cross-fitted columns must be recorded with both statistics or the Gap column misleads.
### V31–V35 — the ruler lands, two config arms close, and one print explains the FE plateau, 2026-10-17 (late). ❌ **No new channel; five closures, and the campaign's measurement floor moved for the first time.**

| arm | axis (one) | parent | ΔCV pooled | folds | ΔLB | verdict |
|---|---|---|---|---|---|---|
| V31 | confident-row exclusion, q=0.0387 by nested \|p−y\| | V18 | **−0.001987** | 0/5 | −0.00199 | ❌ row-noise selection closed by measurement |
| V32 | `n_ens` 8 → 16 | V18 | +0.000023 | 5/5 | +0.00002 | ⚠️ bagging saturated: 1→8 was +0.000886 |
| V33 | seed 42 → 2027 (nothing else) | V18 | **−0.000057** | n/a | **−0.000060** | ✅ the ruler |
| V34 | seed 42 → 2027 on V29's frame | V29 | +0.000010 vs V33 | 3/5 | +0.000060 | ✅ keys read −0.000007 / +0.000010 CV across seeds |
| V35 | `sampling_method` uniform → gradient_based | V11 | −0.000123 | 0/5 | −0.00006 | ❌ MVS published +0.0012, ours −0.00012 |

**Why this batch belongs in the FE ledger even though it added no column.** Three-for-three was the *channel* closure story; this batch closes the *levers* and, more importantly, re-bases the yardstick every future channel will be judged against.

1. **V34 is the second, decisive reading of the Stage-1 TE keys.** At seed 42 they measured **−0.000007 CV / +0.000040 LB**; at seed 2027, **+0.000010 CV / +0.000060 LB**. Same builder, same budget, opposite CV signs, and an LB "advantage" that is larger than the CV effect it is supposed to reflect. The key channel is therefore not small — it is **unmeasurable**, and this retires the last argument for porting screened keys anywhere (tree V23 −0.00005 / V24 −0.00004; net V29 / V34 as above). Fourth transfer, fourth null.
2. **V35's fold-1 importances are the cheapest explanation of the FE plateau this campaign produced** (`_og_teacher` 0.4177 + `_tm_pair_online_boarding__type_of_travel` 0.2199 = **64% of gain in two channels**). A frame that leans 64% on the original-data teacher and one TE pair has no remaining headroom for a third or fourth key, which is *why* the three-for-three transfers happened — not because the channels are badly built. It also says where any remaining published gain must come from: a **different estimator of the conditional law** (the teacher channel), not a different lookup key.
3. **V31 closes the last "clean the labels" idea with the strongest possible evidence**: the 3.87% of training rows a strong nested model most confidently disagreed with were worth **~50× their share** of the ranking (mean |p−y| 0.947 dropped vs 0.087 kept; deleting them cost 0.002 AUC on every fold). Confident learning presumes those rows are errors; here they are the sharpest signal in the frame. Combined with K1b M2(a) (0 byte-identical rows) and M2(c) (ρ̂ = −0.2149), the label-noise route is closed three ways.

**Open in the teacher channel, with external numbers attached (author-printed, not verified here).** The front's remaining published gains sit on the *original-data* side, which is the one channel that has always worked for us: an original-only lookup + teacher at **+0.00048 (5/5 folds)** and a **k-NN label mean against original rows** reported at ~0.95 AUC alone (+0.00015 on a 4-GBM blend). Neither has been tried here — our use of the original set is a GBDT teacher plus mean/shrink lookups, never a nearest-neighbour estimate. See `PLAN_V31_ONWARD.md` §3 V40.

### V29–V30 — the last two channels close, 2026-10-17. ❌ **The two-architecture test is finished and it is unanimous: three channels, three transfers, sign and size preserved.**

| Feature / axis | Construction | Importance % | Impact | Status |
| -------------- | ------------ | ------------ | ------ | ------ |
| The 4 Stage-1 keys on the **net** (V29) | V23's `pair_class__age5`, `pair_inflight_wifi_service__age5` + V24's `n_unrated`, `pair_wifi_band__online_boarding`, appended after V18's key block → `_tm_`/`_freq_`/`_rarity_` only, +12 per-fold cols, **no twins** | not in the net's prints (no gain table) | **CV −0.000007 vs V18, folds 2/5, LB +0.00004 (0.23σ)** — the campaign's highest LB on a tie | ❌ **keys closed on both families** |
| Soft labels from a nested teacher (V30) | `p=(1−α)y+α·p_nested-teacher` as an XGB DMatrix label, α ∈ {.20,.70}, es ∈ {150,400}, evaluated on hard `y_val` | `_og_teacher` still top gain | **fold means 0.96111/0.96111/0.96111**; Spearman(student, teacher) 0.9928–0.9973 **> 0.99 on every fold** ⇒ COLLAPSE at every rung | ❌ **target mixing closed** — a teacher within 0.00011 of its student has nothing to teach |

**The transfer table, now complete.** Every channel this campaign moved from one architecture to the other, with the tree reading first and the net reading second:

| channel | tree | net | verdict |
| ------- | ---- | --- | ------- |
| rating EVs ×13 | **+0.00012** (V11) | **+0.00012** (V18) | transferred, real |
| categorical posteriors ×7 | −0.00004 (V25) | −0.00001 (V28) | transferred, null |
| Stage-1 TE keys ×4 | −0.00005 / −0.00004 (V23/V24) | 0.00000 (V29) | transferred, null |

**Consequence — and it retires a whole class of planned experiment.** "The tree could ignore this column; the net can't" was the standing reason to re-run a null on the second architecture. It has now been tested three times and held three times, so **no null arm earns a second architecture any more** — a channel is closed after one clean reading. That is why the ladder after V30 is scrapped (`PLAN_V21_ONWARD.md` §4) rather than extended, and why `PLAN_V31_ONWARD.md` stops looking for columns and starts looking at what the front is actually built from. **Nothing we have ever measured on this dataset, on either architecture, has produced a CV gain larger than +0.00016.**

### V21–V28 — Stage-1 sweep on V11's frame plus V28 on V18's, LB confirmed 2026-10-16. ❌ **Eight arms, zero passes: the feature channel is closed on both lines, measured by subtraction as well as by addition.**

Parent for V21–V27 is V11 (CV 0.96113 / LB 0.96064) and for V28 it is V18 (CV 0.96129 / LB 0.96106); the bar is Δ>+0.00005 ∧ z>3 ∧ ≥4/5 folds. Deltas below are CV against each arm's own parent, as the runs printed them.

| Feature / axis | Formula or construction | Importance % (fold-1 gain) | Impact | Status |
| -------------- | ----------------------- | -------------------------- | ------ | ------ |
| The 10 rating aggregates **removed** (V21) | V11's recipe with `_mean_excl0`, `_count_0`, `_wrs*`, `_triple_org_mean` etc. never built: 142 base cols vs 152, guard printed 0 of 10 present | n/a | **−0.00001 CV** (3/5 folds), −0.00006 LB — the block has been inert since the EVs arrived | ❌ dead weight |
| The same 10 aggregates on the corrected scale (V22) | `6 − r` for r > 0, 0 kept as the unrated marker, inside the 9 rating-matrix aggregates | n/a | **0.00000 CV** (1/5) despite K1 measuring the reversed scale at composite 0.85658 (+0.04393) | ⚠️ No Improvement |
| `pair_class__age5`, `pair_inflight_wifi_service__age5` (V23) | `c['Class'] * 18 + floor(Age/5)` and `wifi * 18 + floor(Age/5)`, appended after the parent's key block | not in top-15 | **−0.00005 CV** (1/5) — and these were the **two highest z² in the whole K1 screen** (4.6649, 4.0301) | ❌ z² does not buy CV |
| `n_unrated`, `pair_wifi_band__online_boarding` (V24) | `_count_0` keyed as its own column (byte-equal, no duplicate created); wifi band {0, 1-3, 4, 5} × `Online boarding` under `*10` | `_rarity_pair_wifi_band__online_boarding` **0.0228**, 4th feature | **−0.00004 CV** (1/5) with 0 thin cells — used hard and still worthless | ❌ closed |
| `aux_p_<cat>_<level>` ×7 on the **tree** (V25) | label-free P(level) from an XGB on the other 20 columns, original-fitted; Class 3 + Travel 2 + Customer 2 | not in top-15 | **−0.00004 CV** (1/5), 0.00000 LB | ❌ nominal channel is net-only |
| Teacher capacity ladder (V26) | (8,600) → (10,1000) → (12,1500), one model per rung + 80/20 original hold-out AUC | `_og_teacher` 0.4215 on the control rung | deeper rungs **−0.000108 / −0.000125** and teacher hold-out **0.99510 → 0.99486 → 0.99466** | ❌ axis closed at (8,600) |
| Duo original teacher (V27) | `_og_teacher` = sigmoid(logit-mean) of XGB + RealMLP_TD, both orig-only; Spearman +0.95990 vs the 0.990 limit | — | **+0.00002 CV** (4/5) and **+0.00012 LB** — the only arm ahead of its parent on both, below the Δ bar | ⚠️ best-measured, rejected as configured |
| The same 7 posteriors on the **net** (V28) | V25's builder byte-for-byte on V18's frame (296 cols → 381 expanded) | n/a (net frame) | **CV −0.00001 vs V18, folds improved 2/5; LB −0.00001.** First reported LB-only (0.96105) with the CV block missing; the block arrived the same day, so the row exists now and reads as a null, not a loss | ❌ **channel closed on both families** |

**Consequence (the Stage-1 verdict, replacing the z² screen's premise):** every channel that K1's §5 named as a residual — aggregate orientation, age crossings, unrated counts, banded wifi, nominal posteriors, teacher capacity — came home null or negative, on **both** architectures where the posteriors were tested. Two readings survive the sweep. (1) **The screen's z² is a coverage statistic, not an information one**: the two top-ranked keys scored −0.00005 and −0.00004, and the key that took 2.3% of fold-1 gain scored −0.00004. (2) **The only feature moves that have ever produced CV on this dataset are predictions the model was not otherwise given** — the teacher logit and the rating EVs — and both are now at their measured ceiling (V26 closes capacity, V27 shows averaging a weak second member buys 0.00002, V28 shows a third prediction channel adds −0.00001). FE is finished on this dataset on both lines; what is left is the net's own budget (`n_ens`, V32) and the seed datum V33 owes.

### V18 — LB 0.96106 / CV 0.96129 (2026-10-11). 🏆 **Best single of the episode.** V11's aux-EV block handed to the net: **+0.00012 CV vs V12, folds improved 5/5, LB +0.00011 vs V12 / +0.00055 vs V2 — the second arm ever outside the 0.0003 tie band.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `aux_ev_<rating>` ×13 on the **net** frame | identical builder to V11 (EV = Σ k·P̂(rating=k), each P̂ an XGB on the **other** 20 columns, fitted on the overlap-cleaned original 129,859 rows, label-free); +13 numeric cols, 276→289 frame cols → 374 expanded, 1.20 GB, 115 s build, **no EV categorical twins** | n/a (net frame) | **CV 0.96117 → 0.96129 with all 5 folds improved** (+0.00007, +0.00019, +0.00010, +0.00018, +0.00003); fit cost unchanged vs V12 (146 s vs 154 s/fold) — 13 columns are free on this architecture | ✅ **The block is family-independent**: +0.00012 on XGBoost (V11) and +0.00012 on RealMLP (V18), matching the published RealMLP+EV pair (0.961179→0.961344) |
| EV-vs-raw-rating r screen, re-run under a different parent | same 13 columns | n/a | reproduced V11's numbers **exactly** (entertainment +0.9815, wifi +0.9637 … checkin +0.5218, leg room +0.6214) ⇒ the builder is deterministic and frame-independent | ✅ Confirmed as a reusable pre-training diagnostic |

**Consequence (the FE verdict, now measured on two architectures):** the only two feature moves in this campaign that produced CV were **predictions the model was not otherwise given** — the teacher logit / twin representation (V12, +0.00016) and the rating EVs (V11 and V18, +0.00012 each) — and they **stack**, because they are independent channels. Every **re-cut of information already splittable** measured ≈0: `_tmr_*` +0.00002, `_od_*` +0.00002, org-mean shrinkage +0.00000, leaf-wise −0.00002, V16's crosses +0.00006 at 3/5, V19's leaf knob −0.000024. That is the operational definition of "FE is finished on this dataset": new *columns* no longer help, new *channels* still do.

### V16 — LB 0.96048 / CV 0.96089 (2026-10-11). 39 rating×context native crosses on CatBoost: **the last un-disproved published FE delta, now disproved on our frame.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `_x_<rating>__<context>` ×39 (13 ratings × {Class, Type of Travel, Customer Type}) | `df[r].astype('string') + '\|' + df[c].astype('string')`, passed inside `cat_features` (`CATS + CROSS_COLS`), 12–18 levels per column | **absent from fold-1 top-15** (`_og_teacher` 18.67%, `_tm_fd_exact` 4.55%, raw `Inflight wifi service` 1.54%, `_tm_pair_wifi__booking` 1.51% are what it uses) | `add` rung +0.00006 CV over the in-run `none` control, `swap` +0.00007, both on **3/5 folds** → pre-declared ≥4/5 rule keeps `none`; LB −0.00003 = tie band | ❌ Removed from the plan — the interaction channel was already occupied by `_tm_pair_*` TE + the route profile |
| Manual `_tm_pair_*` family dropped (`swap`, 54 cols removed) | lets CatBoost's ordered target statistics own the interactions alone | n/a | `swap` (0.96096) landed within 0.00001 of `add` (0.96095) and still below the ≥4/5 bar ⇒ the manual pair-TE block is **not** what was blocking the crosses; both encode the same thing | ⚠️ Redundant, not harmful |

**Consequence:** the published +0.00029/+0.00034 for this block was measured on a **poorer frame**. Reproducing a public FE delta requires reproducing the base it was measured against; on our 259-column frame the same 39 crosses are worth +0.00006 with 3/5-fold consistency. Also recorded so nobody re-buys it: `eval_metric='AUC'` is **accepted** by CatBoost 1.2.10 with `task_type='GPU'` but each fit prints `Default metric period is 5 because AUC is/are not implemented for GPU` — the metric is evaluated on the CPU, which is why 15 fits cost 63 min, and the metric-alignment delta itself measured **+0.00001**.

### V15 — LB 0.96061 / CV 0.96105 (2026-10-11). LightGBM family swap: no feature change; **+0.00004 CV, folds 4/5 → fails the Δ>+0.00005 bar by 0.00001.** Kept here for its cross-family evidence about the feature set.

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `_og_teacher` | ektarr-spec XGB on the overlap-cleaned original | LGBM split count 2,526 (XGB gain 40%) | Same first place in both families | ✅ Confirmed the dominant channel across families |
| `_tm_fd_exact` | per-fold TE on exact Flight Distance | LGBM split count **2,079** (nearly the teacher); invisible in XGB's top-15 by gain | Second family partitions the same FD signal very differently, AUC lands 0.00004 apart | ⚠️ Representation differs, information does not |
| `_route_*_resid` / `_route_*_mean` tail (age, checkin, baggage, obsrv, legroom, timehc, wifi) | target-free per-FD profile residuals | 752–976 splits each, ~9 of the top 15 | In XGB these were gain-negligible; in LGBM they carry the split budget, same final AUC | ⚠️ No incremental information |

**Consequence (the cross-family test of the FE-closed verdict):** two GBDT families that visibly disagree about *which* columns matter converge to within 0.00004 CV and 0.00010 LB. Disagreement about importance plus agreement about outcome is the signature of a **noise-limited ceiling**, not of an untapped feature.

### V13 — LB 0.96058 (and 0.96051 on a re-submission of the same file) / CV 0.96101 (2026-10-11). Original-mean shrinkage: **the flagged defect was real, removing it changed CV by 0.00000 → arm closed.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `_<feat>_org_mean` (21 cols) + `_triple_org_mean`, shrunk | `(n_k·mean_k + 30·b)/(n_k + 30)`; b recomputed **0.434379** (plan quoted 0.434463, Δ −0.00008417 — recomputed value used, discrepancy printed as a warning) | n/a (same columns, estimator only) | **649 of 1,513 cells moved**; rows mapping to exactly 1.0000/0.0000 went **train 5,999 → 0, test 2,628 → 0** (the plan's `_gate_org_mean` case 1,455/617 included). CV 0.961011 vs V2 0.961009, folds improved 3/5 | ❌ Correctness ≠ value; the defect stays fixed in the archived file, zero gain |
| Same-`sub` re-submission swing | identical `v13_sub.csv` submitted twice | n/a | 0.96058 → 0.96051 = **0.00007 on a byte-identical file** — our own second LB-noise measurement, consistent with the 0.0002–0.00045 band read off the board | 🔬 Method datum: never chase an LB delta below ~0.0003 |

### V12 — LB 0.96095 / CV 0.96117 (2026-10-11). **Best single of the episode (supersedes V9's same-day LB claim).** Net input frame: +0.00016 CV vs V2 with **folds improved 5/5**, +0.00028 CV vs parent V7, LB +0.00044 — the only arm outside the tie band.

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| 17 numeric→categorical `_cat_` twins | `pd.factorize` over the train+test pool as CategoricalDtype, NaN keeps its own level (−1); **every numeric column also kept** → 276 cols handed to fit, expanded estimate 361 (all-one-hot worst case 4,303) | n/a (net frame) | **+0.00028 CV over V7 on an identical recipe** — the representation, not the architecture, was the net's binding constraint. Re-opens the same question for trees (V16/V17) | ✅ Used, largest single move of the ladder |
| `_og_teacher` as logit | `log(clip(p,1e-6,1−1e-6)/(1−clip))`, same function on both frames | n/a | 0 saturated cells after clipping (prob range 0.000002–1.000000): removes the ceiling the net was training against | ✅ Used |
| Skip-rule consequence | plan pre-declared V12 SKIPPED after three net runs below V2 | n/a | The rule conditioned on the net losing **while fed the tree's frame**; the arm then won the episode. **A skip rule must be stated against the arm's own axis** | ⚠️ Falsified; plan corrected same day |

### V11 — LB 0.96064 / CV 0.96113 (2026-10-11). 13 label-free aux-EV columns: **+0.00012 CV, folds improved 5/5 — the only FE arm of this ladder that moved CV.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `aux_ev_<rating>` ×13 | P̂(rating c) predicted from the **other** ratings, predictors fitted on the cleaned original (129,859 rows), label-free; 116 s build, +13 cols (139→152) | fold-1 top-15: `online_boarding` 1.16%, `leg_room_service` 0.73%, `seat_comfort` 0.60% (~2.5% combined) | CV +0.00012 with **5/5 folds** improved (+0.00007…+0.00018); LB +0.00013 = tie band. Reproduces four public measurements of +0.00009…+0.00017 | ✅ CV-positive; ❌ not yet control (DeLong z + seed-2027 pending) |
| EV-vs-raw-rating Pearson r (instability screen) | r of each EV against its own raw rating column | n/a | `inflight_entertainment` **+0.9815** (near-copy → little new), `wifi` 0.964, `online_boarding` 0.917 … `checkin_service` **+0.5218**, `leg_room` 0.621, `gate` 0.804 — low r means the EV can carry independent information | ✅ Kept as a cheap pre-training screen |

### V10 — LB 0.96065 / CV 0.96103 (2026-10-11). Original-distribution distance features: **+0.00002 CV, +0.00014 LB, folds improved 3/5 → gate FAILS, arm closed.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `_od_<col>_pct0/pct1/pctgap` (9 cols: FD, Age, rating-mean) | class-conditional percentile rank of the competition row inside the clean original's distribution, midpoint 0.5 on NaN, clamped out-of-support | **absent from fold-1 top-15** | Plan expected +0.00005…+0.00035 from the S6E3 analogue; measured +0.00002 with zero selection. S6E3's +0.00036 was on a frame with no `_og_teacher` and no wide route profile — here the original's marginal distribution is already consumed by org means + teacher | ❌ Removed from the plan (arm closed, not re-tuned) |

### V9 — LB 0.96067 / CV 0.96103 (2026-10-11). TE row-statistics block: **best LB of the episode, yet +0.00002 CV with folds improved only 2/5 → gate FAILS.**

| Feature | Formula / value | Importance % | Impact | Status |
| ------- | --------------- | ------------ | ------ | ------ |
| `_tmr_mean` | per-row mean over the fold's `_tm_*` TE vector (float32) | **7.96%** (3rd overall, behind `_og_teacher` 41.5%, `_wifi_org_mean` 12.1%) | Heavy use, no ranking gain | ⚠️ Heavily used, no improvement |
| `_tmr_min` | per-row min over `_tm_*` | **6.83%** (4th) | same | ⚠️ Heavily used, no improvement |
| `_tmr_std` / `_tmr_max` / `_tmr_range` | ddof=1 std; max; max−min | below top-15 | n/a | ⚠️ No improvement |
| Whole block (+5 cols, 120→125 per-fold) | S6E5's +0.00040 LB analogue | n/a | CV +0.00002 / LB +0.00016 (tie band) | ❌ Closed |

**Method consequence (the important one):** this is the cleanest measured separation between *"the model uses the feature"* and *"the feature adds information"*. A deterministic summary of quantities the tree already splits on is selected heavily and still moves nothing. It retires the dimensionality-reduction argument for the TE block and independently confirms the 2026-10-07 verdict that the FE axis is finished on this dataset.

### Targeted region dives — 2026-10-07 (`S6E10-research/personal_region_dive.md`, `soft_law_dive.md`). VERDICT: feature engineering is finished on this dataset.

| Claim | Measured | Status |
| ----- | -------- | ------ |
| V2 recovers essentially the whole recoverable law | implied clean AUC **0.9990** (κ=0.924 at ρ=0.036); 95–98% of the 0.039 gap is irreducible; total modeling headroom **0.0009–0.0019** | ✅ Settled |
| Personal-travel slack is unreachable by features | within-Personal AUC: 5 architectures 0.8299–0.8323, best of 31 rank blends **+0.0000618** whole board (1/5 of tie band); `data.csv` Personal law scores **0.8093–0.8144 vs V2 0.8322**, every blend weight worse, monotonically; Spearman(orig,comp) **0.823 Personal vs 0.990 Business** = the generator re-drew the Personal posterior; 89.5% of the deficit is 6,032 positives in the wifi∈{1,2,3} floor where no column exceeds \|AUC−0.5\|=0.013 | ❌ Region closed |
| Original channel has no un-extracted information | best orig lookup in stochastic region **0.887 vs V2 0.937**; 6-key ensemble 0.881; every blend negative (−0.00004 / −0.00017); where they disagree most V2=0.966, lookup=0.866 | ❌ Closed (`_og_teacher` consumed it) |
| **`_phy_minus_dig` / `_count_ge4_physical8` — RETRACTED** | Step 0 scored it 32/32 positive sign-consistency (p=0.0078), but the targeted test shows the digital/physical split (standalone 0.745, the only genuinely absent representation) **reduces** within-Personal AUC when blended with V2. Sign-consistency of residuals ≠ incremental value | ❌ Arm withdrawn |
| Cell-entropy / uncertainty features | standalone AUC **0.457** (symmetric → cannot reorder a ranking); partial corr to V2 loss **−0.002** after removing V2 confidence | ❌ Dead |
| Other candidates in-region | phy−dig 0.505, max−min 0.465, zero-blocks ≤0.545, row-index/adjacency 0.500 (generator shuffled) | ❌ Dead |

Consequence for the plan: V6 (rating×Customer-Type TE keys), V7 (`max_bin`) and V8 (`lossguide`) are no longer expected to move CV — run them only for closure. Remaining axes with real evidence: architecture with a proper training budget, and combining our own models.

### V6 — LB 0.96045 / CV 0.96064 (2026-10-08). RealMLP bagged (`n_ens` 1→8, everything else byte-identical to V5): **+0.000886 vs V5 (z=+13.40, 5/5 folds)** but **−0.000372 vs V2 (z=−5.90, 0/5)** → net is competitive, not a control.

| Change | Formula / value | Importance | Impact | Status |
| ------ | --------------- | ---------- | ------ | ------ |
| `n_ens` 1 → 8 (PyTabKit bagging) | same 6-epoch flat_anneal budget | n/a | V5→V6 CV 0.95975→0.96064. Confirms V5's failure was training budget, not architecture. `max_epochs=6 reached` in 5/5 again → **epoch count still untested** | ✅ Measured; ❌ not a control |
| V6 as combination leg (measured on saved OOFs, equal logit-mean) | mean of logits | n/a | **V2+V4+V6 = 0.961201 (+0.000192** vs V2 alone) > V2+V6 (+0.000145) > V2+V4 (+0.000113). Adding the weak V5 net **dilutes** to +0.000137 → the pool-width rule bites at n=4 | 🔬 Measured (leg value) |
| Runtime | 22.6 min vs my 2 h estimate | n/a | bags share the data pipeline: 8× fit cost was only 1.5× wall time — do not price NN arms by bag count | ✅ Logged |

### V5 — LB 0.96032 / CV 0.95975 (2026-10-07). Architecture axis: RealMLP_TD on the frozen V2 matrix → CV −0.00126 vs V2, REJECTED (LB −0.00019 = tie band).

| Feature / change | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `RealMLP_TD single` (PyTabKit 1.7.3, n_ens=1, n_epochs=6, lr_sched='flat_anneal') | same 259-col V2 matrix, model swapped | n/a (no importances) | CV −0.00126 / LB −0.00019; paired DeLong dAUC −0.001258, z=−15.67, **0/5 folds improved**. `max_epochs=6` reached in 5/5 folds → early stopping never fired, so this is an UNDERTRAINED net, not a falsified class: every published RealMLP number here used n_ens 8–32 bagging | ❌ Rejected as control; iterate in place with n_ens>1 before ruling the architecture out |
| CV→LB gap sign flip | V2 −0.00050 → V5 **+0.00057** | n/a | In-fold early stopping depresses a GBDT's CV; a net with no ES trigger keeps no such handicap — so cross-architecture CV comparisons are not gap-neutral. Audit nets on LB, not just CV | 🔬 Research |
| V2+V5 logit blend (recorded, not a version) | 0.8·logit(V2)+0.2·logit(V5) | n/a | OOF **0.961121 (+0.000112** over V2 alone); resid corr 0.99002, logit corr 0.99221. Same finding as V4: a weaker-but-different net still buys ~1e-4 in combination. Kept as a measurement only — combining is out of scope for versions | 🔬 Measured (not shipped) |
| `fill_continuous_nan` (forced by PyTabKit) | fit-fold medians on numeric, first level on unseen category | n/a | 6,140 cells (0.0014% of the matrix). Declared as a second change in this version: Step 0 bounded fill choices at ΔAUC ≤0.000415, so it cannot explain −0.00126 | ✅ Required for any NN/TFM version here |
| PyTabKit API facts for this build | `n_epochs`/`batch_size`/`lr_sched`/`use_early_stopping`/`early_stopping_*_patience`/`val_metric_name`/`verbosity`/`tfms`; fit is `(X, y, X_val, y_val)` with **DataFrame** X_val; no `__version__` (use `importlib.metadata.version`) | n/a | Three failed runs came from invented names (`epochs`, `batch`, `epoch_scheduler`, `auto_configure`, `X_val.values`). Copy arg names from our own S6E2_V48 / S6E3_V73 / S6E5_V21 scripts, never from docs memory | ✅ Used (unblocks all NN arms) |

### V4 — LB 0.96051 / CV 0.96088 (2026-10-02). CatBoost member: solo −0.00013 vs V2 → NOT a control; KEPT for stack (blend 0.4·CB + 0.6·V2 logit-mean = 0.961122, +0.000113 over best single; residual corr V2~V4 = 0.997).

| Feature | Formula | Importance % (CB) | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| V2+V4 logit-blend (w_CB=0.40) | 0.4·logit(CB)+0.6·logit(XGB) | — | OOF 0.961122 (+0.000113); best w on [0.25,0.75] plateau 0.40–0.50 — first measured stack delta | 🔬 Measured (V7 seed) |
| `_tm_fd_exact` | TE of exact FD (smooth=20) | 4.15 (CB #2) | route lever matters MORE to CB than XGB — diversity confirmed | ✅ Used |
| `_route_wifi_resid` | rating − route-mean(wifi) | 1.33 (CB #4) | route residuals into CB top-5; XGB never ranked them | ✅ Used |
| `_og_teacher` | orig-only XGB d8 pred | 19.57 (CB #1) | same mechanism, lower share than XGB's 40 | ✅ Used |
| CatBoost params | d8 lr0.05 iter6000 ES200-on-Logloss, Bernoulli ss0.8, GPU | — | fold BestIter 940–1195; Logloss ES (GPU lacks AUC metric in build 1.6.1 env) | ✅ Used |

### V3 — LB 0.96065 / CV 0.96096 (2026-10-02). Duo teacher = tie, NOT adopted.

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `_og_teacher` (duo) | 0.5·(logit_xgb_d8 + logit_lgb64), orig-only | 40.02 | CV −0.00005 vs single XGB teacher: tree+tree diversity on the same source adds nothing; goodpjw's +0.00072 was tree+MLP. Channel saturation proven — teacher axis closed | ❌ Removed (V4 reverts to V2 single teacher) |

### V2 — LB 0.96051 / CV 0.96101 (2026-10-02, +0.00040 CV vs V1). ACCEPTED: paired DeLong vs v1_oof ΔAUC +0.000397, SE 5.3e-5, z=+7.44 (gate PASS; 200× bootstrap 0/200 sign flips). Importance % = fold-1 XGB gain share of the V2 run.

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `_og_teacher` (ektarr spec) | XGB d8/cs0.5/ss0.8/600 on all 21 native-cat feats of overlap-cleaned data.csv | 40.43 | Upgrading teacher from d6/17-numeric raised CV +0.00040 and its own precision (share fell while total rose — family sharpened) | ✅ Used |
| `_tm_pair_online_boarding__type_of_travel` | TE smooth=20 on OB×Travel | 17.64 | Sharpest pair; #2 overall | ✅ Used |
| `_count_ge4_inflight8` | 8-col rating aggregate | 3.04 | Tripled its V1 share | ✅ Used |
| `_cust_org_mean` | orig target mean per Customer Type value | 2.52 | New in V2 (all-21 org means) — instant top-5 | ✅ Used |
| `_tm_pair_inflight_wifi_service__type_of_travel` / `_tm_pair_class__inflight_wifi_service` | TE wifi×Travel, Class×wifi | 1.99 / 1.31 | ektarr KEY_PAIRS confirmed | ✅ Used |
| `_triple_org_mean` | orig mean per Travel×Class×Customer cell | 0.91 | New in V2, top-10 first try | ✅ Used |
| `_freq_pair_online_boarding__type_of_travel`, `_tm_pair_wifi__ob`, `_tm_pair_wifi__booking`, `_freq_pair_wifi__booking` | pair freq + TE extensions | 0.48–0.68 | family holds | ✅ Used |
| Wide route profile (std + resid + 10 cat shares + count, per-FD pool) | ektarr distance_profile | not in top-15 | Part of the +0.00040 package; isolated ablation not yet run | ✅ Used |
| Int-coded pair keys (`a*10+b`, `code_a*n_b+code_b`) | replaces '__'.join | n/a (perf) | prep 290s→10s/fold; total 26.7→3.8 min | ✅ Used |

### V1 — Official baseline, LB 0.96019 / CV 0.96061 (2026-10-02). Importance % = fold-1 XGB gain share of logged run.

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `_og_teacher` | XGB (d6, 17 numeric feats) fit on data.csv, avg of 5 fold preds; comp labels never seen | 47.97 | Dominant column; mechanism = clean law prior under ~4% flip noise. Thin version — ektarr-spec upgrade (d8/cs0.5/ss0.8/all 21 native-cat feats) is V2 candidate | ✅ Used |
| `_tm_pair_online_boarding__type_of_travel` | TargetEncoder(smooth=20, inner-cv5) on OB×Travel | 8.03 | Pair TEs carry the stratified law | ✅ Used |
| `_wifi_org_mean` / `_ob_org_mean` etc. | orig per-value target means (ratings + FD//10) | 2.71 (wifi) | Cheap orig prior, matches S6E9 pattern | ✅ Used |
| `_tm_pair_customer_type__type_of_travel`, `_tm_pair_inflight_wifi_service__type_of_travel`, `_tm_pair_class__inflight_wifi_service` | TE on category×rating pair keys | 1.62 / 1.25 / 1.22 | ektarr KEY_PAIRS all landed top-15 | ✅ Used |
| `_count_eq5`, `_count_ge4_inflight8` | 13/8-col rating aggregates | 1.40 / 0.89 | Tree-inexpressible aggregates confirmed useful as features | ✅ Used |
| `_freq_online_boarding`, `_rarity_pair_wifi__booking` | fit-fold freq + −log(freq) on singles/pairs | 0.78 / 0.73 | Frequency family earned its slots | ✅ Used |
| `Inflight wifi service`, `Class`, `Customer Type`, `Type of Travel` raw (wifi 1.03) | raw cols, cats native-encoded (this run) | ~1 each | Raw signal survives the TE cluster | ✅ Used |
| Route profile (15 per-FD pool means + count) | target-free | not in top-15 | Contributed; wide variant (std+residual+shares) untested → V2 | ✅ Used |
| Digit feats (8 pos × 4 numerics) | mod/div digits | below top-15 | Kept; S6E10-public +0.00069 precedent | ✅ Used |
| Sample weights | — | — | Removed from this design (no public kernel uses them) | ❌ Removed |
| Rule-region AND-flags | see Research Batch V1-revision below | dev-run 37.1 | hijacked splits; excluded from V1 run | ❌ Removed |

### Research Batch V0 — 2026-10-02 (pre-V1; sources: S6E10 public notebooks, S6E1–E9 memory logs, S6E9 winner writeups, original-dataset kernels, local measurement)

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `route_profile(FD)` | per-Flight-Distance target-free mean of every other feature + row count | TBD | goodpjw2008 measured +0.00160 CV (0.958763→0.960362), beats TE-with-target; local: 3,474 levels, median 52 rows/level, 99.98% of test FDs seen in train | 🔬 Research |
| `TE(FD) + rounding keys` | fold-safe target encoding of FD, FD//10, FD//100, FD%1000 × {Class, Travel, Age}; k=1 smoothing | TBD | local honest 5-fold AUC alone 0.743 (k=1 ≫ k=200: 0.725); TE(FD)+count_ge4 rank-blend 0.851 vs 0.827 → additive; evgendvorkin +0.00010/0.00006 | 🔬 Research |
| `{col}_org_mean` | original data.csv target mean per value of each column | TBD | evgendvorkin +0.00006, ranks top-2 importance; S6E9 logs: org means always top-10. Local: competition value vocabulary 100% copied from data.csv → per-value priors apply directly | 🔬 Research |
| `og_teacher_logit` | prediction (logit) of a model trained ONLY on data.csv, added as feature (xgb d8 + RealMLP) | TBD | goodpjw2008 measured +0.00072. Explanation (our pair-mining): data.csv carries the CLEAN label law, competition labels have ~4% symmetric flip noise — teacher = denoised margin | 🔬 Research |
| `pool_orig_lift` | freq_pool(value)/freq_orig(value) on categorical + n-gram keys, TE'd | TBD | S6E9: TE_lift_trigram was #1 feature (14.49% XGB gain); S6E10 FD/delay marginals differ heavily from orig (KS 0.22–0.35) so lift has room | 🔬 Research |
| `count_ge4_inflight8` / `count_ge4_all13` | #(of 8 in-flight ratings ≥4) / #(all 13 ≥4) | TBD | local rank-AUC 0.8352 / 0.8269 — strongest composites; sat rate walks 5.7%→93.2% across 0..13 | 🔬 Research |
| `digital_trio_break` flags | (wifi==0)&(time==0), plus (booking==0), (ob==0) masks; count_0 | TBD | dataset README: 0 = "Not Applicable". Local: zeros nest (P(book0\|wifi0)=0.83); wifi0&time0 → orig sat 1.0000 (n=1834), synth 0.941; count_0=3-4 → 0.86-0.95 sat | 🔬 Research |
| `rule-region flags` | wifi≤3&Personal; wifi≤1&Eco; OB5&wifi5; Personal&OB≤2; Business&OB≥4; age<13; age≥65 | TBD | CONFIRMED near-deterministic in orig (0.0000/0.0002/0.9996/0.0175/0.8803/0.135/0.184) → diluted exactly by ~4% flip noise in synth (0.036-0.040/0.955-0.961). Free high-information binaries | 🔬 Research |
| `mean_excl0`, `row_std` | row mean/std of ratings with 0→NaN | TBD | thread 744809 proposes (σ separated classes); local mean_excl0 AUC 0.8127 vs mean_all 0.8013 → 0→NaN handling confirmed; 744890 same idea | 🔬 Research |
| `digit feats` | FD%10, FD%100, FD//10, FD%100==0, Age%10 | TBD | evgendvorkin measured +0.00069 ("generator artifacts"); local: FD last-digit 0 → +7pp sat; last-digit-0 share 5.6% synth vs 9.4% orig (re-rolled); S6E3 +0.00024 | 🔬 Research |
| `delay_gap`, `total_delay`, `delay_per_100km` | Arr−Dep; Dep+Arr; total/(FD km) | TBD | lordozvlad (127 votes, perm importance) + thread 744809 (13.3% delayed subspace, Business Δ0.61); local: corr(dep,arr)=0.84, R²(arr~dep)=0.709, delta sd 3.3 — gap is tight; raw delay AUC only ~0.49 so value must come from interactions | 🔬 Research |
| `arr_delay_imputed` + `arr_isna` | Trees: leave NaN raw (native default-direction routing) + indicator; dep-fill ONLY inside non-tree stack members | TBD | local bound: 292 NaN rows ⇒ max achievable ΔAUC from ANY fill method ≈ 0.0004 (pair-mass argument), below acceptance gate and LB tie band; isna itself sat 0.363 vs 0.444, same bound applies. R²(Arr\|Dep)=0.709 justifies dep-fill directionally; transformer/complex imputation has no evidence for or against — the box is too small to matter | ⚠️ Settled by bound, not by trial |
| `Class×Travel(×Customer) combo` | concatenated category as one column + TE key | TBD | local stratified law: at mean-rating≥3.7, Business 0.917 vs Eco 0.394, Personal caps 0.186; S6E3/S6E1 combo TE paid; yekenot used Class×Travel×Gender for NN | 🔬 Research |
| `TE_row_stats` | mean/std/min/max/range across TE columns | TBD | S6E5 measured +0.00040 LB (best FE of that episode) | 🔬 Research |
| `CatBoost native-cat` | ratings + FD cast categorical (ordered TS instead of manual TE) | TBD | goodpjw2008 +0.00015-0.00019; nina2025 vanilla CatBoost 0.96124 with zero FE — near top of LB | 🔬 Research |
| `soft teacher labels` | train+teacher-soft test rows | TBD | goodpjw2008: +0.0002 single model, ≈0 in stack; S6E9 3rd place it helped, 2nd place it didn't persist | ⚠️ Research |

### Dead ends — split by evidence provenance (rule: only items measured ON THIS COMPETITION may be marked dead here)

**Measured on S6E10 data (this session locally, or public author-measured on the competition):**

| Idea | Where measured | Result | Status |
| ---- | -------------- | ------ | ------ |
| Append data.csv rows as extra training data | goodpjw2008 (public notebook, competition data) | Failed (no gain); S6E9 prior agrees | ❌ Dead on S6E10 |
| Label cleaning / drop flipped rows | goodpjw2008 −0.002; local exception analysis (synthetic exceptions at base rates = pure noise, orig exceptions systematic 69% Personal) | Removes matched noise structure | ❌ Dead on S6E10 |
| Adversarial validation / drift features | local: KS(train,test) ≤ 0.0027 on every column | No shift exists to model | ❌ Dead on S6E10 |
| Broad HPO (Optuna) | goodpjw2008 +0.00003 | Below any gate | ❌ Dead on S6E10 |
| Missing-value fill METHOD as a lever | local bound: pairs involving the 292 NaN rows ≤ 0.000835 of all pairs → max ΔAUC ≈ 0.0004 < LB tie band | Decision matters ≤0.0004; use native GBDT NaN routing + isna indicator, dep-fill only inside non-tree members | ❌ Not worth a version |
| 39 rating×context **native** crosses on CatBoost (the last published FE delta) | **V16** `add`/`swap` rungs, in-run `none` control | +0.00006 / +0.00007 CV, **3/5 folds**, no `_x_*` column in fold-1's top-15; `swap` ≈ `add` ⇒ `_tm_pair_*` + route profile already own the interaction channel | ❌ Dead on our frame (the published +0.00029/+0.00034 was measured on a poorer base) |
| CatBoost `eval_metric='AUC'` on GPU as a metric-alignment gain | **V16** `none` rung vs V4's recorded CV | **+0.00001**, and every fit printed `Default metric period is 5 because AUC is/are not implemented for GPU` — the metric is computed on the CPU (15 fits = 63 min) | ❌ Dead: no mechanism, no measurement |
| LightGBM leaf-count regularization (`min_child_samples` 20→40) | **V19** vs V15 | **−0.000024 CV**, folds ±0.00001, best iters moved up to 329 trees with an unchanged importance profile | ❌ Dead: inert for ranking |
| Longer RealMLP budget (6/8/10 epochs) | **V17** 3-rung ladder vs V12's 3 epochs | **−0.00025 CV** at every rung, folds improved 1/5 ⇒ `best_epoch = budget` was a fixed-budget artifact; PyTabKit's schedules are budget-relative | ❌ Dead: 3 epochs is the optimum for this recipe/frame |
| Outer folds 5→10 on the net (variance lever) | **V36** `S6E10_V36_TenFoldAveraging.py` (LB confirmed 2026-10-23) | **CV +0.00010 (0.96139, our highest) but LB 0.96104 = inside V18's band**; offset for this protocol −0.00035 vs the 5-fold frame's −0.00023 | ⚠️ CV-only gain, board tie: fold-count CV is not on the parent's scale |
| `plr_sigma` 8 and 32 (Deotte's +5.5e-4 knob) | **V37** ladder, V18 frame, per-fold paired | σ=8 **−0.00001**, σ=32 **−0.00008** vs parent; σ=32−σ=8 mean **−0.00006** (one ruler) — the inherited 2.33 already sits on the plateau (library default is 0.1) | ❌ Closed on our frame: the published gain was banked in V7's bundle |

**Prior-episode evidence ONLY (not tried here — transfer suspicion, not proof):**

| Idea | Episode evidence | Status |
| ---- | ---------------- | ------ |
| Hard pseudo-labeling | S6E3 "0/18+ methods", S6E9 never beat non-PL | ⚠️ Prior: disfavored |
| kNN density ratios, all-pairwise cells, pair-lift/novelty, PCA, WoE-over-TE, rank:pairwise, >10 folds | measured null/negative in S6E3/S6E9 logs | ⚠️ Prior: disfavored |
| Frequency encoding, groupby-mean/diff features | S6E1 all ~0.00 OOF | ⚠️ Prior: disfavored |

Caveat: mechanisms CAN differ across datasets — e.g. S6E9's "count features already carry the generator fingerprint" may not hold here where FD/delay marginals were re-rolled (KS 0.22–0.35 vs orig). Disfavored ≠ banned; revisit only if this episode's residuals show the gap the prior predicted would be absent.

### Ceiling note

With ~4% symmetric label noise, achievable LB AUC ≈ 0.5 + 0.92×(clean_AUC − 0.5). Public top-30 spread is 0.0006 around 0.961 — treat any Δ < 0.0003 as a tie; gate features on paired DeLong z>3 per the repo protocol.

### Research Batch V1-revision — 2026-10-02 (after dev-run OOF 0.96009; ektarr kernel source mined; recipe folded back into V1 — versions are only numbered once finalized)

| Feature | Formula | Importance % | Impact | Status |
| ------- | ------- | ------------ | ------ | ------ |
| `Fixed-relation flags` (dev-run `_r_*`, `_is0`, `trio_break`) | 1-3-condition ANDs over raw cols | dev-run fold-1 gain 0.37 on `_r_biz_ob4` alone | RULE: deterministic ANDs are tree-recoverable in ≤3 splits and hijack split selection; ektarr's stronger XGB has none. REMOVED from V1 (revised). Kept: 13-col aggregates (count_ge4 family, mean/std/min_excl0, count_0) + delay arithmetic + digits — tree-inexpressible | ❌ Removed |
| `Curated TE keys` | TargetEncoder smooth=20 inner-cv5 on 17 singletons + 10 pair keys (rating pairs a*10+b incl wifi×ob) + fd exact/div10/div100 + age_5/fd_250 bins | TBD | ektarr recipe (beat V1 design); blanket TE on all features dilutes the colsample pool — V1's #2 feature was `_te__r_biz_ob4`, pure redundancy | 🔬 Research |
| `frequency_ + rarity_` | fit-fold value_counts + −log(clip) on 17 singletons + 10 pairs | TBD | ektarr + evgendvorkin both include; rarity = S6E9 lift cousin | 🔬 Research |
| `Native categorical` | 4 cats as CategoricalDtype (fit vocab, unseen→NaN) + enable_categorical=True | TBD | ektarr uses it; goodpjw measured +0.00015-19 for the CatBoost CTR equivalent | 🔬 Research |
| `Params light` | mcw 5, ss 0.9, cs 0.85, λ5, α0.05, lr 0.04, es 150, 4000 | — | V1 α10/λ10 over-regularization diagnosis confirmed by two independent public configs; no public kernel uses sample weights | 🔬 Research |
| `Teacher overlap-clean` | exclude orig rows exactly matching comp predictors before teacher fit | — | ektarr hygiene; local count ≈21 rows (immaterial here, adopted for correctness) | 🔬 Research |

---

## Appendix — record corrections, 2026-10-12 (read out of the archived code and this file's own rows)

Rule 2 forbids editing the entries above, so nothing here rewrites a historical row; each item states the
row it affects and what the code or the other ledger actually shows. No new feature entry and no score claim
is added, so rule 1 is not engaged. These matter because this file, not the scripts, is what the next arm's
spec gets quoted from.

| # | Row affected | What it says | What the evidence says |
| --- | --- | --- | --- |
| 1 | **V4 — "CatBoost params … GPU lacks AUC metric in build 1.6.1 env"** | attributes **1.6.1** to the CatBoost build | 1.6.1 is **scikit-learn's** version, printed by every Kaggle run of this episode (`training_logs.md` lines 49, 176, 392, 455: "sklearn 1.6.1"). V4's script imports CatBoost but never prints its version, so no CatBoost number was ever measured in that run. The only printed one is **1.2.10**, from V16 (`version('catboost')`, recorded at `training_logs.md:145`). The *mechanism* note is unaffected and is corroborated by V16's own runtime message `Default metric period is 5 because AUC is/are not implemented for GPU` — which is exactly a 1.2.10-era limitation. Consequence: **V28** (ordered target statistics + native crosses on CatBoost) must be written against **1.2.10**, and any param name taken from 1.6-era documentation is a Arg-Name-Law violation waiting to fail the run. |
| 2 | **Research Batch V0 — `count_ge4_inflight8 / count_ge4_all13`, "local rank-AUC 0.8352 / 0.8269"** | 0.8352 for `_count_ge4_inflight8` | Retracted on **2026-10-07**: the value is **0.8132** and the level ordering inverts (the correction is recorded in `daily_log.md` for that date and carried into `PLAN_V21_ONWARD.md` §2 P2). `_count_ge4_all13` at 0.8269 stands. Anyone quoting 0.8352 as the shipped-best composite is buying a number this campaign already disproved — and the corrected value is what V22b/V29 comparisons should be measured against. |
| 3 | **V3 — "tree+tree diversity … adds nothing … Channel saturation proven — teacher axis closed"** | closes the teacher-diversity axis | V3's implementation changed **two** things: it made `_og_teacher` the mean of two teachers *and* wrote the **raw logit** mean into the column (V3:363-366), while V2/V11 store `predict_proba` (V2:342-343, V11:323-324) — so a [0,1] probability became an unbounded signed value at the same step. Under the single-axis law that is not a clean test, and V3's −0.00005 is evidence against **tree+tree on a rescaled column**, not against two-teacher diversity. `PLAN_V21_ONWARD.md` §5 now carries this caveat inline, and V21 (authored 2026-10-12) averages in logit space and maps back through the sigmoid so the estimator set is the only thing that moves. |
| 4 | **V1/V2/V9/V11/V18 rows — column counts quoted in different units** | 139→152 (V11), 120→125 per-fold (V9), 259-col matrix (V5), 276→289 → 374 expanded (V18) | All correct, but the file never says which number is the **base frame** and which is the **fold-expanded matrix**: base 139 + 120 per-fold TE = the 259 the nets inherit, and the net lines (276 = 259 + 17 twins, 289 = +13 EVs) are expanded-matrix counts, not base counts. Quoting one family's number against the other family's is how a frame-width claim goes wrong — `PLAN_V21_ONWARD.md` §7 now states the census in those two units explicitly. |

## Appendix — measured FE facts from the K1 battery, 2026-10-13 (`K1`: CPU analysis over the archived vectors and the 129k original)

Rows above stay as written (rule 2). This appendix records **what was measured on 2026-10-12** by
`S6E10_EDA_K1_CompleteBattery.py` (51.6 s, local CPU, owner-authorised because it fits nothing and trains
nothing). **The script was deleted afterwards while untracked; the surviving evidence is
`%TEMP%/S6E10_workspace/k1_stdout.txt`, and the numbers below are quoted from it, not from memory.**

**1. The residual z² screen re-opens the FE axis, with addresses.** On `v18_oof.csv` (the reference vector,
read for analysis only), cells with n ≥ 200 across 27 keys: **max z² = 4.6649**, against an authoring bar of
1.5 and a healthy-stack reference of 1.06. Per key:

| key family | max z² | top cell | n | gap (obs − mean p) |
| --- | --- | --- | --- | --- |
| `Class × _age_band` | **4.6649** | Business, under 13 | 853 | −0.0172 |
| `_wifi_band × _age_band` | **4.2198** | band 4, age 13-17 | 3051 | +0.0155 |
| `Class × _age_band` | 4.0301 | Eco Plus, 65+ | 1115 | −0.0133 |
| `Online boarding × _wifi_band` | **3.8548** | OB 1, band 4 | 1288 | +0.0169 |
| `Class × _age_band` | 3.7787 / 3.6891 / 3.4165 | Business 13-17 · Eco Plus <13 · Eco Plus 45-54 | 2170 / 1139 / 5030 | +0.0100 / −0.0127 / +0.0072 |
| `_rated_n` | **3.6983** | 4 unrated fields | 2886 | −0.0078 |
| every single column | ≤ 2.7674 | `Cleanliness` highest | — | — |

Five of the top fifteen cells are in the `Class × age` family. **Across V1–V20 `KEY_PAIRS` has never
contained an Age pair at all** — Age enters as a raw column, as digit features, as `age_org_mean` and as the
single `age_5_bin` key — so this is localized structure in interaction families the frame has never carried,
which is what arms **V23** (age keys) and **V24** (unrated-count + banded-wifi keys) test. Decile calibration
is clean (max gap 0.0037), so the residual is structural and local, **not** a scale or calibration problem.

**2. `_rated_n` is not a new feature — it is the shipped `_count_0`.** Both are the count of zero ratings
across the 13, byte-equal. What V24 adds is the **key**, never a duplicate column. Likewise K1's `_wifi_band`
is **{0, 1-3, 4, 5}** (`_wifi_band:0`/`1-3`/`5` cells in the noise map, a `4|…` cell in the screen) — there is
no `4-5` bucket, and the raw `wifi × Online boarding` pair already ships, so banding is the whole axis.

**3. The backwards rating is now a measurement, not a claim, and it is bigger than the ledger said.**
`Departure/Arrival time convenient` univariate AUC **0.47346** — the only rating below 0.5 — with level
satisfaction rates 0.506 at 1 falling to 0.437 at 5, and the original data agrees (ρ = 1.0 on the ordering it
is shipped in). Reversing **only that column** inside the aggregate builder moves the best univariate
composite `_mean_excl0` from **0.81266 → 0.85658 (+0.04393)**, the largest single-column effect ever measured
here, and `_count_ge4_all13` from 0.82694 → 0.85620 via `_c4_thcrev`. It simultaneously pushes `_min_excl0`
the other way (0.67982 → 0.61162) and `_std_excl0` from 0.45586 → 0.51211, so the **net** effect on a fitted
model is still an open question — which is exactly arm **V22**. Nine of the ten aggregates read the rating
matrix; eight change value, because `_count_0` is 0-preserving.

**4. Two composite claims die on the measurement.** `_wrs13 mean` scores **0.81266 = exactly `_mean_excl0`**,
so the recorded 0.9078 for the weighted composite is unreproducible on this frame and P2's "weighted
composite" is a rename of a shipped column. And `_count_ge4_inflight8` measures **0.81325**, the value the
2026-10-07 retraction said; the 0.8352 in the row above remains dead.

**5. The contested age parabola closes.** A centre fitted at 48 scores 0.66981, but under a label-blind centre
it scores **0.60450 — below raw Age at 0.62436**. The claim was an artefact of fitting to the labels it was
then scored against.

**6. Noise headroom, bracketed rather than asserted.** Mass-weighted |competition − original| rate divergence
over pure cells is **0.0468** (min 0.0360, median 0.0618, max 0.2079), while the model-free near-duplicate
floors are 0.0411 at 0.08% coverage and 0.059–0.067 at 64–72%. **0.0468 cannot be a symmetric flip rate**:
substituting it (κ = 1 − 2ρ) implies V18's clean AUC is 1.061, which is impossible — so it includes the
generator re-drawing the Personal posterior (agreement 0.4877 in Personal, 0.9703 in Business) as well as any
flips. Working position: ρ ∈ [0.036, 0.047], cap ∈ [0.9620, 0.9630], headroom ≤0.0019 and plausibly ≤0.0009.
Train↔test drift is absent (max |Δρ| 0.0052), so the standing negative CV→LB gap is submission noise.

**Provenance and defect notice.** The battery's `N_BOOT=120` bootstrap **ran 0 reps on every pair**, so none
of the above has a bootstrap cross-check; its §0b idiom probe raised a `TypeError` (an ndarray into
`Series.corr`), which under-reports pre-flight coverage without affecting §5; and this box is
sklearn 1.9.1 / pandas 3.0.6 against Kaggle's 1.6.1 — the analysis transfers, the training paths do not.
Nothing in this appendix is a version's score claim, so rule 1 is not engaged: V21–V24 and V22 still have no
CV and no LB, and no row for them exists anywhere.

## V38 / V41 measured rows (2026-10-24, CV + LB confirmed) (`LEDGER`)

| Feature / axis | Construction | Impact | Status |
|---|---|---|---|
| `knn_orig_prob` (V38) | K=50 mean original label over the nearest overlap-cleaned original rows, z-scored numerics + one-hot cats; solo AUC 0.94997 | CV −0.00004 / LB 0.96100 vs V18; build cost 51 min | ❌ Dead — dominated by `_og_teacher` (0.95508), which is the same law with a cheaper estimator |
| Net frame minus digit + rating-aggregate blocks (V41) | V18's frame with 37 tree-centric columns dropped after the parent's builders | CV −0.00002 / LB 0.96104, folds 4/5 ahead | ⚠️ Neutral both ways: the RealMLP never used them; the frame is closed in both directions |

**V43 measured row (2026-10-24, CV + LB confirmed)** (`LEDGER`)

| Feature / axis | Construction | Importance % | Impact | Status |
|---|---|---|---|---|
| `_rel_<rating>` ×13 (V43) | `r - _mean_excl0` where r > 0, else 0 — each rating re-expressed against its own respondent's mean rated score | n/a (net arm; no gain importances) | **CV −0.00005 / LB 0.96101, folds improved 1/5** — negative, not a tie | ❌ Closed: the last un-tried transform family; the net does not represent the row mean (V41), so re-referencing cannot add |
