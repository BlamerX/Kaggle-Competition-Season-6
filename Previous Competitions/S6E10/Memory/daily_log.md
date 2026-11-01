# S6E10 Daily Log

> **⚠️ RULES:**
>
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

### 28-10-2026 — continuation: V55 ran and lost — the net memorises the real rows (0.9968 in-sample) and none of it transfers, which closes the last in-house axis

- **Goal**: land the third Kaggle attempt of the originals-as-rows arm and apply the gate as pre-declared.
- **Score**: **LB 0.96094** (owner-confirmed) / **CV 0.96117**, gap **-0.00023** — V18's frame constant again.
  Folds 0.96126 / 0.96037 / 0.96205 / 0.96147 / 0.96106, mean 0.96124 ± 0.00055; pooled − mean = -0.00007, ordinary.
  Against V18 (0.96129 / 0.96106): **ΔCV -0.00012 pooled = -2.1 rulers** (-0.00005 on fold means), **ΔLB -0.00012**,
  **folds improved 1/5** (only fold 4, +0.00005). Total 19.4 min (prep 23 s + fit 144 s/fold) — inside the 19-25 min estimate.
- **The instrument did its job, and that is what turns a loss into a finding**: fold-1 printed
  `in-sample AUC - appended rows 0.99678 | competition fit rows (100k probe) 0.96594 | held-out 0.96126`. Under the
  pre-declared reading the appended rows were unambiguously inside the fit — in fact the net learns them *far* better
  than it learns competition rows. So the -2.1 rulers is **not** "the rows were ignored" and **not** a feature-representation
  failure either: the 289-column frame already represents the real regime better than the synthetic one. It is a
  **transfer** failure — memorising a 4.4%-by-count minority of cleaner rows does not move predictions on the
  competition distribution. V51's appendix hypothesis (the real-data regime as the last untapped inductive bias) is
  refuted on measurement, and with it the premise that any column family is still waiting to be found by mixing in real data.
- **Every guard was exercised and none of them lied**: overlap 129,880 → 129,859; R 103,887 / A 25,972; **0 of 25,972**
  A predictor vectors shared with R; coverage **1,241 of 25,972 = 4.78%** dropped (FD 1.58%, dep-delay 2.04%, arr-delay
  2.32%) reproducing the local pre-run pass **digit-for-digit**, kept-row pos rate 0.4344 → 0.4346; fit frame 584,439
  rows with positives 0.4432 vs 0.4436 competition-only; `n_epochs` read back 3 from the fitted object; EV/rating r
  0.52-0.98 identical to V18's run of record; 1.23 GB transformed estimate, peak reserved 3.20 GB (V18 3.10);
  best epochs [3,3,3,3,3] — V18's expected signature, so starvation is not the explanation. The EV build cost 215 s
  against V18's 115 s, exactly the declared price of fitting the 13 aux-EV producers twice for leakage safety.
- **Gate verdict**: fails three of four criteria outright — solo CV 0.96117 is 0.00023 **under** the ≥ 0.96140
  member-strength bar, ΔCV is negative, folds 1/5. The paired test then ran offline on the archived vectors
  (`gate_v55_calibrated.py`, temp): **dAUC -0.000119 = -2.09 rulers, full-row bootstrap z -3.63** — and the estimator was
  calibrated before being trusted: the same code reproduces the ledger's recorded +3.38 for v50-vs-v49 at **+3.26**. The
  analytic paired DeLong gave z -108 with an SE **30× smaller** than the bootstrap's, and the same 30× overstatement for
  the known pair (analytic +159.7 vs recorded +10.56), so it is quoted as a floor only. **The leg bar was also measured
  rather than relied on: trio + v55 = 0.961471, -0.000114 CV, 0/5 folds improved.** Ranking the four candidate legs by
  damage: v52 -0.000076, v55 -0.000114, v53 -0.000239, v54 -0.000937 — the appended-rows vector is the *second least
  harmful* leg this pool has ever produced and it still lowers the blend. **Not adopted, no board slot, not a leg.**
  V51 **0.96113** remains the submission of record.
- **Timing**: 19.4 min of GPU on the third attempt; the two failed attempts cost ~4 min and ~2 min. Three sessions
  consumed by one arm is the honest bill for authoring a data-regime axis without a local runtime — and the local dry
  run added the same day found two of the three faults before any GPU time was spent.
- **Status**: the in-house route list is now **empty on measurement, not on imagination** — estimator families (V53/V54),
  feature selection (V52), composition in equal-weight and weighted form (the withdrawn V55 leg test + K9 + today's
  leg census), and the data regime (V55) have all been tried and lost with calibrated instruments. What remains needs
  resources outside this repo: the public-OOF tier (K3, task 28) and any external vectors the owner can attach on
  Kaggle — that is the only tier with a *measured* ceiling above ours (kratosyan's 119-member stack OOF 0.962048 → his
  LB 0.96167). `v55_oof.csv` and `v55_sub.csv` are archived, and the arm file now sits in
  `Previously trained files/Archieve/` (hash-identical move, verified before committing).

---

### 28-10-2026 — continuation: V55 repaired by execution — the dry run found three faults the audit could not, and the coverage cap is now measured, not guessed

- **Goal**: "see the correct implementation and fix all the errors" — so stop inspecting the arm and run it:
  `dryrun_v55.py` (temp) executes the whole file on synthetic frames with stubbed torch/xgboost/PyTabKit, which
  reaches every runtime path a second before a Kaggle session costs one.
- **Attempt 2 on the record before the fixes**: the second Kaggle run died at
  `twin Departure Delay in Minutes_cat_: 531 of 25,972 augment rows (2.04%)` — a **2% guard I had invented**. The fix
  was made by design, not by widening my guess: rows the pool's twin codes cannot express are **dropped and counted**,
  the twins function now asserts its own **post-condition** instead of a threshold, and 10% survives only as "the
  design, not the threshold, is wrong". Refusing the wider cap mattered: the old path sent an unencodable long delay
  to `categories[0]`, i.e. it dressed an extreme delay as a zero.
- **Fault 1 (caught by the dry run, would have been silent)**: `competition-row idempotence [teacher]: max |delta|
  3.504e-01`. The arm refit the teacher for the competition rows a *second* time after threading the A frame, so the
  parent-parity guarantee rested on bitwise XGBoost reproducibility and burned an extra 600-tree fit. Restructured to
  **one pass per producer** over all three frames; the in-run snapshot loop is deleted (there is no second call to
  compare against) and the equivalence is proven offline in `check_v55_leak.py` tests 3/4/5.
- **Fault 2 (a real crash on the real data, caught by the dry run)**: `A frame schema differs: missing [] extra []`.
  Empty sets — the *order* differed, because `data.csv` is not in the competition column order and V18 only ever
  selected by name. A is now re-expressed in the competition order before the per-fold check, so that check tests the
  pipeline instead of the input file's layout. The fold loop had never executed before this; it now runs all 5 folds,
  validation competition-only, 285 columns, exactly two output files.
- **Fault 3 (an instrument, not the arm)**: my first route-profile idempotence test **failed on the arm for a reason
  of its own** — `np.array_equal` is False wherever `_route_*_std` is NaN (single-member Flight-Distance groups carry
  NaN in V18 too), so two *identical* calls compared as different and the test had zero resolution. Fixed the way this
  campaign requires: NaN-aware compare, plus a control that runs the identical call twice and prints what the naive
  compare reported. It now passes, with the forbidden variant (A inside the pool) firing at delta **2.900e+01**.
- **Guard honesty**: fold 1 read `n_train_rows` / `n_features_in_` off the fitted object — attribute names **no
  archived arm of ours ever used** (grep), so the banner claimed a read-back the library may not offer. Replaced by
  `_read_held` (V53's mechanism, proven against a real PyTabKit object) for `n_epochs`, plus a measurement that
  resolves the arm's actual question: in-sample AUC of the appended rows against the competition fit rows and the
  fold's held-out level, with the read pre-declared (near fit-row level = they were fitted; near held-out = they were
  not; originals are a different regime, so a middle reading is evidence to weigh, not a pass/fail).
  `fill_continuous_nan` also returns a literal 4-tuple now, so `check_arity.py` can size it — superseding the previous
  entry's "an AST cannot size a list" caveat — and its optional fourth frame is gone because no caller passes None.
- **Measured, not guessed, on the local frames** (`measure_v55_coverage_real.py`, read-only pandas, no fit): union
  unrepresentable share of A = **1,241 of 25,972 = 4.78%** — Flight Distance 1.58%, Departure Delay **2.04% (which
  reproduces the Kaggle trace exactly, so the probe is faithful)**, Arrival Delay 2.32% — kept rows' pos rate
  **0.4344 → 0.4346**, R∩A shared predictor vectors **0**. The cap is 10%, so the third attempt does not stop at the
  coverage step, and the surviving augmentation is not a label-biased subset. Docstring and function now carry these
  figures instead of the single-column guess; A's contribution to the fit is **24,731 rows = +4.4%**.
- **Verification chain, all green**: dry run completes end-to-end and writes exactly 2 files · `check_v55_leak.py`
  **8/8 including controls** · `check_v55_coverage.py` **5/5** · `check_arity.py` **0 problems** · the three parameter
  dicts **AST-identical to V18** (AUX_EV 12/12, REALMLP 31/31, TEACHER 12/12) · docstring **35 lines** at the cap ·
  max line **121**, the house ceiling V18 and V53 also use.
- **Timing**: local CPU only, no GPU consumed. The two failed attempts cost ~4 min and ~2 min of the owner's session.
- **Status**: `Arms/S6E10_V55_RealRowsAugment.py` is **ready to run** and the owner must **re-download it** — every
  one of these faults is absent from the copy already on Kaggle. Task 50 unchanged; admission bar unchanged (solo CV
  ≥ 0.96140 **and** dCV ≥ +0.00005 **and** ≥4/5 folds **and** DeLong z > 3). No score entry is written for a failed run.

---

### 28-10-2026 — continuation: V55's first Kaggle run died at line 890 — one threaded function still returned the parent's 2-tuple

- **Goal**: read the failure, fix it, and add the check that would have caught it before an hour of GPU time.
- **What happened**: the run reached the EV step and raised `ValueError: not enough values to unpack (expected 3, got 2)`.
  `add_aux_ev_features` had its loop threaded for the augment frame but still ended on V18's
  `return train_df, test_df`. Everything before it was healthy and is now on the record: overlap removal
  129,880 → **129,859 clean**; split **R = 103,887 / A = 25,972**; **0 of 25,972 A predictor vectors shared with R**, so
  no appended row is statistically represented in the fit source it is scored by; EV/rating correlations identical to
  V18's run of record (0.52–0.98). The idempotence prints never got a chance to run — the crash came four lines earlier.
- **The real lesson is that `py_compile` is blind to this class.** A return-arity mismatch is a runtime fact, so
  `check_arity.py` (temp) now walks every threaded call site and compares the unpack count against the return paths the
  function can actually take. **Calibrated the way this campaign requires: run against the broken committed version it
  flags exactly `line 890: add_aux_ev_features(...) unpacks 3 <- function returns [1, 2]` — the line that died on
  Kaggle; run against the fixed version that problem is gone.** Two remaining flags are understood, not ignored: the
  dual-arity functions (2-tuple without an augment frame, 3 with) are by design, and `fill_continuous_nan` returns a
  *list*, which an AST cannot size.
- **A second latent defect, found while fixing the first**: the twin column for A was built as
  `pd.Categorical(codes.astype('float64'), categories=<int64 levels>)`. Codes and categories would not match, so **all**
  of A's twin values would land as NaN and then be filled to `categories[0]` — the arm would train with every one of its
  17 categorical twins erased on the appended rows and print nothing wrong, because the unseen count is taken *before*
  the construction. Now the codes are cast to `int64` with `-1` reserved for unseen, and the constructed column's NaN
  count is asserted equal to that count. Proven on synthetic frames: `after_nan == unseen` (8 of 1,600), and the
  falsifier — float-vs-int codes produce **1,600 of 1,600 NaN**, i.e. the assert trips instead of the run silently
  proceeding.
- **Full calibration re-run after the fixes (synthetic frames, no competition data, no xgboost)**: leak absent
  (A label flip moves its org stats by **0.000e+00**), falsifier fires on the wrong fit source (**1.102e-01**),
  competition idempotence holds, twin guard aborts at 5% / passes at 0.5%, twin values match, and the dtype falsifier
  fires. **6/6 PASS.**
- **Timing**: ~4 min of the Kaggle session consumed (data load + FE + the first teacher/EV fits) before the abort.
- **Status**: fixed and re-audited locally; **the owner must re-upload the current file** — his copy predates even the
  EV banner relabel, so a stale notebook is a real risk. Task 50 unchanged: V55 awaits a clean 5-fold run at
  ~20-25 min. No score entry is written for a failed run.

---

### 28-10-2026 — continuation: K9 closes weighted composition, and V55 (originals-as-rows) is authored for Kaggle

- **Goal**: answer the composition question in the form the rules do allow — a stdout probe, not a version — and build the last untried in-house arm.
- **K9 (`Probes/S6E10_K9_WeightCensus.py`, local CPU, wrote nothing, 21 cells on all 699,635 rows)**:
  `(1-w)·logit(trio) + w·logit(leg)` for legs v53/v54/v52, w ∈ {0.05…0.35}, same pre-declared bar.
  **Zero cells clear; 4 of 21 are positive at all; the best is +0.000013** (v53 at w=0.05 — 0.26 of the bar, 5/5 folds,
  DeLong z +2.85: sign-consistent, and a fifth of a ruler in CV). v54's best is +0.000010 and it decays to −0.000450 by
  w=0.35; v52 is negative everywhere. Control reproduced `v51_oof` again (1.17e-13).
  **The null validates its instrument**, which is what makes a stop verdict admissible: each leg's gain falls
  monotonically after its peak, with grid spreads of 2.1 rulers (v53) and 8.1 rulers (v54) — a bar-sized gain would have
  shown. v52's curve is flat (0.6 rulers) and unresolving by itself, but the equal-weight test already closed it.
- **What that settles**: **composition of our own pool is closed in both forms the owner's lift permits** — equal-weight
  (three legs, 0/5 folds each) and weighted (21 cells, none clearing). The pool's own CV ceiling (~0.96160) is already
  where the trio sits, so no recombination reaches the ~0.9621+ tier the public board occupies. No widening request is
  worth the owner's time now: the best cell is 4× below the bar.
- **V55 authored (`Arms/S6E10_V55_RealRowsAugment.py`, ~20 min GPU, awaits the owner's Kaggle session)**: V18's clone with
  one axis — the overlap-cleaned original split **R 80% / A 20% (seed 9121, stratified)** and **A appended to each fold's
  training rows**; estimator, params, frame, folds and validation are V18's unchanged. `REALMLP_PARAMS`,
  `TEACHER_PARAMS` and `AUX_EV_PARAMS` are **AST-identical to the parent (0 differing keys)**, exactly as the config-dict
  rule demands — a banner can lie, a dict cannot.
- **The leakage design, which is the whole arm**: every producer that can see an A row's own `satisfaction` label is
  refitted on **R only** before transforming A (org-statistics, the teacher, and the 13-EV block — V11's own guard is
  "no row's EV may come from a model that saw that row", and A rows *are* original rows). The per-fold target encoder is
  fitted on competition fit-fold labels only and merely transforms A; medians and category vocabularies likewise come
  from the fit fold alone. Competition rows keep V18's exact features, verified by **idempotence snapshots** of the
  route/org/teacher columns taken before and after threading A (abort if any moves >1e-6), and the appended frame's
  schema is asserted equal to the test frame's column-for-column.
- **Guard calibration (synthetic frames, no competition data, no xgboost needed)**: `check_v55_leak.py` (temp) exec'd the
  arm's own AST-extracted functions and ran four tests — **leak absent** (flipping an A row's own label moves its
  org-stats by 0.000e+00), **the falsifier fires** (fitting stats on R+A instead moves that row's mean by
  **1.102e-01**, so test 1 could have failed and did not), **idempotence** holds, and the twin guard **aborts at 5%
  unseen and passes at 0.5%**. ALL PASS.
- **Honest prior, stated before the run**: the external paired test (Sachith7) found appending originals **negative in
  6/6** comparisons on a 180-tree LightGBM, and this episode's own rule of the day says a leg below member strength
  cannot pay — so V55's admission is raised to **solo CV ≥ 0.96140** (v36/v44's level), not merely "beats V18". Below
  that, the arm is a tie and ties are now known to be useless here.
- **Status**: V55 ⏳ awaiting a Kaggle GPU session; K9 ✅ closed and recorded. Task 41 (author) is done, task 50 tracks
  the run. 🏆 **V51 0.96113 is still the submission of record**, and after V55 the only door left with public-scale
  headroom behind it is the foreign-OOF tier, which needs the owner's Kaggle session and a ruling the lift does not cover.

---

### 28-10-2026 — continuation: V55 withdrawn as mis-numbered — it should have been a probe, and two ledger rules were broken

- **Goal**: answer the owner's challenge — what was the point of a version that cannot ship? — and repair the record.
- **The rules I broke** (read from the memory files, not from recall):
  1. **`feedback-check-queue-against-own-falsifiers`, clause (3)**: "if the precondition is unmet but testable cheaply,
     propose a **stdout-only probe (K#)** … **not a version**." I already held the measurement that decided it — v53 at
     10 rulers under the members, v54 at 45, v52 at 0.3 with logit-corr 0.99918 — and an equal-weight fourth leg cannot
     clear a **+0.00005** bar by diluting three stronger members. Authoring it as `V55` spent a version number on a
     measurement whose outcome its own arithmetic predicted.
  2. **Ledger admission rules**: `training_logs.md` rule 1 ("only update after Public LB score is available") and
     `feature_engineering.md` rule 1, which is the strictest of the four. V55 has no LB and no vector, so it qualified
     for **`daily_log.md` only** (its rule 1 allows "experiment OOF available"). I had put a `### Version 55` block in
     the training log and a V55 row in the FE table.
- **Repairs made**: `Arms/S6E10_V55_LegExtensionBlend.py` deleted by `git rm` (still in commit `3e4f4a9` if it is ever
  wanted); the `### Version 55` block removed from `training_logs.md`; the FE section retitled to V53–V54 and narrowed
  to the frame claim. The measured numbers are not deleted — they move here, where their rule allows them.
- **Experiments (the measurement, retained here)**: equal-weight `mean(v27,v36,v44)` + ONE leg, closed list, gate fixed
  before the run (gain ≥ +0.00005 AND ≥4/5 folds AND DeLong z > 3; leg precondition `solo ≥ 0.9600`, declared in V53's
  and V54's own headers). Control verified: recomputing the trio reproduces `v51_oof` to **9.326e-14** in logit space.

| Candidate | Leg solo CV | Pooled | Gain vs V51 trio | Folds | DeLong z | Bootstrap z | max logit-corr | Admitted |
|---|---|---|---|---|---|---|---|---|
| +v53 (TabM) | 0.960667 | 0.961546 | **-0.000039** | 0/5 | -2.00 | -2.34 | 0.98184 (distinct) | ❌ |
| +v54 (DCN-V2) | 0.958693 | 0.961379 | **-0.000206** | 0/5 | -6.47 | -6.50 | 0.98267 (distinct) | ❌ precondition failed |
| +v52 (drop-frame) | 0.961295 | 0.961565 | **-0.000020** | 0/5 | -2.60 | -2.66 | 0.99918 (collapses) | ❌ |

  Ran twice (185.3 s, then 180.4 s) with **every statistic identical digit-for-digit**, so this is deterministic
  arithmetic, not a sample. No vector was written; **V51 0.96113 remains the submission of record**, and the number V55
  is free again for an arm that can actually ship.
- **Key Learning (the campaign's last one)**: **decorrelation is necessary and not sufficient.** Both new families are
  genuinely distinct channels — max logit-corr 0.982, nowhere near the 0.999 collapse rule that sinks v52 — and both
  still *lower* the blend, with the damage tracking the leg's own shortfall: 10 rulers under costs 0.7 rulers, 45 under
  costs 3.6. An equal-weight leg buys independence by diluting three ~0.9614 members into four. A future composition
  therefore needs a leg **near member strength**, and no admission test based on correlation alone should be trusted
  again. ⚠️
- **Status**: V53 ❌ not adopted, V54 ❌ not adopted, V55 withdrawn (never a valid version). 🏆 V51 0.96113 stands. The
  forward plan is in the next entry; nothing further enters `public_scores.md` without a confirmed LB.

### 28-10-2026 — continuation: the forward plan, priced against the remaining budget (3 days, ~5 GPU-h)

Deadline is 31-10-2026 23:59 UTC. Recorded spend today: V53 140.3 min + V54 18.6 min ≈ 2.7 h of the ~8 h weekly T4
allowance, so ~5 h remain in the week and every option below is priced against it. Ordered by expected value, not by
comfort.

1. **K9 — weight census, CPU, ~3 min, stdout-only, writes nothing.** The rule I just broke, applied properly: before
   any further composition *version*, measure whether **any** single weight w ∈ {0.05 … 0.30} of v53 / v52 / v54 mixed
   into the V51 trio clears the standing bar (pooled ≥ +0.00005 AND ≥4/5 folds AND DeLong z > 3 vs `v51_oof`). If none
   does, equal-weight dilution was not the obstacle and the composition phase is formally closed — say so and stop
   spending numbers on it. This also answers the weighted-leg door without needing a widening of the owner's lift first.
2. **V55 (fresh number) — original rows as training rows, ~20 min GPU.** The last mechanism the code audit found
   genuinely untried in-house: originals appear only as teachers, aux-EV producers and statistics ([V18 fit path
   `:791`], never jointly with competition rows in the final estimator. Design constraint from the audit, non-negotiable:
   split the cleaned originals into reference **R (80%)** and augmentation **A (20%)**, fit every original-derived
   producer on R alone, transform A identically to competition rows, and keep validation/OOF competition-only — otherwise
   the appended rows read their own label through `org_stats`/teacher columns, which is the in-sample leak the K8
   cleanup already exposed. Prior against it: Sachith7's paired kit found appending *negative in 6/6* comparisons
   (−0.000446) on a 180-tree LightGBM. Prior for it: that baseline is far weaker than our net, and the V51 appendix names
   this data regime as the one thing that could produce an inductive bias the pool does not contain.
   **Admission is now arithmetic, not hope**: today's dilution measurement says a leg must arrive at **≈ member strength
   (0.9614+)** to help a blend, so this arm ships only if it beats V18's 0.961287 outright — a 0.9606-grade result is
   pre-declared worthless as a leg, and I will not re-test that.
3. **TabM at a converged budget — ~4.4 h GPU, only if 1 and 2 leave both time and quota.** Fold fits ran 26 min at the
   30-epoch ceiling that all five folds hit, so 60 epochs is ≈52 min/fold ≈ 4.3-4.5 h for 5 folds: it would consume the
   rest of the week and cannot be combined with option 2. It is the only *unmeasured* family ceiling we have, and its
   honest expected value is a leg that must again clear 0.9614 to matter.
4. **The documented headroom, blocked on the owner, not on me: foreign OOF pools (K3 / task 38).** The public board reads
   0.96213-0.96427 and the recorded stack evidence is OOF 0.962048 → LB 0.96167 — i.e. **+0.0008 of headroom that no
   in-house mechanism has touched**, and this box has no Kaggle access (`kaggle` module and `~/.kaggle` both absent) and
   no browser path to the assets. It needs the owner's session, plus his ruling on third-party vectors, which the
   composition lift does not currently cover.

**If option 1 shows no weight clearing and option 2 lands below 0.9614, the correct end is V51 0.96113 — a stop chosen on
measurement, not on running out of ideas.** What that would leave genuinely untried, recorded so the ledger does not
overclaim: a converged TabM, the foreign-vector tier, and (if the lift is widened) weighted composition.

**Housekeeping**: `Arms/` no longer exists — the three run arms were moved by the other writer into
`Previously trained files/Archieve/`, which is where archived arms live. The next authored arm recreates `Arms/` for its
pre-run home, as V52 did.

---

### 28-10-2026 — continuation: V53 and V54 scored, V55 tested the legs and shipped nothing — the campaign ends on a dilution rule

- **Goal**: land both new-family runs and settle, as the final planned measurement, whether either extends the champion trio.
- **Experiments**: V53 TabM_D (Kaggle T4, 140.3 min) → **CV 0.960667 / LB 0.96072**, gap **+0.00005**, folds 0.96094 / 0.96013 / 0.96151 / 0.96082 / 0.96029. V54 DCN-V2 (18.6 min) → **CV 0.958693 / LB 0.95965**, gap **+0.00096**, folds 0.95886 / 0.95782 / 0.95979 / 0.95909 / 0.95846. V55 (local CPU, 185.3 s, no fit) scored the closed candidate list against the pre-declared bar.
- **Verdicts**: both families fail their own single-model bars, and **no candidate cleared V55's gate** — +v53 −0.000039 (0/5 folds, z −2.00), +v54 −0.000206 (z −6.47, and it fails its own `solo >= 0.9600` precondition), +v52 −0.000020 (z −2.60). **V55 wrote no vector, so V51 (0.96113) remains the submission of record.** The control was verified, not assumed: recomputing the trio reproduces `v51_oof` to **9.326e-14** in logit space.
- **Key Learning — decorrelation is necessary and not sufficient.** The two new families are genuinely distinct channels (max logit-corr **0.982**, nowhere near the 0.999 collapse rule that sinks v52) and both still *lower* the blend. The damage tracks the leg's own shortfall almost exactly: 10 rulers under the members costs 0.7 rulers, 45 rulers under costs 3.6 rulers. An equal-weight leg buys independence by diluting three strong members into four — so a leg must arrive near member strength. This is the reverse of the S6E9 precedent (TabM as best non-tree leg, +0.000093) because there the pool was tree-dominated and the members were weaker.
- **Second Key Learning — the gap was never a constant.** V53 (+0.00005) and V54 (+0.00096) are both **positive**, where every RealMLP net sits at −0.00018 to −0.00043. The observed spread across this campaign's arms is ~**0.0014**, so the calibrated `−0.00046 ± 0.00001` was an artifact of calibrating on three closely related blends; the honest statement is that the CV→LB gap is family- and mix-specific and unpredictable for a family that has not already been submitted.
- **Third Key Learning — what the two runs prove about their own instruments.** V53's best epochs were **[30,30,30,30,30] against a 30 ceiling**: early stopping never fired, so TabM is **budget-limited, not measured** on this frame — its loss must not be read as the family's ceiling. V54's were **[3,3,2,3,2] of 50**: it peaked in three epochs and degraded for the whole patience window in all five folds, i.e. the family genuinely cannot exploit this frame, and at 2.8 min/fold against a 45 min ceiling that verdict was not budget-limited. V53's canary also validated itself: 26.1 min/fold projected 131 min against 137.9 actual.
- **Integrity notes**: `v54_oof` ships **110 rows at exactly 0.0 and 86 at exactly 1.0** (0.0280%, largest tie group 110) and `v54_sub` has 4 — the first boundary probabilities this campaign has produced; V55 clips them to |logit| 13.82 and prints the count rather than pretending they are interior. V53's variant-discovery print filtered lowercase `'abm'`, so it printed an empty list although `TabM_D_Classifier` imported fine — cosmetic, recorded so nobody reads it as a missing-class alarm.
- **Timing**: 140.3 min + 18.6 min GPU; 185.3 s local CPU for the composition test (the paired bootstrap is the cost, not the averaging).
- **Status**: ❌ V53 and V54 not adopted; V55 not shipped. 🏆 **V51 0.96113 is the final submission of record.** Four things remain unmeasured and were never claimed closed: TabM at a converged epoch budget, a weighted (non-equal) composition — outside the owner's lifting of the composition law — the original-rows-as-training-rows mechanism (task 41), and the public foreign-vector pools (task 38).

---

### 28-10-2026 — continuation: V54 is reassigned to DCN-V2, and the composition arm moves to V55

- **Goal**: the owner directed that the next trained arm be the DCN-V2 family, with the number V54, and that the leg-extension blend arm be deleted.
- **Experiments**: none run. `Arms/S6E10_V54_LegExtensionBlend.py` is deleted (`git rm`, still recoverable from `dc9cff1` if it is ever needed again), and [Arms/S6E10_V54_DCNv2Family.py](../Videos/GitHub/S6E10/Arms/S6E10_V54_DCNv2Family.py) is authored as V18's clone with the estimator family as the only axis.
- **Nothing measured is lost**: the blend arm's two results stay in the record above — the incumbent trio reproduces `v51_oof` to **9.3e-14 in logit space**, and `+v52` measured **−0.000020, 0/5 folds, DeLong z −2.60**, failing its pre-declared bar. **V55 inherits that exact gate** (solo ≥ 0.9600, candidate must beat V51's recorded CV, equal-weight logit-mean only, writes nothing if neither candidate clears), so renumbering does not soften it. V53's own docstring and RESULTS were updated to name V55, so no archived arm points at a deleted file.
- **Model provenance, which is the point of the owner's "make it ours"**: the DCN-V2 is our **own hand-written PyTorch** implementation, copied from S6E9 V13:115-211 with every hyperparameter unchanged (cross 4, low_rank 64, experts 4, dnn [256,128], embedding_dim 16, dropout 0.2, AdamW lr 1e-3, wd 1e-5, batch 4096, chunked inference 8192, max 50 epochs, patience 10 on val AUC with best-checkpoint restore). Unlike V53, this family is not a library class, so there is no argument-name risk — the names are ours by construction.
- **Two deliberate deviations from the borrowed script, both declared in the file**: (i) `StratifiedKFold(5, shuffle, seed 42)` instead of V13's plain `KFold`, because the house partition is frozen and every recorded arm is scored under it; (ii) `pos_weight` is computed from the **fit fold**, where V13 computed it on all training rows — the campaign takes every label-derived statistic from the fit fold only. The class weight also means this arm differs from V18 in the loss as well as the architecture, and that is stated rather than inherited silently.
- **Guards added that no prior DCN/TabM arm had** (verified by grep across the episodes): a fold-0 column-split positive control (21 categorical columns expected = 4 raw + 17 twins, aborting if dtype detection loses or misreads them), a structural read-back on the **constructed** net (embedding count, cross-layer count, `input_dim`, `output.in_features`, cards ≥ 2), a **learning floor** of 0.90 on fold-1 val AUC with the epoch trace printed on failure, and a fold-1 wall-time canary at 45 min/fold (S6E9 V29's mechanism) whose fallback is a declared smaller `DCN_CFG`, never a mid-run change. Calibrated without torch (absent on this box) by replaying the net's own arithmetic: the invariants hold on the real frame shape (~468 inputs from 268 numeric + 200 embedded dims) and **fire** on the known-bad cases — 0 or 20 categorical columns, and a degenerate card of 1.
- **Expectation, read honestly from the prior episodes**: S6E9 V13 ran only **14.4 min** but took **0.94490 OOF**, below that episode's TabM (0.94585) and its XGB control (0.946076) — its ledger says it "did not outperform trees or TabM"; S6E4 V39 called it that episode's "Best NN" at 0.96986. So V54 is budgeted as a **cheap, decorrelated leg for V55**, not as a solo winner, and its own single-model bar (CV ≥ 0.961337, z > 3, ≥4/5 folds) is expected to fail.
- **Timing**: V54 ~30-60 min on one T4 (289 columns vs V13's ~80, so roughly 3.6× V13's 14.4 min, with the canary deciding). V53 unchanged.
- **Status**: two arms await the owner's Kaggle GPU sessions (tasks 45 and 46, V53 then V54); V55 is owed once their vectors exist. No score rows exist for V53/V54 and none will be written before CV **and** LB are both in hand. Best recorded LB remains **V51 0.96113**.

---

### 28-10-2026 — continuation: the previous episodes already ran TabM, and that rewrote V53's config

- **Goal**: answer the owner's prompt to check how we implemented this in earlier competitions before spending a GPU session on V53.
- **Experiments**: read-only search of `C:\\Users\\BlamerX\\Videos\\GitHub\\Kaggle-Competition-Season-6\\Previous Competitions\\` (S5E12, S6E1, S6E2, S6E3, S6E4, S6E5, S6E9 — each with its own `Memory/` ledgers). **Five archived TabM arms exist**: S6E3 V9 / V21 / V71, S6E4 V8:145-163, S6E9 V6:179-197 and V29:140-156, all using `from pytabkit import TabM_D_Classifier`. S6E5 V14's TabM and S6E3 V63's are hand-written implementations, not PyTabKit.
- **Key Learning (the one that mattered)**: **the Arg-Name Law's source is our own archive, and I had used a public kernel instead.** Every prior TabM arm passed the same 15 keys — `device, verbosity, arch_type='tabm-mini-normal', tabm_k, num_emb_type='pwl', d_embedding, d_block, n_blocks, dropout, batch_size, lr, n_epochs, patience, weight_decay, random_state` — and **none** of them used `wd`, `tfms`, `embedding_size`, `max_one_hot_cat_size`, `use_early_stopping` or `val_metric_name`. V53 as first drafted passed exactly those RealMLP names, with `n_epochs 3` and early stopping **off**, which is the failure mode the law exists to prevent: it would either die at construction or fit an undertrained model and then be read as "TabM loses".
- **V53 rewritten** to S6E9 V6's config verbatim (`tabm_k 16, d_block 256, n_blocks 3, batch 1024, lr 1e-3, weight_decay 1e-3, n_epochs 30, patience 10, d_embedding 16, dropout 0.2`) on the unchanged V18 frame. Verified statically: the dict's key set now equals S6E9 V6's with no extras and no omissions, the file compiles, and the guard lists cover every key.
- **Budget honesty**: prior timings are the cost evidence — S6E9 V6 took **43.9 min** on 83 columns; S6E4 V8 took **295 min**; S6E9 V29 needed a top-120-of-312 column budget and a fold-0 canary. Our frame is 289 columns, so V53 now carries a **75 min/fold canary** (the V29 mechanism) whose declared fallback is V29's own wide-frame rung (`k 4 / d_block 128 / n_blocks 2 / batch 4096`) as a re-run, never a mid-run change. V18's 12 GB analytic abort was **removed rather than kept looking strict**: it was calibrated for RealMLP's one-hot/embedding transform, which this arm replaces, so it could fire or stay silent for the wrong reason; the estimate is still printed, labelled as the parent-recipe number, and the wall-time canary decides.
- **Prior verdicts, which set the expectation honestly**: S6E9's TabM **lost its control by −0.00043** with rank correlations 0.995-0.996 and the episode closed the family as "weak and correlated"; S6E5's TabM trailed RealMLP; S6E3 V71 failed and concluded "TabM is bound purely by the feature space, not hyperparameters", with **k=32 optimal and k>32 dead**. But the same S6E9 record says V6's TabM was the **"best non-tree leg"**, blending with V23 for **+0.000093 at z +7.5** — which is precisely the role V54 tests here. So a losing solo V53 is the expected case and still worth having.
- **Second families, priced**: FT-Transformer ran in S6E3 (LB 0.91633, weakest NN, **692 min**), S6E5 (V8) and S6E9 (V7: 0.94568 LB, 159 min, lost to TabM); DCN-V2 in S6E3 (worse), S6E4 (0.96986) and S6E9 V13 (**14.4 min**, 0.94568/0.94490, "did not outperform trees or TabM"); NODE in S6E3/S6E4/S6E5 was the recurring cheap success (S6E4 V21 0.97720, called "a prime candidate for future ensembles"). If TabM's leg fails, **DCN-V2 at ~15 min and NODE are the cheap next doors — not another 700-minute transformer**.
- **Timing**: no compute run today beyond static AST/compile checks and synthetic guard calibration (a build that drops the architecture keys fires; `k` drift fires; the good object is clean).
- **Status**: V53 is now sourced from our own proven recipe and is ready for a Kaggle GPU session (task 45). V54 is unchanged — it reads whatever `v53` produces. My earlier entry today mis-attributed the configuration's provenance; that entry stands as the record of what was believed then, and this one corrects it.

---

### 28-10-2026 — V53 (TabM family) and V54 (leg-extension blend) authored; V52's open blend question is now measured

- **Goal**: act on the owner's ruling that the TabM route must be **our own run, not their vectors** — V53 fits the family, V54 blends it.
- **Why that is the better route**: it removes the two blockers found yesterday — this box has no Kaggle access (`kaggle` module and `~/.kaggle` both absent), and the published TabM vectors carry no embedded IDs/folds, so their row order and exporter provenance cannot be certified from here. A leg we fit ourselves inherits our frame, our seed-42 folds and our IDs.
- **V53 (`Arms/S6E10_V53_TabMFamily.py`, GPU, unrun)**: cloned from V18 so the diff *is* the axis — the fitted object becomes `TabM_D_Classifier` (PyTabKit 1.7.3; the class name and argument names come from a run that executed on this build in this competition, not from guesswork). Frame, 5-fold driver, `n_epochs 3`, `batch_size 256`, early-stopping switch and the input-transform keys stay V18's; RealMLP's architecture/optimiser keys have no TabM counterpart, so the family's published defaults replace them (`tabm_k 8`, `n_blocks 3`, `d_block 512`, `num_emb_type pwl`, `lr 1e-3`, `wd .01`).
- **Guards, each from a recorded failure**: a pre-flight that builds the object **before any fit** and reads the config back (an un-named-but-forwarded kwarg is legitimate in PyTabKit, so "absent from the signature" is not the test — the object's own record is; `_read_held` was calibrated against synthetic objects including a known-bad `plr`-vs-`pwl` case, which fires); the fold-1 read-back on the *trained* model (K8 cost a GPU run measuring a net we do not ship); a hard abort if any of the 14 critical keys is unrecorded; V18's absolute GB guard extended with TabM's k-mixing first-block term, aborting rather than silently shrinking the frame; and `RESULTS` prints which k ran.
- **V54 (`Arms/S6E10_V54_LegExtensionBlend.py`, CPU, runs once v53 exists)**: incumbent = V51's trio, candidate list closed and pre-declared to `[+v53_tabm, +v52_dropframe]`, equal-weight logit-mean only — no weight search and no meta-model on saved OOF. Bar: gain ≥ +0.00005 **and** ≥4/5 folds **and** DeLong z > 3 vs `v51_oof`; if neither candidate clears it writes nothing and V51 stays the submission. It also closes V51's guard gap by requiring every `_sub` id vector to align row-for-row with `test.csv`.
- **Experiments (run today, read-only, no writes)**: the arm's own control was verified rather than assumed — recomputing `mean(v27,v36,v44)` reproduces archived `v51_oof` at **max logit difference 9.3e-14, rank correlation 1.00000000**. The available candidate `v27+v36+v44+v52` scored **0.961565 vs the trio's 0.961585 = −0.000020, 0/5 folds improved, DeLong z −2.60, bootstrap z −2.66** → **it does not clear**.
- **Key Learning**: **V52's blend value is no longer "unverified" — against the standing champion it is negative.** The census explains the mechanism: `v44~v52 0.99918` and `v36~v52 0.99885` cross K3's 0.999 collapse rule, so a fourth leg that is V44's near-twin re-weights the blend instead of adding information, and −0.3 of a ruler is what that costs. Correlation is the mechanism, not the admission test — the measured marginal is the test, and it agrees.
- **Timing**: V53 ~25-50 min on one T4 (V18 was ~18 min; k-mixing makes the real cost unmeasured until it runs). V54 ≈ 1-2 min CPU. Nothing was trained or downloaded locally; the only local execution was arithmetic on our own archived vectors plus source-level guards.
- **Status**: V53 ⏳ awaiting a Kaggle GPU session from the owner — it cannot be run from this box. V54 ⏳ blocked behind `v53_oof.csv`/`v53_sub.csv`. No score-ledger rows exist for either, and none will be written before CV and LB are both in hand. Best recorded LB remains **V51 0.96113**.

---

### 27-10-2026 — V52 result and limits of the exhaustion claim

- **Goal**: record V52 and distinguish a failed experiment from an exhausted search space.
- **Experiments**: user-reported LB **0.96102**; earlier full-row OOF recheck **0.961295**, ΔCV **+0.000008** vs V18, paired-bootstrap **z +0.32**, **2/5** folds improved, gap **−0.000275**, pooled-minus-fold-mean **−0.00001**. The single-model admission bar failed on magnitude and fold count; the bootstrap diagnostic also found no significant advantage.
- **Timing**: not supplied; the attachment was the script, not stdout. Fit widths and fitted-parameter read-backs are specified in code, not independently observed in a run log.
- **Key Learning**: the selected 81-column removal showed no improvement. This does not establish that all feature selection is futile, that predictions are identical, or that the CV→LB gap is invariant to every frame change. The gap differs from V18's by −0.000048.
- **Blend check**: the first V52 subset search timed out; the replacement was stopped because it did not reproduce the intended comparison (V52 could enter the baseline pool; the logit matrix was not initialized before use). Neither supplies a valid blend verdict. Any further computation belongs on Kaggle.
- **Static coverage audit**: the inspected final-estimator paths train on competition rows; original rows supply teachers/auxiliaries/statistics, not joint final-model training ([V18 fit path](../Previously%20trained%20files/Archieve/S6E10_V18_RealMLPFusionEV.py#L791), [V27 original-only teacher](../Previously%20trained%20files/Archieve/S6E10_V27_DuoTeacherOrig.py#L406)). Joint source-row augmentation remains untested, not proven beneficial. It needs a fresh paired control with the same source-reference subset and feature producers: added source rows must be held out of their own target-derived features, and competition validation labels must remain excluded. K2's failed ModernBERT recipe does not constitute a TabM or canonical FT-Transformer test.
- **Composition audit**: V49/V50/V51 ship fixed equal-weight means. The referenced weight-search source and optimizer diagnostics were not found; the recorded SLSQP-uniform claim cannot establish an optimum. A small predeclared weight grid and V52 add-one comparison remain unverified. Reproduce the archived baselines and validate IDs/OOF provenance first; do not reuse historical K3 unchanged, since it fits a logistic meta-model rather than performing the permitted fixed-vector averaging. Correlation and plausible solo AUC alone do not establish alignment or leakage-free provenance.
- **Status**: V52 not adopted as a single-model improvement. Best recorded public LB remains **V51 0.96113**, followed by **V50 0.96112**. Original-row augmentation is a verified coverage gap, not positive evidence of a gain. No new score or theoretical ceiling is claimed.

### 27-10-2026 — continuation: public-source audit changes the remaining priorities

- **Goal**: distinguish genuinely untested mechanisms from stale score claims before ending the campaign.
- **Experiments**: source/record inspection only; no new training, vector download or submission. The live [official public leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard) shows **0.96427 / 0.96352 / 0.96297 / 0.96276** at ranks 1–4, so the old 0.96213 plateau is not the current global public ceiling. Methods are unverified; public scoring uses approximately 20% of test rows and final scoring the remaining 80%.
- **Timing**: no new model runtime. Public notebook runtimes below belong to their publishers, not this campaign.
- **Existing assets, newly reverified**: [Deotte's v1 TFM dataset](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions) lists **11 OOF/test pairs**, nine foundation models plus raw-feature RealMLP/XGBoost; its card specifies 21 raw columns and seed-42 five-fold splits. These assets are not new discoveries. The positional NPYs lack embedded IDs/folds; exporter provenance, actual row order and applicable model-output licenses still require verification before use. A CC0 dataset label alone does not settle upstream restrictions.
- **New model candidate**: [Demidov's October 26 TabM notebook](https://www.kaggle.com/code/yekenot/ps-s6-e10-tabm-pytabkit), v1 / 362491780, has a visible Kaggle **public score 0.96052**, recorded OOF **0.96050**, and OOF/submission outputs. It uses TabM rather than our RealMLP recipe, without an original-data teacher. Its weaker solo score is not evidence of positive blend value; test that marginal before spending GPU time.
- **New positive evidence with an important restriction**: [Upadhyay's October 27 replay](https://www.kaggle.com/code/ashwinupadhyay/airline-oof-stacking-controls-gains-and-replay), v1 / 362818388, prints **17-group meta-CV 0.961794 → 21-group 0.961912**, positive on all five folds after adding four foundation-model groups. However, the equal means are **0.961619 → 0.961615**: the improvement belongs to a fitted logistic stack, not averaging. The author reports LB **0.96145**, but the notebook has no public-score chip and explicitly disclaims independent end-to-end nested validation. This supports investigating the base vectors, not copying the stack under the averaging-only rule or promising its gain.
- **Negative evidence changes the augmentation priority**: [Sachith7's paired kit](https://www.kaggle.com/code/sachith7/s6e10-does-original-data-help-a-paired-cv-kit/notebook), v2 / 358334358, prints six matched comparisons (two seeds, three folds; 180-tree LightGBM). Mean fold AUC is **0.957722 synthetic-only / 0.957276 appended originals / 0.958875 original-only teacher**. Appending loses **0.000446**, negative in **6/6** pairs; no submission was made. This does not settle our neural recipe with its existing teacher, but makes augmentation lower-priority than a cheap vector test.
- **Next test and stop rule**: on Kaggle CPU, first reproduce V50/V51 and verify candidate OOF/test provenance, exact IDs or independently established positional order, finite values and target orientation. Then compare one admitted foreign vector at a time, plus V52, against a frozen incumbent using a small predeclared nonnegative weight grid; no logistic/meta-model fit and no public-score-driven extrapolation. Report pooled and per-fold deltas and selection-aware limitations; correlation or fixed-vector bootstrap alone cannot prove improvement. If only noise-sized/unstable gains remain, retain the incumbent. A matched, leakage-safe source-augmentation pilot is a fallback, not an automatic V53.
- **Status**: research identifies remaining experiments, not a measured improvement. Current-recipe diminishing returns are supported; universal exhaustion is not. Binary foreign vectors have not been downloaded or certified, and no new version has been authored.

---

### 25-10-2026 — continuation: V52 authored — the first feature-selection rung written from a measurement

- **Goal**: turn the K8 verdict into the arm it authorises, with the bar fixed before the run.
- **Experiments**: none run yet. [Arms/S6E10_V52_InertFrameDrop.py](../Videos/GitHub/S6E10/Arms/S6E10_V52_InertFrameDrop.py) clones V18 byte-for-byte in recipe (AST diff of `REALMLP_PARAMS` against the parent: **0 differing keys** — the axis is the frame alone) and deletes the measured-inert families from every fold's fit frames: **digits 20 + org_stats 22 + tm_single_TE 22 + unmapped residual 14 + rating_aggregates 3 = 81 columns, 289 → 208**. The family predicates are copied verbatim from the census, so the drop set is the set that was measured, not a hand-picked list; the raw originals stay (inert as a block, but every other family is derived from them).
- **Guards written into it, each learned from a real failure here**: fold 1 asserts the per-family drop counts **equal** the census's numbers and the width equals 208, aborting before any fit if the rules and the measurement disagree (the K8 lesson: a clone can silently measure a different model); after fold 1's fit it reads `num_emb_type` back off the fitted object and requires no override (the V42 lesson: read parameters back after the fit, not at construction); exactly two `to_csv`, both labelled v52; head and RESULTS prints name V52's own axis, not the parent's (the V40/V43/K8-clone lesson).
- **Pre-declared bar, so the result cannot be re-read afterwards**: as a single model, CV ≥ **0.96140** with ≥4/5 folds improved and offline paired DeLong z > 3 vs `v18_oof`; as a blend member, it must lift the best 5-subset above CV **0.961695**, because LB = CV − 0.00046 ± 0.00001 makes a CV tie a board tie. Honest prior from V41 (37 columns dropped: −0.00002 CV, 4/5 folds ahead) and from the fact that block permutation calls a *redundant* block inert: **a tie is the likely outcome**; the value is either a real gain or a genuinely different vector for the blend.
- **Status**: ⏳ awaiting a Kaggle GPU slot (~18 min). No score-ledger rows exist for V52 and none will be written until CV and LB are both in hand.

---

### 25-10-2026 — continuation: K8 re-run on the right network flips its own verdict — FS is admissible

- **Goal**: land the corrected K8 run and what it authorises.
- **Experiments**: K8 again on the fixed probe, 12.6 min GPU. The read-back proves it measured the shipped model this time (`fitted embedding-type: {'num_emb_type': None}`, fold-1 baseline 0.961393 against V18's recorded 0.96141 — inside the net family's ±0.00003 run jitter), and all three controls are clean again (determinism diff 0.0, alignment 0.49972 < 0.55, ghost −0.00000). **Verdict: 103 of 290 columns inert-or-harmful = 35.5% ≥ the 30% bar → an FS arm is admissible.** The first run's 27.9% was not a near-miss to be argued about; it was the wrong network, exactly as suspected.
- **The census on the model we actually ship**: READS IT — teacher +0.297977, twins_cat +0.002738, route_profile +0.001020, aux_ev +0.000831, freq_rarity_pairs +0.000587, tm_pair_TE +0.000401, freq_rarity_single +0.000371. **Inert** — tm_single_TE +0.000179 (22 cols), org_stats +0.000164 (22), digits +0.000098 (20), the unmapped residual +0.000073 (14), raw_originals +0.000038 (21), the injected noise −0.000004 (1), rating_aggregates −0.000008 (3). Of the 103, **102 are real columns (35.3% of the 289-column frame) and one is the probe's own noise column**. The *recommended drop set* is narrower than the inert set: **org_stats 22 + tm_single_TE 22 + digits 20 + unmapped residual 14 + rating_aggregates 3 = 81 columns (28.0%)** — the 21 raw originals read inert too but they are the inputs every other family is derived from, so removing them is a different (and riskier) experiment than V41 already priced.
- **Key Learning**: the two runs differ on which blocks the net reads (under the plr embedding tm_single_TE read +0.000238 = READS; under the shipped default it reads +0.000179 = inert), which is the whole reason a probe must match the model that ships. Also: the frame's usage is concentrated harder than any ledger note said — one column carries +0.298 of block importance while 81 columns carry under the noise floor.
- **Status**: ✅ FS reopened→**admissible**; next arm is a drop-only rung on V18 (one axis), priced honestly against V41's precedent that dropping columns is a tie, and against the calibrated law that only a CV gain (or a genuinely different vector) can move the board.

---

### 25-10-2026 — continuation: V51 scores 0.96113, settles the offset, and says the own-pool is finished

- **Goal**: land V51 and read its verdict against the two hypotheses it was built to separate.
- **Experiments**: V51 (v27+v36+v44) — CV **0.961585**, **LB 0.96113 = new best**, gap **−0.000455**. Three legs vs V49's four (−0.000484) and V50's five (−0.000475) → **the offset does not shrink with leg count: H1 (family mix) wins, H2 (leg count) is dead**, and the working predictor is **LB ≈ CV − 0.00046 ± 0.00001** over three calibration points.
- **Key Learning**: the campaign now has a calibrated CV→LB law for its own blends, which converts the remaining time into arithmetic: to score 0.9621 a blend needs CV ≈ 0.96256, i.e. **+0.0010 above the pool's ceiling**, and the exhaustive subset search (2..6 members over all 14 single-arm vectors, run with this landing) is the proof that no recombination of what we own reaches it. So the three things still worth a run are all *diversity* purchases, in cost order: the **K8 re-run** (a slimmer frame is the only in-house route to a decorrelated member), the **never-tried real-rows-as-training-data arm** (the only in-house change of inductive bias), and **foreign vectors** (the 0.962 tier, gated on the owner's asset listing). Second: V51's +0.00003 over V29 is 0.5 ruler, so this is a record number and not a proven gain — finishing should follow CV under the calibrated offset, not the public order.
- **Status**: 🏆 V51 0.96113 becomes the standing best number; V50 keeps an equal private claim. No further recombination of the own-pool is justified: the channel is now measured, calibrated and capped.

---

### 25-10-2026 — continuation: V51 is authored and run locally to settle what the offset actually depends on

- **Goal**: turn the K8/V50 findings into the next experiment, with zero GPU.
- **Experiments**: **V51 TrioDiverseBlend** — equal-weight logit-mean of **v27 + v36 + v44**, the best 3-member set over all 14 single-arm vectors (exhaustive over C(14,2) and C(14,3), not greedy). CV **0.961585**, only **0.000010 below V50**, pooled-minus-fold-mean −0.00000, paired bootstrap vs `v50_oof` **z −1.25** (insignificant, as pre-declared — this arm is not a CV claim). Ran in 39.8 s; `v51_oof.csv` + `v51_sub.csv` written, two files, both labelled V51.
- **Why it exists**: V49 (4 legs) and V50 (5 legs) both sit at gap −0.00048 while single-model nets sit at −0.00018/−0.00023, and nothing measured yet separates the two explanations. **H1 (offset set by family mix)** predicts V51 ≈ 0.96110; **H2 (offset grows with leg count)** predicts ≈ 0.96128. The hypotheses are 3.2 rulers apart on the board and one submission apart in cost.
- **What each answer buys**: if H2 holds, the campaign has been paying for averaging it doesn't need and the remaining days belong to *small* diverse blends; if H1 holds, V50's structure is right and the only headroom left is foreign vectors (task #38) — which is also the answer that closes the composition channel honestly.
- **Status**: ⏳ CV-side work complete, **no LB yet, so V51 holds no score-ledger row** (ledger rule 1). Pending the owner's submission: run it once, report the score, and the appendix law gets its third point.

---

### 25-10-2026 — continuation: K8 ran clean controls on the wrong network, and the FS verdict is reopened

- **Goal**: read the K8 stdout, decide whether a feature-selection arm exists, fix what the log exposed.
- **Experiments**: K8 on GPU, 12.3 min, fold-1 fit of the parent's recipe plus one injected noise column. All three controls passed — determinism `0.961276 vs 0.961276 (diff 0.0)`, whole-frame alignment `0.49959 < 0.55`, ghost `_k8_noise −0.00000` — so the instrument is sound and the readings below are real. Verdict printed by the probe: **81 of 290 columns inert-or-harmful = 27.9%, under the pre-declared 30% bar → "NO: feature selection on this frame is closed by measurement"**.
- **The catch, found by reading the log's own header instead of its verdict**: the fit carried **`num_emb_type='plr'`** — the cloned file was V39's, not V18's. V18 leaves that key unset and gets the class default, so the census measured a net at CV 0.96120 / LB 0.96097 instead of the 0.96129 / 0.96106 model we ship; the fold-1 baseline in the log (0.96128) sits 0.00013 under V18's recorded 0.96141, which is the same discrepancy seen from the other side. A verdict that missed its bar by **2.1 percentage points** cannot be read off the wrong network, so **FS is not closed — it is reopened**, and the probe is fixed: the key removed, a pre-flight that *refuses to run* if any embedding-type key is present, and a read-back **after** fold 1's fit asserting the value is None/'pbld`. The config dict now diffs against V18's with **zero** differences, and the 22-check calibration still passes.
- **What the census says about the shipped model** (new knowledge either way, and it agrees with every closure we ever made by argument): the ranking is carried by eight families — teacher **+0.258906** (one column), twins_cat +0.003973, aux_ev +0.000810, tm_pair_TE +0.000800, route_profile +0.000714, freq_rarity_pairs +0.000449, freq_rarity_single +0.000269, tm_single_TE +0.000238 — while org_stats (+0.000155), digits (+0.000067), raw_originals (+0.000052), the unmapped residual (+0.000049) and the rating aggregates (−0.000005) read inert.
- **The methodological caveat that limits what FS could ever buy here**: block permutation reads a **redundant** block as inert because its information is duplicated elsewhere in the frame. The 21 raw originals are the proof — +0.000052 for the block, while every family that re-encodes them (twins, TE, route, freq) reads positive. Inert therefore does **not** mean deletable, and that is exactly why the 30% bar was set in advance rather than negotiated after the number.
- **Status**: 🔁 K8 re-run needed on the fixed probe (same ~12 min GPU) — the only open question it answers is whether the shipped net's inert share clears 30%. Meanwhile the CPU-side move is the small-diverse-blend test (2–3 members: v44+v27, or v45+v27+v11) against the frozen −0.00048 offset, which prices the leg-count term the V44/V45 arms could not, and the external-asset listing (task #38) still gates anything stacked on foreign vectors.

---
### 25-10-2026 — continuation: V50 takes the record, and the offset law splits into diversity vs variance-removal

- **Goal**: log V50 (LB 0.96112) and verify the incoming "Next-Level Strategy" plan against our own measurements.
- **Experiments**: V50 equal-weight logit-mean of v44+v45+v36+v27+v11 — CV **0.961595** recomputed from the shipped vector, **LB 0.96112**, gap −0.00048, folds 5/5 improved vs V29 on the confirmed house partition, bootstrap paired DeLong z **+3.38** vs V49 / **+10.56** vs V18. First arm to clear Δ ≥ +0.0001, z > 3 and ≥ 4/5 folds together. CPU-only, no fit.
- **Key Learning**: the averaging law is really two laws. Inside one recipe (V36 folds, V44/V45 seeds) the CV→LB gap widens by almost exactly the CV bought, so LB never moves. Across **different** arms it froze at −0.00048 from 4 to 5 legs while LB tracked CV ~1:1 (V49→V50: CV +0.000035 → LB +0.000040). Consequence stated in numbers: with the offset frozen, **a board 0.9621 needs CV ≈ 0.9626**, which is +0.0010 above V50 and larger than every single-model gain this campaign has produced combined — so the only routes left are foreign vectors (public-kernel OOF pools, K3's still-pending public run) or a genuinely different frame (K8). Second: **our DeLong stands** — the recorded +8.76 (V49 vs V18) reproduces at bootstrap +8.09 and V50's +3.34 at +3.38, while two analytic ports of mine gave +0.66 and +71.1; hand-ported analytic SEs are hereby banned in this repo. Third: the house partition is now *proved* by reproducing V18's and V29's recorded fold arrays digit-for-digit from the stored vectors.
- **Plan verification**: the incoming document's channel arithmetic is right (teacher +0.00016 CV / **+0.00044 LB** = V12−V2, matching the ledger exactly) and its hybrid-stack instinct follows from our own law. Three of its claims are wrong or unverifiable: V49 is a **4-member** blend, not "all 42"; V48's members correlated **0.962** with `_og_teacher`, the >0.998 was member↔member; V31 does not show the labels are "clean" (ρ ≤ 0.0387 is a measured flip rate). Its "internal pool cannot cross 0.9611" is now falsified on the number by V50, though not on the evidence (+0.00002 = 0.4 ruler). Its TabPFN-3.5 / Summit / Kratosyan assets cannot be reached from here — Kaggle pages are JS-rendered and there is no local API token, so the owner must print them (task #38) before anything is authored on top of them.
- **Status**: 🏆 **V50 is the new best LB (0.96112)**; V29 0.96110 becomes the reference point, and adoption of V50 rests on the CV/fold/z evidence, not on a 0.4-ruler board move. Next: K8 on GPU (restored, three defects fixed, 22 calibration checks passing) and one validated external-asset listing. No `feature_engineering.md` row: V50 adds no column.

---
### 25-10-2026 — continuation: V49's composition blend ties the 2nd-highest LB and measures the diverse-blend offset

- **Goal**: run the composition phase's first version entirely locally (the owner lifted the 2026-10-11 standalone law for composition arms) and submit it.
- **Experiments**: V49 = equal-weight logit-mean of v11+v27+v34+v36 — the hill-climb optimum over all 42 archived OOF vectors (greedy with replacement; SLSQP returns uniform; rank-mean ties). Ran locally in 6.9 s — the first fully locally-verified arm (in-run paired DeLong: z +8.76 vs v18, +6.01 vs v36, +8.89 vs v29). OOF **0.961564** (+0.00028 vs V18) → **LB 0.96108, tied with V32 for 2nd-highest ever**; logged as the Ensembles table's first entry.
- **Key Learning**: the offset law extends to diverse blends: LB − OOF = **−0.00048**, the largest constant yet (V36 −0.00035, V44 −0.00038, V45 −0.00037). Own-pool composition is therefore offset-bound — the OOF ceiling 0.96156 maps to LB ~0.9611, so **no composition of our own 43 vectors passes 0.9611 on the board**. The route above is the public pools: kratosyan's 119-member stack (OOF 0.962048 → LB 0.96167 at his own −0.00038), Deotte's 11 TFM OOFs (tabpfn35 solo 0.961145, above our best single), the 0.96213 wall file, Jiatian's 15-fold GPU stack (OOF 0.96238). Also landed: V48's row (LB 0.96091) — the stacked-single shape closed, and its r2 members reached their ledger configs (0.96105/0.96112/0.96103).
- **Status**: ⚠️ V49 not adopted (V29 0.96110 remains best LB; V49 = 2nd-highest, tied with V32). Next per the battle plan: the public-stack forks — kratosyan fork with LR C=0.003, +Deotte's TFM columns, the wall play with one unrounded-score change, finals = CV-backed stack + wall variant.

---

### 25-10-2026 — two multiseed arms land the campaign's highest OOF and move the board by nothing

- **Goal**: land V44 (V18 frame x 3 seeds) and V45 (V29 frame x 3 seeds), both shipped as the logit-mean of their legs.
- **Experiments**: V44 CV 0.96142 / LB 0.96104 (ΔCV +0.00013, ΔLB −0.00002 vs the parent's band); V45 CV 0.96142 / LB 0.96105 (ΔCV +0.00014, ΔLB −0.00005 vs V29 0.96110). Legs: V44 0.96126/0.96127/0.96132, V45 0.96127/0.96125/0.96129. Timing 46.4 + 48.8 min ≈ 1.6 GPU-h for the two.
- **Key Learning**: **averaging buys CV and pays it straight back as offset** — the CV→LB constant went −0.00023 → −0.00038 (V44) and −0.00018 → −0.00037 (V45), which is the same cancellation V36 measured when it doubled the fold count (+0.00010 CV, −0.00012 offset, LB −0.00002). Three instances, two mechanisms, two frames: an arm that averages more models into its vector must NOT inherit its parent's offset, and both headers that did predicted an LB 2.1–3.2 rulers too high. Second: **`digit-for-digit` reproducibility is tree-only** — V44's same-seed leg reads −0.00003 against V18 and V45's against V29/V34 by 1e-5, so up to half of the ±0.00006 "seed tick" is run-to-run jitter on the net family; a cross-file control is not a control there. Third, and the cheapest: the **census was already available for free** — `v44_oof` sits at logit corr 0.999201 against `v18_oof` and `v45_oof` at 0.999197 against `v29_oof` (both above finding 82's 0.999 collapse threshold), so the two blends are the parent's ranking with noise removed, not a new column, and the null board result was predictable from stored vectors in seconds **before** the 95 GPU-min spent on them.
- **Status**: ⚠️ Neither adopted. Standing best is still **V29 0.96110**; ten of the 43 scored arms now sit inside the 0.96100–0.96106 window the V18 band defined. Highest OOF ever is now 0.96142 (both arms) and the board ranks them 6th and 8th, which is the CV gate's flattest demonstration to date. Unrun doors unchanged: K8 (net-side importance), K7 (TabPFN-3), K3's public-pool run — plus the owed offline paired DeLong on `v44_oof` / `v45_oof`.

---

### 25-10-2026 — continuation: V48 stack-trains the net and loses to its own members, closing the last stacking route

- **Goal**: land V48 (three in-run GBDT members' OOF logits as 3 columns for the V18 net).
- **Experiments**: V48 CV 0.96070 pooled / 0.96099 fold mean / **LB 0.96091** — ΔCV −0.00059 vs V18 (−10.4 rulers), 0/5 folds improved, student −0.00042 below its own best member (LightGBM 0.96112). Members fitted in 36.3 of the 53.8 min. Numbers **46 and 47 have no rows**: their scripts existed in root earlier today and were removed before any run, and no v46/v47 vectors or stdout are on disk — same treatment V14 got, so the sequence gap is a missing arm, not a missing result.
- **Key Learning**: **the census that was free had already priced this.** The three members correlate 0.99776–0.99898 in logits (one channel by finding 82's rule) and ~0.962 with the `_og_teacher` column the frame has carried since V1, so the widening added a near-duplicate worse copy before it added a column — and unlike V44/V45 the vector is *not* collapsed against its parent (v48 vs v18 logit corr 0.996478, below the 0.999 bar), which is the campaign's cleanest demonstration that **independence is necessary, not sufficient**. Second datum: **pooled OOF drifts from fold mean by −0.00029 on a cross-fitted-column vector** where single-model nets stay within 0.00001, so V48's headline **+0.00021 gap is a drift artefact** and both statistics must be quoted for such arms. Third: the arm declared **no implied LB** (the V44/V45 lesson) and so failed its own gate without publishing arithmetic that later needed correcting — the design change worked; adopt it as standard for every shape whose offset is unmeasured.
- **Status**: ❌ Not adopted. Standing best remains **V29 0.96110**; the board is unmoved by six arms since (V38–V48 all land inside or below V18's band). Stack training is closed, FE is closed (V43), averaging is zero-sum (V44/V45/V36). The only unmeasured doors left are K8 (net-side importance), K7 (TabPFN-3) and K3's public-pool run — plus v49, which has shipped vectors but no LB and so cannot hold a row.

---
### 24-10-2026 — five arms land, four library-mechanism failures are found by the arms themselves

- **Goal**: run the §10 queue (V38/V39/V40/V41/V42) and land every verdict.
- **Experiments**: V38 k-NN column 0.96125 CV / 0.96100 LB; V39 `num_emb_type='plr'` 0.96120 / 0.96097; V40 prior-as-`base_margin` 0.95319 / 0.95196 with BestIter 0 in 5/5 folds; V41 subtractive net frame 0.96127 / 0.96104; V42 CatBoost MVS 0.95990 / 0.95965, fold-1 paired −0.001361. None adopted; standing control stays the V18 band, best LB V29 0.96110.
- **Timing**: 68.5 + 17.0 + 4.7 + 16.4 + 14.0 min ≈ 2.0 GPU-h, of which 51 min was one k-NN column build.
- **Key Learning**: four library-parameter failures in one day, all caught by guards that measure rather than by documentation: `scale_pos_weight` is invisible to `inspect.signature` because the wrapper forwards it (`**kwargs`); Kaggle's xgboost has no `eval_base_margin` and rejects a DMatrix eval pair, which turns an offset arm into the prior itself; CatBoost's `sampling_frequency` enum is `PerTree`/`PerTreeLevel`, not the values I named from memory; and `len(CatBoostClassifier)` does not exist, which killed a fold that had already trained. The rule that came out: **copy the archived parent's parameter and change one word; read values back AFTER the fit; let a diagnostic never be able to kill a good run.**
- **Status**: ⚠️ Queue empty — five arms, five closures, zero adopted. Next step is decided in `PLAN_V36_ONWARD.md` §17.

---

### 24-10-2026 — continuation: V43 lands below the control, and FE closes by measurement

- **Experiments**: V43 ipsative ratings — CV 0.96124 / **LB 0.96101** (gap −0.00005 CV vs the control, folds improved 1/5, gap-to-LB −0.00023 for the 6th reproduction). K8 (net-side importance) and K7 (TabPFN-3 feasibility) remain unrun.
- **Key Learning**: the arm's own control line produced a new dataset fact — **no respondent has all 13 ratings unrated** (train 0, test 0; the whole frame is 999,479 = the recorded 699,635 train + 299,844 test), so `_count_0` never reaches 13 and any "disengaged respondent" mechanism is bounded. Second: this arm's two runtime axis prints (the head print and the `[5/5] RESULTS` footer) still printed its ancestor's axis (V12's aux_ev text) because the clone rewrote `CFG` and the docstring but not those strings — the same defect I removed from V40 before it ran. A clone must rewrite every place the axis is printed, not just the identity constants.
- **Status**: ⚠️ V43 not adopted (control unchanged, V29 0.96110 remains the highest LB). Register of transform families is now exhausted by measurement; the two live doors are FS-on-the-net (K8) and the foundation model (K7).
- **Repo layout (owner's instruction, 2026-10-24):** the root keeps **only `README.md`**. The three plan documents
  (`PLAN_V21_ONWARD.md`, `PLAN_V31_ONWARD.md`, `PLAN_V36_ONWARD.md`) and the six stdout probes
  (`S6E10_K1b/K2/K3/K5/K7/K8_*.py`) are removed from the tracked tree, so every `PLAN_... §n` reference in the rows
  above now points at **git history, not a file on disk** — recover any of them with
  `git show <commit>:PLAN_V36_ONWARD.md` (they were all committed before the removal, and the removal commit's parent
  holds byte-identical copies). Nothing in this note edits a historical row: the references stay as written.

---

### 22-10-2026

- **Goal**: answer the owner's question — *why not give ModernBERT a try and see if it really works or not?* — with a
  measurement rather than a citation, since the campaign's verdict that the transformer family is not worth a version
  rests entirely on other people's implementations (`PLAN_V31_ONWARD.md` §2c, thread 751334).
- **Experiments**: **`S6E10_K2_ModernBertProbe.py`** authored, verified, committed (`5cf5a13`) — a K1b-style stdout-only
  probe: fold 1 only, random-init order-free ModernBERT (12×384, every token at position 0, mean pool → logit), fp16 not
  bf16 (a T4 has no bf16 tensor-core path; their recipe was an A100), no weights downloaded so no licence decision.
  **Arm A** = the raw 21 columns; **arm B** = the same stack plus our two channels as summary tokens (`_og_teacher` and
  the mean of the 13 rating EVs, both fitted on original rows only, V18's run-of-record). Three bands pre-declared:
  A < 0.96040 → close the family; 0.96040–0.96140 → the A/B channel effect (bar **+0.0004**) is the only reason to buy
  5 folds; ≥ 0.96141 → author V36, which must first buy its own CV→LB offset.
  **Run 1 aborted before epoch 1** — cost ~1.5 GPU-min, **no AUC, therefore no band reached and no arm closed.** It died
  on its own guard: `ModernBertConfig` in that build has **no `hidden_dropout`**, so the `hasattr` contract check raised.
- **Timing**: header, data path and both feature blocks completed: teacher **3 s**, 13 rating EVs **35 s** (mean EV
  3.293), fold 1 exactly 559,708 / 139,927 — the same rows V18's fold 1 is on. Bin maxima read off the real data:
  **Age 85, Flight Distance 2491, both delays 183**, so the derived value-embedding tables are 87 / 2493 / 185 rows.
- **Key Learning**: **(i) A guard that asserts a library's field names is one assertion per release.** The fix is a
  scan, not a longer list: every attribute the installed config actually exposes whose name mentions dropout is zeroed
  and printed. The alternative was worse — `ModernBertConfig(**kwargs)` **swallows an unknown keyword silently**, so an
  invented name would not have crashed, it would have trained a model with dropout ON and reported it as this one. That
  is the Arg-Name Law's failure mode, caught by a loud stop instead of a quiet wrong measurement.
  **(ii) Two defects were found by re-reading the traceback, because nothing here can be executed locally**: the
  permutation test's own `randperm(23) + 1` indexes a 23-wide slice out of range, and `featurise` had been casting every
  column to float32 before the categorical branch (the four categoricals are strings). Local runtimes re-probed: numpy
  2.5.4 / pandas 3.0.6 / sklearn 1.9.1 / scipy 1.18.1, and **`torch`, `transformers`, `xgboost` all MISSING** — so no
  tensor shape, dtype or attention behaviour has ever run before upload.
  **(iii) The probe now measures its central assumption instead of asserting it.** A permutation test runs the full
  `tokens() → encoder → pool → head` chain in fp32 and fp16 on 8 rows in seconds and raises if permuting a row's tokens
  moves the logit, i.e. if `position_ids=0` is being ignored and the "order-free" model is secretly a sequence model.
  A per-epoch AUC trace also gates the read: a last-epoch gain > +0.002 makes the verdict about **budget**, not family.
- **Status**: K2 fixed and re-committed, ready to re-run on Kaggle GPU (~45–60 min, stdout only). No score, no OOF, no
  submission, no `V#`. The queue otherwise stays as the owner left it on 2026-10-18: **empty** (§3 erasure record).

---

### 22-10-2026 — continuation: run 2 died inside the library, and the audit found four more defects before run 3

- **Goal**: the owner asked for run 2's traceback to be read *and* for every change to be re-checked against what it
  claims, rather than fixing only the line that raised.
- **Experiments**: **K2 run 2 aborted at model construction** (~1.5 GPU-min, again no AUC, again no band reached).
  `ModernBertEmbeddings` builds a token table with `padding_idx=config.pad_token_id`, and this build's default for
  `pad_token_id` is **greater than or equal to my 2-row `vocab_size`**, so torch's own assert
  (`Padding_idx must be within num_embeddings`) fired. The scan-based guard had worked, though: the run printed the four
  dropout knobs that `ModernBertConfig` 5.0.0 really exposes — `attention_dropout`, `classifier_dropout`,
  `embedding_dropout`, `mlp_dropout` — all zeroed, and **no string-typed position field exists at all** in that build,
  which is exactly why order-freeness is measured by permutation instead of asserted by name.
- **Timing**: header and data path reproduced run 1 exactly (teacher 3 s, EVs **36 s**, fold 1 559,708 / 139,927,
  bin maxima Age 85 / Flight Distance 2491 / delays 183). `transformers` is **5.0.0**, now printed by the header because
  two of the three defects are version-dependent.
- **Key Learning**: **the audit found four defects the traceback did not**, and one of them would have made K2's null
  verdict wrong rather than merely delayed:
  **(1) the encoder's LR contradicted the file's own design.** 5e-5 is a *fine-tuning* rate, but the docstring's central
  justification for skipping pretrained weights is their ablation that random init scored the same — under their recipe.
  A random-init 12-layer encoder at 5e-5 for 4,370 steps can print ≈0.5, and band A (`< 0.96040 → close the family`)
  would then have closed a **family on my training setup**. Fixed: 3e-4 with 200-step warmup, plus a printed
  **NO-LEARNING floor** (`< 0.70` = a training failure that closes nothing) so the file cannot deliver a family verdict
  it is not entitled to. Same class of error as the dropout **kwargs** swallow: an internally inconsistent setup that
  still produces a clean-looking number.
  **(2) the printed read could drift from the pre-declared read.** The band edges, the +0.0004 channel bar and the
  +0.002 under-training bar were typed as literals in the prints *and* in the docstring. Now one constant each
  (`BAND_REPRO`, `BAND_CONTROL`, `CHANNEL_BAR`, `UNDERTRAINED_STEP`, `LEARN_FLOOR`, `WARMUP`), and the seed-move
  citation is **derived** (`BAND_CONTROL − SEED_REPRO_FOLD1` = V18 vs V33 on this fold) instead of remembered — which
  is also the no-hardcoded-sibling-numbers law applying itself to my own file.
  **(3) the channel was not V18's channel.** I had written "the exact columns V18 scores with" while fitting the teacher
  at 400/0.8 and the EVs at 150/6/0.06 with seed 42; V18 uses **600 and 300 trees, `max_depth` 8, `colsample_bytree`
  0.5, seed 0**. Copied verbatim, so the A/B measures architecture rather than my re-tuning. One difference remains and
  is now declared in the file rather than hidden: V18 feeds the four categoricals to XGBoost as Categorical dtypes with
  `enable_categorical`, this probe feeds shared integer codes — a weaker partition rule, which biases **arm B downward**,
  the safe direction for a test whose null closes a family.
  **(4) in-place token assembly, and my own typo.** `e[:, j] = e[:, j] + …` mutated a tensor the autograd graph holds —
  the kind of crash that appears on the *backward* pass, i.e. after the forward-only permutation check passes. Replaced
  by slicing (`NUM_COLS` are the first four columns by construction). And the edit that introduced the constants also
  introduced `("B", auc_b and aucs_b)`, a nonsense boolean I wrote and caught by re-reading — which is the honest limit
  of this loop: `torch`, `transformers` and `xgboost` are absent locally, so nothing in the model path has ever
  executed here, and **every one of these was found by reading, not by running**. Guards added for it: `pad_token_id=0`
  is now *checked* after construction, the built encoder is *scanned* for any `nn.Dropout` with `p > 0`, `LinearLR`'s
  keywords are signature-guarded like `ModernBertModel.forward`, and non-finite loss or non-finite eval logits raise
  instead of returning a trustworthy-looking AUC.
- **Status**: K2 fixed, all six probe checks clean, docstring 35/35, binning invariant re-proved; committed and ready
  for run 3. Still **no score, no OOF, no submission, no `V#`**, and the queue stays **empty**.

---

### 22-10-2026 — continuation 2: run 3 died in my own tensor assembly, so the shapes are now proved locally

- **Goal**: read run 3's traceback and stop losing GPU runs to failures that a local check could catch.
- **Experiments**: **run 3 aborted at 8 rows inside `RowSet.tokens()`** — `RuntimeError: The size of tensor a (4) must
  match the size of tensor b (8) at non-singleton dimension 1`. Cause: `em(ids[:, j]).unsqueeze(1)` **and**
  `torch.stack(..., dim=1)` each insert an axis, so `v` came out 4-D `(B, 21, 1, H)` and every later broadcast
  mis-aligned. It had been in the file since the first draft — the earlier `forward()` had the same line, so
  **`tokens()` had never executed successfully on any run.** Fixed to stack 2-D `(B, H)` elements directly, and the
  pre-flight now asserts the assembled shape as `(rows, len(WIDTHS)+3, H_WIDTH)` with both shapes in the message.
  **Progress the guards made**: the model *constructed* (so the `pad_token_id` fix holds), the built-encoder dropout
  scan passed, and the `LinearLR` signature guard passed — all before any epoch, which is why this run also cost
  ~1.5 GPU-min rather than 45.
- **Timing**: header reproduced every run-1/2 datum again (teacher 4 s, EVs **96 s** under V18's copied 300-tree
  recipe, mean EV 3.296, fold 1 559,708 / 139,927, bin maxima unchanged). The derived band print worked:
  **control 0.96141 | seed-2027 repro 0.96083 | pure seed move 0.00058** — computed from two constants now, not typed.
- **Key Learning**: **the shape class is reproducible without torch, because numpy and torch share broadcasting and
  `stack`/`cat` semantics.** New harness `check_k2_shapes.py` (temp) extracts the file's own `tokens()` /
  `logits_from()` / `forward()` source via AST and execs it against shape-only stand-ins for `nn.Embedding`,
  `nn.Linear` and the encoder: it now proves `tokens() → (8, 24, 384)` = one set token + 21 column tokens + 2 channel
  tokens, and `forward() → (8,)`. **Calibrated in both directions**: with run 3's `.unsqueeze(1)` restored it fails
  with the *same* artifact the GPU reported — `(8,4,4,384)` against `(8,4,384)`. What it still cannot check is autocast
  dtype choice and real attention behaviour, so the permutation test on GPU remains the contract check; the harness
  only removes the class of failure that used to cost a run. Broader rule: **the pre-flight is what makes a probe
  cheap** — three aborts, ~4.5 GPU-min in total, and no run ever reached a number it could not trust.
- **Status**: K2 fixed, six probe checks clean, docstring 35/35, binning invariant and now the model-path shapes
  proved locally. Ready for run 4. Still **no score, no OOF, no submission, no `V#`; the queue stays empty.**

---

### 22-10-2026 — continuation 3: K2's first full run landed, and its own verdict logic was the bug

- **Goal**: read run 4's numbers, and check whether the file was entitled to the conclusions it printed.
- **Experiments**: **run 4 completed both arms (20.2 min, fold 1, `EPOCHS=2`)** — the first K2 run to print AUCs at all.
  **Order-free contract holds on real hardware**: permuting a row's tokens moved the logit by **4.1e-08** (arm A) and
  **6.0e-08** (arm B) relative to the logit scale, so `position_ids=0` is honoured by `transformers` 5.0.0 and the
  encoder genuinely reads a row as a set. Model: **29.6M params, 24 tokens/row, peak 3.59 GB, 274 s per epoch-arm.**
  **Arm A (raw 21 columns): 0.87730 → 0.92400** (last epoch **+0.04670**, still climbing). **Arm B (+our two channels):
  0.94635 → 0.94224** — it **peaked at epoch 1 and then lost 0.00410**. Same-epoch gap **+0.06905 at ep1, +0.01824 at
  ep2**; best-vs-best **+0.02235**. Both arms sit below 0.96040, but the file's own horizon guard says that is a
  budget statement, not a family one.
- **Timing**: prep reproduced again (teacher 4 s, 13 EVs 93 s, mean EV 3.296, fold 1 559,708 / 139,927). **My cost
  estimate was 4× pessimistic**: measured 274 s per epoch-arm means a full 5-fold × 6-epoch arm is **~2.3 GPU-h**, not
  the 3.5–4 in the plan.
- **Key Learning**: **three things, and the middle one is a defect in my own reading machinery.**
  **(1) Our channels are worth ~150× more to an under-trained transformer than to our net** — +0.01824 at epoch 2 vs
  +0.00012 for V18 — which is the most interesting number the campaign has printed in weeks. But the gap **collapsed by
  0.05 in a single epoch**, and that is the whole story: a raw-only arm that cannot yet extract the rating→satisfaction
  law borrows it from the channel, and as it learns, the loan is repaid. So the open question is no longer "does the
  transformer like our features" but **whether any part of the gap survives arm A's convergence**. That is now the
  decisive, falsifiable reading: gap > 0.00058 (one seed move on a fold) surviving convergence ⇒ a strong learner still
  cannot get our law from the raw columns, and the family has a route we lack; gap collapsing to noise ⇒ the channels
  were only a training-time shortcut, exactly what the monotone-invariance argument predicts, and the family adds no
  information we don't already hold.
  **(2) `step > bar` was the wrong test for convergence, and it nearly closed a family.** Run 4 printed *"arm B →
  converged at this budget, so the band below binds"* for a curve that peaked at epoch 1 and **fell** — a non-increase
  was being read as a plateau. A decline is a *schedule* signal, not a convergence signal, and letting it feed the band
  logic would have filed "the order-free family is below our control" from two unjudged curves. Now: **the best epoch
  is what is compared** (last-epoch and best-epoch disagree here by 0.004), a peak followed by a decline is labelled
  schedule-limited, a still-rising last epoch is horizon-limited, and the gap is printed **per epoch** so a comparison
  is always same-budget. The guard I built to stop a cheap reading closing a family is what made this visible — it
  fired correctly (`+0.04670 > 0.002`) while the *verdict wording* around it was wrong.
  **(3) The A/B itself was apples-to-oranges.** `channel effect = auc_B_last − auc_A_last` mixes a falling arm with a
  rising one; the honest quantities are the same-epoch gaps and best-vs-best, which differ (+0.01824 vs +0.02235).
- **Status**: K2 re-authored to **6 epochs** with best-epoch verdicts and the per-epoch gap series; docstring re-declares
  the read against run 4's evidence, `EPOCHS=2` bands retired in the open. All local checks green (compile, prose
  35/35, names, flow, dead-construct scan, **model-path shapes**, **binning invariant**). **Run 5 is the decisive one.**
  Still no OOF, no submission, no `V#`; the ladder queue stays **empty**.

---

### 22-10-2026 — continuation 4: §3b moved forward by pre-spending the schedule lever run 4 already bought

- **Goal**: make run 5 able to answer the question §3b now asks — does any part of the B−A gap survive arm A converging.
- **Experiments**: K2 amended before re-running, so no hour is spent re-measuring run 4's decline. **The encoder rate now
  halves every 2 epochs** (3.00e-4 → 2.12 → 1.50 → 1.06 → 0.75 → 0.53e-4 at the epoch boundaries, final factor 0.125),
  applied as an explicit `0.5 ** ((ep + frac) / LR_HALFLIFE)` after warmup rather than a second scheduler, because
  `LinearLR` recomputes the group rates from its own state and two schedulers would overwrite each other. The A/B stays
  paired: both arms and both param groups see the same factor, so the gap remains a same-budget comparison. RESULTS also
  prints the **gap at arm A's best epoch**, since "best-vs-best" compares two different budgets (ep1 vs ep2 in run 4) and
  is therefore not the quantity the band decision wants.
- **Timing**: unchanged in expectation — 6 epochs × 2 arms × 274 s ≈ **55 min**, plus ~2 min prep, still fold 1 only.
- **Key Learning**: **the new local check failed honestly, which is the point.** `check_k2_lr.py` (temp) re-implements
  only the arithmetic and reads the constants from the file's AST; its first version asserted monotonicity across the
  *whole* curve and so reported 199 "rises" — warmup rises by design. The assertion that fired first was itself a bug in
  the checker (it did not handle tuple targets like `WARMUP, LEARN_FLOOR = 200, 0.70`), and it **raised rather than
  printing a clean pass**, which is the behaviour that keeps this class of check worth running. Now the invariants are
  the three that matter: warmup never falls and lands exactly on `ENC_LR`, the decayed leg never rises, and the handoff
  is not an upward jump (measured −6.2%). Second lesson: pre-spending a lever that the previous run already paid for is
  not the same as spending two levers blind — run 4's decline is the evidence, and the horizon stays the only other
  change.
- **Status**: docstring re-declares the read (35/35), all six probe checks plus the **shape**, **binning** and new
  **LR-curve** harnesses green. Ready to upload for run 5. Still no OOF, no submission, no `V#`; ladder queue **empty**.

---

### 22-10-2026 — continuation 5: run 5 settled the level, withdrew my own 150× claim, and ended K2

- **Goal**: read run 5 (EPOCHS=6 + decayed LR) against the bands §3b re-declared, and decide whether the probe has
  anything left it can measure.
- **Experiments**: **K2 ran to completion: 59.8 min, both arms, no aborts.** Arm A (raw 21 columns) 0.94082 | 0.93932 |
  0.93644 | 0.93573 | 0.94026 | 0.93547 — best 0.94082 on epoch 1. Arm B (+our channels) 0.93176 | 0.93786 | 0.94570 |
  0.94650 | 0.92816 | 0.94990 — best 0.94990 on epoch 6. Same-epoch gaps −0.00906 | −0.00146 | +0.00927 | +0.01077 |
  −0.01210 | +0.01443. Order-free contract re-confirmed (4.1e-08 / 6.0e-08). Peak 3.59 GB, 29.6M params, ~290 s/epoch.
- **Timing**: the decay did what it was bought for at the front — arm A reached 0.94082 by epoch 1 where run 4's flat
  3e-4 was still at 0.92400 after epoch 2 — but it did not buy convergence: **both arms are flat within their own
  scatter** (least-squares slope over the six epochs: A −0.000704/epoch ± 0.000536, B +0.001783 ± 0.002171; neither
  exceeds 2 SE).
- **Key Learning**: **three conclusions, the first of which is a retraction of my own headline.**
  **(1) The "+0.01824, ~150× what our channels buy V18" claim from run 4 is withdrawn.** Run 5's six same-epoch gaps
  average **+0.00197 with 1 SE 0.00454 — not separable from zero** — and re-checking run 4 on its own terms makes it
  worse: its two gaps (+0.06905, +0.01824) average +0.0436 with 1 SE 0.0254, i.e. **1.7 SE from zero on the very run
  that printed it.** I derived a 150× effect size from two endpoints of two curves that were moving in opposite
  directions, and passed it to the owner as the run's most interesting finding. The corrected statement: *this
  instrument cannot tell whether our channels help a transformer at all.*
  **(2) What the run CAN settle is the level, and it settles it hard.** Arm A's best epoch is **0.94082**, i.e.
  **0.02059 below V18's fold 1 (0.96141) = 35× the 0.00058 seed ruler**. A level survives epoch jitter that a gap does
  not, so this is a real reading: a 29.6M-parameter random-init order-free ModernBERT, six epochs, decayed LR, is not
  close to our control on this data. Combined with measured cost — 59.8 min for one fold of both arms — a 5-fold V36 is
  ~2.5 GPU-h and the gate's seed-2027 reproduction is another ~2.5, i.e. **~5 of our 8 weekly GPU-hours to enter an arm
  that starts 0.02 behind.** V36 *as specified in §3b* is therefore closed: not "the family is incapable", which this
  probe cannot establish, but "no arm is authorable on this regime at this budget against this deficit".
  **(3) A verdict label must carry its own error bar, or it is noise with adjectives.** This is the third self-inflicted
  misreading in one day, each caught only because the numbers were re-derived rather than re-narrated: the name-pinned
  config guard (run 1), `non-increase = converged` (run 4), and now run 5's print calling arm A "PEAKED, schedule-limited"
  and arm B "STILL CLIMBING" off single adjacent-epoch moves of −0.00479 and +0.02174 — moves *inside* each arm's own
  scatter. Fixed properly: RISING/FALLING prints only when the least-squares slope exceeds 2× its own SE, the gap is
  reported as mean ± 1 SE, and both are computed from the run's own numbers. The new `trend()` is verified locally
  against `np.polyfit` on run 5's printed curves — slopes and SEs agree to 1e-9 — so the statistic is checked before the
  sentence that uses it is allowed to speak.
- **Status**: **K2 is finished; no run 6 is proposed.** Its durable outputs are: the order-free wiring is correct on our
  hardware, this family's deficit at this regime is 35 rulers deep, single-fold transformer readings cannot resolve the
  gate scale, and the campaign's ceiling statement (V18 at the public single-model frontier) stands unchallenged by the
  architecture the owner asked about. All local harnesses green (six probe checks, shapes, binning, LR curve, trend
  statistic). No OOF, no submission, no `V#`. Remaining owner doors unchanged: composition (~+0.0008 public) vs stop.

---

### 23-10-2026

- **Goal**: answer "what is the next step — run K2 again, or is there a version to train?" by checking what the public is
  doing (Kaggle board + threads), what GitHub actually contains for this competition, and what the literature says about
  the mechanisms still on the table.
- **Experiments**: no model was run. **K2 stays finished (no run 6) and no version is authored** — after run 5 the ladder
  queue is empty, so the only legitimate next move is an owner decision, and the evidence for it had to be re-measured.
  New tooling fact: **Kaggle serves thread and leaderboard bodies client-side, so `WebFetch` returns only the JS shell**;
  the browser render route reached four bodies and the board, and everything below is labelled by route
  (`[rendered]` / `[api]` / `[raw]` / `[search]`). Digest:
  `…Temp/S6E10_workspace/EXTERNAL_PASS_2026-10-23.md`.
- **Key Learning**: **the pass mostly corroborated us, and the two things it "found" were already in the file — caught
  before either was logged as a discovery.**
  - **Board, live `[rendered]`:** #1 **0.96387** (158 entries), #2 0.96341, #3 0.96247, #4 0.96243, #5 0.96236, then
    **an exact 0.96213 plateau from rank 11 past rank 50** (~40 teams, 8–220 entries — one widely-submitted kernel).
    Our highest printed LB **0.96110 (V29)** is **0.00103 below rank 20** (control band 0.96100–0.96106). This
    **supersedes §5's 2026-10-18 figures** (0.96190 at rank 20,
    0.96273 at #1), corrected in place and named, not silently overwritten.
  - **The composition lead was already recorded**: §5 cites `kratosyan`'s stacking kernel (0.96205 / 0.96167), and
    kratosyan *is* the author of thread 746148, whose prose prints **0.96200 / 0.96163**. Same author, two artefacts,
    ~5e-5 apart, neither verifiable by us — recorded as a provenance discrepancy rather than smoothed. What the prose
    adds is the **price list**: 10-fold members **+0.0001**, TabPFN-3.5/TabICL **+0.0001**, "every other GBDT or RealMLP
    family **+0.00001 to +0.00003**", plus his own negative — four members scoring 0.9600–0.9611 alone made the stack
    **worse (0.962013 → 0.962002)**: "new kinds of signal matter more than more models of the same kind." That is our
    K1b M6 (−0.00076) and V27 (+0.00002) reproduced by a stranger, and it prices the last gap as unreachable by any arm
    of the kinds we already run. A commenter's per-fold audit splits the scale further: +7 TFM members **+6.3e-6 pooled
    (9/10 folds)**, +6 GBDT/RealMLP members **+0.9e-6 (4/10 = chance)**, and "24 of 40 unused OOF columns were exact
    copies (corr 1.000000)" — i.e. stack-level "+1e-4" is two orders above what a gate can see.
  - **GitHub is not a source of method at our level**: 12 repos, **all 0 stars**, best prints 0.96009 CV / 0.95973 LB
    (LightGBM + flight-distance target encoding), a rank-averaged LGB+XGB+CB blend at 0.9588, a random forest at 0.956.
    Its value was **three digit-level confirmations of measurements we made alone**: orig-only model scores **0.954** on
    synthetic train (our M2(c): **0.95465**); wifi rating 0 → **89% satisfied** (our M1: **0.887**); and "**only 4 of
    699,635** train rows have a flight distance the original does not use", which is the mechanism behind our biggest FE
    lever. His CV→LB gaps (−0.00036 twice) are our offset law in a stranger's numbers.
  - **The idea that looked new was not.** Rating 0 = unrated-but-satisfied invites a fix to `aux_ev_*` (EV = Σk·P(k)
    collapses the marker into a low score). Checked against the shipped frame before proposing anything: V12/V18's rating
    twins one-hot **6 levels including 0** (V31's stdout proves it), trees split `<1` natively, and the explicit
    encodings were measured anyway — V22 **0.00000**, V24 **−0.00004**, V28 **−0.00001**, V29 **−0.000007**, both
    families. Closed by inspection.
  - **RTD named and dismissed with a reason**: it is Replaced Token Detection (TabTransformer's SSL task,
    arXiv 2012.06678), and the literature's own premise is *scarce labels* — we have 699,635. 751334's +3e-4 was measured
    on a transformer our own run 5 showed to be 0.02 under-converged, which inflates every input-side gain. Not a
    one-axis edit; it needs a custom pretraining loop inside pytabkit. Not scheduled.
- **Timing**: research only, no GPU used. Local harnesses untouched.
- **Status**: plan gains **§2d** (this pass, with per-row provenance) and a dated correction inside §5; findings **72–73**
  logged. **The decision is the owner's and it is now priced, not vague**: the remaining **0.00103** to rank 20 is
  composition-shaped, the single-model law excludes blending, and the one lawful shape for it — **one fitted model
  trained on a stack's predictions (distillation)** — is *not* what M6 measured (M6 tested averaging), but does require
  the owner to rule that **third-party public OOFs are a permitted target**, since the 2026-10-11 law was written about
  our own saved OOFs. Nothing is authored until that is answered.

---

### 23-10-2026 — continuation: the goal pass — new models, LLMs, a priced forward plan, and a calibrated composition probe

- **Goal**: the owner's standing instruction — search harder for what can still move the LB, including new model classes
  and whether a custom model or an LLM can be trained on this data, and turn it into a plan built on what we have already
  measured. Five research agents (LLM-for-tabular; the 2024–26 tabular-model field at T4 budget; AUC-aligned objectives;
  this month's threads; public-OOF availability) plus one authoring pass. **No model was run and no version was authored.**
- **Experiments**: two new artefacts. **`PLAN_V36_ONWARD.md`** — Phase A = four named arms on frozen V18 (**V36** folds
  5→10, **V37** `plr_sigma` 2.33→32, **V38** k-NN label mean on the original rows, **V39** XGB `max_bin`), with a
  pre-declared stop-loss after V36+V37; Phase B = the model-class verdicts below, gated by an admission test written
  *before* the reports arrived; Phase C = composition, handed to a measurement instead of a values argument; §6 submission
  rule; §7 spends ~10 GPU-h explicitly, including ~2.5 h deliberately unspent. **`S6E10_K3_StackMarginal.py`** — CPU-only,
  stdout-only, owns no V#: it ingests attached public OOFs, refuses label/fold columns, converts probabilities and
  z-scored logits alike, removes the exact duplicates (24 of 40 in the public pool), and prints our V18 column's drop-one
  marginal **plus a cluster-level marginal** (single-linkage |r|>0.90) with a permutation floor, against three
  pre-declared readings. Verification: 7 static probes clean (`[P2] docstring 25/35`, `[P5] to_csv 0`, `[P7]` no library
  kwargs) and the instrument **calibrated on synthetic members — ALL PASS on 11 checks**, including the check that
  matters: a planted independent family reads **+0.294 joint vs +0.002 best solo**, and no noise column ever reads
  positive. Local only, no GPU, no data read.
- **Key Learning**:
  - **The field verdict is that we already hold the large-data leader.** The same >70k-row Elo table that ranks
    xRFM(T+E) at 1563 puts **RealMLP(T+E) at 1721, above every tuned GBDT** — so our plateau is the data's, not a model
    choice. xRFM is admitted as **K4, a ~20-minute decorrelation probe only** (published 365 s per 1k rows ≈ 71 GPU-h at
    our size as-configured), with pre-declared go/no-go: correlation with V18 < 0.98 **and** solo ≥ 0.955 **and**
    extrapolated full cost ≤ 2.5 GPU-h. TabPFN-3.5/3, TabICLv2, TabFM, TabDPT, CARTE/XTab/TP-BERTa/TALE, diffusion
    generators, iLTM/GTab are killed with reasons in plan §4 — benchmark caps below our row count, memory above a 16 GB
    T4, or published last-place at >70k.
  - **LLMs, answered twice — by regime and by arithmetic.** TabuLa-8B's win is at 1–32 labelled samples (we have
    699,635); a 2026 scaling study reports LLM normalised score at or below majority-class guessing by d≳16 (21 features).
    And 699,635 rows ≈ **63M tokens per epoch** = **20–32 T4-hours against ~10 GPU-h left in the competition**. The one
    honest experiment (a LoRA scaling-slope probe, 6–8 GPU-h) is recorded specifically so it is **not** run: it would eat
    Phase A + Phase C to test a regime our data sits outside. The single 2026 Playground win that used LLMs used them as
    *engineers*, which is an argument about composition, not architecture.
  - **Metric-aligned loss is not admitted, expected +0.0000…+0.0002.** Ranking-loss AUC gains scale with positive-rate
    *sparsity* (+0.0008 at 3.3% CTR, and **−0.263** in a dense regime); ours is 44.4%. Pairwise logistic loss is already
    AUC-*consistent* (Gao & Zhou), so it shares log-loss's optimum. The APIs cannot express our metric (XGB/LGB LTR needs
    per-query `GroupId`; CatBoost states AUC "can't be used for optimization" for ranking), and exact pairs are ~1.2e11
    ≈ 1–2 TB. **What is admissible is a four-minute free screen**: reweight one fold with `scale_pos_weight` 1.5 / focal
    γ=1 — if AUC moves >±0.0001 the finite-model intuition is wrong and the channel reopens.
  - **A mechanism about our own gate, found outside: OOF rows are scored by one model, the shipped test row is the mean
    of all of them** (747182: +0.00023 OOF → +0.00005 on a held-out fake test). Our CV-first gate therefore
    *under-prices* test-time-averaging arms, which is exactly V36's mechanism — hence V36 is gated on pooled paired DeLong
    plus the 2027 reproduction, not on fold arrays. It does **not** re-open bagging: V32 already showed more members
    inside the model moved +0.000023.
  - **Two process lessons, both about my own writing rather than the data.** (1) My edits swallowed a section heading for
    the second time in this file, so structural edits to a long markdown doc now go through a stdlib patch script whose
    every anchor is asserted for uniqueness — and the script was **calibrated by running it on two known-bad copies**
    before trusting it on the real file; both guards aborted before writing. (2) A figure I had carried in notes — that
    public-top-2 selection misses the best private file for 7.2–16.5% of teams — **could not be re-sourced, so it is not
    in the plan.** §6's diversification rule now rests only on our own measured offset law plus the printed
    selection-overfit price, with that provenance stated in the text.
  - **Correction to my own plan, made in the same pass and named here: `plr_sigma` 2.33 is NOT the pytabkit default.**
    Fetched `DefaultParams.RealMLP_TD_CLASS` from pytabkit 1.7.3 (`pytabkit/models/sklearn/default_params.py`, PyPI sdist
    cross-checked against the GitHub raw file): **`plr_sigma` defaults to 0.1**, and 19 of the 21 numbers in V18's dict
    are not defaults (`lr` 0.053 vs 0.04, `wd` 0.0236 vs 0.02, `sq_mom` 0.988 vs 0.95, `embedding_size` 5 vs 8,
    `max_one_hot_cat_size` 18 vs 9, `p_drop` 0.05 vs 0.15, hidden [512,256,128] vs [256,256,256], `act` silu vs selu,
    `ls_eps` 0.01 vs 0.1, `n_epochs` 3 vs 256, `n_ens` 8 vs 1, `lr_sched` flat_anneal vs coslog4, `wd_sched` cos_log_15
    vs flat_cos, `p_drop_sched` expm4t vs flat_cos, `bias_init_mode` neg-uniform-dynamic-2 vs he+5,
    `add_front_scale` False vs True, `plr_hidden_2` 8 vs 4, `plr_lr_factor` 0.1151 vs 0.1; only `batch_size` and
    `plr_hidden_1` match). Two consequences: the plan's sentence "our value is still the library's" was **wrong** and is
    corrected in plan §1.4/§3, and it cuts against the arm's rationale in a useful way — Deotte moved **0.1 → 32** while
    our recipe already sits at 2.33, i.e. 23× toward his endpoint, so part of his +5.5e-4 may be banked. **V37 is now a
    two-rung ladder {8, 32}** (2×17.6 min) because that turns an ambiguous single null into a shape: it measures whether
    goodpjw2008's tuning saturated on his frame. The positive side of the same datum: it upgrades plan §1.4 from an
    inference to a measurement — our control is a *fully* hand-tuned configuration on someone else's frame, which is
    precisely the reason a knob arm is still live here at all.
- **Timing**: no GPU used; local authoring, static checks and calibration only.
- **Status**: **plan is live, nothing is scheduled to run until the owner reads it.** K3 is ready to run the moment public
  OOF datasets are attached to a CPU kernel; it is the gate on the only door with a claimed +0.0008 behind it, and its
  three readings are pre-declared so a null will be a measurement rather than an absence. Phase A's four arms are priced,
  not authored — authoring waits for the owner's go on the plan. The single-model law, the ruler, the offset law and the
  V18 control all stand unchanged; the campaign's ceiling statement survives this pass with more corroboration than it
  had before it.

---

### 23-10-2026 — continuation 2: K3 ran locally, and running it found three defects its own calibration could not see

- **Goal**: the owner asked for K3 to be run locally — it is CPU-only, and every artefact it needs except the *public*
  members is on this box.
- **Experiments**: `run_k3_local.py` and `run_k3_ownpool.py` (temp) **exec the shipped file's own source** with only its
  two input roots and the `ours` regex substituted, so every number below is the real probe's arithmetic on the real
  699,635-row frame, not a paraphrase of it. Run 1 loaded all 35 of our OOFs, id-aligned them, and **refused** —
  `need >= 20 PUBLIC member columns; found 0`, which is the guard doing its job (no public OOF library exists off-Kaggle:
  HF search for `s6e10` / `playground-series-s6e10` / `s6e10-oof` returned nothing, and Kaggle dataset download needs
  credentials this box does not have). Re-pointing `ours` at **v18 only** turns our other 34 arms into the pool, so the
  instrument could finally be run for real — and it died **three times in a row**:
  1. the "informative" filter was written `AUC_FLOOR < s < 1 - AUC_FLOOR` = **0.60 < s < 0.40, an empty range**, so it
     dropped all 35 columns and reported a 0-column stack; fixed to keep a column informative in *either* direction;
  2. `cluster_marginals` crashed (`Found array with 0 feature(s)`) because single-linkage at 0.90 chained the whole pool
     into one group, and dropping that group empties the stack; it now skips an all-covering group and prints the size;
  3. **the duplicate collapse deleted OUR OWN column and ranked someone else's copy instead** — members are ordered
     public-then-ours, so a first-index-wins rule always removes ours; it crashed next at `max()` on an empty list.
     Union-find now keeps our column in its own group and prints which twins it duplicates.
  New calibration `check_k3_realrun.py` execs the shipped source over a synthetic attached pool: **10 checks across
  three scenarios, all OK, and all 7 refuse when the old broken filter is restored** — the previous instrument check had
  *reimplemented* the loader and filter, which is exactly why it passed a file that could not run.
- **Key Learning**: **running the file is a different activity from checking the file.** The three defects were all in
  code paths my static checks and synthetic-paraphrase checks could not reach: `[P3] name resolution` and `[P6] dead
  constructs` say nothing about an inverted comparison; a paraphrase of a filter tests the paraphrase. Numbers from the
  first real run of the instrument on our own ladder:
  - **Our 35 OOFs are 12 columns.** 78 near-duplicate pairs at logit corr > 0.999 collapse the pool to 12 independent
    columns, and even those 12 form **one** connected group at |r| > 0.90 — our ladder is a single correlation blob, not
    35 legs. V18's OOF specifically sits at 0.99942 / 0.99974 / 0.99973 / 0.99971 against **V12, V28, V29, V32** (near-,
    not bit-identical: max logit gap up to 1.35).
  - **V18's marginal contribution to a nested-CV logistic stack of our other arms is +0.000025** (82nd percentile of the
    other members; their median is +0.000001, max +0.000044 from V27). The 12-column stack scores **0.961518** vs V18
    solo 0.96129 — the whole ladder combined buys +0.00023 on OOF, which is ~2 seed-ticks.
  - **The instrument's own floor is +0.000002**, i.e. below the +1e-5 bar, so the probe *can* see at this scale; the
    per-column distribution (+1e-6 for most members) is the saturation statement, in our own numbers.
  - Alignment re-proved on the way: V18's solo AUC read off the local files is **0.96129**, the recorded CV band exactly.
- **Timing**: local CPU only, ~3 min per full run; no GPU, no submission, no `V#`, nothing written except stdout.
- **Status**: K3 is **fixed and locally proven, and still cannot answer its own question** — the public members are only
  on Kaggle, so the run that matters is a Kaggle CPU kernel with the OOF datasets attached (now also worth attaching the
  whole `oof/` folder, since the near-duplicate NOTE is informative). Do **not** read the +0.000025 as the §5 verdict: the
  pre-declared bands were written against the public pool. Plan §5 records what the local dry run established.

---

### 23-10-2026 — continuation 3: Phase A landed, our largest CV gain converted to nothing, and the offset law needed correcting

- **Goal**: run the two arms the owner approved (V36 folds 5→10, V37 `plr_sigma` {8, 32}), land them, and answer
  "why is the LB saturated" with our own numbers instead of with sentiment.
- **Experiments**: **V36** 10-fold on V18's byte-identical frame → **OOF 0.96139 (our highest CV ever) / LB 0.96104**,
  33.8 min, fold mean 0.96140, `pooled − fold mean` −0.000004, best epoch 3 on all ten folds. **V37** ladder →
  σ=8 **0.96128**, σ=32 **0.96121** against the parent's 0.96129, **LB 0.96107** for the rung of record (σ=8, chosen by
  the pre-declared rule), 30.8 min. The rungs are paired per fold (same splits, same prep): σ=32 − σ=8 = +0.00005,
  −0.00013, −0.00014, −0.00010, +0.00002 ⇒ mean **−0.00006**, one ruler.
- **Key Learning**:
  - **The offset constant belongs to the (frame, protocol) pair, not to the frame.** The law as recorded said
    `LB = CV − per-frame constant` (−0.00023 on V18's frame), reproduced to 0.000003 *across seeds*. V36 changed only
    the fold count and its constant is **−0.00035**. A fold-count arm's CV is therefore not on the same scale as the
    arms it is compared against, and **+0.00010 here is not a candidate gain** — part of it is a different estimator.
    This is a correction to a recorded law and it is written as one, in a dated appendix.
  - **Why the board is saturated, measured four ways, all of it ours.** (1) The net family's own CV→LB scatter is
    **0.000174**; V36's +0.00010 is 0.6 of that σ, so the conversion was never going to be visible — an LB tie at this
    effect size is arithmetic, not bad luck. (2) Our 35 OOFs collapse to **12 independent columns** and even those form
    one |r| > 0.90 cluster (K3 local): there is no diversity in the ladder to bank. (3) Every LB delta since V18 has
    stayed inside the ±0.00006 seed tick — V29 +0.00004, V32 +0.00002, V28 −0.00001, V34 −0.00006, V37 +0.00001,
    V36 −0.00002: **six arms, six ties, both yardsticks**. (4) The gap to the 0.96213 plateau is **0.00103 ≈ 17
    rulers**, and that plateau is one widely-submitted stack artefact ~40 teams share; the front's own price list puts
    a new *family* at +1e-4 and everything else at +1e-5…+3e-5.
  - **`plr_sigma` was already turned.** pytabkit's default is 0.1, V7's inherited bundle put 2.33 in it, and Deotte's
    +5.5e-4 describes 0.1 → 32. Going to 8 costs nothing measurable and 32 costs −0.00006, so the plateau sits between
    2.33 and 8 and his gain was banked, not left on the table. The method that found it: fetch the library's default
    dict before claiming a knob is untouched (finding 75) — the claim was checkable, it was wrong, and the correction
    priced a run.
- **Timing**: 1.08 GPU-h of the ~10 h available (V36 33.8 min, V37 30.8 min, 2 rungs × 5 folds with prep shared).
- **Status**: **V18 remains the standing control; V29 (0.96110) remains the highest LB.** The plan's stop-loss did
  **not** fire (it needed both arms < +0.00005 and V36 read +0.00010), but its other pre-declared clause does: *V39 is
  dropped without running if V37 is null* — **V39 is closed**. V38 (k-NN label mean) is the only Phase A arm left and
  it asks a different question (a new *estimator* of the one channel type that ever worked here, not another value of a
  knob at its plateau); ~25 min, the owner's call with the conversion arithmetic above in front of them. Paired DeLong
  for V36/V37 vs `v18_oof` is still pending the two vectors — the CV deltas here come from stdout, and that
  distinction is kept on the record.

---

### 18-10-2026

- **Goal**: land the five runs the owner sent (V31–V35), answer "why did the LB not improve", re-read the public
  board for anything new, and re-aim the queue against what the numbers now support.
- **Experiments**: **V31** confident-row exclusion (q=0.0387 by nested |p−y|, parent V18) — **CV 0.95930 / LB 0.95907**,
  **−0.001987 CV, folds 0/5, LB −0.00199**, 25.6 min. **V32** `n_ens` 8→16 — **CV 0.96131 / LB 0.96108**,
  **+0.000023 CV with 5/5 folds improved**, 29.1 min. **V33** V18 at seed 2027 — **CV 0.96123 / LB 0.96100**.
  **V34** V29 at seed 2027 — **CV 0.96124 / LB 0.96106**. **V35** `sampling_method='gradient_based'` (MVS) on V11 —
  **CV 0.96101 / LB 0.96058**, **−0.000123 CV, folds 0/5**, 5.9 min, and the in-run `booster params` print proves the
  parameter reached the booster, so it is a mechanism null and not a swallowed kwarg.
- **The day's real result is the ruler, and it re-prices the whole ledger.** V33 is V18 with nothing changed but the
  seed: **pooled CV −0.000057, fold mean −0.000056, LB −0.000060.** So the gate's magnitude bar (Δ > +0.00005) sits
  *below* one tick of the noise the gate is meant to detect — V27's +0.00002, V32's +0.000023 and V29's +0.00004 LB
  are all sub-tick, and the three arms that clear two ticks are exactly the three frame channels (twins +0.00016,
  EV-on-tree +0.00012, EV-on-net +0.00012). **§7 now requires Δ ≥ +0.0001 and a seed-2027 reproduction**, and
  V18's "best single" is recorded as a band — CV 0.96123–0.96129, LB 0.96100–0.96106 — not a point.
- **Second structural finding: the public board is CV minus a per-frame constant.** −0.000227/−0.000230 for four arms
  on V18's frame (V18, V31, V32, V33 — a 0.000003 spread) and −0.000180 for both V29-frame arms (V29, V34, identical),
  −0.000493/−0.000430 on the tree frame. Consequence stated plainly: **LB cannot improve unless CV improves**, the
  "LB luck" this file has occasionally celebrated is an offset difference between frames, and the previously recorded
  net-family board scatter of 0.000174 was mixing frames and budgets rather than measuring board noise.
- **Two mechanisms closed, both decisively.** V31 deleted the 3.87% of training rows a strong nested model most
  confidently disagreed with (mean |p−y| **0.947** dropped vs **0.087** kept) and lost 0.002 AUC on every fold: those
  rows are the frame's sharpest signal, not noise — AUC is scored against the *observed* labels, so deleting the
  surprising ones removes the metric's cheapest credit. Third independent confirmation the labels are near-clean
  (with K1b M2(a)=0 duplicates and M2(c)'s ρ̂ = −0.2149). V35 killed the plan's biggest external hope: MVS's published
  **+0.0012 on 97/153 datasets** came back **−0.00012** with a *wider* best-iteration spread (370 rounds), i.e. it
  destabilised early stopping instead of steadying it, at no time saving.
- **Bagging is exhausted**: `n_ens` 1→8 was **+0.000886**; 8→16 is **+0.000023** for ~1.8× the fit. The campaign's
  largest lever has a measured 38× collapse in effect per doubling, so no further rung is proposed. V32's calibration
  pre-declaration also failed to fire — pooled minus fold-mean is −0.000007 at 16 vs −0.000005 at 8 — which retires
  the "pooled OOF hides a bagging/epoch bias" hypothesis on this frame at this budget.
- **Public board, re-read (external, author-printed, none verified by us).** The front's method is now unambiguous
  and it is **stacking over shared public OOF libraries on the community split** — `StratifiedKFold(5, shuffle,
  rs=42)`, the same split as ours: kratosyan "Stacking every public OOF library | nested CV **0.96205** — LB
  **0.96167**"; goodpjw2008's ladder "Twin Stacks + Family Reg | LB **0.96187**", "TabPFN + Route Categories | LB
  **0.96160–0.96164**", and — directly relevant to our one-channel history — "Auxiliary-Task Features | LB
  **0.96129**"; rabari9999 roadmap LB 0.96188; najiama OOF CV 0.96192 / LB 0.96165. A 53-CSV "21 models + 247-pair
  stack" OOF dataset is the shared substrate. Independent corroboration of two of our own measurements: a
  competitor's original-trained model scores **0.995 on the original but 0.954 on synthetic** (our K1b M2(c) printed
  **0.95465**), and **adversarial AUC ≈ 0.50** between train and original (our M5: the class that wins agrees in
  100% of shared cells). Two published single-parameter arms we have never measured: Deotte's
  **`plr_sigma` 0.94→32 = +5.5e-4** on RealMLP ("only lr/wd/lr_sched/n_epochs/n_ens matter otherwise") and the
  XGBoost analogue **`max_bin`=4096** — and `max_bin` is genuinely unmeasured here, because V14's archived OOF is a
  bit-identical copy of V2's, so its row has always been closure-only.
- **A proposal of mine, refuted by the same pass.** I had queued "train on synthetic ∪ original rows" as a
  data-axis arm, reasoning from the learning curve (+0.0006 per doubling) and from M5's shared law. External
  measurement says the opposite: shahed-swe's ablation prints **original-row append −0.00035**, and the discussion
  title "Original rows hurt, most others are within noise" (karttikjangid05) says the same thing twice. **Not
  authored**, and the two facts are consistent with our own ledger: the original earns its place as a *teacher and
  lookup*, which is how V2 built it (+0.00040) and how it still dominates the importances (0.4177 gain share).
- **Repo damage, repaired and recorded.** The concurrent session double-landed this batch — five identical rows in
  `Memory/public_scores.md` and two identical five-entry blocks in `Memory/training_logs.md` — and had deleted
  `PLAN_V21_ONWARD.md`, `PLAN_V31_ONWARD.md` and `S6E10_K1b_VerificationBattery.py` from the working tree with no
  commit. The rows/blocks were deduplicated (byte-identical copies, verified before deleting), the three files were
  restored from `HEAD`, and V31–V35's scripts are now in `Previously trained files/Archieve/` as intended.
- **A public dispute our own data settles - and I first got it wrong, then computed it.** A public author is credited
  in our ledger with `Departure/Arrival time convenient` inverting at **rho = -0.90**, and another competitor contests
  it at **-0.05**. My first reading of our K1b level table declared -0.90 impossible. It is not. From
  P(sat=1) = 0.478/0.506/0.453/0.449/0.402/0.437 for levels 0-5, the **Spearman between level rank and rate rank is
  -0.900 over levels 1-5** (-0.886 including the unrated marker) - the published -0.90 - while the **row-level
  point-biserial is -0.0493** - the contested -0.05. Both authors were right, at different granularities, and they only
  disagree if the 0.104 rate range is forgotten: the *ordering* is firmly inverted, the *information* is small. That one
  line is the whole story of this column, and it explains why correcting the sign convention has never produced CV.
  Two further corroborations in the same external pass, all author-printed: ~0.954 for an original-trained model scored
  on synthetic labels (our K1b M2(c) printed **0.95465**), and 17 train + 4 test exact overlaps with the original (our
  `prepare_original` removed **21**).
- **Key Learning**: **we were arguing about numbers the instrument cannot read.** The campaign spent twenty arms on
  ±0.00002 differences and only built a ruler when it ran its own control twice — and once the ruler existed, three
  "gains" and one "record LB" turned out to be reseed artifacts, while the two genuine mechanism results of the day
  (V31 −0.002, V35 −0.00012) were both *larger than any gain we ever claimed*. The honest queue is therefore short:
  two published in-family parameter moves (`plr_sigma`, `max_bin`), the owner's decision on TFM weights, and the
  owner's decision on the single-model law — because the front's +0.0008 is, on the printed evidence, stacking, not
  a feature we have not thought of.
- **Third external pass, and its headline is a negative result worth more than any trick: no author anywhere public
  has printed a *single* model at LB ≥ 0.9615.** The highest single-model public LBs printed are **0.96098** (one
  LightGBM, thread 747358) and **0.96082** ("best single model, RealMLP v4"); the best single-model **CV** prints are
  0.961492 (TabPFN-3.5 + 9 route/distance features, Deotte) and 0.961344 (RealMLP + aux features). **V18's band,
  CV 0.96123–0.96129 / LB 0.96106, is therefore at the public frontier of the single-model class, and its LB is above
  every single-model LB on record.** Teams 1–15 (~0.96284 → 0.96197) disclose no architecture. This is the answer to
  "why is nothing moving the LB any more", and it is a ceiling statement rather than a search statement.
- **Two corrections and one confirmation from the same pass.** Corrected: I had written that TabPFN-3.5 "is printed
  above our band", which was one author's number — goodpjw2008 prints full-context TabPFN **alone at 0.960729**, below
  our model, using it only as a stack member (LB 0.96156), and a third author reports engineered columns made it worse
  in all five variants tried. So door 2 of §8 is now "plausible, not established". Corrected: the aux-task channel,
  which I had listed as a possible untried idea, is **ours** — 16 label-free targets cross-fitted by multi-class XGB
  into `aux_ev_*` (Σ k·P(k)) plus `aux_p_*`; their single-model effect is +0.00010 on trees and +0.00017 on RealMLP
  against our measured +0.00012 and +0.00012, and *their own ablation* kills the residual hope (all 30 columns
  +0.00004, the 16 probability columns alone **−0.00010**, "twelve smarter variants all lost") — the −0.00010 being
  our V28 null restated. Confirmed: the only public ablation built to a real noise standard — one-at-a-time against a
  LightGBM baseline, **paired 5-fold CV over 3 seeds with random-feature controls** (745098) — independently reaches
  our conclusion: original rows **−0.00038**, ratings-as-categories / Arrival-Delay filling / mean-service rating
  **inside the noise**, and its author judges aux-task features by paired CV because the LB effect was **+0.00002**
  against cited public-LB noise ~0.0003. Its one beyond-noise FE win, **Flight-Distance digits +0.00036**, is closed
  here by our own 2026-10-12 measurement (conditioning on FD//5 flips it to −0.0100, z = −3.3 — a distance confound)
  and by the 12 constant digit columns V35's own log drops.
- **Status**: ✅ five arms landed with CV and LB; ✅ ruler measured and the gate re-based; ✅ the offset law
  established on six arms across two frames and two seeds; ❌ V31 row-noise selection and V35 MVS closed by
  measurement; ⚠️ V29's best-LB claim retired as an offset-plus-seed artifact (V18 stands as control); ✅ public
  board re-read and two of our measurements independently corroborated; 🔬 `PLAN_V31_ONWARD.md` re-aimed at
  V36 `plr_sigma` / V39 `max_bin`, with V38 still behind the owner's weights decision.


---

### 18-10-2026 — continuation: the tabular-transformer thread is read, and the post-V35 queue is erased

- **Goal**: read the discussion the owner pointed at (751334) and the rest of the forum, judge whether it beats our best
  single model, and act on the owner's instruction to remove the plans that came after V35.
- **How it was read, at last.** Kaggle's discussion pages render for an ordinary browser session, so the thread body was
  pulled from the live DOM rather than a proxy. That matters for the record: the previous external passes were limited to
  search snippets, titles and GitHub READMEs, and one of my claims is corrected below as a result.
- **751334 is Chris Deotte, "Part 2 — How to build a Tabular Transformer!", a day old: Q-FT v2.** A ModernBERT-base
  encoder (149M, all weights trained, no LoRA) reading a row as an **unordered set** — `position_ids = 0` for every
  token, so the order shuffling and 5-order averaging of Q-FT v1 both disappear (reordering now changes the output by
  3e-6), head = mean of the 23 non-CLS vectors → Linear(768→1). Two additions on top: **side features (+4e-4)** and
  **replaced-token-detection pretraining (+3e-4)** — 15% of the 21 values swapped from a random other row, per-token BCE,
  3 epochs over 1.13M *unlabelled* train+test+original rows, run once and fold-independent. Fold-0 ladder 0.960368 →
  0.960655 → 0.960977; **5-fold OOF 0.960798** (v1 was 0.959795); ~6.5 min/fold plus 18 min shared pretraining, 11.4 GB
  peak, one A100. Their own negatives: ModernBERT-large no better than base, DeBERTa-v3 worse, **a randomly initialised
  ModernBERT scored the same as the pretrained one**, 10 epochs of pretraining worse than 3, and supervised pretraining
  on the original dataset helped nothing at all.
- **Verdict: it does not beat our best single model, and it cannot be compared on LB at all.** Their print is a
  **CV** of 0.960798 against our control band 0.96123–0.96129 — about **0.00049 below V18** — and no submission number
  is attached to it. The transferable part is also already ours: their side features are count-in-train+test,
  count-in-original and the ratio, per-route mean/spread/original-mean, a label-free self-prediction residual, and
  smoothed original log-odds, i.e. our org stats + route profile + `_og_teacher`, which V2 bought for +0.00040 in
  2026-10-02. Their own ablation makes that explicit — the same feature block is worth **+4e-4 to the transformer but
  +1.6e-3 to a GBDT**. The one genuinely untried mechanism is **RTD**: label-free self-supervision over unlabelled rows,
  which has no GBDT analogue and no cheap RealMLP analogue either, and whose published effect (+3e-4) is on a model
  starting 0.0005 behind us.
- **Corrections and corroboration from the same read.** Corrected: I had recorded TabPFN-3.5 as "printed above our band"
  on one author's 0.961492; on the same forum goodpjw2008 reports full-context TabPFN-3.5 **alone at 0.960729** — below
  our model — used only as a stack member (LB 0.96156), and a third author gets 0.961178 (5-fold) / 0.961275 (10-fold)
  while finding engineered columns hurt it in all five variants tried. So the foundation-model door is now recorded as
  *plausible, not established*. Corroborated: the noise-floor thread (745098) reaches our conclusion by an independent
  route — paired 5-fold CV over 3 seeds with random-feature controls puts original rows at **−0.00038** and ratings-as-
  categories / delay filling / mean-service rating **inside** the noise, and its author judges an aux-task gain of
  **+0.00002 LB** by paired CV because public-LB noise there is ~0.0003. Its one beyond-noise FE win, Flight-Distance
  digits **+0.00036**, is closed locally: we measured that effect as a distance confound (conditioning on FD//5 gives
  −0.0100, z = −3.3) and V35's log drops 12 constant digit columns.
- **The queue is erased.** V36 (`plr_sigma`), V37 (net schedule), V38 (bagged-context TFM), V39 (`max_bin`) and V40
  (k-NN on the original) are removed from `PLAN_V31_ONWARD.md` §3 — none was ever authored, so nothing was renamed or
  deleted from the archive — and the section now carries an erasure record naming each removed arm, what it would have
  tested, and why it had been on the list. §8 is rewritten as the diagnosis that justifies stopping rather than as a
  forward queue. The reason is arithmetic, not mood: with the ruler at ±0.00006 and every surviving candidate published
  at +1.5e-4 to +5.5e-4 *elsewhere*, four more arms could only have produced more points inside the noise band, while
  nobody on this board has produced the remaining ~0.0008 with a single model.
- **Key Learning**: reading the thread in full, rather than through a search snippet, is what turned "a new single model
  above us" into "a new single model 0.00049 below us". Two of my external claims had already hardened from titles; the
  correction is recorded above rather than quietly overwritten.
- **Status**: ✅ 751334 assessed and §2c updated with the full Q-FT v2 method; ✅ the TabPFN over-claim corrected;
  ✅ V36–V40 erased with a named record; ✅ external noise-floor corroboration landed; ⚠️ the two doors (composition,
  external weights) remain the owner's and are recorded as decisions, not plans.

---

### 17-10-2026

- **Goal**: land V29 and V30, decide what the record now supports, and **scrap the remaining ladder** in favour of a plan
  aimed at what is actually moving the board.
- **Experiments**: **V29** (four Stage-1 keys ported to V18's net frame, +12 per-fold columns, 301→386 expanded cols) —
  **CV 0.96128 pooled / 0.96129 fold mean, LB 0.96110, Gap −0.00018, 17.6 min** ⇒ **−0.000007 CV vs V18, folds improved 2/5
  with one tie: gate FAILS on both bars**, and its +0.00004 LB — the highest number this campaign has ever submitted — is
  **0.23σ** of the net family's board scatter. **V18 remains the standing control.** **V30** (soft-label ladder, three
  rungs) — **fold means 0.96111 / 0.96111 / 0.96111** across α ∈ [.20, .70] and es ∈ [150, 400] with **the collapse guard
  firing on every fold of every rung** (Spearman student-vs-nested-teacher 0.9928–0.9973 vs a 0.990 limit), the nested
  teacher itself scoring 0.96100 against the student's 0.96111, and α=0.70 needing 1,387–3,251 rounds to buy nothing ⇒
  **target mixing is closed by its own pre-declared falsifier, not by a score**. V30's control rung also **re-measured
  V20's recovered CV** (0.961104 from the archived vector vs 0.96110 printed fresh, and LB 0.96060 identical) — the first
  validation of a recovered number in this ledger. **Scrapped before running** on the owner's instruction: `V31_RegionAlpha`
  (its parent channel just measured zero across a 3.5× α range) and `V32_NetBaggingCapacity` (a three-rung ladder costing
  70 GPU-min). Both move to the archive as RETIRED, neither deleted. **Correction stated in the same breath**: scrapping
  V32 on cost is defensible, but bagging is the one knob this campaign measured a *large* effect on — V5 (`n_ens=1`) → V6
  (`n_ens=8`) was **+0.000886 pooled / +0.000598 fold-mean** — so its premise is unfalsified and the new plan re-proposes
  it as a single rung (`n_ens` 8→16) rather than a ladder. **V33 keeps its occupant** (V18 at `RANDOM_SEED=2027`, authored and verified, still unrun) and the new plan adds **V34 = V29 at 2027**, so the grid {V18, V29} x {42, 2027} can finally say whether 0.96106 vs 0.96110 is a model difference or the same model under another seed. **V31 and V32 are reused by new arms** (row-noise weights; one bagging rung) because their retired occupants never ran and own no ledger row.
- **Timing**: 57.3 GPU-min for the two arms (17.6 + 39.7); **≈2.05 GPU-h consumed of the ≈4.1 h** the old ladder estimated.
- **Key Learning**: **the two-architecture experiment is finished and unanimous** — rating EVs +0.00012/+0.00012, posteriors
  −0.00004/−0.00001, TE keys −0.00005·−0.00004/0.00000: **three channels tested on both families, three transfers, sign and size
  preserved**. So "the tree could ignore it, the net can't" is retired as a reason to re-run anything, and a null closes a
  channel after one clean reading. And nothing this campaign has ever measured — on either architecture — exceeded **+0.00016 CV**,
  while the board's front (0.96190 at rank 20, 0.96273 at #1) is **~0.0008 of LB above us**, which is roughly everything we have
  ever gained, again. That gap is the real finding of the day: **it is not a missing feature, it is a missing kind of model** —
  and the public arms that crossed it are stacking (banned here by the owner's law) and tabular-foundation-model inference
  (legal, untried).
- **Status**: ❌ ladder scrapped (V31/V32 retired, V33 re-scoped); ⚠️ best-ever LB without a control change; ✅ both arms recorded
  with CV and LB; 🔬 `PLAN_V31_ONWARD.md` is the new instrument, written against the board rather than against the ledgers' hopes.

### 17-10-2026 — continuation: the research pass that re-aimed the plan at variance

- **Goal**: answer "what actually moves the score" from the literature and from our own ledger, then rebuild the queue
  around that direction only — no feature arm, no arm without an external effect size behind it.
- **Experiments**: none ran (a research and authoring pass). Local verification of the dataset facts was **attempted
  and refused** — the project boundary blocks reading `Dataset/*.csv` for EDA even with the owner's grant in the goal
  text — so every number the new plan depends on is routed into `S6E10_K1b_VerificationBattery.py` (committed,
  stdout-only, M1–M7) rather than asserted here.
- **The two findings that changed the plan.** (1) **Symmetric label noise cannot be corrected for a ranking metric.**
  With flip rate ρ the observed posterior is `π_ρ = ρ + (1−2ρ)·π`, strictly increasing for ρ<0.5 (Natarajan et al.
  2013; Blanchard et al. 2016), and AUC is invariant to monotone transforms: the ~4% flips compress the ceiling (our own
  bracket 0.9620–0.9630) **without changing the optimal ordering**. That explains V30's null mechanistically and
  pre-declares any row-reweighting arm to zero. (2) **Config beats features here, and the ranking is ours, not
  borrowed**: `n_ens` 1→8 **+0.00089**, the net's proven recipe +0.00025, representation +0.00016, EVs +0.00012, every
  other arm ≤0.00002. The largest published single-parameter AUC effect of that kind is **Minimal Variance Sampling:
  +0.0012 ROC-AUC, first on 97 of 153 datasets** (Ibragimov et al., KDD 2019) — 6× our biggest channel, and a variance
  lever, which is the only mechanism (1) leaves open.
- **Arms authored and verified**: `S6E10_V32_NetBaggingRung.py` (`n_ens` 8→16 on V18; fold-mean read, with the
  pooled−fold-mean shrink test pre-declared), `S6E10_V34_V29SeedRepro.py` (seed 2027, completing the
  {V18,V29}×{42,2027} grid), `S6E10_V35_GradientRowSampling.py` (`sampling_method='gradient_based'` on V11, 6 min,
  constructor round-trip plus a `get_xgb_params` proof, since the sklearn wrapper forwards unknown keys silently). All
  three pass the six calibrated checks; V35's AST diff touches only `CFG`/`XGB_PARAMS`/`__main__` with zero bare-node
  drift, which is what proves the EV-predictor and teacher dicts stayed still.
- **Queue**: `PLAN_V31_ONWARD.md` **v2** replaces v1 in place. v1's row-noise arm → **V39** (demoted by the invariance
  argument); `DropInertAggregates` **dropped unauthored** (a runtime cleanup, not a score lever); the TFM probe →
  **V38**, re-scoped to the published **bagged small-context** recipe, because a naive single TabPFN-3.5 on this board
  printed **LB 0.95739** and the independent benchmarks give boosted trees the large binary tasks. New: **V36**,
  distillation from a *diverse* in-run multi-family teacher — the only route to the front's gains the single-model law
  permits (a teacher target is explicitly legal), gated on K1b M6 because V30 died precisely for lack of diversity
  (V27 measured tree↔net Spearman **0.9599** where V30's teacher sat above 0.99) — and **V37**, one net schedule value
  read from pytabkit's own list at run time.
- **Closed by evidence, not by taste**: monotonic/interaction constraints (0.1–2.9% AUC cost, **8.6%** on a
  mis-specified direction, parity at best — and our ordinals encode 0 as "unrated", which *is* that trap);
  noise-cleaning on GBDTs (>99% detection yet corrections lowered test performance; "naturally robust");
  pseudo-labelling at our 700k-labelled : 300k-unlabelled ratio; and **label smoothing**, which the RealMLP paper states
  outright harms AUROC — that killed the most tempting parameter on this board before it cost a GPU-minute.
- **Status**: 🔬 queue rebuilt on measured mechanism; ⚠️ two owner decisions open (the single-model law; attaching
  external pretrained weights); ✅ three new arms committed and verified; ❌ no local dataset verification is possible
  under the standing boundary, so K1b carries it.

---

### 17-10-2026 — continuation: K1b's first Kaggle run answered M1 and withdrew my own M2/M3

- **Goal**: run the gate battery the plan is scheduled against, land what it measured, and repair what it broke —
  no version file, no submission, no ledger score row.
- **Run**: `S6E10_K1b_VerificationBattery.py` on Kaggle (pandas 2.3.3, numpy 2.0.2, sklearn 1.6.1, scipy 1.16.3,
  python 3.12). Frames as every archived arm reads them: **train (699635, 23), test (299844, 22), orig (129880, 22)**;
  positive rate **train 0.4436 / orig 0.4345**; the only NaN column in train is `Arrival Delay in Minutes` (**292**);
  the original contributes no column this frame lacks.
  **M1 — answered.** **6 of the 13 ordinals are not monotone in synthetic train**: Inflight wifi service,
  Departure/Arrival time convenient, Ease of Online booking, Gate location, Online boarding, Baggage handling.
  Monotone-up: Food and drink, Seat comfort, Inflight entertainment, On-board service, Leg room service, Checkin
  service, Cleanliness. **Sign agreement with the original: 12 of 13** (only Seat comfort differs). The printed
  constraint vector is `[0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]`. The offenders are all the **0-as-unrated marker**:
  wifi `0:0.887 (13k)` against `1:0.391, 2:0.270`; online boarding `0:0.620` against `1:0.105`; gate `0:0.543`
  against `3:0.337` — unrated passengers are *more* satisfied than those who rate 1–3, and the original shows the
  same shape (`wifi 0:0.997`). So a `monotonic_constraints` vector would be mis-specified on six columns, which is
  the 8.6% case in §2's table, not the 0.1% case. **Closed by measurement.**
  **M2/M3 — the printed numbers are withdrawn.** The 17-column cell key produced **437,904 cells over 699,635 rows
  (1.6 rows per cell)**, so every `n>=200` filter returned empty and "conflict mass 0.0000" measured sparsity, not
  noise. The "conditional-law ceiling 0.79021" was the same failure: 245,033 cells with 0.5% of them at the shrink
  mass means the lookup had degraded to the global prior. **M4 aborted** on `astype('string').agg('|'.join)` —
  `TypeError: sequence item 16: expected str instance, NAType found`, item 16 being that 292-null delay column.
  Wall time is not recorded because the run did not finish.
- **Repairs** (`b90c3a2`): M2 now has three readings that each answer a real question — conflict inside
  **byte-identical** rows via `pd.util.hash_pandas_object`, conflict inside a coarse 5-column key, and a **moment
  estimate of ρ** from the calibration gap of a model fitted on the original alone (`E[y] = ρ + (1−2ρ)E[p]`). M3 is
  ceiling arithmetic on our own control CV (0.961287, V18) rather than a lookup that could not resolve it. M5/M7 move
  to the coarse key. M6 drops the logistic member — it is not one of V36's teachers and a member ~10 AUC points below
  the best makes the ensemble-minus-best gate statistic answer the wrong question — and uses the orig-law model, so
  the census is exactly V36's declared teacher set. **One shared encoding** (V11's `make_key_frames` idiom, vocabulary
  built from both sources) feeds every fit and every hash, so a code means the same value in orig, train and test.
  Operational finding: a **format-on-write hook re-splits any long joined `print("\n[…]…")`**, so the battery emits
  `print()` plus short separate prints.
- **Key Learning**: **the ~4% figure this campaign has been quoting is not a symmetric flip rate.** Requiring the
  clean-data AUC implied by `0.5 + (0.961287−0.5)/(1−2ρ)` to stay ≤ 1 caps ρ at **0.019**. At 4% flips our own CV
  would imply a clean-data AUC above 1, i.e. arithmetic impossibility — so that divergence is generator re-drawing
  (which moves the marginal, not the order) or noise concentrated in rows no model can order. Plan §1(a)'s
  0.9620–0.9630 ceiling bracket is **withdrawn**; the invariance conclusion stands and is now the stricter case.
  A cheap measurement also corrected a premise I had inherited from the literature rather than tested: sparse keys
  do not silently fail, they fail **plausibly** — an empty table reads as "no noise found", which is the opposite of
  what it means.
- **Status**: ✅ M1 answered, monotonic constraints closed by measurement, plan §1(a-i)/§2/§4/§6 corrected;
  ❌ first-run M2/M3 readings withdrawn as my design error; ⚠️ **the battery needs one more Kaggle run** — M2(a–c),
  M3, M4, M5, M6, M7 are still missing, and until they print, **V31, V36 and V38 stay unauthored**.

---

### 17-10-2026 — continuation: K1b's second run closed three arms, and corrected a number I had committed an hour earlier

- **Goal**: run the re-designed battery, land all seven answers, and author V31 as the owner instructed.
- **Run**: `S6E10_K1b_VerificationBattery.py` finished in **0.5 min** of wall time and printed everything.
  **M2(a)**: **0 byte-identical feature vectors in 699,635 train rows**, so the provable label-conflict mass is
  **0.000000** — no two rows share a feature vector, therefore no label pair is demonstrably conflicting.
  **M2(b)** on the coarse 5-column key: train 1,080 cells, 309 at n≥200, **96.50%** of rows covered, minority mass
  **0.0911**; original 963 cells, 120 at n≥200, 75.44% covered, **0.0580**. **M2(c)**: an original-fitted law model
  scores **0.95465** on synthetic train, mean p **0.4605** against an observed rate of **0.4436**, so the moment
  estimator of the flip rate is **ρ = −0.2149** — negative, which a flip rate cannot be.
  **M4**: **0 of 299,844** test rows duplicate a train row on the full vector; within-train duplication **0.00000**.
  **M5**: 120 shared coarse cells covering **83.13%** of train, signed divergence **−0.00889**, weighted mean
  absolute divergence **0.03645**, max 0.1396, and **the two sources agree on which class wins in 100% of them**.
  **M6** (250k fit / 150k hold-out, raw 21-column frame): lgbm **0.95738**, xgb 0.95734, catboost 0.95721, orig-law
  **0.95397**; **all six Spearman pairs DIVERSE** (0.884–0.971, none near the 0.990 collapse limit); **logit-mean of
  the four = 0.95662, so ensemble minus best member = −0.00076**. **M7**: 304 mixed cells covering 96.23% of train,
  majority-positive side 0.4477 vs majority-negative 0.5523, weighted minority fraction **0.0946** → the battery's
  own printed verdict, "skewed: one direction dominates".
- **Gate verdicts**: **V36 dropped unauthored** — its gate needed both diversity and a gain, and the gain was four
  times the +0.0002 bar in the wrong direction; averaging diverse families scored below its own best member, the same
  reading V27's production blend gave (+0.00002). **V38's transductive variant closed by M4** (0.00000 duplication
  against a 2% bar); only the inductive bagged-context form remains, still behind the owner's weights decision.
  **V31 authored anyway, by owner instruction, as §1(a)'s falsifier**: `S6E10_V31_ConfidentRowWeights.py`, parent
  V18, `Q_DROP = 0.0387` fixed before the run, nested scorer = V30's proven teacher pass byte-for-byte (5 inner
  folds, seed base 77, same params), frame/TE/recipe/budget/folds/seed all V18's, validation block never subset so
  the fold AUCs stay comparable. The file's own docstring states that its gate came back empty.
- **A number I published an hour earlier was wrong, corrected the same day.** §1(a-i) asserted ρ ≤ 0.019. The
  contraction is `obs = 0.5 + (1−2ρ)(clean − 0.5)`, so requiring clean ≤ 1 gives ρ ≤ (1 − (0.961287−0.5)/0.5)/2 =
  **0.038713** — I halved twice. The consequence is not cosmetic: the cap sits *just above* the 0.03645 divergence
  M5 measured, so the campaign's ~4% figure is **admitted at its lower edge rather than refuted**, and at the cap our
  own CV implies a clean-data AUC of 1.000 — if the noise were that large, the ranking we already have would be
  essentially perfect on clean labels and there would be no model headroom left to buy. K1b's M3 now computes the cap
  instead of asserting it, and M2(b)'s print no longer calls the coarse-cell mass "a lower bound on ρ".
  **The entry above, written earlier the same day, carries the 0.019 figure and is left unedited; this is its dated
  correction.**
- **Key Learning**: three of this plan's eight arms were closed by thirty seconds of work, and the two most valuable
  readings are the negative ones — a duplicate rate of exactly 0, which kills transduction outright, and an ensemble
  gain of −0.00076, which kills distillation-from-a-diverse-teacher at its premise rather than at its budget. The
  second lesson is about my own output: **a printed label is a claim.** The battery captioned the coarse-key minority
  mass as a bound on ρ while that key omits 16 informative columns; a reader would have taken 0.0911 for a noise
  floor and built an arm on it. Bounds belong to the estimators that actually isolate them.
- **Status**: ✅ battery complete, all seven answers landed; ✅ V31 authored and verified (six checks; the checker
  itself is calibrated — it fires on a moved `REALMLP_PARAMS` value and on a read-before-bind, and stays silent on
  V18 itself); ❌ V36 dropped, V38's transductive route closed; ⚠️ §1(a-i)'s bound corrected the same day it was
  written; 📋 runnable queue is now **V35 → V33 → V34 → V32 → V31 → V37** (≈111 GPU-min).

---

### 16-10-2026

- **Goal**: run the V21→V28 half of the ladder on Kaggle, land its measurements in the ledgers, and settle
  V29 — which the plan had parked as "not authorable yet" because its axis was the *set* of Stage-1 passes.
- **Experiments**: **eight arms run, all eight scored** (parent V11 for V21–V27, parent V18 for V28). Every
  Stage-1
  gate failed: **V21 −0.00001 CV** (the 10 rating aggregates not built at all, guard 0 of 10 in the frame),
  **V22 0.00000** (aggregates on the corrected `6−r` scale), **V23 −0.00005**, **V24 −0.00004** (the four TE
  keys K1's z² screen ranked highest, 4.6649/4.0301/3.8548/3.6983), **V25 −0.00004** (7 `aux_p` posteriors on
  the tree), **V26 −0.000108/−0.000125** on its deeper rungs with the teacher's own hold-out AUC falling
  0.99510→0.99486→0.99466, **V27 +0.00002** at 4/5 folds — ahead of its parent on CV *and* +0.00012 LB, still
  under the Δ>+0.00005 bar. **Stage 1 passed nothing**, so V29's premise is empty and it is authored as the
  transfer test the port was meant to reward (`S6E10_V29_PortStage1KeysToNet.py`, parent V18, +4 keys = +12
  per-fold columns, no twins, verified by the six checks and committed). **V28 came in two parts**: its LB
  (0.96105) arrived with the batch and its CV block only later the same day, so it first sat with **no row**
  under law 9 and now has one — **CV 0.96128 (pooled = fold mean), folds [0.96144, 0.96041, 0.96210, 0.96136,
  0.96112], Gap −0.00023, 17.4 min** ⇒ **−0.00001 CV vs V18 at 2/5 folds and −0.00001 LB: a null, not a loss**,
  and the second-highest LB in the file. That closes the nominal-posterior channel on **both** families
  (V25 −0.00004 on the tree, V28 −0.00001 on the net, same builder) — the one measured case so far where a
  tree-null stayed a net-null, which is exactly the assumption V29 exists to test on the key channel.
- **Timing**: 50.2 GPU-min measured across V21–V27 (5.6 / 6.0 / 5.7 / 6.0 / 6.0 / 14.6 / 6.3 min) plus V28's
  printed **17.4 min** ⇒ **≈1.1 GPU-h** of the 4.1 h the ladder costs in full, leaving V29–V33 (~2.8 h) well
  inside the 8 GPU-h week.
- **Key Learning**: three things, none of them about features. (1) **A pre-training screen can rank a channel
  top and still be wrong about it** — the two highest-z² keys in K1's §5 scored −0.00005 and −0.00004, and the
  key that took 2.3% of fold-1 gain scored −0.00004: z² measures coverage of a difference, not usefulness.
  (2) **V26's control rung printed V11's five fold AUCs digit-for-digit**, which is the first in-house proof
  that this frame is bit-reproducible at fixed seed — run-to-run differences on the ladder are signal, and the
  seed-noise datum V33 still owes is about *seed changes*, not about repeats. (3) **The verification suite was
  partly theatre**: the read-before-binding check (added yesterday after V28's `NameError`) walked `__main__`
  only as a child of a block its `if __name__` test could not reach, so it returned "0 findings" on every file
  including a deliberately misordered copy. It now bites on that copy (line 802) and still reports 0 on all
  21 archived scripts and the five root arms. A check that cannot fail is not a check — calibrate it against a
  known bad case before trusting its silence. (4) **Architecture is not why the feature arms failed.** Two
  channels have now been put through both families — the rating EVs (+0.00012 on XGB in V11, +0.00012 on the net
  in V18) and the categorical posteriors (−0.00004 in V25, −0.00001 in V28) — and **both transferred, within
  0.00003 each time**. That removes the last standing excuse for re-testing a tree null on the net, and it is
  written into V29's own docstring before its run rather than after.
- **Status**: ❌ Stage 1 closed empty and V28 closed the same channel on the net — **feature engineering is
  finished on both lines**, measured by subtraction as well as addition; ⚠️ V27 is the one arm worth re-opening
  (only if the net member gets a real budget), and V28 is the second-highest LB in the file *because* it is a
  null against a control that is already at the ceiling; ✅ V29 authored, verified and committed; 🔬 next
  compute = V33 (the seed datum that de-risks every future ACCEPT) then V32 (the net's bagging budget).

---

### 16-10-2026 — continuation: the board was re-measured from outside, and our premise about it was wrong

- **What the public board actually looks like today** (read through a rendering proxy; Kaggle's own pages return only JS to a
  direct fetch): **#1 = 0.96273, rank 20 = 0.96190**, public scored on ~20% of test. Our best (V18, 0.96106) is **0.00084 below the
  top-20 floor**. The plan's working target — "CV 0.96145–0.96155 → LB 0.96120–0.96132" — was written as if that were near the
  front; it is not, it is lower-mid-board. Sources: [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard),
  [rules](https://www.kaggle.com/competitions/playground-series-s6e10/rules).
- **What the arms ahead are made of.** Author-printed public numbers: **0.96187–0.96190 for stacks** ("Twin Stacks + Family Reg"
  claims 0.96187; a "Secrets… It's Stacking!" / Top-10 Consensus Stack and a "Roadmap" stacking kernel sit at 0.96188–0.9619; a
  stack over **public OOF libraries + tabular foundation models** prints 0.96176 with nested CV 0.96200). A **TabPFN-3.5** thread
  prints **OOF 0.961178 (5-fold) / 0.961275 (10-fold)** at ~560k-row context
  ([747823](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747823)); a notebook titled
  *TabPFN + route categories* claims **LB 0.96164** ([goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96164)) —
  **not verified as a single model**: the page rendered a different, stacking title, so treat 0.96164 as an author claim of unknown
  composition. Mid-pack singles print 0.96097 ("Are You Satisfied") and 0.96026 (LightGBM + OOF).
- **Two external readings that corroborate our own arithmetic, and one that contradicts our gate.** (1) A ten-submission study here
  found OOF 0.96139 vs LB 0.96110 (gap ≈0.0003), "ranking transfers" but **"LB noise masks differences below 1e-4"**
  ([749255](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/749255)) — our own tree-family fit says the same
  thing from 20 arms: slope 0.84, r +0.86, residual sd **0.000063**. (2) A stacking author reports the stack **"got better on CV and
  worse on the public LB"** ([746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148)). (3) **The
  contradiction is the important one**: TabPFN's OOF (0.96118–0.96128) is *equal to our V18's CV*, yet its claimed LB (0.96164) is
  **0.0004 above its own CV** — while **all 28 of our arms land 0.0003–0.0006 *below* their CV**. Mechanism is structural, not luck:
  TabPFN conditions on the whole training context at predict time, so the submitted model is strictly stronger than anything an OOF
  protocol can measure, and 5-fold→10-fold alone moved its OOF +0.0001. **Our gate ("decide on CV, audit on LB") is therefore
  mis-calibrated for that family**: a TabPFN arm would read as a CV tie and submit above it. Any future family swap needs its own
  CV→LB offset measured before its deltas mean anything.
- **External data, re-measured by someone else**: the original file overlaps competition **train by 17 rows and test by 4**, `Gender`
  AUC 0.5056, 91% zero-delay, and `Flight Distance` AUC shifts **0.696 → 0.649** from real to synthetic
  ([745932](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745932)) — consistent with our own "row-append is
  dead" finding and with K1's 21 exact overlaps, and it says the *distribution*, not the labels, is what differs.
- **Why our LB stalled, in one paragraph of arithmetic over our own 28 runs.** CV stopped improving (V21–V28 span −0.00005…+0.00002
  vs their parents). The board is not ignoring us — fitted **within** family, LB tracks CV at slope 0.84 (trees, n=20, offset
  −0.000445 ± 0.000064) and 0.49 (nets, n=7, offset −0.000110); V18's CV predicts its LB to within 0.00002, so nothing is
  "underperforming". But +0.0003 LB costs **CV +0.00036**, and the largest single channel ever found here is +0.00016 — 2.2× short.
  Meanwhile 11 tree submissions sit inside 0.96058–0.96066 (width 0.00008) and a byte-identical re-submission moved the board 0.00007.
  **Conclusion: our stall is an input stall, on a plateau the board cannot resolve, against a front that has moved to a different
  class of method.**
- **Status**: ⚠️ the ladder's own axes (V29/V30/V31/V32/V33) are now known to be worth ≤0.0001 LB each — keep them for closure, not
  for position; 🔬 the two levers the evidence puts above 0.9616 are **stacking** (banned by our single-model law, owner's call) and
  **tabular foundation-model inference** (legal as a single model, untried by us, and needing a Kaggle-T4 context budget we have not
  measured). Nothing in this entry is a score row: it is external, author-printed evidence, labelled as such.

---

### 13-10-2026

- **Goal**: re-establish the V21-onward ladder as **files** after the 2026-10-12 loss, in the continuous
  order the plan dictates (V21→V32 + R27) — one fitted model per arm, no saved OOF as input, and every arm
  proven single-axis before it is committed.
- **Experiments**: **none — no compute ran, so no score row is written anywhere** (law 9). The record for
  the 2026-10-12 evening, which the previous day's entry stops before: with the owner's explicit instruction
  the K1 battery ran locally (CPU only, 51.6 s, sklearn 1.9.1 / pandas 3.0.6 / numpy 2.5.4). It **ACCEPTED
  V18's gate** (vs V12 +0.000115, z=+4.87, 5/5; vs V2 +0.000278, z=+5.12, 5/5) → the standing control moves
  V2 → V18; **recovered V20's CV at 0.961104** (vs V11 −0.000030, z=−1.06, 1/5; vs V2 +0.000094, z=+2.80 < 3
  → closed *as configured*, not as a mechanism); the §5 z² screen maxed at **4.6649** against a 1.5 authoring
  bar, reopening FE with named cells (`Class×age5` 4.6649/4.0301/3.7787/3.6891, `_wifi_band×age5` 4.2198,
  `Online boarding×_wifi_band` 3.8548, `_rated_n` 3.6983, every single column ≤2.77); §6b **closed the
  pool-fitted EV branch** (max |Δρ| 0.1182); no train↔test drift; `timehc`-reversed `_mean_excl0` **0.85658**
  (+0.02964 over the best shipped 0.82694) and `_wrs13 mean` shown to be **numerically identical to
  `_mean_excl0`** (0.81266), so P2's recorded 0.9078 is unreproducible. Two findings were about the
  instrument: its paired bootstrap **executed 0 reps on every pair** (`N_BOOT=120` never ran), so no z has an
  independent cross-check; and the plan's v2 text had by then been rewritten by a concurrent writer to claim
  "Bootstrap 120 reps, 0/120 sign flips" for four pairs — a measurement that never happened. Struck, not
  restated. **Then the working tree lost the work**: seven verified arm files *and*
  `S6E10_EDA_K1_CompleteBattery.py` were deleted by that writer, unrecoverable from git because nothing was
  committed; the sole surviving evidence of the run is `%TEMP%/S6E10_workspace/k1_stdout.txt` (664 lines).
  **Today twelve files were re-authored from the frozen parents** — V21 aggregate ablation, V22 `timehc` sign
  correction (v2 measured the +0.02964 and scheduled no slot for it), V23 age TE keys, V24 unrated-count +
  banded-wifi keys, V25 categorical posteriors on the tree, V26 teacher-capacity ladder, V27 duo orig teacher,
  V28 the same posteriors on the net, V30 soft-label ladder, V31 region-dependent α, V32 `n_ens` ladder, and
  R27 (V18 at `SEED_REPRO=2027`, output stem `r27` so nothing overwrites `v18_oof`). Each was **generated by
  an anchored script that raises unless every anchor occurs exactly once** — none typed from memory — and each
  passed four checks: `compile()`, prose density (docstring 19–27 lines, comments 54–70, inside the archive's
  band), zero unbound `Load` names, and a **docstring-stripped top-level AST diff** whose only moved
  constructs are the identity block, the axis construct(s) and `__main__`. The three ladders additionally
  **re-indent the parent's fold loop and assert the indentation is reversible** — the check that catches the
  failure mode that killed the first V26: a body that compiles cleanly and still reads a variable its
  restructure deleted. Committed as `ab81a2a`, `146e628`, `d9d7258`, `fc02b2f` (plan + ledgers `4dbdc62`).
- **Timing**: no GPU and no Kaggle CPU used. Ladder cost as authored is **≈4.1 GPU-h** of the 8 GPU-h/week
  quota — v3's "≈2.6 GPU-h" was wrong because three arms (V26, V30, V32) are multi-rung and were priced as
  single runs.
- **Key Learning (authoring is a measurement of the plan, not a transcription of it).** Three defects in the
  plan text only surfaced while writing the code that was supposed to follow it: **(a)** K1's `_rated_n` is
  **byte-equal to the shipped `_count_0`** aggregate, so "define `_rated_n` in-run" would have added a
  duplicate column dressed as a new feature — V24 keys the existing column instead; **(b)** the categorical
  posterior block is **8 columns, not 4** (Class 3 + Travel 3 + Customer 2 levels); **(c)** K1's wifi banding
  is **{0, 1-3, 4, 5}**, shown directly by the stdout (`_wifi_band:0`/`1-3`/`5` cells in §4, a `4|13-17` cell
  in §5), not the `'4-5'` bucket the spec claimed. A fourth was in the numbers: v3 quoted §5 as 4.2235 /
  3.8471 / 3.7012 where the record says 4.2198 / 3.8548 / 3.6983 — corrected against the stdout, not from
  memory. The general form: **a spec that cannot be pasted into the parent's real names is not yet a spec.**
- **Key Learning (governance, earned twice now).** Untracked work in this repo is disposable work. The
  commit-on-verification rule adopted after the first loss is what makes this re-authoring survivable: twelve
  verified files are in history rather than in a tree a second session can clear. Its corollary is plan §9,
  which now documents the verification method itself — "verified" without a stated method is exactly the claim
  v2 made while shipping three specs that would have died at launch.
- **Continuation (same day, still no compute): four read-only audit agents checked V21→V33 against their
  frozen parents and against R1-R8, and two of their findings were run-fatal.** **(a) `S6E10_V33_...`
  asserted 8 categorical posterior columns; the fitted vocabulary is 7.** `Dataset/data.csv` has
  Business travel 89,693 + Personal Travel 40,187 and **no Group travel row anywhere in the folder**, so
  Class 3 + Type of Travel **2** + Customer Type 2 = 7; the first version of the arm would have raised at
  minute 15 of a Kaggle run with nothing written. V25 carried the same wrong count in prose only, so it
  would have shipped 7 silently while claiming 8 — both corrected against the data, not the documentation.
  **(b) the R27→V33 rename left `VERSION_NUMBER = 18` in place**, so the reproduction run would have written
  `v18_oof.csv` **over the archived control vector it is measured against** — the exact thing its own
  docstring promised not to do. A new check now enforces it: exactly two `to_csv` calls, both on `{stem}`, no
  `read_csv` of any `*_oof`, and `VERSION_NUMBER`/`VERSION_NAME`/stem/`Parent:` path mutually consistent.
  **Also fixed from the same sweep:** the owner's one-OOF-one-sub rule was violated by all three ladders
  (V26/V30/V32 each wrote a vector per rung) — they now write only the rung of record and every other rung is
  read from its printed fold AUCs, which costs the paired-DeLong half for losing rungs and is stated as
  because the gate only ever needs the vector that would actually ship; V23/V24's new TE keys were inserted
  mid-dict, and since `colsample_bytree` draws features by index under a fixed seed that moved every parent
  column's index, so they are now appended after the parent's key block; V23's injectivity guard checked
  train's max Age but the keys are built on test too; V27 could feed an unfillable categorical NaN into
  RealMLP and printed its own collapse verdict without acting on it, so it now asserts no NaN survives and
  raises at ≥0.990 instead of spending five folds on a premise the falsifier already rejected; V26's in-run
  CV comparison is labelled as choosing only which vector the offline gate looks at. No new rows in `public_scores.md` / `training_logs.md` /
  `feature_engineering.md`: V20's recovered CV is real, but its best-iteration and timing lines cannot be
  recovered from a vector, and every new arm has neither CV nor LB. **V29 is deliberately absent** — its axis
  is "the keys that passed Stage 1", a set we do not own until V23/V24/V25 report, so authoring it now would
  mean either guessing the list or burying a switch inside every future run. Still owed: **R27** (the control
  gate's second half), the four **K1 re-authoring items** (plan §6), and the ledger landing of the numbers
  above.

- **Continuation (same day, still no compute by us): the owner asked for V21-onward to be verified against
  the rules and for **exactly one OOF and one submission per run**, and four read-only audit agents plus a new
  scripted check answered it. Two findings were run-fatal.** **(a)** The R27→V33 renumber (owner's request:
  the reproduction run gets a version number so it is scheduled and logged like an arm, while owning no axis)
  left `VERSION_NUMBER = 18` behind, so the file whose whole purpose is *not* to touch the archived control
  was about to write `v18_oof.csv` over it. **(b)** V28 asserted 8 categorical posterior columns; `Dataset/`
  has **no `Group travel` row anywhere**, so the fitted vocabulary is Class 3 + Type of Travel 2 + Customer
  Type 2 = **7** and the assert would have aborted a GPU run at minute 15 having written nothing. Also fixed
  from the same sweep: all three ladders (V26/V30/V32) were still writing a vector per rung — they now write
  the rung of record only and every other rung is read from its printed fold AUCs, which costs a losing rung
  its paired-DeLong half and is stated as a cost, not hidden; V23/V24's TE keys were inserted mid-block and
  are now **appended**, because `colsample_bytree` draws features by index under a fixed seed, so inserting
  moved every parent column's index and put that movement on the arm's ledger; V23's injectivity guard covered
  train but the keys are built on test too; V27 could feed an unfillable categorical NaN into RealMLP, and it
  printed its own 0.990 collapse verdict without acting on it — it now asserts and raises. **The new check is
  mechanical**: exactly two `to_csv` calls, both on `{stem}`, no `read_csv` of any `*_oof`, and
  `VERSION_NUMBER`/`VERSION_NAME`/stem/`Parent:` mutually consistent — the check that caught (a). Twelve files,
  five checks each, 0 defects.

- **Continuation (same day): V25 became the first arm of the ladder to run, and it died in `[2/5]` — the
  failure was ours, not the plan's.** XGBoost raised `value 0 for Parameter num_class should be greater equal
  to 1` at the first categorical-posterior fit: the wrapper deduced **zero** classes from the label array it
  was handed. That label coding was the only place in the file that built labels differently from the parent —
  `yt[fit_rows].map(codes).astype('int64')` against the inherited EV builder's
  `np.searchsorted(levels, vals).astype('int64')`. Now `pd.Categorical(yt[fit_rows], categories=levels).codes`,
  the same coercion the parent applies to its own categorical features, **plus a pre-fit invariant check**
  (≥2 levels, one code per fitted row, no `-1`, codes spanning exactly `0..k-1`) **and a per-target print of
  the level census and row count**, so any residual cause names itself inside the first minute instead of
  reaching the booster and burning the fold budget. Applied identically to V28, which carries the same
  builder; both re-verified (compile, prose band, 0 unbound names, AST diff still exactly the axis) and
  committed `c1968ab`. Honest limit stated plainly: this makes the mechanism *diagnosable and consistent with
  the proven parent path* — the rerun is the proof, and the census line is now the thing to read first.

- **Correction to the bullet above, same day: my first diagnosis of V25's failure was wrong, and the
  rerun disproved it.** I attributed `num_class = 0` to the label coding (`Series.map(dict).astype('int64')`
  against the parent's `np.searchsorted`) and switched to `pd.Categorical(...).codes`. The rerun printed
  the census I had added — `fitting Class: 3 levels [...] on 129,859 original rows`, then
  `fitting Type of Travel: 2 levels [...] on 129,859 original rows` — **Class fitted, Type of Travel died**,
  so the labels were never the problem: the sklearn wrapper cannot deduce `num_class` for a **two-class**
  `multi:softprob` fit. Fix now applied to V25 and V28: `objective='binary:logistic'` when a target has 2
  levels, which returns the same `[P(l0), P(l1)]` pair, so vocabulary and column count are branch-
  independent. The label-coercion change is kept (it matches the parent and is stricter), but it is logged
  as *not the cause* — the lesson is that a plausible mechanism plus a passing static check is still a
  hypothesis, and the printed census is what turned it into a refutation in one run instead of three.

- **Continuation (same day): V28 reached Kaggle and died at line 737 on `NameError: name 'P_COLS' is not defined` — a defect class every existing check was blind to.** The head print reported the axis with `+{len(P_COLS)}`, but `P_COLS` is only bound later, in `[2/5]`; the name resolves fine globally, so compile, prose density, name resolution and the AST diff all passed it. Fixed by dropping the count from the head print (the builder already prints it, where it exists), and a **read-before-binding check in module execution order** was added as verification item 6 in plan §9. It is calibrated, not asserted: 0 on the three archived parents that ran fine, 0 on all twelve arms, and on the pre-fix text it returns exactly the failure Kaggle reported — `line 740 reads P_COLS, bound at 824`. Worth naming the pattern it generalises: **checks that verify a change are silent about ordering**, and a head print is where an arm is most tempting to make informative.

- **Continuation (same day): V28 then died at line 737 with `NameError: name 'P_COLS' is not defined`, and the cause was a hole in my own verification, not in the arm.** The head print reported the axis as `+{len(P_COLS)}` while `P_COLS` is bound 84 lines later in `[2/5]`; compile, prose density, name resolution and the AST diff all pass such a file, because the name *is* bound and only the order is wrong. Fixed by printing the count where it exists, and **read-before-binding in module execution order** became check 6 of plan §9. It is calibrated, not trusted: 0 on the three archived parents that ran fine, 0 on the twelve arms, and exactly `line 740 reads P_COLS, bound at 824` on the reconstructed pre-fix text. The pattern to keep: **checks that verify a *change* are silent about *ordering***, and the head print is exactly where an arm tempts you to make the log informative before the numbers exist.

---

### 12-10-2026

- **Goal**: land V20, the last live arm of `PLAN_V16_ONWARD.md` §3 — soft targets from an in-run nested teacher on V11's frame, aimed at the ~3.8% flip noise itself.
- **Experiments**: V20 ran to completion on Kaggle GPU after two XGBoost 3.2.0 blockers, both closed on evidence rather than on documentation memory. (1) The sklearn wrapper refuses a float target — `XGBClassifier.fit` infers classes from `y.unique()` and raised "Invalid classes inferred from unique values of `y`. Expected: [0 1], got [0.1 0.9]" — so the student moved to the functional API (`xgb.DMatrix(label=p)` + `xgb.train`), with an import-time probe that forces the functional path to reproduce the wrapper recipe (`reg_lambda=500`, bootstrap off, abort above max |Δp| = 1e-3). (2) `Booster.predict` does **not** inherit the wrapper's `best_iteration` slice (default `iteration_range=(0,0)` = every trained round), so `(0, best_iteration + 1)` was made explicit — without it V20 would have been scored on ~150 trees past its own early stop, i.e. a second axis. **Confirmed public LB 0.96060** vs parent V11 0.96064 and control V2 0.96051 → **−0.00004 vs parent, inside the 0.0003 tie band**. The CV side is **not recorded**: the material supplied was the script text, not the run's stdout, so OOF CV, the five fold AUCs, the pre-declared 0.990 Spearman falsifier and the equivalence number are all still missing. By law 9 (entries only once CV *and* confirmed LB exist) **no `public_scores.md` or `training_logs.md` row was written** — this bullet is the whole record, exactly as the V15 failed attempt was handled.
- **Timing**: outputs executed and archived (`v20_oof.csv`, `v20_sub.csv`, script in `Previously trained files/Archieve/`); wall time pending the log (arm estimated ~18-20 min GPU).
- **Key Learning**: the leaderboard cannot adjudicate this arm in either direction — 0.96060 sits 0.00004 under V11 and 0.00001 under V15, both far inside the band we measured twice in-house — so only the CV criteria plus the anti-collapse Spearman can decide, and neither is captured yet. Two durable API facts for any future soft-target arm: XGBoost's sklearn wrapper is closed to float labels by class inference, and the raw Booster does not inherit the wrapper's early-stop truncation.
- **Status**: 🔬 V20 executed, **CV pending capture** → no score-table entry; LB recorded here only. Ladder state: 🏆 V18 (0.96129 CV / 0.96106 LB) is the best single and candidate control, ❌ V16/V17/V19 closed, V14 still closure-only with no scores, and the Step 0 gate kernel (paired DeLong + `SEED_REPRO=2027`) remains the one thing standing between V18 and control status.

- **Experiments (planning turn, same day — no compute run, so no score rows by law 9):** read all four ledgers, the surviving plan text and the archived V18/V20 configs end-to-end, then authored `PLAN_V21_ONWARD.md` (repo root) as the single-model / no-saved-OOF next stage. It opens by accounting for the owner's complaint honestly — V11/V18's EV block, V12's twin+logit frame and V2's whole package are public recipes — and then spends Tier 1 on the four things that are **ours** and were recorded but never acted on. Three audit findings came out of the re-read rather than out of new experiments: **(1) V20 mis-specified its own arm** — it ran `ALPHA = 0.20` (line 82) with `early_stopping_rounds=150`, while the surviving §V20 pre-declaration is `0.3×label + 0.7×teacher` and the published ladder is monotone up to α=0.70 (0.961272 → 0.961320 → 0.961367 → **0.961394**), whose stated mechanism is training **369 → 1,527** iterations — a window tuned for hard labels can truncate exactly that. So a flat V20 CV would falsify the configuration, not soft labels. **(2) Teacher capacity has never been tested in isolation**: `_og_teacher` is 40.4% of V2's fold-1 gain and #1 by splits in both other families, yet the only capacity move (V1's d6/17-feature → V2's d8/21-feature) was bundled with three other axes, and the teacher is fit on 129,859 clean rows with no held-out split — we have never measured its own quality. **(3) `PLAN_V16_ONWARD.md` is not in this working tree** (and `PLAN_V7_ONWARD.md` survives only as the temp copy), so the V16-era pre-declarations can no longer be quoted from the repo — which is why §V20 above is read from the surviving text and the archived script header.
- **Key Learning (planning turn):** the channel-vs-re-cut rule now dictates the shape of the ladder, not just its verdicts: the only arms worth authoring are **new channels** (the cross-inductive-bias teacher — V3 closed tree+tree and its own note names tree+MLP as the +0.00072 direction; the 3 missing categorical EVs that leave our net arm at 13 of the published 16 aux targets), **producer quality on an existing channel** (teacher capacity, bagging capacity — `n_ens` 1→8 was still our largest internal delta at +0.000886 and 16/32 have never been run), **the target** (soft labels at the pre-registered α), and **one correction to a known-wrong convention** (`Departure/Arrival time convenient` is backwards at ρ=−0.90 and sits inside 8 of the 10 aggregate columns; excluding it lifted `_mean_excl0` 0.8127→0.8383 — recorded 2026-10-07, reproduced by a public author, never given a version). Second, and a hygiene rule worth keeping: two old closure slots were **consumed by renumbering** rather than run — the zero-pattern joint key and the purge + `_wrs13`/`_c4_thcrev` composites (0.9078 / 0.8573 univariate, against our live 0.8132). A renumbered-away slot is an unexecuted arm, and its numbers still live in the ledger as if it were closed.
- **Status (planning turn):** 📋 `PLAN_V21_ONWARD.md` authored — Tier 0 first (**G0** V18's never-run gate: paired DeLong vs `v12_oof`/`v2_oof` + `SEED_REPRO=2027`, ~10 min Kaggle CPU; **G1** capture V20's CV block from the Kaggle stdout; **G2** the z² residual screen on `v18_oof` as the pre-authoring FE gate), then Tier 1 **V21** duo teacher (tree+MLP on the original, the +0.00072 direction V3 named and never built) / **V22** sign convention for the backwards rating / **V23** teacher-capacity ladder with an in-run control rung / **V25** soft labels as a 3-rung ladder that reproduces V20 as its control and isolates the ES window from α / **V26** `n_ens` 16→32, with **V24** (EV 13→16), **V27** (net input purge), **V28** (CatBoost ordered-TS + twins) in Tier 2 and a union ship arm that composes only gate-passed channels. Full pass ≈4.3 GPU-h against the 8 GPU-h/week quota, G0/G2 on CPU. ⚠️ Nothing ran, so `public_scores.md` / `training_logs.md` / `feature_engineering.md` are unchanged.

- **Continuation (EDA turn — owner restored this session's ownership of `PLAN_V21_ONWARD.md` + `Memory/*`; no compute ran, so no score rows by law 9):** the objective for this turn was the EDA pass the plan had been written without: *what patterns in the dataset have we recorded and never acted on*. Because nothing may execute locally (Law 3; the classifier blocked every attempt including the owner-authorised ones), the EDA was done as an **instrument plus a record audit** rather than as measurements. Three artefacts:
  1. **`S6E10_EDA_K1_CompleteBattery.py`** (repo root, 1,356 lines, 20 sections, syntax-verified, fits nothing, writes nothing, Kaggle CPU). It collapses the plan's entire Tier 0 into one run and adds the model-free checks the campaign never had: §2b near-duplicate label-conflict ladder (first *model-free* lower bound on ρ, which the 0.036–0.039 headroom arithmetic has always been an inference about), §2c per-segment agreement between the competition law and the original law, §3 the backwards-rating re-audit, §4 ρ per cell, §5 the z² residual screen, §6b original-vs-pool EV-fitting divergence, §6c per-rating R²-on-others (fit-free substitute for the revoked r-screen), §6d train↔test **joint** drift (the one unexplained fact: every version lands LB below CV while KS ≤ 0.0027 proved only the marginals), §7b EV-column census, §8 paired DeLong over the saved OOFs, §10 a verdict line per decision with `not computed` guards.
  2. **Record audit → 21 findings**, ranked in temp (`EDA_SYNTHESIS_FOR_REVIEW.md`, `EDA_K1_FINDINGS_AND_DECISIONS.md`) with paste blocks D1–D8 for the eventual stdout. Findings that changed the plan are now in **`PLAN_V21_ONWARD.md` §7**.
  3. **Ledger corrections annotated, not edited** — see the dated appendix in `public_scores.md`.

- **Key Learning (EDA turn): the highest-value hidden pattern was in our own code, not in the CSV.** `AUX_EV_FIT_ON_ORIGINAL = True` is hard-set in V11, V18 and V20, and **every one of those files already contains a pool-fitted `else` branch that has never executed in any version** — the campaign has been running one half of a two-branch switch for nine versions and calling the choice a settled fact. That is a switch flip, one axis, ~6 min, and it is now arm **V24b**. Same class of finding: the 10 rating-aggregate columns arrived bundled inside V2's four-axis package (+0.00040, never decomposed) and have **never been ablated** — arm **V29** (~4 min, needs no K1 number) settles both the decomposition and whether V27's purge list can be measured instead of guessed.
- **Key Learning (second, and it constrains how to read every future gate):** the gate as written is **easier to pass the less an arm changes**. The only two recorded paired SEs (5.3e-5, 8.0e-5) mean z>3 demands |Δ| ≈ 0.00016–0.00024, i.e. 3–5× the Δ>+0.00005 bar, while the null arms' dispersion is ±0.000024 — so Δ alone is only ~1.6σ from a measured null, and the paired SE shrinks as an arm's logits converge to the parent's. New reporting rule: Δ, paired SE, z, logit correlation to parent, fold count, always; and *"z>3 at low divergence"* is a higher evidence tier than *"z>3 at high divergence"*.
- **Key Learning (third, self-correction):** three numbers I had published from the ledgers were wrong and are corrected here rather than quietly. (a) The iteration-alert statistic is **31 of 40 logged folds (77.5%)**, not 35/40 — recomputed from the per-version arrays, and the runbook's "alert below 1,000 iterations" rule is deleted as a complexity signal regardless, since the control and the best models trip it and the episode's highest stopping point is 1,325 against a 6,000 cap. (b) **V5's −0.00126 CV deficit is −0.00081 epoch-corrected** — pooled OOF and fold mean diverge by −0.00045 at `n_ens=1`/6 epochs, so the rejection stands but the size was overstated by a third; hence the new rule that net arms print the fold mean and no arm is compared on pooled OOF across `n_epochs`. (c) The frame census inherited from the older plan was **stated-as-verified and wrong**: code gives **40** fold keys (17 singletons, 5 FD/Age encodings, **18** pair keys = 10 `KEY_PAIRS` + 8 `RATING_PAIRS`) × 3 mechanisms = **120** per-fold against "~36/21/10", base **139**, and **9** categorical shares not 10 — V16's `swap` dropping 54 columns independently confirms 18 × 3.
- **Key Learning (fourth, on data integrity as an EDA output):** recompute any derived cell before quoting it. `public_scores.md`'s `Gap` column follows **Gap = LB − OOF**, and 3 of its 24 rows do not satisfy it (V3 in both tables, V4 once — both recorded −0.00045 against correct −0.00031/−0.00037, the identical value being the tell of a copied cell). `training_logs.md` carries **two irreconcilable best-iteration arrays for V2** (`[785, 648, 530, 405, 482]` in its own row vs `[649, 465, 548, 662, 423]` quoted by V8, which fitted no depth-wise control at all); it cannot be adjudicated from the record, so it is flagged, not chosen — which is itself the argument for judging `SEED_REPRO=2027` reruns on paired AUC only.
- **Governance note (why the plan's Tier 0 was rewritten):** `PLAN_V21_ONWARD.md` Tier 0 now reads "collapsed into one analysis kernel" (G0/G1/G2 → K1 §8/§5). The parallel session that briefly held the file had marked Tier 0 ✅ with gate numbers; those numbers were computed **off-platform** and then **every script was deleted and no archived**, leaving ✅ rows with no script and no stdout. The honest status is **reported, not measured**, so V18's gate, V20's CV and the z² screen are all owed again, and the plan says so with that provenance attached.
- **Timing:** no compute this turn. K1's own cost is ~25 min Kaggle CPU for the full battery, with the bootstrap capped at `N_BOOT=120` on the three pairs of record ((18,12), (18,2), (20,11)) after a first draft was measured on paper at 23 min for bootstraps alone.
- **Status (EDA turn):** 📋 EDA **authored, not executed** — K1 is ready to paste onto Kaggle CPU with the OOF archive attached, and until its stdout lands **no §2b–§8 number exists and none is quoted as measured**. Ladder updated in place: +**V24b**, +**V29**, V26 Tier 1 → Tier 2, V25 three rungs → two, **V22 held by owner decision**. 🏆 V18 (0.96129 CV / 0.96106 LB) remains best single, V2 (0.96101 / 0.96051) remains standing control, V20 still has no score row.

- **Continuation (same day, still no compute): V29 authored — the EDA finding that needed no measurement became a runnable arm.** `S6E10_V29_AggregateAblation.py` (repo root, GPU, ~5 min) ablates the **10 rating-aggregate columns** (`_mean_excl0`, `_std_excl0`, `_min_excl0`, `_count_ge4_all`, `_count_ge4_inflight8`, `_count_ge3`, `_count_le2`, `_count_le1`, `_count_eq5`, `_count_0`) from V11's frame. Parent **V11**, one axis, and the diff against the archived parent is exactly those 12 lines plus identity/print lines — EV block, TE keys, delay/bins, fold driver, seed and XGB params untouched; the file raises if an ablated column reaches the frame, and its audit number is **`Total base features`
exactly 10 below V11's run** (only the delta is claimed — the absolute count must come from a Kaggle run). Why it is worth a version at all: the block entered through V2's four-axis bundle (+0.00040) and has **never been decomposed in 28 versions**, and it is the only list the net-purge arm (V27) currently guesses at. Expected outcome is flat CV — the channel-vs-re-cut rule predicts ≈0 for any re-cut of already-splittable information — and flat is the useful answer, because it retires ten columns on measurement.
- **Status (V29):** 🔬 Authored, syntax-checked, **not run** — so no score row anywhere. Two Kaggle jobs are now owed by the owner, in priority order: **V29** (GPU ~5 min, answers a 28-version-old question and unblocks V27's purge list) and **K1** (CPU ~25 min, unblocks V24b/V22/V25 decisions).

- **Continuation (same day): V25 authored — the campaign's own mis-specification turned into a runnable ladder.** `S6E10_V25_SoftLabelLadder.py` (repo root, GPU) is V20's mechanism rebuilt at the configuration the inherited spec actually pre-declared. Two declared rungs, **α 0.20 / ES 400** and **α 0.70 / ES 400**, run of record **α 0.70 pre-declared rather than chosen by the win**, one student per rung, rungs never combined, both OOFs saved as the audit trail. Two design decisions worth recording because they are the arm's whole evidential value: **(1) both rungs share ES=400**, so the rung-to-rung delta is the teacher-weight axis uncontounded — the V16-style confound (α and the early-stop window moving together) cannot occur inside the comparison; **(2) the third rung (α 0.20 / ES 150, i.e. V20 reproduced in-run) was dropped**, because its only information was the ES half and that reads directly against `v20_oof.csv` offline. Since neither α nor ES enters the nested teacher's cross-fit, the 25 inner fits and the per-fold prep are computed **once per fold and shared by both rungs** — declared in the docstring so the wall time is not misread as a third rung's evidence. Cost falls from the plan's ≈60 min to **≈14**, which is why Tier 1 is now ≈115 min GPU and a full pass ≈3.5 GPU-h against the 8 GPU-h/week quota. The mechanism claim being tested is printed and checkable: the published soft target trains **369 → 1,527** iterations, and the run prints both best-iteration arrays so a truncation by ES=150 becomes visible rather than inferred.
- **Status (V25):** 🔬 Authored (syntax-checked, no leftovers from the restructure), **not run**, sequenced **after K1** — not because the α half needs a number, but because V20's never-captured CV is the comparator for the ES half. Ladder now has three runnable artifacts waiting on Kaggle: **V29** (~5 min), **K1** (CPU ~25 min), **V25** (~14 min).

- **Continuation (same day): finding 22 — V3's null was a two-axis arm, and the campaign has been reading it as the closure of teacher diversity.** Read out of the archived code, no measurement needed: V3 stores `_og_teacher` as the **raw logit-mean** of its two teachers (V3:363-366) while V2 and V11 store `predict_proba` (V2:342-343, V11:323-324). So the column changed scale, sign and range — a [0,1] probability became an unbounded signed value — in the same step that its estimator set went from one model to two. `feature_engineering.md` and every downstream closure note have since cited that **−0.00005** as evidence that teacher diversity does not work here. What V3 actually closed is **tree+tree**; **tree+MLP — goodpjw's +0.00072, the largest published single-model delta still un-disproved in this episode — has never been tested cleanly.**
- **Key Learning (authoring consequence):** that reinterpretation is what promotes **V21** above the FE arms, and it had to be authored differently from V3 to be a fair test: V21 (`S6E10_V21_DuoTeacherOrig.py`, parent V11, syntax-checked, not run) averages the two orig-only teachers in logit space and maps the mean **back through the sigmoid**, so `_og_teacher` arrives on the parent's scale and the estimator set is the only thing that moved. `REALMLP_PARAMS` is copied from our own archived V18 run rather than from documentation memory (Arg-Name Law), and the arm carries the pre-flight the plan mandated — both teachers fitted on 80% of the overlap-cleaned original and scored on the untouched 20%, printing each teacher's own AUC (the channel's producer quality, never measured in 21 versions) and the Spearman between their logits, with **> 0.990 ⇒ reject at any CV**. The measurement fits are discarded and the shipped pair is fitted on 100% of the clean rows exactly as the parent fits its one teacher, so the teacher's training set is not a silent second axis.
- **Governance note:** a closure that is quoted as settled should be re-read in code before it is used to rank future arms. Two of this turn's three highest-value findings (V29's never-ablated aggregates, V3's scale confound) are the same operation applied to the record — reading what shipped, not what was written about it. §5 of the plan now carries the V3 caveat inline so the closure list cannot be read without it.

- **Continuation (same day): findings 23–25, from auditing `feature_engineering.md` against the archived code and the other ledgers.** The FE file is what future arm specs get quoted *from*, so its defects propagate. **(23)** Its V4 row says GPU-AUC is unavailable "in build **1.6.1** env" — 1.6.1 is **scikit-learn**'s version (printed as "sklearn 1.6.1" in four `training_logs.md` rows), and V4 never prints a CatBoost version at all; the only measured CatBoost build in the episode is **1.2.10**, from V16's `version('catboost')`. The mechanism note was right (V16's runtime line `AUC is/are not implemented for GPU` is exactly a 1.2.10-era limitation), which is why the mislabel survived: **a wrong label on a correct observation reads as verified.** This binds **V28** — its kwargs must come from V4/V16's archived calls or a signature probe, and whether `boosting_type='Ordered'` runs on CatBoost's GPU path is something **we have never established either way**, so the arm must open with V16's 30-iteration probe rather than assume. **(24)** The ledger still quotes `_count_ge4_inflight8` at univariate AUC **0.8352**, retracted to **0.8132** on 2026-10-07 with the correction living only in this file and the plan. **(25)** Its column counts are in two different units without saying so — 139/152 are base frames, 259/276/289→374 are fold-expanded matrices (139 base + 120 per-fold TE = the 259 the nets inherit) — every number correct, the mixtape wrong, which is why V27/V29 now quote deltas. All three are recorded in a dated appendix at the end of `feature_engineering.md`; rule 2 means the historical rows are untouched, and no feature entry or score claim was added there, so rule 1 is not engaged.
- **Key Learning (the audit method, worth keeping):** three of today's five strongest findings are not dataset measurements at all — they are **disagreements between the record and the code** (a hard-set switch whose other branch never ran, a closure whose arm had two axes, a version number belonging to another library). The cheapest high-yield EDA operation in a campaign this mature is to re-derive every quoted claim from the shipped source, then re-check which of those claims was ever tested in isolation.

- **Continuation (same day): finding 26 — the CV layer of `training_logs.md` passes a full re-summation, and that changes what the net arms' numbers mean.** All **19** recorded fold tables were re-added by hand and each printed `Mean` matches the arithmetic mean of its own five fold AUCs (worst case ±1e-5; the three display mismatches — V2 0.961016→0.96101, V5 0.960196→0.96019, V8 0.960996→0.96099 — are truncation-vs-rounding and each still matches its own recorded pooled OOF). So unlike `public_scores.md`, which carried three wrong `Gap` cells, **this file needed no correction appendix — a negative result worth stating, because the ladder's decisions rest on it.** Differencing pooled OOF against fold mean across all twelve GBDT arms gives **≤ 1e-5 in every single case with no exception**, and the only non-trivial offsets are nets: V7 −0.000012, V12 −0.000008, V18 −0.000002 at 3 epochs, V6 −0.000154 at 6, V17 −0.000172 at 10, V5 −0.000446 at `n_ens=1`/6. **The new pattern is in the last two comparisons: at the same 6-epoch budget, bagging 1→8 cut the offset 3×, so the pooled yardstick degrades with fold-level calibration variance and epoch count is only one source of it.** That is now a pre-declared falsifier on **V26** (`n_ens` 16/32): the offset must shrink further, and if it does, part of every historical net arm's CV was being *lost* by judging on the pooled vector — up to 0.00015, i.e. 3× the Δ>+0.00005 bar — which is why V26 is scored on the fold mean.

- **Continuation (same day): K1 hardened by a static control-flow audit before the owner spends a Kaggle session on it.** Reading the battery for failure modes rather than for findings turned up one real defect: **§5, the z² residual screen that decides whether every FE arm lives or dies, called `Series.groupby(list-of-Series)` five times per key set — an idiom the file's own §0b pre-flight does not prove — and because no section is individually wrapped, a raise there would have taken §6, §6b, §6c, §6d, §7, §7b, §8, §9 and §10 with it**, i.e. the whole decision half of the run. It now performs **one** DataFrame-level named-aggregation groupby per key set (the path §0b validates; §0b also gained a probe covering named agg, `qcut(duplicates='drop')`, `rank().corr()` and `pinv`), same statistics, 5× fewer passes, no dangling temporaries left behind, syntax clean at **1,387 lines**. §10 was checked and is already safe — every line passes through a `val()` helper that renders a skipped section as `not computed` rather than crashing, which is what keeps a partial paste usable.
- **Key Learning (instrument design, not dataset):** in a long single-run battery, either put the exotic pandas idiom inside the pre-flight or don't use it — **the pre-flight's coverage is the run's risk surface**. The other cheap insurance is structural: a closing summary that reads guarded values, so the last section can never be the one that destroys the first twenty minutes.
- **Status (K1):** 🔬 Still authored-and-verified-statically-only, still unrun. Nothing about the dataset has been measured yet, and the four artifacts waiting on Kaggle are unchanged in that respect: **K1** (CPU ~25 min + OOF archive), **V29** (~5 min), **V21** (~10 min), **V25** (~14 min).

- **Continuation (same day): V23 authored — the third Tier-1 arm that needs no K1 number, aimed at the campaign's most-used and least-measured column.** `S6E10_V23_TeacherCapacityLadder.py` (parent V11, GPU) varies `_og_teacher`'s capacity alone — rungs **d8/600 (V11's exact spec, refit in-run as the control), d10/1000, d12/1500**, both values moving together as one declared coordinate, rung of record **pre-declared as d12/1500**, per-rung OOFs saved. Two things the authoring settled from the parent's code rather than from assumptions: **(a) the fold prep is provably shareable** — `make_key_frames` takes its keys from CATS/RATINGS/FD/Age only and never reads `_og_teacher` (which appears in V11 solely where it is created), so one prep per fold serves all three rungs and a rung overwrites exactly one column of the prepared matrices; the arm is therefore ~9 min instead of 3 × 5.8, and the sharing is a single-axis property rather than a shortcut that changes results. **(b) the teacher's own quality finally gets measured** — each rung also fits on 80% of the clean original and is scored on the untouched 20%, printed beside the shipped column, which is always fit on 100% of the clean rows exactly as the parent fits its one teacher, so training-set size never becomes a second coordinate. The file asserts both overridden keys exist in the inherited `TEACHER_PARAMS` before starting, so a rung cannot silently fail to change anything, and the RESULTS block prints how far the control rung reproduces V11's recorded 0.96113 — that gap is itself the run-to-run variance datum the seed-2027 rerun would otherwise lack. Syntax-checked, 780 lines, **not run**; if the per-rung teacher AUC is flat, a flat student result closes the capacity axis cleanly instead of ambiguously, which is the counter-evidence the spec carried openly (129,859 rows, near-deterministic law).

- **Continuation (same day): V26 shipped on a second attempt, and the way it was done is the lesson.** The first pass left the file half-restructured — the new rung-loop opener landed, the old fold body stayed, and the body still loaded `oof_probs`, `prep_times` and `fold_scores`, which the new opener no longer defines. **`py_compile` reported that file clean.** Rather than leave a runnable-looking trap in a repo whose owner spends GPU sessions on it, I deleted it (backup in temp) and rebuilt from the archived parent with a patch script that **asserts every anchor matches exactly once before writing anything**, so a mismatch aborts with the repo untouched. The scan then earned its keep immediately: it found a second, missed `oof_cv = auc_score(y.values, oof_probs)` behind the replaced region. Final verified state of `S6E10_V26_NetBaggingCapacity.py`: syntax OK, docstring 34 lines (cap 35), one `out_dir`/`stem` definition, `for n_ens in N_ENS_LADDER:` with the fold loop correctly nested, per-rung `v26_ladder_nens{8,16,32}_oof.csv` plus `v26_oof/sub.csv` from the pre-declared record rung (32), no undefined loads outside function parameters, no saved OOF read as input. Each rung prints pooled CV, fold mean **and their difference**, which is the §7 finding-26 falsifier: `|pooled − fold mean|` must shrink as bags rise (V5 `n_ens=1` −0.000446 → V6 `n_ens=8` at the same 6-epoch budget −0.000154).
- **Key Learning (verification, and it generalises past this file):** a syntax check is not evidence that a restructure is complete — it proves only that the text parses. What a body-level refactor needs is a **name-resolution pass**: collect every load, subtract every binding, and read what remains. That is also why the ladder's `dCV` lines were audited as a group: each prints `oof_cv − <hardcoded parent CV>`, and across all 25 version scripts every literal matches `public_scores.md` (V2 0.961009, V4 0.96088, V11 0.96113, V12 0.96117, V15 0.96105, V18 0.96129), so no pending run can mis-state its own delta through a stale constant — a defect no gate would ever surface.
- **Status (ladder, end of day):** 🔬 Seven runnable artifacts, **none run**: **K1** (CPU ~25 min + OOF archive), **V29** (5), **V23** (9), **V21** (10), **V25** (14), **V24** (18), **V26** (70) ≈ 5 GPU-h against the 8 GPU-h/week quota. `PLAN_V21_ONWARD.md` §8 now carries the **decision table** for them — each pre-declared threshold paired with the action that follows, including the two that can re-price the whole plan (K1 §2b's ρ floor > 0.039 falsifies the ceiling arithmetic; K1 §8's z ≤ 3 means V18 is not the control and every child of it must be re-judged against V2). Running order was fixed for information-per-GPU-minute, not for optimism: the cheap arms that can *cancel* expensive later ones go first. 🏆 V18 remains best single (0.96129 CV / 0.96106 LB), V2 the standing control, V20 still CV-less, and no score row was written for anything that did not run.

---

### 11-10-2026

- **Goal**: execute the first four arms of `PLAN_V7_ONWARD.md` (V7 net recipe, V8 leaf-wise, V9 TE row-stats, V10 original-distance) on Kaggle GPU, each judged by the frozen-fold gate against V2, and bring the version files to house format (minimal prose, device declared in both the header docstring and the head print).
- **Experiments**: **V7** RealMLP on goodpjw's proven recipe (flat_anneal / 3 ep / ES off) → **0.96089 CV / 0.96049 LB**: +0.00025 vs parent V6, −0.00012 vs V2, folds 0/5 → rejected; best epoch = 3 in **5/5** folds, so every net arm we have run (V5/V6/V7) was epoch-starved rather than disproved. **V8** `grow_policy='lossguide'`, `max_depth=0`, ladder 8/16/32 → **0.96099 CV / 0.96042 LB** (rung 32), folds 1/5, ladder spread only +0.00010 → rejected; S6E5's +0.00089 does not transfer once RowStats is unbundled. **V9** +5 `_tmr_*` row-stats over the `_tm_*` block → **0.96103 CV / 0.96067 LB** — the **highest LB of the episode**, but +0.00002 CV with folds improved 2/5 → gate fails; `_tmr_mean` takes 7.96% and `_tmr_min` 6.83% of fold-1 importance and still adds no ranking information. **V10** +9 `_od_*` original-distribution columns → **0.96103 CV / 0.96065 LB**, folds 3/5, and **no `_od_*` column reaches the top-15** → arm closed. Also: all nine V7–V15 scripts brought to the house verbosity baseline (docstrings 14–28 lines, RESULTS blocks cut to identity+device, CV, per-fold AUCs, best iter, one timing line), `Device:` added to every title line and head print, and a **syntax-breaking defect repaired in V12** (≈100 lines of old docstring left outside the module docstring, which had swallowed the import block — V12 could not have run before this fix).
- **Timing**: V9 3.8 min, V10 3.7 min, V7 13.8 min (143 s/fold fit; V6's 6-epoch fit was 247 s — half the budget for 58% of the time), V8 11.1 min for 3 rungs × 5 folds. Prep stayed 9–19 s/fold; the two FE arms cost nothing at prep time.
- **Key Learning**: **The gate did its job on all four arms, including the two that looked like wins.** Three arms moved LB by +0.00014…+0.00016 (V9, V10, and V3 historically) while CV moved +0.00002 and fold improvement was 2/5 or 3/5 — precisely the 0.0003 tie band the plan measured from the leaderboard itself. So a new best-LB row is *not* a new control, and V2 stands at 0.961009 CV / 0.96051 LB. Second: importance rank and incremental value are independent — V9's block is the 3rd/4th most-used feature group in the frame and changes nothing, which is the strongest evidence yet that the FE axis is genuinely closed here. Third: with three net runs below V2, the plan's pre-declared condition fires — **V12 is SKIPPED, not re-tuned**; the surviving net value is `v7_oof.csv` as the strongest leg for any own-model combination.
- **Status**: ❌ V7, V8, V9, V10 all rejected as control — **V2 remains control (0.96101 CV / 0.96051 LB)**; ⚠️ V9 logged as best LB of the episode (0.96067, inside tie band); ✅ V7 = best net and preferred combination leg; 🔬 V11–V15 authored and format-clean, awaiting execution; ⚠️ V12 skipped by rule; pending: paired DeLong on `v9_oof` / `v10_oof` vs `v2_oof` to record z before the arms are archived.

- **Continuation (same day — V11, V12, V13 landed; the V12 line above is superseded by evidence):** **V12 = CV 0.96117 / LB 0.96095 → the best single of the episode**, +0.00016 CV vs V2 with **folds improved 5/5** and **+0.00044 LB — the only arm outside the 0.0003 tie band**; **V11 = CV 0.96113 / LB 0.96064**, +0.00012 CV with **5/5 folds** (the first tree arm to clear both CV criteria, and the ladder's only CV-positive FE move besides V12); **V13 = CV 0.96101 / LB 0.96058 → exactly V2's CV (Δ +0.00000, folds 3/5)** despite removing every saturated org-mean cell (train 5,999 → 0, test 2,628 → 0, 649/1,513 cells moved). Two further measurements: the plan's shrinkage prior `b = 0.434463` re-derived itself as **0.434379** at runtime (recomputed value used), and **re-submitting the identical `v13_sub.csv` moved LB 0.96058 → 0.96051, a 0.00007 swing on a byte-identical file.** Timings: V11 5.8 min (116 s EV build), V12 15.2 min, V13 3.7 min.
- **Key Learning (continuation):** **the pre-declared V12 skip was wrong and I should have caught it**: the rule read "three net runs below V2 ⇒ the net is disproved for our frame", but those three runs all fed the net *V2's tree-shaped frame*. Changing only the representation (17 numeric→categorical twins + teacher-as-logit, recipe byte-identical to V7) bought +0.00028 over V7 and passed every tree in the episode. Two rules to keep: a skip condition must be stated against the arm's **own** axis, and V12's best-epoch = 3 (the budget) in 5/5 folds means this is still the net's floor, not its ceiling. Conversely V13 proves the other half of the lesson: a defect can be real, provably removed, and worth exactly zero.
- **Status (continuation):** 🏆 **V12 is the new best single and the candidate control** (CV 0.96117 / LB 0.96095); ✅ V11 = strongest tree single, CV gate cleared; ❌ V13 rejected; **V2 remains the standing control until paired DeLong z > 3 and the `SEED_REPRO=2027` rerun confirm V12** — both still pending, and nothing gets submitted to LB on a tie band. Remaining ladder: V14/V15 closure runs, V16 (39 native-cat crosses, CatBoost) and V20 (soft-label targets) are now the live bets, and V12's result re-opens the twin/logit frame for trees.

- **Continuation (V15 attempt, no score):** V15 raised `TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_round'` at fold 1 and produced **no OOF and no LB**, so nothing was logged in the score tables. Kaggle's image is **LightGBM 4.6.0**, not the pinned 4.7.0 — the file's PIN WARNING fired as designed and was recorded instead of ignored. Fixed same day: early stopping in LightGBM's sklearn API is a **callback** (`callbacks=[lgb.early_stopping(150, verbose=False)]`), with `predict_proba` using `best_iteration_`, so V2's ES semantics are preserved. Also switched the arm to **CPU-only by owner decision**: the GPU path is deleted rather than kept as an option (`CFG.USE_GPU` and both `DEVICE_ARGS_*` dicts removed, `device_type='cpu'` + `n_jobs=-1` now inside `LGBM_PARAMS`), and the ektarr teacher keeps V2's hyperparameters but runs `device='cpu'` so the kernel stays off the 8 GPU-h/week quota — a disclosed device-only deviation, logged in V15's docstring.
- **Key Learning (V15):** this is the **third** failure from invented library argument names in this episode (V5's PyTabKit kwargs, V15's `early_stopping_round`). The house rule is not optional: take kwarg names from our own archived scripts or from the installed build's signature, never from documentation memory — and pin warnings exist to be read, not bypassed.
- **Housekeeping (same day): the dead cuDF accelerator switch is removed from all 15 scripts (V1–V15, archived parents included).** `USE_CUDF = False` meant the `cudf.pandas.install()` block never executed, so **no run semantics change** — the archived scores still describe exactly what ran. Because the identical 9-line block was deleted from **both** side of every parent/child pair, the one-axis byte-comparisons stay valid. Printed device labels dropped `cuDF: False`, and docstring technique lines now say only "all numeric matrices float32". Consequence for the ledgers: `training_logs.md` Device lines written today read "…`device='cuda'`, cuDF off" — accurate for those runs, but **future Device lines should not mention cuDF at all**, since the option no longer exists in the code.

- **Experiments (later still — V15 clean run):** **V15 LightGBM single = CV 0.96105 / LB 0.96061** (Gap −0.00044, 28.2 min on **CPU**, `device_type='cpu'`, Kaggle build **4.6.0** not the pinned 4.7.0). Folds improved **4/5** vs V2 (+0.00011, +0.00006, +0.00004, −0.00006, +0.00003) but aggregate CV moved only **+0.00004 — short of the Δ>+0.00005 gate by 0.00001** → rejected. Early stopping fired in 5/5 folds (best iters 1068/1012/838/669/833), the counter-example to every net arm. The API question is now closed by the pre-flight's own print: `fit() exposes: callbacks=True | eval_metric=True | **early_stopping_rounds=False**` — so the callback was the *only* valid form on this build and the "obvious" plural-kwarg fix would have failed too. V14 (`max_bin` ladder) is the only arm still unrun.
- **Key Learning (cross-version read of V1–V15, the thing to carry into the next arms):** across fifteen versions the **LB moved 0.00076 total** (0.96019 → 0.96095) and **13 of 15 submissions sit inside a 0.00026-wide band** (0.96042–0.96067) — i.e. indistinguishable at the tie-band resolution we measured twice (same-file resubmission = 0.00007; board-derived band = 0.0002–0.00045). What actually separates the versions is the **CV**, and only two arms ever raised it: **V11 aux-EV (+0.00012, 5/5)** and **V12 net frame (+0.00016, 5/5)** — both of which feed the model an *independent* channel (a label-free cross-fitted prediction of another column; a representation plus a teacher logit) rather than a new cut of columns it can already split. Every re-cut of existing information measured ≈0: leaf-wise +0.00000 (−0.00002), `_tmr_*` row-stats +0.00002, `_od_*` distance +0.00002, org-mean shrinkage +0.00000, and the whole new family in V15 +0.00004 — while V15's importance profile (route residuals taking ~9 of 15 split slots) shows LightGBm genuinely "sees" the frame differently yet lands 0.00004 away. **Different partition of the same information + identical outcome = the noise ceiling, not an untapped feature.** With κ=0.924 at ρ=0.036 the clean-law AUC V2 already implies is 0.9990, total headroom is 0.0009–0.0019, and the board's 0.96179–0.96184 top-25 are all stacks — so the missing piece for LB is not another axis on this frame, it is either **more effective training budget on the one arm that converts (V12 pins best-epoch = 3 in 5/5 folds ⇒ epochs untested)**, **soft-label targets that let a single model train 3–10× longer without overfitting (V20, published +0.00012–0.00019, single-model legal)**, or **the one published FE delta that is family-specific and still un-disproved here (V16's 39 rating×context native-cat crosses on CatBoost, +0.00029/+0.00034, 5/5)**.
- **Status (V15):** ❌ V15 rejected (V2 stands on the tree side); 🏆 **V12 (CV 0.96117 / LB 0.96095) was the episode's best single and the candidate control** at this point, still awaiting paired DeLong z vs `v2_oof` and the `SEED_REPRO=2027` rerun; ✅ V11 = strongest CV-positive tree arm, V15 = the only different-family OOF we own; 🔬 next actions: V12 epoch-budget arm, V20 soft labels, V16 crosses (V14 remains closure-only, and closing the ladder with more same-frame re-cuts is now predicted-null by fifteen measurements, not by opinion).
- **Experiments (later still — the V16/V17/V18/V19 batch, `PLAN_V16_ONWARD.md` §3, all four arms landed):** 🏆 **V18 RealMLP + V11's 13 aux-EV columns on V12's byte-identical twin+logit frame at `n_epochs=3` = CV 0.96129 / LB 0.96106 — the best single of the episode on both metrics**, +0.00012 CV vs parent V12 with **folds improved 5/5** (+0.00007…+0.00019), +0.00028 CV / **+0.00055 LB vs V2 = the second move ever outside the 0.0003 tie band**, and above the best published single (0.96105). 17.6 min, 289 frame cols → 374 expanded, 1.20 GB, peak reserved 3.10 GB, fit 146 s/fold (V12's 154 s — 13 columns are free), EV build 115 s. ❌ **V17 epoch ladder = CV 0.96092 / LB 0.96085**: rungs 6/8/10 scored 0.96089 / 0.96083 / **0.96092**, i.e. **every rung ~0.00025–0.00034 BELOW V12's 3-epoch budget**, folds improved 1/5, all 15 fits pinned at their own ceiling (`max_epochs=N reached`), 86.4 min. ❌ **V16 CatBoost 3-rung ladder = CV 0.96089 / LB 0.96048** (rung of record `none`): the metric half (V16a) measured **+0.00001 vs V4** and the frame half (V16b) +0.00006 (`add`) / +0.00007 (`swap`) with **3/5 folds each** → the pre-declared ≥4/5 rule kept `none`; 63.0 min, and no `_x_*` cross reached fold-1's top-15. ❌ **V19 LightGBM `min_child_samples` 20→40 = CV 0.96103 / LB 0.96057**, **−0.000024 vs V15** on 4/5 folds of ≈+0.00001 (fold 1 −0.00015); 26.1 min CPU; best iters shifted up to 329 trees (1068→739) while the importance profile stayed V15's column-for-column.
- **Governance decision (same day):** V18 was authored against PLAN_V16_ONWARD's "parent = V17's winning rung", and its own guard correctly refused to start while V17 had not run. The owner chose to **re-anchor V18 to V12 at V12's exact 3-epoch budget** instead of waiting 86 GPU-min for the ladder. That turned out to be the *correct* experimental call, not just a schedule saving: the ladder lost to 3 epochs, so a V18 measured on a V17 rung would have been a comparison against a worse parent. Standing rule extracted: **a child may be re-anchored to the parent's frozen config when the intermediate arm is unrun — the single-axis property is preserved either way, and the frozen-parent reading is the conservative one.**
- **Timing (batch):** V18 17.6 min GPU, V17 86.4 min GPU, V16 63.0 min GPU, V19 26.1 min CPU → **167 min ≈ 2.8 GPU-h**, against the plan's 2.7 h estimate for the whole episode — on budget, and V19 consumed none of it.
- **Key Learning (the batch, and the one that matters most):** **V18 vindicates the channel theory and closes it as a rule.** Across V8–V19 the only two feature moves that ever produced CV are the two that hand the model a *prediction it was not given* — V12's teacher-logit + twin representation (+0.00016, 5/5) and V11's rating EVs, which now score **+0.00012 on XGBoost and +0.00012 on RealMLP, 5/5 folds on both** — and they **stack** rather than overlap (0.96101 → 0.96113 → 0.96117 → 0.96129). Every re-cut of already-splittable information measured ≈0, including the two I had left open as the best bets (`_tmr_*`-style interactions in V16, V19's leaf-count capacity knob). Second, and a self-correction I owe the ledger: **my "the net is epoch-starved" reading was wrong, and the number it came from had no information content.** With `use_early_stopping=False`, PyTabKit always restores the *last* epoch, so "best epoch = 3 = the budget" was mechanically guaranteed in V5/V6/V7/V12 and could never have evidenced starvation; V17 then showed 6/8/10 all lose to 3, because the recipe's schedules (`flat_anneal`, `cos_log_15`, `expm4t`, `sqrt_cos`) are all **budget-relative** — extending the budget stretches every anneal instead of adding more of the same. Third datum, cheap and reusable: at 10 epochs V17's pooled OOF (0.96092) sat **0.00017 below its fold mean** (0.96109), while V12 and V18 show OOF ≈ fold mean — `OOF − fold mean` is now a calibration-drift check on every net arm.
- **Status:** 🏆 **V18 (CV 0.96129 / LB 0.96106) is the new best single and the new candidate control**, superseding V12; **V2 (0.96101 / 0.96051) remains the standing control until paired DeLong z > 3 vs `v12_oof` and `v2_oof` and the `SEED_REPRO=2027` rerun confirm V18** — both still open, and nothing gets promoted on CV+LB alone. ❌ V16, V17, V19 all rejected; their axes are closed by measurement and recorded in the dead-ends table. 🔬 Still unrun: **V20 (soft-label targets — the last live bet, and the only arm aimed at the 3.8% flip noise itself)**, V14 (archived OOF/sub, no scores — cannot enter the score tables by the ledger-integrity rule), and Step 0's gate kernel + `v12_oof` vs `v2_oof` DeLong. ⚠️ The ensembles track (ENS-A over `v11`/`v12`/`v15`/`v4`, ENS-FINAL) is now materially better-founded — V18's OOF is a fourth independent leg — but it is the owner's decision, never a numbered version.

---

### 08-10-2026

- **Goal**: run the V6 architecture arm (bagged RealMLP) that `PLAN_V6_ONWARD.md` put first
- **Experiments**: **V6 = V5 with one knob, `n_ens` 1→8** (AST: all 12 functions identical to V5; 56 other lines are docstring/CFG/prints) → **CV 0.96064 / LB 0.96045**. Paired DeLong: **+0.000886 vs V5 (z=+13.40, 5/5 folds)** and **−0.000372 vs V2 (z=−5.90, 0/5)** → rejected as control, V2 stands. Early stopping never fired again (`max_epochs=6 reached` ×5), so the epoch budget is the remaining untested knob on this class. Also measured on saved OOFs: equal logit-mean V2+V4+V6 = **0.961201 (+0.000192)**, the best own-model combination so far; adding V5 dilutes it to +0.000137.
- **Timing**: 22.6 min (prep 18–19 s/fold, fit 247–249 s/fold). My 2 h estimate was 5× wrong — PyTabKit's bags share one data pipeline, so cost scales ~1.5× not 8×.
- **Key Learning**: architecture is now bounded: a bagged RealMLP on our matrix reaches 0.96064, i.e. within 0.0004 of the control but not past it, with the FE axis measured closed (V2 clean-law AUC 0.9990, headroom 0.0009–0.0019). The only lever with a measured gain left is combining our own legs (+0.00019), which stays below the 0.0003 tie band — so the decision in front of us is submission strategy, not modeling.
- **Status**: ❌ V6 rejected as control; ✅ `v6_oof.csv` = preferred third leg. **V2 remains control (0.96101 CV / 0.96051 LB).**

---

### 07-10-2026

- **Goal**: second research campaign (16 agents: public notebooks/discussions/leaderboard, 386-writeup digest library, academic ceiling theory, local data study, independent audit) → re-plan V5 onward as **single-model only** and run the first arm
- **Experiments**: (1) Public state: LB #1 **0.96184**, ranks 1→49 span **0.00008** (1,054 teams, 20% public), 269 public notebooks, 15 forum threads (all read), 12 public OOF libraries. (2) **Our folds are canonical**: reconstructed `SKF(5,shuffle,rs=42)` reproduces all 20 logged per-fold AUCs to 5 dp and matches golem's `folds_seed42.npy` on 699,635/699,635 rows. (3) Stack route measured locally (109 public legs + LR-on-logits = CV **0.962095**, Δ+0.001086, z=+19.23, leakage-tested clean) → **rejected by the owner: versions must be our own models**; numbers kept, code deleted. (4) Ceiling derived + simulated: `AUC_noisy−½ = κ(AUC_clean−½)`, ρ from orig-pure regions = **0.036–0.039 (heteroscedastic)** ⇒ cap ≈ **0.9630**; the old "~4%" is falsified (it caps at 0.95895 < our own CV). Noise arms provably inert (η̃ monotone in η): region recalibration −0.000032, constant-rank override −0.0002…−0.0066. (5) **Three pinned numbers corrected**: `count_ge4_inflight8` is **0.8132 not 0.8352** (ordering inverts); FD-last-digit-0 is a **distance confound** (condition on FD//5 → −0.0100, z=−3.3) so the digit arm is dead here; missing-value swing is **0.000415 not 0.0004** (no fill helps). `nina2025` vanilla-CatBoost 0.96124 is a **phantom** (no such kernel in the API index). (6) **Backwards rating found**: `Departure/Arrival time convenient` ρ=−0.90 and it sits inside our 13-column aggregates (0/8 mined kernels notice); excluding it lifts `_mean_excl0` 0.8127→0.8383. (7) **V5 RealMLP_TD single** (n_ens=1, flat_anneal, 6 ep) on the frozen V2 matrix → **0.95975 CV / 0.96032 LB**, CV −0.00126 vs V2 → rejected; ES never fired, so the class is undertrained rather than disproved. Three failed submissions en route came from invented PyTabKit arg names — fixed by copying our own S6E2/S6E3/S6E5 scripts.
- **Timing**: V5 15.3 min on Kaggle GPU (prep 21–22 s/fold, fit 154–158 s/fold) vs V2 3.8 min — the net costs ~4x wall time for a worse score at this config.
- **Key Learning**: The single-model route is capped: best published singles here are CV ≈0.9613–0.9614, so even a win is +0.0004…0.0006 LB = tie-band, not a rank climb (the frontier is combiner-built). Also: **copy library argument names from our own archived scripts, never from documentation memory** — three sessions were lost to `epochs` vs `n_epochs`. And CV→LB gap sign flipped for the net (+0.00057 vs V2's −0.00050): cross-architecture CV comparison is not gap-neutral, so audit nets on LB.
- **Key Learning (added after the two targeted region dives)**: **feature engineering is finished here.** V2's implied clean-law AUC is **0.9990** (κ=0.924 at ρ=0.036) ⇒ 95–98% of our gap to perfect is irreducible and the total headroom for any model is 0.0009–0.0019. The Personal-travel slack is unreachable (the generator re-drew that posterior: Spearman orig-vs-comp 0.823 in Personal vs 0.990 in Business; every orig blend is worse; best-of-31 blends = +0.0000618 whole board = 1/5 of the tie band), and the original channel has nothing left (best orig lookup 0.887 in-region vs V2 0.937, all blends negative). **Retracted my own `_phy_minus_dig` arm**: residual sign-consistency across 32 slices is not evidence of incremental value — the digital/physical split *lowers* within-Personal AUC when blended with V2. Method rule learned: test a candidate by blending it against `v2_oof` in the target region, never by residual-sign counting.
- **Status**: ❌ V5 rejected (paired DeLong dAUC −0.001258, z=−15.67, 0/5 folds) — **V2 remains control (0.96101 CV / 0.96051 LB)**. Plan changed: FE arms (TE keys, `max_bin`, `lossguide`) are now expected-null and kept only for closure; the two live axes are **architecture with a real training budget** (bagged RealMLP `n_ens=8–32`, or TabPFN-3.5 inference — published singles 0.96134–0.96141 vs our 0.96101) and **combining our own models** (measured: V2+V4 +0.000113, V2+V5 +0.000112, 4-leg pool +0.000178 z=+6.99). Awaiting the owner's choice.


### 02-10-2026

- **Goal**: Research campaign (dataset deep-dive + public mining) → establish official V1 baseline
- **Experiments**: (1) local EDA: verified label law near-deterministic in data.csv (wifi≤3&Personal → 0.0000 sat) diluted to 0.036–0.040 in synthetic train ⇒ ~4% symmetric flip noise; route-ID FD structure, zero-block survey breaks, digit artifacts, no train/test shift (KS≤0.0027). (2) Public mining: S6E10 discussions/kernels, airline-dataset kernels, S6E9 winner writeups, own S6E1–E9 logs → FE ladder written to feature_engineering.md. (3) Dev-run (house params, blanket TE, weights, rule-region flags): 0.96009 CV — flags hijacked fold-1 gain (0.37 on one AND-flag). (4) **V1 official**: ektarr-aligned XGB → **0.96061 CV / 0.96019 LB** (Gap -0.00042). (5) tomwarrens EDA mined: route-rate +0.0013, row-append +0.0003 (dead again), adversarial 0.4995, same-flight variance +1.5% → random folds safe. (6) **V2 WideProfile** (4 axes: int-key prep, ektarr-spec d8/21-feat teacher, wide route profile std/resid/shares, org means all-21 + triple) → **0.96101 CV / 0.96051 LB** (+0.00040 CV / +0.00032 LB vs V1, all 5 folds improved, 3.4 min vs 26.7 min). (7) **V3 DuoTeacher** (xgb+lgb logit-mean teacher) → 0.96096 CV / 0.96065 LB — tie, tree+tree diversity saturated; V2 remains control. (8) **V4 CatBoostMember** (V2 package, model swapped; two API fixes en route) → 0.96088 CV / 0.96051 LB; solo −0.00013 but V2+V4 blend = **0.961122 (+0.000113)** → stack ingredient secured; day-4 public mine done (frontier 0.96152 = 38-member stack; aux-EV features are the new positive; orig-rows −0.00038 confirmed with noise-floor).
- **Timing**: V1 26.7 min on Kaggle GPU; ~90% of it per-fold prep (string pair-key joins ~290 s/fold). V2 same recipe minus the string-join tax: **3.4 min total, prep 10 s/fold** — the speedup prediction held exactly.
- **Key Learning**: `_og_teacher` = 48% of fold-1 gain — the orig-trained teacher is the denoised law prior, so teacher capacity is the highest-leverage remaining axis; deterministic fixed relations must NOT be trained as features (tree-recoverable, steals splits); target-derived stats (fold-safe TE, orig means, teacher) are legitimate and dominant; NaN handling is bounded ≤0.0004 AUC — never spend a version on it. Int-key factorized pair builder + wide route profile + ektarr-spec teacher staged as V2 candidate.
- **Status**: ✅ V2 remains control (0.96101/0.96051); V4 kept as stack member (+0.000113 blend proof); next: V5 aux-EV features on V2 base, then V7 fold-sealed stack over v2/v4/v5 OOFs

### 28-10-2026 — continuation: V55's second Kaggle attempt failed on my guessed 2% guard — fixed by changing the design, not the threshold

- **Goal**: read the new failure honestly. The EV fix held (the run got 42 lines further, to the twin pass), and this time the arm's **own guard** stopped it: `twin Departure Delay in Minutes_cat_: 531 of 25,972 augment rows (2.04%) carry a value the competition pool never had, above the 2% guard`.
- **Why the guard, not the data, was the defect**: the 2% was a number I invented while authoring, with no measurement behind it. The real coverage gap is a property of the frame (the twin code space is the train+test pool's, V12's axis) and it is ~2% on one delay column. Raising the cap to 3% would have been the post-hoc move the ledger rules exist to prevent - and it would still have shipped 531 rows whose extreme delay was silently encoded as `categories[0]`, i.e. **a long delay recorded as no delay**.
- **The fix (design, not threshold)**: `filter_unrepresentable_rows` now **drops** any A row the pool's twin codes cannot express - a novel value, or a NaN in a column whose pool carries no missing level - and prints the per-column count plus the positive-rate shift on the kept rows (0.4920 → 0.4925 in calibration, i.e. label-blind). `add_numeric_categorical_twins` keeps no threshold at all: it asserts the **post-condition** that every kept row has a real code, so the two views must agree or the run stops. A declared `CFG.AUG_COVERAGE_CAP = 0.10` remains as the "the design is wrong, not the number" limit.
- **Why not instead widen the vocabulary by one "novel" level**: that changes one-hot twin widths for the competition rows as well, which moves the model's input and would make the axis rows **and** features. Dropping rows keeps V18's features byte-identical, which the idempotence snapshots then prove.
- **Calibration (synthetic frames; no competition data, no xgboost)**: 5/5 in `check_v55_coverage.py` — the filter drops exactly the injected 192 of 3,000 rows; the drop mask never reads the label (7 mask lines, none mentions `TARGET`); every kept row maps in all **17/17** twin columns with zero NaN; the **falsifier** — skipping the filter — trips the arm's own assertion (`180 kept A rows still have no pool code`); and competition twin columns are **identical** with and without the A pass. The org-stat leak harness still passes 3/3 (flip moves nothing; the wrong fit source moves it 1.102e-01; idempotence). One harness expectation of mine was wrong and was corrected rather than the code: `CFG.TARGET` legitimately appears once, in the report.
- **Static**: compiles; docstring 35 lines (at the cap); the three parameter dicts are **AST-identical to V18**; 2 `to_csv`; the arity audit's only remaining flag is its known false positive (`fill_continuous_nan` returns a list, which an AST cannot size).
- **Timing**: ~4 min of the second attempt consumed before the stop; two failed attempts now, both caught by checks rather than by a wrong number being recorded.
- **Status**: file fixed and re-audited — **re-download `Arms/S6E10_V55_RealRowsAugment.py`** before the next attempt (task 50). Expect ~25,972 − (a few hundred) appended rows; the coverage line prints the exact number. Admission unchanged: solo CV ≥ 0.96140 or no board slot.
