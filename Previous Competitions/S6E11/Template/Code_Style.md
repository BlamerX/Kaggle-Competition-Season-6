# S6E11 Version Style & Naming Spec

Censused from every archived arm in `Previous Competitions/` — S6E1 (125 scripts), S6E2 (71), S6E3 (88), S6E4 (48), S6E5 (27), S6E9 (51), S6E10 (55). S5E12 has no scripts. Where episodes disagree, §9 shows the counts that picked the winner rather than a preference.

New version files are written from [S6E11_TEMPLATE_ModelArm.py](S6E11_TEMPLATE_ModelArm.py) (an arm that fits) or [S6E11_TEMPLATE_BlendArm.py](S6E11_TEMPLATE_BlendArm.py) (an arm that combines saved vectors).

---

## 1. File naming

| Artifact | Pattern | Example |
| --- | --- | --- |
| Version script | `S6E11_V{N}_{PascalCaseAxis}.py` | `S6E11_V7_AuthMethodRate.py` |
| Retired script | `RETIRED_S6E11_V{N}_{PascalCaseAxis}.py` | `RETIRED_S6E11_V12_SmoteRung.py` |
| OOF vector | `v{N}_oof.csv` — columns `(id, pred)` | `v7_oof.csv` |
| Submission | `v{N}_sub.csv` — columns `(id, is_fraud)` | `v7_sub.csv` |

1. **`V{N}` capital in the script name, lowercase in the artifacts.** All 55 S6E10 scripts use `S6E10_V42_CatBoostMVS.py`; all 29 paired writes use `stem = f"v{CFG.VERSION_NUMBER}"`. Do not unify the case — the split is what separates a script from its vector in a flat listing.
2. **`{PascalCaseAxis}` names what changed, not the model.** `V42_CatBoostMVS` (one sampling key), `V53_TabMFamily` (the family). `V42_CatBoost` is wrong: a reader must learn the axis from the filename.
3. **No `_ladder` / `_rung` suffix in any filename.** A sweep ships one configuration; the rejected rungs live in stdout and the ledger entry.
4. **`RETIRED_` prefix, in place, never deleted.** S6E10 keeps `RETIRED_S6E10_V31_RegionAlpha.py` and `RETIRED_S6E10_V32_NetBaggingCapacity.py` beside their replacements. A superseded arm is the control a later arm was measured against.

## 2. The two-writes law

**An arm writes exactly two files: `v{N}_oof.csv` and `v{N}_sub.csv`.** S6E10 states this in three banners (`# [4/5] SAVE OUTPUTS (the blended single-model vector; two-to_csv law)`) and breaks it in four arms: V8 wrote `{stem}_ladder_{L}_oof.csv` per rung, V14/V16/V17 wrote one OOF per rung on top of the shipped pair.

Why it is a law and not tidiness: an extra vector is an *unledgered* result. A per-rung OOF left in `/kaggle/working` gets blended by a later arm without passing the gate, and `Memory/public_scores.md` has no row for it.

**Sweep arms keep rungs in memory and print them**, then write the pair for the winning rung only. If per-rung vectors must survive, that is two arms: a probe that prints, a ship that writes.

Also banned: model checkpoints, `.pkl`/`.joblib`, weights, intermediate frames, `*_log.txt` (S6E1 has six stray ones — the log is stdout, and the owner pastes it into `Memory/training_logs.md`).

### Where the two files live

| Stage | Path | Set by |
| --- | --- | --- |
| Produced | `/kaggle/working/v{N}_oof.csv` + `/kaggle/working/v{N}_sub.csv` | `out_dir = "/kaggle/working"` in every fit arm (all 52 of S6E10's Kaggle arms) |
| Archived | `<episode checkout>/Previously trained files/oof/v{N}_oof.csv` + `.../sub/v{N}_sub.csv` | where the download lands; the two dirs are created by the first combiner arm |
| Cited | `Memory/public_scores.md` rows name both: `` `v51_oof.csv` `` / `` `v51_sub.csv` `` | the archive path is what makes a ledger row checkable |

A fit arm never writes into the archive — it writes to `/kaggle/working` and the owner downloads. A combiner arm reads the archive and writes its own pair beside it, which is why blend arms carry `OOF_DIR`/`SUB_DIR` in `CFG` and fit arms carry only `out_dir`.

## 3. Header: ≤12 lines, this version only

The header is an index card for **the arm you are writing**, not a digest of the campaign. Census of what the house used to do: docstring medians ran 8-10 lines in S6E1-S6E2, 30 in S6E4, 36 with a 92-line max in S6E9, and S6E10 pulled back to 26 — and its early arms (V1-V6, 39-68 lines) are the ones written before that pull-back. The cap below is where the practice was heading, made explicit.

Seven fields, in this order, one line each unless a second line is unavoidable:

```python
"""
S6E11 V{N} - {Name}: {the one change, in a clause} - {CPU|GPU (cuda)}
GOAL:   {the open question this arm closes — one line}
Axis:   {the ONE thing changed vs the parent; what stays frozen — one line}
Parent: {path to the archived script it clones}
Gate:   {pre-declared numeric adoption test + the ruler it is judged against}
Cost:   {expected minutes, device, and the losing outcome that is live}
Outputs: v{N}_oof.csv (id, pred) + v{N}_sub.csv (id, is_fraud) to /kaggle/working — exactly two writes.
"""
```

**Nothing else goes in the header.** Specifically excluded, each because an episode used it:

| Banned | Was | Why it is out |
| --- | --- | --- |
| `Strategy:` prose | S6E4 28×, S6E5 27×, S6E9 44× | A restatement of the code below it. The code is the strategy. |
| `Key Techniques:` numbered lists | S6E5 9×, S6E10 V1-V6 | Grows with the script and never gets pruned; V1's is 18 lines describing 603 lines. |
| `Pipeline:` | S6E4 30× | Same as Strategy. |
| `Reference:` / `References:` Kaggle URLs | S6E4 27×, S6E9 34× | The attribution belongs in the ledger entry and the writeup, not in every clone of the file. |
| `Dataset Structure:` dumps | S6E9 19× | Column facts the reader gets from `train.csv` in one line. |
| `Golden Rules:` | S6E5 26×, S6E9 17× | House law duplicated per file — it drifts. Law lives in this spec and in the `Memory/` RULES blocks. |
| `Evidence:` / plan-section cites | S6E10 17× | Fold the reference into `GOAL:` in four words, or leave it in `Memory/daily_log.md`. |
| Sibling scores, prior arms' deltas | S6E10 throughout | **No hardcoded numbers about any other version.** A sibling's number moves and prose does not. Cite `Memory/public_scores.md` by name if a bar must be stated. |
| Campaign history, owner-law dates, "the lesson from V44/V45" | S6E10 V49-V55 | The ledger holds the narrative; the header holds the instruction. |
| `=====` underline art inside the header | S6E10 V1 | Decoration consumed three of the 39 lines. |

Three prohibitions that come from arms that cost GPU time in S6E10:

- **A clone rewrites the `Axis:` line and the footer's axis clause, not just `CFG`.** Copy a script, change a number, and the banner still describes the parent.
- **The config dict is diffed against the parent before upload.** A banner can lie, a control can pass, a dict cannot.
- **`Gate:` is written before the run.** A threshold chosen after seeing the number is a description, not a test.

## 4. Footer: the summary prints this version's numbers and nothing else

`[5/5] FINAL RESULTS` is the last thing in the file. It restates the arm, reports its own measurements, and lists its two writes — that is all.

```python
print(f"\n[5/5] RESULTS | {CFG.VERSION_NAME} | Device: GPU (cuda)")
print(f"   Axis: {one clause, this arm's own axis}")
print(f"   OOF ROC AUC: {oof_cv:.6f} | OOF PR-AUC: {oof_pr:.6f}")
print(f"   Fold ROC AUCs: {[f'{s:.6f}' for s in fold_scores]}")
print(f"   Fold mean +/- std: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
print(f"   Fold PR-AUCs:  {[f'{s:.6f}' for s in fold_pr]}")
print(f"   Timing: prep {np.mean(prep_times):.0f}s + fit {np.mean(fit_times):.0f}s per fold | total {total_min:.1f} min")
print(f"   Saved: {stem}_oof.csv + {stem}_sub.csv")
```

Rules for the footer:

1. **Six decimals on AUC, not five.** S6E10's ruler is ±0.00006 CV, so five decimals rounds the quantity the gate is judged on. This is the one place the earlier episodes were wrong, not just different.
2. **No comparison table, no sibling rows, no "vs V18" block.** The footer prints what this run measured; comparison is arithmetic done on the saved vectors, and it belongs in `Memory/public_scores.md`.
3. **No restating the docstring.** If the axis needs three lines to be understood, the arm is two arms.
4. Keep the V53/V42 house form `RESULTS | {CFG.VERSION_NAME} | Device: ...` — one line, indented facts below. S6E10 V1's `=` × 80 rules around the footer are the older shape.

## 5. Fit-arm scaffold: eight fixed slots

```
# =============================================================================
# 1. IMPORTS
# 2. CONFIGURATION
# 3. SEED EVERYTHING
# 4. MODEL PARAMETERS
# 5. METRIC - ROC AUC
# 6. FEATURE ENGINEERING
# 7. PER-FOLD FEATURE BUILDER
# 8. MAIN EXECUTION
# =============================================================================
```

Slots are **fixed and numbered**, and an arm that has nothing for a slot keeps the banner and names its state on the same line — the house already does this: three S6E10 arms write `# 7. PER-FOLD FEATURE BUILDER - byte-identical to V11`. Deleting a banner is what made S6E5 and S6E9 drift (see §9): once a slot can vanish, the numbering below it becomes model-dependent and no two files in the episode agree on where `MAIN EXECUTION` is.

Slot notes:

- **2. CONFIGURATION** — one `class CFG:` holding `VERSION_NUMBER`, `VERSION_NAME`, `EXP_ID`, `PARENT`, `TRAIN_PATH`, `TEST_PATH`, `ORIG_PATH`, `TARGET`, `N_FOLDS`, `RANDOM_SEED`, `SEED_REPRO`, then this arm's axis parameters. The 55-file census has `VERSION_NUMBER`/`VERSION_NAME`/`EXP_ID`/`TARGET`/`TRAIN_PATH` in 53-54 of 55 and `RANDOM_SEED`/`N_FOLDS`/`TEST_PATH` in 52.
- **3. SEED EVERYTHING** — `seed_everything(seed)` over `np`, `random`, `torch` when imported; called before any data work.
- **5. METRIC - ROC AUC** — ROC AUC is the scored metric and keeps the slot name even though this arm also computes PR-AUC; put both in slot 5 and name them `auc_score` / `pr_auc_score`.
- **6 vs 7** — slot 6 is target-free feature work done once on the whole frame; slot 7 is anything that touches `y` (target encoding, rates, EV maps, teacher OOF) and MUST be refit inside each fold on fit rows only. An arm with no target-touching work says so on the slot 7 banner.
- **8. MAIN EXECUTION** — `t0_all = time.time()`, the `=` × 80 head banner (`Version: V{N} | {Name} | Device: ... | Build: ...`), the `Axis:` echo line, then the five stages below.

Stage banners inside slot 8, each with a matching `print`:

```
[1/5] LOAD DATA      [2/5] FEATURE ENGINEERING   [3/5] TRAINING (N-Fold CV, per-fold ...)
[4/5] SAVE OUTPUTS   [5/5] FINAL RESULTS
```

Denominators are the total stage count — S6E10 V49 prints `[5/4] RESULTS`, and that is the bug, not the convention.

## 6. Guards that every fit arm carries

Each of these exists because an S6E10 run was lost without it. They are not optional and they are not documented in the header — the code is the documentation.

| Guard | Where | Recorded failure it prevents |
| --- | --- | --- |
| Library version assertion | slot 1, after imports | `sklearn` ≥ 1.3 for `TargetEncoder`; print the version, `raise ImportError` otherwise |
| Import-with-install for optional libs | slot 1 | `try: from pytabkit import … except ImportError: subprocess pip install` (V53) |
| Pre-flight config build | before slot 8 loads any data | construct the estimator and read every config key back; abort on a key the build neither names nor records |
| Post-fit read-back, fold 1 only | inside the fold loop, after `fit` | construction accepts values the core rejects at fit — V42's `sampling_frequency` and `SqrtDegree` died twice this way; K8 fitted a different net and still printed clean controls |
| Wall-time canary, fold 1 | inside the fold loop | declared ceiling per fold with the fallback rung named in advance; the fallback is a **re-run**, never a mid-run edit |
| Memory probe before the fold loop | slot 8 | neighbour-matrix and full-frame allocations that OOM at fold 3 |
| `del` + `gc.collect()` per fold, `torch.cuda.empty_cache()` for nets | end of fold loop | 5-fold OOM on a 16 GB T4 |
| Fold-1 feature importance print | fold 0 only | one of the two places a wrong target-mean encoder shows up |

## 7. Per-family adjustments to slot 4

Same scaffold, different slot 4 and different read-back. Each recipe names the archived script it must be copied from — copying kwargs out of our own archived scripts is the only provenance that has held; recalling an API from documentation memory cost V42 two dead runs.

| Family | Recipe to copy from | Slot 4 specifics | Mandatory read-back |
| --- | --- | --- | --- |
| XGBoost | `S6E10_V1`, `S6E10_V8` (leaf-wise), `S6E10_V20` (soft labels) | `enable_categorical` must match the dtype work; `early_stopping_rounds` | `best_iteration` per fold |
| LightGBM | `S6E10_V15`, `S6E10_V19`, `S6E9_V46` | `min_child_samples` ladder is its own axis | `best_iteration` per fold |
| CatBoost | `S6E10_V4`, `S6E10_V16`, `S6E10_V42`, `S6E9_V8` | `task_type='GPU'`, `eval_metric='Logloss'` (AUC on that build runs on CPU — V16 measured 63 min for 15 fits) | `bootstrap_type` / `subsample` off the **fitted** object |
| RealMLP / hand-written net | `S6E10_V5`-`V7`, `V17`, `V18`, `V44`, `V48`, `V54` (DCNv2), `S6E9_V47` | epochs + patience, batch, lr, and the `torch` dtype discipline | `n_epochs` and every key off the trained object |
| TabM / pytabkit | `S6E10_V53`, `S6E9_V6`, `S6E9_V29` | `arch_type`, `tabm_k`, `num_emb_type`, `d_block`, `n_blocks`, patience | convergence is early stopping: a run that used every epoch was not converged |
| Combiner / blend | `S6E10_V49`-`V51`, `S6E9_V51` | `MEMBERS`, `REFERENCES`, `_logit`/`_sernels`/`_delong` | id-alignment `assert` on every vector read, OOF and sub |
| Distance / kNN | `S6E10_V38`, `S6E9_V15` | neighbour count, metric, chunking | print the matrix size before the fold loop |
| Imbalance arm (SMOTE / weights / undersample) | new to this episode — no ancestor to copy | `scale_pos_weight` or a sampler, one per arm | assert the resampled positive count and the resulting rate, per fold |

## 8. Imbalance, which is what makes S6E11 not S6E10

S6E10's target was near-balanced. This one is not (~1.70% positive in the original; the competition rate is measured by the first EDA arm and recorded in the ledger, not here), and the difference changes what an arm must report:

- **ROC AUC is the score; PR-AUC is the diagnostic.** Both print per fold and both go in the footer. A ROC AUC lift with a flat PR-AUC is a false-positive-side improvement and the ledger says which.
- **Fold AUCs on an imbalanced 5-fold split are noisy**: report mean ± std plus the pooled OOF, and never call a one-fold move an effect.
- **Resampling, class weights and threshold tuning are axes like any other** — one per arm, named in `Axis:`, parent frozen. A SMOTE arm that also re-tuned depth measured nothing.
- **The decision threshold is not scored.** AUC is rank-based, so `v{N}_sub.csv` needs only monotone probabilities; threshold work belongs in the writeup.
- **Fold counts stay at 5** for comparability with the house ruler, and pooled-minus-fold-mean is printed — the quantity S6E10 used to catch pooled-OOF drift.

## 9. Why this scaffold won the census

| Episode | Scripts | Docstring median (max) | Section banners | Field vocabulary |
| --- | --- | --- | --- | --- |
| S6E1 | 125 | 10 (22) | ad hoc — `1. DATA LOADING` 46×, `1. CONFIGURATION` 29× competing | none systematic |
| S6E2 | 71 | 8 (14) | ad hoc — `1. Import RAPIDS` 11× | none systematic |
| S6E3 | 88 | 20 (43) | one-off banners per file | none systematic |
| S6E4 | 48 | 30 (80) | **scaffold introduced**: 1. IMPORTS 29×, 2. CONFIGURATION 29×, 3. SEED 24×, 4. MODEL PARAMETERS 20×, 5. METRIC 22×, 7/8. MAIN EXECUTION 25× | `Pipeline:` 30, `Strategy:` 28, `Reference:` 27, `Device:` 27 |
| S6E5 | 27 | 29 (62) | slots 1-3 fixed, 4-7 drift (MAIN EXECUTION at 6 in 11 files, 7 in 10) | `Strategy:` 27, `Golden Rules:` 26 |
| S6E9 | 51 | 36 (92) | two numbering systems in one episode — `1. CONFIG`+`2. SEED`+`3. METRIC` in 18 files vs `1. IMPORTS`+`2. CONFIGURATION` in 28 | `Strategy:` 44, `Device:` 27, `Dataset Structure:` 19 |
| **S6E10** | **55** | **26 (68)** | **8 slots, 52/55 agreement on every one, `7. PER-FOLD FEATURE BUILDER` in 47** | `Axis:` 49, `Outputs:` 49, `Parent:` 47, `Expected / Cost:` 45, `Device:` 45, `Gate:` 44, `GOAL:` 40 |

S6E10 is the pick on both axes that matter: the scaffold is the only one that holds across model types (its 55 arms span XGBoost, LightGBM, CatBoost, RealMLP, TabM, DCN-V2, kNN, distillation, stacking and blends under one numbering), and its slot-7 split is what makes it model-type agnostic — target-free work and per-fold work get separate permanent homes, so a family swap does not renumber the file. S6E4 invented the scaffold, S6E5/S6E9 lost it precisely because a slot could be deleted, and S6E10 fixed that by giving slot 7 a job every model type either has or explicitly disclaims.

Its header did **not** win on size — median 26, with late arms at 31-35 and the early ones at 39-68 — which is why §3 caps it at 12 lines and bans by name the fields the other episodes carried. The scaffold is adopted; the prose habit is corrected.

## 10. When the arm is not a model

- **EDA / instrument arms** (no fit, no submission): same header minus `Gate:`, and they write **zero** files. Their output is printed measurements that go into `Memory/feature_engineering.md`. An instrument that writes a CSV becomes a vector someone later blends without a gate.
- **Local-only analysis** (blending lab, offline scorer): `ROOT` points at the local checkout, `OOF_DIR`/`SUB_DIR` are read-only, and the arm still writes at most the one `v{N}_oof.csv` + `v{N}_sub.csv` pair if it ships a submission.
- **Anything that cannot ship an artifact is a probe, not a `V{N}`.** Give it a `lab_` filename without a version number and keep it out of the ledger, so the version sequence stays a sequence of scored experiments.
