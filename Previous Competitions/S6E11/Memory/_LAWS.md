# S6E11 Canon Laws

The four ledgers in this folder cite laws by number — `law 9`, `Law 3`, `the 2026-10-11 standalone law`, `the two-to_csv law` — and **no file in any previous episode states them.** S6E10's `daily_log.md` leans on "law 9" seven times; the reader has to reconstruct it from a parenthetical. S6E5 and S6E9 attacked the same problem the other way, by pasting a `Golden Rules:` block into 26 and 17 individual version scripts, where each copy drifted from the others.

This file is the single statement. The per-ledger `⚠️ RULES` blocks stay, but they now hold only what is specific to that ledger and point here for the rest. A law is listed only when a recorded failure or an owner instruction backs it, and the provenance is given.

---

### Law 1 - Compute happens on Kaggle

Every model is fitted on Kaggle (T4 or Kaggle CPU, chosen per arm's budget). Nothing is fitted or measured locally except a combiner over vectors already archived, which is arithmetic on saved columns and costs seconds.

> **Provenance:** the owner's standing instruction, restated 2026-10-10. S6E10's ledger calls it "Law 3"; the reason it is a law is that a locally-fitted number has no run of record behind it.

### Law 2 - A result needs CV *and* a confirmed LB

A row enters `public_scores.md` only when the OOF score and the public LB score both exist, and `training_logs.md` only after the LB is available. A CV with no submission is an experiment that is still open; an LB with no OOF is an unexplainable number.

> **Provenance:** cited throughout S6E10 as "law 9 (entries only once CV *and* confirmed LB exist)"; `public_scores.md:131` "law 9 needs CV *and* a confirmed LB on both sides", `:260` "law 9 barred the row".
>
> **What this does not bar:** setup days, record audits and instrument runs have no score and are still recorded in `daily_log.md` — labelled `no score entry`, so a reader cannot mistake them for results.

### Law 3 - Standalone: a saved OOF is never a feature or a target

An arm fits one model from data it computes itself inside the run. No archived prediction vector enters its feature matrix, its labels, or its weights. The exception the owner granted on 2026-10-25 is the **composition class**: a combiner arm may read archived `oof`/`sub` vectors and blend them, because that is a model over vectors rather than a model over leaked features. Even then, a blended vector must not be fed back into a fit arm.

> **Provenance:** `S6E10/Memory/training_logs.md:234` — "the owner lifted the 2026-10-11 standalone law for composition arms on 2026-10-25, so reading saved vectors is now lawful for this class."

### Law 4 - Two writes, no more

An arm produces exactly `v{N}_oof.csv` and `v{N}_sub.csv`. Extra vectors are unledgered results: they sit in the working dir, get blended by a later arm, and never pass a gate.

> **Provenance:** stated in `S6E10_V44`'s save banner ("two-to_csv law"), broken by V8, V14, V16 and V17, which each wrote one OOF per rung of a ladder on top of the shipped pair. Naming and artifact paths: [`Template/Code_Style.md`](../Template/Code_Style.md) §1-2.

### Law 5 - The gate is written before the run

`Gate:` in the header carries the numeric adoption test, the ruler it is judged against, and the losing outcome that is live. A threshold chosen after seeing the number is a description of the number, not a test of the hypothesis.

> **Provenance:** every S6E10 mature arm states it; the campaign's costliest self-inflicted wound was pricing an axis against someone else's config because no in-run control was declared (V37).

### Law 6 - History is never rewritten

Entries are added; earlier entries are not edited, re-sorted or deleted. A correction is a new dated entry that cites the entry it corrects and says what changed.

> **Provenance:** rule 2 in every S6E10 ledger. The need for it is concrete: `S6E10_V43`'s entry records that "two inherited banners lied", which is only recoverable if the lying banner is still there.

### Law 7 - Every number has a provenance label

Figures are marked as one of: a Kaggle run of record (with fold values), an owner-confirmed LB, arithmetic over this file's own columns (`LEDGER`), a measurement quoted from a source card, or an estimate. No number is written as fact when its origin is a guess.

> **Provenance:** the owner's standing no-fabricated-numbers rule; S6E10's `public_scores.md` appendices are all labelled `LEDGER` for exactly this reason.

### Law 8 - A record is not a gain

A new highest LB inside the ruler changes the record, not the conclusion. Adoption needs the gate: ΔCV at or above the declared bar, folds improved, paired DeLong `z`, and reproduction at `SEED_REPRO`. When the gate binds harder than the board, say so in the row.

> **Provenance:** S6E10's repeated finding — "+0.00003 = 0.5 ruler: a record number, not a proven gain" (V51), and "z binds harder than Δ" (V50).

### Law 9 - Arms come from the template

New version files are written from [`Template/S6E11_TEMPLATE_ModelArm.py`](../Template/S6E11_TEMPLATE_ModelArm.py) or [`Template/S6E11_TEMPLATE_BlendArm.py`](../Template/S6E11_TEMPLATE_BlendArm.py), keep the eight fixed section slots, carry the header's seven fields at twelve lines or fewer, and print only their own measurements in `[5/5] RESULTS`. A clone rewrites the parent's `Axis:` text and is diffed against the parent's config dict before upload.

> **Provenance:** the banner-lies and dict-drift failures above; the slot discipline is what kept S6E10's 55 arms on one numbering when S6E5's and S6E9's drifted.

### Law 10 - This episode is imbalanced, so report both curves

ROC AUC is the scored metric. PR-AUC is reported for every arm, per fold and pooled, because at this positive rate a ROC AUC gain can come entirely from the false-positive side and mean nothing about catching fraud.

> **Provenance:** the source dataset the host names carries 1.70% positives against S6E10's near-balanced target; the host's own starter ideas ask for PR-AUC rather than accuracy.
