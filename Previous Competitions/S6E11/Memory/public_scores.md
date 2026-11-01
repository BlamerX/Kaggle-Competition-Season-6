# S6E11 Public Leaderboard Scores

> **⚠️ RULES** — canon laws live in [`_LAWS.md`](_LAWS.md); the rules below are specific to this ledger.
>
> 1. **A row exists only when the LB score is confirmed from Kaggle *and* the OOF score is in hand** (Law 2). No score, no row; the attempt belongs in `daily_log.md`.
> 2. **Score cells are frozen. The Notes cell may gain a dated annotation**, never a substitution: `Correction (YYYY-MM-DD): …` below the original text, which stays. Numbers are history, readings are argument (Law 6).
> 3. **INSERT each row at its LB rank inside its category** (highest on top). Chronology lives in `daily_log.md`; this table's order is a ranking, so "newest first" and "highest first" no longer contradict each other.
> 4. **Include:** OOF ROC AUC, OOF PR-AUC, LB, Gap, Training Time, and both artifact names.
> 5. **Gap = LB − OOF, signed, six decimals.** Five decimals rounds the quantity the gate is judged on; the ruler here is the seed tick the first reproduction arm measures.
> 6. **PR-AUC is not optional** (Law 10). A ROC AUC gain with a flat PR-AUC is a false-positive-side improvement, and the Notes cell says which side moved.
> 7. **CATEGORIZE** per the headers below: two-stage models, multiseed arms, and resampling / class-weight / undersampling arms are all **Single Models**; a **Combiner** is an arm whose inputs are saved vectors (Law 3's composition class).
> 8. **The Time cell names the device**; a local combiner reads `CPU (local)`, a Kaggle fit reads `XX min (GPU)`.
> 9. **TOP 5 holds exactly the five highest LBs** (any category); each also appears in its category table — the view is not a copy.
> 10. **Status:** 🏆 Best | ✅ Good | ⚠️ Partial | ❌ Failed/Overfit

---

## 📝 Score Logging Format

The format block stays first; score entries are written below it.

| Version | Date  | LB Score | OOF ROC AUC | OOF PR-AUC | Gap    | Time   | Script    | OOF File  | Sub File  | Notes |
| ------- | ----- | -------- | ----------- | ---------- | ------ | ------ | --------- | --------- | --------- | ----- |
| V#      | MM-DD | X.XXXXXX | X.XXXXXX    | X.XXXXXX   | -0.XXXXXX | XX min | `file.py` | `v#_oof.csv` | `v#_sub.csv` | Notes |

---

### Leaderboard Scores Top 5

The five highest LBs of every submission; each also appears in its category table below (Top 5 is a view, not a copy).

| Version | Date | LB Score | OOF ROC AUC | OOF PR-AUC | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | ----------- | ---------- | --- | ---- | ------ | -------- | -------- | ----- |

---

## Single Models (incl. two-stage, multiseed, resampling / class-weight arms)

| Version | Date | LB Score | OOF ROC AUC | OOF PR-AUC | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | ----------- | ---------- | --- | ---- | ------ | -------- | -------- | ----- |

---

## Ensembles (combiners over saved OOF vectors)

| Version | Date | LB Score | OOF ROC AUC | OOF PR-AUC | Gap | Time | Script | OOF File | Sub File | Notes |
| ------- | ---- | -------- | ----------- | ---------- | --- | ---- | ------ | -------- | -------- | ----- |
