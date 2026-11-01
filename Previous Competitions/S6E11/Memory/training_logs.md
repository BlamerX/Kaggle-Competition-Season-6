# S6E11 Training Logs

> **⚠️ RULES** — canon laws live in [`_LAWS.md`](_LAWS.md); the rules below are specific to this ledger.
>
> 1. **Write the entry after the Public LB score is available** (Law 2) and while the run's stdout is in front of you. This ledger is the run of record; an entry written from memory is a secondhand number (Law 7).
> 2. **DO NOT EDIT or DELETE** a previous entry after submission (Law 6). Corrections are new dated addenda that quote the text they correct.
> 3. **New entries go directly below the Required Format block, newest landed first.** Version-number order is the intent; landing order is the record. A batch that lands out of sequence stays in landing order, annotated — earlier entries are never re-sorted to satisfy the rule.
> 4. **Paste the arm's `[5/5] RESULTS` block verbatim** inside the entry. It is the arm's own measurement, printed by the code the run actually executed; prose summaries drift, stdout does not.
> 5. **Include:** Score (LB / OOF, six decimals, signed gap), Device, Result Δ, the timing breakdown (prep + fit per fold, then total), **every fold's ROC AUC and PR-AUC**, pooled-minus-fold-mean, convergence read-back (`best_iteration` / `n_epochs` / early-stopping state), the fold-1 wall-time canary reading, Strategy in one line, File path, Key Learning, Status.
> 6. **Pooled-minus-fold-mean is mandatory.** It is the quantity that caught pooled-OOF drift masquerading as a gain in the last episode.
> 7. **No sibling scores in the entry body**, except the Δ the declared gate compares against. A relative's number moves; prose does not (Law 7).
> 8. **A retired arm keeps its entry**, marked `RETIRED` with the reason, because it is the control a later arm was measured against.
> 9. **Status:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

## Required Format

The format block stays first; version entries are written below it, newest landed first.

````markdown
### Version [N] ([Description]) - YYYY-MM-DD

**Score**: **X.XXXXXX LB** / X.XXXXXX OOF ROC AUC (Gap: -X.XXXXXX) | OOF PR-AUC X.XXXXXX
**Device**: CPU/GPU
**Result**: **±X.XXXXXX LB**

**Timing:**
| Stage | Time |
|-------|------|
| Prep (per fold) | X.X s |
| Fit (per fold) | X.X s |
| Total | X.X min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | +/- std | Pooled - mean |
|--------|--------|--------|--------|--------|------|---------|---------------|
| 0.XXXXXX | 0.XXXXXX | 0.XXXXXX | 0.XXXXXX | 0.XXXXXX | 0.XXXXXX | 0.XXXXXX | +0.XXXXXX |

**Fold PR-AUCs:** [ ... ]

**Convergence:** best iteration / epoch per fold, early-stopping state
**Canary:** fold-1 wall time vs the declared ceiling

**Strategy:** [One line]
**File:** `Previously trained Files/S6E11_V[N]_[Name].py`

**[5/5] RESULTS as printed:**

```
(paste the arm's own footer block here)
```

**Key Learning:**

> [Takeaway]

**Status:**
````

---
