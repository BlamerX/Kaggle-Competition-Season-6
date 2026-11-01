# S6E11 Feature Engineering Notes

> **⚠️ RULES** — canon laws live in [`_LAWS.md`](_LAWS.md); the rules below are specific to this ledger.
>
> 1. **A feature row is recorded when the column shipped in an arm that has CV *and* a confirmed LB** (Law 2), or when an instrument measured it without shipping — label which in the Impact cell.
> 2. **DO NOT EDIT** a previous version's rows (Law 6). A re-definition is a new row that names the row it replaces.
> 3. **PREPEND** new discoveries (latest first), the format block staying first and feature rows written below it; entries order by version number where that does not conflict with same-day batch run order.
> 4. **Include:** Feature, Formula, Importance %, Impact, Status — plus the two cells this episode needs: **Leak-safe** and **Prior**.
> 5. **Leak-safe** states how the column avoids leaking: `fit-fold only`, `pooled, no label`, `original-data only`, or `n/a`. A target statistic built on the whole train frame is not a feature, it is the label wearing a hat (Law 3).
> 6. **Prior** carries the smoothing for any rate- or target-encoded column — `smooth=`, the clip, and the fallback for unseen values. At this positive rate an unsmoothed rate encoder on a rare key is noise the model will happily fit.
> 7. **Importance % names its source** (`gain`, `split`, `permutation`, `brute-force drop`, `shap`). A tree's gain and a net's permutation share a column name and nothing else; an unsourced number is not comparable across arms.
> 8. **`n/a` is a legal Feature value** for a rows-only or recipe-only arm — say what the axis was instead.
> 9. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format

The format block stays first; feature entries are written below it.

| Feature | Formula | Importance % (source) | Leak-safe | Prior | Impact | Status |
| ------- | ------- | --------------------- | --------- | ----- | ------ | ------ |

---
