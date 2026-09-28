# Digest format rules (v2)

Contract for every entry in `writeup_digests/`. A writeup must only ever be read once: after the
digest exists, it must be possible to **replicate or deliberately skip** that solution without
opening the link again — and to answer, from the file alone, **what architecture the winner used**
(flat ensemble? multi-stage stack? a genuinely new model?).

## Files

| Path | Role | Written by |
|---|---|---|
| `kaggle_writeup_links.md` | URL index + entry IDs; defines what must be covered | `tools/build_link_index.py` (generated) |
| `writeup_digests/season6/S6E8.md` … `season3/S3E1.md`, `tps-2021-2022/TPSMONYYYY.md` | **One file per competition, inside its season folder** — this is the deliverable | digest agent |
| `writeup_digests/architecture_matrix.md` | Rank-vs-architecture rollup across all seasons | `tools/architecture_matrix.py` (generated) |

Season folders: `season6/`, `season5/`, `season4/`, `season3/`, `tps-2021-2022/`. The source index
contains no Season 1 / Season 2 Playground writeups, so those seasons have no folder.

Never create side files: no `notes.md`, no per-entry scratch dumps, no duplicate season volumes.
One competition = one file, and that file is the whole record for it.

**This file (`_FORMAT.md`) is the single rules document for the whole library.** All format,
tagging, fetching, tooling and update conventions live here — do not split them into new files.

## Tooling contract (generated + check scripts)

| Path | Role | Rule |
|---|---|---|
| `tools/manifest.json` | source-of-truth link list (comps → writeups → entry IDs) | regenerate with `tools/build_link_index.py` after editing `kaggle_writeup_links.md`; never hand-edit |
| `tools/verify_digests.py` | structural check: entry counts, IDs, template fields, Link lines, season filing | must report **0 defects** before any season is declared done; advisory lines (rule-10 counts, FLAT-with-stages=2) are tracked, not blocking |
| `tools/refetch_check.py` | independent re-fetch: every quoted score/URL in a digest must appear on the live page | run per file after the audit stage; report flags are **triaged, not auto-failures** — known false-positive patterns: digest-author computed deltas in `Gains`, scores inside collapsed comment replies, leading-zero omission (`.96938` ↔ `0.96938`), citation-line writeups slugs, image/nav/profile URLs |
| `tools/architecture_matrix.py` | regenerates `writeup_digests/architecture_matrix.md` from all `**Architecture:**` lines | re-run after ANY digest edit that touches an architecture line; checked-in copy must stay byte-identical to a fresh run |
| `writeup_digests/_AUDIT_LOG.md` | per-file audit ledger; the ONLY record of which files have passed an independent re-fetch | every on-disk digest file must have exactly one row; statuses are `AUDITED` / `AUDIT PENDING` / `WRITTEN`; a row must name the file path on disk (no filename typos) and state its flag findings |

## Fetching a writeup (verified recipe)

WebFetch is blocked on Kaggle. Use the jina reader proxy over Bash. **The correct URL form depends on
the page type, and they are opposite** — check which one your link is before fetching:

| Link contains | Working form | Why |
|---|---|---|
| `/writeups/<slug>` (Season 6, most Season 5) | `/c/` → `/competitions/` | the `/competitions/` form returns the body **plus the author's comment replies** (~45 KB); `/c/` returns ~5 KB body only |
| `/discussion/<id>` (older comps, e.g. S5E1-S5E6, all of S4/S3/TPS) | keep `/c/` | here the `/competitions/` form returns a ~1.6 KB navigation shell while `/c/` returns the full post **with all comments** (~21 KB) |

```bash
# /writeups/ pages
curl -s -m 120 -H "X-Return-Format: markdown" -o /tmp/<id>.txt \
  -w "HTTP %{http_code} bytes %{size_download}\n" \
  "https://r.jina.ai/https://www.kaggle.com/competitions/<comp>/writeups/<slug>"

# /discussion/ pages
curl -s -m 120 -H "X-Return-Format: markdown" -o /tmp/<id>.txt \
  -w "HTTP %{http_code} bytes %{size_download}\n" \
  "https://r.jina.ai/https://www.kaggle.com/c/<comp>/discussion/<id>"
```

Fallback ladder, in order — a writeup is only `UNFETCHED` after all three:

1. The working form for that page type, as above.
2. Same URL with `-H "X-No-Cache: true"`: jina serves empty cached shells (~160-260 bytes, title
   only) for less-visited pages, and this recovers the full body.
2b. Still a shell: add `-H "X-Engine: browser"` (renders the page) and/or `-H "X-Timeout: 60"` —
    several older `/discussion/` threads only resolve with these.
3. The opposite form (`/c/` ↔ `/competitions/`), with and without the headers above.
4. For a stubborn `/discussion/<id>` page that shells on both: `/competitions/<comp>/discussion/<id>`
   **with** `X-No-Cache: true` (this recovers ~15 KB posts where `/c/` gives 145 bytes), and
   `/c/<comp>/discussion/<id>?sort=polls` variants when you need the full comment tree.

Accept only if the file exceeds ~3,000 bytes **and** contains solution prose; a 150-260 byte file is
a cache shell, not a short writeup. Transient HTTP 429 is per-IP and expected when several agents
fetch at once — space requests and retry.

**Escape hatch when all proxy forms fail:** an older `/discussion/` page can refuse every proxy form
(reported on S5E5: 582700, 582848). Open the page with the browser tools instead
(`mcp__browser-use__navigate_page` then `take_snapshot`) and read body, comments and link hrefs from
the accessibility snapshot. Use that before ever writing `UNFETCHED`, and say in `Status` which route
you used. Only a genuinely 404-ing page (see S6E2-12) may be recorded as unreachable.

## Hard rules

1. **One entry per link in the index. No skips**, including unreachable ones (mark `UNFETCHED`).
2. **Entry ID must match the index exactly** (`S6E8-01`). IDs are stable keys.
3. **Numbers beat adjectives.** `+0.0004 AUC`, `-12s RMSE`, `50 folds`, `weight 0.6`. Never
   "helped a bit", "many models", "slightly better".
4. **Never invent a hyperparameter.** Quote only what the author (or an author reply in the comments)
   states. Fill gaps with `not stated`. **Never satisfy a mandatory field by carryover**: a number,
   model name, `novel=` entry, `mod=` tag or commenter credit taken from a *different* writeup in the
   same competition is a fabrication even if it is true elsewhere. When the source is silent, write
   `source silent` — an empty field is correct; a filled one is not.
4b. **Absence claims require a direct check.** "0 comments", "nothing technical in replies", "no
   novel meta-learner", "publishes no score" may only be written after counting that specific page.
   Auditors have found several of these invented, and they propagate into the consensus block.
4c. **Anything read out of an image is labelled as such.** Some authors put their architecture or
   scores only in a figure. Transcribe it, then mark the field `(from figure: <filename/url>)`; if you
   cannot open the image, write `screenshot-only, content unconfirmed` rather than guessing from the
   caption. Unverified diagram readings are the most common source of confident, wrong detail.
5. **Preserve every URL cited inside the writeup body** (notebooks, datasets, discussion, GitHub) in
   `Artifacts`, verbatim. Do not fetch them. Losing a link is a defect.
6. **Never edit `kaggle_tabular_playground_writeups.md`** — it is the user's original file.
7. **`Architecture` is mandatory and machine-parseable** (see taxonomy). It is the field the user
   queries most; leaving it blank fails the file's purpose.
8. Style: one fact per bullet, no prose paragraph over 2 lines, no emojis. Nested bullets and small
   tables are allowed wherever a flat bullet list would mangle a hyperparameter grid or a credit
   table (rule 7's spirit is scannability, not shortness).
9. A missing score is normal: `LB not stated/not stated` is valid; a fabricated score is not.
10. **Every count in the consensus block must name its supporting entry IDs.** "Agreed on (4 of 5)"
    is only valid if the four IDs are listed; auditors have repeatedly found these counts inflated by
    one entry that never stated the practice. If you cannot list the IDs, lower the count.
11. **Write to disk incrementally.** Digest one competition, write its file and consensus block, then
    move to the next. A run that ends early must leave complete files behind, never four unwritten
    ones — an agent once fetched 15 pages and lost all of it by holding the output in memory.

## Architecture taxonomy

Choose one **primary** tag = the structure that produced the *submitted* prediction. Add modifier
tags for everything else present. `stages` counts model-training stages, not submissions.

| Primary tag | Meaning | Distinguishes |
|---|---|---|
| `SINGLE` | One model/architecture submitted (internal seed/fold averaging allowed) | vs everything below |
| `SEED` | Only multi-seed / multi-fold averaging of one architecture | no second family |
| `FLAT` | Blend of level-0 models: weighted / rank / geometric average, hill-climbing, greedy selection — no meta-learner trained on OOF | vs `STACK2` |
| `STACK2` | Two-level stacked generalization: L1 models → OOF matrix → trained meta-learner (LogReg/Ridge/NN/GBM) | one meta layer |
| `STACKN` | Three or more levels (stack of stacks, multi-layer blender) | ≥3 levels |
| `CASCADE` | Sequential: an earlier stage's *output becomes an input feature* of the next stage (residual boosting, init-score, embedding→GBM, denoise→retrain) | data flows forward, not into a blender |
| `PSEUDO` | Semi-supervised loop: predictions on test feed back into training (transductive rounds, self-training) | loop, not DAG |
| `AUTOML` | AutoGluon / H2O / autosklearn-style black-box preset stack | vendor-managed topology |
| `AGENT` | LLM-agent-driven search/hill-climb pipeline that owns model discovery (possibly ending in SINGLE) | how it was built |

Modifiers (comma-separated, any that apply): `public-oof` (community OOF/prediction sets in the
pool) · `copula` · `neg-weights` · `nested-cv-select` · `multi-view` (different feature sets per
branch) · `ovo` (one-vs-one / one-vs-rest decomposition) · `multitarget` · `embed` (NN embeddings
as GBM features) · `gp` (genetic-programming feature search) · `adv` (adversarial validation) ·
`threshold` (metric-optimal cut-off tuned) · `rank-blend` · `pseudo` (pseudo-label loop present).

## Tag rulings (ambiguous cases — apply these, do not re-decide)

These rulings exist because a tag that shifts meaning between competition files makes
`architecture_matrix.md` wrong. Auditors enforce them too.

1. **Hill-climbing / greedy / TPE / quasi-MC weight search = `FLAT`.** Selecting weights is not
   training a meta-learner. `STACK2` requires a model fitted on the OOF matrix (LogReg, Ridge,
   CatBoost, NN, ...).
2. **Class-prior, threshold or calibration fixes applied after `predict_proba` = `Post-processing`**,
   never a stage — even when they are worth +0.05. Promote to a stage only if the correction is
   itself a model trained on OOF predictions.
3. **Public OOF / prediction files in the pool add members, not stages** → `mod=public-oof`, and the
   pool counts go in `l1`.
4. **Seed or fold averaging inside one architecture = `SEED`** (or `SINGLE` if the author treats it
   as one model), and never counts as a level.
5. **NN embeddings consumed as FEATURES by a downstream GBM = `CASCADE` with `mod=embed`** — data
   flows forward, not into a blender. If instead the embedding model's *prediction* is one more OOF
   column into a meta, that is `STACK2`.
6. **AutoGluon as one level-1 member is not `AUTOML`.** Use `AUTOML` only when the submitted stack is
   the vendor-managed one.
7. **LLM/agent-driven search ending in one model:** primary tag follows the **submission**
   (`SINGLE`), with `mod=agent`. `AGENT` is the primary only when the submitted prediction is itself
   the agent system's output.
8. **One-vs-rest / multi-output decomposition:** the recombination counts as a stage, so a OvR set of
   binary stacks under a combiner is `STACK2` with `mod=ovo,multitarget` — not `FLAT`.
9. **Pseudo-label loop that produces the winning model = `PSEUDO` as primary;** a loop that merely
   augments training data for level-1 members is `mod=pseudo`.
10. **Multiple final slots with different architectures:** tag the one that earned the reported rank,
    and describe the other in the expansion bullets.
11. **A submitted deterministic formula with no fitted estimator is still `SINGLE`** — but its first
    expansion bullet must say `no fitted estimator: hand-built decomposition` so the rollup never
    implies a trained model. As soon as any coefficient is learned from data, it stops being this case.
12. **When the topology genuinely cannot be derived, the primary is `not stated (proof: ...)`,**
    naming what the page omits (e.g. "names no estimator; notebook title alone cannot separate FLAT
    from STACK2"). A deleted/unreachable page uses `UNKNOWN (topic deleted - forms tried: ...)`. Both
    are honest values, counted separately from tag defects; guessing `SINGLE` because a title says
    "ensemble" is not.
13. **A submitted deterministic formula with no fitted estimator is `SINGLE`** — but its first
    expansion bullet must say `no fitted estimator: hand-built decomposition` so the rollup never
    implies a trained model. As soon as any coefficient is learned from data, it stops being this case.

`novel=` names an architecture that was *not* a stock sklearn/GBM/MLP choice at the time — e.g.
`TabM`, `RealMLP`, `FT-Transformer`, `Lookup-Transformer`, `NODE`, `TabPFN`, `GANDALF`, `DANet`,
`Trompt`, `MLP+embedding`, `ResNet-tabular`. Write `none` for plain GBM/LR stacks. This is the field
that answers "did anyone use a new architecture, and did it win?".
**`novel=` is a machine-parsed list: bare names separated by commas, or the single word `none`.**
No prose, no parentheticals, no qualifiers in that slot — explain a judgement ("the autoencoder is a
feature extractor, not the submitted model") in the expansion bullets under the tag instead. The
rollup drops non-conforming tokens and lists them as defects.

`topo=` a one-line DAG, `→` between stages, `+` in parallel, `[n]` counts, e.g.
`21 families[556 streams] → rank+logit → LogReg(C=0.01..3) ×6 → avg ranks` .

## Entry template

Every field mandatory; empty → `not stated`.

```markdown
### <ID> · <rank> · <author handle> · LB <public>/<private> · CV <cv score>

- **Link:** <url>
- **Status:** FETCHED (<attempts/notes>) | UNFETCHED (<reason>)
- **TL;DR:** <2-3 bullets, one line each>
- **Architecture:** <PRIMARY> · stages=<n> · l1=<#models/>'n/s' · l2=<meta-learner or none> ·
  novel=<arch or none> · mod=<tags or none> · topo=<one-line DAG>
  - <1-3 bullets expanding the tag: what each stage consumes, whether the winner was the stack or a
    single member, and how the ensemble stage differs from a plain average>
- **Setup:** <rows/cols, split sizes, submission slots, structural quirk exploited>
- **Features:** <exact engineered feature names + formulas; encodings; GP/generated features;
  feature-source kernels; columns dropped and why>
- **Models:** <per family: library + author-published hyperparameters + seeds + fold count>
- **CV:** <splitter, #folds, #repeats, stratification, seeds; CV-vs-LB gap; author's trust verdict>
- **Ensembling:** <blend/stack mechanics with weights, meta-learner params, hill-climb/ridge details,
  shared-OOF usage, number of models actually averaged>
- **Post-processing:** <thresholds, calibration, clipping, rank-gauss, rounding, label tricks>
- **Gains:** <ranked change → Δmetric, biggest first>
- **Failed:** <dead ends with stated reason — highest-value anti-knowledge in the file>
- **Comments:** <technical content that only exists in the author's comment replies (ablations,
  withheld FE, clarifications); or `nothing technical`>
- **Compute:** <wall-clock, GPU/CPU, RAM, Kaggle limits hit and workarounds, cost>
- **Artifacts:** <URLs cited by the author, verbatim; or `none cited`>
- **Lesson:** <one transferable sentence>
```

## Competition section wrapper

`writeup_digests/<season>/<COMP>.md` opens with the section header and closes with a synthesis block.
The synthesis is what makes cross-reads unnecessary: it aggregates agreement instead of repeating it
per entry.

**Unknown metric:** three competitions in the index (S3E1, S3E2, S3E3) carry a placeholder metric. Read
the real one off the writeup page header or body, print it in the section header as
`- Metric: <name> (recovered from writeup pages; index had no metric)`, and use it for every delta
sign in that file. Never infer it from the problem statement.

```markdown
## <COMP> — <full competition title>
<title line from kaggle_writeup_links.md: task | metric | problem>
- Kaggle display title: "<as shown>"
- Competition: <url>
- Writeups covered: <n> of <n>
- Score ladder: <rank> `<pub>/<priv>` … (as reported by the authors)

### <ID> · …   (entries, best rank first)

## <COMP> — consensus recipe
- **Architecture distribution:** <rank → primary tag, one line per entry, so the winner's topology is
  visible at a glance; end with `winner topology: <tag>`>
- **New architectures at the board:** <which `novel=` models appeared, at which ranks, best score
  each; explicitly say "none" if every entrant was GBM/LR>
- **Agreed on (<count> of <n>):** <FE primitives, CV scheme, model families, blend type shared by
  ≥half the entries, with which ranks>
- **Divergences:** <where top ranks disagreed with mid ranks, and the resulting score gap>
- **Highest-leverage single trick:** <one line + source entry>
- **Nothing worked:** <things multiple authors tried and reported useless>
```

## Definition of done for a competition file

- [ ] entry count == writeup count for that competition in the index
- [ ] every entry has all template fields, including a parseable `**Architecture:**` line
- [ ] every URL in the source writeup appears in the competition file
- [ ] comment replies mined (fetch used `/competitions/` form)
- [ ] no `<placeholder>` survives
- [ ] `Architecture distribution` + `New architectures` blocks filled

## Audit stage (separate agent, mandatory)

A second agent re-verifies each competition file after it is written. The auditor is adversarial: its
job is to find where the writer invented, dropped, or mis-transcribed something — not to approve.

**Ordering constraint:** an auditor must only start after its file's writer has finished. A concurrent
writer will overwrite audited content, and a raced file can contain a stale entry beside a corrected
one. If you suspect a race (entries that contradict each other, a consensus block that references an
older tag), say so in your report instead of editing around it.

1. Re-fetch **every** link with the `/competitions/` recipe; do not trust the writer's quotes.
2. For each entry, verify: rank + author handle, every score quoted in the header and `Gains`,
   hyperparameter values, the `Architecture` primary tag against the taxonomy, and that each
   `Artifacts` URL really appears in the source.
3. Verify entry count against `kaggle_writeup_links.md` for that competition; verify each entry ID
   matches its URL by rank.
4. **Fix defects in place** — same file, do not append a review report, do not create a notes file.
5. Append nothing to a correct file.

Auditor output is a short report, not a file: per competition, `n entries audited, k fixed,
list of material corrections` plus any entry it could not verify and why.

## Update workflow (future seasons / new episodes)

To extend the library with a new season or episode:

1. **Extend the source index** (`kaggle_writeup_links.md`, hand-maintained): add the competition
   block with its writeup links + ranks, matching the existing per-competition format.
2. **Regenerate** `tools/manifest.json` and the link index side via `tools/build_link_index.py`.
3. **Write digests** with one agent per competition using the verified fetch recipe + entry template
   above; write to disk incrementally (hard rule 11). New season → new folder `season<n>/`, or
   `tps-YYYY-YYYY/` for a new TPS year range; entry IDs stay `<COMP>-<rank2d>`.
4. **Independent audit**: dispatch one re-fetching auditor per file, only after its writer
   finishes (ordering constraint above); auditors fix defects in place.
5. **Close the loop, in this order**:
   - `python tools/verify_digests.py` → 0 defects
   - `python tools/refetch_check.py writeup_digests/<file>.md` per new/changed file → triage flags
   - `python tools/architecture_matrix.py` → regenerate the matrix
   - add one `AUDITED` row per file to `_AUDIT_LOG.md` (fix the section header counts to match)

Invariants that must hold after any update: manifest writeup count == digest `**Link:**` count;
every on-disk digest file has exactly one audit-log row; `architecture_matrix.md` is byte-identical
to a fresh regeneration; no `AUDIT PENDING` or `WRITTEN` rows remain.

## Known permanent gaps (mirror)

Canonical list lives in `_AUDIT_LOG.md` ("Known permanent gaps"). If a new gap is found (deleted
topic, not-available comp), record it there, not here.
