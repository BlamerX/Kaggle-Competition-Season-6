# Audit ledger

Which competition files have been re-verified by an **independent auditor agent** that re-fetched every
source page (`_FORMAT.md` → "Audit stage"). Statuses here are driven only by auditor reports actually
received; `WRITTEN` means one agent pass and provisional deltas/prose.

| Status | Meaning |
|---|---|
| AUDITED | auditor re-fetched every source, corrected defects in place, reported back |
| AUDIT PENDING | auditor dispatched, report not yet received |
| WRITTEN | entries + architecture tags present, no independent verification yet |

Architecture primary tags have proven the most durable field; deltas, attributions and consensus prose
the least. Corrections cited below are auditor-applied.

## Season 6 - 8/8 AUDITED

| File | Entries | Auditor findings |
|---|---|---|
| season6/S6E1.md | 10 | 9 fixes: fabricated CV splitter, invented author URL, RMSE sign error |
| season6/S6E2.md | 15 | 12 entries corrected; 12th place confirmed deleted on Kaggle; `stages` 2→1 |
| season6/S6E3.md | 11 | 8 entries; "0 comments" claim on an 8-comment page; 149/199/167 count conflict resolved |
| season6/S6E4.md | 9 | 18 fixes: fabricated +0.00043 gain, three wrong subtractions |
| season6/S6E5.md | 9 | 8 entries: wholly fabricated Comments field; `novel=` padding by carryover |
| season6/S6E6.md | 14 | 11 entries: 12th retagged `STACKN`→`STACK2`; TabPFN-3-as-meta confirmed |
| season6/S6E7.md | 5 | 6 fixes: consensus count 4-of-5 → 2-of-5 |
| season6/S6E8.md | 5 | 14 fixes: 1st's published LB 0.97174 had been recorded as unstated; `novel=` de-padded |

## Season 5 - 12/12 AUDITED

| File | Entries | Auditor findings |
|---|---|---|
| season5/S5E1.md | 7 | 13 fixes: "0 Comments" on a 38-comment page; cdeotte pseudo-label walkthrough recovered |
| season5/S5E2.md | 3 | AUDITED - part of the 5-file small-comp batch: fabricated data-generation mechanism |
| season5/S5E3.md | 4 | AUDITED - part of small batch: invented +0.00114 gain |
| season5/S5E4.md | 6 | 18 fixes: "-1.19" gain was -0.19; private LB labelled as CV |
| season5/S5E5.md | 14 | 14 fixes: false browser-only claim; `novel=none` where author named ResMLP |
| season5/S5E6.md | 11 | 13 fixes: 4th/5th rank contradiction adjudicated; 21st retagged `STACK2`→`STACKN` |
| season5/S5E7.md | 1 | AUDITED - part of small batch: clean; 42nd-vs-Top#3 rank conflict documented |
| season5/S5E8.md | 14 | ~25 fixes: fabricated 2-comment thread; HC-vs-Ridge direction reversed |
| season5/S5E9.md | 1 | AUDITED - part of small batch: `PSEUDO` primary upheld |
| season5/S5E10.md | 7 | 12 fixes: "0 Comments" on an 84-comment page; 04 settled at `STACK2` |
| season5/S5E11.md | 7 | ~25 fixes: 1st's unpublished score recovered from 5 collapsed replies |
| season5/S5E12.md | 3 | AUDITED - part of small batch: consensus claim reversed; model count 4→5 |

## Season 4 - 12/12 AUDITED

| File | Entries | Status |
|---|---|---|
| season4/S4E10.md | 5 | AUDITED - `AUTOML` survived ruling 6; audited twice due to a writer/auditor race |
| season4/S4E11.md | 4 | AUDITED - the library's only `AUTOML` first place |
| season4/S4E12.md | 3 | AUDITED - winner is `SINGLE`, 611-feature count confirmed |
| season4/S4E1.md | 3 | AUDITED - 6 fixes: CV delta sign wrong (-0.0015 → +0.0026), paddykb credit 3-of-3 → 2, handle typo |
| season4/S4E2.md | 4 | AUDITED - 5 fixes: two rule-10 consensus counts lowered to 3-of-4 |
| season4/S4E3.md | 2 | AUDITED - threshold-absence claim **verified true by grep**; `STACK2 mod=multitarget` stands |
| season4/S4E4.md | 6 | AUDITED - winner `FLAT` (49 models, Nelder-Mead on OOF, negative weights) and 2nd's `AUTOML` both upheld |
| season4/S4E5.md | 3 | AUDITED - `STACKN` upheld (LR fitted per subset); invented original-data gain → `source silent`; R2 signs clean |
| season4/S4E6.md | 2 | AUDITED - **worst defect found in the library**: "0 Comments" claim was fabricated (page renders 23) and had spread into consensus |
| season4/S4E7.md | 6 | AUDITED - winner `STACKN`/stages=3 confirmed (78 stage-1+2 columns → one XGBoost); 12 fixes, 5 dropped URLs restored |
| season4/S4E8.md | 6 | AUDITED - 7 url flags (4 false positives: trailing-URL normalization in collapsed comments / cache-shell artifacts); 0 score flags; architecture tags (`STACKN`, `AUTOML`, `FLAT`, `STACK2`) upheld |
| season4/S4E9.md | 5 | AUDITED - 2 url flags (trailing-fragment normalization in proxy render; URLs verified live); 0 score flags; all architecture tags (`CASCADE`, `AUTOML`, `STACK2`, `FLAT`) upheld |

## Season 3 - 24/24 AUDITED

| File | Entries | Status |
|---|---|---|
| season3/S3E1.md, S3E2.md, S3E4.md | 9 | AUDITED - S3E1 **RMSE** and S3E2 **ROC AUC** recoveries verified verbatim against Evaluation pages; `AUTOML` (AutoGluon submitted as-is) and `SEED` (50 identical-config XGBoosts) both survive; two more fabricated "0 Comments" claims caught |
| season3/S3E3.md | 6 | AUDITED - metric ROC AUC **recovered** with quoted evidence; 12th demoted to `not stated` |
| season3/S3E5.md | 8 | AUDITED - 4/4 `mod=threshold` attributions held; winner `SINGLE` survives |
| season3/S3E6.md | 5 | AUDITED - "no entry publishes an RMSE" proved **half wrong**: 43rd's chart image is an RMSE table, transcribed; `AUTOML` and `SINGLE` upheld |
| season3/S3E7.md | 6 | AUDITED - five `FLAT` tags stand; +.014 relabelled LB, not public |
| season3/S3E8.md | 5 | AUDITED - `STACK2` and `AUTOML` confirmed |
| season3/S3E9.md | 2 | AUDITED - `mod=gp` verified; 23rd-vs-12th rank conflict documented |
| season3/S3E10.md | 8 | AUDITED - two `CASCADE` tags retagged `FLAT` under ruling 5 |
| season3/S3E11.md | 4 | AUDITED - two "no CV published" claims overturned by chart images; S3E6→S3E11 carryover text removed |
| season3/S3E12.md | 4 | AUDITED - 0 score flags, 0 url flags; handles + scores verified |
| season3/S3E13.md | 9 | AUDITED - 8 score flags are all false positives (digest-author-computed deltas in Gains, not Kaggle page quotes); 0 url flags |
| season3/S3E14.md | 3 | AUDITED - 2 url flags are commenter-cited notebook URLs (not in proxy-rendered body but verified live); 0 score flags |
| season3/S3E15.md | 8 | AUDITED - all three `CASCADE` (trained imputer → predictor) verified; screenshot claim held for 3 of 6, not 6 |
| season3/S3E16.md | 3 | AUDITED - 0 score flags; 22 url flags are author-profile/handle URLs not in proxy-rendered body (known limitation); architecture tags upheld (`not stated` / `SINGLE` / `FLAT` per ruling 12) |
| season3/S3E17.md | 4 | AUDITED - 0 score flags, 0 url flags; handles + scores verified |
| season3/S3E19.md | 10 | AUDITED - **1st and 4th topics confirmed deleted on Kaggle** after every ladder rung + browser; their `SINGLE` tags removed, winner topology now "not recoverable" |
| season3/S3E20.md | 2 | AUDITED - 1 score flag is false positive (digest-author delta in Gains, not Kaggle page quote); 0 url flags |
| season3/S3E21.md | 2 | AUDITED - 0 score flags, 0 url flags; handles + scores verified |
| season3/S3E22.md | 1 | AUDITED - `FLAT` did **not** survive: it rested on a notebook URL slug → `not stated` |
| season3/S3E23.md | 1 | AUDITED - figures verified, clean |
| season3/S3E24.md | 4 | AUDITED - **clean**: 4/4 `FLAT` sweep verified page by page, nothing fitted on an OOF matrix |
| season3/S3E26.md | 3 | AUDITED - 1 score flag is false positive (~0.401 lives in collapsed comment replies); 0 url flags; 39th's "CatBoost final model" quote attributed to correct page

## TPS 2021-2022 - 22/22 AUDITED (1 comp not available: TPSOCT22, no writeups, no on-disk file)

| File | Entries | Status |
|---|---|---|
| tps-2021-2022/TPSJAN21.md | 5 | AUDITED - 2 score flags (~0.698, ~0.702) are unlabelled CV/LB gap commentary (false positives); 0 url flags |
| tps-2021-2022/TPSFEB21.md | 7 | AUDITED - 3 score flags (0.84148, 0.84248, 0.84228) live in collapsed comment replies; verified via browser render; all correct; 1 arxiv url not in proxy body |
| tps-2021-2022/TPSMAR21.md | 5 | AUDITED - 0 flags |
| tps-2021-2022/TPSAPR21.md | 1 | AUDITED - 0 flags |
| tps-2021-2022/TPSMAY21.md | 3 | AUDITED (re-fetch 2026-09-28) - 1 url flag is a writeups-slug citation link not in proxy body; 0 score flags; `STACK2`/l1=5 families + Ridge-in-CalibratedClassifierCV upheld |
| tps-2021-2022/TPSJUN21.md | 2 | AUDITED - 1 url flag (citation-line writeups slug not in proxy body); 0 score flags |
| tps-2021-2022/TPSJUL21.md | 1 | AUDITED - 0 flags |
| tps-2021-2022/TPSAUG21.md | 1 | AUDITED - 0 flags |
| tps-2021-2022/TPSSEP21.md | 8 | AUDITED - both `STACKN` tags confirmed genuine three-level stacks from author replies; 1 commenter-cited url not in proxy body |
| tps-2021-2022/TPSOCT21.md | 4 | AUDITED - 2 url flags are malformed/partial URLs extracted from page text (`/input`, trailing-colon); 0 score flags |
| tps-2021-2022/TPSNOV21.md | 4 | AUDITED (re-fetch 2026-09-28) - 0 url flags, 0 score flags; rank-claim disagreement (3rd congratulated "on the win" vs index 5th's "+141") documented in-file |
| tps-2021-2022/TPSDEC21.md | 1 | AUDITED - 2 url flags are profile URLs not in proxy body |
| tps-2021-2022/TPSJAN22.md | 4 | AUDITED - 0 flags; handles + scores verified |
| tps-2021-2022/TPSFEB22.md | 2 | AUDITED - 6 url flags are `/competitions/...` navigation URLs not in proxy body (known limitation); 0 score flags |
| tps-2021-2022/TPSMAR22.md | 2 | AUDITED - 2 url flags are trailing-URL normalization issues; 0 score flags |
| tps-2021-2022/TPSAPR22.md | 6 | AUDITED - writer's brief named wrong competition; corrected in-file; score flags are false positives (digest-author deltas) |
| tps-2021-2022/TPSMAY22.md | 3 | AUDITED (re-fetch 2026-09-28) - metric confirmed **ROC AUC binary classification** (not clustering/ARI, which is TPSJUL22); 4 url flags are competition header/thumbnail images, a malformed `kaggle.com/cv13j0` path and a profile link (none in proxy body); 0 score flags |
| tps-2021-2022/TPSJUN22.md | 6 | AUDITED - 4 score flags are digest-author deltas in Gains (false positives); 1 trailing-fragment url flag |
| tps-2021-2022/TPSJUL22.md | 1 | AUDITED - 1 url flag is citation-line writeups slug not in proxy body; 0 score flags |
| tps-2021-2022/TPSAUG22.md | 3 | AUDITED - 2 url flags are commenter-cited notebook/profile URLs not in proxy body; 0 score flags |
| tps-2021-2022/TPSSEP22.md | 2 | AUDITED - 0 flags |
| tps-2021-2022/TPSNOV22.md | 3 | AUDITED (re-fetch 2026-09-28) - 2 url flags are author-cited notebook URLs with trailing-`notebook"` quote artifacts (links not in proxy body); 0 score flags; index RMSE vs page AUC metric conflict resolved in-file (ROC AUC, higher better) |
| tps-2021-2022/TPSOCT22.md | 0 | NOT AVAILABLE - source index lists "Not available" (no writeup)

## Known permanent gaps (not fixable by re-running)

- `S6E2-12` - writeup deleted on Kaggle (404 on all forms).
- `S3E19-01`, `S3E19-04` - topics deleted; only the 21 and 8 surviving comments carry content.
- `S3E25`, `S3E18`, `TPSOCT22` - the source index lists no writeup for these ("Not available").
