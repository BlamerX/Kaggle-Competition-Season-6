## TPSAUG22 — Tabular Playground Series - Aug 2022
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Aug 2022"
- Competition: https://www.kaggle.com/c/tabular-playground-series-aug-2022
- Writeups covered: 3 of 3
- Score ladder: 9th `not stated/not stated` · 14th `0.58675/0.59090` as submitted (`0.58741/0.59110` after the bug fix the
  author says cost him 8 ranks) · 17th `not stated/0.59087` (private only; his two other published numbers are 0.59109
  private unsubmitted and 0.59053 for "the two final submissions")
- Absolute AUCs sit near 0.59 because of the train/test modality shift in `product_code`; 14th is the only entry here that
  publishes public and private for every member. A commenter on 17th's page (Katarina 00, 426th, not an author) frames the
  board: "frustrated that nobody hit that 0.60".

### TPSAUG22-09 · 9th · takanashihumbert (Joseph Zhou) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-aug-2022/discussion/349297
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 4.0 KB; all 5 comments rendered inline; the
  discussion id resolves to the writeup page `sawaimilert-tps-aug22-9th-solution`)
- **TL;DR:** One sentence of method: 4 logistic regressions, each on a different feature set, blended — "to avoid
  overfitting". He calls the result luck ("Really a shake, lucky") and publishes no score.
- **Architecture:** FLAT · stages=1 · l1=4 LogisticRegression models (one per feature set) · l2=none · novel=none ·
  mod=multi-view ·
  topo=4 × LogisticRegression, each on its own feature subset → blend of the 4 predictions [mechanic not stated]
  - The whole architecture is the **multi-view** idea: the members are the same estimator (LR), differentiated only by
    which columns they see; he states the purpose as avoiding overfitting, not as gaining diversity for its own sake.
  - `FLAT` not `STACK2`: no meta-learner is mentioned; only "I just use 4 LR models". Whether the 4 are averaged or
    weighted is not stated on the page.
  - `l1=4` is taken from his own count; the individual feature sets are not enumerated anywhere on the page or in comments.
- **Setup:** rows/cols, split sizes, fold count, submission slots: not stated.
- Structural quirk exploited: not stated on this page (the `product_code` shift is what the other two entries build on).
- **Features:** not stated — no column names, no encodings, no formulas.
- His stated position on FE, verbatim from a comment reply: "Actually I have no idea how to do FE, so I follow 'the simpler
  the better'."
- A commenter (Naruke, 28th) refers to the author's separate notebook on **adversarial validation on test data** as helpful;
  whether that AV work feeds the 9th-place submission is **not stated**, so no `adv` modifier is claimed here.
- **Models:** 4 × Logistic Regression (scikit-learn assumed, library not named); feature sets, regularization values,
  solver, C, seeds, fold counts: not stated.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV number and no LB number published.
- CV-vs-LB gap: not computable.
- Author's trust verdict: not stated; he attributes the placement to the shakeup instead of to validation ("Really a shake,
  lucky").
- **Ensembling:** blend of the 4 LR predictions; weights, averaging mechanic, and whether OOF or test predictions were
  blended: not stated.
- **Post-processing:** not stated. (Related: a commenter asks why transforming `loading` with `np.log` pushed his own AUC
  below 0.5 — see Comments; the author never answers it.)
- **Gains:** none quantified — the page contains no delta for anything.
- **Failed:** none named; the page states no dead ends either (counted body + all 5 comments).
- **Comments:** 5 comments (counted). Technical content:
  - Edvard (420th) asks the difference between `np.log` and `np.log1p`, reporting that log-transforming `loading` drove his
    own AUC below 0.5; the author replies only with the "no idea how to do FE / simpler the better" statement above, so the
    question is left unanswered on the page.
  - Naruke (28th) credits the author's adversarial-validation-on-test-data notebook.
  - Jose Cáliz (57th) twice: that he followed the author's competition-thread advice, and "it's hard to do FE when not
    knowing the nature of the data".
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/takanashihumbert/tps-aug22-9th-solution/notebook (his solution notebook, the only link in
    the body)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/writeups/sawaimilert-tps-aug22-9th-solution
    (citation line)
  - https://www.kaggle.com/competitions/33108/images/thumbnail
  - https://www.kaggle.com/takanashihumbert · https://www.kaggle.com/vishnu123 · https://www.kaggle.com/edvardstepanuk ·
    https://www.kaggle.com/jcaliz
- **Lesson:** Four logistic regressions on disjoint feature views is a complete top-10 submission here — but a solution with
  no numbers published is a pointer, not something to replicate score-for-score.

### TPSAUG22-14 · 14th (could have been 6th) · medali1992 + nourhadrich (Med Ali Bouchhioua, nour hadrich) · LB 0.58675/0.59090 (0.58741/0.59110 after fix) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-aug-2022/discussion/349810
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 5.8 KB, complete body; page reports
  "## 0 Comments" and renders none)
- **TL;DR:** 3-train-vs-2-validate `product_code` folds (AmbrosM's scheme) keep him honest about the leaderboard; 7 member
  models blended; a `model.predict()` vs `model.predict_proba()` bug on the CatBoost member depressed the blend and cost
  roughly 8 ranks.
- **Architecture:** FLAT · stages=1 · l1=7 members (TabNet, Logistic Regression, LightGBM, NN, XGBoost, CatBoost, KNN) ·
  l2=none · novel=TabNet · mod=none ·
  topo=10 × 3-vs-2 product_code folds → 7 families each producing a prediction set → blending notebook
  [weights/mechanic not stated] → 0.58675/0.59090, corrected to 0.58741/0.59110
  - Single stage: the blending notebook consumes the 7 member predictions directly; no second model is trained on an OOF
    matrix, so this is `FLAT` rather than `STACK2`. The blend weights are not published.
  - The 10-fold 3-vs-2 structure is a **validation** scheme, not a level: each member is fitted 10 times and the fold
    predictions are aggregated into one member column, which rule 4 forbids counting as a level.
  - `novel=TabNet`: the author calls TabNet his best single model (0.58908 public / 0.59098 private), the only non-stock
    family on this board.
- **Setup:** rows/cols and submission slots not stated. Train has 5 product codes (A–E in his fold dict); test carries
  **different modalities** — his words: "the difference in the modalities between the measurements in the train set and
  those in the test set. This made us ensure to use these new modalites (test set) in the validation process."
- Fold construction (verbatim from his code block): 10 folds, each training on 3 codes and validating on 2 —
  `Fold 1 [['C','D','E'],['A','B']]`, `Fold 2 [['B','D','E'],['A','C']]`, `Fold 3 [['B','C','E'],['A','D']]`,
  `Fold 4 [['B','C','D'],['A','E']]`, `Fold 5 [['A','D','E'],['B','C']]`, `Fold 6 [['A','C','E'],['B','D']]`,
  `Fold 7 [['A','C','D'],['B','E']]`, `Fold 8 [['A','B','E'],['C','D']]`, `Fold 9 [['A','B','D'],['C','E']]`,
  `Fold 10 [['A','B','C'],['D','E']]` (i.e. all 10 two-code held-out pairs).
- **Features:** no engineered features are named in this entry.
- Encoding used: `product_code` membership is the split key, selected with `df_train['product_code'].isin(...)`.
- The exact feature list is only given as `features` in the code block; column names not stated.
- **Models:** 7 families, each with a published notebook and both board scores (author's table, verbatim):
  - Tabnet 0.58908 / 0.59098 (his stated best single model)
  - Logistic Regression 0.58974 / 0.59018 (best public of the seven)
  - LightGBM 0.58381 / 0.58831
  - Neural network 0.58828 / 0.59064
  - XGBoost 0.58288 / 0.58898
  - CATBoost 0.58484 / 0.58964
  - KNN 0.58121 / 0.58995
  - Ensembling (submitted) 0.58675 / 0.59090
  - Ensembling After (same notebook, bug fixed) 0.58741 / 0.59110
- Hyperparameters, seeds, library versions for every family: not stated (only notebooks linked, not fetched).
- **CV:** the 3-vs-2 scheme above, 10 folds, deterministic (code-level, no shuffling, no repeats, no seeds).
- He credits the scheme as decisive, verbatim: "The 3 vs 2 cross validation scheme by AmbrosM was **the key** to not
  **overfit** the leaderboard."
- No CV score is published, so the CV-vs-LB gap is not computable from this page.
- **Ensembling:** one blending notebook over the 7 member prediction sets; weights and mechanic (average vs rank vs
  hill-climb) not stated.
- Measured ensemble effect: the submitted blend (0.59090 private) sits between his best member TabNet (0.59098) and the
  public-best LR (0.59018) — i.e. as submitted the blend did **not** beat his best single model on private; after the fix it
  did (0.59110 vs 0.59098).
- **Post-processing:** none stated beyond the bug correction itself (`predict_proba` restored).
- **Gains:** (+0.00066 public, +0.00021 private) fixing the CatBoost member's `model.predict()` → `model.predict_proba()`
  in the ensemble: 0.58675/0.59090 → 0.58741/0.59110.
- (+0.00012 private) blend over best single member after the fix (0.59098 TabNet → 0.59110).
- Rank consequence stated by the author in the title: the corrected blend "could have been 6 th" instead of 14th.
- (structural, unquantified) adopting the 3-vs-2 folds is what he names as the thing that stopped leaderboard overfitting.
- **Failed:** The one named failure is the ensemble bug: `model.predict()` instead of `model.predict_proba()` for CatBoost
  inside the blending model — "a **huge mistake** in one of the Ensembling models … that affected the whole score and our
  rank".
- Second-order dead end visible in his own table: XGBoost (0.58898 private) and LightGBM (0.58831) are the two weakest
  members and still below the KNN (0.58995) and LR (0.59018) members — the GBMs did not carry this board.
- **Comments:** nothing technical — the page itself reports "## 0 Comments" (counted on the fetched page, not inferred).
  All of the entry's quantitative content therefore lives in the body and its notebook table.
- **Compute:** not stated.
- **Artifacts:** (verbatim from the body and its table; not fetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/341896 (AmbrosM's 3-vs-2 CV post)
  - https://www.kaggle.com/code/purist1024/principled-3-vs-2-cv-splitting-on-product-code (sklearn-style implementation)
  - https://www.kaggle.com/code/medali1992/aug-tps-tabnetclassifier
  - https://www.kaggle.com/code/medali1992/tps-aug-logistic-regression
  - https://www.kaggle.com/code/medali1992/tps-aug-lightgbm
  - https://www.kaggle.com/code/nourhadrich/tps-aug-neural-network
  - https://www.kaggle.com/code/nourhadrich/tps-aug-xgboost
  - https://www.kaggle.com/code/nourhadrich/tps-aug-catboost
  - https://www.kaggle.com/code/nourhadrich/tps-aug-knn
  - https://www.kaggle.com/code/nourhadrich/tps-aug-ensembling (linked twice: "Ensembling" and "Ensembling After")
  - https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8877777%2F03f41d787de7ebb74b6c795c02029b46%2Fembed.PNG?generation=1662145162975604&alt=media
    (the rank-vs-score figure)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/writeups/black-list-14-th-solution-that-could-have-been-6-t
    (citation line) · https://www.kaggle.com/competitions/33108/images/thumbnail
  - https://www.kaggle.com/medali1992 · https://www.kaggle.com/nourhadrich · https://www.kaggle.com/purist1024
- **Lesson:** On a shift-heavy board, your split must mirror the shift (3 codes train / 2 validate) — and audit the blend's
  member calls, because one `predict()` vs `predict_proba()` slip cost 8 ranks, more than any model choice here.

### TPSAUG22-17 · 17th · rsizem2 (Robert Sizemore) + chrismiddlebrooks · LB not stated/0.59087 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-aug-2022/discussion/349541
- **Status:** FETCHED after 3 attempts. Step 1 (`/c/` + `X-Return-Format: markdown`) returned a 407-byte cookie shell with
  "This page maybe not yet fully loaded"; step 2b (`X-Timeout: 60`) on the same `/c/` URL returned the full 7.8 KB post with
  all 6 comments. A parallel `/competitions/…/writeups/ahmed-abdelmagid-17-th-solution` guess returned 677 bytes and was
  discarded (the real slug on the page is `kagglesrl-17th-place-solution`).
- **TL;DR:** Aggressive subtraction: 4 hand-made features + 6 selected columns, everything else discarded; WOE-encoded
  `attribute_0`; log `loading`; RobustScaler; then one LR (and one gblinear XGBoost) averaged over the 10 3-vs-2 folds.
- He says not spending time is what saved him: "I imagine this saved us from overfitting to the public leaderboard".
- **Architecture:** SEED · stages=1 · l1=1 LogisticRegression family, prediction averaged over the 10 3-vs-2 folds ·
  l2=none · novel=none · mod=none ·
  topo=drop all but 6 base cols (+4 engineered) → WOE on attribute_0, log(loading), RobustScaler → LR fit once per 3-vs-2
  fold [10] → average the 10 fold predictions → 0.59087 private
  - Rule 4: the 10 fold models are one architecture averaged, so this is `SEED`/`SINGLE`, not a level; the author treats the
    split-average as the model ("For the final prediction, we averaged our predictions over each split").
  - A second family, XGBoost with the **linear booster** (`booster=gblinear`), was also selected as a winner-class model;
    its best version scored 0.59109 private and was **not submitted**, so the two families were never blended — the
    submission is single-family.
  - The author's own account of which family occupied the two final slots is not self-consistent (see Comments): 0.59087
    LR is "what we were ranked with", while "the one I chose for the two final submissions" is 0.59053. Per ruling 10 the
    tag follows the ranked score (LR); if the ranked slot was actually the gblinear XGBoost the primary tag is unchanged,
    because it is still one family fold-averaged.
- **Setup:** rows/cols not stated; **2 final submission slots** ("the two final submissions").
- Structural quirk exploited: the test set's unseen product-code modalities, handled by training on 3 codes and validating on
  2, then averaging over all 10 such splits.
- **Features:** 4 engineered features, verbatim from the post:
  - missing value indicator for `m3`
  - missing value indicator for `m5`
  - area feature = product of `attribute2` and `attribute3`
  - average of `m3` through `m17`
- Selected base columns ("all of the other features were discarded"): `loading` (used as `log(loading)`),
  `attribute_0` (WOE-encoded), and measurements `0, 1, 2, 17`.
- Encodings: WOE (weight-of-evidence) on `attribute_0` — and only that column.
- Imputation: pourchot's scheme; both the KNN variant from that notebook and **median per product code** gave similar results.
- Scaling: `RobustScaler` (link cited in the body).
- Feature-source kernels / GP-generated features: not stated (his model-breadth notebook is linked instead).
- **Models:** Logistic Regression — winner family, private 0.59087 ("what we were ranked with").
- XGBoost with linear boosting (`boosters=gblinear`), best private 0.59109, unsubmitted; the variant used for the final
  slots scored 0.59053 private.
- Model-breadth screen: "We tested a bunch of models using mostly default settings" (his own notebook
  `tps-08-22-comparing-models`) and settled on the two above.
- Hyperparameters: "a minimal hyperparameter search to optimize the regularization parameters" — values not stated;
  seeds, fold repeats, library versions not stated.
- **CV:** custom split, 10 folds = all 3-train/2-validate `product_code` combinations, deterministic, no shuffle/repeat
  counts stated; the scheme is credited to @purist1024's notebook.
- CV score: not published; CV-vs-LB gap not computable.
- Trust verdict: implicit and strong — he stopped experimenting deliberately ("me and my teammate were busy most of the
  month… I imagine this saved us from overfitting to the public leaderboard"), and reports "It is encouraging that the
  private and public leaderboard scores were so similar for my final submissions".
- **Ensembling:** fold averaging only — predictions averaged over the 10 splits; no cross-family blend in the submission.
- Shared-OOF usage: none mentioned on the page (counted).
- **Post-processing:** log transform of `loading` before fitting (feature-side); no threshold, calibration, clipping, or
  rank transform stated.
- **Gains:** none quantified — the page publishes no before/after delta for the 4 features, the 6-column cut, the WOE
  encoding, the RobustScaler, or the gblinear choice.
- Ordering he does state: best XGBoost (0.59109) > best LR (0.59087) on private, yet the LR is what he was ranked with.
- **Failed:** WOE encoder fitted per-fold "instead of the full training data", motivated by leakage fear: "It didn't seem to
  help at all on the private leaderboard."
- XGBoost as the theoretically better member (0.59109) failed to become the submission — the better model was left
  unsubmitted.
- Not a failure but the reason the entry is thin: "I don't think our code is particularly presentable for sharing" — no code
  is published, only a notebook of model comparisons.
- **Comments:** 6 comments (counted, all rendered by step 1 of the retry). Technical content, all from the author
  (@rsizem2):
  - The three private numbers quoted above (XGBoost best 0.59109, chosen-for-final 0.59053, best LR 0.59087 = ranked score)
    plus the figure https://i.imgur.com/bchSfEv.png.
  - His WOE ablation: he tried two encodings — fitting the WOE encoder on each split individually versus on all training
    data — "to reduce data leakage"; it did not help on private.
  - Attitude/anti-overfitting: "Frankly, I had written off this competition by the end. Mostly just submitted on the last
    day so I wouldn't be tempted to do any more work."
  - Commenters: Sanal (642nd) praises the 3-vs-2 validation as "a clever move"; Katarina 00 (426th) asks for the XGBoost
    private score and observes nobody on the board reached 0.60; 1 appreciation comment (C4rl05/V, 53rd) with no content.
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/purist1024/principled-3-vs-2-cv-splitting-on-product-code (cited as "this notebook by
    @purist1024", written in the body as `/c/tabular-playground-series-aug-2022/discussion/purist1024`)
  - https://www.kaggle.com/code/pourchot/hunting-for-missing-values (imputation scheme)
  - https://www.kaggle.com/code/rsizem2/tps-08-22-comparing-models (their model-breadth test)
  - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
  - https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-linear-booster-booster-gblinear
  - https://i.imgur.com/bchSfEv.png (from the author's comment)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/writeups/kagglesrl-17th-place-solution
    (citation line) · https://www.kaggle.com/competitions/33108/images/thumbnail
  - https://www.kaggle.com/chrismiddlebrooks · https://www.kaggle.com/rsizem2 · https://www.kaggle.com/purist1024 ·
    https://www.kaggle.com/pourchot · https://www.kaggle.com/sanalps · https://www.kaggle.com/katarina00 ·
    https://www.kaggle.com/cv13j0
- **Lesson:** On this shift-dominated board, deleting 20 of the 26 columns and keeping a WOE-encoded linear model costs you
  nothing — and a deliberately idle final week is a valid anti-overfitting strategy.

## TPSAUG22 — consensus recipe
- **Architecture distribution:** 9th `FLAT` (4 LR members, one per feature view) · 14th `FLAT` (7 members across TabNet/LR/
  LGBM/NN/XGB/CatBoost/KNN into one blending notebook) · 17th `SEED` (one LR family averaged over the 10 3-vs-2 folds).
  - **winner topology: `FLAT`** — 9th averages 4 same-estimator/different-view members; nobody on this board built a
    stacked level 2 (`l2=none` on all three pages).
  - Read-across: the two entries that published full member tables show blends that barely beat their own best single
    (14th: +0.00012 private after the fix; 17th: his blend family's best model, XGBoost at 0.59109, never submitted), so the
    ensemble is not where the score is on this dataset.
- **New architectures at the board:** `TabNet` only — 14th's best single model (0.58908 public / 0.59098 private), still
  below the corrected blend (0.59110) and above his LR member on private (0.59018).
  - 9th and 17th are `novel=none`: logistic regression, and LR + gblinear XGBoost respectively.
  - Everything else on the board is stock: LR, LightGBM, XGBoost, CatBoost, KNN, generic MLP, WOE encoding, RobustScaler.
  - No novel architecture appears as a blender anywhere — there are no trained blenders at all in this set.
- **Agreed on (2 of 3 — TPSAUG22-14, TPSAUG22-17) — the defining primitive of this competition:** the 3-train / 2-validate
  `product_code` CV scheme, credited to the same two sources on both pages (AmbrosM's discussion/341896 and
  purist1024's `principled-3-vs-2-cv-splitting-on-product-code`), used with all 10 code-pair folds and averaged into the
  final prediction. TPSAUG22-09 states nothing about its validation scheme, so it cannot be counted.
- **Agreed on (3 of 3 — TPSAUG22-09, TPSAUG22-14, TPSAUG22-17):** logistic regression is a first-class member of the
  submission, not a throwaway baseline — 9th's submission is 4 LRs and nothing else; 14th's LR member is the best of the
  seven on public (0.58974) and second-best on private; 17th's ranked score is his LR (0.59087).
- **Agreed on (2 of 3 — TPSAUG22-14, TPSAUG22-17):** treat the test set's product-code distribution as the problem — 14th
  "made us ensure to use these new modalites (test set) in the validation process", 17th builds the split so every model is
  validated on 2 codes it never trained on. Neither names a concrete adversarial-validation model (17th's WOE-per-fold test
  is a leakage control, not AV).
- **Agreed on (2 of 3 — TPSAUG22-09, TPSAUG22-17) — and this is the board's own verdict:** minimal or zero feature
  engineering, with the rest attributed to luck/shakeup. 9th: "I have no idea how to do FE… the simpler the better",
  "Really a shake, lucky". 17th: 4 features + a 6-column keep-list, "mostly default settings", a "minimal" regularization
  search. 14th names no FE at all.
- **Divergences:** rank order here is not explained by complexity — the widest pool (14th, 7 families including TabNet)
  finished behind the smallest one (9th, 4 LRs on disjoint views) by ~0.00030 private (0.59090 submitted vs a 9th place whose
  score he never publishes) and behind 17th's single-family LR (0.59087).
- Where top beat mid: 9th's diversity is bought with **feature subsets on one estimator**; 14th bought **family diversity**
  instead and got less per unit of work, with 4 of his 7 members (XGB 0.58898, LGBM 0.58831, CatBoost 0.58964, KNN 0.58995)
  all private-below his own LR and TabNet.
- The largest single documented loss on this board is not architectural: 14th's `predict()` vs `predict_proba()` bug is
  worth +0.00021 private / +0.00066 public, i.e. ranks 14 → ~6, more than any member choice.
- **Highest-leverage single trick:** the 3-vs-2 `product_code` CV (TPSAUG22-14, who calls it "the key to not overfit the
  leaderboard"; independently adopted by TPSAUG22-17, whose only stated validation idea it is). Closest runner-up in
  measured value: 14th's predict_proba bug fix (+0.00021 private, 8 ranks).
- **Nothing worked:** fitting the WOE encoder per fold to avoid leakage — "It didn't seem to help at all on the private
  leaderboard" (TPSAUG22-17).
- **Nothing worked:** GBMs as ensemble members on this dataset — XGBoost, LightGBM and CatBoost are the three weakest members
  of 14th's table (private 0.58898 / 0.58831 / 0.58964 vs 0.59098 TabNet and 0.59018 LR); 17th also had to abandon his
  better XGBoost (0.59109) because it never made the submission.
- **Nothing worked:** leaving the strongest model unsubmitted — 17th's best private number (0.59109, gblinear XGBoost) is
  0.00022 above the score he was ranked with.
- Not countable as agreement: 9th publishes neither scores, features, folds nor failures, and 14th has 0 comments, so this
  competition's aggregate knowledge is thin by design; the two documented ideas above (3-vs-2 CV, LR-as-main-model) are the
  only ones supported by more than one page.
