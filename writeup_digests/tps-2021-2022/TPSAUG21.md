## TPSAUG21 — Tabular Playground Series - Aug 2021
- Task: Tabular (Regression) | Metric: RMSE | Problem: Practice regression
- Kaggle display title: "Tabular Playground Series - Aug 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-aug-2021
- Writeups covered: 1 of 1
- Score ladder (RMSE, LOWER better; `public/private` as reported by the authors): 1st `not stated / not stated` — the page publishes no
  RMSE anywhere, in the body or in its 12 comments; the only number in the published method is `learning_rate = 0.2`.
- The metric itself is confirmed on the page by the author's own wording ("rmse is huge"), and it is the only score-related term used.
- Single-entry file: every consensus line below is stated by TPSAUG21-01 alone, and no agreement claim is made.

### TPSAUG21-01 · 1st · Ivan Kontic (ivankontic) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-aug-2021/discussion/270051
- **Status:** FETCHED after the ladder: the plain `/c/.../discussion/270051` call returned a 164 B cache shell; `X-No-Cache: true` on
  the same URL recovered 7,352 B (body + pseudocode + all 12 comments), `X-Engine: browser` + `X-Timeout: 60` returned the identical
  7,352 B, and the `/competitions/.../discussion/270051` variant gave 7,828 B of the same content without the citation block
- **TL;DR:** A self-invented transductive boost: fit LightGBM on the **test** features against pseudo-labels, compute the training-set
  residual against that model, fit a second LightGBM on the residual, and push `0.2 x` the predicted error back into the pseudo-labels
  — repeated while "our score is getting better".
- Two `lgb.LGBMRegressor()` calls with no arguments are the entire published model; nothing else about the pipeline is quantified.
- **Architecture:** PSEUDO · stages=2 per round, round count not stated ("several times") · l1=2 LGBMRegressor per round (one fit on
  test+pseudo-labels, one fit on the train residual) · l2=none (no blender; each round rewrites `test_y` in place) · novel=none ·
  mod=none ·
  topo=pseudolabels → LGBM fit(test_df, pseudo_y) → new_y = train_y - LGBM.predict(train_df) → LGBM fit(train_df, new_y) →
  test_y += 0.2 x error_prediction(test_df) → repeat, last test_y is submission.csv
  - This is a loop, not a DAG: the previous round's test predictions *are* the next round's labels, which is the `PSEUDO` definition;
    the residual-boosting shape alone would be `CASCADE`, but the loop state (evolving `test_y`) is what produces the submission, so
    `PSEUDO` is primary per ruling 9.
  - `mod=pseudo` is deliberately not added — that modifier is for a loop that merely augments level-1 members of some other ensemble.
  - `novel=none` is about the estimator: both stages are stock LightGBM regressors. What is new here is the **topology**, and the
    author claims he could not find prior art for it: "I tried to find if someone use something similar, unsuccessfully."
- **Setup:** rows/cols, target name, split sizes, submission slots: all not stated — the pseudocode comment only says
  "test_df, train_df, train_y are self explanatory".
- Structural quirk exploited: the test set is used as a *training corpus* (the first model is fit on `test_df` against pseudo-labels),
  so the model's own in-sample fit on test defines the residual that gets learned next.
- **Features:** none stated — the page never lists, engineers, encodes or drops a single column (`source silent`).
- **Models:** `lgb.LGBMRegressor()` twice per round (LightGBM sklearn wrapper), called with **no arguments** in the published
  pseudocode, so the submitted configuration is LightGBM defaults as far as the page shows.
- `learning_rate = 0.2` in the pseudocode is a shrinkage factor on the error update, **not** a LightGBM hyperparameter — do not
  transcribe it into `LGBMRegressor(learning_rate=...)`.
- Author's comment on why the damping exists: the label-fitting model is "trained on very poor dataset for a given label, rmse is
  huge", so it "poorly predicted error-values", and "in two consecutive iteration, model predicts errors with different signs for same
  item" → the 0.2 factor was added "in try to prevent fluctuation", and "Later test have shown better result for with lr then without
  it. So, just intuition."
- Rounds, seeds, early stopping, library versions: not stated.
- **CV:** no CV value, splitter, fold count or repeat count on the page.
- The loop's own stopping rule is "We can do this several times while our score is getting better" — the page never says whether that
  score is a local CV or the leaderboard, so the iteration count is attached to an unspecified score.
- CV-vs-LB gap: not computable; the author publishes no number of any kind.
- Trust verdict: not stated.
- **Ensembling:** none — no blend, no weights, no meta-learner. The only "combination" is the additive update
  `test_y = test_y + error_prediction * 0.2`, i.e. a damped self-correction of the pseudo-labels.
- Shared-OOF usage: none cited.
- Number of models averaged: 0 averaged; 2 fitted per round.
- **Post-processing:** not stated (no threshold, clipping, rounding, calibration or label trick appears). The 0.2-shrunk error update is
  part of the training loop, not output post-processing.
- **Gains:** NOT STATED — there is no delta, no iteration-by-iteration score, and no final RMSE on the page.
- The only comparative claim is unquantified: with the `learning_rate` damping beats without it (author's "Later test have shown better
  result for with lr then without it").
- **Failed:** no prior art found: "I tried to find if someone use something similar, unsuccessfully."
- Undamped iterations failed by construction: consecutive rounds flipped the error sign for the same row, which is what forced the 0.2
  damping.
- He searched for a similar published method and found none, so the scheme had no external baseline before he used it.
- Publication gaps that block replication: the winning run is not shared — "my final notebook is really mess and I'm embarrassed to
  make it public"; only a cleaned initial notebook is linked, and the round count / stopping score are never given.
- **Comments:** 12 comments; 1 deleted; the author's replies carry the algorithm's only rationale (the damping explanation and the
  huge-RMSE diagnosis, quoted under `Models`).
  - Mark Babayev (667th) reports the method transfers: "I added it to my notebook with H2O AutoML model and it returned the best
    result, even better then the built-in H2O model stacking" — his notebook is linked in the comment.
  - grayjay (462nd): "Very nice. I myself tried 41 (other) approaches; none worked."
  - Realtimshady (398th): "I also tried a pseudolabel approach but with only 1 iteration" — i.e. the same family, no repetition loop.
  - kailai (shown on this page as 4th in this Competition) — one line: "You are a genius!"
  - Nobody asks for the LightGBM parameters, the initial pseudo-label source, or a score, and none is volunteered.
- **Compute:** not stated.
- **Artifacts:** (verbatim from the page; not fetched)
  - https://www.kaggle.com/ivankontic/003-final-my-boost-1st-place?scriptVersionId=73878373 (body citation — "new initial notebook")
  - https://www.kaggle.com/markbquant/aug-21-alternative-h2o-stacking-and-blending (comment citation — H2O AutoML port of the method)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2021/writeups/ivan-kontic-1st-place-pseudocode (citation line)
  - https://www.kaggle.com/competitions/tabular-playground-series-aug-2021
  - https://www.kaggle.com/ivankontic (author)
  - comment links: https://www.kaggle.com/sergeyzemskov, https://www.kaggle.com/markbquant, https://www.kaggle.com/bernhardklinger,
    https://www.kaggle.com/grayjay, https://www.kaggle.com/realtimshady, https://www.kaggle.com/kailai,
    https://www.kaggle.com/choondrise
  - where the initial `test_y = pseudolabels` came from: no artifact and no description on the page
- **Lesson:** Treat the test set as a trainable corpus: fit on your pseudo-labels, learn your own residual on the train set, and write
  back a damped correction — but publish the score per round, or nobody can tell you how many rounds won.

## TPSAUG21 — consensus recipe
- **Architecture distribution:** 1st `PSEUDO` (transductive residual loop: LGBM on test+pseudo-labels → LGBM on the train residual →
  `test_y += 0.2 x error`, repeated) .
  - **winner topology: `PSEUDO`** — one writeup on this board, so the label has no comparator inside the file; the loop is also
    structurally a residual-boost cascade, which is why the primary follows ruling 9 rather than the CASCADE pattern alone.
- **New architectures at the board:** none. `novel=none` on the only page (checked against its full text): both stages are stock
  `lgb.LGBMRegressor` calls with no arguments. The genuinely new thing is the **training topology**, and the author himself says he
  "tried to find if someone use something similar, unsuccessfully" — so this is a novel *scheme*, not a novel *model*, and 2021
  Playground regression here contains no tabular-NN, no transformer and no vendor-managed submitted stack.
- **Stated by the sole entry (TPSAUG21-01; no cross-writeup agreement is possible in a 1-entry file):** a two-LightGBM pseudo-label
  loop with 0.2 damping is what produced the winning submission, and nothing else on the page (no FE, no blending, no
  post-processing) is claimed as a contributor.
- **Stated by the sole entry (TPSAUG21-01):** iteration is the point — the author repeats "while our score is getting better", and
  the one commenter who tried the same family reports doing "only 1 iteration" (Realtimshady, 398th, on this page).
- **Divergences:** none observable — single-entry file. The board's own information asymmetry is that 1st publishes a method with zero
  numbers, so rank-vs-method cannot be cross-checked within this file.
- **Highest-leverage single trick:** the damped self-correction update `test_y = test_y + 0.2 x error_prediction(test_df)`
  (TPSAUG21-01) — the damping is the only hyperparameter the page publishes, and the author's stated reason is that without it
  consecutive rounds flipped the predicted error's sign for the same row.
- **Nothing worked:** undamped repeated rounds (sign-flipping error predictions between consecutive iterations) and, per the thread,
  41 other approaches tried by a commenter with none working (grayjay, 462nd) — the latter is a commenter's report, not the author's.
- Not stated anywhere on this page, and therefore not to be assumed: the source of the initial pseudo-labels, the number of rounds, the
  score used to stop, every LightGBM hyperparameter, all public/private RMSE values, any CV protocol, feature list, row counts and
  compute budget.
