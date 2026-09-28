## TPSMAR22 — Tabular Playground Series - Mar 2022
- Task: Tabular (Regression) | Metric: Mean Absolute Error | Problem: Practice regression
- Kaggle display title: "Tabular Playground Series - Mar 2022" (writeup page header; the page's own subtitle line is not published)
- Competition: https://www.kaggle.com/c/tabular-playground-series-mar-2022
- Writeups covered: 2 of 2
- Score ladder (MAE, LOWER IS BETTER): 1st `not stated/not stated` · 3rd `not stated/not stated`.
  Neither author publishes an MAE value; the only quantitative facts in the set are structural (300 Optuna trials, 3/5/10-day
  windows, 7/14-day lags, 8 epochs, lr 0.005677, layers [1066, 931], batch 256) and rank movement ("the LB shakeup was as high as
  it was", "jumping more than 300 places").
- Task shape: one target per `x, y, direction` congestion series, 5-minute resolution, test period matched to a weekday
  (`hour-minute` and day-of-week filtering is the winner's whole validation design).

### TPSMAR22-01 · 1st · El Conde de Ñáñaño (ottpocket) · LB not stated/not stated · CV not stated (no number published)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2022/discussion/316271
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/316271`; 13,335 bytes, body + 16 comments incl. 7 topic-author replies)
- **TL;DR:** "The winning score was a single lgbm with no post-processing" — heavy lag + likelihood-encoding FE, then Optuna used to
  pick which features to keep and which training rows to use.
- He validated on the week exactly before the test, filtered to the same day-of-week and hour-minute combinations as the test period,
  and ignored the public LB.
- **Architecture:** SINGLE · stages=1 · l1=1 (LGBMRegressor) · l2=none · novel=none · mod=none ·
  topo=[lag aggregates (3/5/10-day rolling + expanding; mean/var/median/min/max + 1-interval shift) per `x-y-direction`, per day and
  weekday] + [likelihood encodings: min/max/median/var/mean per `xy` and per `x-y-direction` at every `hour-minute`] → Optuna
  feature-in/out selection (300 trials) + train-row filter flags → one LightGBM → prediction, no post-processing
  - Fitted estimator is LightGBM (a stock GBM regressor); the Optuna step selects a feature subset and two training-row filters, it
    trains no meta-learner, so the topology stays single-stage (rule 1: weight/feature selection is not stacking).
  - GLOBAL model, explicitly: one regressor over all `x-y-direction` series with the location structure pushed into features. He
    tried the multi-output alternative — "train a regressor on all the spatial points at a given time simultaneously" — and it failed
    (see Failed), so the submitted model is single-target.
  - Not `CASCADE`: nothing in the winning path consumes another model's output; it consumes another notebook's *encoding recipe*
    (PyCaret notebook borrowed for the `hour-minute` x location crossing). Not `AUTOML` either — per rule 6, borrowing an idea from
    Pack-Man's PyCaret notebook is not a vendor-managed stack.
- **Setup:** rows/cols not stated. Validation = "the data that were exactly 1 week before the test", with the training data filtered
  "to only be during the same day and hour-minute combinations as the test period".
- Two-stage notebook split: one notebook builds the features, a second runs the Optuna selection; the winning feature build is pinned
  at `scriptVersionId=91804908`.
- Structural quirk exploited: the test is a single weekday at 5-minute resolution, so only rows sharing that day-of-week and
  hour-minute neighbourhood are worth training on — he let Optuna decide that rather than assuming it.
- **Features:**
  - Lags: for every `x-y-direction` combination, on both the day and the weekday, he computed "means, variances, medians, minimums,
    maximums, and 1 interval shifts", using 3, 5 and 10 day rolling windows plus expanding windows.
  - Likelihood encodings: "the minimus, maximums, medians, variances, and means for every `xy` and `x-y-direction` combination at all
    `hour-minute` combinations" — the encoding recipe taken from Pack-Man's AutoML notebook ("From the first I took a way to
    likelihood encode based on the cross between hours-minute time and location").
  - Differentiation from the public kernels: "Unlike most public kernels, I used many other features than the medians."
  - Extra spatial crossing added in the final sub (author comment): encodings on `x-y-hour-min` in addition to
    `x-y-direction-hour-min` — "The most spatially oriented thing I did for the final sub".
  - Feature count after step 1: "too many features", hence Optuna. Exact final feature count: not stated.
  - Leakage hygiene he admits skipping: "The encodings I did here were not very pretty as they did not use k-fold validation to
    prevent leakage. Another case of not having enough time."
- **Models:** LightGBM `LGBMRegressor` — the single submitted member ("a single lgbm").
- Hyperparameters: not stated (a commenter quotes `model = LGBMRegressor()` from the feature-selection notebook; the author confirms
  he tuned features only, not `num_leaves` / `learning_rate`).
- Optuna settings: `trial.suggest_categorical(feat_in_question, [True, False])` per candidate feature, 300 trials, plus Optuna flags
  for "whether I should train on only day 0" and "if I should only train on the same hour-minute times as the test"; "After 300
  trials, Optuna got very good at finding the best features."
- Seeds, library versions, fold counts: not stated.
- **CV:** no named splitter — a hand-built validation window: exactly one week before the test, filtered to matching day-of-week and
  hour-minute combinations of the test period.
- No CV/MAE number is published, so the CV-vs-LB gap cannot be computed from this writeup.
- Author's stance on the LB (his own words): "I was quite surprised to find that the LB shakeup was as high as it was. I almost
  didn't check tonight because I was so disappointed in my scores" — i.e. public scores were anti-correlated with the outcome;
  Laurent Pourchot (291st) confirms from the outside: "you were not focused on Public LB".
- **Ensembling:** none — one model, and the "hybrid"/blended boards other competitors ran (Martynov Andrey's Hybrid Regressors, 387th)
  are cited as inspiration for post-processing only, not used.
- **Post-processing:** none, and explicitly so: "The winning score was a single lgbm with no post-processing. I kinda ran out of time."
  He names the idea he took from Martynov Andrey's Hybrid Regressors notebook ("From the second, I liked the post-processing") and did
  not use it. No rounding, clipping or MAE bias correction is described — for MAE the corresponding correction would be a
  median-shift, and he does not mention one.
- **Gains:** nothing is quantified (no MAE anywhere). Ranked ordering the author gives:
  1. Lag + likelihood-encoding build well beyond the community's median-only features (stated as the difference from public kernels).
  2. Optuna feature-subset selection over the surplus ("After step 1 I had too many features ... I used Optuna to pick the best features").
  3. Optuna-chosen training-row filters (day 0 only; test-matching `hour-minute` only) folded into the same 300-trial search.
  4. Adding `x-y-hour-min` encodings alongside `x-y-direction-hour-min` (author comment; unquantified).
  5. Choosing the submission on the week-before-test validation rather than public LB — the shake-up is the evidence, its size is not stated.
- **Failed:** Wide neural networks (https://www.kaggle.com/code/ottpocket/wide-neural-predictions-with-optuna): "Given the spatial
  relationship of the data, it would make sense to train a regressor on all the spatial points at a given time simultaneously. This
  did not work out for me, however."
- Cost of that failure (author comment): "This month I spent about 95% of my effort on that failed neural submission!"
- A planned 1-d convolutional network predicting all `x-y-direction` coordinates for N time stamps was never finished — "I almost
  didn't even have the above submission because I wanted to create a 1-d convolutional network ... I opted for the easy
  lagged-encoded approach due to time constraints."
- "No generators for me this month!" — no time-series generator/augmentation work attempted.
- Diagnosed conclusion (author comment): "I was disappointed that using the spatial nature of the data did not affect much of the
  models in use."
- k-fold-safe encodings: skipped for lack of time, acknowledged as a leakage risk in the final model.
- Tuning model hyperparameters and features simultaneously: not done ("I did not do both because I was desperate and running out of
  time"), and he states it "would lead to a different subsection of features".
- **Comments:** 16 comments, 7 of them topic-author replies (counted).
- To liartem (57th), who asks whether Optuna's feature selection is model-specific: "You can easily tune both the model parameters
  (`num_leaves`, `learning_rate`,etc.) and features using this method. I did not do both because I was desperate and running out of
  time! I do not doubt that tuning both simultaneously would lead to a different subsection of features."
- To Laurent Pourchot (291st): the 95%-of-effort-on-failed-NN figure, the abandoned 1-d CNN plan, and "No generators for me this month!"
- To Pablo Prieri (91st): spatial data "did not affect much of the models in use"; final sub's only spatial content is the
  `x-y-hour-min` encodings; "I really did want to see some cool graph transformers in use here."
- To iamagoose (2nd in this competition, commenting here): "I really liked you lagging on the `x-y-dir-hour`. I wish I thought of
  that." — the 2nd-place lag recipe is named, but 2nd's own writeup is not in this index.
- Samuel Cortinhas (229th here, TPSJAN22-16 author) and Martynov Andrey (387th, whose hybrid-regressor notebook supplied the
  post-processing idea the author liked — the encoding recipe instead came from Pack-Man's PyCaret notebook) both congratulate; no
  numbers.
- **Compute:** the feature build "takes around 5-10 minutes to run"; the same pipeline runs the 300 Optuna trials; GPU/CPU, RAM,
  wall-clock of the Optuna search and Kaggle limits: not stated.
- **Artifacts:** (cited by the author, verbatim, not fetched)
  - https://www.kaggle.com/code/ottpocket/encodings-and-lags (feature build; winning version pinned
    https://www.kaggle.com/code/ottpocket/encodings-and-lags?scriptVersionId=91804908)
  - https://www.kaggle.com/code/ottpocket/feature-selection-notebook (Optuna feature/train-filter selection)
  - https://www.kaggle.com/code/ottpocket/wide-neural-predictions-with-optuna (the failed wide-NN notebook)
  - https://www.kaggle.com/code/packinman/tps-mar-2022-automl-pycaret-regression (source of the hour-minute x location likelihood encoding)
  - https://www.kaggle.com/code/martynovandrey/tps-mar-22-hybrid-regressors (source of the post-processing he liked but did not use)
  - https://www.kaggle.com/competitions/tabular-playground-series-mar-2022/writeups/el-conde-de-a-o-1st-disbelief (citation line)
  - https://www.kaggle.com/competitions/tabular-playground-series-mar-2022 , https://www.kaggle.com/ottpocket ,
    https://www.kaggle.com/packinman , https://www.kaggle.com/martynovandrey
  - commenter profiles linked on the page: https://www.kaggle.com/pabloprieri , https://www.kaggle.com/akmalmir ,
    https://www.kaggle.com/samuelcortinhas , https://www.kaggle.com/pourchot , https://www.kaggle.com/iamagoose ,
    https://www.kaggle.com/ifashion , https://www.kaggle.com/liartem , https://www.kaggle.com/mpwolke
- **Lesson:** On a single-weekday test window, spend the budget on feature-subset search and on training-row filtering that mirrors
  the test period — one untuned LightGBM on Optuna-selected lag/encoding columns beat every blended board, and the neural-spatial
  detour cost 95% of the month for nothing.

### TPSMAR22-03 · 3rd · Eric (hamstard) · LB not stated/not stated · CV not stated (no number published)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2022/discussion/317661
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/317661`; 3,741 bytes — a genuinely short writeup, body + 1 comment;
  no cache shell, solution prose present)
- **TL;DR:** One "slightly tuned" fastai tabular neural net with four date-part features and 7/14-day target lags, judged with
  `TimeSeriesSplit`; 3rd place after "jumping more than 300 places".
- **Architecture:** SINGLE · stages=1 · l1=1 (fastai `tabular_learner`) · l2=none · novel=none · mod=none ·
  topo=[day of week, is Monday, hour of day, minute of hour] + [target lagged 7 days, target lagged 14 days] →
  fastai tabular learner MLP [1066, 931], 8 epochs, lr 0.005677, batch 256 → prediction
  - Fitted estimator: a fastai tabular learner (embedding + MLP head), trained 8 epochs. Nothing is hand-built and deterministic
    here, so rule 11's `no fitted estimator` wording does not apply.
  - Single stage, no blender, no meta-learner, no second model.
  - `novel=none` is the honest tag: the page presents it as "a slightly tuned fastai tabular learner", a stock MLP-with-embeddings,
    not as a new architecture — and the author never frames it as an experiment in model class.
  - Global vs per-series: `source silent`. The writeup lists only time features and lags and never says whether `x`, `y`, `direction`
    enter as categorical features of one global fit or whether one model is trained per series; a single tabular learner implies one
    global fit, but this entry does not claim it.
- **Setup:** title-only slot information: "3rd place, 3rd month, 3 submissions" — three submissions mentioned, never explained.
  Rows/cols, split sizes: not stated.
- Structural quirk he relies on: the target repeats weekly, so 7-day and 14-day lags are the informative history.
- **Features:** exactly four time features plus two lags, verbatim: "Time features: day of week, is Monday, hour of day, minute of
  hour" and "Lag the target by 7 and 14 days".
- `is Monday` is a separate flag from `day of week` — i.e. Monday is treated as its own regime, which is the same insight as 1st's
  Optuna flag for "train on only day 0".
- No encodings, no aggregate lag statistics (unlike 1st's means/variances/medians/min/max), no external data, no dropped columns stated.
- **Models:** fastai tabular learner (https://docs.fast.ai/tabular.learner.html), "slightly tuned":
  - Number of epochs: 8
  - Learning rate: 0.005677
  - Layers: [1066, 931]
  - Batch size: 256
  - Optimizer, embedding sizes, dropout, seeds: not stated.
- **CV:** `TimeSeriesSplit` (sklearn) used "to evaluate model performance".
- Fold count, gap, repeats, seeds: not stated. CV-vs-LB gap: not computable, no MAE published.
- Trust verdict (implicit, his only validation statement): the public board is not an objective — "I surprisingly made the third place
  by jumping more than 300 places. But if you check out the top 20 in the private leader board, you'll notice all did similar jumps.
  So I guess I was quite lucky."
- **Ensembling:** none.
- **Post-processing:** not stated (no rounding, clipping or MAE bias correction described).
- **Gains:** none quantified — the writeup publishes no metric value at all. The published structure is a feature list and four
  hyperparameters; the only outcome given is rank and the 300+ place jump.
- **Failed:** nothing is reported as tried-and-rejected (checked: the body is two bullet groups and a sign-off; the single comment
  carries no author reply). No ablation of the 7-day vs 14-day lag, no mention of GBM alternatives tried and dropped.
- What he volunteers instead: luck — the private shake-up was shared by the whole top 20.
- **Comments:** 1 comment, 0 from the topic author (counted).
- Andy8744: "It does appear that in a lot of competitions, the top public LB scores are misleading and leads to overfitting to public
  test set." — the only substantive reply on the page.
- Page tags (as published): Regression, Time Series Analysis, sklearn, Tabular.
- **Compute:** not stated (8 epochs, batch 256 — no wall-clock or device given).
- **Artifacts:** (cited by the author, verbatim, not fetched)
  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
  - https://docs.fast.ai/tabular.learner.html
  - https://www.kaggle.com/competitions/tabular-playground-series-mar-2022/writeups/theoriginalguesstimator-3rd-place-3rd-month-3-subm (citation line)
  - https://www.kaggle.com/hamstard (author profile), https://www.kaggle.com/andy8744 (commenter profile)
  - tag search links on the page: https://www.kaggle.com/search?q=regression+tag%3A%22regression%22 ,
    https://www.kaggle.com/search?q=time+series+analysis+tag%3A%22time+series+analysis%22 ,
    https://www.kaggle.com/search?q=sklearn+tag%3A%22sklearn%22 ,
    https://www.kaggle.com/search?q=tabular+tag%3A%22tabular%22
- **Lesson:** In a shake-up month a 9-line writeup with one lightly-tuned NN, two weekly lags and `TimeSeriesSplit` can medal — the
  edge was validating on the honest next-period window, not the architecture.

## TPSMAR22 — consensus recipe
- **Architecture distribution:** 1st `SINGLE` (one `LGBMRegressor` on Optuna-selected lag + likelihood-encoding features, no
  post-processing) · 3rd `SINGLE` (one fastai tabular learner, 8 epochs / lr 0.005677 / layers [1066, 931] / batch 256).
  - **winner topology: `SINGLE`** — and it is the topology of this file at both ranks; no `FLAT`, `STACK2`, `CASCADE`, `PSEUDO` or
    `AUTOML` submission appears in the set, even though the winner borrowed his encoding recipe from a PyCaret AutoML notebook and his
    admired post-processing idea from a hybrid-regressor notebook (rule 6: borrowed ideas are not vendor-managed stacks).
  - Global vs per-series: both entries are silent or implied-global; 1st is genuinely global by construction (one single-target
    regressor, all location structure pushed into lag/encoding features, and he reports the multi-output spatial version failed),
    3rd never states it.
  - Fitted-estimator audit (rule 11): 1st fits LightGBM on selected columns, 3rd fits an NN — both learn coefficients, so neither is
    a deterministic decomposition and neither gets `no fitted estimator` wording.
- **New architectures at the board:** none — `novel=none` at both ranks. The two submitted estimators are a stock gradient-boosted
  tree and a stock fastai tabular MLP-with-embeddings.
  - Non-architecture ingredients worth naming: likelihood encodings crossed over `hour-minute` x location (1st), Optuna
    per-feature in/out selection (1st), `TimeSeriesSplit` validation (3rd).
  - Architectures attempted and NOT shipped: a wide NN predicting all spatial points at one timestamp and a 1-d CNN over all
    `x-y-direction` coordinates for N timestamps (1st, both failed/abandoned).
- **Agreed on (2 of 2 — TPSMAR22-01, TPSMAR22-03):** lagged target history is the core feature family. 1st: 3, 5 and 10-day rolling
  plus expanding windows over `x-y-direction`, storing means, variances, medians, minimums, maximums and 1-interval shifts, on both
  the day and the weekday. 3rd: the target lagged 7 and 14 days.
- **Agreed on (2 of 2 — TPSMAR22-01, TPSMAR22-03):** date-part features, with Monday singled out. 3rd writes `day of week` AND
  `is Monday` as separate features; 1st lets Optuna decide whether to "train on only day 0", i.e. treat that weekday as its own regime.
- **Agreed on (2 of 2 — TPSMAR22-01, TPSMAR22-03):** never optimize against the public LB, and both finished on a huge private
  shake-up: 1st "the LB shakeup was as high as it was ... I almost didn't check tonight because I was so disappointed in my scores"
  (corroborated by commenter Laurent Pourchot, 291st: "you were not focused on Public LB"); 3rd "jumping more than 300 places ...
  if you check out the top 20 in the private leader board, you'll notice all did similar jumps".
- **Divergences:** the model family and the search axis split the two. 1st ran a large feature/row-filter search (300 Optuna trials over
  per-feature booleans and training-row flags) on a GBM with untuned hyperparameters; 3rd ran no feature search at all (6 features) on
  an NN with four published hyperparameters and no tuning narrative. Rank gap between them: 1st vs 3rd, MAE unpublished on both pages,
  so the cost of the divergence cannot be quoted.
- Validation design diverges: 1st hand-built a test-mirroring window (exactly one week before the test, filtered to the same
  day-of-week and `hour-minute` rows); 3rd used a generic `TimeSeriesSplit` with no gap/fold settings published.
- Leakage discipline diverges: 1st states his encodings "did not use k-fold validation to prevent leakage"; 3rd's `TimeSeriesSplit`
  excludes future rows by construction, though his 7/14-day lags are computed with no stated fold-safety note.
- Post-processing diverges by intent: 1st deliberately shipped none ("a single lgbm with no post-processing") after admiring
  Martynov Andrey's post-processing; 3rd simply does not mention any.
- **Highest-leverage single trick:** run the feature selection as an Optuna boolean per column (`trial.suggest_categorical(feat,
  [True, False])`, 300 trials) on top of an over-supplied lag/encoding build, and let the same search choose the training-row filter —
  one LightGBM then beat every blended board (TPSMAR22-01).
- **Nothing worked:** modelling the spatial structure as a multi-output problem — 1st's wide NN over all spatial points at a given time
  failed, and his planned 1-d CNN over all `x-y-direction` coordinates consumed the rest of his effort ("about 95% of my effort on
  that failed neural submission"), with his diagnosis that "using the spatial nature of the data did not affect much of the models in
  use". Note the boundary: 3rd's neural net did medal, so the failure is the all-locations-simultaneously formulation, not NNs.
- **Nothing worked (adopted-then-skipped):** post-processing — the winner named it as the thing he liked from another notebook and
  shipped without it for lack of time (TPSMAR22-01).
- **Nothing worked:** time-series generators / synthetic augmentation — explicitly skipped ("No generators for me this month!", 1st),
  so it is an untested option rather than a measured dead end; recorded here because the top of this board ran out of time, not ideas.
- **Nothing worked (measurably):** nothing in this set is quantified as a loss — neither page publishes an MAE, so all "did not work"
  statements in this file are the authors' own qualitative verdicts.
