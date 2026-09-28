## TPSMAY21 — Tabular Playground Series - May 2021
- Task: Tabular (Multiclass Classification) | Metric: Multiclass Loss | Problem: Practice multiclass classification
- Kaggle display title: "Tabular Playground Series - May 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-may-2021
- Writeups covered: 3 of 3
- Score ladder (multiclass log loss, LOWER better; `public/private` as reported by the authors): 1st `1.08564/1.08763` ·
  3rd `1.08514/1.08769` · 13th `not stated` (the page publishes no score).
- The public/private order flips between 1st and 3rd: 3rd has the better public (1.08514 < 1.08564) but 1st the better private
  (1.08763 < 1.08769) — final rank was decided on private by 0.00006 log-loss units (TPSMAY21-01, TPSMAY21-03).
- Metric direction from the 1st page: log loss, lower better — the winner clips probabilities because it "helps mitigating the
  extremes of too small or too large values in the log_loss metric".

### TPSMAY21-01 · 1st · jj vanman (jjvanman) · LB 1.08564/1.08763 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/243054
- **Status:** FETCHED (jina `/c/.../discussion/243054`, 15,828 B in the audit re-fetch: full body + the "31 Comments" thread with
  all author replies; the page title itself reads "[1st place] Solution description")
- **TL;DR:** Five *different* base models (LogReg, RandomForest, XGBoost, LightGBM, CatBoost), each run over 3 seeds × 5-fold
  stratified CV, their OOF predictions fed to a 5-fold-stacked RidgeRegression meta wrapped in sklearn's `CalibratedClassifierCV`;
  the submission is the stack's clipped probabilities (0.05–0.95).
- He names no NN and no feature subsets: "due to the non existence of any dominant features or highly associated features with the
  target ... I kept all the features".
- What he credits the win to: "trusting my local cv results and not using public lb and not blindly using the automl results".
- **Architecture:** STACK2 · stages=2 · l1=5 model families (LogReg, RF, XGB, LGBM, CatBoost) × 3 seeds × 5-fold stratified CV,
  OOF predictions per fold · l2=RidgeRegression meta in `CalibratedClassifierCV`, trained on the stage-1 OOF matrix in 5-fold
  stacking · novel=none · mod=none ·
  topo=5 families [×3 seeds ×5 folds] → OOF matrix → Ridge+CalibratedClassifierCV stack (5-fold) → clip probs to [0.05,0.95] → submission
  - Tag verdict against the earlier draft of this entry: **not `SEED` and not `multi-view`** — the submitted prediction is produced
    by a meta-learner fitted on the stage-1 OOF matrix (that is the definition of `STACK2`), and there are no per-branch feature
    subsets anywhere on the page (all features kept; only encodings differ per model), so `mod=multi-view` is unsupported.
  - Seed/fold averaging is internal to level 0 (ruling 4) and never a stage; the diversity comes from five different families, not
    from randomness.
  - The author tested the `FLAT` alternative and rejected it: "I tried weighted average of models and it did worse than stacking".
- **Setup:** rows/cols, split sizes, submission slots: not stated on the page (nor the class count). Workflow in Google Colab.
- Structural quirk exploited: none named; the opposite — he explicitly declined the public pretrained/automl submissions.
- **Features:** no feature engineering survived: "I tried PCA, clustering, also played with featuretools but all led to either
  horrible overfitting or poor local cv so at the end I did not pursue feature engineering much."
- All columns kept (see the Setup quote above); no dropped columns named.
- Encodings only: one-hot via scipy sparse for the LogisticRegression (to fit in-memory); OrdinalEncoder (label encoding) for all
  tree-based models. GP/generated features and feature-source kernels: none.
- **Models:** LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost — "each trained on a 5fold stratified cv on three random
  seeds, using out of fold predictions for each fold and averaging across folds for predicting the test set".
- "I used Optuna to lightly tune each model (you will see my final parameters in the code above)" — the values live in the linked
  Colab gist, not on this page (`source silent`).
- Class weighting "did not really work for LogisticsRegression"; the other models "used the defaults class weights".
- Fold counts, seeds and per-model round counts other than 5 folds × 3 seeds: not stated.
- **CV:** 5-fold stratified CV per model, OOF predictions as the stacking input; no CV number appears on the page ("local cv" is
  only invoked as a decision rule) — CV-vs-LB gap not computable.
- Author's trust verdict (verbatim): "what worked at the end of was trusting my local cv results and not using public lb".
- **Ensembling:** "I used a 5fold stacking of all models above and used a meta model of RidgeRegerssion, using CalibratedClassifierCV
  in sklearn." (typos are the author's).
- Weighted averaging tried, judged worse than stacking, not used.
- Shared-OOF usage: none — public automl results "did not help my local cv so I did not use them at the end".
- Number of models averaged: 5 families × 3 seeds × 5 folds = 75 base fits feeding the fold-stacked meta (author states the factors,
  not the product).
- **Post-processing:** "I ended up clipping the probabilities (below 0.05 and above .95) to help the log_loss metric" — motivation
  cited to the Medium article in `Artifacts`.
- **Gains:** winning submission 1.08564 public / 1.08763 private; no per-step deltas published.
- Ordering claimed by the author: stacking > weighted average; probability clipping helps log loss; both without numbers.
- Declined hypothetical gain: "blending with top public notebooks could have brought the private LB scoe down to 1.08742" — i.e. an
  available −0.00021 private he chose not to take.
- **Failed:** PCA, clustering and featuretools FE — "horrible overfitting or poor local cv".
- Class weighting on the LogisticRegression.
- Weighted-average blending (lost to stacking).
- Public automl/pretrained submissions: no local-CV gain, excluded.
- Over/undersampling never tried: after reading Remek Kinas's notebook showing "it likely would not work", "I did not have time, so
  did not really pursue any further".
- **Comments:** 31 comments; the thread carries several author replies — the technical extras live there.
  - Author on the pipeline: "I never ran this pipeline end-to-end actually. I ran cv code for each model segment separately, wrote
    the results into csv/txt files and then separately ran the stacking pipeline."; compute = "google colab standard GPU"; "Random
    Forest and Logistic regression were slow as they do not use the GPU in sklearn implementation (you could use RAPIDS I guess)";
    "you could make Logistic Regression faster by removing the calibration cv with minimal impact on overall score" — confirms the
    CalibratedClassifierCV wrapper was active and cost time, not score.
  - Author on tuning: "The hyperparameter tuning segments also were not all run fully as in most cases it did not matter so I stopped
    with just a few experiments and manually tuning instead."
  - Unanswered: tharun_01 (34th): "how did you perform stacking without oof preds? I see that in your notebook you save the oof
    preds as a txt file ... the text file looks like not in a correct file format. Please help me with this" — no reply in the tree.
  - anthony (no rank shown) opens with "first place without denoising encoders (they are amazing though)"; the author answers
    "probably because masters of DAE got bored with these series lol" — 1st used no autoencoder/denoising stack.
  - Ranks recorded in the thread: Sharlto Cope (5th), Remek Kinas (40th), hayahiko (37th), Ryan Barretto (89th), Ansh Gupta (108th),
    Azakk/andrewkkchoi (163rd), Anand Philip (189th), Hikmet Sezen (61st), Dave E (307th), jeong (505th), tharun_01 (34th).
- **Compute:** Google Colab standard GPU; wall-clock, RAM and cost not stated; tuning only partially run.
- **Artifacts:** (verbatim from the page; not fetched)
  - https://colab.research.google.com/gist/academicsuspect/0aac7bd6e506f5f70295bfc9a3dc2250/tabular-may-baseline.ipynb?authuser=1#scrollTo=LtC_S97E8ep_
    ("the bulk of my code on Google Colab for individual models and cv (does not include the stacking stage)")
  - https://medium.com/@egor_vorobiev/how-to-improve-log-loss-score-kaggle-trick-3f95577839f1 (the clipping-for-log_loss article)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216087 (fatihozturk's Jan 3rd place — CV framework and
    stacking basis)
  - https://www.kaggle.com/remekkinas/tps5-is-about-sparsity-shap-extensive (the over/undersampling analysis he followed)
  - https://www.kaggle.com/ryanbarretto/boring-blend-of-stack-and-weights/comments (recommended stacking framework)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021/writeups/jj-vanman-1st-place-solution-description
    (citation line)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021
  - https://www.kaggle.com/jjvanman (author), https://www.kaggle.com/fatihozturk, https://www.kaggle.com/hiro5299834,
    https://www.kaggle.com/remekkinas, https://www.kaggle.com/ryanbarretto (credits)
  - comment links: https://www.kaggle.com/tharun2001, https://www.kaggle.com/andrewkkchoi, https://www.kaggle.com/anthonydwan,
    https://www.kaggle.com/aphilip, https://www.kaggle.com/devhunmin, https://www.kaggle.com/davidedwards1,
    https://www.kaggle.com/jeongyoonlee, https://www.kaggle.com/hayahiko, https://www.kaggle.com/victoriamazina,
    https://www.kaggle.com/hikmetsezen, https://www.kaggle.com/dwin183287, https://www.kaggle.com/ansh422, https://www.kaggle.com/fmigas
- **Lesson:** On a log-loss board, diversity at level 0 plus a calibrated Ridge stack plus probability clipping to [0.05, 0.95] beat
  weighted averaging, every FE attempt and all public pretrained submissions here — and the win margin was only 0.00006 private.

### TPSMAY21-03 · 3rd · MICADEE (adegladius; Adeyinka Michael Sotunde) · LB 1.08514/1.08769 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/243093
- **Status:** FETCHED (audit re-fetch: the plain `/c/.../discussion/243093` form returned 9,017 B with the full body + all 11
  comments; the writer first hit a ~2.5 KB cached shell and recovered it with `X-No-Cache: true`)
- **TL;DR:** He "ensembled just only three submission files", chosen by hand as the ones where Class_2 probability was highest,
  and reached 3rd without touching any public pretrained submission.
- Members: one LightAutoML run, one "AutoLightgbm + DAE" run, one LightGBM tuned with Optuna (named only in a comment reply).
- **Architecture:** FLAT · stages=1 · l1=3 own submissions (LightAutoML, AutoLightgbm+DAE, LightGBM+Optuna) ·
  l2=not stated (no meta-learner named) · novel=none · mod=none ·
  topo=own submissions filtered on Class_2 probability mass → pick best 3 → ensemble of 3
  - Selection is hand-made on a class-probability criterion (files where "Class_2 possesses a better probability values"), then the
    three are combined — member selection plus a blend with no fitted meta-learner, hence `FLAT` (ruling 1).
  - The blending math itself (average / weighted / rank) is not stated anywhere on the page: `not stated (proof: the body says only
    "I ensembled just only these three submission files" and names no weights, no metric-driven weight search and no second-stage model)`.
  - `novel=none` is the truthful value even though one member is called "AutoLightgbm + DAE": the autoencoder is inside a level-0
    member, is never described, and is not the submitted architecture. LightAutoML is likewise only one member, so `AUTOML`
    does not apply (ruling 6).
- **Setup:** 2 final slots implied by a Kaggle Playground of that month but never stated; rows/cols/split sizes not stated.
- Structural quirk exploited: Class_2 is "the most common class in the training data" (author's reply) and he asserts a model that
  scores Class_2 well scores well overall.
- **Features:** none named anywhere on the page — no engineered feature, encoding, GP feature or dropped column is mentioned; the
  three members' own preprocessing is `source silent`.
- **Models:** three level-0 members, named only in the author's comment reply to @tharun2001:
  - one LightAutoML pipeline,
  - one AutoLightgbm + DAE pipeline,
  - one LightGBM with Optuna.
- Hyperparameters, folds, seeds, library versions: not stated.
- **CV:** no CV number, splitter, fold count or trust statement on the page; only LB is quoted (`source silent`).
- **Ensembling:** exactly 3 submission files combined into the final entry → Public 1.08514, Private 1.08769.
- Explicitly excluded: "any of these predictions (submission files) from these so called 'Pretrained Models'" — refused from the
  beginning to the end of the competition.
- Shared-OOF usage: none (the refusal above).
- **Post-processing:** not stated (the Class_2 filter is a *selection* rule for which submissions to blend, not a transformation of
  the probabilities; the page describes no threshold, calibration, clipping or rounding).
- **Gains:** final blend = 1.08514 public / 1.08769 private; no intermediate single-model score is published, so the blend's own
  gain cannot be computed.
- Ordering only, from the author's narrative: hand-filtering his own submissions on Class_2 probability quality > blind blending of
  everything he had > using public pretrained submission files (rejected a priori).
- **Failed:** nothing is named as a failed model or feature.
- Anti-knowledge of record is his *refusal*, not a measured dead end: the pretrained public submission files were never tested
  ("I refused to use any of these predictions ... for ensemble from the beginning to the end of this competition").
- The page publishes no per-member preprocessing or tuning detail either — what he actually ran (LightAutoML, AutoLightgbm+DAE,
  LGBM+Optuna) is named only, never described.
- **Comments:** 11 comments; the author replies several times and the model list exists **only** in the replies.
  - Reply to @yichian (697th): "only those predictions where 'Class_2' is highly favored had better score ... any prediction where
    'Class_2' possesses a better probability values gives a better score"; on why Class_2: "Yes, because 'Class_2' can tell us more
    about other Classes" (confirming it is the most common class).
  - Reply to @tharun2001 (34th): "I used one LightAutoML, one AutoLightgbm + DAE and one Lightgbm with Optuna. My sole target is the
    submission files (or predictions) where 'Class_2' is highly favored."
  - Two commenters (Hikmet Sezen, 61st; tharun_01, 34th) ask for a notebook / more detail; none is shared.
  - One comment on the thread has been deleted ("This comment has been deleted.").
  - Lázaro (lazaro97, 13th in this Competition — the TPSMAY21-13 author) replies: "Three times in a row? That's awesome!" and
    "A complex proposal (more of 20 models) is difficult to implement in the real world", i.e. 3rd's 3-file ensemble vs his own 20+
    model pipeline.
- **Compute:** not stated.
- **Artifacts:** the body cites no notebooks or datasets.
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021/writeups/adeyinka-michael-sotunde-third-3rd-place-solution-
    (citation line / writeup-card URL)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021
  - https://www.kaggle.com/adegladius (author profile)
  - comment links only: https://www.kaggle.com/yichian, https://www.kaggle.com/tiwariayan, https://www.kaggle.com/hikmetsezen,
    https://www.kaggle.com/dwin183287, https://www.kaggle.com/tharun2001, https://www.kaggle.com/lazaro97
- **Lesson:** If one class dominates the prior, rank your own submissions by how well they predict that class and blend only the top
  few — but expect to publish zero transferable detail.

### TPSMAY21-13 · 13th · Lázaro (lazaro97 / Virtual-ML) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/243028
- **Status:** FETCHED (first `/c/.../discussion/243028` jina call returned 7,418 B: full body + code + the 1 comment; the `/c/`
  retry is byte-identical)
- **TL;DR:** Two-stage LightGBM pipeline: stage 1 emits per-model OOF and test probability matrices saved as `.npy` with the log loss
  in the filename; stage 2 merges them with linear weights optimized against the metric.
- Publishes no score, no hyperparameter values and no notebook link ("i have a ban for using gpu and I cant share the link").
- **Architecture:** FLAT · stages=2 · l1="multiple LightGBM models (count not stated), each a StratifiedKFold run saving train+test
  prediction matrices" · l2="linear weights over stage-1 submissions, optimized with the metric; no estimator named" ·
  novel=none · mod=none ·
  topo=StratifiedKFold LightGBM (Optuna params) [n models] → saved OOF/test `.npy` → linear-weight merge tuned on log loss
  - Stage 2 is weight selection against the metric, which is `FLAT` by ruling 1; the page never says a classifier/regressor was
    fitted on the OOF matrix, so `STACK2` would be an upgrade the source does not support.
  - `l1` count is not stated: the code saves one train/test pair per `txt` label, so the pool is several labelled LGBM configs,
    but how many is never given.
  - 4 output classes are visible in the code (`yv=np.zeros((len(X_train),4))`), and each fold's fold-level log loss plus the overall
    `log_loss(y_train, yv)` is printed and baked into the saved filenames.
- **Setup:** rows/cols not stated; 2 submission slots not stated; 4-class multiclass visible from the code.
- Split: `StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)` — the constants live in a config file the
  page does not show, so fold count and seed are `not stated`.
- Structural quirk exploited: none named.
- **Features:** one encoding helper published as code: `num_encode(train_df, test_df, column, encoding)` creates `num_{col}` columns
  by fitting the passed encoder on the training frame only and transforming both frames.
- The encoder object passed in is not named, so target / ordinal / other is not stated.
- Preprocessing philosophy (verbatim): "About preprocessing, i think is better try diversity of datas but i'm not sure."
- Column drops, GP features, feature-source kernels: not stated.
- **Models:** LightGBM only (`lgb.train` with a `lgb.Dataset` valid set and `EARLY_STOPPING_ROUNDS`).
- Hyperparameters come from Optuna: `get_params(study, obj, dir_save)` calls `study.optimize(obj, timeout=TIME)` and
  `study.enqueue_trial(lgb_tune)` (seeds the study with the current best config); the searched space and the resulting `lgb_tune`
  values are not stated.
- `TIME`, `EARLY_STOPPING_ROUNDS`, `NUM_FOLDS`, `RANDOM_STATE`, `path` are read from a config cell not shown.
- Motivation for the split into `lmodelv` / `training_lgbv`: so the algorithm "can optimize ... in two ways", citing
  https://www.kaggle.com/awwalmalhi/extreme-fine-tuning-lgbm-using-7-step-training
- Library versions, seeds, GPU/CPU per model: not stated.
- **CV:** the in-code CV metric is `log_loss` per fold and over the full OOF matrix, and the full-OOF value is baked into the saved
  `.npy` filenames (`preds/train_{txt}_{log_loss}.npy`); the page itself states no CV or LB number.
- No CV or LB number is published anywhere on the page, so the CV-vs-LB gap cannot be computed.
- Trust verdict (verbatim): "Finally the cv is the most important thing".
- **Ensembling:** stage-2 merge — "Some submissions can merge with linear weigths, but this can be optimized with a metric".
- Weights, weight bounds, whether the optimization is hill-climbing or a fitted linear model: not stated.
- Test-side fold averaging is explicit in code: `yt += model.predict(X_test)/NUM_FOLDS`.
- Shared-OOF usage: none mentioned.
- Number of models averaged: not stated.
- **Post-processing:** not stated.
- **Gains:** NOT STATED — the page contains no delta for any step, model or preprocessing choice.
- Ordering only: he credits Optuna-driven LGBM + saved OOF matrices + metric-optimized linear weights as the whole ranking path,
  and says "in general, this works properly (in this case)".
- **Failed:** "I didn't use autoencoder but i should did" — an acknowledged omission, i.e. a self-reported gap rather than a tested
  dead end.
- He reports preprocessing diversity as unproven in his own hands: "i think is better try diversity of datas but i'm not sure".
- Self-critique of the whole pipeline: "I think the code and all this workflow can be better".
- He was banned from GPU usage on Kaggle at the time, which is why the stage-2 notebook is not shared.
- **Comments:** the page counts 1 comment + 1 appreciation comment; the sole non-deleted one is "Thanks for sharing!" from dib /
  diegoiglesias (261st in this Competition), listed under Appreciation — the other comment slot is deleted; no technical content.
- **Compute:** Kaggle GPU ban stated; wall-clock, RAM and limits otherwise not stated.
- **Artifacts:** (verbatim from the page; not fetched)
  - https://www.kaggle.com/awwalmalhi/extreme-fine-tuning-lgbm-using-7-step-training (cited in the body for the two-way tuning idea)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021/writeups/virtualml-13th-my-solution-2nd-stage-of-modeling
    (citation line / writeup-card URL)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2021
  - https://www.kaggle.com/antoreepjana, https://www.kaggle.com/lazaro97 (Authors block; the writeup's own `Authors` list shows
    antoreepjana and Lázaro)
  - https://www.kaggle.com/diegoiglesias (commenter)
  - workflow figure: https://storage.googleapis.com/kagglesdsdata/datasets/1251020/2086737/stacked.PNG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210601%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210601T003655Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=28df58bc7badf65434984dadd1f077745b689e5803953d61ed8d688c453c4e38d89d25d353a7d7642c98111b30dece4bca7a18db82bb4bc265db8d0691589817c5884579e3ca59e9b1447bcd22b295ef2f6d204fa7762f71f5fa0edfdbabd8a0aa1c14666279048227a0fb6d9fa705f2e78628b2ba0c5b09b3b1e297909da683b526a0a8df0861aa15d4e7fc528e773b91600700578423c72732c60a9a3e04cc640e4c7fb2292aa75bc2d51f9863a44dc4ae1bdbd1bee22e501a6882889d3998f299732935ad9fc193da99ddb264464dc1411cfbd799918529ae623f65c0dd770fdb1af9c6570279a7c2c5160d20ea3d3c6db82929e1d402df88e6a5c3683666
  - the author refers to "this dataset" (his stage-2 outputs) without a URL
  - page tags: Advanced, Multiclass Classification, Intermediate
- **Lesson:** Persist per-model OOF and test matrices with their log loss in the filename and your second stage becomes a
  one-cell experiment — but if you publish no numbers, nobody can tell which of your stages earned 13th.

## TPSMAY21 — consensus recipe
- **Architecture distribution:** 1st `STACK2` (5 families × 3 seeds × 5-fold OOF → Ridge meta in CalibratedClassifierCV, 5-fold
  stacking, probabilities clipped) · 3rd `FLAT` (hand-picked 3 own submissions blended; blend math not stated) · 13th `FLAT`
  (stage-1 LightGBM OOF/test matrices merged with metric-optimized linear weights).
  - **winner topology: `STACK2`** — the only trained meta-learner on an OOF matrix in this set belongs to the winner
    (TPSMAY21-01); 13th's "linear weights ... optimized with a metric" is weight selection, i.e. `FLAT` per ruling 1, and 3rd fits
    nothing at all.
  - Read-across: the winner's two-stage stack beat 3rd's 3-file hand-picked blend by only 0.00006 on private (1.08763 vs 1.08769) —
    on this board the architecture gap bought a medal, not score (TPSMAY21-01, TPSMAY21-03).
- **New architectures at the board:** none. `novel=none` on all three pages (TPSMAY21-01, TPSMAY21-03, TPSMAY21-13), each checked
  against its own text.
  - The winner used no neural net at all; a commenter flags this explicitly ("first place without denoising encoders", TPSMAY21-01
    comments) and the author jokes that the DAE crowd got bored of the series. The only other NN-shaped mentions are counterfactual
    or undescribed: 13th "I didn't use autoencoder but i should did" (TPSMAY21-13) and 3rd's member label "AutoLightgbm + DAE"
    (TPSMAY21-03, never elaborated).
  - No page in this set names a transformer, TabNet, NODE, TabPFN or any later-season tabular-NN family; the vendor-managed stack
    appears once and only as one level-0 member (LightAutoML, TPSMAY21-03), which is not `AUTOML` per ruling 6.
- **Agreed on (3 of 3 — TPSMAY21-01, TPSMAY21-03, TPSMAY21-13):** LightGBM is present in every solution — one of 1st's five base
  families, two of 3rd's three members (AutoLightgbm+DAE, LightGBM+Optuna), and 13th's entire stage 1.
- **Agreed on (3 of 3 — TPSMAY21-01, TPSMAY21-03, TPSMAY21-13):** no public/community OOF or "pretrained" submission files in the
  blend pool. 1st: the automl results "did not help my local cv so I did not use them at the end" (and he quantifies the blend he
  declined: 1.08742 vs his 1.08763); 3rd refuses the "Pretrained Models" files "from the beginning to the end of this competition";
  13th's pool is only his own saved `.npy` matrices (he names no shared predictions).
- **Agreed on (3 of 3 — TPSMAY21-01, TPSMAY21-03, TPSMAY21-13):** Optuna-assisted tuning is in every pipeline. 1st: "I used Optuna
  to lightly tune each model"; 3rd names "one Lightgbm with Optuna"; 13th's `get_params`/`study.optimize` is Optuna.
- **Agreed on (2 of 3 — TPSMAY21-01, TPSMAY21-13):** CV beats LB as the decision signal. 1st: "what worked at the end of was
  trusting my local cv results and not using public lb"; 13th: "Finally the cv is the most important thing". (3rd is silent.)
- **Divergences:**
  - The meta layer splits the board: 1st trains a Ridge meta inside CalibratedClassifierCV on OOF, 3rd hand-picks 3 files on a
    Class_2-probability heuristic, 13th optimizes linear weights against the metric — and only 1st publishes the numbers his
    architecture produced (1.08564/1.08763).
  - FE: 1st tried PCA/clustering/featuretools and rejected all of it ("kept all the features"); 13th ships the `num_encode` helper
    plus "diversity of datas" preprocessing; 3rd is silent on features entirely.
  - Validation disclosure is the inverse of rank-spread: 13th shows the StratifiedKFold code but publishes no score at all, while
    1st discloses 5-fold stratified CV × 3 seeds plus full Colab code and publishes 1.08564/1.08763 (TPSMAY21-13, TPSMAY21-01).
- **Highest-leverage single trick:** clip the probabilities to [0.05, 0.95] before log loss (TPSMAY21-01) — named as part of the
  winning submission itself, and the only post-processing step on this board; runner-up mechanics (stacking > weighted average) is
  claimed without a number.
- **Nothing worked:** every FE attempt by the winner (PCA, clustering, featuretools — "horrible overfitting or poor local cv"), class
  weighting on LogisticRegression, and weighted-average blending instead of stacking, all TPSMAY21-01, none with numbers.
- **Nothing worked (reported as an omission, not a test):** the autoencoder 13th says he should have used (TPSMAY21-13); 3rd's
  refusal of pretrained files was a principled skip, not a measured loss (TPSMAY21-03).
- Beyond that, nothing else is reported as failed on this board: neither TPSMAY21-03 nor TPSMAY21-13 names a dead end with a number,
  so the "nothing worked" list is short on purpose rather than complete.
