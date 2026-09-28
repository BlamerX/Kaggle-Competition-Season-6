## TPSNOV22 — Tabular Playground Series - Nov 2022
- Task: Tabular (Regression) | Metric: RMSE | Problem: Practice regression on synthetic data
  *(index title line, reproduced as printed)*
- **Metric: ROC AUC, higher better, binary classification (recovered from the writeup pages; the index title line says
  Regression/RMSE).** Evidence on the fetched pages: TPSNOV22-07's own board numbers are 0.51430 public / 0.51892 private
  (chance level for RMSE would not be 0.5); TPSNOV22-01 names "Top-K AUC" feature selection, a sigmoid output with
  `binary_crossentropy` in the posted model code, and "models had more troubles to classify negative labels";
  TPSNOV22-03's "isotonic calibration" is a probability-calibration step. All deltas in this file use AUC sign convention
  (higher is better).
- Kaggle display title: "Tabular Playground Series - Nov 2022"
- Competition: https://www.kaggle.com/c/tabular-playground-series-nov-2022
- Writeups covered: 3 of 3
- Score ladder (as authors report it): 1st `public rank 7 / private rank 1 (no AUC printed)` · 3rd
  `public rank 172 / private rank 3 (no AUC printed)` · 7th `0.51430 (public, rank 97) / 0.51892 (private, rank 7)`
- Only structural number published anywhere in the set: the **5000 provided features** (TPSNOV22-07).
- This board is defined by the public→private collapse: all three entries report a rank swing of 6 to 169 places.

### TPSNOV22-01 · 1st (private; 7th public) · jcaliz + samuelcortinhas (Jose Cáliz, Samuel Cortinhas) · LB not stated/not stated (public rank 7 / private rank 1) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/369674
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 12.4 KB; body + 21 comments rendered, one of them
  marked "This comment has been deleted")
- **TL;DR:** A scipy.minimize-weighted blend over 5 deliberately different members (LightAutoML, XGBoost, LightGBM, 2 dense
  NNs), each trained on its own selected feature subset, with a hand-tuned ±delta push on predictions above/below 0.5 as
  post-processing.
- He placed 7th public and 1st private — the win came from CV discipline, not from the visible leaderboard.
- **Architecture:** FLAT · stages=1 · l1=5 members (1 LightAutoML output + 1 XGBoost + 1 LGBM + 2 dense NNs) · l2=none ·
  novel=none · mod=multi-view ·
  topo=feature selection (Top-K AUC / L1-LR / importance≈RFE / permutation) → 5 member sets [XGB, LGBM, 2 NN, LightAutoML]
  → logits → scipy.minimize weights → ±delta post-processing on either side of 0.5 → submission
  - Ruling 1 is what keeps this `FLAT` and not `STACK2`: "Stacking using scipy.minimize" is a *weight search* over member
    predictions, not a model fitted on the OOF matrix — there is no meta-learner to name.
  - Ruling 6 keeps LightAutoML from making this `AUTOML`: it contributes one member column, so its vendor-internal stack
    stays hidden inside level 1 ("LightAutoML provided by @sergiosaharovskiy, @alexryzhkov").
  - The members are differentiated by **feature subsets as much as by algorithm** — his own phrasing: "Blend different types
    of models, not only on on the algorithm but o the features" — which is why `multi-view` is the defining modifier here.
  - `novel=none`: the NN members are plain dense Keras MLPs (code posted in the comments, quoted under Models).
- **Setup:** rows/cols, fold sizes per member, submission slots: not stated by him (the 5000-column scale is stated only in
  TPSNOV22-07).
- Structural quirk exploited: none named; instead he names a *data-analysis void* — "there were almost zero data analysis,
  I felt it was more like shoot everything until it works".
- **Features:** no column names or formulas published; only the selection machinery:
  - Selection methods used to build the per-member feature sets: Top-K AUC; L1 with LogisticRegression; feature importance
    ("similar to RFE"); permutation importance.
  - Encodings, GP/generated features, dropped columns, feature-source kernels: not stated.
- **Models:** members of the final blend, as enumerated in the body:
  - LightAutoML (provided via @sergiosaharovskiy / @alexryzhkov kernels) — 1 member.
  - "One XGBoost models trained with different features" — 1 member.
  - "One LGBM models trained with different features" — 1 member.
  - "Two dense NN trained with different features" — 2 members.
- NN architecture (Samuel Cortinhas's comment reply, verbatim code): Keras `Sequential` →
  `Input(shape=(X_train.shape[1],))` → `CFG['depth']-1` blocks of `Dense(units=CFG['units'])` + `BatchNormalization()` +
  `Activation(CFG['activation'])` + `Dropout(rate=CFG['dropout_rate'])` → `Dense(units=1, activation='sigmoid')`, compiled
  with `Adam(lr=0.001)`, `loss='binary_crossentropy'`, `metrics=['binary_accuracy']`.
  - `CFG` values (depth, units, activation, dropout rate) are **not stated** anywhere on the page.
- NNs were run with **multistart** to mitigate training noise (his key point 3).
- Activations preferred: Swish and Mish (key point 8). Optimizer LR: 0.001 (from the posted code).
- Libraries named: XGBoost, LightGBM, LightAutoML, Keras, Weights and Biases ("among others").
- Per-model hyperparameters, seeds, library versions: not stated.
- **CV:** two fold counts are stated — "we used 10 StratifiedKfold and other models with 20 StratifiedKfold".
- #repeats, shuffle seeds, which member got 10 vs 20 folds: not stated.
- CV score: not published; no CV-vs-LB gap computable.
- Trust verdict (verbatim, key point 7): "Trust your CV its easy to overfit" — and the outcome supports it: 7th public,
  1st private.
- **Ensembling:** scipy.minimize over member predictions (credited to @pourchot's "stacking-with-scipy-minimize" kernel).
- Blend operates in **logit space**: "Use logits instead of probablities, thanks @ambrosm".
- Number of members in the final blend: 5 (enumerated above); "Lots of ensembles with different models architectures and
  different features" precede it, with no counts or scores for those earlier ensembles.
- Shared-OOF usage (author comment): "The final submission and blending was done using only OOF and submissions preds from a
  kaggle dataset" — his team's own stored OOF/submission predictions, not a community set, so no `public-oof` modifier.
- **Post-processing:** key point 4, verbatim: "Post processing by adding little values to predictions above and below 0.5.
  Always trusting the CV to select the positive and negative delta. Earlier in the competition EDA notebooks shown that
  models had more troubles to classify negative labels."
  - Tagged as `Post-processing` per ruling 2 (applied after the probabilities exist; the CV picks the sign and magnitude),
    never counted as a stage.
  - Delta magnitudes: "little values" — no number published.
- **Gains:** No AUC delta is published for any step (feature selection, stacking, multistart, post-processing).
- Rank-level gains only: the whole pipeline converted public rank 7 into private rank 1.
- Orderings he does state: members were chosen for diversity (algorithm × feature subset); 20-fold CV used for the models
  that needed it over the default 10-fold.
- **Failed:** three named dead ends, all in the "Several ideas that didn't work" section:
  - Boltzmann ensemble — "we didn't go deeper after the first couple of iterations".
  - Geometric mean — "similar to Boltzman".
  - Residual blocks in the NN — "a simple neural network was enough".
- Self-criticism recorded as a process failure: "I spent too much time trying to do things from scratch instead of reading
  others work."
- **Comments:** 21 comments (counted; one is marked deleted, plus 3 appreciation entries). Technical content:
  - Samuel Cortinhas (co-author) posts the full Keras `build_model()` function quoted under Models, in answer to
    @nurbektastan (207th) asking "Is it possible to get your NN architecture?".
  - Jose Cáliz (author) to @sainikhilesh333: no single reproducible notebook exists — "there were several versions of
    multiple notebooks (some locals) saving OOF preds and submissions… The final submission and blending was done using only
    OOF and submissions preds from a kaggle dataset."
  - @sahal_dissanayaka asks whether TCN was tried instead of LSTM; **no author reply appears on the page**, and neither
    TCN nor LSTM appears in the body, so nothing about recurrent members can be claimed here.
  - Alexander Ryzhkov (46th, whose LightAutoML kernels the winner used): "That's nice to see that our LightAutoML helped you
    to be on top"; Mikhail Kuznetsov (85th, LightAutoML author) and Laurent Pourchot (322nd, whose scipy-minimize kernel was
    credited) both reply-only congratulations.
  - No scores, weights, or feature counts appear in any comment.
- **Compute:** Weights and Biases used for experiment tracking and to "save oof preds and validate your hypothesis";
  wall-clock, GPU/CPU, RAM, Kaggle limits: not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/mikhailkuz/lightautoml-nn-happiness
  - https://www.kaggle.com/code/alexryzhkov/5k-features-not-a-problem-for-lightautoml
  - https://www.kaggle.com/code/ambrosm/tpsnov22-eda-which-makes-sense
  - https://www.kaggle.com/code/pourchot/stacking-with-scipy-minimize
  - https://www.kaggle.com/code/sergiosaharovskiy/tps-nov-2022-in-automl-we-trust
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/writeups/1st-place-solution (citation line)
  - https://www.kaggle.com/competitions/33111/images/thumbnail
  - https://www.kaggle.com/jcaliz · https://www.kaggle.com/samuelcortinhas · https://www.kaggle.com/alexryzhkov ·
    https://www.kaggle.com/sergiosaharovskiy · https://www.kaggle.com/mikhailkuz · https://www.kaggle.com/pourchot ·
    https://www.kaggle.com/leehann · https://www.kaggle.com/sainikhilesh333 · https://www.kaggle.com/flaviocavalcante ·
    https://www.kaggle.com/danushkumarv · https://www.kaggle.com/sahandissanayaka · https://www.kaggle.com/nurbektastan ·
    https://www.kaggle.com/perrypineapple · https://www.kaggle.com/ephraimx · https://www.kaggle.com/alexandershumilin ·
    https://www.kaggle.com/gazu468 · https://www.kaggle.com/cyrilbourgeois · https://www.kaggle.com/pvtrmalli ·
    https://www.kaggle.com/teckmengwong · https://www.kaggle.com/ambrosm
- **Lesson:** When the public board is noise, the win is a blend whose members differ by feature subset *and* family, weights
  found by a plain optimizer over logits, and a calibration shift chosen by CV rather than by the leaderboard.

### TPSNOV22-03 · 3rd (private; 172nd public) · guoyaobit (solo) · LB not stated/not stated (public rank 172 / private rank 3) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/370126
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 3.6 KB; body complete, both comments rendered)
- **TL;DR:** Column filtering (values outside (0,1)) + a Spearman-with-target cut at 0.42 + isotonic calibration, then
  AutoGluon does the modeling. 172nd public → 3rd private; he calls the shakeup a shock and claims no complicated trick.
- **Architecture:** AUTOML · stages=1 (as submitted; the vendor preset's internal level count is not stated) ·
  l1=1 AutoGluon-trained model/stack · l2=inside AutoGluon, not stated · novel=none · mod=none ·
  topo=drop out-of-range columns → drop columns by Spearman with target (threshold 0.42) → isotonic calibration → AutoGluon
  fit → its built-in ensemble → submission
  - `AUTOML` is the right primary per ruling 6: the **submitted** prediction is the vendor-managed stack itself, unlike
    TPSNOV22-01 where LightAutoML is only one blend member.
  - Ruling 2 applies to isotonic calibration: it is a calibration step, listed here as a pipeline step and recorded under
    Post-processing; it is not counted as a stage.
  - His own lessons line — "Calibration and model stacking is useful tricks" — is the only reference to stacking; whether a
    stack of his own sits under AutoGluon is **not stated**.
- **Setup:** rows/cols, folds, submission slots: not stated (the 5000-column scale appears only in TPSNOV22-07's page).
- Structural quirk exploited: the generator's column ranges — features "contain values out of range of (0,1)" are treated as
  invalid and dropped, i.e. the valid feature space is a subset with known support.
- **Features:** subtractive only; no engineered feature is named.
  - Step 1: "Drop features contain values out of range of (0,1)."
  - Step 2: "Drop features using spearman corrlation with ground truth. 0.42 is the threadhold." (verbatim, including his
    spelling). The page does **not** state the direction of the cut (whether columns above or below |0.42| are dropped) and
    publishes no surviving-column count — both left as not stated rather than inferred.
  - Failed FE variant (his section 4): "Drop features using ece."
- **Models:** AutoGluon — presets, time/memory limits, included families, and the internal ensemble recipe are all not
  stated; AutoGluon version not stated.
- Rejected alternative: FLAML (listed under "ideas that doesn't work").
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated. He only asserts "Trust your local CV."
- CV score: not published; CV-vs-LB gap not computable — his public position was rank 172 with no AUC printed, private rank
  3, so the swing itself (169 places) is the only validation signal on the page.
- Author's trust verdict (verbatim, lessons learned 1): "Trust your local CV."
- **Ensembling:** none hand-built — the blend is whatever AutoGluon's preset ensemble produces; weights, members, and level
  count not stated. Shared-OOF usage: none mentioned (counted).
- **Post-processing:** isotonic calibration (step 3 of his list, stated before the AutoGluon fit; which stage's outputs the
  calibrator wraps is not stated). No threshold, clipping, or rounding mentioned.
- **Gains:** No numeric delta for the two column filters, the 0.42 Spearman cut, or the calibration.
- Rank-level: 172nd public → 3rd private, which he describes as "the shakeup shocked me".
- **Failed:** two named dead ends:
  - Feature dropping by ECE (expected calibration error) — "Several ideas that doesn't work".
  - FLAML — same section, no detail on why.
- **Comments:** 2 comments (counted), nothing technical: luoyun777 posts an off-topic team-application email request
  ("I sent you a team application email (Santa2022) written in Chinese… Sorry to bother"); ErrorMan (654th) thanks him for
  the list. No author reply appears on the page.
- **Compute:** not stated (no time limit, hardware, or AutoGluon `time_features`/`time_train` settings published).
- **Artifacts:** none cited in the body; the page carries only profile/citation links:
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/writeups/guoyaobit-3rd-place-solution
    (citation line)
  - https://www.kaggle.com/competitions/33111/images/thumbnail
  - https://www.kaggle.com/guoyaobit · https://www.kaggle.com/luoyun777 · https://www.kaggle.com/jaygun84
- **Lesson:** On a board where public ranks are meaningless, a vendor stack plus two ruthless column filters can beat every
  hand-built ensemble — and can also leave you 172nd in public, so the filters must be justified locally, not visually.

### TPSNOV22-07 · 7th (private; 97th public) · mpware (solo) · LB 0.51430 (public) / 0.51892 (private) · CV not stated (5-fold, value not published)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2022/discussion/369731
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 5.9 KB; body + both comments rendered)
- **TL;DR:** Two serious feature-selection runs over the 5000 supplied columns (CatBoost LossFunctionChange, best subset =
  250; null-importance with 100 permutation runs, kept 350), then a 9-member **simple average** of LightGBM/GBDT, NN and
  CatBoost models fit on the 250/275/350-column views; the whole pipeline repeated over 5 random seeds to measure ±0.005
  instability.
- **Architecture:** FLAT · stages=1 · l1=9 members (3 LGBM/GBDT + 3 NN on 250/275/350 columns + 3 best singles: 2 LGBM/GBDT
  at 250 and 629 columns + 1 CatBoost at 275) · l2=none · novel=none · mod=multi-view ·
  topo=5000 cols → CatBoost LossFunctionChange FE + null-importance FE (100 runs) → views of 250/275/350 cols → 9 models
  (3 GBDT + 3 NN + 2 GBDT@250/629 + 1 CatBoost@275) → simple average → submission, × 5 random seeds for the whole process
  - "Finally my **submission is just a simple average**" — equal weights, single stage, so `FLAT` (no weight search, no
    meta-learner; the L2 experiments he reports as insignificant appear in `Failed`, not in the submission).
  - `multi-view` carries the architecture: the same three families are re-trained on three *column-set sizes*, so member
    diversity comes from the views, exactly as 1st describes his own blend.
  - Rule 4: the 5 random seeds wrap feature selection and modeling, and are used for variance measurement, not as a
    second level or as a separate ensemble stage.
- **Setup:** **5000 provided features** (his number, the only structural figure in this file); rows, fold sizes, submission
  slots: not stated.
- Structural quirk exploited: the feature catalogue is huge and mostly noise, so the contest is won by which 250-350 columns
  survive — not by the estimator.
- **Features:** selection only; no engineered feature is named.
  - CatBoost `select_features` based on **LossFunctionChange** ("the difference between the loss value of the model with this
    feature and without it"); sizes explored: 1000, 500, 275, 250, 200 — **best result at 250**.
  - **Null importance** feature selection with **100 runs**: shuffle the target, build the importance distribution, keep only
    columns whose real importance stands outside the shuffled null → **350 features selected**.
  - A third size, **629**, appears as the feature set of one of his two best single LGBM/GBDT models; its selection method is
    not stated.
  - Encodings, GP/generated features, dropped-column rationale beyond the two selectors: not stated.
- **Models:** families named: LightGBM/GBDT (best CV), CatBoost, TensorFlow/Keras NN ("NN was not bad").
- Final 9 members with their column-view sizes: LGBM/GBDT at 250, 275, 350; NN at 250, 275, 350; best singles LGBM/GBDT at
  250 and 629; CatBoost at 275.
- Hyperparameters: none published. His statement on tuning: "Hyperparameters grid search result was not significant."
- Seeds: 5 random seeds over the entire process; per-model seeds not stated. Library versions not stated.
- **CV:** 5 folds ("Best CV (5 folds) was for classic LightGBM/GBDT"); splitter type (stratified or not), repeats, seed list
  on the folds: not stated.
- Stability measured over the 5 full-process seeds: "It was around +- 0.005 on both CV/LB. LB shake up risk might be high
  with such variance."
- Published scores: public 0.51430 (rank 97) / private 0.51892 (rank 7); the gap private − public = **+0.00462**, within the
  ±0.005 band he measured, i.e. his public rank was noise, not signal.
- CV score value: not published, so the CV-vs-LB gap is not computable.
- Author's trust verdict (verbatim): "Trust your CV, play with random seeds to see how stable are your predictions."
- **Ensembling:** simple (equal-weight) average of the 9 listed members; nothing weighted, no rank blending, no meta.
- Shared-OOF usage: not stated. AutoGluon is discussed separately (below) as reference results only.
- **Post-processing:** tested and rejected — "Shifting predictions by -+ 0.00XX" appears in his "ideas that doesn't work"
  list. No calibration or clipping is applied in the submission.
- **Gains:** No CV or LB delta is published for the 250 vs 275 vs 350 vs 629 column views, nor for adding the NN members.
- Ordering stated: subset size ranking by CV — 250 best, then the rest of {1000, 500, 275, 250, 200} explored; family
  ranking — LightGBM/GBDT best, CatBoost next, NN "was not bad".
- His two stated score-moving levers, in order: "Features selection" first, "NN + GBM ensembled models" second.
- **Failed:** five named dead ends, all verbatim from his list:
  - "Shifting predictions by -+ 0.00XX" — the same post-processing trick that 1st reports as a key point.
  - "LOFO for FE" (leave-one-feature-out scoring).
  - "Drop features with hard samples (because they were already in my best list)".
  - "All kind of L2 models were not significant" — his level-2 attempts (stacked learners) added nothing.
  - AutoGluon, tried for the first time: "It did not help to improve score but it gave quick reference results that help to
    go ahead in the right direction."
- Hyperparameter grid search: "was not significant".
- **Comments:** 2 comments (counted), both by the topic author, both technical and both adding published artifacts:
  - "With the Null Importances FE kernel used: https://www.kaggle.com/code/mpware/tps-null-importances-fe"
  - "CatBoost FE kernel with loss improvements charts:
    https://www.kaggle.com/code/mpware/tps-catboost-fe/notebook" plus the chart figure
    https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F698363%2F8d662877397144c277625b0bb1aad1ed%2Ffe.png?generation=1669932783369922&alt=media
  - No scores or hyperparameters appear in the comments; no other user replies.
- **Compute:** not stated (no wall-clock, hardware, or per-model training time).
- **Artifacts:** (verbatim; not fetched)
  - https://catboost.ai/en/docs/concepts/python-reference_catboost_select_features (cited in the body for LossFunctionChange)
  - https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances (the null-importance method he followed)
  - https://www.kaggle.com/code/mpware/tps-null-importances-fe (his own run, from his comment)
  - https://www.kaggle.com/code/mpware/tps-catboost-fe/notebook (his own run, from his comment)
  - https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F698363%2F8d662877397144c277625b0bb1aad1ed%2Ffe.png?generation=1669932783369922&alt=media
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/writeups/mpware-7th-place-solution
    (citation line)
  - https://www.kaggle.com/competitions/33111/images/thumbnail
  - https://www.kaggle.com/mpware
- **Lesson:** On 5000 mostly-noise columns, publish a null-importance run and a LossFunctionChange run, take the 250-350
  survivors, then equal-weight three families over those views and measure ±0.005 across 5 seeds before trusting anything.

## TPSNOV22 — consensus recipe
- **Architecture distribution:** 1st `FLAT` (5 members, logits + scipy.minimize weights, LightAutoML as one member) ·
  3rd `AUTOML` (AutoGluon is the submitted stack, preceded by two column filters and isotonic calibration) · 7th `FLAT`
  (9 members over 250/275/350/629-column views, simple average).
  - **winner topology: `FLAT`** — and both `FLAT` entries put their diversity in the **feature views**, not the blender;
    the one `AUTOML` entry took 3rd private from 172nd public.
  - Read-across on the blender question: 1st's optimizer-found weights finished 1st private; 7th deliberately used a simple
    average and finished 7th; 7th separately reports "All kind of L2 models were not significant". No entrant fitted a
    meta-learner on the OOF matrix, so this board contains no `STACK2`.
- **New architectures at the board:** none. `novel=none` on all three entries — no TabM/FT-Transformer/NODE-class model is
  named on any page; the NN members are plain dense Keras MLPs (1st, code posted) and unnamed Keras nets (7th).
  - Non-stock items here are **weighting/selection machinery**, not architectures: scipy.minimize logit stacking (1st),
    CatBoost LossFunctionChange + 100-run null importance (7th), isotonic calibration (3rd).
  - Vendor stacks appear at two different levels: LightAutoML as one member (1st, so not `AUTOML` per ruling 6) versus
    AutoGluon as the submission (3rd, so `AUTOML`), with FLAML rejected outright by 3rd and AutoGluon called
    score-neutral by 7th.
- **Agreed on (3 of 3 — TPSNOV22-01, TPSNOV22-03, TPSNOV22-07):** feature **subsetting** is the primary lever, and every
  entrant prunes the supplied columns with a named rule: 1st uses Top-K AUC, L1 with LogisticRegression, importance-based
  selection (RFE-like) and permutation importance; 3rd drops out-of-(0,1)-range columns and applies a Spearman-with-target
  cut at 0.42; 7th uses CatBoost LossFunctionChange (best 250) and null importance with 100 runs (350 kept).
- **Agreed on (3 of 3 — TPSNOV22-01, TPSNOV22-03, TPSNOV22-07):** trust local CV and treat the public leaderboard as noise,
  stated almost verbatim by all three ("Trust your CV its easy to overfit" / "Trust your local CV." / "Trust your CV, play
  with random seeds to see how stable are your predictions").
  - The board proves them right: 1st was 7th public, 3rd was 172nd public, 7th was 97th public at 0.51430 and finished 7th
    private at 0.51892 (+0.00462, inside the ±0.005 spread he measured).
- **Agreed on (2 of 3 — TPSNOV22-01, TPSNOV22-07):** ensemble diversity must come from **algorithm × feature subset**, not
  from algorithm alone. 1st: "Blend different types of models, not only on on the algorithm but o the features"; 7th: three
  families instantiated on the 250/275/350 views plus two extra views (629). TPSNOV22-03 runs no hand-built ensemble, so it
  cannot count.
- **Agreed on (2 of 3 — TPSNOV22-01, TPSNOV22-03):** no AUC numbers published at all — both give ranks only, so the file
  cannot quote a single delta from them; only 7th publishes board scores (0.51430/0.51892), and no entry publishes a CV value.
- **Divergences:** the winner's most-cited key point is the one the 7th place independently tested and rejected.
  - 1st, key point 4: adding small deltas to predictions above/below 0.5, with CV picking the sign — reported as an
    improvement (magnitude not stated).
  - 7th, "ideas that doesn't work": "Shifting predictions by -+ 0.00XX".
  - Same trick, opposite verdict, and the difference is the selection instrument: 1st tunes it on 10/20-fold CV over a
    blended member set, 7th applies it to a simple average whose own CV variance is ±0.005 — larger than any delta he could
    shift by. Neither publishes a number that settles it.
- Second divergence: AutoML presets. 3rd wins a medal with AutoGluon as the submitted model; 7th says AutoGluon "did not
  help to improve score" and kept it only as a reference; 1st (TPSNOV22-01) uses LightAutoML as a blend member and calls
  "Be really open to learn new things as required, AutoML is a great example" a lesson learned. Three ranks, three verdicts.
- Third divergence: weight search vs equal weights. 1st optimizes blend weights with scipy.minimize over logits (5 members)
  and takes 1st private; 7th equal-weights 9 members and takes 7th; the gap is unmeasurable here because 1st publishes no
  score.
- Public-vs-private placement swing across these three entries (7th public → 1st private, 172nd public → 3rd private,
  97th public → 7th private) means rank, not score, is the only comparable signal on this board — which is why every
  architecture claim above is tied to a stated method rather than to a leaderboard position.
- **Highest-leverage single trick:** run a **null-importance / permutation-null** selection over the 5000 columns and take
  the survivors (TPSNOV22-07: 100 shuffled-target runs → 350 kept, CatBoost LossFunctionChange → best subset
  250) — it is the only published *procedure* on this board that is reproducible without the author's code, and 1st's four
  selection methods (Top-K AUC, L1-LR, RFE-like importance, permutation importance) are the same idea at smaller cost.
- **Nothing worked:** the negative knowledge is *not* corroborated across authors on this board — each entry's dead ends are
  its own, and the format's "multiple authors" bar is not met by any of them:
  Boltzmann ensemble, geometric mean, residual NN blocks (1st); ECE-based feature dropping, FLAML (3rd); prediction
  shifting, LOFO FE, dropping hard-sample features, all L2 models, AutoGluon gain, hyperparameter grid search (7th).
- **Nothing worked (the one item two entries share, stated as a caveat not a finding):** exotic blending. 1st abandoned
  Boltzmann and geometric-mean blends after a couple of iterations; 7th found "all kind of L2 models" insignificant — but 1st
  still won with an optimizer-found blend, so the shared lesson is "the blend type is not the lever; the column subset is".
