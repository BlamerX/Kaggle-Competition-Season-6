## TPSJAN21 — Tabular Playground Series - Jan 2021
- Task: Tabular (Regression) | Metric: RMSE (lower is better) | Problem: Predict loss associated with loan defaults
- Kaggle display title: "Tabular Playground Series - Jan 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-jan-2021
- Writeups covered: 5 of 5
- Score ladder (RMSE, pub/priv, as reported by the authors): 1st `0.69530/0.69381` (winning ridge stack; CV 0.69289) ·
  1st alt `0.69620/0.69472` (NNs-only averaged stack) · 2nd `not stated/not stated` · 3rd `0.700X single-NN cv/lb, stack not stated` ·
  4th `0.69579/0.69500` (blend) · 11th `not stated/not stated`
- Earliest competition in the library. Only three of five pages publish any RMSE number; 1st and 4th give full pub/priv,
  2nd/3rd/11th give rank only. No page states rows/cols or split sizes.

### TPSJAN21-01 · 1st · danzel (Daniel, springmanndaniel) · LB 0.69530/0.69381 · CV 0.69289 (winning ridge stack)

- **Link:** https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/danzel-1st-place-solution
- **Status:** FETCHED. Plain `/competitions/` form returned a 174-byte empty-body shell; recovered full body + all 33 comments
  (15,714 bytes) with `-H "X-No-Cache: true" -H "X-Engine: browser" -H "X-Timeout: 60"` (ladder step 2b).
- **TL;DR:** Denoising-Autoencoder (DAE) transform turns the whole dataset into a new feature space; heavily-regularized MLPs on
  DAE data + LightGBM models, combined by a ridge stack — the first place win on a dataset everyone assumed was trees-only.
- **Architecture:** STACK2 · stages=2 · l1=multiple DAE-MLPs + some LightGBM models · l2=ridge regression (+ simple averaging) ·
  novel=none · mod=none ·
  topo=train+test → level-1 DAE (swap noise) → its weights as a new dataset → regularized MLPs (+ LightGBM) → ridge stack + averaging → submission
  - The submitted prediction is the ridge-stacked ensemble of DAE-MLPs and LightGBM, not a single member.
  - The DAE is a *feature extractor*, not the submitted model: both train and test are pushed through a first-level denoising
    autoencoder and its learned weights are reused as the input dataset for a second-level NN (author's diagram). So DAE is not
    counted as a stack level and is not tagged `novel` (rule: an autoencoder feeding a downstream model is a forward transform).
  - LightGBM branch is a set of plain tree models (author: "based on slightly adjusted @kailex params").

- **Setup:** rows/cols, split sizes, submission slots: not stated. CV described as "local 10 fold cv".
- Structural quirk exploited: "the model's weights will include a lot of feature information (no more feature engineering needed)"
  — the DAE is used as an automatic feature transformer.
- **Features:** none hand-engineered for the DAE-MLP branch — the DAE representation *replaces* feature engineering (author's words).
  Original tabular columns are fed in raw to the DAE; noise is injected during DAE training (row-wise / column-wise / random,
  batch-wise); exact noise-level, layer widths, encoder/decoder sizes: not stated in this writeup (deferred to the linked
  "turn your data into DAETa" notebook).
- LightGBM uses the plain original features (params adapted from kailex).
- **Models:** MLP = "Baseline NN (simple MLP with keras)", "(dense layer + l2 reg + dropout) x2", "nothing special here" — the
  author's own earlier phrasing quoted in a comment; heavily regularized to fit DAE-transformed data.
- Denoising Autoencoder: architecture, layers, noise level, epochs: not stated (only "I spent most of the time on DAE training/validation").
- LightGBM: exact hyperparameters not stated; credited to kailex's public params.
- Number of DAE-MLPs and of LightGBM models in the stack: not stated ("multiple DAE-MLPs and some lightgbm models").
- **CV:** 10-fold local CV. Winning ridge stack CV 0.69289 -> public 0.69530 (+0.00241) -> private 0.69381 (+0.00092 above CV).
- All-NNs-averaged-only stack: CV 0.693773 -> public 0.69620 -> private 0.69472.
- Seeds: not stated.
- **Ensembling:** ridge regression stack over the DAE-MLP and LightGBM OOF predictions, plus simple averaging; ridge params not stated.
- Shared-OOF usage: not stated.
- **Post-processing:** not stated.
- **Gains:** winning stack private 0.69381 vs the NNs-only averaged stack private 0.69472 = -0.00091 RMSE from adding the LightGBM
  branch and the ridge combination (author's two published ensembles).
- Public-vs-private: private improved over public by -0.00149 (0.69530 -> 0.69381), i.e. the final private was his best of the set.
- (unquantified) the DAE representation is credited as the whole reason NNs beat trees here.
- **Failed:** none named in the body. (From comments: two other competitors — 3rd Fatih and 81st kailex — say their own DAE attempts
  failed; kailex "tried to use DAE but without success." That is *their* result, not this author's.)
- **Comments:** all technical content lives in author replies, mined here:
  - Author to @l0glikelihood (7th): "I'm curious now - any success in training a tree booster with this type of data?" — DAE features
    were fed to NNs; whether the LightGBM used DAE features is not stated.
  - Author to @fatihozturk (3rd): "adding noise is crucial and it needs some experience to apply it correctly."
  - Author to @adityaecdrid on **stacked DAEs**: "Let's say you have 3 hidden layers. The final 'stack' is the 3 layers combined
    'columnwise'." On noise: "define some noise-level and replace parts of the input data with sampled data … (rowwise, colwise, random)."
  - Author to @gunesevitan on batch-wise noise: "Doing it batch-wise makes the noise distribution more diverse and it is more memory
    friendly … Your goal should be to show your model as much noise as possible."
  - @fatihozturk (3rd, top commenter): his own DAE failed because "I completely missed the noise adding part and instead just run the
    dae model for a few epochs and used its output as a new dataset. It only added diversity to my final blending a bit."
  - @kailex (81st, the LightGBM-param source): "I tried to use DAE but without success."
  - The linked external writeup is referenced for DAE config detail (6.2.1 column-wise swap noise: "noise only 20% of the batch by
    touching only full columns") — that text is on the *linked notebook*, quoted here by a commenter, not in this page body.
- **Compute:** not stated (no wall-clock, GPU/CPU, RAM or cost).
- **Artifacts:** (author-cited, verbatim; not fetched)
  - https://www.kaggle.com/springmanndaniel/dae-representation/ (custom DAE-representation dataset, shared for others to test)
  - https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta (author's longer solution writeup)
  - https://www.kaggle.com/kailex (params credited, tagged without a notebook URL)
  - figure: https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F606532%2Fce663fd11930c8a41648525a20b2cc96%2Fdeepstack_input_output_color.png?generation=1724062680729078&alt=media
  - https://www.kaggle.com/springmanndaniel (author) · https://www.kaggle.com/fatihozturk · https://www.kaggle.com/erickeniukews ·
    https://www.kaggle.com/l0glikelihood · https://www.kaggle.com/adityaecdrid · https://www.kaggle.com/gunesevitan (commenter profiles)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/danzel-1st-place-solution (citation line)
- **Lesson:** A well-tuned DAE (with proper noise injection) turns tabular data into a feature space where a plain regularized MLP can
  beat every GBM — but the win is the feature transform, not the network depth.

### TPSJAN21-02 · 2nd · Ren (ryanzhang) · LB not stated/not stated · CV 0.698–0.702 (his NN only)

- **Link:** https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/ren-2nd-solution-write-up
- **Status:** FETCHED. Plain `/competitions/` form returned a 178-byte empty-body shell; recovered full body + all 5 comments
  (6,757 bytes) with `X-No-Cache: true` + `X-Engine: browser` + `X-Timeout: 60` (ladder step 2b).
- **TL;DR:** A "boring" month-long pool of 20 GBM/linear/NN base models saved as OOF, stacked by ridge regression; the interesting
  part is a swap-noise MLP fed sklearn RandomTreesEmbedding features ("dwarf DAE") that he got down to 0.698 CV.
- **Architecture:** STACK2 · stages=2 · l1=~23 base models (10 GBT + 2 SVM + 1 KNN + 2 Ridge + 1 RF + 1 RegularizedGreedyForest + ~6 NNs) ·
  l2=ridge regression · novel=none · mod=none ·
  topo= (trees/SVM/KNN/Ridge/RF/RGF + swap-noise MLPs trained on original+RTE features) → OOF → ridge stack → submission
  - Submitted prediction is the ridge regression stacking "all the above"; the winning structure is the stack, not any one member.
  - The MLP branch is a CASCADE-within-a-member: sklearn `RandomTreesEmbedding` features are horizontally stacked onto the original
    inputs and swap-noise is added only to the original inputs; this is a feature transform (leaf one-hots), not a meta level, so it
    does not make this `STACKN`, and `RandomTreesEmbedding` is a stock sklearn ensemble so `novel=none`.
- **Setup:** rows/cols, split sizes, submission slots: not stated. Code written in 2 hours, then one spare home machine ran it "for the
  entire month".
- **Features:** RandomTreesEmbedding leaf-encoding of the original columns (his "dwarf version of DAE"), concatenated with the raw
  inputs for the NN branch; trees/linear models use the plain features. No manual feature formulas; no dropped columns named.
- **Models:** level-0 pool, verbatim counts: 3 LightGBM, 4 XGBoost, 3 NGBoost (10 GBT), 2 SVM, 1 KNN, 2 Ridge, 1 Random Forest,
  1 Regularized Greedy Forest, plus ~6 MLP/Keras NNs "with slight differences". Hyperparameters for the tree/linear models: not stated
  ("tweak my old competition code"); NN uses a `add_swap_noise_torch(X, ratio=.15, col_to_apply=[])` routine.
- **CV:** NN branch: 0.702 CV with swap-noise MLP alone; 0.698 CV after adding RandomTreesEmbedding features to the inputs (author's own
  figures, and he links discussion 208429 for it). Final stack CV and public/private LB: not stated. Seeds/folds: not stated.
- **Ensembling:** single ridge regression fitted on the OOF predictions of all ~23 base models; ridge params not stated.
- Shared-OOF usage: none — all base models are his own.
- **Post-processing:** not stated.
- **Gains:** (0.702 -> 0.698 CV, -0.004 RMSE) the NN improving from swap-noise-only to swap-noise + RandomTreesEmbedding features.
- (unquantified) swap noise "did helped" the NN vs no noise.
- **Failed:** DAE itself — "I tried out DAE briefly, but not able to get it working" — so he substituted RandomTreesEmbedding as a
  cheaper stand-in. He also states he has "a feeling that each input variable is a composite of multiple components" but ran out of
  time to model that properly.
- **Comments:** technical content is only in one commenter, @gunesevitan (123rd): "I was able get my nn closer to 0.7 with
  RandomTreesEmbedding + entity embeddings of components extracted gaussian mixture model. I guess I was missing the noise part."
  Author states he will not share more code ("an exhausted dad's code at 2 AM is not readable").
- **Compute:** one spare home machine running for a full month; wall-clock, GPU/RAM not stated.
- **Artifacts:**
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html (the "dwarf DAE" he used)
  - https://www.kaggle.com/ryanzhang/ridge-on-random-tree-features (his shared Ridge-on-RTE notebook)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/208429#1161897 (where he noted the 0.702 NN CV)
  - https://www.kaggle.com/springmanndaniel (1st, tagged)
  - https://www.kaggle.com/ryanzhang (author)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/ren-2nd-solution-write-up (citation line)
- **Lesson:** When a DAE is too costly to tune, sklearn RandomTreesEmbedding + swap-noise-augmented inputs is a cheap way to drag an MLP
  close enough to trees to earn its slot in a ridge stack.

### TPSJAN21-03 · 3rd · Fatih Öztürk (fatihozturk) · LB 0.700X (single NN)/not stated · CV not stated for the stack

- **Link:** https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/fatih-3rd-place-solution
- **Status:** FETCHED on the first try (plain `/competitions/` form, `X-Return-Format: markdown`, 8,428 bytes, body + 11 comments).
- **TL;DR:** Four model families — plain LGBM, a two-stage threshold classifier/regressor LGBM, a DAE-augmented LGBM, and an
  MLP+Embedding — stacked by linear regression, then squeezed with non-linear interaction features between the OOF columns.
- **Architecture:** STACK2 · stages=2 · l1=4 model families (plain lgbm, 2-stage threshold lgbm, DAE-augmented lgbm, MLP+embedding) ·
  l2=linear regression stack over OOF (+ non-linear interaction features between OOF columns) · novel=MLP+embedding · mod=none ·
  topo=4 model families → OOF → [add OOF-OOF non-linear interaction features] → LinearRegression stack → submission
  - Submitted prediction is the linear-regression stack over saved OOF predictions ("much better than manual blending"), not a member.
  - `novel=MLP+embedding` names only one of the four level-1 members: an MLP with Embedding layers on categorical features crafted from
    the original continuous columns, worth 0.700X CV/LB alone. The DAE in model 3 is a feature/data-augmentation extractor, not the
    submitted model, so it is not tagged novel.
  - The 2-stage lgbm (model 2) is an internal classifier-then-two-regressors composition inside one member, not a stack level.
- **Setup:** rows/cols, split sizes, submission slots: not stated. Single NN reaches "0.700X cv/lb".
- **Features:** model 4: categorical features crafted from the original continuous columns (binning into categorical ids) so embedding
  layers can be added along with the original inputs. Model 3: DAE output used to *augment the training data* during CV (rows added).
  Model 1 uses plain features. Final stack adds non-linear interaction features between the OOF predictions themselves.
- **Models:** 1) regular LightGBM on plain features. 2) two-stage LightGBM: threshold the target into a binary column (e.g. threshold=8),
  train a stage-1 classifier for P(class=1), train two stage-2 regressors (target > threshold, and target < threshold), combine as
  `final = p0*pred_0 + p1*pred_1`; varying the threshold mints many diverse models. 3) regular LightGBM on the DAE-augmented training set.
  4) MLP with Embedding layers (keras; 0.700X single-model CV/LB). Exact hyperparameters: not stated in the body (in the linked kernels).
- **CV:** single NN 0.700X CV; final stack CV: not stated. Splitter/folds/seeds: not stated.
- **Ensembling:** linear regression on top of the OOF predictions of each saved model; plus non-linear interaction features between the
  OOF columns as "a very final squeezing step" that improved CV and LB "in 4th-5th decimals". Shared-OOF usage: none (all his own models).
- **Post-processing:** not stated beyond the 2-stage threshold combination (which is part of member 2, not post-hoc).
- **Gains:** (+0.700X from a single NN) MLP+embedding on crafted categoricals — "improved my final stacking quite well".
- (4th-5th decimal, both CV and LB) adding non-linear interaction features between OOF predictions in the stacker.
- (stated as the largest stack contributor) the 2-stage varying-threshold LGBM models — "those model predictions coming from different
  thresholds were the main contributor to my final stacking" (author comment).
- **Failed:** DAE — "my dae was far from being successful as 1st place's dae solution"; using its output to augment the LGBM training
  data "ended up improving final stacking well enough" but far weaker than the winner's DAE. He also says he "completely missed the noise
  adding part" (his own comment on 1st's page).
- **Comments:** author reply to @springmanndaniel (1st): "those model predictions coming from different thresholds were the main
  contributor to my final stacking :)". @jamesd5 (320th): "Enjoyed reading your approach to the binary classification as predictive feature."
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/fatihozturk/nn-with-embedding-part-of-3rd-place-solution?scriptVersionId=53252723 (NN kernel)
  - https://www.kaggle.com/fatihozturk/models-stacking-3rd-place-solution?scriptVersionId=53266247 (LGBM models + stacking)
  - https://www.kaggle.com/springmanndaniel (1st, tagged)
  - https://www.kaggle.com/fatihozturk (author) · https://www.kaggle.com/hamzaghanmi (11th commenter) ·
    https://www.kaggle.com/jamesd5 · https://www.kaggle.com/ashokkumarbibbab/twitter-trump-insult (unrelated comment link)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/fatih-3rd-place-solution (citation line)
- **Lesson:** On a loss-regression target, minting diverse members by sweeping the binarization threshold of a two-stage LGBM beats trying
  to make every base model individually strong — feed them all into a plain linear stack.

### TPSJAN21-04 · 4th · Dave E (davidedwards1) · LB 0.69579/0.69500 · CV slightly below LB (not stated numerically)

- **Link:** https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/dave-e-4-lb-notebooks
- **Status:** FETCHED on the first try (plain `/competitions/` form, `X-Return-Format: markdown`, 9,716 bytes, body + 3 comments).
- **TL;DR:** A fixed-weight blend of 2 Keras NNs + LGBM + XGB; the NNs are the interesting part — GaussianMixture sub-distribution
  split features, optuna-sized MLP, last-5-epoch TTA averaging over 4 seeds x 10 folds.
- **Architecture:** FLAT · stages=1 · l1=4 models (LightGBM, XGBoost, Keras NN1, Keras NN2) · l2=none (fixed weighted average, no meta-learner) ·
  novel=none · mod=none ·
  topo= GMM-split features → 2 Keras MLPs (4 seeds x 10 folds, last-5-epoch avg) + LGBM + XGB → weighted average → submission
  - Submitted prediction is a fixed weighted average of the four members: LGBM 0.374224, XGB 0.161811, Keras NN1 0.243573, Keras NN2
    0.220393 — that is `FLAT` (rule 1: weight selection, not a meta-learner fitted on the OOF matrix).
  - Seed and epoch averaging (4 random seeds x 10 folds, weights from the last 5 epochs re-predicted and averaged) happens inside each
    NN, so it does not add a stage.
- **Setup:** one submission only ("One submission which was the best score I had (CV / public LB)"; turned out to have his best private).
  Rows/cols and split sizes: not stated.
- **Features:** for the NNs: sklearn GaussianMixture splits each original feature into sub-distributions ("worked better than kmeans");
  each sub-distribution becomes 2 columns (a value = original data, a label = 1/0). More splits kept improving CV, so the NN used "a large
  number of additional feature columns". Inputs rescaled and target reset around zero. LGBM/XGB use public parameters with his extra
  features (a copy error, so their scores may not match other notebooks with the same params).
- **Models:** LightGBM and XGBoost with public parameters. NN1 (more features): public 0.69753 / private 0.69724. NN2 (fewer features,
  smaller train/valid gap): public 0.69809 / private 0.69747. LGBM: public 0.69694 / private 0.69585. XGB: public 0.69870 / private 0.69766.
  Keras/tensorflow MLP, ~5-6 layers with dropout and dense regularisation from an early optuna run; pytorch attempted and abandoned.
- **CV:** 10 folds, 4 random seeds per NN; saved weights from last 5 epochs and averaged their re-predictions (TTA), which "made the CV (on
  paper at least) lower". He performed some groupings across folds, so "my CV estimate came out a bit lower than LB". Final blend public
  0.69579 / private 0.69500.
- **Ensembling:** fixed weighted average with the four weights above; blend score public 0.69579 / private 0.69500. NN2 "added just a small
  amount to the blended LB score". Shared-OOF usage: none named (but LGBM/XGB parameters are public-notebook derived).
- **Post-processing:** target reset around zero and input rescaling are pre-processing; no threshold/clipping/rounding stated.
- **Gains:** blend private 0.69500 beats his best single member (LGBM 0.69585 private) by -0.00085 RMSE; the two GMM-featured NNs are what
  push it under the tree baseline ("it made quite a large difference to LB position by providing a mix from a non-tree model").
- NN2 over NN1: -0.00023 private (0.69747 vs 0.69724 is NN1 better; NN2 is worse alone but adds blend diversity).
- **Failed:** Ridge regression on the NN GMM-split features "did a lot better than with original features (down to 0.709 or so)" but he
  "couldn't get any positive effect based on my blending calculation" — a strong-ish solo model that contributed nothing to the blend.
  PyTorch NN attempted: results "just not looking comparable" to Keras/tf, so he left it. Optuna run was early and never re-tuned against
  the final feature inputs. Decreasing batch size "seemed to cause it to learn more steadily but end result didn't seem to look too much different".
- **Comments:** @springmanndaniel (1st): "You scared the hell out of me during the last day - coming closer and closer." Author replies the
  blend surprised him ("I was a bit surprised myself when I first submitted the blended submission"). Note: "Changing the activation on the
  output layer is a nice touch" is the author complimenting **danzel's** solution, not describing his own — do not read it as this entry's architecture.
- **Compute:** not stated (no wall-clock, GPU/CPU, cost).
- **Artifacts:**
  - https://www.kaggle.com/davidedwards1/jan21-tabular-playground-4-lb-final-blend (blend notebook)
  - https://www.kaggle.com/davidedwards1/jan21-tabplayground-nn-final-more-features (NN1)
  - https://www.kaggle.com/davidedwards1/jan21-tabplayground-nn-final-fewer-features (NN2)
  - https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-inference-tta (where he picked up last-epochs/TTA averaging)
  - https://www.kaggle.com/hamzaghanmi/xgboost-hyperparameter-tuning-using-optuna (public XGB params source)
  - https://www.kaggle.com/hamditarek/tabular-playground-series-xgboost-lightgbm (public LGBM/XGB params source)
  - https://www.kaggle.com/springmanndaniel (1st, tagged) · https://www.kaggle.com/davidedwards1 (author)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/dave-e-4-lb-notebooks (citation line)
- **Lesson:** Splitting continuous features into GaussianMixture sub-distributions (value + indicator columns) is enough to make a plain MLP
  competitive with trees — but only worth it inside a blend; a solo Ridge on those features added nothing.

### TPSJAN21-11 · 11th · Hamza (hamzaghanmi) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/hamza-11th-solution
- **Status:** FETCHED via the browser escape hatch. The `/competitions/` writeups form and the `/c/` form each returned only a
  2,800-2,900 byte navigation shell even with `X-No-Cache: true` + `X-Engine: browser` + `X-Timeout: 90`; recovered the real body with
  `mcp__browser-use__navigate_page` + `take_snapshot`. The writeup itself is genuinely minimal (a two-paragraph note + one notebook link) and
  the snapshot shows a "0 Comments" heading.
- **TL;DR:** A short learning-oriented writeup: two regression GBMs (CatBoostRegressor and LGBMRegressor), some feature scaling /
  transformation / selection, results and details live entirely in the linked notebook, not in the post.
- **Architecture:** not stated (proof: the page names two regressors — catboostRegressor and lgbmRegressor — and links a notebook, but never
  states whether the submitted prediction is one model, a weighted blend, or a stack; no weights, no meta-learner, no combination method,
  no CV/LB number is given anywhere on the page, so the notebook title alone cannot separate FLAT from STACK2 or SINGLE) ·
  stages=not stated · l1=not stated (2 regressor families named, count unknown) · l2=not stated · novel=none · mod=none · topo=not stated
  - No submission topology is claimed by the author; the only architecture facts on the page are "regression algorithms like catboostRegressor
    and lgbmRegressor". Anything further is in the linked notebook, which this digest does not fetch (rule 5).
  - `novel=none`: CatBoost and LightGBM are stock GBM regressors.
- **Setup:** rows/cols, split sizes, submission slots: not stated. This is the author's second Kaggle competition after "MOA prediction".
- **Features:** technique categories only — "features scaling, transformation selection"; exact engineered features, encodings, dropped columns:
  not stated (deferred to the notebook).
- **Models:** CatBoostRegressor and LightGBMRegressor ("first time I used regression algorithms like catboostRegressor and lgbmRegressor");
  library versions, hyperparameters, seeds, fold count: not stated.
- **CV:** splitter, folds, repeats, seeds, CV-vs-LB gap: not stated. Author publishes no CV or LB number.
- **Ensembling:** not stated — the page neither describes nor numbers any blend/stack.
- **Post-processing:** not stated.
- **Gains:** NOT STATED — no quantified delta of any kind appears on the page.
- **Failed:** not stated.
- **Comments:** nothing technical — the page snapshot reads "0 Comments"; no replies exist.
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/hamzaghanmi/11th-solution (his notebook — the only substantive content; not fetched per rule 5)
  - https://www.kaggle.com/hamzaghanmi (author)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/writeups/hamza-11th-solution (citation line)
- **Lesson:** A top-12 finish here with no published numbers means the whole solution is in the notebook; this entry is a pointer, not a
  replicable recipe.

## TPSJAN21 — consensus recipe
- **Architecture distribution:** 1st `STACK2` (ridge stack over DAE-MLPs + LightGBM; DAE is a feature extractor) ·
  2nd `STACK2` (ridge stack over ~23 own base models incl. swap-noise MLPs with RandomTreesEmbedding features) ·
  3rd `STACK2` (linear-regression stack over 4 model families incl. MLP+embedding, + OOF interaction features) ·
  4th `FLAT` (fixed weighted average of 2 Keras NNs + LGBM + XGB, no meta-learner) ·
  11th `not stated` (names CatBoost + LGBM regressors, no combination method or number on the page).
  - **winner topology: `STACK2`** — 1st submitted a ridge stack; ranks 1-3 all stacked on OOF with a linear meta, 4th used a plain weighted
    blend, 11th states nothing.
- **New architectures at the board:** exactly one listed novel architecture appears — `MLP+embedding` at 3rd (one of his four level-1
  members; 0.700X single-model CV/LB; not the submitted model, which is the linear stack).
  - No modern tabular transformer anywhere (this is 2021): no TabM / FT-Transformer / NODE / TabPFN.
  - The genuinely *notable* 2021 technique was the Denoising Autoencoder, but it is a feature extractor not a submitted model, so it is not
    tagged novel: it carried 1st (winning pipeline), was attempted-and-failed at 2nd ("not able to get it working"), and used only for data
    augmentation at 3rd ("far from being successful"). Competitors at 4th and in the 1st comments repeat that only 1st got the noise
    injection right.
  - Other non-stock parts: `RandomTreesEmbedding` (2nd, as a "dwarf DAE"), NGBoost + RegularizedGreedyForest (2nd), GaussianMixture
    sub-distribution features (4th).
- **Agreed on (5 of 5 — TPSJAN21-01, -02, -03, -04, -11):** every solution includes at least one gradient-boosting tree model
  (LightGBM/XGBoost/CatBoost/NGBoost). Even the two NN-led top finishes stack or blend trees with the NNs.
- **Agreed on (4 of 5 — -01, -02, -03, -04):** a neural network contributes to the final ensemble. 1st submits DAE-MLPs, 2nd stacks ~6
  swap-noise MLPs, 3rd adds an MLP+embedding, 4th blends 2 Keras NNs. Only 11th (-11) is trees-only. This is the through-line of the
  competition: the medalists all cracked "how to make an NN beat/match trees here."
- **Agreed on (3 of 5 — -01, -02, -03):** the submitted artifact is a linear meta-learner stack on OOF (ridge regression for 1st and 2nd,
  plain linear regression for 3rd). 4th uses fixed weights (FLAT) and 11th states no method.
- **Agreed on (4 of 5 — -01, -02, -03, -04):** the winning lever is a feature transform aimed at the NN, not hand-crafted business features —
  1st DAE representation, 2nd RandomTreesEmbedding leaf features, 3rd categorical-embeddings crafted from continuous columns, 4th
  GaussianMixture sub-distribution (value+label) columns.
- **Divergences:** 1st replaced all feature engineering with a DAE and stacked everything with ridge (private 0.69381); 4th stayed with a
  plain fixed-weight blend of 2 NNs + 2 trees and no meta (private 0.69500) — a 0.00119 RMSE gap that the linear meta + DAE explains.
  Mid-table 11th finished on two stock GBM regressors with no published combination, showing trees alone still reached top-12 but not the medals.
- **Highest-leverage single trick:** the DAE with *proper noise injection* (1st) — batch-wise, column/row/random swap noise "as much noise as
  possible" — the one thing competitors repeatedly say they missed and the reason an MLP won a supposedly trees-only regression.
- **Nothing worked:** DAE attempts without getting the noise part right (2nd "not able to get it working"; 3rd "far from successful"; kailex
  at 81st "without success"). A strong solo model that never earns its blend slot (4th's Ridge on NN features, 0.709 alone, contributed
  nothing). PyTorch at 4th (results "not comparable" to Keras, abandoned). More folds/seeds at the margin (4th "diminishing returns").
