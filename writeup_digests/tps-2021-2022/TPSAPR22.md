## TPSAPR22 — Tabular Playground Series - Apr 2022
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Apr 2022" / "Practice your ML skills on this approachable dataset!"
- Competition: https://www.kaggle.com/c/tabular-playground-series-apr-2022
- Writeups covered: 6 of 6
- Metric as reported on the pages: **ROC AUC (HIGHER better)** — confirmed in-body (02 "the variation between each
  fold's output in terms of AUC", 03 "The extra percentage of AUC", 01 "I tested the predictive model CV"). Task is
  sequence classification: each row is a group of 60 time steps × 13 sensor columns with a `subject` identifier
  (06 states the (n, 60, 13, 16) shape after projection; 02 reshapes to (-1, 60, features)).
- Score ladder: 1st `not stated (public)/0.99249` CV 0.9918 · 2nd `0.987/0.989` for its best single model, stack CV 0.99,
  stack LB not stated · 3rd `0.99037 (public, stacked)` from `0.98052/0.97706` pre-stack · 4th `not stated/not stated` ·
  5th `0.98248/0.97816` (its LGBM member; blend score not stated) · 6th `0.985/0.9839` (its GRU member) → blend `not
  stated/0.98797`
- Note on 1st's numbers: the author edits his own post — "apologies, in the initial version of this topic I incorrectly
  copied and pasted public LB model scores instead of private" — so 0.99249, 0.99134, 0.98668, 0.98961 are **private**.

### TPSAPR22-01 · 1st · davidedwards1 (Dave E) · LB not stated/0.99249 · CV 0.9918

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322259
- **Status:** FETCHED (1 attempt, `/c/.../discussion/322259` + `X-Return-Format: markdown`, 13,817 bytes, body + 8 comments
  + 1 appreciation comment)
- **TL;DR:** A swap-noise denoising autoencoder (2-layer LSTM) trained on train **and** test, whose layer outputs become
  features for a second predictive network; the submission is that family plus LGBM/TF runs blended with ElasticNet weights.
- Best single DAE pipeline 0.98999 CV / 0.99134 private; the full blend 0.9918 CV / 0.99249 private.
- **Architecture:** STACK2 · stages=3 · l1=4 main PyTorch DAE pipelines + 2 TensorFlow(TPU) predictive models +
  LGBM/XGB/RandomForest/ExtraTrees · l2=ElasticNet (fitted to produce blend weights; many inputs zeroed) ·
  novel=Denoising-Autoencoder · mod=embed ·
  topo=scaled raw sensors → LSTM-DAE (swap noise, trained on full train+test) → selected AE layers as features →
  predictive NN (PyTorch GPU / TF TPU) + LGBM/XGB/RF/ET → ElasticNet-weighted blend
  - Stage 1 (DAE) never predicts the target: it is trained to detect where swap noise was injected, and its **internal
    layer outputs** are consumed as input features by stage 2 → the DAE→predictive-NN path is a forward cascade
    (rule 5: embeddings used as features = cascade), while the submitted artifact is stage 3.
  - Stage 3 is a real meta-learner, not weight hill-climbing: ElasticNet is fitted to assign model weights, so the
    primary tag is `STACK2` (one meta layer), not `FLAT`.
  - The DAE is not new to 2022 — the author says the idea is "all contained in the topics" of TPS Jan/Feb/Mar 2021 and a
    Porto Seguro thread, and links his own March-21 DAE starter notebook; the `novel` slot records the architecture, not
    its novelty.
- **Setup:** this page states only "each sample is a sequence with multiple sensors" and names sensors/steps in his noise
  examples (`Seq1-Sensor00-Step02`, `Seq1-Sensor00-Step00` swapped with `Seq1-Sensor00-Step50`); row/column counts are not
  stated here (the 60-step × 13-sensor shape is stated on entries 06 and 02, not on this page).
- Split: **10 folds, every `subject` fully contained in a single fold** (group split on subject).
- Structural quirk exploited: the DAE is trained transductively on **train + test** data ("using both the train and test
  data"), so test rows inform the representation even though their labels are unknown.
- **Features:** none engineered for the DAE — "I used only the raw sensor data (scaled)".
- Tried and rejected: adding a `subject` count feature and a `subject` embedding — no improvement, on only 2 single
  experiments he had time for.
- Predictive TF(TPU) branch used base data + feature engineering: **shift and rolling mean**.
- One LGBM member used tsflex-generated features from the notebook he credits at the end of the post.
- DAE feature files had to be exported and read as **TFRecords** to feed the TensorFlow branch.
- "I don't think that I came up with any new feature engineering for any of the models."
- **Models:** DAE = 2 LSTM layers (more layers not tested); increasing LSTM size improved results until training time
  capped it.
- Predictive model = neural network (PyTorch GPU) fed with AE-layer outputs **+** the original scaled sensor data
  (better than AE features alone).
- TensorFlow branch: Bi-LSTM starter forked from @hamzaghanmi; GRU slightly better than LSTM in TF; BatchNorm added a
  small improvement in the final run; relatively smaller/simpler models matched or beat bigger ones.
- GBM branch: "the usual LGBM, XGB, RandomForest, ExtraTrees etc".
- Key predictive-net regularizer: **spatial dropout, often > 0.35 rate** — improved the end score and stabilized training.
- Hyperparameters beyond the above (lr, epochs, exact sizes, seeds): not stated.
- **CV:** 10 group folds on subject; 20 folds gave "maybe a +0.001" improvement but the GPU/PyTorch training-time
  tradeoff was judged poor.
- DAE epoch count chosen by scoring 1-fold predictive CV at DAE checkpoints.
- Published pairs: best single DAE pipeline CV 0.98999 / private 0.99134 (+0.00135); best TF model CV 0.9851 / private
  0.98668 (+0.00158); full blend CV 0.9918 / private 0.99249 (+0.00069).
- Author's trust verdict: "the CV/Leaderboard correlation seemed reasonably strong to me, particularly after blending
  all the models together (I burned through a lot of unnecessary submissions to have a better feeling of correlation)".
- **Ensembling:** ElasticNet used to derive final model weights over the pool; "many of the inputs were given 0 weight
  and the main contributors were the DAE runs, some TPU runs, and the LGBM models".
- Pool = 4 main DAE runs (PyTorch/GPU) + TF(TPU) with base data + shift/rolling-mean FE + TF(TPU) with saved DAE features
  + LGBM/XGB/RF/ET (one LGBM with tsflex features).
- DAE features worked substantially worse in TensorFlow than in PyTorch (author's own comparison; cause not found).
- **Post-processing:** none stated (no threshold, calibration or rank transform).
- **Gains:** (+0.00115 private AUC) full ElasticNet blend over the best single DAE pipeline, 0.99134 → 0.99249.
- (+0.00581 private) best TF model 0.98668 → the 0.99249 blend.
- (+0.00135 CV→LB on one pipeline) group-fold CV proved well calibrated for the DAE branch.
- (stated as clearly better, unquantified) feeding the predictive model both the DAE features **and** the original
  scaled sensor data.
- (stated as better than AE features alone) spatial dropout > 0.35 on every predictive net.
- (+0.00173 private, stated as "slightly worse" for the alternative) bigger DAE LSTM 0.99134 vs smaller DAE LSTM 0.98961.
- **Failed:** swap-noise levels around 50% — "didn't seem to work as well"; best results below 30% noise.
- Feeding **all** autoencoder layers to stage 2 — "Only some of the layers ... seemed to help. Some layers seemed to
  make the outcome worse."
- Extra DAE feature engineering: none helped; raw scaled sensors only.
- `subject` count feature and `subject` embedding: no improvement (2 experiments each).
- 20-fold CV: only ~+0.001 for a large time cost.
- More than 2 LSTM layers in the DAE: never tested (time quota).
- DAE features in the TensorFlow branch: "substantially worse than Pytorch"; he stopped trying.
- **Comments:** 8 comments counted; the author's own post-edit is the correction (public→private numbers). Non-author
  replies worth keeping:
  - @azzamradman (2nd): "DAE is something I tried, but with Conv2D and Conv2DTranspose layers and it didn't work.
    LSTM should have been given a try."
  - @group16 (3rd): "impressive to see how well DAE are performing here! ... Cool that our notebook was of any help as well"
    (the tsflex/powershap notebook he credits as one LGBM member).
  - @liartem (246th), @fionacarson (696th), @niekvanderzwaag (122nd), @zhixx018 (223rd, appreciation), SRK, one anonymous: congratulations only.
- **Compute:** "All within the time quotas of Kaggle GPU". Best DAE run ≈ **8 h** DAE training on full train+test +
  **5.5 h** for the predictive model over 10 folds on Kaggle GPU; the smaller DAE variant ≈ 4.5 h per stage.
  TF branch ran on TPU. TFRecords pipeline needed for DAE features. RAM, cost: not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/discussion/216037
  - https://www.kaggle.com/competitions/tabular-playground-series-feb-2021/discussion/222745
  - https://www.kaggle.com/competitions/tabular-playground-series-mar-2021/discussion/229833
  - https://www.kaggle.com/competitions/tabular-playground-series-mar-2021/discussion/229868
  - https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
  - https://www.kaggle.com/code/davidedwards1/tabularmarch21-dae-starter
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/dave-e-1-lb-solution (author citation)
  - https://www.kaggle.com/code/hamzaghanmi/tps-april-tensorflow-bi-lstm
  - https://www.kaggle.com/code/jeroenvdd/tpsapr22-best-non-dl-model-tsflex-powershap?scriptVersionId=94240450
  - https://www.kaggle.com/springmanndaniel , https://www.kaggle.com/ryanzhang , https://www.kaggle.com/davidedwards1
  - https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322259
- **Lesson:** A self-supervised representation trained on train+test with a task-tuned noise level, then blended as just
  one family among GBMs under a fitted ElasticNet, is worth ~+0.0012 private over the best single pipeline.

### TPSAPR22-02 · 2nd · azzamradman (DataRegressor) · LB 0.987/0.989 (best single model) · CV 0.963 (single), 0.99 (stack)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322257
- **Status:** FETCHED (3 attempts: plain `/c/` gave a 164-byte empty shell, `/c/` + `X-No-Cache: true` gave 6,681 bytes
  with body + 7 comments (used), `/competitions/` + `X-No-Cache` gave 6,727 bytes with the same comments)
- **TL;DR:** A 40-model LGBM stack over OOF meta-features, where the strongest member is a 4×2D-CNN + GRU + GMP net whose
  per-fold runs are triple-repeated and the two losing repeats become extra meta-features.
- Stack bought +0.00130 over the members; the single model alone "can land the 6th place in this competition".
- **Architecture:** STACK2 · stages=2 · l1=40 models (each ≥0.96 on both local CV and LB) · l2=LightGBM trained on the
  saved OOF meta-features · novel=none · mod=none ·
  topo=StratifiedKFold on reshaped (-1, 60, features) → 40 models ×(3 repeats per fold, best kept) → saved OOF matrix →
  LGBM meta → submission (CV 0.99)
  - Level 0 is a wide pool of CNN/RNN nets and GBMs; level 1 is a **trained LGBM on the OOF matrix**, so this is `STACK2`
    and not a weighted blend.
  - The fold-repeat trick is the distinctive part: for each fold the net was run 3 times, the best AUC run became the
    submission-quality prediction and the other 2 runs were kept as additional meta-feature columns — selection happens
    inside the fold and the discarded repeats are reused, which is why no `mod=` tag is needed beyond the stack itself.
- **Setup:** sequences reshaped to `(-1, 60, train.shape[-1])` — 60 steps per sample, all sensor columns as the last axis;
  after reshaping "you don't need to care for the groups as each index now is a full group by itself".
- Row/subject counts, submission slots: not stated.
- CV is **StratifiedKFold stratified on the label** over the reshaped index.
- **Features:** deliberately minimal — "I didn't use new features compared to the public one", except grouping
  `sensor_02` readings by an engineered **`count`** feature, which he does not think moved the final score.
- Encodings, tsflex/tsfresh, dropped columns: not stated.
- **Models:** best single model = **4×2D-CNN + GRUs + GMP** (Keras), published as
  https://www.kaggle.com/code/azzamradman/tps04-best-single-model-0-989 with the submission notebook
  https://www.kaggle.com/code/azzamradman/best-single-model-submission-0-989.
- Pool: 40 models, all ≥0.96 on CV and LB; per-model libraries/hyperparameters/seeds/fold counts: not stated.
- Rejected architectures: Conv2D/Conv2DTranspose + GlobalAveragePooling2D denoising autoencoder; Transformer encoder.
- **CV:** single best model CV 0.963 vs public 0.987 / private 0.989 — CV understates by ~+0.024 for the single net.
- Final stack CV AUC 0.99 (LB for the stack: not stated in this post).
- Fold-level monitoring: "I noticed the variation between each fold's output in terms of AUC" — the reason for 3 repeats.
- **Ensembling:** save OOF predictions ("meta-features") of every model ≥0.96, fit a final LGBM on them; stated gain
  **+0.00130**. Stacking notebook: https://www.kaggle.com/azzamradman/tps-04-blending.
- Number of models actually averaged/stacked: 40.
- Shared public OOF sets: not stated (all members are his own models).
- **Post-processing:** none stated.
- **Gains:** (+0.00130 AUC) LGBM stack over the 40-member pool (author's stated figure for "this step").
- (+0.027 CV) single model 0.963 CV → 0.99 stack CV.
- (stated as fold-noise reduction) best-of-3 repeats per fold plus the 2 discarded runs as meta-features.
- (cross-check, not an author gain) @kailai (10th) retrained the published notebooks from this page and reports private
  0.99024 / public 0.99003 — +0.00124 private and +0.00303 public over the author's own 0.989/0.987 for the same model.
- **Failed:** Denoising Autoencoder (his variant): sequences treated as images, Conv2D + MaxPooling2D to compress,
  GlobalAveragePooling2D to flatten, Conv2DTranspose + Conv2D to reconstruct — listed under "Things didn't work well",
  and he concedes "DAE seems to be working with LSTM as in @davidedward12's solution" (the layer type is the difference).
- Transformer Encoder: "its training was tricky and the results were unstable".
- New feature engineering: he reports nothing beyond the `sensor_02`/`count` grouping paid off.
- **Comments:** 7 comments; author reply: notebooks are shared in the post (to @raviista, 48th).
  Non-author technical content: @kailai (10th) retrained the published notebooks and got private 0.99024 / public 0.99003;
  @davidedwards1 (1st) notes he "didn't try any Conv2D" — i.e. the conv DAE failure is architecture-specific.
- **Compute:** not stated (no wall-clock/GPU/TPU figures; fold repeats imply ≥3× the single-model cost, but he does not
  quantify it).
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/azzamradman/tps04-best-single-model-0-989
  - https://www.kaggle.com/code/azzamradman/best-single-model-submission-0-989
  - https://www.kaggle.com/azzamradman/tps-04-blending
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/azzam-radman-2nd-place-solution (author citation)
  - https://www.kaggle.com/davidedward12 (handle as written in his post) , https://www.kaggle.com/azzamradman
- **Lesson:** When fold-to-fold AUC variance is the problem, run each fold three times, keep the winner and spend the two
  losers as extra OOF columns — a free meta-feature that costs only compute.

### TPSAPR22-03 · 3rd · group16 / jeroenvdd / moeflon (VD Brothers: Gilles Vandewiele, Jeroen Van Der Donckt, Vic Degraeve) · LB 0.99037 (public, stacked)/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322269
- **Status:** FETCHED (3 attempts: plain `/c/` 164-byte shell, `/c/` + `X-No-Cache: true` 12,393 bytes with body + 9
  comments (used), `/competitions/` + `X-No-Cache` 12,711 bytes with the same comments)
- **TL;DR:** Feature engineering first (tsflex + powershap + deep **shapelets** mined in PyTorch), then a CatBoost
  meta-learner fitted on a stack of own and public model OOFs.
- Shapelet-augmented feature model alone: 0.98052 public / 0.97706 private; the stack reached 0.99037 public (+0.00985).
- **Architecture:** STACK2 · stages=2 · l1=own feature-based models (older versions included) + 7 public notebooks ·
  l2=CatBoost fitted on the stacked OOF predictions · novel=Shapelet-NN · mod=public-oof,embed ·
  topo=raw sequences → tsflex sliding-window features (+powershap pruning) → shapelet mining net (features passed in) →
  own feature models + 7 public DL/GBM notebooks (OOF stored, GroupKFold on subject, model reset per fold) →
  CatBoost meta + subject-level aggregates of the predictions → submission
  - The `Shapelet-NN` name in the novel slot is the **level-0** extractor, not the submitted model: shapelets are short
    predictive subsequences learned by gradient descent (tslearn's approach ported from Keras to PyTorch with lr
    scheduling, early stopping and pass-through extra features). The submitted artifact is the CatBoost stack.
  - Ordering matters: shapelet mining ran **after** feature extraction, and the already-extracted features were fed to
    the shapelet net so the mined shapelets are *complementary* to them.
  - Level 1 is a trained meta-learner (CatBoost) on OOF columns → `STACK2`, with public notebooks as additional members
    (rule 3: they add members, not stages).
- **Setup:** **10 GroupKFold on `subject` for all trained models** — "This resulted in CV scores that corresponded very
  well to LB scores."
- Team of three (first Kaggle competition outside class for two of them). Row/col counts, slots: not stated.
- Structural quirk exploited: subject-level aggregation of predictions (a subject's sequences share a label pattern).
- **Features:** tsflex sliding-window statistical features (his team's in-house package) + `powershap` used to
  automatically exclude many unhelpful feature types.
- Deep-mined shapelets added on top as extra features.
- Baseline they built from: @ambrosm's "tpsapr22-best-model-without-nn".
- Public notebooks' own FE was reused as-is (their stack members came with their features).
- **Models:** level 0 = "some older versions of our feature-based approaches" (LGBM-family GBMs on tsflex features +
  shapelets) plus 7 public models (6 deep + 1 XGBoost): dlaststark tfv1/tfv2, bannourchaker part2 BiLSTM+DenseNet+RNN, part3
  CNN-InceptionTime, part4 hybrid CNN-LSTM parallel, davidedwards1 TF Bi-LSTM 10-fold spatial dropout, siukeitin XGBoost
  baseline on 2500 features.
- Shapelet extractor: tslearn's keras implementation re-written in PyTorch by @moeflon.
- Meta-learner: **CatBoost** fitted on the stacked predictions.
- Per-model hyperparameters, seeds, epochs: not stated.
- **CV:** 10 GroupKFold on subject; author-stated calibration: GroupKFold on subject matched LB, whereas many public
  notebooks grouped on **sequence** instead of subject (their #1 fix).
- Published: 0.98052 public / 0.97706 private for the pre-stack submission; 0.99037 public after stacking; the stack's
  private is not stated.
- **Ensembling:** stack different models and fit a CatBoost on it; per-member weights: not stated (a model, not a grid).
- Also: "group our predictions by subject and calculate aggregates (mean, min, max, std, …)" — worth **+0.002 CV/LB**,
  described as "nice & easy".
- Modifications applied to the public notebooks to make them stack-safe: store OOFs during CV, GroupKFold on **subject**
  (not sequence), reset the model in every fold instead of continuing to fit across folds.
- **Post-processing:** subject-level aggregation of predictions (mean/min/max/std) applied to the submission; treated
  here as a feature/ensembling step per the author's own framing, plus an explicitly rejected all-zero-subject
  classifier post-processing model (see Failed).
- **Gains:** (+0.00985 public AUC) stacking over the best single feature-based submission: 0.98052 → 0.99037.
- (+0.002 AUC) subject-level prediction aggregates (mean, min, max, std).
- (stated as the reason LB tracked CV, unquantified) GroupKFold on `subject` rather than on sequence id.
- (stated as valuable, unquantified) feeding already-extracted features into the shapelet net so shapelets are complementary.
- **Failed:** Pseudo-labeling the confident test-set predictions.
- Hidden Markov Models per subject (sequences ordered by sequence id) — "to no avail"; estimating HMM parameters from
  training labels gave only a marginal improvement, and there was no accurate label-free way to estimate them.
- Stitching together the sequences of the same subject.
- A post-processing model predicting which subjects contain only label-0 sequences (~9% of subjects, ~5% of the data;
  an AUC-1.0 model is fittable) — "adding this model does not bring any added value (our models already implicitly
  learned these things)".
- Many engineered feature types, automatically excluded with powershap.
- **Comments:** 9 comments counted; the author reply is the highest-value withheld detail in this file:
  - Gilles Vandewiele (topic author, to @bannourchaker, 9th): "do GroupKFold on subject instead of sequence ID and more
    importantly to reset your model in every fold! If you call .fit() in keras, it will resume training from its last
    point. As such, in your second fold, the model will have actually already trained on the current validation data
    which is why you get val AUC close to 1." — i.e. public notebooks' near-1.0 val AUC was leakage.
  - @bannourchaker's counter-argument: he grouped by sequence deliberately "to have more general model (as I add that
    information about subject in feature engineer)".
  - @davidedwards1 (1st): "Will be reading more about the platelets idea" (typo for shapelets).
  - @azzamradman (2nd), @liartem (246th), SRK, @bannourchaker, one anonymous: congratulations.
- **Compute:** not stated (no wall-clock/GPU/TPU figures or limit notes).
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/jeroenvdd/tpsapr22-best-non-dl-model-tsflex-powershap
  - https://www.kaggle.com/code/ambrosm/tpsapr22-best-model-without-nn
  - https://github.com/predict-idlab/tsflex
  - https://github.com/predict-idlab/PowerSHAP
  - https://tslearn.readthedocs.io/en/stable/user_guide/shapelets.html
  - https://github.com/tslearn-team/tslearn/
  - https://www.kaggle.com/code/dlaststark/tps-apr22-tfv2
  - https://www.kaggle.com/code/dlaststark/tps-apr22-tfv1
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part3-cnn-inceptiontime-con11
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con166
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part2-bilstm-densenet-rnn-con6
  - https://www.kaggle.com/code/davidedwards1/tps-april-tensorflow-bi-lstm-10f-spatialdropout
  - https://www.kaggle.com/code/siukeitin/tps042022-baseline-xgboost-model-on-2500-features
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/vd-brothers-3-solution (author citation)
  - https://www.kaggle.com/group16 , https://www.kaggle.com/jeroenvdd , https://www.kaggle.com/moeflon ,
    https://www.kaggle.com/davidedwards1 , https://www.kaggle.com/azzamradman , https://www.kaggle.com/ambrosm ,
    https://www.kaggle.com/bannourchaker
- **Lesson:** Stack only after fixing the public notebooks' validation (group on subject, reset the model each fold,
  store OOFs) — the meta-learner's +0.00985 was mostly bought by making the members' OOFs honest.

### TPSAPR22-04 · 4th · ymatioun (Youri Matiounine) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322558
- **Status:** FETCHED (1 attempt, `/c/.../discussion/322558` + `X-Return-Format: markdown`, 3,620 bytes — above the shell
  threshold and contains full solution prose; body + 2 comments)
- **TL;DR:** Skip building your own LSTM; rerun the best public LSTMs to harvest their **out-of-fold train predictions**,
  then feed those (plus subject- and sequence-level aggregates) into a LightGBM.
- "This form of model stacking seems to produce better synergy than simple blending."
- **Architecture:** STACK2 · stages=2 · l1=best public LSTM models (rerun by the author to obtain train OOF predictions) ·
  l2=LightGBM trained on aggregated features **including** the LSTM OOF predictions · novel=none · mod=public-oof ·
  topo=public LSTM models → rerun for OOF train predictions → aggregate by subject and by sequence (many ways) →
  LightGBM (features = aggregates + LSTM OOF predictions) → submission
  - The level-0 predictions enter level-1 as **features of a GBM**, not as terms of a weighted blend, so this is a fitted
    meta-learner → `STACK2`. Ruling-5 check: the page feeds the LGBM the LSTMs' final *predictions* ("fed predictions from
    best LSTM models as one of the features"), explicitly harvested **out-of-fold** on train — not LSTM representations —
    so this is `STACK2`, not `CASCADE mod=embed`; the author himself calls it "this form of model stacking".
  - His own LSTM attempt exists but was excluded: it was worse than the public models, so level 0 is entirely public work.
  - No score, no CV scheme, no hyperparameters are published anywhere on this page — the mechanism is the whole record.
- **Setup:** sequence-level data with a subject grouping (his inputs are "aggregated by subject as well as by sequence");
  row/col counts, folds, slots: not stated.
- Structural quirk exploited: every LSTM sees one sequence at a time, so all subject-level information is missing from
  level 0 and must be recovered at level 1.
- **Features:** aggregates by **subject** and by **sequence**, computed "in many ways" (specific statistics not listed);
  the aggregated LSTM predictions are among them.
- tsflex/tsfresh, encodings, dropped columns: not stated.
- **Models:** level 0 = "best public LSTM models" (identities not named on this page) + his own LSTM implementation,
  discarded for being weaker.
- Level 1 = one LightGBM model. Hyperparameters, seeds, folds: not stated.
- **CV:** not stated. The one validation-relevant act reported is re-running the LSTM models to capture their **out-of-fold**
  train predictions in addition to test predictions — i.e. leak-free OOF harvesting.
- **Ensembling:** stacking via LGBM-on-OOF rather than blending; weights: not applicable (a trained meta-learner);
  number of LSTM members: not stated.
- **Post-processing:** not stated.
- **Gains:** (qualitative, the page's core claim) stacking beats simple blending for LSTM+GBM synergy; his own LSTM →
  public LSTMs was itself an improvement, though he publishes no numbers for either.
- **Failed:** Building his own LSTM — "i tried to create my own, but it was worse than best public LSTM models, so i
  ended up using public LSTM models instead of my own ones".
- No other dead ends listed.
- **Comments:** 2 comments counted; **nothing technical resolved in them** — SRK congratulates ("Interesting way to stack
  the LSTM model with the GBM one") and an anonymous commenter asks which LSTM models were better or whether he just took
  the best one; **no author reply is present on the page**, so the LSTM identities stay unknown.
- **Compute:** not stated (re-running several LSTM models for OOF is the implied cost; no figures).
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/youri-matiounine-4-solution (author citation)
  - https://www.kaggle.com/ymatioun , https://www.kaggle.com/sudalairajkumar
- **Lesson:** When public models beat your own, the winning move is to re-run them for clean OOF predictions and let a GBM
  learn the aggregation the sequence models structurally cannot see.

### TPSAPR22-05 · 5th · cabaxiom (Cabaxiom) · LB 0.98248/0.97816 (LGBM member); 0.98131/0.98259 (LSTM member) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322277
- **Status:** FETCHED (1 attempt, `/c/.../discussion/322277` + `X-Return-Format: markdown`, 6,403 bytes, body + 7 comments)
- **TL;DR:** A single tsfresh-featured LGBM (~9000 auto-generated features, trimmed by recursive feature elimination, plus
  per-subject normalized copies of every feature), weighted-blended with his own LSTM and seven public notebooks.
- No stacking — and he says that is what cost him the top 3.
- **Architecture:** FLAT · stages=1 · l1=1 LGBM (tsfresh+RFE) + 1 own LSTM + 7 public deep-learning notebooks ·
  l2=none (fixed weights only) · novel=none · mod=public-oof ·
  topo=raw sequences → tsfresh ~9000 features → RFE trim → LGBM (w≈0.4) + own LSTM (w≈0.2) + 7 public notebooks (w≈0.4 split)
  → weighted average → submission
  - Weights are hand-assigned proportions (0.4/0.2/0.4), and no model was fitted on the OOF matrix → `FLAT`, per ruling 1.
  - His LSTM out-predicted his LGBM on private (0.98259 vs 0.97816) yet got the smaller weight — the blend is not CV-optimal.
  - The tsfresh branch is the interesting part: for **every** generated feature he added a normalized-by-subject twin.
- **Setup:** ~9000 tsfresh features before trimming; rows/cols/folds/slots: not stated.
- Structural quirk exploited: normalizing each feature within its subject (subject-relative signal the raw window
  statistics miss).
- **Features:** tsfresh auto-features, each duplicated as "the normalized feature value by subject".
- Trimming: recursive feature elimination (RFE) down from ~9000; final feature count: not stated.
- **Models:** (1) LightGBM on the tsfresh+RFE set — 0.98248 public / 0.97816 private.
- (2) his own LSTM — 0.98131 public / 0.98259 private.
- (3–9) seven public notebooks used as blend members: dlaststark tfv1/tfv2, dmitryuarov "sensors-deep-analysis-0-98",
  bannourchaker parts 2/3/4, hasanbasriakcay "tpsapr22-fe-pseudo-labels-bi-lstm".
- Hyperparameters, seeds, fold counts for any model: not stated.
- **CV:** no CV number published — every score on the page is an LB score.
- Public→private movements: LGBM 0.98248 → 0.97816 (−0.00432, private worse); LSTM 0.98131 → 0.98259 (+0.00128, private better).
- Author's trust verdict: not stated.
- **Ensembling:** fixed weights — LGBM ≈0.4, own LSTM ≈0.2, remaining ≈0.4 split across the 7 public notebooks;
  per-notebook weights within that 0.4: not stated. Shared OOF usage: none reported ("I did not use stacking").
- **Post-processing:** none stated.
- **Gains:** (+0.00443 private AUC) his own LSTM (0.98259) over his own tsfresh LGBM (0.97816) — yet the LSTM got the
  smaller blend weight (≈0.2 against the LGBM's ≈0.4).
- (+0.00128 public→private on the LSTM branch) 0.98131 public → 0.98259 private; the LGBM branch moved the other way
  (0.98248 public → 0.97816 private, −0.00432), i.e. the member weighted most heavily was the one that lost on private.
- (stated as the missing step, unquantified) stacking instead of hand weights — "perhaps if I had my score would be
  closer to the top 3".
- Score of the final weighted blend: not published on this page.
- **Failed:** the page names no failed experiment. What is documented: ~9000 tsfresh features had to be trimmed by
  recursive feature elimination before the LGBM could use them (stated; the reason for trimming is not given), and the
  LGBM's public edge did not survive to private (−0.00432). The author's own comment adds the ensemble itself as the
  weak part ("My ensemble was particularly weak compared to others, as I did not use stacking").
- **Comments:** 7 comments counted; the technical content is in the author's replies:
  - To @akmalmir (13th): "I think potentially this solution could have been enough to get a similar score to the top 3,
    but not 1st. Actually, the #3 solution was not too dissimilar to mine, just with a few extra steps. **My ensemble was
    particularly weak compared to others, as I did not use stacking**, perhaps if I had my score would be closer to the
    top 3, but who knows!"
  - Confirms to @bannourchaker (9th) that his public notebooks "were very useful" — i.e. they are blend members, not baselines.
  - @lachlangillian, an anonymous commenter ("I like the recursive feature elimination approach"), SRK: congratulations.
- **Compute:** not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/dlaststark/tps-apr22-tfv1
  - https://www.kaggle.com/code/dlaststark/tps-apr22-tfv2
  - https://www.kaggle.com/code/dmitryuarov/sensors-deep-analysis-0-98
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part2-bilstm-densenet-rnn-con6
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part3-cnn-inceptiontime-con11
  - https://www.kaggle.com/code/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con166
  - https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-fe-pseudo-labels-bi-lstm
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/cabaxiom-5-solution (author citation)
  - https://www.kaggle.com/cabaxiom , https://www.kaggle.com/bannourchaker , https://www.kaggle.com/akmalmir
- **Lesson:** A hand-weighted blend of a strong GBM and strong LSTMs plateaued below the fitted stacks above him (his
  blend score was never published) — if you have the OOFs, fit a meta-learner instead of guessing weights.

### TPSAPR22-06 · 6th · sebastianvangerwen (Sebastian van Gerwen) · LB 0.985/0.9839 (GRU member); final blend not stated/0.98797 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322622
- **Status:** FETCHED (3 attempts: plain `/c/` 2,506-byte nav shell, `/c/` + `X-No-Cache: true` 4,658 bytes full body +
  5 comments (used), `/c/` + `X-No-Cache` + `X-Engine: browser` re-shelled at 2,506, `/competitions/` + `X-No-Cache`
  4,704 bytes equivalent)
- **TL;DR:** A per-sensor GRU tower network: project each 60-step sequence to 16 dims, then run 13 **separate** 4-layer
  GRU nets, one per sensor stream, to stop the net exploiting noisy cross-sensor covariates.
- That net alone: 0.985 public / 0.9839 private; blended with XGBoost, a 1D-CNN and public LSTM models → 0.98797 private.
- **Architecture:** FLAT · stages=1 · l1=1 multi-tower GRU net + 1 XGBoost + 1 1D-convolutional model + public LSTM models ·
  l2=none · novel=none · mod=public-oof,multi-view ·
  topo=(n, 60, 13) → linear dense projection to 16 dims → (n, 60, 13, 16) → 13 parallel 4-layer GRU towers (one per
  sensor) → prediction; + XGBoost + 1D-CNN + public LSTMs → ensemble (private 0.98797)
  - `multi-view` is the architecture: each tower sees exactly one sensor stream, so the 13 views are disjoint feature sets
    rather than 13 copies of the same input.
  - The submission is an ensemble of that net with a GBM, a 1D-CNN and public LSTM predictions; no meta-learner is
    mentioned and no weights are published → `FLAT`, and the blend mechanic itself is `not stated`.
  - The projection layer (13 → 16 dims per step) is part of the trained net, not a separate stage.
- **Setup:** data reshaped as 60 timesteps × 13 sensors per sample; projection adds a 16-dim channel axis → (n, 60, 13, 16).
- Row/subject counts, fold count, submission slots: not stated.
- Structural quirk exploited: noisy covariates **between** sensor sequences — separating towers removes that leakage path.
- **Features:** none engineered — raw sequences projected linearly.
- Encodings, tsflex/tsfresh, dropped columns: not stated.
- **Models:** GRU net: one linear dense projection (→16 dims) + **13 separate GRU networks of 4 GRU layers each**, one per
  sensor sequence.
- Ensemble partners: XGBoost, a 1D-convolutional model, and public LSTM models (identities not named on this page).
- Optimizer, lr, epochs, batch size, dropout, seeds, folds: not stated.
- **CV:** not stated — both published numbers are LB scores (GRU net 0.985 public / 0.9839 private; ensemble private
  0.98797, public not stated).
- Public→private on the single net: 0.985 → 0.9839 (−0.0011, private worse).
- **Ensembling:** "Ensembling this model with XGBoost, 1D-convolutional, and public LSTM models" — weights and mechanic not
  stated; number of members not stated beyond 3 named families plus public LSTMs.
- **Post-processing:** not stated.
- **Gains:** (+0.00407 private AUC) ensemble over the single multi-tower GRU net: 0.9839 → 0.98797.
- (stated as the reason the tower split works, unquantified) preventing the GRU from overfitting to noisy covariates
  between sequences, "and allowed the network to converge better".
- **Failed:** No failed model, feature or hyperparameter is listed on this page (counted: the body has one architecture,
  one ensemble, no dead-end section).
- **Comments:** 5 comments counted (one marked deleted by Kaggle). Technical content:
  - SRK caught a typo in the post: the single net's private score read "0.839" and asked whether it should be 0.9839.
    Author reply: "Yes my mistake. Thanks! It's fixed now." — so 0.9839 is the correct private figure.
  - @ravi20076 (318th here): thanks for sharing (appreciation comment).
- **Compute:** not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/writeups/sebastian-van-gerwen-6th-place-solution (author citation)
  - https://www.kaggle.com/sebastianvangerwen , https://www.kaggle.com/sudalairajkumar , https://www.kaggle.com/ravi20076
  - no notebooks, datasets or GitHub links are cited in the body
- **Lesson:** When sensor streams are noisy relative to each other, give each stream its own tower and let a later
  ensemble fuse them — splitting the input beats regularizing one shared net.

## TPSAPR22 — consensus recipe
- **Architecture distribution:**
  - 1st `STACK2` — swap-noise LSTM denoising autoencoder (trained on train+test) feeds predictive nets; ElasticNet-derived
    blend weights over DAE/TPU/LGBM runs; CV 0.9918 / private 0.99249
  - 2nd `STACK2` — 40 models → saved OOF meta-features → LGBM meta; stack CV 0.99, +0.00130 from stacking
  - 3rd `STACK2` — tsflex + deep-shapelet features → own models + 7 public notebooks → CatBoost meta; 0.99037 public
  - 4th `STACK2` — public LSTMs rerun for OOF → LightGBM on subject/sequence aggregates; no scores published
  - 5th `FLAT` — fixed weights 0.4 LGBM / 0.2 own LSTM / 0.4 over 7 public notebooks; members 0.97816–0.98259 private
  - 6th `FLAT` — 13-tower GRU net + XGBoost + 1D-CNN + public LSTMs, unweighted description; 0.98797 private
  - winner topology: `STACK2` (cascade DAE features → predictive models → ElasticNet-weighted blend). **4 of 6 entries
    (01, 02, 03, 04) fitted a meta-learner (ElasticNet, LGBM, CatBoost, LGBM); the 2 that only weighted (05, 06) landed
    5th and 6th.** No `STACKN`, `PSEUDO`, `AUTOML` or `AGENT` submission in this set.
- **New architectures at the board:** `Denoising-Autoencoder` at 1st (private 0.99249 for the blend; 0.99134 private for
  one DAE pipeline) — but 1st documents it as imported from TPS Jan/Feb/Mar 2021 (entries 01's four discussion links) and
  from Porto Seguro, so it is a revived, not a novel, 2022 architecture. `Shapelet-NN` (gradient-descent-mined shapelets,
  tslearn ported to PyTorch) at 3rd, as a level-0 feature extractor (entry 03). Second's Conv2D/Conv2DTranspose DAE
  variant **failed** (entry 02), as did Transformer encoders (entry 02) — the DAE only worked with LSTM layers, per the
  1st/2nd comment exchange. Nothing else beyond stock CNN/GRU/LSTM nets, LGBM/XGBoost/CatBoost/RF/ET and ElasticNet/LGBM/CatBoost metas.
- **Agreed on (4 of 6: 01, 02, 03, 04):** a trained meta-learner on stored OOF predictions rather than hand weights —
  and 5th confirms the inverse from below ("My ensemble was particularly weak compared to others, as I did not use
  stacking", entry 05).
- **Agreed on (5 of 6: 01, 03, 04, 05, 06):** public notebooks/models as pool members — 1st forked a TF Bi-LSTM starter and
  one tsflex LGBM run, 3rd stacked 7 public notebooks, 4th used public LSTMs as level 0 entirely, 5th weighted 7 public
  notebooks, 6th blended public LSTM models. Only 2nd built all 40 members himself.
- **Agreed on (5 of 6: 01, 02, 03, 04, 05):** subject is the unit of structure — 1st and 3rd used 10 folds with each subject
  inside one fold (GroupKFold on subject), 2nd reshaped so each index is a whole sequence-group, 4th aggregated the LSTM
  predictions by subject as LGBM features, 5th normalized every tsfresh feature by subject; 3rd's author comment states the
  correct grouping (subject, not sequence) is what makes CV track LB and what invalidates public notebooks showing val AUC
  near 1.
- **Agreed on (3 of 6: 01, 02, 03):** the blend/stack step is worth roughly +0.001–+0.01 at the top — +0.00115 (1st's blend over its
  best pipeline), +0.00130 (2nd's stack over its pool), +0.00985 (3rd's stack over its best single submission); all three
  name the ensemble step, not new features, as the win.
- **Divergences:** the top 3 all **built** a level-0 artifact the others consumed (1st an unsupervised DAE representation;
  2nd a 4×2D-CNN+GRU+GMP net at 0.989 private single-model; 3rd shapelet features + public model hygiene), while 4th and
  5th consumed public models almost unchanged — 4th still stacked (rank 4, no published score) and 5th only weighted
  (rank 6 equivalent members: 0.98248 public / 0.97816 private). The measured cost of choosing weights over a meta-learner
  is about +0.0013–0.00985 AUC of forgone blend gain; choosing sequence-level grouping over subject-level grouping
  invalidates the CV rather than costing a fixed delta (entry 03's comment thread).
- **Highest-leverage single trick:** group predictions by `subject` and add mean/min/max/std aggregates — "+0.002" public
  and CV, described as nice and easy (entry 03); runner-up: rerun each fold 3 times, keep the best and feed the two
  rejected repeats to the meta-learner as extra OOF columns (entry 02).
- **Nothing worked:** pseudo-labeling confident test predictions (03) and the author's own LSTM vs public LSTMs (04);
  Conv2D/Conv2DTranspose denoising autoencoder and Transformer encoders (02); HMMs per subject and stitching a subject's
  sequences (03); a post-processing classifier for the ~9% all-negative subjects — reachable at AUC 1.0, worth nothing
  (03); high swap-noise ~50% and AE layers fed wholesale into stage 2 (01); subject count/subject-embedding features (01);
  20-fold CV at ~+0.001 for the extra GPU time (01); new hand-built features beyond public ones (01, 02).
