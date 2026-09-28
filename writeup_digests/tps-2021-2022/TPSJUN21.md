## TPSJUN21 — Tabular Playground Series - Jun 2021
- Task: Tabular (Multiclass Classification) | Metric: Multiclass Loss | Problem: Practice multiclass classification
- Kaggle display title: "Tabular Playground Series - Jun 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-jun-2021
- Writeups covered: 2 of 2
- Score ladder (multiclass log loss, LOWER better; `public/private` as reported by the authors): 1st `not stated / not stated`
  (the only private numbers on the page are his **step-2 intermediate** scores `1.73900` and `1.73890`) · 2nd `not stated / not
  stated` (publishes no score at all).
- Neither page states the final winning score, so the 1st-vs-2nd gap on this board cannot be computed from the library.
- 9 classes are stated only through 1st's XGBoost parameters (`num_class = 9`); 2nd's page never states the class count.

### TPSJUN21-01 · 1st · kailai · LB not stated/1.73900 then 1.73890 (author's step-2 intermediates, explicitly called "private score") · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2021/discussion/250046
- **Status:** FETCHED (jina `/c/.../discussion/250046` gave 14,574 B with all 35 comments; the `?sort=polls` retry was byte-equivalent
  in content, and the browser snapshot was then used to expand the one sub-thread Kaggle renders collapsed as "3 more replies",
  which recovered the author's most detailed reply — quoted under `Comments`)
- **TL;DR:** ~15 XGBoost models from 3 hand-written parameter lists, whose class-relabelled probability columns (~90 of them) become
  the ONLY inputs of an R `neuralnet` MLP; the several MLPs are then weighted-averaged.
- Quantified content of the whole writeup is one number pair: private 1.73900 with tree predictions only vs 1.73890 once some NN
  predictions are added as stage-2 features.
- **Architecture:** STACK2 · stages=2 · l1=~15 XGBoost models producing ~90 prediction columns · l2=R `neuralnet` MLP, hidden layers
  (63, 27), sigmoid/sigmoid, then a weighted average of several such MLPs · novel=none · mod=ovo,multi-view ·
  topo=3 XGB param sets x seeds x class-relabelled targets [~15 models -> ~90 pred cols] → neuralnet MLP(63,27) trained on
  predictions only [no original features] → weighted average of MLPs
  - Stage 2 is a trained meta-learner whose input matrix is stage-1 output columns and nothing else ("No original features"), and the
    author's own word for the design is stacking: "difference between models were best for stacking" — so this is `STACK2`, not
    `CASCADE` (ruling 5 applies only when the earlier output is one feature among originals) and not `FLAT` (ruling 1).
  - The class-order-rotation and class-combination members are an `ovo`-style target decomposition; per ruling 8 the recombination is
    the stage-2 MLP, so the primary stays `STACK2`.
  - `novel=none`: the meta-learner is a stock 2-hidden-layer MLP from the R `neuralnet` package with sigmoid activations. The
    newsworthy part in 2021 was an NN as the *blender*, not a new architecture.
- **Setup:** 9 classes (`num_class = 9`), 3 parameter sets → "about 15" XGBoost models → "about 90 predictions as features".
- Rows/cols and split sizes: not stated on the page (verified against both the jina render and the full browser re-render).
- Submission slots: not stated.
- Structural quirk exploited: relabelling the target (rotate which class is index 0, merge the weak classes) multiplies the number of
  decorrelated prediction columns available to the stack — 15 models yield ~90 columns, not 15.
- **Features:** no feature engineering at all: "I did not do scaling/normalizing etc., package of ann2 do it by itself, that is why I
  chose ann2" (author comment) and `standardize = TRUE` in the call.
- Stage-2 input = stage-1 prediction columns only. Variants named by the author:
  - deleting the class_4 and class_5 tree predictions from the nnet feature set;
  - training tree models on merged targets: class_4+class_5 vs the rest, class_4+class_5+class_1 vs the rest, "etc.";
  - rotating class order: "Class_8 as 0, others as 1,2,3,4---, Class_6 as 0, others as 1,2,3,4---".
- Original 2,000-style raw columns are explicitly NOT fed to stage 2.
- Encodings, GP/generated features, feature-source kernels, dropped columns: not stated.
- **Models:** XGBoost (R wrapper), three parameter sets published verbatim:
  - Set 1.1 (leafwise): `tree_method=hist, max_bin=512, max_leaves=150, min_child_weight=110, grow_policy=lossguide, eta=0.009,
    max_depth=0, subsample=0.7, colsample_bytree=0.11, colsample_bylevel=0.90, lambda=0, alpha=22, objective=multi:softprob,
    eval_metric=mlogloss, num_class=9, max_delta_step=10` (`colsample_bynode=0.80` present but commented out).
  - Set 1.2 (depthwise): `eta=0.006, max_depth=22, min_child_weight=110, gamma=0.01, subsample=0.7, colsample_bytree=0.1,
    colsample_bylevel=0.90, colsample_bynode=0.80, lambda=1.5, alpha=21, objective=multi:softprob, eval_metric=mlogloss, num_class=9,
    max_delta_step=10`.
  - Set 1.3 (boosted random forest): `tree_method=hist, max_bin=512, max_leaves=200, grow_policy=lossguide, min_child_weight=110,
    eta=1, alpha=22, lambda=0, subsample=0.70, colsample_bytree=0.11, colsample_bylevel=0.90, num_parallel_tree=110,
    objective=multi:softprob, eval_metric=mlogloss, num_class=9` (`max_depth=3` commented out).
  - Author's tuning rationale (comment): "Leafwise, depthwise and boosted random forest catch different information ... First set
    growthwise, then tune other parameters."
- Stage-2 MLP verbatim (page form): `bst<-neuralnetwork(X,Y,hidden.layers = c(63,27),standardize=TRUE, optim.type = 'adam',
  learn.rates = 0.0004, val.prop =0.2 ,batch.size=320,random.seed=8888688 ,L1=2,L2=0,activ.functions=c('sigmoid','sigmoid')
  ,n.epochs =200)`. The body's call name is `neuralnetwork`; the author's comment resolves it — "ann2 is neuralnet in r language bst
  is the name of the model" — so the package is R `neuralnet`.
- XGBoost seed values and per-model round counts: not stated; "I changed seeds and parameter" only.
- **CV:** no CV number, splitter, fold count or seed protocol is published ("its cv is not good" is the only CV sentence on the page,
  about LightGBM; the single seed on the page is the meta's `random.seed = 8888688`, listed under `Models`) — CV-vs-LB gap not
  computable.
- Trust verdict: not stated; the final step is chosen by LB-flavoured reasoning the author admits he never ablated: "I did not test the
  difference, just reasoning doing so maybe difference between models were best for stacking".
- **Ensembling:** two blend layers: stage-2 MLP stacking stage-1 outputs, then a weighted average across the MLPs (author confirms
  "it is a weighted average of NNs ... only one set of parameters and one seed, but different features").
- Weights of the final average: not stated.
- Adding some stage-1 **NN** predictions to the tree-prediction feature set: private 1.73900 → 1.73890 (author's "may be" numbers).
- Shared-OOF usage: none cited; every member is his own run.
- Number of models averaged: "about 15" XGBoost models feeding the stack; MLP count not stated.
- **Post-processing:** none stated (no threshold, calibration, clipping, rounding or label trick appears on the page).
- **Gains:** (-0.00010 private, 1.73900 → 1.73890) adding some nnet predictions as extra stage-2 features next to the tree predictions.
- Unquantified but claimed as the load-bearing choices: the three different XGB growth policies for decorrelation, and the
  class-relabel / class-merge target sets that turn 15 models into ~90 columns.
- **Failed:** LightGBM: "Tried lightgbm, but its cv is not good" — XGBoost only.
- Scaling/normalizing by hand: unnecessary, and he says that is why he picked the `neuralnet` package.
- He never validated his own central diversity trick ("I did not test the difference"), so ~90-vs-15 columns is an unmeasured choice.
- Growth path reported: 4 models to test, then 10, "then more" — i.e. no principled stopping rule.
- **Comments:** 35 comments, of which the sub-thread under Maxim Kazantsev (295th) is collapsed by Kaggle and was recovered with the
  browser. The author's replies are the only place the real method appears:
  - "For the second step, I mainly used XGB predictions, some NN predictions. No original features."; "it is a weighted average of
    NNs. Yes, I used only one set of parameters and one seed, but different features. There is more space to improve."
  - "I used these 3 sets of parameters to build about 15 models, chose about 90 predictions as features to train nnet"; "Yes, only
    xgb. Tried lightgbm, but its cv is not good. At the beginning, I built 4 models to test, it is good, then I built 10 models to
    test, then more. Yes, I changed seeds and parameter. ann2 is neuralnet in r language bst is the name of the model."
  - Key expanded reply: "Class_8 as 0,others as 1,2,3,4---, Class_6 as 0,others as 1,2,3,4---, also seed is different. I did not test
    the difference, just reasoning doing so maybe difference between models were best for stacking. Other techniques are: Deleting
    tree predictions of class_4,class_5 from features used for nnet; Training tree models using different target sets, including:
    combining class_4,class_5 vs other classes; combining class_4,class_5,class_1 vs other classes; etc. All these predictions of
    models were used as features of nnet. I did not do scaling/normalizing etc., package of ann2 do it by itself, that is why I chose
    ann2".
  - Tuning-philosophy reply: leafwise (`grow_policy=lossguide` + `max_leaves`), depthwise (`lossguide` with `max_leaves` off),
    boosted random forest (`eta=1` + `num_parallel_tree`).
  - Unanswered: Samvel Kocharyan's data-leak challenge — "It seems like data leak happened on 2nd step. Models at 1st step seen ground
    truth. And their predictions used at second step. Where am I wrong here?" — no author reply anywhere in the expanded tree, and the
    page never says whether stage-1 columns are OOF or in-sample. This is the single biggest replication hazard in the file.
  - Diogo Santiago (603rd) notes the meta uses no ReLU; Joseph Margaryan (3 years later) reports he tried to port the recipe to a newer
    multilabel problem and the simple NN "overfitted to the training data".
- **Compute:** R (XGBoost + `neuralnet`); wall-clock, GPU/CPU, RAM, cost: not stated.
- **Artifacts:** body cites no notebooks or datasets; the citation block and comment links on the page:
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2021/writeups/kailai-1st-place-solution-everyone-can-be-a-winner
    (citation line)
  - https://www.kaggle.com/kailai (author)
  - comment links: https://www.kaggle.com/samvelkoch, https://www.kaggle.com/jonaspalucibarbosa,
    https://www.kaggle.com/vineethakkinapalli, https://www.kaggle.com/andrewkkchoi, https://www.kaggle.com/mustafasenol95,
    https://www.kaggle.com/maximkazantsev, https://www.kaggle.com/adhithia, https://www.kaggle.com/dsantiago,
    https://www.kaggle.com/shardulmehetar, https://www.kaggle.com/tunguz, https://www.kaggle.com/azzamradman,
    https://www.kaggle.com/javiervallejos, https://www.kaggle.com/seongwook93, https://www.kaggle.com/davidbroberts,
    https://www.kaggle.com/josephmargaryan, https://www.kaggle.com/billyweiye, https://www.kaggle.com/jiyunkang
- **Lesson:** A tiny NN meta-learner over ~90 class-relabelled tree columns was enough to win this board — but the writeup never says
  whether those columns are OOF, so reproduce it with fold-safe predictions and expect the gain to shrink.

### TPSJUN21-02 · 2nd · KhanhVD (duykhanh99) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2021/discussion/250060
- **Status:** FETCHED after the ladder: `/c/.../discussion/250060` returned a 1,599 B shell; `X-No-Cache: true` recovered 5,841 B
  (body + all 11 comments); `X-Engine: browser` + `X-Timeout: 60` (5,374 B) adds the writeup-card author/citation block; the
  `/competitions/.../discussion/250060` form returned a 157 B shell and was not used
- **TL;DR:** A four-member weighted average: an NN+GBT recipe copied from a public kernel, an Embeddings→Conv1D→Residual NN with KNN
  features, a 1DCNN+2DCNN+Residual NN, and LightAutoML.
- Seven sentences, no parameters, no scores, no folds — the whole method is the member list.
- **Architecture:** FLAT · stages=1 · l1=4 members (NN+GBT, Embedding-Conv1D-Residual NN + KNN features, 1DCNN+2DCNN+Residual NN,
  LightAutoML) · l2=none · novel=MLP+embedding · mod=none ·
  topo=[NN+GBT, Embedding→Conv1D→Residual NN (+KNN features), 1DCNN+2DCNN+Residual NN, LightAutoML] → weighted average
  - One blend stage, no member's output is refed as a feature and no meta-learner is trained, hence `FLAT` (ruling 1).
  - `AUTOML` is deliberately NOT the primary: LightAutoML is one of four members and the submitted blend is his own (ruling 6).
  - `novel=MLP+embedding` is the taxonomy's label for the author's own words "Simple NN (Embeddings -> Conv1D -> Residual)" plus
    "KNN features"; the 1DCNN+2DCNN+Residual member is a second hand-rolled Keras CNN. These are 2021 Keras recipes taken from public
    kernels, not a named tabular-Transformer family, and no later-season architecture name is implied by the tag.
- **Setup:** rows/cols, class count, split sizes, submission slots: not stated anywhere on the page.
- Structural quirk exploited: not stated.
- **Features:** "KNN features" are the only named feature work, attached to member 2 (from Remek Kinas's kernel).
- No engineered column names, formulas, encodings, dropped columns or feature-source kernels beyond the two credited kernels.
- **Models:** four members, described only at architecture level; all hyperparameters, seeds, epochs and fold counts not stated:
  - member 1: "NN + GBT similar from" hiro5299834's kernel `tps06-nns-gbts-optimization` (what that NN/GBT split is inside his blend is
    not stated).
  - member 2: "Simple NN (Embeddings -> Conv1D -> Residual) + KNN features" from the kernel
    `remekkinas/keras-tuner-knn-features-simplex-optimization`.
  - member 3: "Simple NN (1DCNN +2DCNN + Residual)".
  - member 4: "LightAutoML".
- Library versions: not stated.
- **CV:** no CV number, splitter or fold protocol (`source silent`); no trust verdict either.
- **Ensembling:** "simple ensemble (weighted averaging) from some models" — weights not stated, member count = 4.
- Shared-OOF usage: he credits and reuses public *kernels* (methods), and never mentions using other people's prediction files.
- Number of models actually averaged: 4 as listed; member 1 may itself bundle an NN and a GBT (not stated).
- **Post-processing:** not stated.
- **Gains:** NOT STATED — no delta, no per-member score, no leaderboard number appears on the page.
- Ordering is not even given: the list is a member inventory, and he credits the community as the reason he placed ("I would like to
  thank the notebooks shared in this competition, I wouldn't be here without them!").
- **Failed:** nothing is reported as failed.
- Anti-knowledge of record is what he never answers (see `Comments`), plus the silence on member weights — a reader cannot tell whether
  LightAutoML or the CNNs carried the 2nd place.
- **Comments:** 11 comments; 2 of them deleted (both "This comment has been deleted." markers on the page); the author's replies
  contain no technical content (three thank-you replies, to remekkinas, to kailai (1st), and to hiro5299834 — "your kernel help me
  alot").
  - Unanswered: Saurav Joshi (456th) asks exactly the replication question this file exists for — "Can you please explain the 3rd model
    which is Simple NN (1DCNN +2DCNN + Residual) ... I had created a Triple Layered 1d CNN but the inclusion of 2DCNN is confusing me."
  - Remek Kinas (120th, kernel author) confirms his notebook was used: "I am really happy that my notebook was helpful."
- **Compute:** not stated.
- **Artifacts:** (verbatim from the page; not fetched)
  - https://www.kaggle.com/hiro5299834/tps06-nns-gbts-optimization (body citation, member 1)
  - https://www.kaggle.com/remekkinas/keras-tuner-knn-features-simplex-optimization (body citation, member 2)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2021/writeups/khanhvd-2nd-place-solution (citation line)
  - https://www.kaggle.com/duykhanh99 (author), https://www.kaggle.com/hiro5299834, https://www.kaggle.com/remekkinas
  - comment links: https://www.kaggle.com/cv13j0, https://www.kaggle.com/sauravjoshi23, https://www.kaggle.com/kailai,
    https://www.kaggle.com/dathudeptrai, https://www.kaggle.com/uniabhi
- **Lesson:** In the 2021 Playground, rebuilding two popular public kernels plus LightAutoML and averaging them was a silver medal —
  but a writeup with no weights and no scores is a member list, not a recipe.

## TPSJUN21 — consensus recipe
- **Architecture distribution:** 1st `STACK2` (~15 XGBoost models → ~90 class-relabelled prediction columns → `neuralnet` MLP meta,
  MLPs then weighted-averaged) · 2nd `FLAT` (weighted average of 4 members: NN+GBT, embedding-CNN+KNN NN, 1D/2D CNN NN, LightAutoML).
  - **winner topology: `STACK2`** — the gold went to the only entrant here that trains a meta-learner, and it is an *NN* meta-learner,
    which is why the 1st place's stage-2 architecture (a plain 63/27 sigmoid MLP) matters more than his level-0 diversity.
  - Silver used no stacking at all: 2nd's blend never refeeds predictions, so the 1st-vs-2nd difference on this board is
    meta-learner-vs-average, and neither page publishes the score gap (TPSJUN21-01, TPSJUN21-02).
- **New architectures at the board:** one, and it is at 2nd, not 1st.
  - `MLP+embedding` (TPSJUN21-02): "Simple NN (Embeddings -> Conv1D -> Residual)" plus a "1DCNN + 2DCNN + Residual" NN, both rebuilt
    from public kernels; no score is attached to either because the page publishes none.
  - 1st is `novel=none`: XGBoost plus a stock 2-layer sigmoid MLP as the meta-learner (the unusual choice in 2021 was an NN blender,
    flagged by a commenter: "Nice to see neural nest to the rescue, and even not using relu activations").
  - Across both pages there is no transformer, no named tabular-NN family (TabPFN/FT-Transformer/Node-style), and no vendor-managed
    submitted stack: LightAutoML appears once, as one `FLAT` member (TPSJUN21-02).
- **Agreed on (2 of 2 — TPSJUN21-01, TPSJUN21-02):** neither entrant engineers features from the raw 2021 Playground columns; 1st's
  stage-2 input is "No original features" and 2nd's only named feature work is kernel-supplied "KNN features".
- **Agreed on (2 of 2 — TPSJUN21-01, TPSJUN21-02):** a neural network is inside the submitted prediction, and neither page publishes a
  fold scheme or a CV number, so both are unreproducible score-for-score (the only seed anywhere is 1st's meta `random.seed = 8888688`;
  2nd publishes no parameters at all).
- **Stated by only 1 of 2 (TPSJUN21-01):** GBMs alone were not the answer — LightGBM is discarded ("its cv is not good") and XGBoost is
  used only as a column generator for the meta. TPSJUN21-02 never reports a GBM verdict; his only tree content is inside the LightAutoML
  and "NN + GBT" members, with no weights given.
- **Divergences:** 1st built depth (2 stages, ~90 stacked columns) where 2nd built breadth (4 parallel members, one stage); the only
  quantified number in the whole file is 1st's +0.00010 private gain for adding NN columns to his stack (1.73900 → 1.73890), which
  says stacking *width* bought nearly nothing at that point.
- 1st published every XGBoost and MLP hyperparameter he used and no score; 2nd published no hyperparameter and no score. For
  replication the useful half of this file is therefore TPSJUN21-01 plus the browser-recovered comment.
- **Highest-leverage single trick:** multiply the stage-1 column count without adding model families — rotate the class order and merge
  the weak classes (class_4+class_5, class_4+class_5+class_1 vs the rest) so ~15 XGBoost runs yield ~90 decorrelated prediction
  columns for the meta (TPSJUN21-01).
- **Nothing worked:** LightGBM at level 1 ("Tried lightgbm, but its cv is not good"), hand-done scaling/normalizing (unnecessary;
  `neuralnet` standardizes), and any further stage-2 feature growth (the reported private gain of adding NN predictions was
  0.00010) — all from TPSJUN21-01.
- TPSJUN21-02 reports nothing as failed; the absence is stated here rather than filled by the other entry.
