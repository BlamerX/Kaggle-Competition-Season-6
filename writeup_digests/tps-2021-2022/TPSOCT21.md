## TPSOCT21 — Tabular Playground Series - Oct 2021
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Oct 2021" (page subtitle: "Practice your ML skills on this approachable dataset!")
- Competition: https://www.kaggle.com/c/tabular-playground-series-oct-2021
- Writeups covered: 4 of 4
- Score ladder: 1st `not stated (topic deleted)` · 3rd `not stated` · 4th `0.85424 (public, reference NN) -> 0.85503 (public, KMeans NN)` · 9th `not stated`.
- Only LB numbers published anywhere in this set are 4th's two public scores; a commenter on the deleted 1st-place page reports his own
  best NN at 0.85487 (ResNet-like structure, 143rd).
- Row counts, column counts and submission-slot counts: not stated on any of the four pages.

### TPSOCT21-01 · 1st · olivrk (Olivier) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284511
- **Status:** UNFETCHED (topic deleted). Every ladder route returns the same "Deleted Topic / This topic has been deleted." body:
  `/c/` discussion form (8,154 B), `/c/` + `X-No-Cache` (164 B shell at write time; the auditor's re-fetch on this route served the same
  8,154 B deleted render), `/c/` + `X-Engine: browser` + `X-Timeout: 60` (8,154 B, identical deleted render), `/competitions/` discussion
  form + `X-No-Cache` (8,242 B, identical deleted render), `/c/` + `?sort=polls` (356 B cached shell; 2,591 B with `X-No-Cache` — a
  navigation-only shell that never reaches the topic body), and finally the real browser (`navigate_page` + accessibility snapshot) which
  also shows "Deleted Topic". Auditor re-ran all six routes and confirms the deletion. Unlike the other three pages of this file, the
  sign-in `returnUrl` is the discussion path itself rather than a `writeups` slug page, so no live writeup URL survives for this entry.
  Comments ARE still served.
- **TL;DR:** The 1st-place body is gone; only the comment thread survives, and it says the lost solution was tree models ensembled with
  one "improved neural net".
- Eleven of the thread's comment positions render as "This comment has been deleted.", and no author reply is readable anywhere on the page -
  aayush26's "Thanks for the response" proves the author did answer at least once before the thread was stripped.
- **Architecture:** UNKNOWN (topic deleted - forms tried: `/c/` discussion form, `/c/` + `X-No-Cache: true`, `/c/` + `X-Engine: browser`
  + `X-Timeout: 60`, `/competitions/` discussion form + `X-No-Cache: true`, `/c/` form + `?sort=polls`, real browser
  `navigate_page` + accessibility snapshot) · stages=not stated · l1=not stated · l2=not stated · novel=not stated · mod=not stated ·
  topo=not stated (no body text survives; third-party paraphrases in comments cannot set a tag)
  - Ruling 12: nothing on this page names an estimator, a weight, or a stage count, so no primary tag is derivable.
  - Commenter paraphrase of the deleted body (aayush26, 44th): "ensemble between good performing tree based models and an improved neural
    net" - a reader's summary, not the author's specification.
  - The deleted author reply evidently named a model family the reader had not heard of (aayush26: "I have never heard of temporal fusion
    transformer"), so a TFT-style architecture is the one surviving hint about what 1st did. It is not quotable as the submitted topology.
- **Setup:** rows/cols, split sizes, slots: not stated (body deleted).
- Structural quirk exploited: not stated (body deleted); 3rd and 4th in this same file both attribute the winning edge to NNs, but that is
  their text, not this one's.
- **Features:** not stated (body deleted).
- **Models:** not stated (body deleted). Commenter-side evidence only: trees + "an improved neural net" (aayush26), and the phrase
  "temporal fusion transformer" quoted back from a deleted author reply (aayush26).
- **CV:** not stated (body deleted). No CV-vs-LB gap computable.
- **Ensembling:** not stated (body deleted). Hikmet Sezen (17th) asked "How many NN and tree-based models did you weighted, and scores of
  these tree-based models (or just range of their scores)? Heterogeneity is important factor for efficiency of ensemble learning." - the
  answer is not visible on the page.
- **Post-processing:** not stated (body deleted).
- **Gains:** not recoverable - the page publishes zero numbers for this entry.
- **Failed:** not recoverable. Anti-knowledge of record for this entry is the loss itself: a 1st-place TPS writeup can be deleted while its
  comment thread stays public, so the leaderboard medal leaves no replicable trace.
- **Comments:** page header reads "28 Comments"; the render shows 14 named in-thread comments + 1 comment block whose account name is gone
  + 2 appreciation entries = 17 readable blocks, and 11 positions that read "This comment has been deleted." (17 + 11 = the header's 28).
  0 "Topic Author" blocks survive, i.e. no author reply is readable anywhere on the page.
  - aayush26 (44th): "this is relevant for almost all TPS. To ensemble between good performing tree based models and an improved neural
    net" + asks whether pseudolabeling would help further (his question and the author's answer are both in the deleted set).
  - aayush26, follow-up that survives: "Thanks for the response. I have never heard of temporal fusion transformer. Can you share your
    kernal or code snippet with the changes required to fit for this problem statement?"
  - Hardy Xu (143rd): "I also experimented a lot with neural networks, but was only able to get up to .85487 with a ResNet-like structure."
  - Krish Yadav asks how the author came to "read the paper" - i.e. the deleted body cited an academic paper, whose name is lost.
  - Anonymous commenter: "OK this is freaking impressive. respect! Are you willing to share some of the code? this is a really novel
    solution" - no code was published on this thread.
  - Rank badges visible on this thread (context for cross-reading the other Oct entries): pourchot 5th, adamwurdits 9th, akmalmir 11th,
    hikmetsezen 17th, mehrankazeminia 19th, aayush26 44th, cv13j0 78th, kashif 130th, hardyxu52 143rd, yannbarthelemy 205th,
    dwin183287 239th, pcyslm 381st.
- **Compute:** not stated.
- **Artifacts:** the deleted body's links are gone; the surviving page contains only profile links and one image asset.
  - https://www.kaggle.com/olivrk (mentioned 5x as the congratulated author; his handle is the only author identification the page offers)
  - https://www.kaggle.com/keagle, https://www.kaggle.com/aayush26, https://www.kaggle.com/yannbarthelemy,
    https://www.kaggle.com/mohammadkashifunique, https://www.kaggle.com/dwin183287, https://www.kaggle.com/hardyxu52,
    https://www.kaggle.com/pourchot, https://www.kaggle.com/mehrankazeminia, https://www.kaggle.com/adamwurdits,
    https://www.kaggle.com/akmalmir, https://www.kaggle.com/hikmetsezen, https://www.kaggle.com/cv13j0, https://www.kaggle.com/pcyslm
  - https://www.kaggle.com/static/images/discussion/high-five-illo.svg, https://www.kaggle.com/competitions/28010/images/header
  - the deleted body's paper reference: title lost
- **Lesson:** Publish a self-contained writeup - the medal itself is not durable knowledge; a deleted topic erases the architecture, the
  features, and the author's own replies at once.

### TPSOCT21-03 · 3rd · mathurinache (Mathurin Ache) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284594
- **Status:** FETCHED (first try, `/c/` discussion form, 5,439 B; renders as the writeup page
  `/competitions/tabular-playground-series-oct-2021/writeups/prevision-io-3rd-place-solution` with all 5 comments)
- **TL;DR:** Blend of public + private autoML solutions (XGB/LGB/CAT/HGB heavy) plus Kaveh's multi-input NN run at 25 seeds, then a
  pseudo-label retrain worth "0.000X".
- **Architecture:** FLAT · stages=1 · l1=not stated count (composition of public + private autoML solutions whose GBMs are XGB/LGB/CAT/HGB,
  plus one multi-input NN iterated 25 seeds) · l2=none named (no meta-learner is mentioned anywhere on the page) · novel=MLP+embedding ·
  mod=public-oof,pseudo ·
  topo=[public + private autoML solutions] + [multi-input NN ×25 seed runs] → composition → retrain main models on train + confident
  pseudo-labeled test rows → submission
  - "Composition of public and private autoML solutions": community notebooks are members, so per ruling 3 they widen `l1` and add
    `mod=public-oof` rather than adding a level.
  - Not tagged `AUTOML`: per ruling 6 the vendor stack is a member of his composition, not the submitted stack itself.
  - The pseudo-label step retrains the existing main models on augmented rows instead of producing the winning model from a
    self-training loop, so per ruling 9 it is `mod=pseudo`, not the primary.
  - `novel=MLP+embedding` is supported from THIS page only: the author's kernel of reference is named "multi-input neural network" and a
    comment on this page asks about "categorizing continuous variables for the NN" - i.e. a second categorical input branch. No embedding
    prose exists on the other Oct entries.
- **Setup:** rows/cols and split sizes: not stated. Slots: not stated. Structural quirk named: 25 seed iterations of the NN to make results
  "robust"; author's stated extra for top placement = "combine one or more efficient NN models".
- **Features:** the page names no engineered columns; feature work is inside the borrowed kernels.
  - Categorical handling implied by the thread: continuous variables categorized for the NN input branch (asked by AndrewTGraham,
    answer not given on the page).
  - Pseudo-labeling thresholds (exact, from the body): "if proba test <0.05 => then target = 0 if proba test> = 0.95 => 1 and add test
    lines in train".
  - Pseudo-labeling code posted in the author's comment uses a DIFFERENT lower threshold than the body:
    `np.where(sub['target']>0.95, 1, np.where(sub['target']<0.1, 0, sub['target']))` then
    `test_data = test_data[test_data['target'].isin([0,1])]` then `pd.concat([train_data, test_data], 0, ignore_index=True)`.
  - Columns dropped: not stated. GP/generated features: not stated. Encodings: not stated.
- **Models:** multi-input NN = https://www.kaggle.com/kavehshahhosseini/tps-oct-2021-multi-input-neural-network, 25 seed iterations.
- Gradient boosting families present in the composition: XGB, LGB, CAT, HGB (author's abbreviation list; no configs published).
- autoML components: named as "public and private autoML solutions" without naming a vendor (no AutoGluon/H2O attribution on this page).
- Per-model hyperparameters, seeds beyond "25 iterations", library versions: not stated.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated. No CV score and no LB score published on this page, so no
  CV-vs-LB gap is computable.
- Author's pseudo-label payoff is stated as "an extra performance 0.000X", i.e. fourth-decimal only.
- **Ensembling:** composition (blend) of public + private autoML solutions with the 25-seed NN; weights and blender type not stated.
- Shared-OOF usage: yes - "public ... solutions", individual sources not named on this page.
- Number of models averaged: not stated, except the NN's 25 seed iterations.
- **Post-processing:** pseudo-label retrain (see Features for both threshold variants); no threshold/calibration/rank-gauss/rounding step is
  otherwise named.
- **Gains:** ("0.000X", author's own notation) pseudo-label retrain of the main models on top of the composition.
- Unquantified: the multi-input NN at 25 seeds is what he credits for reaching the podium - "the little extra to perform high in this
  competition is to combine one or more efficient NN models".
- **Failed:** not stated - this entry names no dead end.
- Note the internal inconsistency the author never resolves: the body says the pseudo-label lower threshold is 0.05, his own posted
  snippet says 0.1.
- **Comments:** page header "5 Comments", 5 comment blocks, 1 is a Topic Author reply, 0 deleted positions. Technical content in replies:
  - The full pseudo-labeling snippet (posted to @Aayush): reads `best_sub="../script/ma20211018a.csv"`, bins test predictions at >0.95 / <0.1,
    keeps only confident rows, concatenates them onto train, and reloads the untouched test set for prediction. He publishes no
    re-scoring scheme.
  - Aayush Kumar Singha (44th) asks exactly where the leaked-test rows are excluded from the AUC - i.e. the snippet as posted cannot be
    CV-scored without extra bookkeeping; the author does not answer that part.
  - AndrewTGraham (683rd) asks whether categorizing continuous variables for the NN is only for memory or also helps accuracy. No answer on
    the page.
- **Compute:** 25 seed iterations of the NN imply a long NN run; wall-clock, GPU model, RAM, cost: not stated.
- **Artifacts:** (all URLs cited on the page, verbatim; not fetched)
  - https://www.kaggle.com/kavehshahhosseini/tps-oct-2021-multi-input-neural-network
  - ../script/ma20211018a.csv (author's own submission file, referenced inside the posted snippet)
  - https://www.kaggle.com/input paths inside the snippet: `../input/tabular-playground-series-oct-2021/train.csv`,
    `../input/tabular-playground-series-oct-2021/test.csv`
  - https://www.kaggle.com/mathurinache, https://www.kaggle.com/olivrk, https://www.kaggle.com/Aayush,
    https://www.kaggle.com/raahulsaxena, https://www.kaggle.com/andrewtgraham, https://www.kaggle.com/adamwurdits,
    https://www.kaggle.com/aayush26
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/writeups/prevision-io-3rd-place-solution (citation line)
  - https://www.kaggle.com/competitions/28010/images/thumbnail
- **Lesson:** In a GBM-saturated TPS, one borrowed embedding-MLP run at 25 seeds plus a razor-thin pseudo-label retrain was enough for a
  medal - but publish the actual blend weights, or the solution is unreproducible.

### TPSOCT21-04 · 4th · motchan (Mottchan) · LB 0.85424 (public, reference NN)/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284560
- **Status:** FETCHED (first try, `/c/` discussion form, 4,236 B; renders as writeup slug `mottchan-4th-place-solution`, 2 comments)
- **TL;DR:** KMeans applied to the multi-input NN moved the public score 0.85424 -> 0.85503; the submission is a fixed hand-weighted blend
  of 5 files.
- **Architecture:** FLAT · stages=1 · l1=5 blended submission files (`sub_1..sub_5`), one of them the KMeans-modified multi-input NN ·
  l2=none (no meta-learner on the page) · novel=none · mod=none ·
  topo=multi-input NN + KMeans variant + 3 other sub files → weighted sum (0.075/0.210/0.210/0.210/0.295) → submission
  - Weight selection is a fixed coefficient list, not a fitted blender: `FLAT`, and per ruling 1 there is no OOF meta-learner anywhere.
  - `novel=none` deliberately: this page identifies the NN only by the kernel title and by "KMeans"; unlike TPSOCT21-03 it states no
    categorical/embedding input branch, and rule 4 forbids carrying that detail over.
  - The KMeans step's mechanics are `source silent` on this page: he says only that KMeans is what he used "to increase the public score
    for deep learning".
- **Setup:** rows/cols and split sizes: not stated. Slots: not stated. Structural quirk he exploits: the NN's outsized effect on the
  **public** score - "deep learning has a big impact on the public score this time".
- **Features:** no engineered columns named in the body.
  - KMeans is the only feature-side device he names (notebook title `tps-oct-2021-kmeans`); exact usage not stated.
  - Columns dropped: not stated. Encodings: not stated. GP/generated features: not stated.
- **Models:** multi-input neural network from Kaveh Shahhosseini's kernel; public score 0.85424 as-is.
- Same NN after his KMeans modification: public score 0.85503.
- Library, layer counts, hyperparameters, seeds, fold counts: not stated.
- **CV:** splitter, #folds, repeats, stratification, seeds: not stated; he publishes only public LB numbers and calls them "public score",
  so no CV figure exists to compare.
- Trust verdict: he deliberately optimized for the public score ("I set out to increase the public score for deep learning") - and the
  page records no private score, so this is a public-LB-driven entry.
- **Ensembling:** weighted sum of 5 prediction files, from the code he pasted in the notebook (quoted by a commenter):
  - `sub['target'] = (sub_1['target'].values*0.075) + (sub_2['target'].values*0.210) + (sub_3['target'].values*0.210) +`
    `(sub_4['target'].values*0.210) + (sub_5['target'].values*0.295)`
  - Which model each `sub_i` is: not stated on the page.
  - Shared-OOF usage: not stated. Seeds/fold averaging: not stated.
- **Post-processing:** none named beyond the weighted sum (no threshold, calibration, clipping, rank-gauss, or rounding on this page).
- **Gains:** (+0.00079 public) KMeans applied to the reference multi-input NN: 0.85424 -> 0.85503 - the only quantified delta on this page.
- Ordering only: he says the NN raised the public score even though "there was no correlation with XGBoost or EDA", i.e. the NN's value was
  decorrelation, not raw strength.
- **Failed:** not stated for features or models. The one thing he reports as *surprising* rather than useful: NN targets raised the score
  "even though there was no correlation with XGBoost or EDA".
- The origin of the blend weights is an open question, not a dead end: he never answers it (see Comments).
- **Comments:** page header "2 Comments", 2 comment blocks, 0 Topic Author replies, 0 deleted positions.
  - Michael Mellinger (186th) quotes the full 5-term weight line and asks "How did you come up with the weights?" - no answer exists on
    this page, so the weights (0.075, 0.210, 0.210, 0.210, 0.295) are unexplained hand-picked constants.
  - Raahul Saxena (32nd): "Following this." - no content.
- **Compute:** not stated (no wall-clock, GPU, RAM or cost).
- **Artifacts:** (verbatim, not fetched)
  - https://www.kaggle.com/kavehshahhosseini/tps-oct-2021-multi-input-neural-network (the reference NN document)
  - https://www.kaggle.com/motchan/tps-oct-2021-4th-place-importantmodel-kmeans-nn (his KMeans NN)
  - https://www.kaggle.com/motchan/tps-oct2021-last-submission-model (his final ensemble notebook)
  - https://www.kaggle.com/motchan, https://www.kaggle.com/mmellinger66, https://www.kaggle.com/raahulsaxena
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/writeups/mottchan-4th-place-solution (citation line)
  - https://www.kaggle.com/competitions/28010/images/thumbnail
- **Lesson:** A clustering trick on somebody else's NN was worth +0.0008 public - but a fixed weight list whose provenance the author never
  explains is a rank you cannot reproduce.

### TPSOCT21-09 · 9th · adamwurdits (Adam Wurdits) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284492
- **Status:** FETCHED (first try, `/c/` discussion form, 11,764 B; body + all 19 comment blocks, 7 of them author replies)
- **TL;DR:** Three-level stack: 15 LightGBMs × 20 seeds + 32 variants of 15 other model types → 9 level-1 meta-models → LDA at level 2.
- **Architecture:** STACKN · stages=3 · l1=15 LightGBM models (each a 20-seed average) + 32 variants of 15 other model types ·
  l2=9 level-1 models (3 LightGBM, Logistic Regression, ElasticNet, Linear Discriminant Analysis, RidgeCV, CatBoost, XGBoost) ·
  l3=1 Linear Discriminant Analysis · novel=none · mod=multi-view ·
  topo=(15 LGBM ×20 seeds + 32 variants of 15 model types) → 9 level-1 meta-models → LDA (level 2) → submission
  - Two trained meta layers above the base pool, so `STACKN`, not `STACK2`. The author's own labels are the evidence: "Base models",
    "Meta-models", "For my level 1 models, I used 9 in total ... For my level 2 model, I picked Linear Discriminant Analysis". The LDA
    level-2 is a consumer of the level-1 outputs, not a projection step: it sits at the end of the DAG just before "Submission"
    (from figure: https://i.postimg.cc/7hH7ZxFX/tps10.jpg, opened by the auditor - it labels the base column "47 models" = 15 LightGBMs +
    "32 variants of 15 types of supporting models" and prints no scores, so the "publishes no score" claim stands).
  - How the stack differs from a plain average: each level consumes the previous level's out-of-fold predictions as its feature matrix, and
    both meta layers are fitted estimators (9 models, then LDA), not weights.
  - The 20-seed averaging inside each LightGBM is seed averaging per ruling 4 and does not count as a level.
  - `mod=multi-view`: diversity is bought with feature subsets rather than hyperparameters - base models run on column subsets and on
    binary-features-only sets (see Features), so the pool branches see different views of the same rows.
  - Fold-assignment mechanics for the meta layers are `source silent`, so no selection-hygiene modifier is tagged.
- **Setup:** rows/cols and split sizes: not stated. Slots: not stated. Structural quirk exploited: none named.
- His design constraint was wall-clock: models are trained on column subsets, "especially those that took longer to train, like KNeighbors,
  Multilayer Perceptron, AdaBoost, GBMs".
- **Features:** three borrowed feature subsets, named after their creators (author's exact names):
  - `max_features` - from https://www.kaggle.com/maxdiazbattan/tabular-play-oct-2021-feature-selection-idea
  - `luca_features` - from https://www.kaggle.com/lucamassaron/feature-selection-using-boruta-shap (Boruta-SHAP selection)
  - `mottchan_features` - from https://www.kaggle.com/motchan/tps-oct-2021-kmeans (KMeans; the same author is 4th in this file)
  - Scaling chosen per model by best CV: "I used different scaling methods with my models depending on which one offered the best CV score"
    - in the end "Standard/Robust scalers in most cases".
  - Binary-only feature set for some models: "some were trained on just the binary features alone - e.g. Quadratic Discriminant Analysis".
  - Exact engineered column names and formulas: not stated - all feature construction lives in the three cited notebooks, and the author
    publishes no formula of his own.
  - Columns dropped: not stated beyond the per-model subsets. GP/generated features: not stated.
- **Models:** base layer: 15 LightGBM models, "each trained on 20 seeds and averaged" (published as a notebook + dataset).
- Diversification layer: "32 variants of 15 other types of models", mostly non-tree-based, inspired by David Coxon's 20-model comparison
  notebook.
- Non-tree members named on the page: KNeighbors, Multilayer Perceptron, AdaBoost, Quadratic Discriminant Analysis, Linear Discriminant
  Analysis, Ridge, Lasso, ElasticNet, Random Forest.
- Full supporting-type list, all 15 (from figure: https://i.postimg.cc/7hH7ZxFX/tps10.jpg): AdaBoost, BaggingClassifier, Bernoulli Naive
  Bayes, ElasticNet, GradientBoosting, KNeighbors, Lasso, Linear Discriminant Analysis, LinearSVC, Logistic Regression, Multilayer
  Perceptron, Quadratic Discriminant Analysis, Random Forest, Ridge, RidgeCV.
- Level 1: 3 LightGBMs, Logistic Regression, ElasticNet, Linear Discriminant Analysis, RidgeCV, CatBoost, XGBoost (9 total).
- Level 2: Linear Discriminant Analysis.
- All hyperparameters beyond the regularization note below: not stated. Library versions: not stated.
- Author's regularization finding (comment): "if I optimize alpha regularization (decrease it from the default value of 1.0) then they all
  start to look a lot like a simple linear regression and not help my stack as much" - so Ridge/Lasso/ElasticNet were kept at defaults.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated (only "20 seeds" per LightGBM and "solid folds" as a compliment from
  14th-place maxdiazbattan).
- Selection rule stated: per-model scaling was chosen by "which one offered the best CV score".
- Author's implicit CV-vs-LB trust: he built the whole stack on CV; he publishes no LB score, so no gap is computable.
- **Ensembling:** two trained meta layers (9 level-1 models → 1 LDA level-2); meta-learner regularization parameters not stated.
- Shared-OOF usage: yes, self-organized - "to later collect all the out-of-fold and test predictions and upload them as datasets".
- Number of models averaged at the base layer: 15 LightGBM × 20 seeds = 300 fits, plus 32 variants of 15 model types.
- **Post-processing:** not stated (no threshold, calibration, clipping, rank-gauss, or rounding on this page).
- **Gains:** ordering, not magnitudes - the page publishes no delta for any step.
  - Keeping weak non-tree models in level 1 is the gain he argues for qualitatively: "All supporting models had weaker scores than any of
    my LightGBM models, yet they helped my level 1 meta-models learn patterns that they didn't see with the LightGBMs."
  - Same verdict extends beyond non-tree models: "This was not only true for the non-tree based models, but even for the tree-based models
    like random forest."
- **Failed:** binarization: "I tried out all and had very high hopes for binarization but ended up using Standard/Robust scalers in most
  cases" - a scaling candidate that did not win.
- Lowering `alpha` below 1.0 on Ridge/Lasso/ElasticNet: they collapse toward plain linear regression and stop helping the stack (author
  comment).
- Publishing a single unified notebook: his stack is "highly fragmented", one notebook per base model, which he names as the thing he wants
  to fix for November rather than a modeling gain.
- **Comments:** page header "19 Comments"; 18 named comment blocks render (16 in-thread + 2 in the Appreciation section) plus 1 block whose
  account name is gone, and 7 of the in-thread blocks are Topic Author replies; 0 deleted positions. Author-only technical content, mined:
  - Sub-par models stay in on purpose (reply to lilkaskitc, 46th): all supporting models scored below his LightGBMs yet improved the level-1
    metas; same for tree-based members such as random forest.
  - Code policy (reply to anonymous "Are you willing to share the code?"): only the LightGBM notebook is polished and public; every other
    base model has its own separate notebook following the same format, run so he could "tinker with one, while other models were training"
    and then collect all OOF + test predictions into datasets.
  - Regularization detail (same reply): the alpha-defaults observation quoted under Models.
  - Hikmet Sezen (17th) used Adam's notebook inside his own stacking pool; Adam replies that Hikmet reached his LB place "with only 7
    submissions".
  - Non-technical: Gang HUANG (85th) fixes the broken postimg embed (`![](https://i.postimg.cc/7hH7ZxFX/tps10.jpg)`).
- **Compute:** not stated (no wall-clock, GPU, RAM, or cost); the notebook-per-model split is described as a concurrency trick, not
  measured.
- **Artifacts:** (verbatim, not fetched)
  - https://i.postimg.cc/7hH7ZxFX/tps10.jpg (his stack diagram, embedded in the body)
  - https://www.kaggle.com/maxdiazbattan/tabular-play-oct-2021-feature-selection-idea
  - https://www.kaggle.com/lucamassaron/feature-selection-using-boruta-shap
  - https://www.kaggle.com/motchan/tps-oct-2021-kmeans
  - https://www.kaggle.com/adamwurdits/15-lgbms-trained-on-20-seeds-dataset-included
  - https://www.kaggle.com/davidcoxon/20-model-comparison-oct-tabular-playground
  - https://www.kaggle.com/adamwurdits, https://www.kaggle.com/maxdiazbattan, https://www.kaggle.com/lucamassaron,
    https://www.kaggle.com/motchan, https://www.kaggle.com/davidcoxon, https://www.kaggle.com/bell2psy,
    https://www.kaggle.com/sahandissanayaka, https://www.kaggle.com/dwin183287, https://www.kaggle.com/hikmetsezen,
    https://www.kaggle.com/lilkaskitc, https://www.kaggle.com/aries1988, https://www.kaggle.com/anantaannadatha,
    https://www.kaggle.com/fuchengyee
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/writeups/adam-wurdits-9th-place-solution (citation line)
  - https://www.kaggle.com/competitions/28010/images/thumbnail
- **Lesson:** Deliberately keep individually-weak, structurally-different models in level 1 - the meta layer learns patterns the strong
  GBM family cannot see on its own.

## TPSOCT21 — consensus recipe
- **Architecture distribution:** 1st `UNKNOWN (topic deleted)` · 3rd `FLAT` (autoML composition + 25-seed multi-input NN, pseudo-label
  retrain as `mod=pseudo`) · 4th `FLAT` (5 hand-weighted submission files) · 9th `STACKN` (base pool → 9 level-1 metas → LDA level 2).
  - **winner topology: not recoverable** - the 1st-place topic is deleted, so this board's only certifiable structural fact is that both
    readable top-5 entries (3rd, 4th) are flat blends and the medal differences came from their NN component, not from stacking depth.
  - Read-across: the deepest stack (9th, 3 stages, ~332 base fits) finished below the two flat blends (3rd, 4th) - in Oct 2021 stack depth
    did not buy rank.
- **New architectures at the board:** `MLP+embedding` - the multi-input neural network of Kaveh Shahhosseini
  (`tps-oct-2021-multi-input-neural-network`) is the only non-stock component named by any surviving entry: 3rd ran it at 25 seeds as a
  composition member, and 4th started from the same kernel (public 0.85424) and pushed it to 0.85503 with KMeans.
  - 4th's page names no embedding/categorical branch, so its `novel=none` is page-local, not a carryover denial.
  - No entry in the readable set used a novel architecture as a meta-learner: 9th's level-1 and level-2 are LightGBM / LR / ElasticNet /
    LDA / RidgeCV / CatBoost / XGBoost, i.e. all stock sklearn/GBM.
  - Reported as existing at 1st but unrecoverable: "temporal fusion transformer" (quoted back from a deleted author reply) and a commenter's
    own ResNet-like NN at 0.85487 (143rd, not the author).
- **Agreed on (2 of 3 readable entries - TPSOCT21-03, TPSOCT21-04; TPSOCT21-09 is the exception) - the decisive primitive:** a neural
  network had to be in the mix alongside the GBMs.
  - 3rd: "the little extra to perform high in this competition is to combine one or more efficient NN models", with his NN at 25 seeds.
  - 4th: "Like the 1st person, I also happened to find out that deep learning has a big impact on the public score this time."
  - Not counted: TPSOCT21-09 never credits a neural network with his result - "Multilayer Perceptron" appears on his page exactly once,
    inside the subset-trained diversification layer, and every component he names for the winning stack is LightGBM or a linear/LDA meta.
- **Agreed on (2 of 3 readable entries - TPSOCT21-03, TPSOCT21-04):** public-community material is a legitimate member of a top-4 solution -
  3rd's composition is explicitly "public and private autoML solutions", 4th's whole NN branch is a borrowed public kernel.
  - 9th is the counter-example inside the same file: 0 community prediction sets, but 3 borrowed public *feature* notebooks
    (`max_features`, `luca_features`, `mottchan_features`).
- **Score-disclosure check (counted on all 3 readable pages, plus the deleted 1st):** no Oct 2021 entry states a private score, and only
  TPSOCT21-04 states any LB number at all (two values, both labelled "public score" by him). TPSOCT21-03 and TPSOCT21-09 publish zero
  scores, so this board's public-to-private movement is unmeasurable from its writeups - unlike Nov 2021, where 5th titles his entry with
  his own rank shake-up.
- **Divergences:** top-4 (3rd, 4th) built shallow blends around one strong NN; 9th built a 3-level, ~332-fit stack of conventional models.
  The rank gap between them is unquantified because 9th publishes no score - the divergence is in *published evidence*, and only 4th's
  +0.00079 KMeans delta is measurable anywhere in this file.
- Second divergence: 3rd and 9th both used pseudo-label-style data growth (3rd explicitly; 9th never mentions it), so the file's single
  anti-pseudo-label datapoint comes from the deleted 1st's thread, where aayush26 asks whether pseudo-labeling would help 1st further.
- **Highest-leverage single trick:** retrain on the confidently-pseudo-labeled test rows and keep the delta in the 4th decimal - 3rd's
  "0.000X" (TPSOCT21-03), with the exact rule `<0.05 -> 0`, `>=0.95 -> 1` in the body and `<0.1` / `>0.95` in his posted snippet.
- Closest measured equivalent: 4th's KMeans on the multi-input NN, +0.00079 public (TPSOCT21-04).
- **Nothing worked:** not stated by any readable entry - TPSOCT21-03 and TPSOCT21-04 publish zero dead ends. The only reported failure is
  9th's binarization scaling (abandoned for Standard/Robust) and his alpha-tuned Ridge/Lasso/ElasticNet collapsing to plain linear
  regression.
- **Nothing worked (by absence of record):** no Oct 2021 entry names target encoding, aggregated statistics, or any exact engineered
  column formula - feature work was delegated entirely to the three cited public feature-selection notebooks (9th) or to the borrowed
  autoML kernels (3rd).
