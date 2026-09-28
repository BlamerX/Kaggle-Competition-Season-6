## TPSMAR21 - Tabular Playground Series - Mar 2021
- Task: Tabular (Binary Classification) | Metric: AUC | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Mar 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-mar-2021
- Writeups covered: 5 of 5
- Note on the metric: the index title line (used above) says binary classification / AUC; every score on these pages is in the 0.89-0.90 higher-is-better range consistent with AUC. This file therefore records MAR 2021 as AUC; no number in this file is a loss (RMSE/MAE) and no delta is signed as "lower is better".
- Score ladder (as reported by the authors; format private/public): 1st `not stated (post-competition single model 0.89999/0.89528; best in-comp single 0.89964/0.89482)` · 2nd `0.90053/0.89599 (stack CV 0.9008564)` · 3rd `topic deleted` · 9th `not stated` · 11th `not stated (top 1%)`

### TPSMAR21-01 · 1st · Dave E (davidedwards1) · LB not stated/not stated · CV 0.894-0.898 (600-epoch guide), single model 0.89964 private/0.89482 public

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229833
- **Status:** FETCHED (1st `/c/` attempt returned full 19370 bytes, body + all 19 comments)
- **TL;DR:** Feb's winner (ryanzhang's transformer-DAE) reused: 3 DAE runs feed an MLP; XGB/LGBM Optuna models join; Optuna weights over OOF make the final blend.
- Key original contribution: decay the swap-noise level every epoch instead of a fixed schedule.
- **Architecture:** FLAT · stages=1 (blend level; members are themselves DAE→MLP cascades) · l1=3 DAE-MLP models + top-5 Optuna XGB + top-5 Optuna LGBM · l2=none (Optuna weight search, ruling 1) · novel=Transformer-DAE · mod=embed,multi-view ·
  topo=[3× transformer-DAE (noise decay per epoch, 1200-1800 epochs, one extended to 2400) → DAE features → MLP inference] + [5 XGB (one-hot) + 5 LGBM (label enc)] → Optuna weight blend on OOF
  - Submitted artifact is the blend; no meta-model fitted on the OOF matrix, so `FLAT` per ruling 1, not `STACK2`.
  - Diversity came from encodings (one-hot vs label vs one run with KMeans-binned all-categorical input) and from DAE epoch-weights taken at several points.
  - Inside each DAE member the learned features become MLP *inputs* — a `CASCADE` sub-topology within level-0, not a stack level.
- **Setup:** stratified k-fold, 10 folds; "10 fold was definitely better than 5 fold"; rows/cols/slots not stated on the page.
- **Features:** the DAE runs take the raw numerical columns plus the categorical columns (the page does not state which categorical encoding the DAE itself used; label encoding is named only for LGBM, one-hot only for XGB). One DAE run binning continuous variables with KMeans to a 100% categorical feed (idea from https://www.kaggle.com/siavrez/kerasembeddings), slightly worse CV but kept for diversity.
- XGB got one-hot columns with the Optuna option to drop low-count columns (exclude any column with sum < 40).
- DAE feature output later augmented with the X reconstruction and the mask prediction (Variation 2) — "seemed to improve CV as well as giving the best single model".
- **Models:** transformer-DAE = ryanzhang's Feb code (github repo linked); learning rate 3e-04 decreasing; start noise ~50% flat across all columns; 27 DAE runs × 600 epochs evaluated by CV on the output features; 3 runs used for the submission, one with weights from several epochs; total 1200-1800 epochs, one extended to 2400 ("very very small improvement").
- Noise schedule: SwapNoiseMasker updated every epoch to decay noise; LR decay 0.998, noise decay 0.999 → 50% noise at epoch 0 lands ~27% after 600 epochs.
- MLP inference stage: dropout/hidden size tried, only fractional gains; details not stated.
- Tree models: 2 Optuna cycles each for XGB and LGBM (first to narrow ranges), top 5 kept per family; hyperparameter values not stated.
- **CV:** 5-fold (simple) used as the run-assessment guide: CV 0.894-0.898 after 600 epochs (author comment); paired run comparison at 600/1200 epochs — Run 1: 0.897925/0.89860, Run 2 (faster noise decay): 0.8982/0.8984 — faster decay better at 600, worse at 1200.
- Post-competition re-run combining all small improvements: single model, no blend, Private 0.89999 / Public 0.89528 — author: "would still have been #3 LB".
- Seeds/splits beyond stratified 10-fold: not stated. CV-vs-LB: author's capped-weights experiment found CV-best blend ≈ test-best blend on this dataset.
- **Ensembling:** Optuna + OOF predictions → per-model weights (values not stated). Second submission capped max weight per model "in case Optuna mix fitted CV better than test" → slightly worse on both public and private → cap dropped.
- **Post-processing:** not stated.
- **Gains:** (delta not stated) adding the X reconstruction + mask prediction to the DAE feature output (his "Variation 2") gave "the best single model score I have": 0.89964 private / 0.89482 public — the page publishes no before/after pair, so no gain number exists.
- (+0.000275 CV at 600 epochs, then reversal) per-epoch noise decay: Run 2 (faster decay) 0.8982 vs Run 1 0.897925 at 600 epochs; Run 1 back ahead at 1200 epochs (0.89860 vs 0.8984, i.e. Run 2 -0.0002). Net judgement: decay was used in "All DAEs used in submission".
- (no number stated) 5-fold → 10-fold inference: "10 fold was definitely better than 5 fold" — the page publishes no CV pair for the change.
- (-? final) blend over best single: author's final blend score is not published on the page, so the exact delta is not stated.
- **Failed:** "What Didn't Seem To Work" list (author's words, all from limited testing): larger batch size; smaller embed dim (larger dim also no gain); different loss and emphasis weights; changing emphasis over time; random noise; lower noise; much higher noise; varying noise with cardinality (both directions); lower and/or flat start learning rate.
- LGBM with one-hot columns + low-count dropping: "results seemed worse than just using original columns and label encoder".
- Capping blend max weight: worse on both public and private.
- **Comments:** 19 comments (1 tagged appreciation). Technical author replies: the 600-epoch CV-check habit forced by Colab cutting off notebooks; the 0.894-0.898 CV range; the two-run 600/1200-epoch comparison; admission "I didn't find many improvements on your starting point [from end of February comp]... changing them just made it worse" (reply to ryanzhang); best-settings training notebook + weights dataset shared in reply (links in Artifacts); to danzel (2nd): "really it is your idea originally in applying to this competition, and the difference between our scores is probably less than the impact of changing one random state! And I think you used fewer models"; to Pradeep Boopathy: he will share the run once his Kaggle GPU quota returns. Noise-decay idea credited to the January winner's second lower-noise training round (link in Artifacts); commenter Khurram Siddiqui credits danzel: "springmanndaniel idea showed up that autoencoder especially DAE worked well if one can handle noise level nicely".
- **Compute:** Colab for DAE training (600-epoch chunks because "Colab starts to cut off my notebook automatically"), Kaggle GPU for the re-run — "My GPU quota is running a little low"; wall-clock/RAM not stated.
- **Artifacts:** (all URLs cited by the author or in author replies, verbatim; not fetched)
  - https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter (his starter shared during the comp)
  - https://github.com/ryancheunggit/Denoise-Transformer-AutoEncoder (original DAE code)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216037 (January winner: second lower-noise training round → his decay idea)
  - https://www.kaggle.com/siavrez/kerasembeddings (all-categorical binning idea)
  - https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter-cv-inference (inference notebook; v6 = post-comp 0.89999 run)
  - https://www.kaggle.com/davidedwards1/tabmar21-tabular-blend-final-sub/output (blend notebook; individual model submissions + OOF in version 2 outputs, inputs private)
  - https://www.kaggle.com/davidedwards1/tabmar21-best-dae-settings-030421 (best DAE training notebook, stages v2/v3/v4 = epochs 0-600/600-1200/1200-1800)
  - https://www.kaggle.com/davidedwards1/tabmar21-final-dae-030421 (saved stage weights)
  - https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229868 (2nd's improvement idea he points at)
- **Lesson:** Import the previous month's winning architecture and change exactly one axis (the noise schedule) — the run-selection CV at 600 epochs was good enough to steer 27 experiments to a win.

### TPSMAR21-02 · 2nd · danzel (springmanndaniel) · LB 0.89599 public/0.90053 private · CV 0.9008564

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229868
- **Status:** FETCHED (1st `/c/` attempt returned full 11607 bytes, body + all 20 comments)
- **TL;DR:** DAE rebuilt around embeddings: noise added to label-encoded categoricals *before* the embedding layer, clean embeddings reconstructed on the fly.
- Stacked lgbm + 3 DAE-MLP runs via XGBLinear for the winning 0.90053 private / 0.89599 public.
- **Architecture:** STACK2 · stages=2 · l1=1 LGBM + 3 DAE→MLP models (each member is itself an embed→MLP cascade) · l2=XGBLinear (fitted on OOF) · novel=Transformer-DAE · mod=embed ·
  topo=[1× LGBM] + [3× (embedding-layer transformer-DAE, fixed 0.25 noise, 1000 epochs → 2-layer MLP 1000r-1000r-s)] → XGBLinear stack → 0.9008564 CV / 0.89599 public / 0.90053 private
  - Level 2 is a fitted estimator (XGBLinear) on the OOF outputs → `STACK2` per ruling 1, unlike 1st's weight search.
  - The DAE's embedding layers are trained during reconstruction; noise is applied pre-embedding so a categorical never feeds as "all types at once" (his objection to one-hot + noise).
  - His own page carries no comment from 1st; the "difference between our scores is probably less than the impact of changing one random state" line is Dave E's reply on TPSMAR21-01's page, not on this one.
- **Setup:** rows/cols/folds not stated on the page; scores published per member (Models).
- **Features:** no hand-built features stated; representation learning replaces FE. Numerical input as-is; categoricals label-encoded, noise swapped on them, then embedded (dim 5 for every categorical, including binaries).
- **Models:** DAE-Transformer (combined ryanzhang's + his own TPS-Jan winning pieces): fixed noise 0.25 randomly distributed, 1000 epochs, no noise-schedule tuning; stage-2 MLP = two layers, "(1000r-1000r-s)".
- LGBM single. Member scores: 1× LGBM CV 0.89743 / pub 0.89304 / priv 0.89769; 3× DAE-MLP CV 0.90042 / pub 0.89560 / priv 0.90012 (one score listed for all runs — "haven't uploaded every single dae-mlp run").
- **CV:** splitter/#folds not stated; final stack CV 0.9008564 vs private 0.90053 — CV ~0.0003 optimistic; author's trust verdict not stated.
- **Ensembling:** XGBLinear stacker over LGBM + 3 DAE-MLP OOF/predictions; stacker hyperparameters not stated.
- **Post-processing:** not stated.
- **Gains:** (+0.00043 CV over the best DAE-MLP member) XGBLinear stack vs single-family CV 0.90042.
- (+0.003 CV) DAE-MLP members over the LGBM member (0.90042 vs 0.89743).
- (unquantified) embedding reconstruction over one-hot+noise — the whole point of the architecture; "because this worked so well I also trained a lightgbm".
- **Failed:** none listed in the body beyond stated tuning abstentions: noise fixed at 0.25 "didn't try to increase/decrease"; embed dim 5 uniform, per-categorical tuning left as future work; epoch count (1000) by trial and error (comment reply).
- Commenter JiangTT (21st): his own nn.Embedding attempt failed; ryanzhang's reply names the reason: reconstructing the embeddings themselves causes representation collapse.
- **Comments:** 20 comments (1 tagged appreciation). Technical author replies: embeddings are learned during the DAE reconstruction of the clean data, with logic to "grab the right embeddings at the right time" (reply to delai50); confirmed the linear-excite/subspace layer from ryanzhang's architecture; 1000 epochs = "Trial and error"; reveals he won TPS-Jan so no prize eligibility.
- **Compute:** not stated.
- **Artifacts:** (every URL the page itself carries; the body cites no notebook or code URL — the only other hrefs are profile handles, e.g. https://www.kaggle.com/springmanndaniel, https://www.kaggle.com/ryanzhang)
  - https://s4.gifyu.com/images/new_1c86ab636a2a3f5f6.png and https://gifyu.com/image/Yqti (diagram 1, basic one-hot idea)
  - https://s4.gifyu.com/images/new_4.png and https://gifyu.com/image/Yqtj (diagram 2, noise added pre one-hot)
  - https://s4.gifyu.com/images/new_3.png and https://gifyu.com/image/Yqtu (diagram 3, noising post one-hot)
  - https://s4.gifyu.com/images/new_2.png and https://gifyu.com/image/Yqtm (the mask under the embedding scheme)
  - Rule 4c note: the four figures are the author's own uploads and were NOT transcribed here — every architecture statement in this entry comes from the body prose and his comment replies; the pictures' contents are unconfirmed.
- **Lesson:** If one-hot noise pollutes your autoencoder's input distribution, move the noise in front of a learned embedding and reconstruct the clean embedding target — the representation, not the blender, is worth the 0.003 AUC.

### TPSMAR21-03 · 3rd · not recoverable (author handle lost with the post) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/230101
- **Status:** UNFETCHED — topic deleted, INDEPENDENTLY RE-CONFIRMED BY THE AUDITOR on a fresh fetch. Forms re-run: `/c/` plain → 3198 bytes, and that render is not a nav shell but the tombstone itself: heading "Deleted Topic" + "This topic has been deleted." + "Please sign in to reply to this topic" + the 2 surviving comments; `/c/` + `X-No-Cache` → 164-byte empty shell; `/competitions/` + `X-No-Cache` + `X-Engine: browser` + `X-Timeout: 60` → 3275 bytes, again "### Deleted Topic / This topic has been deleted. / ## 2 Comments"; browser `navigate_page` + `take_snapshot` → accessibility tree shows heading "Deleted Topic" (level 3), StaticText "This topic has been deleted.", and exactly two comment nodes (Anuar Assamidanov, MICADEE "9th in this Competition"). No author handle, title, body text or score is recoverable from any route.
- **TL;DR:** The 3rd-place writeup was deleted from Kaggle; nothing of it survives except two comments under the tombstone.
- Commenters' breadcrumbs: the post evidently mentioned "LGBMRegressor w/ DAE" and something about DeepTables — neither claim verifiable against the body.
- **Architecture:** not stated (proof: the re-fetched page is a "Deleted Topic" tombstone — no author body, handle, estimator, blend step, or score anywhere on it; the only method words on the page, "LGBMRegressor w/ DAE" and "DeepTables", are commenter questions, not author claims, so no topology separates SINGLE from FLAT from STACK2; forms tried: /c/ plain, X-No-Cache, X-Engine browser + X-Timeout, /competitions/ + X-No-Cache + browser, browser snapshot, per ruling 12) · stages=not stated · l1=not stated · l2=not stated · novel=not stated · mod=not stated · topo=not stated
  - No estimator, blend, or score can be attributed to this entry; the surviving comments are reader questions, not author content.
- **Setup:** not stated (page deleted).
- **Features:** not stated (page deleted).
- **Models:** not stated (page deleted).
- **CV:** not stated (page deleted).
- **Ensembling:** not stated (page deleted).
- **Post-processing:** not stated (page deleted).
- **Gains:** not stated (page deleted).
- **Failed:** not stated (page deleted).
- **Comments:** 2 comments survive under the deleted topic. Anuar Assamidanov asks the author to explain "LGBMRegressor w/ DAE"; MICADEE (9th in this competition) replies he "thought of using DeepTables but forgot to use it at the end of the day" — both are commenters' words; the 3rd-place author never appears.
- **Compute:** not stated (page deleted).
- **Artifacts:** none recoverable (page deleted).
- **Lesson:** Late-posted TPS writeups can vanish — cite them, but never build a consensus claim on a tombstone.

### TPSMAR21-09 · 9th · MICADEE (adegladius) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229985
- **Status:** FETCHED (1st `/c/` attempt returned full 6137 bytes, body + all 4 comments)
- **TL;DR:** Fine-tuned Dave E's DAE implementation, then "combined" its score with his own XGB, LGB, CatBoost and an "octopus" ML model.
- Zero numbers published: no CV, no LB, no fold or weight detail.
- **Architecture:** FLAT · stages=1 · l1=5 (DAE, XGB, LGB, CatBoost, Octopus ML model) · l2=none (no meta-learner named) · novel=DAE · mod=none ·
  topo=[DAE (fine-tuned davidedwards1 impl) + XGB + LGB + CatBoost + Octopus] → combine model scores (mechanic not stated)
  - `novel=DAE` because an autoencoder is not a stock GBM/LR member; the page says only "DAE model implemented by @davidedwards1", so it is the borrowed transformer-DAE family, and this author contributes no new architecture of his own.
  - "Combine different scores generated from my different models" is the only ensemble description; weights/mechanic (average vs rank vs stack) cannot be derived, though no fitted meta-model is claimed, hence FLAT with the mechanic flagged.
- **Setup:** not stated.
- **Features:** not stated.
- **Models:** DAE from davidedwards1's starter (linked in the Mar code ecosystem), "fine tuned... as best I could"; XGB, LGB, CatBoost, Octopus ML model — all hyperparameters not stated.
- **CV:** not stated; no numeric CV anywhere on the page.
- **Ensembling:** score combination of the five members; mechanic not stated.
- **Post-processing:** not stated.
- **Gains:** not stated.
- **Failed:** not stated on this page.
- **Comments:** 4 comments. Technical content: Elvin Aghammadzada (16th) on why industry rarely uses such stacks (interpretability, "no one really cares too much about 0.001 precision", efficiency/lower resource cost in prod); the author then quotes ryanzhang's answer on DAE for real data — the quoted text is reproduced by MICADEE in his own reply, ryanzhang's original comment is not shown on this page: "usually the larger the dataset the more likely it can be close or better than trees... the non-transformer DAEs can always degrade to a non-linear PCA".
- **Compute:** not stated.
- **Artifacts:** none cited with URLs on this page (handle links only: ryanzhang, davidedwards1, elvinagammed).
- **Lesson:** A borrowed, untuned DAE plus your usual GBDT zoo still reaches top 10 — but with zero published numbers this page is a pointer, not a recipe.

### TPSMAR21-11 · 11th · Lázaro (lazaro97) · LB not stated/not stated (top 1%) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229834
- **Status:** FETCHED on 3rd ladder attempt (`/c/` plain returned 2506-byte nav shell; `X-No-Cache` returned 407-byte shell; recovered 8140 bytes with `X-No-Cache: true` + `X-Engine: browser` + `X-Timeout: 60`)
- **TL;DR:** A generic stacked model with a Ridge at stage 2, built from the best few models per family; the page is lessons/philosophy, code never published.
- **Architecture:** STACK2 · stages=2 · l1=not stated ("best models for each type", families LGBM/XGB/CatBoost named) · l2=Ridge regression on the stacked outputs · novel=none · mod=none ·
  topo=[best-of each GBM family] → stage-2 Ridge → submission
  - Author names his own level-2 explicitly: "at least in my ridge on the 2nd stage of modeling" — a fitted meta-model → `STACK2`.
  - Level-1 composition, fold scheme, counts: not stated; promised code cleanup never appeared on this page.
- **Setup:** not stated (top 1%, 11th).
- **Features:** not stated.
- **Models:** LightGBM, XGBoost, CatBoost named as ensembled "diverse models"; logistic regression/SVM/HistGB/NN mentioned as things also worth trying (general advice, not confirmed members); hyperparameters not stated.
- **CV:** not stated; validation philosophy only ("see the valid score and use your model in new observations").
- **Ensembling:** stacking of diverse models; "only use the best models for each type of model"; weights: Ridge fits them.
- **Post-processing:** not stated.
- **Gains:** not stated.
- **Failed:** caution (not a named failure): adding public participants' models can be data leakage — "review its source and evaluate if the new information is good"; adding a model shrinks the Ridge's other weights ("if you add a model, other model decreases its participation... at least in my ridge").
- **Comments:** 7 comments (1 tagged appreciation); author replies only emoji-level (reply to Vadim Irtlach about dirty code). Nothing technical.
- **Compute:** not stated.
- **Artifacts:** (learning resources cited by the author, verbatim; not fetched)
  - https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter
  - https://www.kaggle.com/hiro5299834/tps-mar-2021-rank-averaging-and-stacking
  - https://www.kaggle.com/rmiperrier/tps-mar-lgbm-optuna
  - https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter-cv-inference
  - https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/225929
  - https://www.kaggle.com/tunguz/tps-mar-2021-eda
  - https://www.kaggle.com/siavrez/kerasembeddings
  - https://www.kaggle.com/theawm/0-8917-stratified-kfold-xgboost-eda
  - https://mlwave.com/kaggle-ensembling-guide
- **Lesson:** Public starter kernels + a plain Ridge stacker is a complete top-1% recipe on early TPS — but audit every borrowed model's provenance before it enters your OOF matrix.

## TPSMAR21 - consensus recipe
- **Architecture distribution:** 1st `FLAT (Optuna weights; DAE→MLP cascade members; novel=Transformer-DAE)` · 2nd `STACK2 (XGBLinear over LGBM + 3 DAE-MLP; novel=Transformer-DAE)` · 3rd `not stated (topic deleted - tombstone re-verified by the auditor; all slots not stated)` · 9th `FLAT (5-model score combine, mechanic not stated; novel=DAE borrowed)` · 11th `STACK2 (Ridge stage-2)` — winner topology: FLAT blend of DAE-MLP cascades + Optuna-tuned trees.
- **New architectures at the board:** Transformer-DAE at 1st (post-comp single 0.89999 private / 0.89528 public; in-comp best single 0.89964 private / 0.89482 public; final blend score never published) and 2nd (stack 0.90053 private / 0.89599 public, CV 0.9008564) — the two top ranks; 9th ran the same borrowed DAE family (novel=DAE, his own page publishes no score). danzel's embedding-layer DAE variant (embed dim 5, pre-embedding noise, clean-embedding reconstruction) is the board's only genuinely new architecture, and its stack holds the best published private score in the file (0.90053; 1st's private 0.89999 is a post-competition single model, not his submission). No GBM/NN novelty beyond the DAE family; 11th was stock GBDT + Ridge.
- **Agreed on (3 of 4 retrievable; IDs):** DAE as the key model family (TPSMAR21-01, TPSMAR21-02, TPSMAR21-09 all ran one; TPSMAR21-03 unknown/deleted, TPSMAR21-11 states no DAE). Stacking/ensembling diverse families under one final layer (TPSMAR21-01 Optuna blend, TPSMAR21-02 XGBLinear, TPSMAR21-09 score combine, TPSMAR21-11 Ridge = 4 of 4). Optuna/automated tuning: strictly 1 of 4 (only TPSMAR21-01 states Optuna — for his XGB/LGBM sweeps and for blend weights); TPSMAR21-02 and TPSMAR21-11 never name it, and TPSMAR21-11 only links an Optuna starter kernel as a learning resource, which is not a stated practice.
- **Divergences:** 1st vs 2nd is one fitted meta-model apart: Optuna weight search (FLAT, final score not published) vs XGBLinear stack (0.90053 private / 0.89599 public). The "difference between our scores is probably less than the impact of changing one random state" line is Dave E's reply on TPSMAR21-01's page, not a claim on 2nd's page. Top ranks = representation learning (DAE); 9th/11th = generic stacks around borrowed DAE scores with no published numbers, and the leaders' member-level AUC gap over a plain GBDT is 0.00299 CV (0.90042 DAE-MLP vs 0.89743 LGBM, TPSMAR21-02).
- **Highest-leverage single trick:** per-epoch noise decay on the SwapNoiseMasker (1st, TPSMAR21-01: LR decay 0.998, noise decay 0.999, 50%→~27% over 600 epochs) — the one change that turned Feb's architecture into Mar's win; runner-up trick: noise before the embedding layer with clean-embedding reconstruction (2nd).
- **Nothing worked:** 1st's failed-DAE-tuning list (batch size up, embed dim down, loss/emphasis variants, random noise, noise up or down, noise-by-cardinality, lower/flat LR); LGBM with one-hot + low-count dropping (worse than label encoding, TPSMAR21-01); capping Optuna blend weights (worse both boards, TPSMAR21-01); 5-fold in place of 10-fold (TPSMAR21-01); reconstructing embeddings themselves → representation collapse (commenter-exchange anti-knowledge, TPSMAR21-02).
