## TPSAPR21 - Tabular Playground Series - Apr 2021
- Task: Tabular (Binary Classification) | Metric: Categorization Accuracy (higher better) | Problem: Synthanic survival prediction
- Kaggle display title: "Tabular Playground Series - Apr 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-apr-2021
- Writeups covered: 1 of 1
- Metric note: Accuracy, higher better, per the index title line and the page's own score style (0.81325 on the private LB = share of correct predictions); every number in this file is an Accuracy, so a gain is always positive.
- SINGLE-SOURCE FILE: the index lists one writeup for this competition, so the consensus block below reports only what that one page states and claims no cross-writeup agreement.
- Score ladder (as reported by the author): 1st `final winning score not stated; the shared public notebook version "should get 0.81325 on private LB and get 3rd place"`

### TPSAPR21-01 · 1st · JiangTT (jiangtt) · LB not stated/not stated (shared version: private 0.81325) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/235739
- **Status:** FETCHED (1st `/c/` attempt returned full 9956 bytes, body + all 11 comments)
- **TL;DR:** Swap noise (the DAE trick) ported from autoencoders into GBDT training as data augmentation: 30 LGBM + 30 CatBoost + 30 DecisionTree models, each trained under a different noise level, mixed together.
- The actual winning submission was run locally and blended with two public results (hiro5299834's DAE, alexryzhkov's AutoWoe); the shared notebook reproduces only a 3rd-place-equivalent 0.81325.
- Author's own honesty flag: "I don't remember if I turned on swap noise in my best submission."
- **Architecture:** FLAT · stages=1 (blend level; internal pseudo-label rounds not stated) · l1=90 own GBDTs (30 LGBM + 30 CatBoost + 30 DT, different swap noise each) + 2 external prediction sets · l2=none · novel=none · mod=public-oof,pseudo ·
  topo=[30×LGBM + 30×CatBoost + 30×DT, per-model swap noise (ryanzhang's function modified into hiro5299834's pseudo-labeling voting-ensemble pipeline)] + [hiro5299834 DAE preds] + [alexryzhkov AutoWoe preds] → blend (weights not stated)
  - The submitted winner is a blend of his own noisy-GBD ensemble with two other people's prediction outputs — hence `mod=public-oof`; per ruling 6 the AutoWoe member does not make this `AUTOML` because the submitted artifact is his hand blend, not the vendor stack.
  - The pipeline he modified is a pseudo-labeling voting ensemble (its title on the linked notebook), so `mod=pseudo`; round count and label-selection rule: not stated on this page.
  - "Trained many (30 for lgbm, catboost and dt each) models with different noise and mix them" — the noise-variant mixing is his substitute for per-epoch noise, which he says he didn't know how to do.
- **Setup:** synthetic Titanic-style survival task; rows/cols/slots not stated; the winning run was local ("I ran my best submission in my local machine"), the shared Kaggle notebook needs DEBUG→False to replicate.
- **Features:** not stated on this page (pipeline inherited from hiro5299834's notebook; only the swap-noise augmentation added to it). NaN strategy not described by the author (commenter Daedalus reports his own failed KMeans/cluster-mean NaN fills, not the winner's method).
- **Models:** LightGBM ×30, CatBoost ×30, DecisionTree ×30, each trained with a different swap-noise setting; all hyperparameters: not stated (live in the linked notebooks). External blend members: hiro5299834's DAE model, alexryzhkov's LightAutoML AutoWoe pipeline.
- **CV:** not stated; no numeric CV published on this page. Author attributes his result partly to luck in "prevent[ing] overfitting" rather than a validated ranking.
- **Ensembling:** final = his best (noisy GBDT ensemble) submission blended with the two external prediction sets; blend weights and mechanic not stated. Shared notebook version = pseudo-labeling voting ensemble of the 90 noisy models alone → 0.81325 private (would be 3rd).
- **Post-processing:** not stated.
- **Gains:** (size not stated) blending hiro5299834's DAE predictions and alexryzhkov's AutoWoe predictions into his own ensemble is the stated step between the shared notebook's "0.81325 on private LB and... 3rd place" and his 1st place; the page never prints his winning Accuracy, so no delta can be computed (higher = better).
- (unquantified) swap noise as GBDT augmentation: his stated thesis "data augmentation will be useful in GBDT training"; no ablation numbers published, and he cannot confirm it was on in the best run ("I don't remember if i turned on swap noise in my best submission").
- **Failed:** "no one seems to be interested in my discussion thread" — his pre-competition idea post drew no engagement, so he validated it himself (author's words).
- Per-epoch varying noise during GBDT training: not achieved ("I don't know how to apply different noise in every epoch"), replaced by the 30×3 noise-variant mix.
- From comments (not the author's own pipeline): Eugenee (4th) — extra hyperparameter tuning and other models "didn't work for me" on top of hiro's notebook; Daedalus (325th) — "smart" NaN filling and KMeans cluster-mean filling did not work.
- **Comments:** 11 comments (1 tagged appreciation). Technical commenter content: Björn (248th) answers the author's open problem with two per-epoch augmentation routes for trees — (1) 1-epoch train/save/resume loops via xgboost `xgb_model` option (with the LightGBM continue-training StackOverflow link), (2) one very big augmented training set, never augmenting validation — plus his caveat that he mostly failed to find good tabular augmentations; Eugenee (4th) reports reaching 4th by swapping hiro's DecisionTree for XGBoost. The author posts no comment of his own on this thread, and no representation-collapse exchange appears here (that one lives on the Mar-2021 2nd-place page; nothing from it is attributed to this entry).
- **Compute:** best submission local (machine specs not stated); shared notebook on Kaggle; GPU/CPU time not stated.
- **Artifacts:** (all URLs cited by the author or in comments, verbatim; not fetched)
  - https://www.kaggle.com/jiangtt/tps-apr-2021-pseudo-labeling-voting-ensemble (his shared submission notebook)
  - https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta (where he learned swap noise is key to DAE)
  - https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble (the pipeline whose training he modified with the swap-noise function, and whose DAE result he blended)
  - https://github.com/sberbank-ai-lab/LightAutoML (LightAutoML; AutoWoe author alexryzhkov's framework)
  - https://github.com/mljar/mljar-supervised (MLJAR)
  - https://stackoverflow.com/questions/45654998/lightgbm-continue-training-a-model (commenter Björn's link)
- **Lesson:** Cross-pollinate techniques between model classes — port the winner's noise mechanism from autoencoders into 90 diversified GBDT trainings, then let two public prediction sets cover your own uncertainty.

## TPSAPR21 - consensus recipe
SINGLE-SOURCE: one writeup in the index; every line below reports only what TPSAPR21-01 states. No agreement/divergence claims are possible.
- **Architecture distribution:** 1st `FLAT (90 noisy GBDTs + 2 public prediction sets blended; weights/mechanic not stated)` — winner topology: FLAT with mod=public-oof,pseudo; novel=none on this page.
- **New architectures at the board:** none named — the page's models are stock LGBM/CatBoost/DecisionTree (swap noise is a training technique, not an architecture), plus borrowed external DAE and AutoWoe predictions in the blend; no hyperparameters or scores published for the actual winning artifact.
- **Agreed on (1 of 1; TPSAPR21-01 is the only entry, and every item below is stated on its page):** blending his own result with two other people's public prediction outputs (TPSAPR21-01: hiro5299834's DAE result + alexryzhkov's AutoWoe result); many noise-variant models per family instead of tuning (30 LGBM + 30 CatBoost + 30 DecisionTree, each with different swap noise); and his own acknowledgement that validation did not drive the win — "The result really surprised me! Guess I'm lucky enough to prevent overfitting" (TPSAPR21-01).
- **Divergences:** not applicable with a single entry; note the internal split between the shared notebook (voting ensemble, 0.81325) and the unshared winning blend (local run + two public results).
- **Highest-leverage single trick:** adding two other people's public prediction files (DAE + AutoWoe) to his own ensemble — the stated step between a 3rd-place 0.81325 and 1st (source TPSAPR21-01).
- **Nothing worked:** per the author's own comments thread: his idea discussion attracted no engagement pre-comp; per-epoch noise for GBDTs unsolved by him; per commenters on this page: extra tuning/other models (Eugenee, 4th), smart NaN imputation and KMeans cluster-mean fills (Daedalus, 325th).
