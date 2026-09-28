## TPSJUL22 — Tabular Playground Series - Jul 2022
- Task: Tabular (Clustering) | Metric: Adjusted Rand Index | Problem: Practice clustering on tabular data
- Kaggle display title: "Tabular Playground Series - Jul 2022" / "Practice your ML skills on this approachable dataset!"
- Competition: https://www.kaggle.com/c/tabular-playground-series-jul-2022
- Writeups covered: 1 of 1
- Metric as reported on the page: **not named anywhere on this page** — the strings "Adjusted Rand", "Rand" and
  "ARI" do not occur in the body or the 27 comments, and no numeric score of any kind appears (direct check: the
  pattern `0.<digit>` has 0 matches). The author refers to his results only as "the LB score" / "my best LB score".
  ARI (higher better) is taken from the competition index, not from the writeup. No accuracy or NMI figure exists
  on the page, so none is quoted here.
- This is the library's only unsupervised competition in this file: there are no labels for entrants, so the
  record describes representation, algorithm, cluster-count selection and the label-mapping step instead of
  folds, targets and blends.
- Score ladder: 1st `not stated/not stated` (no ARI number published; author's only quantitative statement is
  directional — 1 component per cluster "greatly reduced the LB score")

### TPSJUL22-01 · 1st · ymatioun (Youri Matiounine) · LB not stated/not stated · CV not stated (unsupervised; no CV described)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jul-2022/discussion/341023
- **Status:** FETCHED (1 attempt: `/c/.../discussion/341023` + `X-Return-Format: markdown`, 13,058 bytes, full body
  + all 27 comments flat. The alternate `/competitions/` form and the `?sort=polls` variant also return full pages
  (~12.8-13.1 KB) — the `/competitions/` render adds the citation line, whose URL is recorded under Artifacts — so the
  first fetch is the complete record: 22 visible comments + 3 marked deleted
  + 2 in the Appreciation section = the 27 the thread claims.)
- **TL;DR:** One custom EM Gaussian-mixture run, constrained by the generator's structure reverse-engineered from a
  `GaussianMixture(n_components=42)` on the float columns only.
- The winning move is data archaeology, not algorithm: 42 real clusters arranged as 7 groups x 6, submitted as 7 labels.
- Hard-coding the floating-point means to 0 / -1 / +1 is what he credits for his best LB score; no ARI number is published.
- **Architecture:** SINGLE · stages=1 · l1=1 (custom EM Gaussian mixture, 42 components) · l2=none (nothing fitted above it) ·
  novel=Custom-EM-GMM · mod=none ·
  topo=14 cols [7 int power-transformed + 7 float raw] → custom EM GMM[42 comps, float means hard-coded 0/-1/+1, block covariance]
  → 42 clusters → collapse each group of 6 → 7 submitted labels
  - The submitted prediction is one clustering run: the winning path contains exactly one fitted estimator and no blend,
    no second model, no seed/fold averaging — so `SINGLE`, not `SEED` and not `FLAT`. (The
    `GaussianMixture(n_components=42)` he also fitted is EDA that informed the constraints, not a member of the submission.)
  - What is fitted: an EM procedure the author wrote himself for a Gaussian mixture "taking into account details of the
    data structure". What is fixed instead of fitted, and stated as such: the means of all floating-point variables
    (0, -1, +1) — hard-coded "to ensure convergence to true cluster centers".
  - `novel=` is his hand-written EM, not a stock call: sklearn's `GaussianMixture`/`BayesianGaussianMixture` cannot fix
    per-component means or impose the 7x7 + 5x5 + 2x2 block covariance he describes. The stock route the community used
    (`BayesianGMMClassifier`, named without a URL) is only discussed as the thing he read the source of.
  - The 42 -> 7 label collapse is a relabel step after the clustering, so it is `Post-processing`, not a stage (ruling 2).
  - No embedding stage: the representation is the 14 selected columns themselves (no PCA, no UMAP — neither string
    appears anywhere on the page).
- **Setup:** Unsupervised tabular clustering; entrants get one dataset and submit cluster labels; no labels are available,
  so there is no train/test target split to describe.
- Relevant variables named: 14 total — `f_07`-`f_13` (7 integers) and `f_22`-`f_28` (7 floating point); "the others are not
  used at all", a fact the author attributes to being "discovered early on in the competition", not to himself alone.
- Row counts, total column count, submission-slot count: not stated.
- Structural quirk exploited: the generator's own hierarchy — 42 clusters in 7 groups of 6, where integer variables only
  separate the 7 groups and the floating-point variables separate the 6 sub-clusters inside each group.
- Author's framing of why this data was hard to validate: there was no ground truth; commenter martinerrazquin (192nd)
  states it explicitly — "there was no score other than public LB which to optimize against".
- **Features:** Column selection: keep the 14 variables above, drop every other column from the clustering entirely.
- Integers `f_07`-`f_13`: power-transformed "into a more 'normal' shape" and then treated as multivariate normal
  variables. The specific transform (Box-Cox / Yeo-Johnson / log1p) is not named — source silent.
- Floats `f_22`-`f_28`: deliberately NOT transformed — "Floating point variables already appear to have a normal
  distribution, so there is no need to transform them."
- No scaling/standardization step is named (direct check: "scal*" does not occur on the page).
- Structure mined from EDA and then hard-coded into the model, verbatim from the body:
  - Integer variables vary only across the 7 cluster groups, so "they truly form 7 clusters, not 42"; their means and
    covariances look completely independent across the 7 groups, with some covariances zero.
  - Per cluster group, 2 of the 7 floats have mean 0, sd 1, and are uncorrelated with every other variable.
  - The remaining 5 floats have means of -1 or +1 and a full 5x5 covariance matrix; they are uncorrelated with the integers.
  - Consequence he uses as the model's shape: the 14x14 covariance is a 7x7 block (integers) + a 5x5 block (correlated
    floats) + a 2x2 identity block (uncorrelated floats).
- Encodings, GP/generated features, feature-source kernels: none named — source silent.
- **Models:** Submitted: 1 custom EM for a Gaussian mixture model, 42 components (7 groups x 6), with the block covariance
  structure above and the floating-point means hard-coded to 0 / -1 / +1.
- EDA model that revealed the structure: `GaussianMixture(n_components=42)` run on the floating-point variables only,
  then reading off the component means — "This data structure can be easily uncovered by" that.
- The community baseline he reverse-engineered: `BayesianGMMClassifier` (named, no URL) — its source uses
  `n_components=7` for each of the 7 clusters, i.e. 49 components in total.
- Parameters of his EM (initialisation, max iterations, tolerance, `n_init`, random seed), library/language and versions:
  not stated; the code is published as a Kaggle notebook (link in Artifacts, not fetched).
- **CV:** No CV exists and none is described: the task is unsupervised, so there is no label to hold out. Direct check —
  the body contains no validation scheme, no folds, no internal metric.
- The only feedback signal the author reports using is the public leaderboard itself ("greatly reduced the LB score",
  "resulted in my best LB score").
- CV-vs-LB gap and author's trust verdict: not stated / not applicable.
- **Ensembling:** None. No blend, stack, voting, consensus partition or run-averaging is described anywhere on the page;
  source silent on whether the EM was run more than once with different initialisations.
- Shared-OOF usage: not applicable (no predictions market exists for an unsupervised task; none mentioned).
- **Post-processing:** Label mapping — the 42 discovered clusters are collapsed 6-at-a-time into their cluster group,
  giving "7 distinct cluster labels" in the submission. Per ruling 2 this is post-processing, not a second stage.
- Parameter fixing at the model level (not post-processing): floating-point means hard-coded to 0 / -1 / +1 before fitting.
- No threshold, calibration, clipping, rank-gauss or rounding step is named; nothing on the page describes aligning or
  permuting labels across runs.
- **Gains:** No ARI figure is published for any step, so every entry below is directional, in the author's own words.
- (+LB, unquantified, largest claimed) hard-coding the floating-point means to 0/-1/+1 in the custom EM —
  "And this is what resulted in my best LB score."
- (+LB, unquantified) modelling 42 sub-clusters instead of 7: collapsing the BGMMClassifier's 7-per-cluster setting down to
  1 component per cluster "greatly reduced the LB score", so the sub-cluster structure is worth the entire gap.
- (+LB, unquantified) replacing `n_components=7` per group with 6: "further experimentation showed that each cluster
  group actually has 6 clusters, not 7."
- (enabling, unquantified) `GaussianMixture(n_components=42)` on floats-only + inspecting component means: without it the
  block structure could not have been hard-coded.
- (fallback, unquantified) power-transforming the integers instead of modelling them: "the same thing as everybody else".
- **Failed:** Building a generative model for the integer columns: "I tried to construct a model for them, but could not
  come up with anything that worked" — his stated bar was a distribution producing negative correlations but no negative
  values AND having a tractable probability mass function. He names this as "probably where the model could be improved
  the most".
- 1 component per cluster in the Bayesian-GMM route: greatly reduced the LB score.
- His own prior belief that the `BayesianGMMClassifier` author's 7-sub-component choice was a mistake: wrong — the truth is
  6, and the total cluster count is >7; that misreading is what the LB drop corrected.
- Commenter-reported dead ends (not the author's, on this page): RomainBdt (16th) found 1 component "bad for LB score" and
  6 vs 7 "similar results", with only ~6 sub-clusters carrying weight; martinerrazquin (192nd) "failed to detect further
  structure, and stuck to 7 sub-components".
- Unanswered on the page: two commenters ask how the 14 relevant variables were identified and how to do feature selection
  here; no reply from the author appears anywhere in the 27-comment tree (direct check).
- **Comments:** Technical content that exists only in comments (all from commenters; the author does not reply):
  - RomainBdt (16th): independently reproduced the 1-vs-6-vs-7 component experiment; sub-cluster weights show ~6 active
    clusters; cannot explain why `BayesianGMMClassifier` beats a plain Bayesian Gaussian Mixture; takeaway — "spend more
    time on understanding the data than fine tuning on model".
  - martinerrazquin (192nd): mechanism for why that classifier works — each major cluster is a union of Gaussian
    sub-clusters, and sklearn's `score_samples` returns a log likelihood, so the per-class probability is
    `likelihoods[i] / sum(likelihoods)`.
  - martinerrazquin (192nd): "there was no score other than public LB which to optimize against", and he shares the
    request that the organizers publish the data generator.
  - Joseph Zhou (7th): "Seems that 42 plays an important role not only in seeds." (the only mention of 42 outside the
    cluster count; no seed value is published).
  - Questions with no answer on the page: how the 14 variables were found (user7777777; Safrizal Ardana Ardiyansa, who
    asks specifically why `f_07`-`f_13` and `f_22`-`f_28`), and how the 6-sub-cluster count was realised (Yan Mazas, 135th).
  - Thread bookkeeping: 27 comments = 22 visible + 3 deleted + 2 appreciation; 6 "thank you" reactions on the post.
- **Compute:** Not stated — the page names no wall-clock, no GPU/CPU choice, no RAM figure and no Kaggle limit.
- **Artifacts:** (cited by the page, not fetched)
  - https://www.kaggle.com/code/ymatioun/tps22jul-custom-em (the author's own code notebook, cited in the body:
    "see the actual code at ...")
  - https://www.kaggle.com/ymatioun (author profile)
  - https://www.kaggle.com/c/tabular-playground-series-jul-2022/discussion/341023 (this page)
  - https://www.kaggle.com/competitions/tabular-playground-series-jul-2022 (page navigation: overview, data, code,
    models, discussion, leaderboard, rules all link off the same `/competitions/` base)
  - https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/writeups/youri-matiounine-1-solution
    (citation line, on the `/competitions/` render)
  - https://www.kaggle.com/competitions/33107/images/header (competition header image)
  - Commenter profiles linked from the thread: https://www.kaggle.com/takanashihumbert (Joseph Zhou, 7th),
    https://www.kaggle.com/romainbdt (16th), https://www.kaggle.com/yuriturygin (29th), https://www.kaggle.com/yanmazas (135th),
    https://www.kaggle.com/sinclairg (162nd), https://www.kaggle.com/martinerrazquin (192nd),
    https://www.kaggle.com/arturra (206th), https://www.kaggle.com/javigallego (213th),
    https://www.kaggle.com/samuelcortinhas (213th), https://www.kaggle.com/mrcljns (216th),
    https://www.kaggle.com/willcramptonn, https://www.kaggle.com/rickyyoung4364, https://www.kaggle.com/user7777777,
    https://www.kaggle.com/safrizalardanaa, https://www.kaggle.com/oscarm524, https://www.kaggle.com/zzzzzzzbob,
    https://www.kaggle.com/robertgouldie,
    https://www.kaggle.com/circlehalf0723, https://www.kaggle.com/muhammedtausif, https://www.kaggle.com/flaviafelicioni (252nd)
  - named without a URL: `BayesianGMMClassifier` (the Kaggle notebook/tool whose source he read),
    `sklearn` `GaussianMixture`, `BayesianGaussianMixture` / Bayesian GMM (mentioned only as "simple Bayesian Gaussian
    Mixture model"), `score_samples`
- **Lesson:** In a synthetic clustering playground, fit one high-k mixture and read the generator out of its component
  means — then hard-code what you learned (block covariance, fixed means) rather than tuning a black-box GMM.

## TPSJUL22 — consensus recipe
- **Single-source file:** this competition has exactly 1 writeup in the index, so no cross-writeup agreement can be
  asserted. Every claim below is supported by TPSJUL22-01's page only, unless it is labelled as a commenter statement
  from that same page. Rule 10 count: all shared-practice lines are "1 of 1".
- **Architecture distribution:** 1st `SINGLE` (one custom EM Gaussian-mixture run, 42 components, post-hoc 42->7 label
  collapse).
  - **winner topology: `SINGLE`** — stages=1, one fitted estimator, no meta layer, no blender, and no run averaging
    described; the only thing fitted is the mixture itself.
- **New architectures at the board:** `Custom-EM-GMM` (1st) — a hand-written Expectation-Maximization routine for a
  structure-constrained Gaussian mixture; nothing else appears, and the stock sklearn equivalents (`GaussianMixture`,
  `BayesianGaussianMixture`/`BayesianGMMClassifier`) are named as the baselines he read and moved past.
  - This is the only entry in the library where the "new architecture" is a custom fitting procedure rather than a new
    network family, and it is also the only unsupervised winner.
- **Agreed on (1 of 1 — TPSJUL22-01):** the representation is a *column-subset + shape correction*, not an embedding:
  14 of the variables kept (`f_07`-`f_13` power-transformed integers, `f_22`-`f_28` untouched floats), no PCA, no UMAP,
  no scaler named.
- **Agreed on (1 of 1 — TPSJUL22-01):** cluster-count selection by *reading component means* — `GaussianMixture(n_components=42)`
  on the float columns only, plus the negative control of cutting the BGMM's 7-sub-component setting to 1 (LB dropped),
  which proved the true count is >7 and specifically 6 per group.
- **Agreed on (1 of 1 — TPSJUL22-01):** the submission is a *label-mapping* of a finer clustering (42 -> 7 groups) done
  as post-processing per ruling 2.
- **Corroborated only inside the comments (not by any second writeup):** the 7x6 hierarchy and the
  `score_samples` likelihood mechanism are restated by RomainBdt (16th) and martinerrazquin (192nd) on this same page.
  Their ranks are the only evidence available about alternative routes; they are commenters, not entries in this index.
- **Divergences:** not measurable inside this file (1 entry). The page itself does show the split at the board: the
  author won with a structure-constrained custom EM on 42 components, while the commenters describing their own attempts
  (16th, 192nd) ran the community `BayesianGMMClassifier` with 7 sub-components and could not see past 7. No score is
  published by anyone, so the resulting gap is unquantified.
- **Highest-leverage single trick:** hard-code what the data generator did — fix the floating-point means at 0/-1/+1
  inside the EM so it converges to the true centres (TPSJUL22-01; author's own "this is what resulted in my best LB
  score", magnitude not published).
- **Nothing worked:** modelling the sub-cluster count as 1 component per cluster (author: "greatly reduced the LB score";
  independently reproduced by RomainBdt, 16th).
- **Nothing worked:** building a tractable generative model of the integer columns — the author wanted negative
  correlations without negative values plus a tractable PMF and reports no success; he fell back to the power transform
  everyone else used, and calls it the biggest remaining upside.
