## TPSSEP21 — Tabular Playground Series - Sep 2021
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Sep 2021 — Practice your ML skills on this approachable dataset!"
- Competition: https://www.kaggle.com/c/tabular-playground-series-sep-2021
- Writeups covered: 8 of 8
- Score ladder: 2nd `not stated/not stated` · 6th `0.81875 public (blend), 0.81851 public (single LGBM)/not stated` ·
  7th `not stated/not stated` · 9th `not stated/not stated (final private rank 9th)` ·
  10th `not stated/not stated (best private 0.81756, deliberately NOT selected)` ·
  14th `0.81868/0.81752` · 24th `not stated/not stated` · 45th `not stated/not stated`
- Only three of the eight pages publish any AUC number, so the ladder is mostly ranks; the densest numeric ladder is 14th's
  (nine public scores + five private scores) and the public->private drop of -0.00116 is the only one measurable in this set.

### TPSSEP21-02 · 2nd · vkonstantakos (Vasilis Konstantakos) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275740
- **Status:** FETCHED. `/c/.../discussion/275740` returned a 164-2506 byte navigation shell on the plain form, with
  `X-No-Cache`, and with `X-Engine: browser`; recovered the full post + all 29 comments (16,679 bytes) on the
  `/competitions/tabular-playground-series-sep-2021/discussion/275740` + `X-No-Cache: true` form (ladder step 4).
- **TL;DR:** 115 level-0 models (Optuna-tuned XGB/LGBM/CatBoost/HistGB; TabNet and pyGAM GAMs tuned by their own sweeps)
  stacked by ElasticNet, then a SECOND ElasticNet fed the first one's output as an extra column.
- He states he used "mostly basic methods and techniques" and did not expect 2nd place; no LB score is published anywhere on
  the page (body or comments).
- **Architecture:** STACKN · stages=3 · l1=115 base models (XGB, LGBM, CatBoost, HistGradientBoosting, TabNet, pyGAM GAMs) ·
  l2=ElasticNet regression (default params, best CV of L1/L2/ElasticNet) · l3=ElasticNet regression again (same columns + l2's
  output as one extra input) · novel=TabNet · mod=multi-view ·
  topo=per-row stats + mixed imputers/scalers + kmeans/outlier/t-SNE/UMAP cols → 115 L0 models → ElasticNet₁(115 L0 preds) →
  ElasticNet₂(same 115 preds + ElasticNet₁ output) → submission
  - Three model-training stages, so this is `STACKN` and not `STACK2`: the level-0 pool, then two chained ElasticNet meta
    layers, the second one receiving the first one's prediction as an additional feature alongside the unchanged level-0
    columns (author's own words in his reply to @raj401).
  - Winning structure is the stack, not a member; the metas are linear regressions fitted on the level-0 outputs (probabilities
    vs logits not stated on his page). He picked ElasticNet for having the best CV among L1/L2/ElasticNet; a commenter
    (@gauravbrills, 231st) asks why ElasticNet over LogisticRegression and the author never answers on this page.
  - `novel=TabNet` names only a level-0 member: he reports TabNet reached "acceptable performance" with or without
    pre-training; the submitted model chain is GBM outputs + two elastic nets. `pyGAM` GAMs sit at level 0 too and by his own
    comment were "not as accurate as the XGB, CB, and LGB models, but they added diversity".
- **Setup:** rows/cols, train/test sizes, fold count and submission slots: not stated anywhere on the page.
- Structural choices he names: per-instance (per-row) statistics, feature-dependent imputation, and "different parameters and
  features" across level-0 models — i.e. the 115 members differ in the view they consume, not only in hyperparameters.
- He deliberately left the meta-models un-optimized "to avoid overfitting due to so many base models".
- **Features:** exact families as listed by the author under "Preprocessing":
  - Per-instance statistics: number of missing values, standard deviation, min, max, average, "etc."
  - Null filling with mean, median, and mode, "depending on the variable's distribution" (all-or-some nulls).
  - Scaling with different scalers (standard, robust, "etc.").
  - Categorical features created from the initial features. Author's reply to @faroqaltam names the columns he binarized by
    distribution: `f29, f40, f42, f65, f70, f74, f75`, with the cut-off "determined manually". Caveat recorded as stated on
    this page: those column ids are outside this competition's `f0..f12`-style schema, and the page gives no reconciliation.
  - Added k-means features, outlier features, t-SNE features and UMAP features.
  - Level-0 feature subsets used for `pyGAM`: only the most important features, ranked from his earlier XGBoost feature
    importances; combinations of feature subsets × spline-term counts tried, most accurate kept.
  - Encodings, GP/generated features, dropped columns: not stated.
- **Models:** level-0 pool = 115 models ("The most extreme implementation was the number of the base models, reaching a
  number of 115 at the end").
  - XGBoost, LightGBM, CatBoost: trained on CPU and/or GPU, optimized with Optuna.
  - Histogram-based Gradient Boosting Classifier: also optimized with Optuna.
  - TabNet: "with or without pre-training", "acceptable performance".
  - Generalized Additive Models: `LogisticGAM` from `pyGAM` (author comment), first Python use of GAMs; all-feature fits died
    with memory errors, so he cut the feature set and swept spline terms.
  - Meta: L1 (Lasso), L2 (Ridge) and ElasticNet regression all tried; ElasticNet won on CV and was submitted twice (chained).
  - Per-model hyperparameters, exact Optuna search spaces, seeds, fold counts, library versions: not stated.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated (the only fold remark on the page is his agreement
  with 24th in a comment on 24th's page: more splits better, at a time cost).
- No CV AUC and no public/private AUC appear on the page, so the CV-vs-LB gap cannot be computed from this entry.
- Author's trust verdict (verbatim): "implement a stable and working pipeline early on and trust it works throughout this
  process (CV was reliable in the final submission)."
- **Ensembling:** two chained linear metas over the 115-column level-0 matrix.
- ElasticNet₁: default parameters, trained on the level-0 predictions (his page says "stacking" but never prints the word
  "out-of-fold"); selected because it had the best CV among L1/L2/ElasticNet.
- ElasticNet₂: same columns again plus ElasticNet₁'s own output as one more input; also default parameters.
- No hyperparameter optimization anywhere in the meta chain (author's explicit decision, to limit overfitting from 115
  inputs); no weights, hill-climbing, rank averaging or shared community OOF sets are mentioned.
- **Post-processing:** not stated. The chained second ElasticNet is a meta stage, not post-processing.
- **Gains:** NO numeric delta is published on this page (no AUC at all). Ordering statements only:
  - ElasticNet > L1 and L2 linear metas on CV (his meta bake-off).
  - Adding the second ElasticNet layer on top of the first: no before/after numbers are printed, and the page never actually
    claims the addition improved the score — a commenter (@sergeyzemskov) asks exactly "why predictions first predictions of
    meta-model improve overall score" with no author answer rendered on the page.
  - GAMs contributed diversity, not accuracy (author comment).
  - More stratified splits = better validation, at a time cost (author's comment on TPSSEP21-24's page).
- **Failed:** Modeling ALL GAM features with varying spline terms: always ended in memory errors; forced down to
  XGBoost-importance feature subsets.
- Pseudo labeling of the test set: "improved my previous submissions but not the final one. Maybe it needed more exploration,
  but it showed some promise! The final submission did not use any pseudo labeling."
- Optimizing the meta-models: deliberately not done (he judged optimization of the metas as an overfitting risk with so many
  base models).
- His whole solution "is divided into many notebooks and it's currently a mess" — no organized code release at writeup time.
- Unanswered questions on the page (so the information is genuinely absent, not withheld by this digest): which tool made the
  diagram, whether base/meta training sets must be separate, the collective memory footprint of 115 models, why the second
  meta helps, the k-means/outlier/t-SNE/UMAP feature recipe.
- **Comments:** the page has 29 comments (27 threaded + 2 appreciation); all material content is author replies, mined above:
  - @gauravbrills (231st) reports HistGB gave him a bad CV so he omitted it, and asks why ElasticNet over LogisticRegression —
    no author reply rendered.
  - @maxdiazbattan (48th) asks the GAM library + RAM question; author's full pyGAM recipe (above), "I had to use a machine with
    approximately 25 GB RAM to run the final notebook. You might achieve to run it at Kaggle, one model at a time."
  - @roberterffmeyer (491st) asks about pseudo labeling; author's rejection (above).
  - @raj401 (126th) asks whether the two ElasticNets share features and whether they were tuned; author's default-params +
    self-feed answer (above).
  - @faroqaltam (95th) asks which categoricals; author's `f29, f40, f42, f65, f70, f74, f75` answer (above).
  - @hikmetsezen (3rd), @mlanhenke (185th), @tunguz (1041st, "Really great to see that Stacking is still the way to go"),
    @ivankontic (78th), @faroqaltam (95th), @himanshunitrr (222nd), @adamwurdits (251st), @shravankoninti (188th),
    @towhidultonmoy (45th), @heyrobin (443rd), @hao0o0o (249th), @sergeyzemskov (200th), @anubhavchhabra, @pritiyadavml,
    @ankurlimbashia, @jaideepvalani: congratulations/code requests, no numbers.
- **Compute:** GAM notebook needed a machine with ~25 GB RAM (author comment); all-feature GAM attempts hit memory errors;
  GBMs on Kaggle CPU and/or GPU. Wall-clock, Kaggle limit hits, cost: not stated.
- **Artifacts:**
  - https://www.kaggle.com/vkonstantakos/generalized-additive-model-gam-stacking (the author's own GAM implementation notebook,
    linked in his comment reply)
  - https://i.postimg.cc/bvVttL7m/tps-sep.png (structure figure; page link https://postimg.cc/gwRj9ykh)
  - https://www.kaggle.com/mlanhenke
  - https://www.kaggle.com/realtimshady
  - https://www.kaggle.com/tunguz
  - https://www.kaggle.com/lucamassaron
  - https://www.kaggle.com/firefliesqn
  - https://www.kaggle.com/raj401 (the author replies under the handle @vasunarasimman in that commenter's greeting)
  - https://www.kaggle.com/vkonstantakos
  - https://www.kaggle.com/account/login (sign-in links rendered in the shell, no content)
  - named without URLs: Optuna, pyGAM `LogisticGAM`, XGBoost, LightGBM, CatBoost, HistGradientBoostingClassifier, TabNet,
    sklearn L1/L2/ElasticNet regression, k-means, t-SNE, UMAP
  - page tags: Tabular, Classification, Ensembling, Gradient Boosting, Optimization
- **Lesson:** With 100+ level-0 members, the winning edge is a deliberately un-tuned linear meta — and feeding the meta's own
  output back as one extra column is worth a stage, provided you accept that it will not scale to a laptop.

### TPSSEP21-06 · 6th · ankitkalauni (Ankit Kalauni) · LB 0.81875 public (final blend) / not stated · CV not stated (single LGBM: 0.81851 public)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275690
- **Status:** FETCHED on the first form (`/c/.../discussion/275690` + `X-Return-Format: markdown`), 6,178 bytes, body plus all
  8 comments.
- **TL;DR:** Borrowed 14th's VotingClassifier idea for one LGBM (0.81851 public), then blended notebooks to 0.81875 public;
  spent his last submission slot on the faster `liblinear` variant, so the `saga` blend — which later scored "the same score
  as the 4th position" — never reached the leaderboard.
- The whole technique came from community notebooks; he names seven people he learned from and states this was his first TPS.
- **Architecture:** FLAT · stages=1 · l1=not stated (>=1 seed-averaged LGBM plus other base models, then a blend of notebooks) ·
  l2=not stated (blending notebook contains a LogisticRegression, `solver='saga'` and `solver='liblinear'` variants; whether
  it is fitted on OOF or on LB is not stated, so this page cannot be tagged `STACK2`) · novel=none · mod=none ·
  topo=LGBM (+ other base models) via VotingClassifier → blend notebooks (LogisticRegression, solver saga | liblinear) → submission
  - The submitted artifact is a blend from a notebook titled "Blending (0.81875)"; no meta-learner fitted on an OOF matrix is
    claimed by the author, so rule 1 keeps this `FLAT`.
  - `mod=none` is deliberate: his only named ensemble device is sklearn `VotingClassifier` over copies of the same LGBM
    (which is rule 4 seed averaging, not a level), and he never states weights or ranks.
  - The two solver variants are the same architecture differing only in the solver — an accidental A/B test of LogisticRegression
    solvers on the final blend.
- **Setup:** rows/cols, split sizes, fold counts: not stated. Submission slots: he was "left with only one submission of the
  day" at the deadline (the structural constraint that cost him).
- **Features:** not stated at all in the writeup body — no engineered feature, encoding, imputer or dropped column is named.
- He thanks @realtimshady and others for notebooks generally, and lists no feature names.
- Feature-source kernels: the notebooks of the people he credits (handles only, no feature list).
- **Models:** LightGBM as the base model he names explicitly ("a single LGBM model"), combined through a `VotingClassifier`
  (idea credited to @martynovandrey).
- Blending step uses sklearn `LogisticRegression` — the two committed notebooks differ by `solver='saga'` vs `solver='liblinear'`.
- All hyperparameters, seeds, fold counts, other families in the blend: not stated.
- Author's credit line: "thanks to @martynovandrey for sharing about the voting classifier it helped me a lot, because of the
  voting classifier I got `0.81851 Pubilc LB` with a single LGBM model." (typo `Pubilc` is the author's)
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV score published.
- Only LB numbers on the page: 0.81851 public (single LGBM via voting) and 0.81875 public (final blend, from the notebook
  title); private not stated.
- **Ensembling:** two-stage in the colloquial sense only: seed-voted LGBM base, then a "blending" notebook over several
  submissions. Blend weights, meta-learner fit data, shared OOF usage: not stated.
- **Post-processing:** not stated.
- **Gains:** (+0.00024 public) single seed-voted LGBM 0.81851 -> final blend 0.81875 (both operands printed on the page).
- VotingClassifier over one LGBM: the author's framing is that the voting classifier is what carried him to 0.81851 public
  with a single model family; the pre-voting score is not stated, so the delta cannot be computed.
- (slot decision, negative) submitting `liblinear` instead of `saga` for his last slot: he reports the `saga` notebook scored
  "the same score as the 4th position" — his own `:(` marks it as the lost upgrade; no AUC given.
- **Failed:** `solver='saga'` took too long to run, so it could not be submitted inside his last daily slot.
- `DataTable` is never mentioned on this page (it is 45th's dead end); no FE is described here at all.
- No dead-end model or feature is named — the writeup is a two-paragraph credit list plus the solver anecdote.
- **Comments:** 8 comments on the page (counted: Edrick Kesuma, author, Martynov Andrey, author, ramkiran55 devireddy, author,
  Eugene Pasechnikov, author). Technical content:
  - Martynov Andrey (14th in this competition, the origin of the voting idea): "Congratulations! 0.81851 is great! I'm happy
    that it works." — confirms the number and the cross-competition transfer.
  - Author to @edrickkesuma (20th): exams, joining TPS Oct as a team, "I have almost done with the base models".
  - Eugene Pasechnikov (271st): "Great job! I am glad to help!" — his notebooks are in the author's credit list.
  - No author reply in the comments adds any hyperparameter, feature name or fold count.
- **Compute:** not stated except that the `saga` solver run was the slow job ("was taking too much time to run").
- **Artifacts:**
  - https://www.kaggle.com/ankitkalauni/6th-place-solution-blending-0-81875 (his final submission notebook)
  - https://i.ibb.co/XZFCrr5/newplot.png (the saga-vs-liblinear plot; page link https://ibb.co/TBPgxx0)
  - https://www.kaggle.com/mlanhenke
  - https://www.kaggle.com/realtimshady
  - https://www.kaggle.com/ivankontic
  - https://www.kaggle.com/dlaststark
  - https://www.kaggle.com/eugenebee
  - https://www.kaggle.com/shreyaspj
  - https://www.kaggle.com/edrickkesuma
  - https://www.kaggle.com/martynovandrey
  - https://www.kaggle.com/ramkiran55devireddy
  - https://www.kaggle.com/ankitkalauni
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/ankit-kalauni-6th-place-solution-tps-sep-2021 (citation line)
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021
  - page tags: Beginner, Tabular, Python, pandas, Data Visualization
- **Lesson:** Steal the seed-voting classifier: it turns one mediocre LGBM into a medal-position submission — then never spend
  your last daily slot on whichever notebook is still running.

### TPSSEP21-07 · 7th · stevenrferrer (Steven Ferrer) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/277256
- **Status:** FETCHED on the first form (`/c/.../discussion/277256`), 8,587 bytes, body + all 5 comments.
- **TL;DR:** Baseline zoo (LGBM/CatBoost/HistGB/XGB) built almost entirely from other people's hyperparameters, a
  VotingClassifier of three of them, and a final blend of all five; publishing every prediction as a dataset.
- His single biggest jump came from one insight: the NaNs are informative, so capture them before imputing.
- **Architecture:** FLAT · stages=1 · l1=5 members (LightGBM, CatBoost, HistGradientBoosting, XGBoost, VotingClassifier of
  LGBM+CatBoost+HistGB) · l2=none (blend; weights not stated, no meta learner claimed) · novel=none · mod=none ·
  topo=NaN-capture FE → {LGBM, CatBoost, HistGB, XGB} + VotingClassifier[LGBM+CatBoost+HistGB] → blend of all five → submission
  - The `VotingClassifier` is itself an average of three families, so the submission is an average of averages; no estimator
    is fitted on top of the members, which is why this is `FLAT` and not `STACK2`.
  - XGBoost appears only as a standalone member — it was excluded from the voting ensemble for cost (9 hours, unfinished).
  - He publishes the intermediate predictions of every member as a dataset, i.e. the level-0 matrix is reusable by others.
- **Setup:** rows/cols, split sizes, fold counts, submission slots: not stated.
- Structural quirk he exploited: NaN values carry signal and must be captured BEFORE imputation.
- **Features:** no feature names or formulas are published in the writeup; FE is credited, not described.
  - FE ideas: @realtimshady's notebook (URL under Artifacts) for "FE ideas".
  - EDA and the missing-values story: @dwin183287 and @lucamassaron.
  - His own statement of the fix: he was "using `SimpleImputer` without capturing the usefulness of NaN values"; after
    "learning that the NaN values are actually useful and capturing it before imputation, the results got better even with
    just the base models."
  - Exact encodings, GP features, dropped columns: not stated.
- **Models:** five members, each a public-hyperparameter baseline:
  - LightGBM baseline — hyperparameters "Most likely using @mlanhenke hyperparams" (author cannot recall the source and asks
    commenters to identify it; nobody does in the 5 comments).
  - CatBoost baseline — @mlanhenke's `tps-09-single-catboostclassifier` hyperparameters; also thanks @shenurisumanasekara's
    step-by-step notebook.
  - HistGradientBoosting baseline — @tunguz's Optuna-derived hyperparameters.
  - XGBoost baseline — again "forgot whose hyperparam", most likely @mlanhenke's.
  - VotingClassifier of LightGBM/CatBoost/HistGradientBoosting with the same per-model hyperparameters; idea credited to
    @dmitryuarov's voting notebook.
  - Numeric hyperparameter values, seeds, fold counts, library versions: not stated for any member.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV score published.
- Early local/leaderboard scores he quotes as ranges only: `0.5xx` - `0.79x` before NaN capture; no final AUC.
- **Ensembling:** "For blending, I combined all of the above models to produce the final submission" — blend weights,
  averaging type, meta-learner: not stated.
- Shared-OOF usage: he consumed others' hyperparameters, and he published his own predictions as a dataset
  (`tps-september-2021-preds`) for others to consume.
- He credits @ankitkalauni (6th) for "the visualisations on model correlations" used to pick the blend.
- **Post-processing:** not stated.
- **Gains:** unquantified in the source: the NaN-capture-before-imputation change moved him out of `0.5xx`-`0.79x` into
  competitive base models ("the results got better even with just the base models") — no post-change number is printed.
- (negative, quantified as time) adding XGBoost to the VotingClassifier: 9 hours of training and it "did not finish"; removing
  it changed nothing measurable and he calls the exclusion lucky.
- **Failed:** `SimpleImputer` applied without a NaN indicator: his models stuck at `0.5xx`-`0.79x`.
- XGBoost inside the VotingClassifier on GPU: 9 hours, unfinished, plus "I ran out of GPU"; the CPU-only voting ensemble scored
  well, which he reports as the better outcome.
- Unresolved attribution: two of his five members have hyperparameters whose original author he cannot remember, and the
  comment thread never supplies it — so the entry is not fully replicable from his page alone.
- **Comments:** 5 comments (counted: F. AL-Tam, Towhidul.Tonmoy, Sharlto Cope/dwin183287, Mohammad Kashif, Ankit Kalauni); no
  author replies, so the body is the whole record on the author's side. The one technical comment:
  - Ankit Kalauni (6th): "it was my first competition. I haven't added histgbm … etc, only lgbm xgb and catboost in base. I
    realize now that diversity in models gives better generalization in stacking."
  - Towhidul.Tonmoy (45th) links his own writeup: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275720#1530732
- **Compute:** XGBoost training ~9 hours without finishing; ran out of Kaggle GPU quota; the good voting ensemble ran on CPU.
  RAM, wall-clock totals: not stated.
- **Artifacts:**
  - https://www.kaggle.com/realtimshady/single-simple-lightgbm
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-lightgbm-baseline
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-catboost-baseline
  - https://www.kaggle.com/mlanhenke/tps-09-single-catboostclassifier
  - https://www.kaggle.com/shenurisumanasekara/catboost-step-by-step-improved
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-histgradientboosting-baseline
  - https://www.kaggle.com/tunguz/tps-09-21-histgradientboosting-with-optuna
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-xgboost-baseline
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-votingclassifier-baseline
  - https://www.kaggle.com/dmitryuarov/tps-voting-xgb-cb-lgbm
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-blend-lgb-xgb-cb-hgb-voting
  - https://www.kaggle.com/stevenrferrer/tps-september-2021-preds (dataset of all his prediction outputs)
  - https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275720#1530732 (from a comment)
  - https://www.kaggle.com/mlanhenke · https://www.kaggle.com/tunguz · https://www.kaggle.com/lucamassaron ·
    https://www.kaggle.com/dwin183287 · https://www.kaggle.com/craigmthomas · https://www.kaggle.com/mohammadkashifunique ·
    https://www.kaggle.com/davidcoxon · https://www.kaggle.com/faroqaltam · https://www.kaggle.com/towhidultonmoy ·
    https://www.kaggle.com/ankitkalauni · https://www.kaggle.com/stevenrferrer
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/steven-ferrer-7th-place-solution-lots-of-trial-and (citation line)
  - page tags: Binary Classification, Tabular, Beginner, Intermediate
- **Lesson:** Capture the NaNs before you impute them, then let model diversity (a fourth family plus a voting member) do the
  climbing — and publish your per-model predictions so the next beginner can skip your 9-hour XGBoost mistake.

### TPSSEP21-09 · 9th · aries1988 (Gang HUANG) · LB not stated/not stated (final standing 9th, private) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276248
- **Status:** FETCHED. `/c/.../discussion/276248` returned a 407-byte shell; `X-No-Cache: true` on the same form returned the
  full post (10,305 bytes) including its 1 comment + 1 appreciation comment.
- **TL;DR:** 34 level-0 models → LinearRegression at L1 → LinearRegression at L2 consuming both L0 and L1 outputs; then the
  "fit to all" trick (retrain on all data with `n_estimators` pushed past the CV optimum) and a final flat blend.
- He rates the fit-to-all trick: "This may be the most important thing that I've learned in this competition".
- **Architecture:** STACKN · stages=3 · l1=34 L0 models (XGBoost/LightGBM variants, seed replicas, several public
  LGBM/stacking notebook recipes — CatBoost is never named on his own page) · l2=LinearRegression (author's L1 meta) · l3=LinearRegression (author's L2 meta, fed both L0 and L1
  outputs) · novel=none · mod=public-oof ·
  topo=nan-count + nan-based target encoding → 34 L0 models → LinearRegression(L0 OOF) → LinearRegression(L0+L1 outputs) → flat blend with 5 "fit to all" models + @mlanhenke's public submission
  - Three trained model stages, so `STACKN`: level 0 (34 models), his "L1" meta, and a second meta that reuses the L0
    columns AND the L1 prediction as inputs.
  - The submitted prediction is not the stack alone: the final submission averages the stack's L1 prediction, five "fit to all"
    models and one external public submission, so a `FLAT` layer sits on top of the stack.
  - He considered weights/power blending for that final layer and decided against it, i.e. the final blend is as close to a
    plain average as the page allows; the exact weights are not stated.
  - `mod=public-oof` covers both the L0 recipes he took from community notebooks and @mlanhenke's published submission inside
    the final blend.
- **Setup:** rows/cols and fold sizes of the stack: not stated; Optuna ran 5-fold CV on a 20% random sample of the training
  data because the file was too large; mid-competition position "about 30th position on the public LB"; final 9th on private.
- **Features:**
  - Per-row number of NaNs — the feature that put him above 0.8 AUC, discovered by @prikshitsingla and discussed in the
    competition thread (URL under Artifacts); he notes it "has been shown to be correlated the target value".
  - Target encoding based on the number of NaNs in a row.
  - He states "I've not done much EDA in this competition" and lists no other feature names.
  - Seed replication of models ("used various seeds to replicate some models, which seemed to work").
  - Encodings beyond TE, dropped columns, GP/generated features: not stated.
- **Models:**
  - Level 0: 34 models in total, assembled from his own XGBoost/LightGBM Optuna-tuned classifiers plus replicas of
    @hiro5299834's single-LGBM, @ivankontic's colsample-LGBM, @mlanhenke's blend&stacking LGBM, @realtimshady's
    single-LightGBM, and stacking recipes from @vishwas21 and @manabendrarout.
  - "One model voting": the four LightGBM recipes above plus one of his own XGBoost models, each run with **7 different
    `random_state` seeds** and blended.
  - "Fit to all": 5 models retrained on the entire training set with `n_estimators` extended beyond the CV early-stopping
    optimum; early stopping in CV used `early_stopping_rounds` around 300-500.
  - Meta L1: LinearRegression. Meta L2: LinearRegression (inputs = L0 outputs + L1 output).
  - Rejected metas: XGBoost and LightGBM classifiers as meta models, with Optuna tuning at L1.
  - Per-model hyperparameter values, fold counts per member, library versions: not stated.
- **CV:** 5-fold CV inside Optuna, on a 20% random subsample (for speed).
- CV for the stack itself: splitter type and fold count not stated.
- Public-vs-private behaviour: for the seed-voting step, "the resulting rise in the public LB is more pronounced than the rise
  in the private LB. So the 'effectiveness' of these improvements may be less than, say, what stacking models gives."
- Trust verdict: he trusts CV for hyperparameters but distrusts it for model capacity, hence the fit-to-all experiment; no AUC
  numbers published, so no gap computable.
- **Ensembling:** L0 34 models → LinearRegression → LinearRegression; then a final flat blend of {stack L1 prediction, 5 fit-to-all
  predictions, @mlanhenke's submission} with weights deliberately left alone (no power blending applied).
- Model-comparison device: kernel density plots of the predicted values across submissions to spot "eccentric" models.
- Shared-OOF usage: yes — other competitors' notebooks/submissions feed both the L0 pool and the final blend.
- **Post-processing:** none stated. Power blending is named and then declined; no threshold, clipping or rounding.
- **Gains:** no AUC deltas anywhere on the page; the ladder is qualitative and ordinal:
  - NaN-count feature: lifted AUC "above a sound 0.8" (his words, no number).
  - Stacking (34 L0 + linear metas): took him to ~30th on the public LB ("about halfway into my climb").
  - One-model voting (7 seeds per recipe): "all gave a noticeable improvement of score", smaller on private than public.
  - Fit-to-all with extended `n_estimators`: "A further improvement" — the one he rates most important.
  - Final flat blend with a public submission: put him "among the top 10 on the private LB".
- **Failed:** Further Optuna searches after the first 20%-sample sweep: "led to nowhere and gave no improvement on the public
  LB score"; he explicitly says "I was not quite successful in uncovering a lot of great models".
- XGBoost or LightGBM as the meta model (with Optuna tuning at L1): did not help his stacking score.
- Blindly adding mediocre L0 members for diversity's sake: names `RandomForestClassifier` and `LogisticRegression`.
- Pseudo labeling (the Aug-2021 1st-place idea he links): did not help his stacking score.
- Working on the full file with Optuna: had to fall back to a 20% random sample.
- **Comments:** the page header counts 1 comment plus 1 appreciation comment; the only rendered entry is Saurav Maheshkar's
  "Thanks for sharing. Gr8 work". No author replies at all despite his closing "If you have questions please leave a message" —
  so nothing technical exists outside the body.
- **Compute:** Optuna on a 20% sample to stay inside time/memory; no wall-clock, GPU or RAM figures stated.
- **Artifacts:**
  - https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/270206 (the NaN-count-correlation post)
  - https://www.kaggle.com/c/30-days-of-ml/discussion/265755 (@abhishek22211's stacking explanation)
  - https://www.kaggle.com/vishwas21/tps-sep-21-3-level-custom-stacking
  - https://www.kaggle.com/manabendrarout/custom-stacking-of-classifiers-gpu-tps-sep2021/
  - https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb/
  - https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/273253 (@edrickkesuma on power blending)
  - https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/274404 (@martynovandrey's one-model-voting post)
  - https://www.kaggle.com/c/tabular-playground-series-aug-2021/discussion/270051 (the pseudo-labeling post he tried and rejected)
  - https://www.kaggle.com/hiro5299834/tps-sep-2021-single-lgbm
  - https://www.kaggle.com/ivankontic/004-2o-lightgbm-colsample-tps-sep-2021
  - https://www.kaggle.com/realtimshady/single-simple-lightgbm
  - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.kde.html
  - https://i.postimg.cc/Gpj0tPsf/score.png (log-scale gap-vs-best-progression figure)
  - https://i.postimg.cc/fbDGNCvJ/results-6-1.png (KDE density plot of all submission predictions)
  - https://www.kaggle.com/pourchot · https://www.kaggle.com/prikshitsingla · https://www.kaggle.com/abhishek22211 ·
    https://www.kaggle.com/vishwas21 · https://www.kaggle.com/manabendrarout · https://www.kaggle.com/mlanhenke ·
    https://www.kaggle.com/edrickkesuma · https://www.kaggle.com/martynovandrey · https://www.kaggle.com/hiro5299834 ·
    https://www.kaggle.com/ivankontic · https://www.kaggle.com/realtimshady · https://www.kaggle.com/aries1988 ·
    https://www.kaggle.com/sauravmaheshkar
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/when-september-ends-my-9th-place-solution-to-the-s (citation line)
- **Lesson:** Two stacked LinearRegressions that re-consume their own previous output beat every exotic blender, and the last
  real gain came from retraining on all data with more estimators than CV said was optimal — a place CV cannot see.

### TPSSEP21-10 · 10th · pliptor (Oscar Takeshita) · LB not stated/not stated (best private 0.81756, unselected) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276009
- **Status:** FETCHED, with the escape hatch. `/c/.../discussion/276009` returned the full body (9,904 bytes) but only 9 of the
  15 comments; the `/competitions/` + `X-No-Cache` form and the `?sort=votes` variant returned byte-identical truncations, so
  the last 6 replies were recovered by opening the page with the browser tools and expanding the "6 more replies" control
  (accessibility/`innerText` snapshot), which renders all 15 comments.
- **TL;DR:** 17 level-0 models (5 LightGBM, 4 XGBoost, 2 CatBoost, 6 Keras) stacked by plain LogisticRegression, with exactly
  one engineered feature: the per-row NaN sum.
- He came back after 3 years away, wanted top 10, preferred a single model, and "succumbed to stacking".
- **Architecture:** STACK2 · stages=2 · l1=17 (5 LightGBM, 4 XGBoost, 2 CatBoost, 6 Keras) · l2=LogisticRegression ·
  novel=MLP+embedding · mod=none ·
  topo=raw features (count not stated) + per-row NaN sum → 17 L0 models (GBM x11 + Keras NN x6) → LogisticRegression on OOF → submission
  - Two trained stages, single meta layer: exactly the "single level stacking structure" he attributes to @vkonstantakos (2nd),
    with a smaller pool.
  - `novel=MLP+embedding` names only the Keras level-0 members: his kernels are derivatives of @lukaszborecki's
    `tps-09-nn`, which has embedding layers over 96 binned inputs plus a separate un-binned input, and `swish` activation.
    They were kept purely for decorrelation, not strength — "Keras didn't give me scores anywhere close to the gradient boost
    ones."
  - The meta is stock `LogisticRegression`, chosen partly because its coefficients expose which base models carry weight; the
    Keras member's embedding is not fed to the GBMs, so this is a stack, not a `CASCADE`.
- **Setup:** rows/cols and fold counts not stated; 2 submission slots implied by the public/private selection story
  (Kaggle auto-selects public; he had to choose the private himself).
- Structural quirk exploited: the row-wise missingness count as the only feature beyond the raw columns.
- **Features:** one engineered feature in every single-model design: "the sum of NA's in the row. It was clearly effective from
  discussions and it was well understood why it works from past competitions."
- He tried other engineered features and dropped them "in the interest of time" — names and results not stated.
- The Keras branch internally uses 96 bins per input for its embedding layers (his description of the source kernel).
- Encodings for categorical inputs, dropped columns, GP features: not stated.
- **Models:**
  - Families tried: XGBoost, LightGBM, CatBoost, SVM, Keras.
  - Final pool 17 members: 5 LightGBM, 4 XGBoost, 2 CatBoost, 6 Keras.
  - Hyperparameters tuned with Optuna (he switched to Optuna from the Bayesian optimization of 3 years prior; he does not say
    which family got which search space).
  - One borrowed recipe he singles out: @mlanhenke's "xgb1" component with `max_level` 18 (the author's exact token) —
    poor alone, "seemed crucial for the overall stacking", "doing the job of adding a strong uncorrelated component".
  - Seeds, per-member fold counts, library versions, Keras layer widths: not stated.
- **CV:** CV scheme details (splitter, folds, repeats, stratification, seeds) not stated; no CV AUC published.
- Private selection rule: "I selected my private by using local cross validation."
- Published LB numbers on the page: "My best private score was 0.81756 but I didn't get to select that submission because CV
  was worse. This may explain a performance gap." — a commenter (@hikmetsezen, 3rd) observes his private is worse than his
  public; neither number is stated for the submission he actually placed with. The only other AUC on the page is the discarded
  LinearSVC's public 0.80104 (author comment), i.e. the one number he does publish for a single non-GBM model.
- **Ensembling:** one-level stack, meta = LogisticRegression over the 17 OOF columns.
- XGBoost was tried as the meta and "wasn't performing well", so it was dropped.
- Why LogisticRegression: "LogisticRegression also gives how much weight each of its input component is being used so it's
  easier to tell which base models to keep."
- Meta hyperparameters, blend weights of members, shared-OOF sets from others: not stated (he used others' recipes, not others'
  predictions).
- **Post-processing:** not stated for the submitted stack. The only calibration device named anywhere on the page is isotonic
  calibration (`CalibratedClassifierCV(..., method='isotonic', ensemble=False)`) on the discarded LinearSVC branch, in his
  comment reply.
- **Gains:** no AUC delta is published for any change. Ordinal statements only:
  - NaN-sum feature: the single engineered feature he kept across all designs, justified by prior competitions.
  - Stacking over single models: the move that made him competitive — "the single models were not scoring competitively at the
    leader board."
  - Keeping the weak-but-uncorrelated `xgb1` in the pool: named as the component-level lesson of the whole comp.
- **Failed:** SVM: discarded on score and on convergence — `sklearn.svm.LinearSVC(dual=False, penalty='l1', fit_intercept=True,
  max_iter=1000)` wrapped in `CalibratedClassifierCV(svm, method='isotonic', ensemble=False)`, CPU only (LinearSVC has no GPU
  path), "it failed to converge, and I only got 0.80104 on the public score with it so I discarded it" (author comment reply;
  @tunguz's answer to the size problem was cuML Rapids SVM, which he did not try).
- Keras NN as a standalone model: never close to the gradient-boosted scores.
- XGBoost as the stacking meta: underperformed LogisticRegression, discarded.
- Other engineered features beyond the NaN sum: tried, then dropped for time; results not stated.
- His best private submission (0.81756): failed to be selected because its CV was worse — the cost of choosing private by CV
  after Kaggle fixed the public slot.
- **Comments:** all 15 comments live in the thread (9 rendered by every proxy form, the last 6 only after opening the page in
  the browser and expanding "6 more replies"). Technical content, all in author replies:
  - To @hikmetsezen (3rd): "I used plain one level stacking" + "we don't get control over the public score. Kaggle always pick
    the highest scoring one. The private one, however, you must make a selection. I selected my private by using local cross
    validation. My best private score was 0.81756 but I didn't get to select that submission because CV was worse. This may
    explain a performance gap."
  - To @tunguz, the full SVM recipe and its score: "I was using `sklearn.svm.LinearSVC`. I think this version doesn't support
    GPU ... `svm = LinearSVC(dual=False, penalty='l1', fit_intercept=True, max_iter=1000)` /
    `model = CalibratedClassifierCV(svm, method='isotonic', ensemble=False)` ... This is the configuration I tried during the
    competition. It runs on CPU, it failed to converge, and I only got **0.80104** on the public score with it so I discarded it."
  - @tunguz (1041st) asks which SVM version and notes "The dataset is too large for sklearn SVM, but maybe could be handled with
    a GPU version", then points to cuML Rapids SVM (URL in Artifacts).
  - @arnabbiswas1 (arnab, 234th in this competition) and @tunguz discuss why he is leaving Kaggle ("I really hate the 'code'
    competition format. And I don't like the fact that we haven't had a proper tabular data competition in years") — non-technical
    for replication, but it dates the board.
  - @towhidultonmoy (45th) and @adamwurdits (251st): congratulations only.
- **Compute:** "many short sleeping nights"; SVM ran CPU-only and never converged (author comment). Wall-clock per model, GPU
  quota, RAM: not stated.
- **Artifacts:**
  - https://www.kaggle.com/lukaszborecki/tps-09-nn (the Keras kernel all his NN kernels derive from)
  - https://medium.com/rapids-ai/fast-support-vector-classification-with-rapids-cuml-6e49f4a7d89e (cuML Rapids SVM, cited by
    commenter @tunguz in the expanded reply thread, not by the author)
  - https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb (source of the `xgb1` uncorrelated component)
  - https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275740 (his reference for the single-level stacking
    structure = the 2nd-place writeup)
  - https://www.kaggle.com/lukaszborecki · https://www.kaggle.com/tunguz · https://www.kaggle.com/mlanhenke ·
    https://www.kaggle.com/vkonstantakos · https://www.kaggle.com/pliptor · https://www.kaggle.com/hikmetsezen ·
    https://www.kaggle.com/adamwurdits · https://www.kaggle.com/towhidultonmoy · named without URL: @arnabbiswas1 ("arnab",
    234th in this competition), whose avatar appears in the collapsed-reply strip
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/oscar-takeshita-10th-place-method-moderate-stackin (citation line)
  - https://storage.googleapis.com/kaggle-avatars/thumbnails/417337-fb.jpg ·
    https://storage.googleapis.com/kaggle-avatars/thumbnails/1020983-kg.png ·
    https://storage.googleapis.com/kaggle-avatars/thumbnails/238237-kg.jpg
  - page tags: XGBoost, LightGBM, Keras
- **Lesson:** The reason to keep a bad model in a stack is decorrelation, not accuracy — a component that scores poorly alone
  (`xgb1`, all six Keras NNs) can be the reason the meta beats every single model.

### TPSSEP21-14 · 14th · martynovandrey (Martynov Andrey) · LB 0.81868/0.81752 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276141
- **Status:** FETCHED. `/c/.../discussion/276141` returned the full body + all 4 comments (6,823 bytes) on the first form.
- **TL;DR:** The origin of "one model voting": VotingClassifier over identical LGBM/CatBoost estimators differing only in
  `random_state`, then hand-set [0.7, 0.3] weighted blends across three such branches, finishing 0.81868 public / 0.81752 private.
- His method was copied by at least two other competitors on this board (6th, and 9th — the latter also confirming it in his
  comment below).
- **Architecture:** FLAT · stages=1 · l1=3 branches (seed-voted LGBM, seed-voted CatBoost, seed-voted modified-ivankontic
  LGBM) plus one external public submission in the middle of the chain · l2=none (hand-set [0.7, 0.3] weights, no fitted meta) ·
  novel=none · mod=public-oof ·
  topo=LGBM{seed-voting}=0.81839 + CatBoost{seed-voting}=0.81816 → weights[0.7,0.3]=0.81846 → blend with @mlanhenke's public
  submission=0.81854; [ivankontic LGBM modified]=0.81835 → {seed-voting}=0.81845 → weights[0.7,0.3]=0.81868 public / 0.81752 private
  - Everything inside a branch is rule 4 seed averaging (same architecture, different `random_state`), so it never counts as a
    level; the submitted object is a weighted average of two branch predictions, hence `FLAT`.
  - The 0.7/0.3 weights are hand-chosen, not fitted: no OOF-fitted meta-learner appears anywhere on the page.
  - `mod=public-oof` refers to the intermediate 0.81854 step, where he blended his 0.81846 with @mlanhenke's published
    stacking submission; the page does not say whether that intermediate survived into the final 0.81868 blend.
- **Setup:** rows/cols, fold counts: not stated. Footnote he adds himself: "Some notebooks executed on local PC, so the scores
  above may differ with notebooks shared."
- Structural quirk exploited: identical hyperparameters with only `random_state` varied — cheaper than CV per model and works
  for classification.
- **Features:** none described. The models he names (his LGBM notebook, CatBoost notebook, @ivankontic's colsample LGBM) carry
  their own preprocessing; he states no feature names, no encodings, no NaN-count feature of his own.
- The only column-level change he reports is `colsample` (via the notebook he modified) and `random_state`.
- **Models:** public scores in the order he reports them:
  - Traditional single LightGBM: 0.81802 (`tps-september-lgbm`).
  - Traditional CatBoost: 0.81751.
  - LGBM with one-model voting: 0.81839.
  - CatBoost with one-model voting: 0.81816.
  - Weighted average of the two, weights [0.7, 0.3]: 0.81846.
  - Blended with @mlanhenke's [[TPS-09] Simple Blend & Stacking (XGB, LGBM, CATB)] submission: 0.81854.
  - Modified @ivankontic "[004-2o] lightGBM colsample" notebook: 0.81835 → with one-model voting: 0.81845.
  - Final blend, weights [0.7, 0.3]: 0.81868 public / 0.81752 private.
  - Author's table in comments (columns are private, public, in that order per his header "**Classifier**, **private**,
    **public**"): LGBM 0.81704/0.81802, Voting LGBM 0.81730/0.81839, CAT 0.81657/0.81751, Voting CAT 0.81721/0.81816.
  - Number of seeds per voting classifier, hyperparameter values, fold counts: not stated.
- **CV:** no CV scheme or CV score published; his evidence is submission scores only, plus his own caveat that some notebooks
  ran locally and shared-notebook scores may differ.
- Public-to-private delta for the final blend: 0.81868 -> 0.81752 = -0.00116, the only such gap computable in this set.
- **Ensembling:** hand-set weighted average, weights [0.7, 0.3], applied twice (once at 0.81846, once at the final 0.81868).
- VotingClassifier with same-architecture/different-`random_state` estimators is the core device, published as his own notebook.
- Shared-OOF usage: one external published submission (@mlanhenke's) blended in at the 0.81854 step.
- **Post-processing:** not stated.
- **Gains:** (+0.00037 public) LGBM 0.81802 -> one-model voting 0.81839.
- (+0.00065 public) CatBoost 0.81751 -> one-model voting 0.81816.
- (+0.00026 private / +0.00037 public) LGBM voting, from his comment table: 0.81704 -> 0.81730 private, 0.81802 -> 0.81839 public.
- (+0.00064 private / +0.00065 public) CatBoost voting: 0.81657 -> 0.81721 private, 0.81751 -> 0.81816 public.
- (+0.00007 public) weighted [0.7, 0.3] of the two voting branches: 0.81839 -> 0.81846.
- (+0.00008 public) blending with @mlanhenke's submission: 0.81846 -> 0.81854.
- (+0.00014 public) switching the second branch to the modified ivankontic LGBM + voting, then the final [0.7, 0.3] blend:
  0.81854 -> 0.81868.
- (+0.00010 public) the modified @ivankontic notebook 0.81835 -> voting 0.81845.
- **Failed:** The method's own ceiling: his final rank "was decreased by 3" on the private LB. He attributes it to three
  specific competitors (private places 7, 8 and 13) climbing +62, +37 and +15 places past him — i.e. the loss was other people's
  private behaviour, not a failed model.
- He names no failed feature or model family; his stated conclusion is positive ("the idea of single model voting do works in
  classification").
- **Comments:** 4 comments on the page (counted: Hikmet Sezen, author, Gang HUANG, author). Material content:
  - Author's private/public table for the four member notebooks (above) — posted in reply to @hikmetsezen, who asked precisely
    so readers could judge which method is robust.
  - Gang HUANG (aries1988): "I used your method in the later part of this competition and it allowed me to further improve my
    previous stacking model. It has saved me time too, since it costs less time to train on all data once than doing
    cross-validations for each model." Note for cross-reading: this commenter is the 9th-place entrant (TPSSEP21-09), whose own page
    credits @martynovandrey's one-model-voting post as well — the two pages corroborate each other on the transfer.
  - Author's explanation of +62/+37/+15 = private places 7, 8 and 13.
- **Compute:** not stated (local PC used for some notebooks, which is why he caveats the scores).
- **Artifacts:**
  - https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837
  - https://www.kaggle.com/martynovandrey/tps-september-lgbm
  - https://www.kaggle.com/martynovandrey/one-model-catboost-voting
  - https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb
  - https://www.kaggle.com/ivankontic/004-2o-lightgbm-colsample-tps-sep-2021
  - https://www.kaggle.com/mlanhenke · https://www.kaggle.com/ivankontic · https://www.kaggle.com/edrickkesuma ·
    https://www.kaggle.com/martynovandrey · https://www.kaggle.com/hikmetsezen · https://www.kaggle.com/aries1988
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/martynov-andrey-14th-place-solution-single-model-w (citation line)
  - page tags: Classification, Ensembling, LightGBM, Tabular
- **Lesson:** Voting over `random_state` clones of one architecture is the cheapest real AUC in a Playground (+0.00037 LGBM,
  +0.00065 CatBoost for a single training pass) — but it buys public LB, and public gains do not survive the private reshuffle.

### TPSSEP21-24 · 24th · saurabhbagchi (Old Monk) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276038
- **Status:** FETCHED. `/c/.../discussion/276038` gave a 1,647-byte shell; `X-No-Cache: true` on the same form returned the full
  post + all 5 comments (5,518 bytes).
- **TL;DR:** Three GBM families, 10 stratified folds, OOF predictions saved into a meta-set, one LogisticRegression on top;
  a jump of ~80 places at the close because his public LB was not representative.
- Everything he did is a smaller copy of @mlanhenke's public stacking notebook, which he credits for teaching him both Optuna
  and meta-learners.
- **Architecture:** STACK2 · stages=2 · l1=XGBoost + CatBoost + LightGBM base models (member count not stated) ·
  l2=LogisticRegression on the saved OOF meta-set · novel=none · mod=none ·
  topo={XGB, CatBoost, LGBM} × 10 stratified folds → saved OOF predictions as meta-set → LogisticRegression → probability submission
  - A textbook two-level stack: level-0 models are trained inside the folds, their OOF output becomes the meta-set, and one
    logistic model is fitted on it — the only trained meta-learner in the pipeline.
  - 10 folds is a deliberate capacity choice, not a default: "I also increased the stratified K-fold split to 10 to achieve
    better validation and more robust model."
  - No community predictions in the pool (unlike 9th and 14th), hence `mod=none`.
- **Setup:** rows/cols, member counts, submission slots: not stated. He was outside the public top 100 and finished 24th —
  "Pleasantly surprised to see my submission jump up by almost 80 places".
- **Features:** no feature engineering of his own is described. The page names none; the pipeline he copies is
  @mlanhenke's, so encodings/imputation/column handling are all `source silent`.
- **Models:** XGBoost, CatBoost, LightGBM as base learners; the approach "similar to what @mlanhenke did in his public notebook".
- He trained "different base models" and saved the `oof_predictions` to build the meta-set at the end.
- Hyperparameters, Optuna search space (he says he learned Optuna from mlanhenke's notebooks but never states a tuned value),
  seeds, fold counts per family: not stated; 10 stratified folds for the OOF scheme.
- **CV:** Stratified K-fold, 10 splits, chosen for "better validation and more robust model".
- No CV score and no LB score published, so the CV-vs-LB gap cannot be computed.
- Trust verdict implied by outcome: he had "almost given up after my submissions could not get me in the top 100 in the public
  leader board", then the private reshuffle rewarded his CV-driven choice of 10 folds.
- **Ensembling:** LogisticRegression meta-learner over the concatenated OOF predictions of the three families; meta
  hyperparameters not stated; no blending of other competitors' submissions.
- **Post-processing:** not stated (the meta outputs the submitted probability directly).
- **Gains:** (rank +80 places at close, no AUC) the private-over-public reshuffle — his own headline result and the only
  measured effect on the page.
- (qualitative, +2 authors' agreement) 5-fold -> 10 stratified folds: "better validation and more robust model"; commenter
  Vasilis Konstantakos (2nd in this competition) on this page: "Increasing the number of splits was better but at the cost of time!"
- **Failed:** Public LB as a selection signal: his public standing (outside top 100) was wrong by ~80 places, and he had given
  up before the close.
- Time: the stratified 10-fold runs were slow enough that he could not finish them interactively.
- He names no failed model or feature.
- **Comments:** 5 comments (counted: Vasilis Konstantakos, author, Vivek Chowdhury, author, Vivek Chowdhury). Technical:
  - Vasilis Konstantakos (2nd) independently confirms the more-splits-better result and its time cost.
  - Vivek Chowdhury asks whether stratified K-fold was slow for him and whether he speeded it up; author: "yes it took a lot of
    time for me as well. I used to leave it running overnight and then submit the completed notebook next day morning."
- **Compute:** no accelerator named; the workaround for Kaggle's interactive timeout was overnight batch runs, submitted the
  next morning. Wall-clock figures not stated.
- **Artifacts:**
  - https://www.kaggle.com/mlanhenke (the credited public notebooks, named by handle only — no notebook URL in the body)
  - https://www.kaggle.com/vkonstantakos · https://www.kaggle.com/vivek468 · https://www.kaggle.com/saurabhbagchi
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/old-monk-24th-place-solution-logistic-model-on-top (citation line)
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021
  - page tags: XGBoost, Optimization, LightGBM, Ensembling, Logistic Regression
- **Lesson:** A three-family, 10-fold, logistic-meta stack is the smallest honest stack that still medals here — and public LB
  position on this dataset tells you almost nothing (24th came from outside the public top 100).

### TPSSEP21-45 · 45th · towhidultonmoy (Towhidul.Tonmoy) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275720
- **Status:** FETCHED. `/c/.../discussion/275720` returned a 164/345-byte shell on the plain, `X-No-Cache` and
  `X-Engine: browser` forms; recovered the full post + all 7 comments (10,468 bytes) via
  `/competitions/.../discussion/275720` + `X-No-Cache: true` (ladder step 4).
- **TL;DR:** A lessons-learned list rather than a solution: memory reduction, stratified folds, `SimpleImputer(mean)` +
  StandardScaler, seed-replicated GBM ensembles, hand-tuned weighted average plus a power transform of the blend.
- Publishes no model configuration, no fold count per model and no score at all.
- **Architecture:** FLAT · stages=1 · l1=not stated (LightGBM + XGBoost + CatBoost variants, each replicated with different
  random states) · l2=none (weighted average, no fitted meta) · novel=none · mod=none ·
  topo=SimpleImputer(mean)+StandardScaler → {LGBM, XGB, CatBoost} × random_state variants → weighted average (hand-set weights
  summing to 1) → power blend of the submission → rank-averaging available as alternative
  - The blend weights were chosen by trial: "I first gave the weights randomly and tracked the output accuracy" — weight
    search without an OOF-fitted meta, so `FLAT` per ruling 1.
  - He also endorses rank averaging ("Rank the outputs first, average the ranks, scale the averaged ranks between 0-1") as a
    technique in his reply, but never says it was in the submitted blend.
  - "Powered submission" (power blending of probabilities) is the one thing he attributes to community knowledge
    (@edrickkesuma-style power blending is named on 9th's page, not his; his own text just says "powered submission").
- **Setup:** no split sizes, no slot count, no row/col counts. Fold schedule: K=2 for fast local iteration, then K=10 —
  "K=10 gave me the best result".
- Structural quirk he works around: the dataset is too large for Kaggle GPU quota, so most work moved to CPU + cloud
  background runs.
- **Features:** no engineered feature named. Preprocessing only:
  - `SimpleImputer` in a sklearn pipeline; `mean` strategy beat `median`.
  - `StandardScaler` on the imputed values.
  - He saved the processed result as a Kaggle dataset so he did not have to reprocess it every run.
  - Missing-value handling note: "LGBM and XGBoost boosting algorithms can handle missing values whereas CATBoost algorithm
    gives errors."
- **Models:** LightGBM, XGBoost, CatBoost; multiple copies of the same algorithm with different random-state numbers inside
  the ensemble.
- Optuna: not claimed by him (he only mentions tracking hyperparameters with neptune.ai).
- Fold types tried: `StratifiedKFold`, plain `KFold`, repeated K-fold — "Stratified k fold gave me the best results."
- Hyperparameters, seeds, member counts: not stated.
- **CV:** stratified K-fold, K=2 for screening then K=10 for the real runs (the K=2 gate: "If I saw the CV score is good enough,
  then I had chosen a large number of fold"). No CV value published, so no CV-vs-LB gap.
- **Ensembling:** weighted average with hand-set weights summing to 1 (heavier weight on the better-performing model), plus a
  power transform of the blended output; rank averaging described as an alternative he points readers to.
- **Post-processing:** "powered submission" — predictions passed through a power transform before submission, exponent not stated.
- **Gains:** no numbers on the page. Stated-positive steps only: K=10 over other fold counts; stratified over plain/repeated
  K-fold; `mean` over `median` imputation; ensembling LGBM+XGB+CatBoost "gave good accuracy"; decreasing memory usage
  "came out very handy"; saving the processed data as a dataset "saved a lot of time".
- **Failed:** `DataTable` library for the large file: "it didn't come out so efficient for me" — and it silently recoded binary
  columns to `True`/`False` when converted back with `to_pandas()`, forcing a manual re-conversion to binary.
- Kaggle GPU quota: ran out twice, forcing the CPU path.
- Colab GPU LightGBM: `LightGBMError: GPU Tree Learner was not enabled in this build. Please recompile with CMake option
  -DUSE_GPU=1` — his from-source recompile block fixed it; the simpler fix suggested by 2nd place in comments did not work
  for him ("I tried that part. But unfortunately, it didn't come out well for me. I still got the error.").
- Median imputation: worse than mean.
- Plain KFold and repeated K-fold: worse than stratified.
- **Comments:** 7 comments (counted: Old Monk, author, Vasilis Konstantakos, author, Adam Wurdits, author, Adam Wurdits).
  Technical content, all outside the body:
  - Vasilis Konstantakos (2nd) gives the shorter Colab fix: `!pip uninstall lightgbm -y` then
    `!pip install lightgbm --install-option=--gpu` — "It seemed to work for me"; author reports it failed on his run.
  - Author's full weight-selection recipe to Adam Wurdits (251st), including the rank-averaging procedure and six resource
    URLs (all copied into Artifacts).
  - Adam Wurdits' own dead end: he built an Optuna objective from an old Titanic notebook but "couldn't manage the weights to
    add up to one", and after reading the resources reports "Two of these notebooks seem to use weights that are entirely
    subjective and the third adjusts the weights based on intuition."
  - Old Monk (24th) and the author exchange congratulations only.
- **Compute:** Kaggle GPU quota exhausted twice; Google Colab tried and LGBM-GPU build fixed by recompiling from source;
  CPU-only runs judged too slow, solved by screening at K=2 and pushing full K=10 runs to the cloud with "save version".
  Wall-clock totals not stated.
- **Artifacts:**
  - https://github.com/Microsoft/LightGBM (the repo in his recompile block)
  - https://neptune.ai/
  - https://towardsdatascience.com/track-and-organize-your-ml-projects-e44e6c7c3f9d
  - https://www.kaggle.com/shaz13/magic-of-weighted-average-rank-0-80
  - https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/205564
  - https://www.kaggle.com/mmotoki/generalized-weighted-mean
  - https://www.kaggle.com/mlanhenke/tps-09-simple-basic-stacking-lgbm-catb-xgb
  - https://www.kaggle.com/yus002/blending-tool-tps-aug-2021
  - https://www.kaggle.com/aayush26/tps-aug-2021-simple-weighted-ensemble
  - https://www.kaggle.com/saurabhbagchi · https://www.kaggle.com/vkonstantakos · https://www.kaggle.com/adamwurdits ·
    https://www.kaggle.com/towhidultonmoy
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/writeups/towhidul-tonmoy-45th-place-what-i-have-learnt-thro (citation line)
  - page tags: LightGBM, XGBoost
- **Lesson:** In a Playground, a hand-weighted GBM blend plus K=10 stratified folds and a memory-reduced pipeline gets you to
  the top 5% of ~1,000+ teams without a single published hyperparameter — which is also why it cannot be replicated exactly.

## TPSSEP21 — consensus recipe
- **Architecture distribution:** 2nd `STACKN` (115 L0 → ElasticNet → ElasticNet fed its own output) · 6th `FLAT`
  (seed-voted LGBM + blending notebooks with a LogisticRegression whose fit data is not stated) · 7th `FLAT`
  (LGBM/CatBoost/HistGB/XGB + a VotingClassifier member, blended) · 9th `STACKN` (34 L0 → LinearRegression →
  LinearRegression, then a flat blend on top) · 10th `STACK2` (17 L0 → LogisticRegression) · 14th `FLAT` (three seed-voted
  branches + [0.7, 0.3] hand weights) · 24th `STACK2` (XGB/CB/LGBM 10-fold OOF → LogisticRegression) · 45th `FLAT`
  (seed-replicated GBM trio, hand weights + power blend).
  - **winner topology: `STACKN`** — 2nd place is the deepest stack on the board and no entrant above 9th used a flat blend.
  - Split is even: 4 stackers (2nd, 9th, 10th, 24th) vs 4 flat blends (6th, 7th, 14th, 45th); the stackers hold 2nd plus 3 of
    ranks 9-24 (9th, 10th, 24th), the flat blends 6th, 7th, 14th and 45th. No entry is `SINGLE` or `SEED` alone, and 6th, 14th
    and 45th all contain `SEED`-style same-architecture voting inside a member (ruling 4); 7th's VotingClassifier mixes three
    different families, not seeds.
  - Depth did not correlate with rank once a stack existed: the two `STACKN` entries are 2nd and 9th, the two `STACK2` are
    10th and 24th, and the strongest flat blend (14th, 0.81868 public) sits within 0.00007 public of 6th's blend (0.81875).
- **New architectures at the board:** exactly two, and neither is the submitted winner.
  - `TabNet` (2nd, TPSSEP21-02) as a level-0 member, "with or without pre-training", self-rated as only "acceptable
    performance"; the 2nd-place submission is GBM outputs through two ElasticNets, not TabNet.
  - `MLP+embedding` (10th, TPSSEP21-10) as 6 of 17 level-0 members — embedding layers over 96 binned inputs plus an
    un-binned input, `swish` activation, all derived from Lukas Borecki's `tps-09-nn`; never competitive alone, kept for
    decorrelation inside the stack.
  - Every other entry (`6th`, `7th`, `9th`, `14th`, `24th`, `45th`) is `novel=none`: GBMs with a linear/elastic-net meta or a
    hand-weighted blend. Also `pyGAM` GAMs at 2nd (a statistical additive model kept for diversity, "not as accurate as the
    XGB, CB, and LGB models"), which is not a neural architecture.
  - Read: this board predates the tabular-NN wave — the 2021 meta layer is LogisticRegression / LinearRegression / ElasticNet,
    never a NN or a GBM.
- **Agreed on (5 of 8 — 02, 07, 10, 24, 45):** the full LightGBM + XGBoost + CatBoost trio as the model zoo; 07 adds
  HistGradientBoosting. Partial zoos: 09 names XGBoost + LightGBM, 14 names LightGBM + CatBoost, 06 names LightGBM only and
  leaves the rest of his blend unlabelled. LightGBM is named by all 8; XGBoost by 6 (02, 07, 09, 10, 24, 45); CatBoost by 6
  (02, 07, 10, 14, 24, 45).
- **Agreed on (4 of 8 — 02, 07, 09, 10):** missingness-derived features are the core primitive of this dataset —
  2nd builds per-row missing counts alongside std/min/max/average; 7th escaped `0.5xx-0.79x` only by capturing NaNs before
  imputing; 9th's per-row NaN count is what pushed him "above a sound 0.8" and he then target-encoded by NaN count; 10th's
  only engineered feature is "the sum of NA's in the row". 24th and 45th never mention NaN features; 45th handles missingness
  by imputation only; 6th and 14th publish no features at all.
- **Agreed on (2 of 8 — 24, 45):** a high stratified fold count as the committed setting, paid for in wall-clock — 24th raised
  to 10 splits "to achieve better validation and more robust model"; 45th screens at K=2 then commits K=10 ("K=10 gave me the
  best result") and picks StratifiedKFold over plain and repeated K-fold. Corroborated from outside: 2nd place, commenting on
  24th's page, "Increasing the number of splits was better but at the cost of time!". Fold counts are not stated at all by 02,
  06, 07, 10 and 14; 9th's only fold number is the 5-fold inside his Optuna runs.
- **Agreed on (4 of 8 — 02, 09, 10, 24):** Optuna as the tuner (2nd on XGB/LGBM/CatBoost/HistGB; 9th on a 20% sample; 10th as
  his new tool replacing Bayesian optimization; 24th as the thing he learned from mlanhenke's notebooks). 45th states no tuner,
  only neptune.ai tracking.
- **Agreed on (4 of 8 — 06, 09, 14, 45):** same-architecture seed replication (VotingClassifier over different `random_state`)
  is the cheapest known gain, and it is 14th's invention: 6th credits him for 0.81851 public on one LGBM; 9th ran 7 seeds per
  recipe; 45th "use the same algorithm for multiple types with different random state numbers"; 14th measured it at
  +0.00037 (LGBM) and +0.00065 (CatBoost) public.
- **Agreed on (4 of 4 stackers — 02, 09, 10, 24):** the meta-learner is LINEAR — ElasticNet ×2 (2nd), LinearRegression ×2
  (9th), LogisticRegression (10th), LogisticRegression (24th) — and both entries that fitted a GBM as meta report it failing
  (10th: XGBoost meta "wasn't performing well"; 9th: "xgboost or lightgbm classifiers as meta model ... and using Optuna to
  tune them at L1" listed as not helping).
- **Agreed on (8 of 8 — 02, 06, 07, 09, 10, 14, 24, 45):** every entrant builds on named community material rather than from
  scratch. Public *submissions/predictions* consumed as ensemble members: 09 (@mlanhenke's submission in the final blend),
  14 (@mlanhenke's submission at the 0.81854 step), 07 (publishes his own prediction dataset outward). Public *notebook
  recipes* consumed as models: 02 (mlanhenke, realtimshady, tunguz, lucamassaron, firefliesqn), 06 (seven named handles +
  martynov's voting classifier), 07 (realtimshady, mlanhenke, tunguz, shenurisumanasekara, dmitryuarov), 09 (hiro5299834,
  ivankontic, mlanhenke, realtimshady, vishwas21, manabendrarout), 10 (lukaszborecki's Keras kernel, mlanhenke's `xgb1`,
  vkonstantakos's stacking structure), 24 (mlanhenke), 45 (from his comment reply: mlanhenke's `tps-09-simple-basic-stacking`
  notebook, the shaz13 / mmotoki / yus002 / aayush26 weighted-average and blending-tool notebooks, and the ranzcr-clip
  discussion thread 205564).
- **Agreed on (4 of 8 — 02, 07, 24, 45):** hitting a hard compute limit is normal and it shapes the architecture — 2nd needed a
  ~25 GB-RAM machine for the GAM notebook after all-feature GAMs died on memory, 7th dropped XGBoost from his voting ensemble
  after 9 unfinished hours plus exhausted GPU quota, 24th ran his 10 folds overnight and submitted the next morning, 45th ran
  out of Kaggle GPU twice and recompiled LightGBM for Colab.
- **Agreed on (2 of 8 — 02, 09):** pseudo labeling tried and rejected for the submission (2nd: improved earlier submissions but
  "the final submission did not use any pseudo labeling"; 9th lists it under "What didn't help my stacking score").
- **Divergences:** depth of stack vs width of blend: 2nd (115 members, 3 stages, 2nd place) and 9th (34 members, 3 stages,
  9th) both say the win came from the trained meta; 14th and 45th say the win came from hand-set weights on seed-voted clones
  and never fit a meta at all. The measurable gap between those philosophies is tiny: 14th's flat hand-weighted blend scored
  0.81868 public and 0.81752 private, while 6th's voting-plus-blend scored 0.81875 public (private never published) — 0.00007
  apart on the board that is published twice. The two stackers who publish anything publish almost nothing: 10th never states
  the score of the submission he placed with, only that his best-ever private 0.81756 was rejected by his own CV-based slot
  rule, 2nd publishes no AUC at all, and 9th only the inequality "above a sound 0.8".
- Divergence on FE: 2nd (k-means/outlier/t-SNE/UMAP + manual binarized categoricals) and 9th (NaN-count target encoding) go
  beyond missingness; 10th deliberately stays at one engineered feature, 7th publishes none, 24th and 45th publish none, and
  14th publishes none — the top of the board did not require feature engineering, only the NaN signal.
- Divergence on public-vs-private trust: 14th (public 0.81868 -> private 0.81752, -0.00116) and 9th ("the resulting rise in the
  public LB is more pronounced than the rise in the private LB") both saw seed-voting gains evaporate privately, while 24th
  gained ~80 places privately after being shut out of the public top 100, and 10th chose his private slot by CV and lost the
  0.81756 he had available.
- **Highest-leverage single trick:** the per-row NaN count / NaN indicator (9th: above 0.8 AUC from it alone; 10th: his only
  engineered feature; 7th: the difference between `0.79x` and a competitive model; 2nd: first bullet of preprocessing),
  followed closely by 14th's one-model voting, which is the only trick in this file with published before/after numbers on both
  boards (+0.00037 public / +0.00026 private on LGBM, +0.00065 public / +0.00064 private on CatBoost).
- **Nothing worked:** a GBM as the meta-learner — XGBoost as stacker (10th), XGBoost or LightGBM as meta plus Optuna tuning of
  them (9th).
- **Nothing worked:** adding individually weak models purely for diversity — RandomForestClassifier and LogisticRegression at
  level 0 (9th), pyGAM GAMs at level 0 (2nd, kept for diversity but weaker than every GBM).
- **Nothing worked:** further Optuna searches once a decent config existed: "led to nowhere and gave no improvement on the
  public LB score" (9th).
- **Nothing worked:** imputation that ignores the NaN signal — `SimpleImputer` without capturing missingness (7th, `0.5xx-0.79x`);
  `median` where `mean` worked (45th).
- **Nothing worked:** pseudo labeling on this dataset for the final submission (2nd, 9th).
- **Nothing worked:** the large-file convenience stack — `DataTable` silently turning 1/0 into True/False (45th); SVM at this
  data size (10th: `LinearSVC(dual=False, penalty='l1', max_iter=1000)` + isotonic calibration, CPU, never converged, public
  0.80104, discarded); all-feature GAMs that die on memory (2nd); XGBoost inside a GPU VotingClassifier (7th, 9 hours,
  unfinished); pip-installed Colab GPU LightGBM without rebuilding (45th, still the same `GPU Tree Learner` error).
