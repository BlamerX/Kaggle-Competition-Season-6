## TPSJAN22 — Tabular Playground Series - Jan 2022
- Task: Tabular (Time Series / Forecasting regression) | Metric: SMAPE | Problem: Practice forecasting
- Kaggle display title: "Tabular Playground Series - Jan 2022" (page header subtitle: "Practice your ML skills on this approachable dataset!")
- Competition: https://www.kaggle.com/c/tabular-playground-series-jan-2022
- Writeups covered: 4 of 4
- Score ladder (SMAPE, LOWER IS BETTER; every delta in this file is signed so that negative = improvement):
  1st public `4.11991` (private never published; that public would rank 306th) · 5th `4.13522/4.66955`, train SMAPE `4.29899` ·
  16th `not stated/not stated` (author publishes no score, only "jumped 277 places", "top 2%") · 40th `not stated/not stated`.
- Structural fact that dominates all four entries: the public LB is the first quarter of 2019, which contains none of the
  interesting holidays, so public order and private order are unrelated.

### TPSJAN22-01 · 1st · AmbrosM (AmbrosM) · LB 4.11991/not stated · CV not stated (no number published)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304355
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/304355` through r.jina.ai with `X-Return-Format: markdown`;
  16,081 bytes, body plus a 42-comment tree)
- **TL;DR:** One Ridge regression on a log-transformed target, with every feature hand-built from residual analysis; he chose the
  submission from `GroupKFold`-by-year CV while the public LB pointed the other way.
- The winning notebook's public score (4.11991) would have ranked 306th publicly; it won privately.
- **Architecture:** SINGLE · stages=1 · l1=1 (Ridge regression) · l2=none · novel=none · mod=none ·
  topo=hand-built features (holiday flags with per-country lengths + product-specific Fourier terms + OECD consumer confidence) →
  Ridge(log target, ColumnTransformer of several MinMaxScalers) → fixed KaggleRama/KaggleMart ratio conversion
  - Fitted estimator is present, so rule 11's `no fitted estimator` wording does NOT apply: Ridge coefficients are learned by
    least squares on the log target; only the store-ratio step is deterministic arithmetic.
  - One model per submission slot, not a blend — he submitted "the two notebooks with the best cv" as his two entries and the
    winner was one of them; there is no meta-learner and no averaging of different families.
  - Global vs per-series: the page never says either. The feature shape (country/product offsets, per-holiday flags, stickers
    with no Fourier term inside one Ridge) is consistent with ONE global fit over the country x store x product panel, so this entry
    does not claim per-series models.
- **Setup:** 3 countries, 2 stores (KaggleRama, KaggleMart) and 3 products are the panel this January TPS is built on; the page itself
  names only Norway ("The Easter holiday in Norway differs from the Easter holiday in the other two countries"), the two stores, and
  the sticker series — the other country/product names are `source silent` here. Exact row/column counts not stated.
- 2 submission slots used for the two best-CV notebooks.
- Structural quirk exploited: public LB = Q1 2019 only, so Easter, Midsummer Day, National Day and Christmas effects are
  invisible publicly — "the public leaderboard gives no information at all about the quality of a model's holiday features".
- **Features:** all four changes are relative to his earlier linear model, and all were "found by a detailed analysis of the residuals":
  - Fourier coefficients re-selected; the sticker series gets NO Fourier coefficients at all, so "the prediction for the stickers
    is constant over the whole year".
  - Small changes in the length of the holiday windows.
  - The Easter holiday in Norway is coded as its own feature, separate from the other two countries.
  - External data: the OECD consumer confidence index added as a feature (dataset `ambrosm/oecd-consumer-confidence-index`),
    as suggested in discussion 302694.
  - Target is log-transformed inside the Ridge pipeline.
  - Per-feature regularization achieved by scaling: "I did not simply use a `StandardScaler`, but a `ColumnTransformer` with
    several `MinMaxScaler`s".
  - Number of Fourier terms, holiday lengths, alpha, exact column names: not stated.
- **Models:** sklearn Ridge regression, log-transformed target; regularization controlled by the single parameter `alpha`
  (value not stated).
- No GBM and no NN in the winning path — a deliberate choice with a stated reason (see Failed).
- Notebooks named: `tpsjan22-10-advanced-linear-model-with-cci` (final) and `tpsjan22-03-linear-model` (earlier).
- Seeds, fold counts, library versions: not stated.
- **CV:** `GroupKFold` with the years as groups.
- SMAPE was reported twice per split: January-March separately from the rest of the year, and he "consistently optimized my model
  for the latter".
- CV-vs-LB: the selected notebook sat at public 4.11991 = public rank 306 while CV said it was the best → CV and public disagreed
  by roughly 290 ranks; the author calls the final choice "quite some courage".
- Author's trust verdict: trust CV, ignore public ("one could practice ignoring the public leaderboard").
- No CV number, fold count, repeat count or seed is published.
- **Ensembling:** none — single model, no weights, no meta-learner, no community prediction files.
- **Post-processing:** one deterministic conversion instead of a fitted component: "The ratio between KaggleRama and KaggleMart
  sales is always the same and does not depend on any other features. A direct calculation is more accurate than linear regression."
- No SMAPE bias/level correction, no rounding published (rounding was raised by 5th as an idea, not by the winner).
- **Gains:** (-0.00047 SMAPE public, measured by a commenter, not by the author) aldparis (11th) merged date coefficients that
  take identical values in his Ridge ("some date coefficients have the same values ... we can gather some dates", e.g. the June
  Wednesday): public 4.11944 vs the winner's 4.11991, with "better CV score ... and better private score 4.58778".
- Unquantified, listed by the author as the four feature changes above (sticker Fourier removal, holiday lengths, Norwegian Easter,
  + OECD CCI), each from residual analysis.
- Unquantified but decisive: selecting the submission on rest-of-year CV instead of public LB (public rank 306 → 1st).
- **Failed:** the author publishes no failed feature or model of his own (checked across body and comments).
- Gradient boosting is rejected on argument, not on a measured score: it removes the FE burden because "decision trees determine
  automatically which countries and products a holiday affects", but "if you use gradient boosting and don't engineer the features
  yourself, you give up control" — trees find a 10-day Easter to be nine or eleven days, and "if you tune the hyperparameters so
  that the holiday has exactly ten days, the model will overfit somewhere else".
- The public LB is declared useless for everything except checking the yearly GDP trend.
- The winner's private score is not in the record, so the size of his shake-up cannot be quoted.
- **Comments:** 42 comments (11 of them appreciation-only, per the page header). Topic-author replies appear twice.
- To akmalmir (147th), confirming "linear models sometimes could beat trees and nn": "I think your conclusion is correct."
- To inversion (Kaggle staff): the competition demonstrated "the importance of cross-validation" and that "the complex machinery
  of gradient boosting and deep learning sometimes is overkill".
- aldparis (11th): his own final solution was "a blend of 3 models (Lasso, Ridge, HuberRegressor) with almost your original FE";
  he was "the first to reach under 4.10 in Public LB 10 days after the beginning", and credits AmbrosM's notebooks; plus the
  coefficient-merging improvement quoted in Gains with a public 4.11944 / private 4.58778.
- Mohamed Ayoub Chettouh (359th): the mechanism for why linear won — "Linear Models extrapolate very well ... If the data the model
  will be tested on is from a different 'Set' ... a well-made linear model should outdo trees and nn"; uniform test sampling would
  favour trees/NN "at least in theory".
- Kartushov Danil (256th): "i had top10 soltuion, but my GB was overfitter ... LB doesnt show how our model is correct aproximating data".
- John Mitchell (40th in this competition = the TPSJAN22-40 author) and Remek Kinas (41st), adaubas (11th), Peleg Shilo (100th),
  Andrew Schleiss (101st) all congratulate; nothing further technical.
- tand (783rd) questions the legitimacy of `GroupKFold` by year on an uptrending series (training on 2016-2018 to validate on 2015,
  "At least, I think if you use lag-feature, this CV is not valid"). The topic author does not answer this question on this page.
- **Compute:** not stated (no wall-clock, hardware or RAM).
- **Artifacts:** (cited by the author, verbatim, not fetched)
  - https://www.kaggle.com/ambrosm/tpsjan22-10-advanced-linear-model-with-cci (final winning notebook)
  - https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model (earlier linear model)
  - https://www.kaggle.com/ambrosm/oecd-consumer-confidence-index (external dataset)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/302694 (where the CCI idea came from)
  - cited inside comments: https://www.kaggle.com/adaubas/tpsjan22-advanced-linear-model-with-cci (11th's coefficient-merged variant)
  - page/author links: https://www.kaggle.com/ambrosm (author profile),
    https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304355 (this page)
  - commenter profile links on this page (navigation only, no content): https://www.kaggle.com/inversion ,
    https://www.kaggle.com/jbomitchell , https://www.kaggle.com/vad13irt , https://www.kaggle.com/sergiosaharovskiy ,
    https://www.kaggle.com/akmalmir , https://www.kaggle.com/mohamedayoubchettouh , https://www.kaggle.com/alincijov ,
    https://www.kaggle.com/codewithgreen , https://www.kaggle.com/aldargarmaev , https://www.kaggle.com/finitearth ,
    https://www.kaggle.com/clementtassart , https://www.kaggle.com/remekkinas , https://www.kaggle.com/samarthvavadiya ,
    https://www.kaggle.com/viji1609 , https://www.kaggle.com/romainbdt , https://www.kaggle.com/sohailshaik272 ,
    https://www.kaggle.com/taranmarley , https://www.kaggle.com/davidedwards1 , https://www.kaggle.com/adaubas ,
    https://www.kaggle.com/slythe , https://www.kaggle.com/kartushovdanil , https://www.kaggle.com/pelegshilo ,
    https://www.kaggle.com/waldemar , https://www.kaggle.com/ksrajavel , https://www.kaggle.com/nebipeker ,
    https://www.kaggle.com/ant3ng , https://www.kaggle.com/paulaehab
- **Lesson:** When the public LB cannot see the effect you are modelling, the whole competition is won by CV design (group by year,
  score the un-scorable months separately) — and a 300th-place public score is compatible with 1st place.

### TPSJAN22-05 · 5th · 🐢 Jun Koda (junkoda) · LB 4.13522/4.66955 · CV not stated (train SMAPE 4.29899)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304369
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/304369`; 8,883 bytes, body + 7 comments)
- **TL;DR:** Physics-style parsimony: one linear regression with 29 coefficients + 1 bias on log(num_sold/GDP), where all 20 holiday
  parameters are ONE shared day-offset profile.
- Public 4.13522 → private 4.66955 (+0.53433 worse): he never touched the trend and never used CV.
- **Architecture:** SINGLE · stages=1 · l1=1 (linear regression, 29 coefficients + 1 bias) · l2=none · novel=none · mod=none ·
  topo=log(num_sold/GDP) + [7 constant offsets + 2 annual Fourier amplitudes + 10 standard-holiday day flags + 10 Christmas day flags]
  → ordinary least squares (MSE) → forecast, no extrapolation into 2019
  - Fitted estimator is present, so `no fitted estimator` does not apply: "Just minimizes the mean squared error of
    log(num_sold/GDP), i.e., standard linear regression".
  - The parameter count closes arithmetically: 7 (weekday/store/country/product offsets) + 2 (mug cosine, hat sine) + 20 (holiday
    day flags) = 29, matching the author's own "29 coefficients + 1 bias".
  - GLOBAL by construction and explicit: country, store, product and weekday enter as constant log-offsets in ONE fit, i.e. the
    3x2x3 series share a model; only the holiday profile is pooled ("a common Gaussian shape" for all holidays).
- **Setup:** 29 coefficients + 1 bias for a 3-country x 2-store x 3-product panel; all years treated equally; no extrapolation of the
  trend into 2019. Rows/cols and slot count not stated.
- Structural quirk exploited: the holiday response is a shift-invariant pulse — the same shape for every holiday, only shifted in time.
- **Features:**
  - [1] Constant factors as log(num_sold) offsets: country, store, product and weekday (Friday, weekend) — "7 parameters: Friday,
    weekend, Norway, Sweden, Hat, Sticker, Rama", explicitly "as in AmbrosM's great notebook".
  - [2] Pure annual modulations, one amplitude each: cosine for Mug, sine for Hat, none for Sticker; "I don't see phase shift or
    higher Fourier modes" (2 parameters).
  - [3] Holiday boosts as binary flags "today is n days after the holiday" for 0 <= n < 10: 10 parameters for a standard holiday,
    10 more for Christmas, which shares the shape but has "much larger height"; the flag representation "can handle overlapping
    holidays, which change every year due to fixed vs non-fixed dates".
  - Target scaling: divides by GDP, then models the log — GDP taken from Carl McBride Ellis's discussion thread.
  - Encodings, GP features, dropped columns: not stated (the model is the whole feature list).
- **Models:** ordinary linear regression, 29 coefficients + 1 bias, MSE on log(num_sold/GDP) (solver, library, regularization: not stated).
- Alternative fit he ran and reports as equal: a nonlinear Gaussian holiday fit — shift ~0.45 days, amplitude ~0.15 in
  log(num_sold), width sigma ~3 days; "Nonlinear fitting with Gaussian works equally well"; the Gaussian has smaller statistical
  error because it has fewer parameters "but can underfit; I did not do detailed comparison".
- Christmas: "slightly wider but might have the same shift and width by choosing more than 1 day for Christmas, but I did not try".
- **CV:** none — the only published internal number is "Train SMAPE 4.29899"; splitter, folds, repeats, seeds not stated.
- CV-vs-LB: no CV exists to compare; public 4.13522 vs private 4.66955 = +0.53433 SMAPE (much worse privately).
- Author's trust verdict: he did not build a validation scheme; he instead used parsimony as the defence — "The model has a small
  number of parameters and therefore avoids overfitting".
- **Ensembling:** none (single model, no averaging, no shared predictions).
- **Post-processing:** none applied — and he names the two he should have done:
  - "Rounding the prediction to integer" (with AmbrosM's figure as the evidence, discussion 301249).
  - "Since the error seems Gaussian in log(num_sold), using mean squared error is reasonable, but prediction value should be tuned
    for SMAPE" — i.e. the SMAPE bias/level correction on the log scale was left on the table.
- No 2019 trend extrapolation, deliberately.
- **Gains:** no delta is published for any step; what is published are fit values, not gains: common holiday shape, peak +4.5 days
  after the holiday, shift ~0.45 d, amplitude ~0.15, sigma ~3 d, and the shared-Christmas-height exception.
- Structural gain, unquantified but stated: replacing per-holiday parameters with one common day-profile is what kept the model at
  29 parameters (1st's AmbrosM: "I somehow felt that the number of parameters for the holidays should be reduced, but I didn't
  realize that the holidays had a common shape").
- **Failed:** long-term trend: "I did not understand the long-term evolution other than the GDP and gave up a week ago"; and
  "giving up future extrapolation cannot be the best approach".
- External data search: "I looked at some external data, but did not find anything useful other than the GDP and the dates of
  official holidays."
- Public-LB probing of the 2019 linear term: "I thought probing for the linear function for the year 2019 using the public
  leaderboard would be useful but I was lazy doing that."
- Unexplained residual structure left in the record: residuals look piecewise-linear in time, different and discontinuous each
  year; slope possibly common among the 3 countries but offsets different; large error at the beginning of 2015; correlated errors
  lasting 1-2 months with no pattern found.
- Two required post-processing steps and the detailed Gaussian-vs-flag comparison were never done (see Post-processing).
- **Comments:** 7 comments.
- aldparis (11th): "i did almost the same, but by hand (i'm an old school statistician) : your solution is more elegant and more
  efficient (5th place)" — the same shared-holiday idea was reached independently from the statistical side.
- Author's own reply: "I also have physics background, reluctant to add thousands of parameters and features, and it seems that our
  approach for holidays outperformed automatic approach like gradient boosting this time."
- AmbrosM (1st): "This is a great solution! I somehow felt that the number of parameters for the holidays should be reduced, but I
  didn't realize that the holidays had a common shape. Congratulations!"
- Taran Marley (206th): "the analysis from linear regression can reveal the most about the dataset in an interpretable way".
- **Compute:** not stated (no wall-clock; the fit is a 29-coefficient OLS).
- **Artifacts:** (cited by the author, verbatim, not fetched)
  - https://www.kaggle.com/junkoda/holiday-kernel (his code)
  - https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model (the notebook he copies the constant-factor idea from)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298911 (GDP data by Carl McBride Ellis)
  - https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301249 (AmbrosM's rounding figure)
  - https://junkoda.github.io/figs/playg/holiday.png (holiday shape per country)
  - https://junkoda.github.io/figs/playg/residual_country.png (residuals by country)
  - https://junkoda.github.io/figs/playg/piecewise_linear.png (piecewise-linear residuals)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/writeups/jun-koda-minimum-linear-regression-5th-place (citation line)
  - https://www.kaggle.com/junkoda (author profile)
- **Lesson:** Pool the holiday response into one shared day-offset profile across all series instead of per-holiday parameters;
  parsimony is what carried 5th place, and the missing integer rounding plus SMAPE-specific bias correction is why it was 5th and not 1st.

### TPSJAN22-16 · 16th · Samuel Cortinhas (samuelcortinhas) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304413
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/304413`; 6,770 bytes, body + 9 comments + 1 appreciation)
- **TL;DR:** A self-described "hybrid model" whose several members were tuned by grid search and then "ensembled together"; the
  writeup publishes no model families, no scores and no CV.
- Zero prior time-series knowledge ("precisely 0"); most ideas credited to other people's notebooks.
- **Architecture:** FLAT · stages=1 · l1=not stated (an unnumbered set of grid-searched models) · l2=none (grid search tunes each
  member's own parameters; no meta-learner trained on an OOF matrix is described) · novel=none · mod=none ·
  topo=Fourier-order-1 + the rest of the feature set (unnamed on the page; "most of the ideas are from other people") → several
  models, grid-searched per model → ensembled (blend)
  - Rule 1 applies: per-model grid search plus blending is weight/parameter selection, not stacking, so the primary stays FLAT.
  - Tag evidence is the author's own wording — "I added a grid search function to fine tune the parameters of my models and then
    ensembled these together" — which is blend language; nothing on the page says a second model was trained on a first model's
    residuals or forecast, so this is NOT tagged CASCADE.
  - The writeup names no estimator at all: which families (the notebook does), how many members, their hyperparameters, and whether
    one global model or per-series models are used are all `source silent`; the notebook is linked and deliberately not fetched.
- **Setup:** no rows/cols/slot counts stated. The only leaderboard fact given: "I woke up this morning to find that I had jumped
  277 places on the leaderboard and finished in the top 2%" — the same public/private disconnect that 1st exploited.
- **Features:**
  - "Fourier features of order 1" — presented as one of his own ideas that he had not seen others try, justified by the seasonality
    ("If it walks like a sine wave and quacks like a sine wave, then you can treat it as if it were a sine wave", paraphrasing
    teckmengwong).
  - GDP per capita tried, then dropped (see Failed).
  - Everything else: "most of the ideas are from other people", and the exact list is not repeated on the page — the feature
    inventory is only in the linked notebook. Encodings, dropped columns, generated features: not stated.
- **Models:** "hybrid model" of several members; families, libraries, hyperparameters, grid ranges, seeds, fold counts: not stated.
- **CV:** not stated — no splitter, no CV value, no CV-vs-LB discussion.
- **Ensembling:** the members were "ensembled together" after grid search; blend weights, metric used to set them, and member count: not stated.
- **Post-processing:** not stated.
- **Gains:** none quantified. The only ordering published: ordinary GDP is "(slightly) higher correlated to the target feature"
  than GDP per capita.
- **Failed:** GDP per capita — "I was excited when this improved my public LB score but when I looked more into it, ordinary GDP was
  (slightly) higher correlated to the target feature. I ended up not using GDP_PC in the end but it was fun to try."
  The public-LB improvement he initially saw is the same trap the winner built his whole strategy around.
- Nothing else is reported as tried-and-rejected (checked in body and comments).
- **Comments:** 9 comments plus 1 appreciation; the topic author does not reply on this page (counted).
- ARShaik (325th) asks two unanswered questions: whether the consumer confidence index would help, and "Why you opted for grid search
  instead of optuna?" — both open, so no answer exists to mine.
- Teck Meng Wong (182nd): "Congrats. Your GDP per capital will be top5. But nothing come close to ambrosm kernel."
- KimBoSoek, luoluona (1062nd), Ammar Abbasi1040 (1222nd) and Vicente Pallares (184th) thank him for the hybrid/grid-search framing;
  no numbers.
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/samuelcortinhas/tps-jan-22-quick-eda-hybrid-model (his full notebook; not fetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/writeups/samuel-cortinhas-16th-place-hybrid-model (citation line)
  - thanked by handle only (no URLs in the body): https://www.kaggle.com/carlmcbrideellis , https://www.kaggle.com/ambrosm ,
    https://www.kaggle.com/lucamassaron , https://www.kaggle.com/adamwurdits , https://www.kaggle.com/vpallares ,
    https://www.kaggle.com/mfedeli , https://www.kaggle.com/fergusfindley , https://www.kaggle.com/teckmengwong ,
    https://www.kaggle.com/vad13irt , https://www.kaggle.com/remekkinas
- **Lesson:** If you cannot say which models you blended, your writeup is not replicable — the architecture tag here can only be
  derived from the words "ensembled these together", which is exactly why member families belong in the writeup.

### TPSJAN22-40 · 40th · John Mitchell (jbomitchell) · LB not stated/not stated · CV not stated (no CV used)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304353
- **Status:** FETCHED (single attempt, `/c/<comp>/discussion/304353`; 6,686 bytes, body + 2 comments incl. 1 appreciation)
- **TL;DR:** Boltzmann Ensemble: weight every community notebook by `exp(b*(S-x))` where x is that notebook's PUBLIC LB score, and
  average. One adjustable parameter, zero modelling.
- An ensembling rule imported from statistical thermodynamics, not a model of the sales data.
- **Architecture:** FLAT · stages=1 · l1=not stated (a pool of shared Kaggle notebooks; the count is never published) · l2=none
  (weights are a closed-form function of each member's public score; only b is tuned, and no meta-learner is fitted on an OOF
  matrix) · novel=none · mod=public-oof ·
  topo=public notebooks [N] with public scores x_i → w_i = exp(b*(S-x_i)), q = sum(w_i) → weighted average of their predictions
  - Rule 1: tuning one parameter b is weight selection, so FLAT and not STACK2. Rule 11 does not apply either way — this entry
    fits nothing to the target itself (the members are other people's fitted models), so it is not a "no fitted estimator"
    hand-built decomposition, it is a blender over fitted models.
  - Weights come from PUBLIC LB scores, not from OOF — the whole method is a public-LB objective, which is the practice the winner
    of this competition rejected.
  - No per-series/global question arises: the ensemble operates on submitted prediction files only.
- **Setup:** no data facts at all; inputs are other people's notebooks/versions plus their public scores. Rows/cols/slots: not stated.
- Structural quirk exploited: in a Playground "many Kagglers shared excellent notebooks containing models scoring close to the top
  of the leaderboard", so a strong pool is free to take.
- **Features:** none — the method consumes predictions, not columns.
- **Models:** "each model represented in the ensemble"; families and the member count are not stated.
- **CV:** none — the score x in the weight formula IS the public LB score, so there is no local validation at all; the author is
  explicit that the method "works best when the target is a continuous numerical quantity" (SMAPE here, sign-flipped convention if
  higher-is-better).
- **Ensembling:** the entire solution.
  - Member weight `w_i = exp(b*(S - x_i))`, x_i = that member's public LB score.
  - `b` is the "one meaningfully adjustable parameter"; larger b decays weights faster with worsening score; it is "analogous to
    1/kT" and called beta in thermodynamics.
  - `S` is a calibration constant: set it to the best single model's score and the top weight is exactly 1.0 — "convenient, but not essential".
  - `q = sum of the weights` is the partition-function analogue and measures "an effective number of contributing models".
  - Sign convention flips for metrics where higher is better.
- **Post-processing:** none stated.
- **Gains:** none published — no SMAPE, no CV, no comparison table at all.
- **Failed:** the author's own limitation list is the anti-knowledge here:
  - "I found that the best value of b changed significantly during the competition" — a single tuned beta is not stable over time.
  - Membership is binary: "a model is either fully in or fully out".
  - "in this presentation there is no account taken of similarity between models" — penalizing correlated members "would increase
    the number of parameters".
  - "there is no flexibility here to include the sometimes useful possibility of negative weights".
  - Transfer warning: it worked because the notebooks were near the top; "if you plan to use this or any other ensembling approach
    there then you will probably need to be able to include some fairly strong private models of your own".
- **Comments:** 2 comments (1 of them an appreciation comment); no topic-author replies (counted).
- inversion (Kaggle staff): "I love reading about approaches that are off the beaten path."
- Ammar Abbasi1040 (1222nd): "Insightful. Thanks for this." No technical content in the replies.
- **Compute:** not stated; by construction it only reads prediction files and their scores.
- **Artifacts:**
  - https://www.kaggle.com/jbomitchell/boltzmann-ensemble-without-overfitting (the ensemble notebook; not fetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/writeups/john-mitchell-40th-place-boltzmann-ensemble-a-solu (citation line)
  - https://www.kaggle.com/jbomitchell (author profile)
- **Lesson:** A one-parameter blender over public-LB-scored community notebooks reaches 40th with zero modelling, but it inherits
  public-LB overfitting wholesale and structurally cannot express negative weights or member decorrelation.

## TPSJAN22 — consensus recipe
- **Architecture distribution:** 1st `SINGLE` (Ridge on log target, hand-built holiday/Fourier/CCI features, public 4.11991 → 1st) ·
  5th `SINGLE` (OLS, 29 coefficients + 1 bias, on log(num_sold/GDP), 4.13522/4.66955) ·
  16th `FLAT` (grid-searched "hybrid" ensemble; member families unnamed) · 40th `FLAT` (Boltzmann weights over community notebooks,
  one tuned b). Ranks 2nd, 3rd and 4th-15th have no writeup in the index, so the distribution covers 4 of the board.
  - **winner topology: `SINGLE`** — both published head-to-head leaders (1st, 5th) submitted ONE hand-engineered linear model with
    no blending; the two ensemble entries finished 16th and 40th.
  - Global vs per-series, stated explicitly: the two linear winners are GLOBAL — 5th writes it out (country/store/product/weekday
    are shared constant log-offsets, the holiday day-profile is shared across holidays and countries); 1st's feature shape is the
    same global pattern but he never says it. 16th and 40th are silent.
  - Fitted-estimator audit (rule 11): 1st fits Ridge by least squares, 5th fits OLS on log(num_sold/GDP) — so both are
    "coefficient-learned", NOT deterministic decompositions; 16th's blend step fits nothing itself (a grid search tunes members it
    trained from scratch, families unnamed) and 40th fits nothing to the target at all (a one-parameter weight formula over other
    people's fitted models).
- **New architectures at the board:** none — no entry carries a `novel=` token. Nothing beyond stock sklearn was submitted at the
  top two ranks (Ridge, plain linear regression), and the two lower entries used unnamed families and a weighting rule.
  The genuinely non-standard ingredients are not architectures: an external-economics feature (OECD consumer confidence index, 1st),
  a shared Gaussian holiday response (5th), and a statistical-thermodynamics weight function (40th).
- **Agreed on (3 of 4 — TPSJAN22-01, TPSJAN22-05, TPSJAN22-16):** explicit annual-cycle/Fourier terms, with the sticker series given
  no seasonality at all. 1st: sticker rows get "no Fourier coefficients at all" so their prediction is flat across the year;
  5th: cosine amplitude for mug, sine amplitude for hat, "none for the sticker", no phase shift and no higher modes; 16th:
  "Fourier features of order 1". 40th states no features.
- **Agreed on (3 of 4 — TPSJAN22-01, TPSJAN22-05, TPSJAN22-16):** the public LB is not a usable objective, and each publishes or
  quotes the evidence. 1st: public 4.11991 = public rank 306 for the notebook that won, because Q1 2019 contains none of the
  holidays; 16th: "jumped 277 places"; 5th: public 4.13522 → private 4.66955, a +0.53433 SMAPE swing.
- **Agreed on (2 of 4 — TPSJAN22-01, TPSJAN22-05):** national GDP as the macro-level covariate (5th divides the target by it and
  models log(num_sold/GDP); 1st says the public LB "is only good to verify whether the model deals correctly with the yearly GDP"
  and adds the OECD consumer confidence index on top). 16th touched GDP variants but never says GDP is in the final model.
- **Agreed on (2 of 4 — TPSJAN22-01, TPSJAN22-05):** hand-built holiday structure beats machine-learned holiday structure.
  1st gives the control argument (trees find Easter as 9 or 11 days; tuning it to 10 overfits elsewhere); 5th's author reply states
  "our approach for holidays outperformed automatic approach like gradient boosting this time". Corroborated from the comments by
  Kartushov Danil (256th, "my GB was overfitter") and by aldparis (11th, blend of Lasso/Ridge/HuberRegressor "with almost your
  original FE") — commenters, not entries, so they do not raise the count.
- **Divergences:** validation design split the board rather than the model class. 1st ran `GroupKFold` by year and scored Jan-Mar
  separately from the rest of the year; 5th ran no CV at all (train SMAPE 4.29899 only) and still took 5th on 29 parameters;
  16th grid-searched per model with no published CV; 40th substituted public LB scores for validation entirely.
- Score gap on the only common column: 1st public 4.11991 vs 5th public 4.13522 = 0.01531 SMAPE, while 5th's model is ~29
  coefficients and 1st's coefficient count is never published — the top of this board was separated by hand-built holiday detail,
  not by capacity.
- Divergence on extrapolation: 5th deliberately does not extrapolate the trend into 2019 and admits "giving up future extrapolation
  cannot be the best approach"; 1st encodes the trend in GDP/CCI features plus Fourier terms instead.
- Divergence on post-processing: 5th lists integer rounding and an SMAPE-tuned point prediction as mandatory work he did not do;
  1st's only deterministic correction is the KaggleRama/KaggleMart ratio.
- **Highest-leverage single trick:** optimize the SMAPE of the months the public LB cannot see (Jan-Mar split out from the rest of
  the year under `GroupKFold`-by-year) — TPSJAN22-01; the runner-up free win was parsimony on the linear model itself: merge date
  coefficients that already take identical values, worth -0.00047 public and a better private (4.58778) on top of the winning
  notebook — aldparis's comment on TPSJAN22-01.
- **Nothing worked:** the automatic-holiday route (gradient boosting) as judged by 1st's control argument and 5th's stated outcome,
  plus 256th's "my GB was overfitter" — note that no entry publishes a CV number for a GBM, so the rejection is qualitative.
- **Nothing worked:** hunting external data beyond GDP and official holiday dates — 5th "did not find anything useful"; 16th's GDP
  per capita was correlated worse than plain GDP and was dropped; 1st's consumer confidence index is the one external addition that
  survived, and it came from a residual analysis, not from a broad search.
- **Nothing worked:** trusting a public-LB bump — 16th saw GDP_PC "improve my public LB score" and discarded it, and the board shook
  by hundreds of places (1st's winning notebook sat at public rank 306, 16th "jumped 277 places").
- **Nothing worked:** 5th's unexplained residual structure (piecewise-linear per year, discontinuous year boundaries, large 2015
  error, 1-2 month correlated errors) — he published it as unsolved rather than patched it with more features.
