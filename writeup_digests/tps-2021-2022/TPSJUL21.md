## TPSJUL21 — Tabular Playground Series - Jul 2021
- Task: Tabular (Regression, multi-target) | Metric: Mean Columnwise RMSLE | Problem: Practice multivariate regression
- Kaggle display title: "Tabular Playground Series - Jul 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-jul-2021
- Writeups covered: 1 of 1
- Score ladder (mean columnwise RMSLE, LOWER better; `public/private` as reported by the authors): 1st `not stated / not stated` —
  the page publishes no score at all, only a rank movement: averaging her output with a public LightAutoML notebook "improved my
  final rank from 2 to 1".
- The page never names the metric or the target list; the three targets it does name by example are carbon monoxide, benzene and
  nitrogen (her words), and the sentinel for missing values is `-200`.
- Single-entry file: every consensus line below is stated by TPSJUL21-01 alone, and no agreement claim is made.

### TPSJUL21-01 · 1st · Laura Desplans (desplanl) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jul-2021/discussion/256486
- **Status:** FETCHED (single jina `/c/.../discussion/256486` call: body + 24 main-thread comments incl. 1 deleted + the 2 appreciation
  comments = the "26 Comments" header count)
- **TL;DR:** Won on the **data leak**: no new features at all, a two-stage GBT pipeline that only imputes the `-200` gaps in the
  leaked data and retrains on train + leaked + imputed, 5 seeds averaged, then blended with Alexander Ryzhkov's public LightAutoML
  notebook — the blend alone moved her from 2nd to 1st.
- Every engineered variable she tried made the score worse once the leak was in play.
- **Architecture:** CASCADE · stages=2 · l1=5 GBT models (different seeds) averaged, the same 5-model set in each stage ·
  l2=none (final output is a plain average with a public LightAutoML prediction set) · novel=none · mod=public-oof,multitarget ·
  topo=stage-1 GBT x5 seeds (avg) estimates the -200 gaps in the leaked data → stage-2 GBT x5 seeds (avg) retrained on
  train + leaked + imputed → average with LightAutoML notebook output
  - Stage 1's *output* (imputed values) becomes stage 2's *training input*, i.e. data flows forward into a retrain rather than into a
    blender — the taxonomy's `denoise → retrain` case, so `CASCADE` and not `STACK2`/`FLAT`.
  - The submitted prediction is the average of her two-stage output and someone else's public notebook, which is a plain average with
    no fitted meta-learner and no weights published.
  - `multitarget` because she keeps the other targets as predictors of each target ("use benzene and nitrogen to predict carbon
    monoxide"); that is the only feature-side idea she retained.
- **Setup:** rows/cols, split sizes, submission slots: not stated on the page.
- Structural quirk exploited: **leaked data whose targets are known**, plus its missing values encoded as `-200`, which she
  two-stage-estimates; the second stage trains on the longer sample "training data + leaked data + the imputed missing values".
- She also observed that Kaggle's provided targets differ from the leaked targets beyond the `-200` cells: the difference "averaged
  zero over time and it appeared to be correlated with the sensors".
- **Features:** zero new columns: "Weirdly enough, I ended up not including any new variables beyond what was provided (+ the leaked
  data)."
- Tried one at a time and REJECTED because each worsened the score: month dummies, weekday dummies, hour dummies, combinations of the
  weather variables, several interactions, lags.
- Retained: the other targets used as features (benzene + nitrogen → carbon monoxide); "It was also particularly helpful to include the
  targets as features".
- Her reading of why the raw set won: the extra variables "were of course helpful prior to the leakage; post-leakage they didn't seem to
  help".
- Encodings, GP/generated features, dropped columns, feature-source kernels: not stated.
- **Models:** "5 gradient boosted tree models, each with a different seed" — averaged predictions.
- GBM library is never named on the page (a commenter calls them "pure XGB models"); hyperparameters, learning rate, round counts,
  seed values: not stated.
- "I used the same models in each stage; I didn't have time to try using a different model for the second stage."
- Second system in the final blend: LightAutoML (Alexander Ryzhkov's public notebook
  `alexryzhkov/tps-lightautoml-baseline-with-pseudolabels`), used as an outside prediction set, not as her own level-1 member, so
  `AUTOML` is not the primary (ruling 6).
- **CV:** no CV number, splitter, fold count, repeat count or seed protocol is published; the page has no validation section
  (`source silent`).
- CV-vs-LB gap: not computable — no local or leaderboard number appears anywhere on the page or in its 26 comments.
- Trust verdict: none stated; her only validation-flavoured argument is that refusing FE "may have prevented overfitting the private
  dataset and hence may have helped my final rank".
- **Ensembling:** two levels of averaging, both unweighted as far as the page says:
  - average of the 5 differently-seeded GBT predictions (per stage).
  - average of her final output with Ryzhkov's LightAutoML notebook predictions → this step alone moved her from rank 2 to rank 1.
- Shared-OOF usage: yes — a public notebook's predictions are half of the final submission.
- Number of models averaged: 5 GBT seeds per stage + 1 external LightAutoML pipeline.
- Weights of the final average: not stated.
- **Post-processing:** none stated. The `-200` handling is imputation of training data (stage 1), not output post-processing; no
  threshold, calibration, clipping, rank-gauss, rounding or label trick appears on the page.
- **Gains:** rank 2 → rank 1 (the medal) from averaging her output with the public LightAutoML notebook — the only quantified outcome
  on the page, and it is a rank, not an RMSLE delta.
- Qualitative gains she asserts: including the other targets as features was "particularly helpful"; the two-stage estimation of the
  `-200` cells made the longer training sample usable.
- No score delta is attached to any other choice.
- **Failed:** every feature she engineered worsened the score once the leak existed: month / weekday / hour dummies, weather-variable
  combinations, several interactions, lags — "each of them worsened my score".
- Modeling the Kaggle-target vs leaked-target difference (zero-mean, correlated with the sensors): "I didn't manage to improve it
  despite the correlation with the sensors"; those lines are left commented out in her code.
- Using a different model family for stage 2: not attempted ("I didn't have time").
- **Comments:** 26 comments per the header = 24 main-thread blocks (incl. 1 deleted) + 2 appreciation comments; no hyperparameters were
  ever extracted from her in the thread.
  - lilkaskitc reframes her unresolved observation as reverse-engineering the generator: "I suppose what you are investigating is
    indeed to decipher the transformations done by Kaggle on the original dataset, re-engineering", and quotes the competition's own
    line that the dataset "is based on a real dataset, but has synthetic-generated aspects to it ... predicting air pollution in a city
    via various input sensor values (e.g., a time series)".
  - Author's own replies: "I think the code ended up being relatively simple at the end compared to everything else I tried!" and
    "Feel free to use it! Happy to answer any questions."
  - Alexander Ryzhkov (27th in this Competition, the credited baseline author) replies with congratulations and a knowledge-sharing
    note; his notebook is the one blended into the winner.
  - Commenters who only congratulate, with their ranks recorded: Bojan Tunguz (1293rd), Jonas Paluci Barbosa (915th, whose linked post
    she cites), Sharlto Cope (links her solution into his EDA notebook), Solomon Kimunyu (31st), 610/kanamemuto (102nd), Jagunn (208th),
    Ash/katnoria (450th), Devashree Madhugiri (140th), Tekina/aniket007 (465th), ss_sv (807th), Rakesh Jarupula (633rd), Yash Damle
    (no rank badge shown).
  - One comment on the thread has been deleted.
- **Compute:** not stated (no wall-clock, GPU/CPU, RAM, limits or cost anywhere on the page).
- **Artifacts:** (verbatim from the page; not fetched)
  - https://www.kaggle.com/alexryzhkov/tps-lightautoml-baseline-with-pseudolabels (body citation — the notebook blended into the winner)
  - https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738 (body citation — Ryzhkov's April missing-value post)
  - https://www.kaggle.com/c/tabular-playground-series-jul-2021/discussion/256321 (body citation — Jonas Paluci Barbosa's post on
    private-set overfitting)
  - https://www.kaggle.com/competitions/tabular-playground-series-jul-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-jul-2021/writeups/laura-desplans-holy-cow-i-ranked-1st-solution-disc
    (citation line)
  - https://www.kaggle.com/desplanl (author), https://www.kaggle.com/alexryzhkov (credited)
  - comment links: https://www.kaggle.com/lilkaskitc, https://www.kaggle.com/shreyashvaish, https://www.kaggle.com/aniket007,
    https://www.kaggle.com/tunguz, https://www.kaggle.com/jarupula, https://www.kaggle.com/jonaspalucibarbosa,
    https://www.kaggle.com/dwin183287, https://www.kaggle.com/katnoria, https://www.kaggle.com/yashdamle,
    https://www.kaggle.com/kanamemuto, https://www.kaggle.com/jagunn, https://www.kaggle.com/metasummattho,
    https://www.kaggle.com/devsubhash, https://www.kaggle.com/ndegwakimunyu
  - page tags: Time Series Analysis, Tabular, Pollution, Gradient Boosting
  - no notebook or dataset URL of her own is given on the page, even though she refers to "the code" ("I commented some of these lines
    in the code") — checked against the full URL list rendered on this page.
- **Lesson:** When the test labels leak, the marginal value of feature engineering goes to zero and the last medal-moving move is a
  plain average with the best public baseline — so check for a leak before you build anything.

## TPSJUL21 — consensus recipe
- **Architecture distribution:** 1st `CASCADE` (stage-1 GBT×5-seed imputation of the `-200` leaked cells → stage-2 GBT×5-seed retrain
  on the augmented sample → plain average with a public LightAutoML notebook).
  - **winner topology: `CASCADE`** — with one writeup on the board no alternative is observed here, and the file makes no claim that a
    flat blend or a stack would have scored differently.
- **New architectures at the board:** none. `novel=none` on the only page in this file (checked against its full text), and the whole
  model inventory is unnamed gradient-boosted trees plus LightAutoML. The only pseudo-labeling on the board arrives second-hand, inside
  Ryzhkov's public LightAutoML notebook that she averages with — the title of that notebook is the page's only statement about it.
- **Stated by the sole entry (TPSJUL21-01; no cross-writeup agreement is possible in a 1-entry file):** no new features beyond the
  provided columns plus the leaked data; every engineered variable (month/weekday/hour dummies, weather combinations, interactions,
  lags) worsened the score.
- **Stated by the sole entry (TPSJUL21-01):** other targets as features (benzene + nitrogen → carbon monoxide) is the single retained
  feature idea, and seed-averaged GBTs (5 seeds, averaged) are the model.
- **Stated by the sole entry (TPSJUL21-01):** a public community artifact was decisive: averaging with Ryzhkov's LightAutoML notebook
  moved her rank 2 → 1.
- **Divergences:** none observable — a single-entry file; the only internal tension is that her pre-leak FE was "of course helpful"
  while her post-leak FE was worthless, which is a statement about the dataset, not about rank behaviour.
- **Highest-leverage single trick:** average your output with the strongest public baseline (TPSJUL21-01) — a rank-2 to rank-1 swing for
  one line of blending, with no fitted meta-learner.
- **Nothing worked:** feature engineering after the leak (month/weekday/hour dummies, weather-variable combinations, interactions, lags),
  and modeling the zero-mean Kaggle-vs-leaked target difference that correlated with the sensors (TPSJUL21-01, lines left commented out
  in her code).
- Not stated anywhere on this page, and therefore not to be assumed: the GBM library, all hyperparameters, any CV or LB number, the
  splitter, and any wall-clock cost.
