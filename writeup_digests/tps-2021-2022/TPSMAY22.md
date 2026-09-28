## TPSMAY22 — Tabular Playground Series - May 2022
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - May 2022"
- Competition: https://www.kaggle.com/c/tabular-playground-series-may-2022
- Writeups covered: 3 of 3
- Metric/task note: confirmed on all three fetched pages as **binary classification scored by ROC AUC** — the page header
  is "Practice your ML skills on this approachable dataset!", the target is quoted from the description as "whether the
  machine is in state 0 or state 1" (casati8's comment on TPSMAY22-01), and every score on the board is an AUC
  (0.998 range). No entry on these pages mentions clustering or Adjusted Rand Index; the ARI/clustering Playground of
  2022 is TPS Jul 2022 (`TPSJUL22`), a different competition, so nothing in this file is a clustering pipeline.
- Score ladder: 1st `not stated/not stated (rank only)` · 4th `0.99829/0.99825` · 5th `0.99829/0.99824`
- No CV number is published anywhere in this set (checked all 3 pages: 1st publishes no metric, 4th publishes public and
  private only, 5th's scores come from his comment reply, public and private only).

### TPSMAY22-01 · 1st · ambrosm (AmbrosM), team with pourchot (Laurent Pourchot) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2022/discussion/328336
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 11.5 KB; all 20 comments rendered inline)
- **TL;DR:** The Overview hint "feature interactions" was turned into an interaction *graph*, which had exactly two
  connected components; the network was then split into two branches so no interaction crosses components.
- The submitted prediction is a blend of variants of that one two-branch Keras network.
- A LightGBM using the same idea reached LB 0.99778 and was deliberately left out of the blend.
- **Architecture:** SEED · stages=1 · l1=variants of one architecture (count not stated) · l2=none · novel=none ·
  mod=none ·
  topo=numeric f_xx columns + the `ch0`–`ch9`/`unique_characters` columns → xgbfir interaction graph (2 connected
  components) → two-branch Keras net [left 18 cols |
  right 26 cols] → merge → sigmoid → blend of variants of that architecture
  - Data flow: each branch sees only the columns of one connected component of the interaction graph, so a cross-component
    interaction is structurally impossible; the author's stated reason is that "other interactions produce noise".
  - `SEED` not `FLAT`: the author says the submission is "a blend of some variants of the following two-branch network" —
    one architecture, several variants; the second family (LightGBM) was trained but excluded, so no family blending occurs.
  - Not a stack: no meta-learner is mentioned anywhere on the page; the identical notebook otherwise is the author's own
    "Advanced Keras" kernel plus the branch split ("Except for the two-branch architecture, our final implementation
    doesn't differ from the Advanced Keras notebook").
- **Setup:** rows/cols, split sizes, fold count, submission slots: not stated on the page.
- Task structural quirk exploited: the generator's feature interactions form a graph with **two** connected components, so
  the interaction constraint is expressible as an architecture, not as a hyperparameter.
- The branch feature lists contain 10 single-character columns `ch0`…`ch9` plus `unique_characters`, i.e. the string-ish
  input has already been expanded into per-character columns; the page never names the raw source column they come from.
- **Features:** left branch (18 columns, verbatim): `['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19',
  'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28', 'f_30', 'ch7']`.
- Right branch (26 columns, verbatim): `['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
  'f_17', 'f_18', 'f_29', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch8', 'ch9', 'unique_characters',
  'i_02_21', 'i_05_22', 'i_00_01_26']`.
- Named interaction features: `i_02_21`, `i_05_22`, `i_00_01_26` (pair/triple products implied by the naming; formulas not
  written out on the page).
- `unique_characters` is credited by the author to @cabaxiom as the introducer of that feature; its exact definition is not
  written on the page.
- Interaction discovery method: Xgbfir (limexp) run by @wti200 / published as a dataset by @sudalairajkumar; the author
  then "constructed a graph of these interactions" by hand ("on the back of an envelope").
- Encodings beyond the `ch0`–`ch9` character split, dropped columns, GP/generated features, feature-source kernels: not stated.
- **Models:** Keras two-branch network (figure of the architecture in the post); several variants blended, per-variant
  hyperparameters not stated.
- LightGBM with `interaction_constraints=[features_left, features_right]` — reached LB 0.99778, not blended.
- Library versions, seeds, fold counts, layer widths: not stated.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV score is published on the page.
- CV-vs-LB gap: not computable (neither side is published).
- Author's trust verdict: not stated on this page.
- **Ensembling:** blend of variants of the two-branch network; weights, count of members and averaging mechanic not stated.
- Shared-OOF usage: none mentioned (rule 4b — the page names no community prediction set).
- **Post-processing:** not stated. (A commenter notes the submission values exceed 1; the author's teammate replies that
  AUC only cares about ranking, so no rescaling was applied — see Comments.)
- **Gains:** The page publishes no deltas.
- Ordering available from it: the two-branch Keras blend (1st) beat the best decision-tree variant it replaced,
  LightGBM at LB 0.99778, by at least the gap to the top of the board (the winning score itself is not printed).
- **Failed:** LightGBM + `interaction_constraints` scored LB 0.99778 and "was not enough to be included in the blend" —
  the only named dead end in this entry.
- Anti-knowledge: applying the winning idea to a GBM instead of a NN lost the competition for that branch of work.
- **Comments:** 20 comments on the page (counted); technical content, all from replies:
  - Laurent Pourchot (teammate, 1st): "it was a fruitful collaboration, you gave the best ideas and I tuned it"; the
    solution code is published at https://www.kaggle.com/code/pourchot/tpsmay22-keras-test-tuned and "The solution was
    blended in the last days to reach the highest score."
  - casati8 (175th) asked why the submitted targets are greater than 1, since the description says the target is state
    0/1; Pourchot: "the AUC metric is sensitive to the ranking", linking
    https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/ — i.e. raw unclipped outputs were
    submitted on purpose.
  - SRK (@sudalairajkumar, 123rd): confirms the xgbfir dataset is what made the win possible ("Glad that the xgbfir
    dataset helped").
  - Remaining comments are congratulations (DeeperNet 9th, ArturRa 114th, Samuel Cortinhas 187th, Chirag Desai 399th,
    Ravi Shukla 238th and others); no further numbers.
- **Compute:** not stated (no GPU/CPU, wall-clock or limit mentioned).
- **Artifacts:** (verbatim from the body and comments; not fetched)
  - https://i.imgur.com/AKLatCP.png (two-branch network figure)
  - https://i.imgur.com/WKbwEGt.png (feature-interaction graph figure)
  - https://github.com/limexp/xgbfir
  - https://www.kaggle.com/datasets/sudalairajkumar/tps-may22-xgboost-feature-interactions
  - https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras
  - https://www.kaggle.com/code/pourchot/tpsmay22-keras-test-tuned (from Pourchot's comment)
  - https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/ (from Pourchot's comment)
  - https://www.kaggle.com/competitions/33105/images/header
  - https://www.kaggle.com/ambrosm · https://www.kaggle.com/pourchot · https://www.kaggle.com/sudalairajkumar ·
    https://www.kaggle.com/cabaxiom · https://www.kaggle.com/wti200 · https://www.kaggle.com/casati8 ·
    https://www.kaggle.com/cid007 · https://www.kaggle.com/raviista · https://www.kaggle.com/onedatareader ·
    https://www.kaggle.com/adamwurdits · https://www.kaggle.com/anandparthiban ·
    https://www.kaggle.com/maxdiazbattan · https://www.kaggle.com/deepernet · https://www.kaggle.com/arturra ·
    https://www.kaggle.com/samuelcortinhas · https://www.kaggle.com/adaubas · https://www.kaggle.com/manfau ·
    https://www.kaggle.com/roczy1 (commenter/honoree profile links on the page)
- **Lesson:** When the generator documents its interactions, find the interaction graph first — its connected components
  tell you how to wire the network, and a GBM given the same constraint is the cheapest way to prove the NN needs it.

### TPSMAY22-04 · 4th · obougacha (Omar Bougacha) · LB 0.99829/0.99825 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2022/discussion/328441
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 4.4 KB; the discussion id resolves to the writeup
  page `omar-bougacha-4-solution-multi-activation-branches`; both comments rendered)
- **TL;DR:** One Keras net whose parallel branches each use a *different activation function* (swish/selu/relu) to explore
  interaction space; then a straight 50-50 average with a community notebook's submission file to finish 4th.
- **Architecture:** FLAT · stages=1 · l1=2 (1 own multi-activation-branch NN + 1 community submission file) · l2=none ·
  novel=none · mod=public-oof,multi-view ·
  topo=shared Swish trunk → parallel branches [swish | selu | relu] → merge → NN 0.99826/0.99822 → 50-50 average with
  mehrankazeminia's published submission → 0.99829/0.99825
  - The submitted artifact is a 2-member weighted average (weights stated: 50-50), so it is `FLAT`, not `SINGLE`: his own
    net alone was 0.99826/0.99822 and the blend moved it to 0.99829/0.99825.
  - `multi-view` here means one network whose branches apply different activation functions to the same columns, the
    author's own rationale being that this "allows to explore different possibilities of feature interaction".
  - `public-oof` in the loose sense used by this competition: the second member is a *published submission* (test
    predictions) from someone else's kernel, not an OOF matrix; no meta-learner is fitted on it, so no level 2 exists.
- **Setup:** rows/cols, split sizes, fold count not stated; submission slots: 1 known final submission (the blend), count
  not stated.
- Structural quirk exploited: the same feature-interaction structure the winner used; his support for the first days was
  AmbrosM's Advanced Keras notebook "especially when he added the interactions".
- **Features:** none named by this author — no columns, encodings, or formulas appear in the post.
- He inherits whatever AmbrosM's kernel builds (stated as the starting point of his code); the per-character columns and
  interaction columns, dropped columns: not stated here.
- **Models:** Keras multi-branch network; common layers use Swish; parallel branches, all layers of a branch sharing one
  activation: swish, selu, relu.
- Best single model (that net): public 0.99826 / private 0.99822.
- Hyperparameters (depth, width, folds, seeds, batch, epochs): not stated. Library version: not stated.
- Second member is not his model: the submission file of @mehrankazeminia's "tpsmay22-auc-ensembling" notebook.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV score published (only public/private LB).
- CV-vs-LB gap: not computable.
- Author's trust verdict: not stated; he instead describes disengaging early (see Failed).
- **Ensembling:** single mechanic, stated exactly: "Final solution is blending my best model to the submission file from
  @mehrankazeminia notebook with weights 50-50."
- Result of that blend: public 0.99829 / private 0.99825 versus 0.99826 / 0.99822 alone (+0.00003 public, +0.00003 private).
- Number of models actually averaged: 2 submission files (the internal content of the community file is not described here).
- **Post-processing:** none stated (no threshold, calibration, clipping or rank transform).
- **Gains:** (+0.00003 public, +0.00003 private) the 50-50 blend with the community submission file over his own best net.
- (unquantified) using different activation functions per branch — his own claim is "I guess I was right about that", with
  no ablation numbers published.
- **Failed:** No failed model, feature, or blender is named on this page (counted: the body names only the branch design and
  the final blend).
- Named non-technical dead end, verbatim: "I lost interest in the competition very quickly since scores where all around
  0.998, I couldn't see the interest in keeping in the race." — i.e. he stopped tuning; the +0.00003 he did take came from
  borrowing, not from modeling.
- **Comments:** nothing technical (counted: 2 comments — HATIM_CHIFA "well done friend that was so helpfull" and What
  (@tttrrraaahhh, 5th in this competition) "You've done a great job. Congratulations!"; no author reply).
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras
  - https://www.kaggle.com/code/mehrankazeminia/tpsmay22-auc-ensembling
  - https://i.postimg.cc/gkX2QXhn/solution.png (architecture figure) and its page https://postimg.cc/ppvvndqH
  - https://www.kaggle.com/competitions/33105/images/thumbnail (competition thumbnail shown in the writeup header block)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2022/writeups/omar-bougacha-4-solution-multi-activation-branches
    (citation line)
  - https://www.kaggle.com/ambrosm · https://www.kaggle.com/mehrankazeminia · https://www.kaggle.com/obougacha ·
    https://www.kaggle.com/hatimchifa · https://www.kaggle.com/tttrrraaahhh
- **Lesson:** A 50-50 average with one strong published kernel is the cheapest +0.00003 on this board — and on a leaderboard
  packed at 0.998, that is the difference between 4th and 5th.

### TPSMAY22-05 · 5th · tttrrraaahhh (Islam Tlupov) · LB 0.99829/0.99824 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-may-2022/discussion/328553
- **Status:** FETCHED. `/c/` form (4.8 KB), `/competitions/<comp>/writeups/islam-tlupov-5-solution-catboost-keras` form
  (5.0 KB) and `/c/…?sort=hotness` with `X-Engine: browser` + `X-Timeout: 60` (4.8 KB) all collapsed 3 replies; resolved
  with the browser escape hatch (navigate + click "3 more replies" + accessibility snapshot), which exposed all 6 comments.
- **TL;DR:** CatBoost with Langevin posterior sampling (depth 8, huge regularization) blended with a Keras net using a
  hand-modified Mish activation; published scores arrive only in the author's comment reply.
- **Architecture:** FLAT · stages=1 · l1=2 (1 CatBoost-Langevin + 1 Keras NN) · l2=none · novel=none · mod=none ·
  topo=CatBoost(posterior_sampling=True, depth 8, very large reg) + Keras NN(custom Mish) → blend of the two
  predictions [weights not stated] → 0.99829/0.99824
  - Two different families are averaged, so this is `FLAT` rather than 1st's single-architecture blend; no meta-learner is
    trained on OOF anywhere on the page.
  - The blend is only worth +0.00002 private over his Keras alone (0.99822 → 0.99824) — the Keras member carries it.
  - `novel=none`: the custom Mish is an activation tweak (formula under Models), and Langevin is a CatBoost sampling
    option (`posterior_sampling=True`), not a new tabular architecture.
- **Setup:** rows/cols, split sizes, fold count, submission slots: not stated.
- Structural quirk exploited: none named by this author; he says only that AmbrosM's two-branch solution "is very unusual,
  I learned a lot from it" — whether his own net uses the branch split is **not stated** on the page.
- **Features:** not stated — no column names, encodings, or formulas appear in the body or the comments.
- **Models:** CatBoost with Langevin sampling: `depth = 8`, "a huge regularization coefficient (this helps the training not
  get stuck in a dead point)"; his comment advice: "use large regularization parameters (500 and above) and the Langevin
  function (`posterior_sampling=True`)".
- Keras NN using a modified Mish, verbatim from the body:
  `def custom_mish(x): return tf.keras.layers.Lambda(lambda x: x*tf.keras.backend.tanh(1+tf.keras.backend.log(tf.keras.backend.exp(x*0.7978845608028654))))(x)`
  - Author's own caveat on it: "I made the activation more sensitive (be careful with it, it often gives a gradient vanishing)".
- Individual scores (author's comment reply): CatBoost 0.99799 public / 0.99820 private; Keras 0.99825 public / 0.99822
  private; CatBoost + Keras ensemble 0.99829 public / 0.99824 private.
- Seeds, epochs, full hyperparameter grid: **not stated, and withheld by the author's own statement** — "The problem with
  Catboost is that it can produce completely different accuracy with the same training parameters. Therefore, I can hardly
  remember the exact parameters I used."
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV score published.
- CV-vs-LB gap: not computable.
- Author's trust verdict: not stated.
- **Ensembling:** blend of the CatBoost and Keras predictions; weights and mechanic (average vs rank vs hill-climb) not
  stated beyond "Then I combined CatBoost model results with the Keras solution".
- Measured effect: +0.00004 public and +0.00002 private over the Keras member alone.
- **Post-processing:** not stated.
- **Gains:** (+0.00002 private, +0.00004 public) blending CatBoost into the Keras net: 0.99822 → 0.99824 private.
- (+0.00004 private, the larger of the two steps) the Keras member over the CatBoost member: 0.99820 → 0.99822 private.
- No feature-engineering or tuning deltas are published in this entry.
- **Failed:** Not stated on the page: no failed model, feature, or blender is named anywhere in the body or the replies
  (counted all 6 comments).
- Reproducibility caveat that reads as a failure mode: CatBoost with identical parameters "can produce completely
  different accuracy" (author), so his `posterior_sampling=True` run is not re-runnable from this record.
- **Comments:** 6 comments (counted, all expanded via the browser snapshot). Technical content, all from the author:
  - Reply to @delai50 (62nd): the three score pairs quoted above plus the recipe advice "large regularization parameters
    (500 and above) and the Langevin function (`posterior_sampling=True`)".
  - Reply to @chrismunch (248th): CatBoost instability across identical settings; he declines to publish the exact
    parameters and promises a notebook from a different competition instead.
  - Chris Munch's own datapoint (commenter, not the author): his best CatBoost under Optuna tuning was ~0.9968 — about
    0.0012 below the 5th-place Keras member.
  - Remaining: 1 appreciation comment ("Thanks for the sharing!") with no content.
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://i.postimg.cc/4yd7k3RB/results-5-1.png (the "results" figure embedded in the body)
  - https://www.kaggle.com/competitions/tabular-playground-series-may-2022/writeups/islam-tlupov-5-solution-catboost-keras
    (citation line)
  - https://www.kaggle.com/competitions/33111/images/thumbnail (not cited here — page thumbnail is
    https://www.kaggle.com/competitions/33105/images/thumbnail)
  - https://www.kaggle.com/ambrosm · https://www.kaggle.com/tttrrraaahhh · https://www.kaggle.com/delai50 ·
    https://www.kaggle.com/chrismunch · https://www.kaggle.com/cv13j0 (author/commenter profile links on the page)
- **Lesson:** On a board compressed into 0.998, a second family buys 2e-5 — spend the effort on the NN member and on
  remembering your hyperparameters, because the blend will not save you.

## TPSMAY22 — consensus recipe
- **Architecture distribution:** 1st `SEED` (blend of variants of one two-branch Keras network; LightGBM excluded) ·
  4th `FLAT` (own multi-activation-branch NN, 50-50 with a community submission file) · 5th `FLAT`
  (CatBoost-Langevin + Keras, weights not stated).
  - **winner topology: `SEED`** — 1st submitted several variants of a single architecture and explicitly kept the GBM out;
    the two blended (`FLAT`) solutions finished 4th and 5th.
  - Read-across: every entry here is single-stage (`l2=none` on all three pages). Nobody on this board built a stack, and
    the two blends that mixed families added +0.00003 (4th) and +0.00002 (5th) private — i.e. less than the 0.00001 that
    separated 4th from 5th by a factor of 2-3.
- **New architectures at the board:** none. `novel=none` on all three entries — no TabM/FT-Transformer/NODE-class model
  appears anywhere on these 2022 pages.
  - The genuinely non-stock elements are wiring tricks, not architectures: 1st's **interaction-graph-constrained two-branch
    MLP** (the only idea on this board that changed a score by more than 1e-5 in the winner's hands), 4th's
    **per-branch activation families (swish/selu/relu)**, 5th's **custom Mish** and **CatBoost Langevin
    (`posterior_sampling=True`)**.
- **Agreed on (3 of 3 — TPSMAY22-01, TPSMAY22-04, TPSMAY22-05):** a Keras neural network is inside every submitted
  solution, and the NN beats the tree on this data.
  - 1st trained a LightGBM with `interaction_constraints` and left it out at LB 0.99778.
  - 5th's Keras member (0.99825/0.99822) is above his CatBoost member (0.99799/0.99820) on both boards.
  - 4th's submitted member is a NN plus one borrowed submission file; he names no tree at all.
- **Agreed on (2 of 3 — TPSMAY22-01, TPSMAY22-04):** express the interaction structure as *branches* in the network —
  1st splits branches by connected component of the xgbfir interaction graph, 4th splits branches by activation function
  "to explore different possibilities of feature interaction". 5th names no branch structure (not stated).
- **Agreed on (2 of 3 — TPSMAY22-01, TPSMAY22-04) — the community is the second member:** 1st builds on the Xgbfir
  interaction dataset published by @sudalairajkumar and the `unique_characters` feature from @cabaxiom; 4th's +0.00003 is
  literally a 50-50 average with @mehrankazeminia's published notebook output. 5th cites no borrowed artifact.
- **Agreed on (0 of 3 published):** no entry publishes a CV score or a CV-vs-LB gap; validation schemes are entirely
  undocumented on this board (verified by reading all three pages). Replication therefore has to rebuild the split itself —
  4th and 5th both inherit the shape of AmbrosM's public kernel rather than their own described protocol.
- **Divergences:** 1st used the interaction hint to *restrict* the model (two branches, cross-component interactions
  forbidden) and won; 4th used the same hint to *diversify* one model (three activation branches) and got 4th; 5th ignored
  the branch idea in favor of a second family (CatBoost Langevin) and got 5th, +0.00002 private over his own best member.
- Mid-rank vs top-rank gap on this board is tiny: 4th private 0.99825 vs 5th private 0.99824 = 0.00001, and both report the
  identical public 0.99829. Rank order here is decided inside the public-leaderboard noise, so the architecture label — not
  the score — is the useful information in this file.
- **Highest-leverage single trick:** the xgbfir interaction graph (TPSMAY22-01) — the Overview hint was public to everyone,
  and turning it into a two-component graph, then into a two-branch Keras architecture, is what separated the winner from
  the GBM variants that "were not enough to be included in the blend".
- **Nothing worked:** transferring the winning idea to a gradient-boosting tree — 1st's LightGBM with
  `interaction_constraints=[features_left, features_right]` stopped at LB 0.99778 (the only measured dead end on this
  board). Chris Munch's independent attempt at the tree path (Optuna CatBoost, ~0.9968, in the TPSMAY22-05 comment thread)
  lands in the same place.
- **Nothing worked** is otherwise unreported: TPSMAY22-04 and TPSMAY22-05 name no failed model, feature or blender, so no
  negative knowledge beyond the tree-vs-NN ordering can be aggregated from this competition.
