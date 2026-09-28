## TPSJUN22 — Tabular Playground Series - Jun 2022
- Task: Tabular (Regression) | Metric: RMSE | Problem: Practice regression
- Kaggle display title: "Tabular Playground Series - Jun 2022" / "Practice your ML skills on this approachable dataset!"
- Competition: https://www.kaggle.com/c/tabular-playground-series-jun-2022
- Writeups covered: 6 of 6
- Metric as reported on the pages: **RMSE on imputed null values (LOWER better)** — this is a null-value
  imputation task, not a multiclass/log-loss task. Every score-quoting page quotes RMSE-scale numbers
  (0.83–1.05; the 8th's page publishes no numbers at all), 16th states "with RMSE metric", and 4th calls it "the first unsupervised model in this series". The delegation
  brief for this file said "multiclass log loss, mushroom species"; no page in this set mentions mushrooms,
  species, classes or log loss, so the brief is wrong and the pages were followed.
- Score ladder: 1st `not stated (public)/0.83343` (DAE alone 0.83351) · 2nd `~0.8358 (unlabeled)` ·
  4th `not stated (his GBM-stage scores ranged 0.90–0.86)` · 8th `not stated` · 16th `0.83784/0.83593` for the superseded
  PyTorch NN and `0.89131/0.88957` for the superseded CatBoost version; final Keras score not stated ·
  17th `0.94694 (public, with NA counts)` vs `1.04555 (public, without NA counts)`; final score not stated

### TPSJUN22-01 · 1st · sebastianvangerwen (Sebastian van Gerwen) · LB not stated/0.83343 · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334331
- **Status:** FETCHED (1 attempt, `/c/.../discussion/334331` + `X-Return-Format: markdown`, 14,990 bytes, body + 20 comments)
- **TL;DR:** The whole competition is imputing the conditional distribution of `F_4` cells where 2+ values are missing.
- A denoising autoencoder (DAE) with feature-wise mask embeddings predicts all masked cells at once — no per-column iteration.
- Final −0.00008 RMSE came from a conditional ensemble: single-attribute models used only on rows whose `F_4` null count is 1.
- **Architecture:** FLAT · stages=1 · l1=6 DAE runs (3 PyTorch + 3 TensorFlow) + single-attribute prediction runs ·
  l2=none (no meta-learner fitted) · novel=Denoising-Autoencoder · mod=none ·
  topo=mask+data → per-feature embeddings (data+mask, added) → MLP[7 dense, mish, LayerNorm+skip] → masked-MSE →
  DAE×6 avg + conditional single-attribute models (F_4 null count == 1 rows only)
  - One stage only: the DAE consumes the masked row **and** the input mask vector; nothing consumes another model's
    output, so this is not a stack and not a cascade.
  - The submission is the *averaged* DAE pool, then blended with the single-attribute family restricted to rows with
    exactly one `F_4` null — the blend is conditional/rule-based, no weights learned on OOF.
  - Averaging across frameworks is the unusual part: TF runs scored better than PyTorch runs, and the average of all
    six beat either family (0.83351 private).
- **Setup:** rows/cols not stated. Column groups named: `F_1`, `F_2`, `F_3`, `F_4` (no `F_0` on this page).
- `F_1` and `F_3`: mean-imputed; `F_2` completely ignored; only `F_4` is modeled.
- Split sizes, submission slots: not stated.
- Structural quirk exploited: the RMSE is computed only over cells that were null in the given test rows, so the model
  only has to estimate a conditional distribution at masked positions.
- **Features:** no feature engineering beyond the missingness machinery itself.
- `source null matrix` = locations of the original data nulls.
- `random mask` = binomial mask (probability 5%) multiplied by the data to zero extra values; OR-ed with the
  "at least one masked value per row" mask; the function is `random_mask()` in his notebook.
- `input mask` = combination of the source-null vector and the random mask vector, embedded like the features.
- Embeddings: each feature and each mask entry linearly projected to embedding dimension **16** (final submission),
  data and mask embeddings **added**, then flattened.
- Alternatives tried: dot-product attention between mask and linear embeddings (per arXiv:2106.16057) — performed worse.
- Columns dropped: `F_2` ignored entirely; `F_1`/`F_3` mean-imputed rather than modeled.
- **Models:** DAE = custom MLP decoder, 7 dense layers, mish activation, layer normalization + skip connections.
- Layer sizes: performance kept increasing with width; only tried up to 2048 (batch size had to be cut dramatically).
- Trained in both PyTorch and TensorFlow; 3 runs each (6 total). Loss = masked MSE (loss forced to 0 at originally
  present values). Output computation: masked positions take the dense head, unmasked positions pass the input through
  (zero gradient for present values under MSE).
- Exact hyperparameters (lr, batch size, epochs, seeds, optimizer): not stated.
- Second family: "single-attribute prediction runs" (one attribute at a time) — architecture/library not stated.
- **CV:** splitter, #folds, #repeats, stratification, seeds: not stated; no CV number published at all.
- Reported numbers are LB only: DAE pool 0.83351 private, final conditional ensemble 0.83343 private (delta −0.00008).
- CV-vs-LB gap and trust verdict: not stated.
- **Ensembling:** average of 6 DAE runs (3 PyTorch + 3 TF); "tensorflow runs performed significantly better, but the
  average of all performed the best".
- Then `Conditional Ensemble`: single-attribute model outputs substituted only where the row-wise `F_4` null count == 1.
- Weights are plain averaging + a hard row condition; no hill-climbing, no ridge/LogReg meta, no shared public OOF cited.
- **Post-processing:** none described beyond the conditional ensemble rule; no clipping, rounding or rank transform stated.
- **Gains:** (−0.00008 RMSE private) conditional ensemble over the DAE-only pool, 0.83351 → 0.83343.
- (stated as "much better") feature-wise mask embeddings over plain dropout on the masked data.
- (stated as "the average of all performed the best") TF+PyTorch cross-framework averaging over TF-only runs.
- (not quantified) 5% binomial mask probability, chosen over other probabilities he experimented with.
- (author's framing) persisting with the autoencoder idea rather than iterating per feature was the decisive move.
- **Failed:** dot-product attention between mask and linear embeddings — "performed worse than simply adding them together".
- Autoencoder variants without a mask embedding/attention mechanism "performed quite poorly" (author comment).
- Widths above 2048: not tried — computation time forced smaller batch sizes.
- The linked PyTorch reference notebook "scored poorly because I messed up the submission dataframe" (author's own note).
- **Comments:** 20 comments; technical author replies mined above and here:
  - Mask probability is binomial p=5% (best of several probabilities tried); "at least one value" is an extra OR-ed step
    purely to avoid rows with zero masked values wasting compute; the implemented function `random_mask()` is therefore
    not strictly binomial (reply to @arturra, the 2nd-place finisher).
  - Masks are drawn independently per row, regenerated for every batch (reply to @thedevastator, 79th).
  - "I actually don't use the DAE to generate features. The DAE is the model itself" — training simulates missingness via
    the random mask; prediction passes the real null vector as mask (reply to @creamiracle).
  - He tried the DAE first, deliberately to avoid iterating over features; points to @masatomurakawamm's masked-LM-like
    notebook and @ehekatlact's autoencoder/GAN thread as prior art.
- **Compute:** not stated except that layer width beyond 2048 became compute-limited and required small batch sizes.
- GPU model, wall-clock, RAM, Kaggle limit hits, cost: not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/sebastian-van-gerwen-1-solution-denoising-autoenco (author citation)
  - https://www.kaggle.com/code/sebastianvangerwen/1st-place-solution-tps-jun-denoising-ae
  - https://arxiv.org/abs/2106.16057 (mask/feature dot-product attention paper he tried and rejected)
  - https://arxiv.org/abs/2002.08338 (loss zeroed at data nulls — same method)
  - https://www.kaggle.com/code/masatomurakawamm/tps-jun22-application-of-masked-language-model
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/330054
  - https://www.kaggle.com/sebastianvangerwen , https://www.kaggle.com/masatomurakawamm , https://www.kaggle.com/ehekatlact
  - https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334331
  - https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8094969%2Fc96e318817a66ad510461cb5fa92d35a%2Fdae%20schematic.png?generation=1656645783893261&alt=media (architecture drawing)
- **Lesson:** When the metric only scores missing cells, train a model to impute under a simulated-missingness mask and
  embed the mask itself — the mask is the most informative feature, not noise to be dropped.

### TPSJUN22-02 · 2nd · arturra (ArturRa) · LB not stated/not stated · CV not stated (author reports "got stuck at about 0.8358", unlabeled)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334319
- **Status:** FETCHED (1 attempt, `/c/.../discussion/334319` + `X-Return-Format: markdown`, 8,099 bytes, body + 10 comments)
- **TL;DR:** Reuses the May-2022 TPS 1st-place Keras NN and splits `F_4` into six groups by NaN count (0–5), training one
  multi-output model per NaN-pattern with the number of outputs equal to the number of NaNs.
- "A lot more than just 80 regressors"; mean for `F_1`/`F_3`; plateaued at ~0.8358.
- **Architecture:** SINGLE · stages=1 · l1=>80 Keras NN instances (one per NaN-count group × target-column pattern) ·
  l2=none · novel=none · mod=multitarget ·
  topo=F_4 rows bucketed by NaN count (0=training, 1..5=prediction) → per-pattern multi-output Keras MLP → assembled submission
  - One architecture, many instances: for the group with two NaNs in `F_4_0`/`F_4_1` the model has 2 outputs and 13 inputs
    (`F_4_2`…`F_4_14`) — inputs and outputs change per pattern, the topology does not.
  - No blender and no meta-learner: rows are routed by their NaN count to the model trained for that pattern, so the
    "ensemble" is a partition, not a weighted average.
  - `F_1`/`F_3` are filled with the mean — a deterministic fill, not a trained stage (`l2=none`).
- **Setup:** group `F_4` named with 15 columns `F_4_0`…`F_4_14` (from the 13-feature example). Row/col counts, split
  percentages, slots: not stated.
- Structural quirk exploited: rows with **zero** NaNs in `F_4` are fully observed and become the training set; rows with
  1–5 NaNs are prediction targets only.
- Hint source: the May 2022 TPS overview text "The dataset has similarities to the May 2022 Tabular Playground" — so he
  ported that competition's winning notebook.
- **Features:** no engineered columns described.
- `F_1`, `F_3`: mean only ("I found nothing more useful than just the mean technique").
- `F_4`: raw columns only, used as inputs minus the NaN positions of the target pattern.
- Encodings, GP/generated features, dropped columns: not stated (work is spread across many notebooks).
- **Models:** Keras neural networks, architecture inherited from
  https://www.kaggle.com/code/pourchot/tpsmay22-keras-test-tuned (itself built on @ambrosm's May-2022 work).
- Output head sized to the pattern's NaN count (2 outputs for a 2-NaN pattern, etc.).
- Model count: "a lot more than just 80 regressors" (author's phrase; exact count not stated).
- Libraries/hyperparameters/seeds/folds: not stated.
- **CV:** splitter, folds, repeats, seeds: not stated. The only score given is "got stuck at about 0.8358" — public or
  private not labeled; earlier single-target tuning of `F_4` was stuck "at about 0.85".
- **Ensembling:** none — per-pattern models are assembled by NaN-count routing; no weights, no averaging across families,
  no shared OOF stated.
- Work is distributed over separate training notebooks because of training time (example: 4-NaNs group notebook).
- **Post-processing:** mean fill for `F_1`/`F_3`; no threshold, clipping, calibration or rounding stated.
- **Gains:** (−0.0142 RMSE-ish) single-target `F_4` tuning ~0.85 → NaN-count-grouped multi-output models ~0.8358.
  (Both numbers are the author's own "about" values; delta stated as approximate.)
- (structural) porting the May-2022 winner instead of designing a new net — he credits that as the starting point.
- (credit, not his own idea) the NaN-count split was suggested by @ehekatlact's notebook.
- **Failed:** plain mean/median-only imputation for `F_4` — nothing better found for `F_1`/`F_3` either, but that was a
  dead end for `F_4`.
- One-column-as-target modelling plateaued at ~0.85 and could not be pushed further.
- "After all I got stuck at about 0.8358 and couldn't do anything for further improvement."
- Training cost was the binding constraint: too many models to fit in one notebook.
- **Comments:** 10 comments; author replies add:
  - Agrees with @rizqyad's experiments that `F_1`/`F_3` features are **independent**, hence the conditional mean is the
    optimum for them, and cites @sebastianvangerwen's "laconic proof" in discussion/332985.
  - Confirms he is the 2nd-place finisher replying on his own thread; no withheld hyperparameters published.
  - Non-author (relevant anti-knowledge): @grahambroughton (132nd) — a few `F_1`/`F_3` columns deviate from ~0 mean/1 std
    (`F_1`: 7, 12, 13; `F_3`: 19, 21) and some have different distributions in null-containing rows; nobody showed it mattered.
  - @thedevastator (79th) links the community solutions summary post (discussion/334415).
- **Compute:** "it took a lot of time for training" — the reason for the many-notebook structure; wall-clock, GPU, RAM: not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/pourchot/tpsmay22-keras-test-tuned
  - https://www.kaggle.com/code/ehekatlact/tps2206-the-na-count-of-each-record-is-critical/notebook?scriptVersionId=98204877
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/328369
  - https://www.kaggle.com/code/arturra/nn-with-four-nans/notebook
  - https://www.kaggle.com/code/arturra/analysis-of-nans
  - https://www.kaggle.com/sebastianvangerwen
  - https://www.kaggle.com/code/rizqyad/a-million-imputation (commenter's notebook, quoted here)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/332985
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/arturra-2-solution (author's own citation line)
  - https://www.kaggle.com/ambrosm , https://www.kaggle.com/pourchot , https://www.kaggle.com/ehekatlact , https://www.kaggle.com/rizqyad
  - named without URLs: "TPS2206 The na count of each record is critical!"
- **Lesson:** Before designing anything, read the next competition's intro text against the previous one — a shared data
  generator means last month's winning notebook is this month's baseline.

### TPSJUN22-04 · 4th · ravi20076 (Ravi Ramakrishnan) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334497
- **Status:** FETCHED (1 attempt, `/c/.../discussion/334497` + `X-Return-Format: markdown`, 7,295 bytes, body + 8 comments)
- **TL;DR:** Baseline ladder (mean/median → SimpleImputer → LGBM/CatBoost/XGBoost) then a plain dense Keras/TF net for
  `F_4` plus linear models + descriptive statistics for `F_1`/`F_3`.
- Explicitly a "much simpler than the winners" solution built almost entirely on public notebooks.
- **Architecture:** FLAT · stages=1 · l1=1 Keras/TF dense NN (`F_4`) + linear models + descriptive-stat fills (`F_1`/`F_3`) ·
  l2=none · novel=none · mod=none ·
  topo=per column group: F_4 → dense NN (NA-count-conditioned) | F_1,F_3 → linear models + mean/median → one submission
  - Different families serve different **column groups**, so the "blend" is a partition by target, not a weighted average
    of competing predictions for the same cell.
  - No meta-learner fitted on OOF anywhere in the writeup; the earlier GBM/SimpleImputer models were superseded, not blended.
  - The empty `novel` slot is deliberate: he names DAE and TabNet only as things he *learned about* from others, not
    models he used.
- **Setup:** column groups `F_1`, `F_3`, `F_4` modeled/filled separately. Rows/cols, split sizes, slots: not stated.
- Structural quirk exploited: NaN count per record determines which model/feature set applies (from @ehekatlact's notebook).
- Timeline: "countless hours over a month", first such Kaggle placement for the author.
- **Features:** NA-count-style conditioning inherited from public notebooks ("TPS Jun 2022 PyTorch Lightning with NA counts",
  "TPS2206 The na count of each record is critical!").
- Cross-group prediction idea: "using other columns in the table to predict a column with lots of nulls" — applied to
  predict `F_1` and `F_3` too, alongside mean/median submissions.
- Exact engineered feature names, encodings, dropped columns: not stated.
- Baselines tried: SimpleImputer with descriptive statistics (mean, median), `IterativeImputer` for one baseline submission.
- **Models:** final `F_4` model = "a simple dense neural network (without a callback)" on Keras + TensorFlow.
- One `F_4` candidate from the PyTorch Lightning NA-count notebook was finetuned and used.
- `F_1`/`F_3` final = combination of linear models + descriptive statistics.
- Superseded: LGBM, CatBoost, XGBoost (score range 0.90–0.86).
- Hyperparameters, layers, seeds, fold counts: not stated.
- **CV:** splitter, folds, seeds: not stated; author's own scores other than "0.90–0.86" at the GBM stage are not stated.
- No CV-vs-LB gap discussion; no validation-credibility claim.
- **Ensembling:** no stack described — `F_4` from the NN, `F_1`/`F_3` from linear + stats, one finetuned PyTorch Lightning
  candidate folded in; weights: not stated. Shared OOF usage: not stated.
- **Post-processing:** mean/median fills used as submitted values for `F_1`/`F_3` (descriptive statistics kept in the
  final combination). No threshold/clipping/rounding stated.
- **Gains:** (score range 0.90–0.86) the author's stated range **at the LGBM/CatBoost/XGBoost stage** — the page gives
  no SimpleImputer-baseline score, so this is a stage range, not a baseline→GBM delta.
- (0.86 → 4th place, delta not stated) GBM stage → NA-count-conditioned dense NN for `F_4`.
- (qualitative, biggest unlock) "TPS2206 The na count of each record is critical!" "provided me a new perspective".
- **Failed:** pure descriptive-statistic imputation ceiling — mean/median and `IterativeImputer` baselines were not
  competitive enough to keep.
- Modeling `F_1`/`F_3`: "used the mean and median values of these columns as well, making some submissions through this
  time, trying to figure out a strategy" — no gain reported from it.
- Callbacks: omitted deliberately "considering my hardware constraints" — the stated reason is hardware, not measured loss.
- **Comments:** 8 comments; counted, and **nothing technical in the replies** — all 8 are congratulations, thanks, and
  @thedevastator's link to the summary post (https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415).
- Body (not replies) holds the technical content: model list, the 0.90–0.86 range, and the four credited notebooks.
- **Compute:** self-reported hardware constraints (no training callbacks, single-machine Keras); wall-clock, GPU, RAM: not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/ravi-ramakrishnan-4th-place-approach (author citation)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415
  - https://www.kaggle.com/ravi20076 , https://www.kaggle.com/juanmorenod , https://www.kaggle.com/mpwolke , https://www.kaggle.com/thedevastator
  - named without URLs: "TPS Jun 2022 PyTorch Lightning with NA counts", "TPS2206 The na count of each record is critical!",
    "0.89 IterativeImputer + Optuna", "TPS22Jun Tensorflow with NA counts", Abhishek Thakur's null-value-imputation tutorial,
    his own EDA notebook ("My own EDA notebook for this competition received quite a few upvotes")
- **Lesson:** On an imputation task, the missingness pattern (NA count per record) is worth more than model complexity —
  a callback-free dense NN plus a correct NA-count framing took 4th place.

### TPSJUN22-08 · 8th · daru191854 (Daru) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334567
- **Status:** FETCHED (4 attempts: plain `/c/` returned a 2,506-byte shell, `X-No-Cache` on `/c/` returned 164 bytes,
  `/c/` + `X-No-Cache` + `X-Engine: browser` + `X-Timeout: 60` gave 3,499 bytes, and `/competitions/.../discussion/334567`
  + `X-No-Cache` gave the full 3,560-byte body with both comments — the last route is what was digested)
- **TL;DR:** An ensemble of 5 neural networks, each structured "like MLM" (masked language model): mask cells, train to
  reconstruct them, then ensemble the nets (combination mechanic not stated).
- The writeup is three sentences plus one architecture screenshot and one code link; no scores published.
- **Architecture:** FLAT · stages=1 · l1=5 neural networks · l2=none · novel=Masked-Language-Model-NN · mod=none ·
  topo=masked-input NN (MLM-style reconstruction head) ×5 → ensemble → submission
  - Level 0 only: 5 same-idea networks whose reconstruction outputs are combined; no model consumes another's predictions.
  - The MLM-like structure means the network is trained on masked cells and predicts the masked values directly (same
    idea as the 1st place; the page does not mention entry 01, so independence is not stated).
  - The `novel` slot carries only the author's own label ("Neural Net like MLM"); the exact layer stack is inside his
    notebook, which is not quoted here.
- **Setup:** not stated (no row/col counts, no groups, no slots, no split sizes).
- Structural quirk exploited: masked-cell reconstruction matches the RMSE-over-nulls metric; stated only implicitly by the
  MLM framing.
- **Features:** not stated.
- **Models:** 5 neural networks, "Each Network is like MLM"; one-network reference implementation
  https://www.kaggle.com/code/daru191854/single-network-like-mlm.
- Library, layers, activation, loss, optimizer, epochs, seeds: not stated.
- **CV:** not stated; no number of any kind appears on the page.
- **Ensembling:** "ensemble of 5 neural Networks" — combination mechanic and weights: not stated.
- **Post-processing:** not stated.
- **Gains:** not stated — the page contains no before/after numbers.
- **Failed:** not stated — no dead ends reported.
- **Comments:** 2 comments; counted, and **nothing technical** — @ravi20076 (4th) congratulates, @thedevastator (79th)
  endorses the MLM framing as generalizable and links
  https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415.
- **Compute:** not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/daru191854/single-network-like-mlm
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/daru-8-solution-neural-net-like-mlm (author citation)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415
  - https://www.kaggle.com/daru191854 , https://www.kaggle.com/ravi20076 , https://www.kaggle.com/thedevastator
  - https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F191854%2Fb26b36d0083ededb1987d066bb3263a1%2F2022-07-02%20115537.png?generation=1656730567280810&alt=media (architecture screenshot)
- **Lesson:** Masked-reconstruction training is the natural objective when only missing cells are scored — but a writeup
  with zero numbers is unreusable, so the code link is the whole solution.

### TPSJUN22-16 · 16th · tttrrraaahhh ("What" / Islam Tlupov) · LB 0.83784/0.83593 (PyTorch NN, superseded); final not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334358
- **Status:** FETCHED (1 attempt, `/c/.../discussion/334358` + `X-Return-Format: markdown`, 7,674 bytes, body + 8 comments)
- **TL;DR:** Three-stage personal ladder: per-NaN CatBoost (0.89131/0.88957) → heavy PyTorch NN per NaN count
  (0.83784/0.83593) → a lighter Keras residual net that fits one notebook.
- The binding constraint was Kaggle's 36,000-second notebook limit, not model capacity.
- **Architecture:** SINGLE · stages=1 · l1=1 Keras MLP per NaN-count notebook (residual `Add` layers) · l2=none ·
  novel=none · mod=none ·
  topo=rows bucketed by NaN count → Keras MLP per count [swish, residual Add connections, 70 epochs, batch 16384, AdamW] → submission
  - The submitted model is one Keras net (per missing-count slice); CatBoost and the heavy PyTorch net were earlier
    submissions he replaced, not ensemble members.
  - Residual `Add` layers sum input tensors to allow depth (author's comment explaining his own diagram).
  - No blender, no meta-learner, no second family in the final path.
- **Setup:** the page names no column groups (target columns and column-group structure not stated); exact rows/cols not stated.
- Structural quirk exploited: NaN-count-conditioned training, one notebook per count (PyTorch stage needed NaN count 1,
  2, … run in series).
- Kaggle constraint: 36,000 s execution limit; final Keras net "fit into one notebook", the PyTorch one did not.
- **Features:** target = one column at a time in the CatBoost stage ("cycle through all of NaN using Catboost");
  NaN count used as the conditioning variable; the page cites no notebook names.
- `lecun_uniform` as `kernel_initializer` — an accuracy improvement he did not have time to upload.
- Encodings, dropped columns, engineered names: not stated.
- **Models:** CatBoost — `lr=0.6222, depth=4, n_estimators=12000, leaf_estimation_method="Newton",
  leaf_estimation_iterations=10, posterior_sampling=True, l2_leaf_reg=7, random_strength=3`, RMSE metric.
- PyTorch NN — "heavy architecture (256-128-64… etc)" with BatchNorm; 0.83784 public / 0.83593 private.
- Keras NN (final) — swish activation on all layers except the output Dense, AdamW `lr=0.012, amsgrad=True,
  weight_decay=4e-7, beta_1=0.95`, 70 epochs, batch size 16384, no callbacks, residual Add connections.
- Seeds and fold counts: not stated.
- **CV:** no local CV at all — every number quoted is a Kaggle LB score, and model selection was by LB.
- Public→private movements reported: CatBoost 0.89131 → 0.88957 (−0.00174, private better), PyTorch NN 0.83784 → 0.83593
  (−0.00191, private better).
- **Ensembling:** none in the final solution — sequential replacement of one model by another; no averaging or stacking stated.
- **Post-processing:** none stated.
- **Gains:** (−0.05347 public / −0.05364 private RMSE) CatBoost-per-NaN → heavy PyTorch NN (0.89131/0.88957 → 0.83784/0.83593).
- (stated as improving accuracy, unquantified) `lecun_uniform` kernel initializer — never submitted.
- (structural) shrinking weights so the net fits one notebook's runtime, which is what made the final submission possible.
- **Failed:** mish activation — "in this task is much worse" than swish.
- Training callbacks — "it reduced performance and time was important in this task".
- RAdam optimizer — better accuracy than AdamW but too slow to fit in 36,000 s; 300 epochs + RAdam "would have given
  amazing results" but his AMD GPU (no CUDA) refused.
- Heavy PyTorch net with BatchNorm — did not run on Kaggle within execution time; "I killed two weeks for that".
- **Comments:** 8 comments; author replies add:
  - The PyTorch (not Keras) solution was the one split across 4 notebooks run in series; the final Keras solution was light
    enough for one notebook.
  - "The Add layer is Residual Connection. Add simply sums the input tensors", benefit = deeper models, better accuracy and
    training stability (reply to @grahambroughton, 132nd).
  - "if I had run my final solution with more epochs, it might have been a winner" — self-reported under-training.
  - @ravi20076 (4th) congratulates; no hyperparameters published in replies beyond the body.
- **Compute:** Kaggle notebook 36,000 s limit hit repeatedly; local hardware is an AMD GPU without CUDA, so no GPU training
  path on Kaggle-scale epochs; wall-clock per run: not stated.
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/islam-tlupov-16-deep-magic (author citation)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415
  - https://www.kaggle.com/tttrrraaahhh , https://www.kaggle.com/ravi20076 , https://www.kaggle.com/thedevastator ,
    https://www.kaggle.com/grahambroughton , https://www.kaggle.com/hskhawaja
  - https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3435682%2Fd5d00579ecae1a7e103afb9621089176%2Fmodel.png?generation=1656656884246071&alt=media (Keras model diagram)
  - named without URLs: "TPS Jun 2022 PyTorch Lightning with NA counts"-style NA-count notebooks are not cited here; only
    his own model diagram and the summary post
- **Lesson:** On a runtime-capped Playground task, parameter count per epoch is the real hyperparameter — halving weights
  to fit 36,000 s beat keeping a heavier net that never submits.

### TPSJUN22-17 · 17th · nikhilkhetan (Nikhil) · LB 0.94694 (public, best stated)/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334343
- **Status:** FETCHED (4 attempts: plain `/c/` 2,506-byte shell, `/c/` + `X-No-Cache: true` 6,809 bytes full body + 3
  comments (used), `/c/` + `X-No-Cache` + `X-Engine: browser` + `X-Timeout: 60` 6,809 bytes identical,
  `/competitions/` + `X-No-Cache` 6,856 bytes equivalent)
- **TL;DR:** NA-count-conditioned modelling plus a public-LB probe: he deduced that the public leaderboard is almost all
  `F_4` cells, so `F_1`/`F_3` work is invisible on public.
- Simple average of two models (TF re-implementation of @ehekatlact's masked NN + a LightGBM/XGBoost regressor combo),
  applied separately to `F_1`, `F_3`, `F_4`.
- **Architecture:** FLAT · stages=1 · l1=2 models per column group (NN + LGBM/XGB combo), applied to 3 groups · l2=none ·
  novel=none · mod=none ·
  topo=per group (F_1 | F_3 | F_4): [TF masked NN + LightGBM/XGBoost w/ NA count] → simple average → submission
  - Level 0 only and a plain unweighted average — no meta-learner, no hill-climbing, no rank blending.
  - The two models compete on the *same* target cells and are averaged, which is what makes this `FLAT` rather than a
    per-group partition.
  - Slot strategy is part of the architecture: 2 submissions were used to test two mutually exclusive `F_1`/`F_3` paths
    (mean fill vs modelling) at once.
- **Setup:** exactly **2 submission slots** allowed, used as the A/B probe.
- Public-set discovery: "most of the public dataset is comprised of F4 columns and maybe a few records with missing in F1
  and F3" — found by "a very simple probe".
- Structural quirk exploited: model by number of missing values per record (@ehekatlact's finding).
- **Features:** NA count per record added to the GBM feature set ("LightGBM and XGboost regression with NA count").
- `F_2` (categorical, no missing values): 10 days of slicing/dicing, then discarded — see Failed.
- Validation set built from the training data in his own public notebook ("Setting up a Validation set").
- Encodings and full column lists: not stated.
- **Models:** (1) NN based on https://www.kaggle.com/code/ehekatlact/tps2206-the-na-count-of-each-record-is-critical ,
  re-implemented in TensorFlow; (2) LightGBM + XGBoost regression combination with NA count.
- "A few other models" developed but dropped by public-LB selection.
- Hyperparameters, seeds, folds: not stated.
- **CV:** no CV numbers published — the published scores are public LB: 1.04555 (first submission, no NA counts) →
  0.94694 (same model + NA counts + modifications). He also states the private moved with public on the two-slot test.
- Author's trust verdict on public LB: modelling `F_1`/`F_3` gave "public score hardly moved" and he attributes it to the
  public set containing few `F_1`/`F_3` cells, i.e. public LB is a *biased* proxy for the `F_1`/`F_3` part only.
- **Ensembling:** "took a simple average of their results" — 2 models, unweighted average, run once per column group.
- **Post-processing:** none; `F_1`/`F_3` mean imputation is offered as an alternative path, not as a correction, in one
  of the two submissions.
- **Gains:** (−0.09861 public RMSE) adding NA counts + modifications to the first model: 1.04555 → 0.94694.
- (−0.00005 public and private, worth ~3 LB places) modelling `F_1`/`F_3` instead of mean-imputing them — "ultimately it
  did not matter much".
- (unquantified) publishing his validation-set notebook — enabled his own model comparison.
- **Failed:** `F_2` exploration: "After slicing and dicing in every possible way using the F_2 features and not finding
  anything, I decided to move on" — ~10 days lost.
- Direct modelling of `F_1`/`F_3`: "I tried lots of models and tactics but the public score hardly moved."
- Mean imputation for `F_1`/`F_3` was *not* clearly beaten, but he notes we "simply do not have any line of sight for the
  change in performance" — i.e. no way to validate that branch.
- Selecting models by public LB on the `F_1`/`F_3` slice: he flags this as unreliable for exactly those columns.
- **Comments:** 3 comments; counted, and **nothing technical** — @thedevastator (79th) links the summary post
  (discussion/334415), @jgraham20 (184th) remarks the top scores are extremely close, @ravi20076 (4th) congratulates.
  All technical content is in the body.
- **Compute:** not stated (no wall-clock, GPU, or limit notes).
- **Artifacts:** (verbatim, unfetched)
  - https://www.kaggle.com/code/nikhilkhetan/f-2-features-search-for-meaning
  - https://www.kaggle.com/code/nikhilkhetan/setting-up-a-validation-set
  - https://www.kaggle.com/code/ehekatlact/tps2206-the-na-count-of-each-record-is-critical
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/writeups/kheti-17th-place-solution (author citation)
  - https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/334415
  - https://www.kaggle.com/nikhilkhetan , https://www.kaggle.com/ehekatlact , https://www.kaggle.com/thedevastator ,
    https://www.kaggle.com/jgraham20 , https://www.kaggle.com/ravi20076
- **Lesson:** Probe the leaderboard before trusting it — knowing which columns the public set actually scores tells you
  where validation is impossible and where a −0.00005 gain is real.

## TPSJUN22 — consensus recipe
- **Architecture distribution:**
  - 1st `FLAT` (6 masked-embedding DAE runs averaged, then conditionally blended with single-attribute models) — novel
  - 2nd `SINGLE` (>80 instances of one Keras MLP, one per NaN-count pattern, routed not blended)
  - 4th `FLAT` (Keras dense NN for `F_4` + linear models + mean/median for `F_1`/`F_3`; partition by column group)
  - 8th `FLAT` (ensemble of 5 MLM-style NNs)
  - 16th `SINGLE` (one light Keras residual MLP; CatBoost and the heavy PyTorch net were replaced, not blended)
  - 17th `FLAT` (unweighted average of a TF masked NN and a LightGBM/XGBoost+NA-count combo, per column group)
  - winner topology: `FLAT`, single-stage; **0 of 6 entries (01, 02, 04, 08, 16, 17) report fitting a meta-learner on an
    OOF matrix** — no `STACK2`, `CASCADE`, `PSEUDO`, `AUTOML` or `AGENT` submission appears in this set (entry 08 never
    states its blend mechanic at all, so its `FLAT` rests on "ensemble of 5 neural Networks" alone).
- **New architectures at the board:** `Denoising-Autoencoder` at 1st (the only model published on that page: DAE pool
  private 0.83351, final conditional ensemble private 0.83343) and `Masked-Language-Model-NN` at 8th (no score published). Everything else is a stock MLP, GBM (CatBoost/LGBM/XGBoost) or
  linear/descriptive-stat fill. TabNet is named only as something 4th learned about from other people's code (04), not used.
  A masked autoencoder-style variant also appears in the comments of 01 as prior art
  (https://www.kaggle.com/code/masatomurakawamm/tps-jun22-application-of-masked-language-model).
- **Agreed on (4 of 6: 02, 04, 16, 17):** condition every model on the row's NaN/NA count — the single most-cited unlock,
  with 02, 04 and 17 thanking @ehekatlact's "TPS2206 The na count of each record is critical!" notebook by name and 16
  running one notebook per NaN count. 01 handles the same information as an explicit mask embedding, 08 masks cells like
  an MLM, so all six encode missingness but only four use the count formulation.
- **Agreed on (3 of 6: 01, 02, 17):** `F_1`/`F_3` are not worth real modelling — 01 mean-imputes them and ignores `F_2`,
  02 states the mean is the best technique (independence, per 01's proof in discussion/332985), 17 tested modelling them
  and gained ~0.00005. Entry 04 is the exception: his final `F_1`/`F_3` fill combines linear models *and* descriptive
  statistics; entries 08 and 16 do not state what they did for those groups.
- **Agreed on (4 of 6: 01, 02, 04, 16):** the submitted `F_4` predictions come from a neural net, not a GBM; entry 17
  averages a TensorFlow NN with a LightGBM+XGBoost combo, and 16th's GBM-only version (0.89131/0.88957) was superseded by
  his own NN. Entry 08's page never names a column group, so it is not counted (its model list is all-neural, though).
- **Agreed on (3 of 6: 02, 04, 16):** Kaggle notebook runtime, not model capacity, is the binding constraint — 02 spread
  training over multiple notebooks because it "took a lot of time", 04 dropped training callbacks "considering my hardware
  constraints", 16 hit the 36,000 s limit and shrank the net to fit one notebook.
- **Divergences:** 1st models the whole masked vector jointly in one network with mask embeddings; 2nd decomposes into
  pattern-specific multi-output nets (0.83343 vs ~0.8358, −0.0024 for joint masking, mixing 01's private with 02's
  unlabeled figure); mid-table entries instead iterate per column or average NN with GBM and land far worse (16th's
  CatBoost private 0.88957 is ~0.056 above 1st, 17th's public 0.94694 ~0.11 above 1st's private — 17th's number is public
  LB, so treat that gap as indicative only).
- **Highest-leverage single trick:** add the row's NA count to the feature/model-conditioning set — 17th measured
  1.04555 → 0.94694 public (−0.09861) from adding NA counts plus modifications (entry 17).
- **Nothing worked:** per-column GBM imputation as an endpoint (16th's CatBoost 0.89131/0.88957, 04's 0.90–0.86 plateau);
  `F_2` categorical exploration (17th, ~10 days, zero yield); modelling `F_1`/`F_3` at all (17th ~0.00005; 02's and 01's
  mean-only results matched it); mish instead of swish (16th); training callbacks (16th); RAdam for accuracy (16th, too
  slow for the 36,000 s limit); dot-product mask/feature attention instead of addition (1st); mask-less autoencoder
  variants (1st, "performed quite poorly"); heavy PyTorch nets that cannot finish a notebook (16th).
