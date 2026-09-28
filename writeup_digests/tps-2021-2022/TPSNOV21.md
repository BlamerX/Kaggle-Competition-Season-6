## TPSNOV21 — Tabular Playground Series - Nov 2021
- Task: Tabular (Binary Classification) | Metric: ROC AUC Score | Problem: Practice binary classification
- Kaggle display title: "Tabular Playground Series - Nov 2021" (the comp where the leaderboard shook up: 5th's title is literally
  "+141 from Public Leaderboard")
- Competition: https://www.kaggle.com/c/tabular-playground-series-nov-2021
- Writeups covered: 4 of 4
- Score ladder: 1st `0.749 -> 0.75010 -> 0.75070 (LB of his three NN rounds), final LB not numbered` · 2nd `not stated` ·
  3rd `not stated (his 18 half-chunk public AUCs are the raw material, not a personal score)` · 5th `not stated (+141 rank places public->private)`.
- Commenter-side scores on these pages: 4th-place @pourchot reports 0.75020 from "around 10 files blended ... without 'data
  manipulation'"; 13th-place @obougacha reports 0.75169 without probing vs 0.75266 with probing.
- Rank-claim check: all four pages carry Kaggle badges matching their index IDs (1st/2nd/3rd/5th). The disagreement on this board is in
  commenter language, not in the badges - 3rd's page is congratulated "on the win" and told "You are the real winner" while the index and
  the page's own label keep him at 3rd. Index IDs are kept; both claims are recorded in the entry.
- Rows/cols: only 3rd publishes geometry (test = nine chunks of 60,000 rows each). No entry states a column count or a CV number.

### TPSNOV21-01 · 1st · jayjay75 (jayjay) · LB not stated/not stated (round ladder 0.749 / 0.75010 / 0.75070) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291883
- **Status:** FETCHED on the 2nd ladder step: first `/c/` discussion request returned a 1,581-byte empty shell; `X-No-Cache: true` on the
  same URL returned the full 6,351-byte page (writeup slug `jayjay-1-solution`, dated May 19, 2022) with all 10 comment blocks; the
  `X-Engine: browser` variant returned the identical 6,351 bytes.
- **TL;DR:** Three sequential "simple NN" retrains - clean, then relabeling the top 5% mislabeled training rows from round 1's OOF, then
  adding 5% pseudo-labeled test rows - blended in rank space with @ambrosm's and @pourchot's public kernels.
- The winning ingredient per his own comment is a fork of the best public probing notebook:
  `tpsnov21-012c-leaderboard-probing`.
- **Architecture:** FLAT · stages=3 · l1=3 blend members (his round-3 NN + @ambrosm's probing kernel + @pourchot's kernel) · l2=none ·
  novel=none · mod=cascade,pseudo,public-oof,rank-blend ·
  topo=NN(1) [OOF on train] -> relabel top 5% mislabeled rows -> NN(2) [LB 0.75010] -> +5% pseudo-labeled test rows -> NN(3) [LB 0.75070]
  -> rank(pct=True) blend with ambrosm + pourchot kernels -> submission
  - `stages=3` counts his three NN training rounds: each round's *output* becomes the next round's *labels/rows*, which is forward data
    flow (`mod=cascade`), not a blender.
  - The submitted prediction, however, is the blend, so the primary is `FLAT`: per ruling 9 a pseudo-label loop that only feeds one blend
    member is `mod=pseudo`; the loop would be primary only if it produced the submission itself.
  - `rank-blend` is his explicit recommendation, verbatim: "the only thing to modify is to use **rank** when you blend your prediction,
    since AUC is all about rank". Blend weights: not stated.
- **Setup:** rows/cols: not stated. Submission slots: not stated (3rd, on his own page, gives the comp-wide limit of 5 per day).
- Structural quirk exploited: the chunked/leaked test set - "I was too lazy to investigate the chunk thing so I mostly rely on your kernel
  for that", i.e. he consumed @ambrosm's chunk machinery instead of building it.
- **Features:** NONE PUBLISHED. The body names no engineered column, no encoding, and no feature source; "a simple NN" is the entire
  feature-side description. Exact formulas: source silent.
- **Models:** "a simple NN" trained three times; architecture, layer count, units, activation, optimizer, library: not stated.
- Round 1: "training a simple NN and getting the oof on the train" - LB ~0.749.
- Round 2: same NN retrained on relabeled data - LB ~0.75010.
- Round 3: same NN retrained with 5% pseudo-labeled test rows added - LB ~0.75070.
- Seeds, fold counts, blend weights of the final submission: not stated.
- **CV:** splitter, #folds, repeats, stratification, seeds: not stated. He reports LB after every round instead, and publishes no CV score
  at all, so no CV-vs-LB gap is computable.
- Round-to-round LB deltas (his numbers): 0.749 -> 0.75010 = +0.0011; 0.75010 -> 0.75070 = +0.0006.
- **Ensembling:** final blend = his round-3 NN + @ambrosm's probing kernel + @pourchot's kernel, mixed in rank space
  (`sub['target'] = sub['target'].rank(pct=True)`); weights not stated.
- Shared-OOF usage: yes, whole kernels rather than OOF columns - he says he relies on the two named public notebooks.
- His justification for rank space: "Submission probability coming from different models are on different scale so it's certainly better to
  do something like sub['target'] = sub['target'].rank(pct=True) before using them".
- **Post-processing:** the rank transform is the post-processing step he names; no threshold, calibration, clipping, or rounding otherwise.
- **Gains:** (+0.0011 LB) relabeling the top 5% mislabeled training rows using round 1's OOF: 0.749 -> 0.75010.
- (+0.0006 LB) adding 5% pseudo-labeled test rows and retraining: 0.75010 -> 0.75070.
- (unnumbered, "moved to my final LB") the rank-blend with the two public kernels - the step that actually took 1st, and the only one he
  does not quantify.
- (author comment) forking the best public probing kernel: "as you will see it increases the AUC by a small margin" - margin not stated.
- **Failed:** He admits not investigating the chunk structure himself ("too lazy") and buying it instead from two public notebooks - the
  anti-knowledge is that on this board the non-expert who blends the right public artifacts beat the experts who built them.
- Not stated as failures: no dead-end model, feature, or blender is named on this page.
- **Comments:** page header "10 Comments"; 10 named comment blocks render, 1 is a Topic Author reply, 0 deleted positions.
  - Author reply (the most technical content on the page, beyond the body): "I have forked the best public kernel and apply it, as you will
    see it increases the AUC by a small margin" + https://www.kaggle.com/jayjay75/tpsnov21-012c-leaderboard-probing/notebook
  - @akmalmir (15th) asks for the notebooks behind the tricks - not provided; @ankitkalauni (126th) cannot follow the rank trick and asks
    for reference links - not provided; both are why this entry is thin.
  - @charliex asks whether blending the top 3 (or 5, or 10) submissions would be even better - unanswered on the page.
  - @mhslearner (91st): "i wasn't sure that blending will win this time, but you proved me wrong".
  - @pourchot (4th): "Incredible score the last day !" - the last-day jump 4th-place witnessed is the probing/overfitting sprint described
    on TPSNOV21-03's page.
  - Related cross-page claim about this entry, stated on TPSNOV21-03's page (not here): @pourchot calls him "the last sprint winner
    [@jayjay75] with overfitix ... He certainly had a very good first solution."
- **Compute:** not stated (no wall-clock, GPU, RAM, or cost).
- **Artifacts:** (verbatim, not fetched)
  - https://www.kaggle.com/jayjay75/tpsnov21-012c-leaderboard-probing/notebook (author's own fork of the probing kernel, in his comment)
  - https://www.kaggle.com/ambrosm, https://www.kaggle.com/pourchot (the two kernel authors he credits by handle)
  - https://www.kaggle.com/jayjay75, https://www.kaggle.com/akmalmir, https://www.kaggle.com/charliex,
    https://www.kaggle.com/ankitkalauni, https://www.kaggle.com/yannbarthelemy, https://www.kaggle.com/mhslearner,
    https://www.kaggle.com/mathurinache, https://www.kaggle.com/edz123
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/writeups/jayjay-1-solution (citation line)
  - https://www.kaggle.com/competitions/28011/images/thumbnail
- **Lesson:** Relabeling your own most-misfit training rows is a cheap +0.0011, and once the metric is AUC, blending in rank space is the
  one post-processing step worth automating - but on a leaked board the unnumbered final step to 1st came from public kernels, not from
  his NN.

### TPSNOV21-02 · 2nd · gb4gb4 (GB4 GB4) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291903
- **Status:** FETCHED (first try, `/c/` discussion form, 4,401 B; writeup slug `gb4-gb4-2-solution`, 2 comments)
- **TL;DR:** One neural net, then per-chunk shrinkage toward 0.5 tuned by probing the public LB chunk by chunk.
- **Architecture:** SINGLE · stages=1 · l1=1 NN (raw output submitted as the baseline) · l2=none · novel=none · mod=none ·
  topo=NN -> raw submission (baseline) -> 9 LB probes, one rescaled chunk each -> per-chunk (c*target+0.5)/(1+c) -> final submissions
  - The submitted prediction is one model's output with a deterministic per-chunk transformation applied after `predict_proba`, so per
    ruling 2 the chunk rescaling is Post-processing and adds no stage.
  - The rescaling is not fitted on OOF either - each `c` was chosen from a public-LB probe of that chunk's 60k-row block, so no meta-learner
    exists anywhere on this page.
  - Not `PSEUDO`: no prediction is fed back into training; test rows never enter the training set.
- **Setup:** test set = 9 chunks (the chunking itself is credited to @grayjay's finding, and the flipped-target work to @motloch).
- Flipped-target rate across the whole dataset "about 25%", per-chunk range "about 24.8 to 25.2%".
- Slots: 9 probe copies plus final submissions; exact daily limit stated on TPSNOV21-03's page (5/day), not here.
- **Features:** none named - the page describes no engineered column, encoding, or feature subset. Source silent.
- Column dropped / GP features / aggregations: not stated.
- **Models:** "a neural network" - library, depth, units, activation, optimizer, seeds, fold count: all not stated.
- Credits for the NN itself: https://www.kaggle.com/chaudharypriyanshu and https://www.kaggle.com/dlaststark notebooks (handles only).
- **CV:** splitter, #folds, repeats: not stated; he publishes no CV number and no LB number at all.
- His selection signal is entirely the public LB: he submitted 9 copies of the baseline, each rescaling one different chunk, and read off
  which chunks improved.
- Outcome of that probe (his wording): "Chunk 6 turned out to benefit the most from being rescaled, while chunks 1 and 8 got worse."
- **Ensembling:** none - no blend, no meta-learner, no shared OOF on this page.
- **Post-processing:** per-chunk shrink toward the base rate: `(c*target + 0.5)/(1+c)` applied to each of the 9 test chunks with a
  different `c` per chunk; the 9 `c` values are not published.
- Rationale, verbatim logic: "if a chunk in the test set had a slightly higher rate of flipped targets then this chunk would contribute
  more to the overall error, and I should push the predictions of that chunk towards 0.5."
- **Gains:** no numeric gain is published anywhere on this page; the only measured facts are directional - chunk 6 gained the most, chunks
  1 and 8 lost.
- Ordering only: raw NN (baseline) < per-chunk rescaled NN, by his account of winning 2nd with the rescaled version.
- **Failed:** rescaling chunks 1 and 8 made the score *worse* - two of nine probes are named dead ends.
- His stated takeaway is an admission of misread data: "One thing I learned from this competition is to look closer at the datasets and
  not take anything for granted. It seems like @motloch and @grayjay saw the simple patterns that the neural networks missed."
- **Comments:** page header "2 Comments"; 2 named comment blocks, 0 Topic Author replies, 0 deleted positions. No numbers in comments.
  - @chaudharypriyanshu (115th, whose NN notebook he credits): "I am very much glad, my work came into your use."
  - @akmalmir (15th): praise only, no technical question answered.
- **Compute:** not stated.
- **Artifacts:** (verbatim; the page cites its subjects by handle, and the handles resolve to the profiles below)
  - https://www.kaggle.com/chaudharypriyanshu, https://www.kaggle.com/dlaststark, https://www.kaggle.com/motloch,
    https://www.kaggle.com/grayjay, https://www.kaggle.com/gb4gb4, https://www.kaggle.com/akmalmir
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/writeups/gb4-gb4-2-solution (citation line)
  - https://www.kaggle.com/competitions/28011/images/thumbnail
- **Lesson:** When labels are corrupted at a slightly different rate per block, the highest-value model is a calibration vector over blocks,
  not a better classifier - and 9 LB probes are enough to fit it.

### TPSNOV21-03 · 3rd · ambrosm (AmbrosM) · LB not stated/not stated · CV "GroupKFold cv scores are useless, stratified cv scores are even more useless"

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291766
- **Status:** FETCHED (first try, `/c/` discussion form, 12,240 B; writeup slug `ambrosm-3-solution-don-t-trust-the-cv-scores`, dated
  Dec 1, 2021; all 14 comment blocks, 2 of them author replies)
- **TL;DR:** He trained 5 real estimator families, watched them lose to DummyRegressors on the leaked data, then measured the public AUC of
  18 half-chunks and turned those 18 numbers into per-chunk target probabilities.
- **Architecture:** FLAT · stages=1 · l1=20 components (18 probe-derived half-chunk probabilities + 2 public notebooks) · l2=none ·
  novel=none · mod=public-oof ·
  topo=[classify train-vs-test] + [GroupKFold(10) over chunks] -> abandon supervised models -> 18 submissions of half-chunk constants ->
  per-half-chunk target probabilities -> 92% probe + 8% two public notebooks -> submission
  - The 92% component is not a fitted estimator: each half-chunk gets a constant probability recovered analytically from that
    half-chunk's public AUC, i.e. DummyRegressor-style output. Nothing is trained on the OOF matrix, so `STACK2` is impossible here.
  - The 92/8 weights were chosen by submitting combinations to the public LB; per ruling 1 weight selection stays `FLAT`.
  - Rejected predecessors, all trained and then discarded: linear regression, kernel ridge regression, 1000 nearest neighbors, LightGBM,
    neural networks (see Failed).
- **Setup:** test = 9 chunks × 60,000 rows each (his statement; 540,000 by that product). Each chunk split into two half-chunks at the
  hyperplane given by the leaked training data → 18 half-chunks.
- Submission budget: "at five submissions per day, 18 half-chunks took four days" - one submission per half-chunk.
- Column count, train rows, final slots: not stated.
- **Features:** the exploitable structure is geometric, not engineered:
  - Chunk membership (9 blocks) and the train/test separating hyperplane from the leaked data, used to cut each chunk in half.
  - No engineered column, encoding, aggregate, or target-encoding variant appears anywhere on this page.
- **Models:** discarded after training: linear regression, kernel ridge regression, 1000 nearest neighbors, LightGBM, neural networks.
- Winner: a pair of DummyRegressors trained on the leaked data - "They all lost against a pair of DummyRegressors trained on the leaked
  data."
- Hyperparameters of the discarded estimators: not stated (he names families only).
- **CV:** GroupKFold with 10 groups, trained on nine chunks and validated on the tenth, adopted after the chunking became known; he
  publishes no CV number for any model, because his conclusion is that the metric is meaningless.
- His verdict, verbatim: "GroupKFold cv scores are useless, stratified cv scores are even more useless, the public lb was the best we could
  get."
- Diagnostic that set the strategy: a classifier can separate training from test samples (his link to discussion/291003), so train-region
  validation cannot predict test-region performance.
- **Ensembling:** final blend = 92% half-chunk probe probabilities + 8% two public notebooks; the notebooks are named only as "two public
  notebooks"; the individual AUC values of the 18 probes are not printed on the page.
- Math for turning 18 public AUCs into probabilities is delegated: "@safavieh explains the math in detail in his notebook".
- **Post-processing:** none beyond the probe estimate itself; no threshold, calibration, clipping, or rank transform on this page.
- **Gains:** (+0.00097 on a *commenter's* submission, this page) @obougacha (13th): model without probing 0.75169 -> 0.75266 with probing
  using @pourchot's notebook - the only number in this file that prices what probing bought somebody who was not on the podium.
- (unnumbered) the probe-and-blend path itself: he states no personal public or private score.
- **Failed:** Weighted training data - upweighting training samples near the test region: "The result on the test set (i.e. leaderboard
  score) was disappointing."
- Every supervised family he tried lost to a DummyRegressor: linear regression, kernel ridge regression, 1000 nearest neighbors, LightGBM,
  neural networks.
- Stratified CV is worse than GroupKFold here, and both are useless: his stated reason is that nine training chunks carry no information
  about the tenth - "If nine training chunks don't give any information about the tenth one, why would the ten training chunks give any
  information about the test chunks?"
- **Comments:** page header "14 Comments"; 13 named comment blocks render, 1 position shows "This comment has been deleted."; 2 are Topic
  Author replies. This is the richest technical thread of the four Nov entries:
  - Author reply on why arbitrary groups cannot be probed (to @siukeitin's proposal): "The assumption is that within a chunk, the samples
    are independent and identically Bernoulli-distributed, and the Bernoulli parameter is a function of the chunk. Under these assumptions,
    the public leaderboard of the chunk gives an unbiased estimate for the parameter of the chunk's Bernoulli distribution. If we create
    random sectors, the public leaderboard part of the sector doesn't give any information on the private part. In a random sector, public
    and private parts are independent random variables with expected value 0.5."
  - @siukeitin (33rd) contributes the probe-budget math: 2 probes estimate the sector positives `p` and the test-set positives `P`;
    amortized probes per sector `1 + 1/n`; with `d` days of probing you could probe `5d - 1` sectors.
  - @obougacha (13th): 0.75169 without probing vs 0.75266 with @pourchot's probing notebook, plus "Note to myself: Always trust your CV….
    :'(", and his image link https://i.postimg.cc/8P2Y9N8g/cv.jpg.
  - @pourchot (4th): "my solution was very simple: around 10 files blended to reach the best score (0.75020) without 'data manipulation'.
    Then, Ambrosm code + my little update", and he names 1st as "the last sprint winner @jayjay75 with overfitix" with a link to
    https://www.kaggle.com/pourchot/my-super-simple-overfitting-tool-explained.
  - @lucaszborecki (124th) argues the top did not overfit: "many of top public remained so you haven't overfitted".
  - @lucamassaron (186th): "it has been the only one where DNN worked better than lightgbm" - a reader's generalization, not the author's.
  - Criticism and the author's reply: @padova (74th) calls it "a competition that leaks data, then redacts it, is spoiled from the start ...
    like acquiring skills for a videogame"; AmbrosM: "Personally, I did learn a lesson for the real world: Always assess the quality of the
    training data. And if the quality is bad, take the consequences."
  - @keysersoze309 (6th): "You are the real winner." - the clearest rank-claim conflict on this board; Kaggle's own badge on every one of
    his comments says 3rd, and the index ID TPSNOV21-03 stays 3rd.
- **Compute:** the only cost stated is submission budget: 18 probes over 4 days at 5/day, plus the remaining submissions spent on blend
  weights. Wall-clock, GPU, RAM: not stated.
- **Artifacts:** (verbatim, not fetched)
  - https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291003 (train-vs-test classifier exists)
  - https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/286731 (the data is chunked)
  - https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/290810 (his switch to GroupKFold)
  - https://www.kaggle.com/ambrosm/tpsnov21-012-leaderboard-probing (the probing kernel)
  - https://www.kaggle.com/safavieh/probing-test-set-chunks-with-math (the math behind the probe)
  - https://www.kaggle.com/ambrosm/tpsnov21-012c-leaderboard-probing (his final submission notebook)
  - https://www.kaggle.com/pourchot/my-super-simple-overfitting-tool-explained and
    https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/url (both as the page prints them, in @pourchot's comment)
  - https://i.postimg.cc/8P2Y9N8g/cv.jpg and https://postimg.cc/kBywtrXT (in @obougacha's comment)
  - https://www.kaggle.com/ambrosm, https://www.kaggle.com/siukeitin, https://www.kaggle.com/vipin20, https://www.kaggle.com/obougacha,
    https://www.kaggle.com/safavieh, https://www.kaggle.com/lukaszborecki, https://www.kaggle.com/pourchot,
    https://www.kaggle.com/lucamassaron, https://www.kaggle.com/padova, https://www.kaggle.com/junjifan,
    https://www.kaggle.com/keysersoze309, https://www.kaggle.com/kpretomazi
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/writeups/ambrosm-3-solution-don-t-trust-the-cv-scores
    (citation line)
  - https://www.kaggle.com/competitions/28011/images/thumbnail
- **Lesson:** Before tuning a CV scheme, test whether a train-vs-test classifier separates the sets - if it does, per-block public-AUC
  inversion can be worth more than every model you own.

### TPSNOV21-05 · 5th · andrewmatic (Andrewmatic / Kushnerov) · LB not stated/not stated (+141 rank places public->private) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291846
- **Status:** FETCHED (first try, `/c/` discussion form, 3,894 B; writeup slug
  `andrew-kushnerov-5-solution-and-141-from-public-le`, dated Dec 1, 2021, 2 comments)
- **TL;DR:** One Keras NN (4 dense layers from 600 units, dropout 0.3, swish) on MaxAbs-scaled data plus `sin(f27)`, a chunk-aligned
  10-fold CV, and an SVM post-processing step - and the discipline to stop submitting.
- **Architecture:** SINGLE · stages=1 · l1=1 Keras NN · l2=none · novel=none · mod=none ·
  topo=MaxAbsScaler + trig features -> Keras NN (4 Dense, swish, dropout 0.3) trained with KFold(shuffle=False, 10 splits) -> SVM
  post-processing of predictions -> submission
  - One architecture submitted; no blend, no meta-learner, no second family appears on the page.
  - The SVM step touches predictions after they exist, so per ruling 2 it is Post-processing and adds no stage; its mechanics are `source
    silent` ("SVM postprocessing some improved the result (thanks @ambrosm)").
  - His rank gain is *not* from a stronger model: 141 places came from the public-to-private shake-up, which he attributes to not chasing
    the overfitted top.
- **Setup:** CV aligned to the leakage structure: "simple KFold(shuffle=False) with 10 splits - for using exactly one chunk to validate and
  nine to learn" (same design 3rd calls GroupKFold).
- Rows/cols/slots: not stated; `f27` implies at least 28 numeric feature columns, and that is the only column name on the page.
- **Features:** exact, as named:
  - Scaling: `MaxAbsScaler` chosen over Robust, Standard, and Power(Gauss)/quantile transforms - "I found that MaxAbsScaler transform data
    better than Robust / Standart or Power (Gauss)".
  - Trigonometric transforms: "Some trigonometric functions slightly improved the result, like `sin(f27)`". Magnitude: "slightly" is the
    author's own word; no number attached.
  - Aggregated statistics, target encoding, interactions, dropped columns: not stated. Source silent.
- **Models:** Keras neural network: 4 Dense layers, "high amount of units (starting from 600)", dropout 0.3, `swish` activation
  ("better than relu"), long-epoch training with a chosen learning rate and learning-rate scheduler.
- Exact layer widths, LR values, scheduler type, epochs, seeds, batch size, optimizer: not stated.
- NN hyperparameter selection method: asked in the comments, not answered (see Comments).
- **CV:** `KFold(n_splits=10, shuffle=False)` so each validation fold is exactly one data chunk; adopted only "when the situation became
  clear with chunks".
- No CV number, no LB number, and no CV-vs-LB gap appears on this page; the single reported metric is his rank movement of +141 places from
  public leaderboard to private.
- Implied trust verdict: he kept a chunk-aligned CV and refused to probe, and that combination is what the +141 measures.
- **Ensembling:** none stated - no blend members, no weights, no shared OOF on this page.
- **Post-processing:** "SVM postprocessing some improved the result (thanks @ambrosm)" - the target of the SVM correction, its parameters,
  and whether it was fit on OOF data are all not stated.
- **Gains:** (+141 leaderboard rank places, public -> private) his stated result of not joining the end-of-competition overfitting race.
- Qualitative, unquantified, in his own ranked wording: MaxAbsScaler over Robust/Standard/Power; `sin(f27)` "slightly"; swish over relu; the
  correct LR + LR scheduler described as "important"; SVM postprocessing "some improved the result".
- No AUC delta is attached to any of these five - the page is a list of directions, not magnitudes.
- **Failed:** Rejected scalers by name: Robust, Standard, and Power (Gauss) - all worse than MaxAbsScaler on his CV.
- Rejected activation: `relu`, worse than `swish` in his runs.
- Failed strategy (his 7th point): chasing the public board - "When it became clear that the top was strongly overfitted, just waiting for
  the end of the competition :)" - i.e. the leaderboard top was the thing to avoid copying.
- **Comments:** page header "2 Comments"; 2 named comment blocks, 0 Topic Author replies, 0 deleted positions.
  - @mhslearner (91st) asks the one question that would make this reproducible: "For choosing NN hyperparameters was it based on trial and
    error or on some optimization method" - no answer exists on the page.
  - @pourchot (4th) names the winning behaviour: "Your 'self-control' at the end of the overfitting competition has made the difference".
- **Compute:** "long epoch training" is the only compute statement; wall-clock, GPU, RAM, cost: not stated.
- **Artifacts:** the body and comments cite no notebook, dataset, or link other than profile URLs - the SVM post-processing credit to
  @ambrosm is by handle with no link.
  - https://www.kaggle.com/andrewmatic, https://www.kaggle.com/ambrosm, https://www.kaggle.com/pourchot, https://www.kaggle.com/mhslearner
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/writeups/andrew-kushnerov-5-solution-and-141-from-public-le
    (citation line)
  - https://www.kaggle.com/competitions/28011/images/thumbnail
- **Lesson:** On a broken board, a clean single model plus a chunk-aligned CV is a rank strategy: he gained 141 places by *not* spending
  submissions on the public score everyone else was overfitting.

## TPSNOV21 — consensus recipe
- **Architecture distribution:** 1st `FLAT` (3-round cascade/pseudo NN chain, rank-blended with 2 public kernels) · 2nd `SINGLE` (one NN +
  per-chunk shrinkage toward 0.5, chosen by 9 LB probes) · 3rd `FLAT` (18 probe-derived half-chunk probabilities at 92% + 2 public
  notebooks at 8%) · 5th `SINGLE` (one Keras NN + SVM post-processing).
  - **winner topology: `FLAT`** — 1st place is a rank-space blend, and the author's own comment credits the decisive last step to his fork
    of @ambrosm's public probing kernel (`tpsnov21-012c-leaderboard-probing`); the two single-model entries took 2nd and 5th.
  - Read-across: nobody on this board won with stack depth. Two of the four submissions contain no fitted estimator at their centre at all
    (2nd's per-chunk `c` vector, 3rd's dummy-style chunk constants).
- **New architectures at the board:** none. `novel=none` on all four pages: the models named are a "simple NN" (1st), an unnamed neural
  network (2nd), discarded linear/kernel-ridge/kNN/LightGBM/NN estimators plus DummyRegressors (3rd), and a 4-layer Keras MLP with swish
  (5th). This is the 2021 baseline board: no tabular-specific architecture appears anywhere.
  - The genuinely new machinery on this board is not a model class but a measurement technique: public-AUC inversion of test chunks
    (3rd, and 2nd's per-chunk rescaling), plus 1st's OOF relabeling of the top 5% mislabeled training rows.
- **Agreed on (4 of 4 - TPSNOV21-01, -02, -03, -05):** the test set is chunked into 9 blocks and that structure, not modelling, decided the
  competition. 1st "rely on your kernel for that [the chunk thing]"; 2nd probes "the 9 chunks in the test set"; 3rd "nine chunks of size
  60000 each"; 5th "when the situation became clear with chunks I started to use simple KFold(shuffle=False) with 10 splits".
- **Agreed on (2 of 4 - TPSNOV21-03, TPSNOV21-05):** the same chunk-aware validation design - 9 chunks to learn, 1 held out - named
  `GroupKFold` by 3rd and `KFold(shuffle=False, 10 splits)` by 5th. 1st and 2nd publish no splitter at all.
- **Agreed on (3 of 4 - TPSNOV21-01, -02, -03):** the public leaderboard, not CV, is the working objective; decisions were made by
  submitting. 1st reports LB after each of 3 rounds; 2nd fit nine `c` values from nine probe submissions; 3rd states it outright and spent
  18 submissions over 4 days at the 5/day limit.
  - The exception is the counter-evidence that makes this board famous: 5th did the opposite and moved +141 places public→private, and
    13th's comment on 3rd's page prices naive probing at only +0.00097 (0.75169 → 0.75266).
- **Agreed on (3 of 4 - TPSNOV21-01, -02, -05):** a neural network is the core model. 1st retrained the same simple NN three times; 2nd
  started from an NN baseline; 5th's submission is one Keras NN. 3rd is the exception and the warning - he trained NNs among five families
  and dropped all of them to DummyRegressors.
- **Agreed on (3 of 4 - TPSNOV21-01, -03, -05):** community public notebooks/kernels as literal components of a podium submission. 1st blends
  @ambrosm's and @pourchot's kernels into his winning submission; 3rd ends with 8% of two public notebooks and credits @safavieh for the
  probe math; 5th applies @ambrosm's SVM post-processing. 2nd credits four people but publishes no external artifact in the submission.
- **Agreed on (4 of 4):** nobody published a CV score, a hyperparameter grid with values, an engineered feature set (beyond 5th's
  `sin(f27)` + MaxAbsScaler), or a private leaderboard number. Any replication of this board is therefore architecture-level only.
- **Divergences:** 1st spent nothing on the data structure and bought it from public kernels, while 3rd reverse-engineered the structure at
  the cost of 18 submissions and finished behind him - the rank gap between "build it" and "blend it" is exactly this board's 1st vs 3rd.
- Divergence in model ambition: 1st, 2nd and 5th all submitted NN-centred predictions; 3rd submitted essentially no model, and Luca
  Massaron's comment on his page ("the only one where DNN worked better than lightgbm") is the community's way of saying the same thing the
  3rd-place author proved by discarding his own NN.
- Divergence in slot strategy, the one that reshuffled the board: 5th's title is a rank delta (+141), and 4th's own comment here says his
  non-manipulated blend of ~10 files reached 0.75020 - i.e. mid-board scores clustered within ~0.002 while rank order moved hundreds of
  places, so on this dataset score closeness told you nothing about rank.
- **Highest-leverage single trick:** the per-chunk public-AUC inversion (TPSNOV21-03): 18 half-chunk probes -> target probabilities -> 92% of
  the final blend. Its transferable core is 2nd's far cheaper version - shrink each block's predictions toward the base rate with
  `(c*target + 0.5)/(1+c)`, nine submissions to fit nine `c`.
- **Nothing worked:** every conventional supervised estimator, at every rank that tested one. 3rd: linear regression, kernel ridge
  regression, 1000 nearest neighbors, LightGBM, neural networks all lost to a pair of DummyRegressors; his weighted-training-toward-test-
  region idea was "disappointing"; and both GroupKFold and stratified CV are declared useless.
- **Nothing worked:** relu (5th, replaced by swish) and Robust/Standard/Power-Gauss scaling (5th, replaced by MaxAbsScaler).
- **Nothing worked:** trusting the public board - stated by 5th (top "strongly overfitted"), by 13th's comment ("Note to myself: Always trust
  your CV…. :'("), and by 124th's counter-argument that the top public stayed, which is itself evidence the scramble was the defining event.
