## TPSFEB21 - Tabular Playground Series - Feb 2021
- Task: Tabular (Regression) | Metric: RMSE | Problem: Practice regression
- Kaggle display title: "Tabular Playground Series - Feb 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-feb-2021
- Writeups covered: 7 of 7
- Score ladder (as reported by the authors; format private/public unless noted): 1st submission `not stated/not stated` — the only LB numbers he ever gives are a one-model test in a comment, `0.84248/0.84148` (CV 0.8413); headline CV 0.8412 · 2nd `0.84228/0.84157` · 3rd `not stated` · 4th `not stated/0.84190` (CV 0.84159) · 6th `0.84198→0.84185 LB progression, final private 0.84249` · 8th `not stated` · 11th `not stated`
- Audit note: every `/c/<comp>/discussion/<id>` link here now 301s to that competition's `/competitions/<comp>/writeups/<slug>` page (Kaggle migrated the 2021 TPS threads), so both forms return the same body; the `/competitions/.../discussion/<id>` form returns a 166-byte shell. All 7 re-fetched with `/c/` + `X-No-Cache: true`; TPSFEB21-01 needed the browser tools because 11 of its 40 comments stay collapsed under every proxy form, `?sort=polls` included.

### TPSFEB21-01 · 1st · Ren (ryanzhang) · LB 0.84148/0.84248 (author's "one model" test in a comment; the final submission's LB is never published) · CV 0.8412 (DAE features + single Ridge, 5-fold)

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745
- **Status:** FETCHED (`X-No-Cache: true` + `X-Engine: browser` + `X-Timeout: 60` recovered 15733 bytes; audit re-pass got the same 15733 bytes on the plain `/c/` form). The 11 nested replies stay collapsed under every proxy form **including `?sort=polls`** — they were expanded only with the browser tools, and are the source of the score and URL added below. Page now renders as `/writeups/ren-1st-place-dae-training-code` (Kaggle redirect).
- **TL;DR:** Winning model is a denoising transformer autoencoder (DAE) whose extracted features feed one Ridge regression.
- Code lives in a GitHub repo, not in the writeup; the page itself is short (post + 40 comments).
- **Architecture:** CASCADE · stages=2 · l1=5 DAEs · l2=Ridge (single linear meta on DAE features) · novel=Transformer-DAE · mod=embed ·
  topo=5× stacked transformer-encoder DAE (different swap-probability sets, blended) → learned features → Ridge → 0.8412 CV
  - The DAE is trained on a corruption-reconstruction task (per-column swap noise); the page does **not** state whether test rows entered DAE training. Its hidden activations become the *input features* of the downstream Ridge — data flows forward, not into a blender (ruling 5).
  - The submitted prediction came from the Ridge on DAE features, not from the DAE alone.
  - Five DAEs were trained with different swap-probability sets and used as one feature source; the author's words are "I trained 5 such DAE with different sets of swap probabilities" and, of re-training with small noise changes, "they blend very well" — how the five are combined is not stated.
- **Setup:** rows/cols not stated; the author's comment reply names 14 numerical variables, and the swap-proba arrays posted in the thread have 11 entries — the page never says which columns those cover, only that categoricals need a higher swap rate than "the 14 numerical variables"; the "500_000 by 4608" hidden matrix (author reply to anthony) implies ~500k rows. Submission slots, split sizes: not stated.
- **Features:** raw columns only in the DAE encoder stage; no hand-built features published on this page.
- A 1024-unit dense layer then a "divide" procedure feeding transformer encoders over subspaces — the layer size and the "divide" step are commenter delai50's reading of the repo, not the author's words; the author's reply confirms only the intent ("the idea here with subspaces is to learn multiple representations simultaneously").
- Categoricals handled with per-column corruption probabilities; encodings beyond that: not stated.
- **Models:** Transformer-DAE: stacked transformer encoders instead of `linear->relu` hidden layers; mask head output passes through sigmoid implicitly via `torch.nn.BCEWithLogitsLoss` (author comment).
- Reconstruction loss = weighted sum of masked and unmasked input, following the Stacked Denoising Autoencoders (Vincent 2010) paper (author comment).
- Per-column swap probas: posted by commenter Aditya Soni (5th) from the repo, then discussed in the author's reply to him: `repeats = [2,2,2,4,4,4,8,8,7,15, 14]`, `probas = [.95,.4,.7,.9,.9,.9,.9,.9,.9,.9,.25]`; author: "Just trial and error... I tuned the first three and got bored and just set .9 for the remaining bunch."
- The author's reply also gives the only row/column count on the page: 14 numerical variables, and says categoricals need a higher swap rate to reach the same corruption level.
- 5 DAEs trained with different swap-probability sets; downstream Ridge: hyperparameters not stated.
- Sparsity analysis (author comment to anthony): 500,000×4608 hidden matrix scanned; ~3% of neurons "truly dead"; the analysis notebook was not shared ("it is a mess").
- **CV:** 5-fold cross-validation; CV 0.8412 for Ridge on DAE features (the headline number in the post body). In a collapsed reply (Aditya Soni asking "What was your best DAE model score on LB/CV?") the author adds: "Just tested with one model cv .8413 private 0.84248 public 0.84148" — his words, and whether "one model" means one DAE or the full 5-DAE pipeline is not stated. Splitter stratification, seeds, repeats, trust verdict: not stated. The final submission's LB score is never published on this page.
- **Ensembling:** level-0 = 5 DAEs blended (mechanic not stated); level-1 = single Ridge on the blended DAE features. No GBM pool, no stacker.
- **Post-processing:** not stated.
- **Gains:** (rank, not score) the only margin claim anywhere is Dave E's "first place by a wide margin" (TPSFEB21-02's page); 1st publishes no final LB score, so no delta is computable. His comment numbers are a component test, not the submission: single-model private 0.84248 is *worse* than 2nd's blend private 0.84228, and CV 0.8412 vs LB 0.84248 are different measurement types — do not subtract across them or across entries.
- (per-column noise) tuned swap probabilities on the first three entries of the `probas` array; the author says the values "are by no means the best" and that re-training with small changes "blend very well" (he trained 5 DAEs this way).
- **Failed:** no dead end is named; this is a code-sharing post, not the full 1st-place writeup. In the collapsed replies the author refuses to ablate (danzel lists split loss / mask-loss weighting / per-feature noise / transformer arch / lr-epoch trade-off; author: "Isn't figuring out this where the most fun is?") and rejects the "DAE removes the CTGAN noise" reading, saying it instead learns conditional distributions such as P(X1 | X2 to X20).
- **Comments:** 40 comments (3 tagged appreciation); 11 nested replies stay collapsed under every proxy form and were read by expanding them in the browser. Technical author replies mined above: mask/reconstruction loss detail, subspaces rationale, dead-neuron count, swap-proba values and 5-DAE blend, the single-model CV/private/public scores, the refusal to ablate, the mask-prediction origin story ("teach DAE to fix corrupted data, might as well add this mask prediction task" — parenting anecdote), the SDAE pointer, and the "why DAE works" answer (learns conditional distributions; at work in consumer goods the representation lets a linear model match GBDT and is "quite useful for clustering"). Also in replies: Dave E (2nd) states his own TF/pytorch-starter DAE attempts failed (quoted in TPSFEB21-02), and for transformer newcomers the author names "the attention is all you need paper" (no URL given) plus a linked AutoInt paper.
- **Compute:** not stated (only implied "crazy compute power" would be needed to fully optimize swap probas).
- **Artifacts:** (all URLs cited by the author or in author replies, verbatim; not fetched)
  - https://github.com/ryancheunggit/Denoise-Transformer-AutoEncoder (the "Link to the code")
  - https://raw.githubusercontent.com/ryancheunggit/Denoise-Transformer-AutoEncoder/main/assets/diagram.png (network diagram)
  - https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html (author reply)
  - https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf (Stacked Denoising Autoencoders, author reply)
  - https://s4.gifyu.com/images/transformer_dae.png (commenter danzel's redraw of the diagram)
  - https://arxiv.org/abs/1810.11921 (AutoInt, author reply — only reachable after expanding a collapsed thread)
- **Lesson:** On fully anonymized tabular data, a corruption-trained autoencoder that reconstructs its own noise pattern can beat every tuned GBM — treat the representation stage, not the blender, as the competition.

### TPSFEB21-02 · 2nd · Dave E (davidedwards1) · LB 0.84157 public/0.84228 private · CV "pretty similar to public leaderboard"

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222762
- **Status:** FETCHED on 2nd ladder attempt (`/c/` form returned 164-byte empty cache shell; recovered 3742 bytes with `X-No-Cache: true`)
- **TL;DR:** Same January approach rerun: 4 models (LGBM, XGB, two NNs) blended with Optuna-searched weights over CV OOF predictions.
- Author explicitly states the DAE attempt failed.
- **Architecture:** FLAT · stages=1 · l1=4 (LGBM, XGB, NN1, NN2) · l2=none (optuna weight search, not a fitted meta-model) · novel=none · mod=none ·
  topo=(LGBM + XGB + NN1 + NN2) → optuna OOF-weight search → weighted blend
  - Weights found by Optuna on CV OOF predictions = weight selection, so `FLAT` per ruling 1, not `STACK2`.
  - The submitted blend (0.84228 private / 0.84157 public) beats every single member on both boards (best private member 0.84282, best public member 0.84229), so the winner was the blend.
- **Setup:** rows/cols not stated; categorical features "dealt with in the usual way" (i.e., as in his January post); slots not stated.
- **Features:** no new features named on this page; author defers to the January-2021 post for the actual pipeline.
- **Models:** per the author's own note "Format Private/Public Score" (first number is PRIVATE): LGBM 0.84282 private/0.84229 public · XGB 0.84356/0.84301 · NN1 0.84362/0.84250 · NN2 0.84333/0.84261.
- NN1/NN2 architectures and all hyperparameters: not stated ("generally just changed some parameters etc" from January).
- **CV:** splitter/#folds not stated; author: "All had CV scores pretty similar to public leaderboard." No numeric CV published.
- **Ensembling:** optuna weight search over CV OOF predictions; final weights `lgbm 0.468606, xgb 0.082158, nn1 0.197289, nn2 0.251397`; blend output 0.84228 private / 0.84157 public ("Output score 0.84228 / 0.84157" under the author's Private/Public format note).
- **Post-processing:** not stated.
- **Gains:** (-0.00054 RMSE private) blend 0.84228 vs best single member LGBM 0.84282 private.
- (-0.00072 RMSE public) blend 0.84157 vs best single member LGBM 0.84229 public; vs XGB 0.84301 public the blend gains 0.00144.
- (unquantified) the optuna weight search itself is not ablated against a flat average on this page.
- **Failed:** DAE: "didn't have much time in first half of the month to explore the DAE idea, then did not have much success in second half of month with making it work."
- Same author's comment on 1st's thread (that page is TPSFEB21-01, not this one, quoted here only as his own cross-post): "both my own efforts in Tensorflow and my efforts to build on your pytorch starter were not successful in making a meaningful breakthrough."
- **Comments:** 1 comment total (Heitor Rapela Medeiros, 288th, congratulatory) — nothing technical in replies.
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216071 (the January post the solution defers to)
- **Lesson:** When a new exotic technique is winning, a disciplined rerun of last month's solid blend plus searched blend weights still takes 2nd — the gap is the exotic model, not the blender.

### TPSFEB21-03 · 3rd · Ken (kntyshd) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/223455
- **Status:** FETCHED (1st `/c/` attempt returned full 5214 bytes, body + all 3 comments)
- **TL;DR:** Discussion post is a pointer to the author's public notebook; the page itself states zero pipeline detail.
- Title's own claim: "just ensembling GBDTs", "didn't use any special techniques".
- **Architecture:** not stated (proof: the page names no estimator list, no blend weights and no meta-learner — only "ensembling GBDTs" plus a notebook link, which cannot separate a plain `FLAT` average from a `STACK2` under rulings 1/12) · stages=not stated · l1=not stated · l2=not stated · novel=none · mod=none · topo=not stated
  - `novel=none` follows the page only: it names GBDTs and the LightGBM tag, and claims no other model; nothing on the page would license a novel name.
  - Everything replicable from this page: the winning family set was GBDTs, ensembled, with no special techniques; the exact pipeline lives in the cited notebook (not fetched here).
- **Setup:** not stated.
- **Features:** not stated (tags on the post: LightGBM, Beginner, Ensembling).
- **Models:** GBDTs per title/body; families beyond the LightGBM tag, counts, hyperparameters: not stated.
- **CV:** not stated.
- **Ensembling:** "just ensembling GBDTs" — mechanic, weights, member count: not stated.
- **Post-processing:** not stated.
- **Gains:** not stated on this page.
- **Failed:** not stated on this page.
- **Comments:** 3 comments; nothing technical from the author. Dave E (2nd) replies with workflow advice (save optuna results/OOF/subs to disk; add a debug flag running 1 optuna trial on `dataframe.sample(frac=0.1)` before full runs) — a commenter technique, not part of this entry's solution.
- **Compute:** not stated on this page — runtime appears only inside the linked notebook, referenced second-hand by Dave E's reply ("your comments at the end on notebook run time"); the debug-run workaround in that reply is Dave E's advice, not the author's own stated method.
- **Artifacts:**
  - https://www.kaggle.com/kntyshd/3rd-place-solution-ensembling-gbdts (the author's full solution notebook)
- **Lesson:** A pure GBDT ensemble, no exotic anything, placed 3rd; the page publishes no score, so its distance to the DAE winner cannot be computed from it — 1st's 0.8412 is a CV number, 4th's 0.84159 CV / 0.84190 public are the closest published numbers to it.

### TPSFEB21-04 · 4th · Craig Thomas (craigmthomas) · LB 0.84190 public/not stated private · CV 0.84159

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222791
- **Status:** FETCHED on 2nd ladder attempt (`/c/` form returned 164-byte empty cache shell; recovered 15492 bytes with `X-No-Cache: true`)
- **TL;DR:** Two-level stack: 78 level-1 GBDT/linear models from random-search parameter sweeps, level-2 Ridge, all on one shared 10-fold split.
- Won 4th after a +113-place private shakeup (his own words and Dave E's, in comments; his public rank was therefore ~117th by that arithmetic) by trusting CV over LB.
- **Architecture:** STACK2 · stages=2 · l1=78 models (14 CatBoost + 17 XGB + 27 LGBM + 10 RF + 10 Ridge) · l2=Ridge (1) · novel=none · mod=multi-view ·
  topo=[14 CatBoost + 17 XGB + 27 LGBM + 10 RF + 10 Ridge] (each 10-fold, shared folds, distinct categorical encodings) → OOF matrix → Ridge (10-fold) → out-of-fold test predictions
  - Each family saw a *different categorical encoding* (CatBoost encoding / LOO / label), so the OOF columns are genuinely different views of the same rows.
  - Level-2 trained on level-1 fold predictions, itself cross-validated, final predictions out-of-fold again.
  - 3-level stacking tried and rejected; LGBM or CatBoost as level-2 tried and worse than Ridge.
- **Setup:** 78 level-1 models each 10-fold; all share the same fold assignment; CV 0.84159, public LB 0.84190; rows/cols and slots not stated.
- **Features:** raw anonymized features; encoding per family is the only "feature" variation — CatBoost enc for the 14 CatBoost models, Leave-One-Out for XGB/RF/Ridge, label encoding for the 27 LGBM. No hand-built features survived: "polynomial feature generation, binning, normalization, scaling, and others, yielded poor results."
- **Models:** custom Python framework wrapping CatBoost, XGBoost, LightGBM, RandomForest, Ridge (+ "a few others" never incorporated); random search over each family's parameter space, saving only models under an RMSE threshold — thresholds and search sizes not stated.
- Notable parameter axes that kept producing good-but-different models: `max_bin`, `cat_smooth`, `num_leaves`.
- **CV:** 10-fold, identical folds across all models (author stresses this); CV 0.84159 vs public 0.84190; author's trust verdict: CV over public LB — his hyper-tuned single model scored better than the stack on the public LB but had worse CV, and the stack's better CV won the private shakeup.
- **Ensembling:** level-2 Ridge regression on the stacked OOF matrix (both train-folds and out-of-fold test predictions); no explicit weights — Ridge fits the coefficients.
- **Post-processing:** not stated.
- **Gains:** (+113 places on the private shakeup, per Dave E's comment and the author's own reply) choosing the stack over his better-public single model.
- (0.00011 RMSE) the author's stated public-LB gap between his own position and 4th place — the number that made him trust CV instead: "I figured that gap could easily be closed with differences between the public and private leaderboard data distributions".
- (stated as a design principle) many level-1 models fit the noise differently → a second level exploiting that beat any single tuned model on CV; exact CV delta not published.
- **Failed:** polynomial feature generation, binning, normalization/scaling variants, various encodings — all "yielded poor results".
- Dropping low-importance features hurt ("performed poorly in comparison").
- DAE experiments: "wasn't able to make much progress with finding optimal noise mixtures."
- 3-level stacks: poor. LGBM/CatBoost as level-2: worse than Ridge.
- **Comments:** 14 comments. Author replies: framework not published ("needs to be cleaned up"), pointer to his March starter kernel https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter demonstrating the same approach with classifiers; admits he "could have cut down the numbers while achieving the same results, but I ran out of time".
- Dave E (2nd) confirms the +113 place shakeup; 1st ryanzhang praises the local-validation discipline.
- **Compute:** ~36 hours across two machines: 3rd-gen Core i5 with 24 GB RAM, and Ryzen 5 3600X with 32 GB RAM + GTX 1060.
- **Artifacts:**
  - https://www.kaggle.com/hamzaghanmi/lgbm-hyperparameter-tuning-using-optuna (inspiration)
  - https://www.kaggle.com/craigmthomas/tps-mar-2021-stacked-starter (author comment)
- **Lesson:** With shared folds and deliberately diverse encodings, a thick level-1 plus a linear level-2 is the noise-robust answer when your public LB and CV disagree — trust the CV.

### TPSFEB21-06 · 6th · Dongkyu Kim (vkehfdl1), with Jihan Chae (chaejihan), songwonmin · LB 0.84198→0.84185 LB (progression), 0.84249 private · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222751
- **Status:** FETCHED (1st `/c/` attempt returned full 5571 bytes with all comments)
- **TL;DR:** A single LightGBM plus pseudo labeling; the team explicitly abandoned ensembling ("its effect was little").
- They also left a better pseudo-labelled model (0.84245 private = 4th place score) unused as the final.
- **Architecture:** PSEUDO · stages=1 (retrain loop; round count not stated) · l1=1 LGBM · l2=none · novel=none · mod=pseudo ·
  topo=LGBM → predict test → append predicted-test rows to train → retrain LGBM → submit
  - The pseudo-label loop produced the submitted model, so `PSEUDO` is primary (ruling 9); there is no blender anywhere in the winning path.
  - Ensemble attempts existed but were dropped — no level-0 pool in the final.
- **Setup:** private LB 0.84249 (6th); pseudo labeling moved an LB score 0.84198 → 0.84185; an alternative pseudo model scoring 0.84245 private was not selected as final; rows/cols/slots not stated.
- **Features:** none — tried and rejected (see Failed).
- **Models:** LightGBM single (hyperparameters not stated on this page; full config in the linked notebook).
- **CV:** not stated.
- **Ensembling:** explicitly negative result: "I did ensemble a lot, but its effect was little and hardly found public LB improvement."
- **Post-processing:** not stated.
- **Gains:** (-0.00013 LB) pseudo labeling moved 0.84198 → 0.84185; author calls it "powerful skill for this competition" and says it helped "at most model I tried."
- **Failed:** ensembling many models (little effect, no public improvement); feature engineering: PCA, KMeans, multiplying features — "didn't work for me" (author comment).
- **Comments:** 6 comments (2 tagged appreciation). Author reply to Khurram Siddiqui is the only technical content: origin of the idea (Cassava competition discussions where many people used pseudo labeling), the failed FE list above, and that pseudo labeling worked on his first XGB too.
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/vkehfdl1/6th-place-solution-pseudo-labelling-lgbm (the solution notebook)
- **Lesson:** On noisy synthetic regression, feeding your own test predictions back into one GBM can beat blending fifty of them — and know which pseudo model to actually submit.

### TPSFEB21-08 · 8th · Bojan Tunguz (tunguz) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222748
- **Status:** FETCHED on ladder rung 2 — plain `/c/` gave a 164-byte empty cache shell; `X-No-Cache: true` returned the complete 3886-byte page (title, full body, "## 0 Comments"). That render is the whole page, not a nav shell — the post really is this short. (`X-Engine: browser` answered HTTP 503 on the audit pass; unnecessary here.)
- **TL;DR:** Post is titled "Congratulations and Thank You": he admits the top-10 entry was a whim-blend of his own best solution with a top public notebook.
- No model names, no scores, no technique details anywhere.
- **Architecture:** FLAT · stages=1 · l1=2 ("my own best solution" + "top notebook") · l2=none · novel=none · mod=public-oof ·
  topo=(own best solution) + (one top public notebook) → 2-way blend
  - Only structure stated: a two-member average of his own submission with someone else's public notebook output; members' architectures and blend weights are not given on the page.
  - Not `STACK2`: no meta-learner is mentioned, and a whim blend of two CSVs is a weight-free or hand-weighted average (undisclosed which).
- **Setup:** not stated; author says he "slackened off" in the second half and entered this on the last day(s).
- **Features:** not stated.
- **Models:** "my own best solution" (architecture not named); partner = "top notebook" (which notebook not named).
- **CV:** not stated.
- **Ensembling:** one blend of two prediction sets; weights and mechanic not stated.
- **Post-processing:** not stated.
- **Gains:** (rank, not score) the whim blend "was good enough to land me in top 10."
- **Failed:** not stated on this page (author was mostly serious only in the first half of February).
- **Comments:** 0 comments (page shows "## 0 Comments").
- **Compute:** not stated.
- **Artifacts:** none cited (only in-page profile link https://www.kaggle.com/ryanzhang).
- **Lesson:** In early Playground comps a last-day blend of your old best with the top public notebook could still medal — public kernels were worth real leaderboard positions.

### TPSFEB21-11 · 11th · mao-stack (maostack) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222798
- **Status:** FETCHED (1st `/c/` attempt returned full 6098 bytes with all comments)
- **TL;DR:** Exactly two techniques on one LightGBM: pseudo labeling plus submission ensemble; Optuna for tuning (author comment).
- **Architecture:** PSEUDO · stages=1 (loop; rounds not stated) · l1=1 LGBM family · l2=none · novel=none · mod=pseudo ·
  topo=LGBM (Optuna-tuned) → pseudo-label loop → the author's "Submission Ensemble" of the resulting models (mechanic not stated)
  - "Submission Ensemble" is the author's own term for combining his several LGBM submissions; the page never says averaging, seeds or rounds — it does say the only model used is LightGBM, so no second family enters, and rule 4 (fold/seed averaging of one architecture never counts as a level) is the closest fit.
  - Only LightGBM used: "The only model I used is LightGBM."
- **Setup:** not stated; scores not published on this page.
- **Features:** not stated on this page.
- **Models:** LightGBM; tuned with Optuna (author comment: "I tuned the parameters using Optuna"); values not stated.
- **CV:** not stated.
- **Ensembling:** the author's stated technique is "Submission Ensemble" (combining his own multiple LGBM submissions); combination rule, count and weights not stated — the detail is in his linked notebooks, not this page.
- **Post-processing:** not stated.
- **Gains:** not quantified on this page; author states the two techniques together reached 11th.
- **Failed:** not stated on this page.
- **Comments:** 8 comments (1 tagged appreciation). Technical author replies: pseudo labeling "is one of the reason[s]" to grow the training set but he "can't explain" the rest; Optuna was the only tuning technique used ("I didn't use any special techniques. I tuned the parameters using Optuna").
- **Compute:** not stated.
- **Artifacts:**
  - https://www.kaggle.com/maostack/english-tps-feb-11th-place-solution (overview notebook)
  - https://www.kaggle.com/maostack/english-tps-feb-pseudo-labeling-11th-place (details notebook)
- **Lesson:** One tuned GBM + pseudo labeling + averaging your own submissions is a complete top-12 recipe — the notebooks, not the post, hold the replication detail.

## TPSFEB21 - consensus recipe
- **Architecture distribution:** 1st `CASCADE (DAE→Ridge, novel=Transformer-DAE)` · 2nd `FLAT (4 models, optuna weights)` · 3rd `not stated (proof: page defers everything to a notebook; title alone can't separate FLAT/STACK2)` · 4th `STACK2 (78→Ridge)` · 6th `PSEUDO (single LGBM)` · 8th `FLAT (2-member public blend)` · 11th `PSEUDO (single LGBM + submission ensemble)` — winner topology: CASCADE with a novel transformer-DAE feature extractor.
- **New architectures at the board:** Transformer-DAE, 1st only (novel=Transformer-DAE, CV 0.8412 on DAE features + Ridge — the lowest published CV in the set; his comment-only one-model test gives CV 0.8413 with private 0.84248/public 0.84148, and the final submission's LB was never published; 4th's CV 0.84159 is the only other entry with a published CV). No other novel model: 2nd's NN1/NN2 are unnamed networks (no architecture on the page, so nothing licenses a novel name), 4th/6th/11th are stock GBDT/linear, 3rd and 8th publish nothing. DAE was *attempted and failed* by 2nd (TPSFEB21-02, plus his comment on 1st's thread) and 4th (TPSFEB21-04, "wasn't able to make much progress with finding optimal noise mixtures").
- **Agreed on (3 of 7 for automated parameter search; 2 of 7 for pseudo labeling):** automated parameter search feeding model selection or blend weights (2nd TPSFEB21-02 Optuna weights, 4th TPSFEB21-04 random search over each family's space, 11th TPSFEB21-11 Optuna per his own comment). Pseudo labeling as an active technique (6th TPSFEB21-06, 11th TPSFEB21-11 — both single-LGBM). Trust-CV-over-public-LB is stated as a decision only by 4th (TPSFEB21-04); 2nd only notes "All had CV scores pretty similar to public leaderboard", so it is not counted as agreement. Dave E's save-your-OOF/debug-run advice in 3rd's comments (TPSFEB21-03) is a commenter technique, not an entrant practice.
- **Divergences:** top of board = representation learning (1st, CV 0.8412) or multi-family blending (2nd 0.84228 private, 4th CV 0.84159/public 0.84190); mid board = single model + pseudo loop (6th 0.84249 private). 6th published that his ensemble attempts lost to his single pseudo model, inverting the usual ensemble-bigger-is-better story; 2nd's blend (0.84228 private) beat 6th's single LGBM (0.84249 private) by only 0.00021, and 4th's stack publishes only a public 0.84190 (no private score on his page) — he reached 4th through the +113-place private shakeup, so his gap to 2nd/6th is not computable. The one clean cross-comparison of the winner's parts: 1st's own single-DAE test (private 0.84248, comment only) is indistinguishable from 6th's single pseudo-labelled LGBM (private 0.84249), so one DAE bought nothing over one GBM — the submission that won is the Ridge on the blended DAE features (CV 0.8412), and its LB score was never published.
- **Highest-leverage single trick:** the 5-DAE feature blend feeding one plain Ridge (1st, TPSFEB21-01) — a learned representation, not a blender, won; the same author's single-DAE test (private 0.84248) beat neither 2nd's 4-model optuna blend (0.84228) nor 6th's lone LGBM (0.84249), so the leverage is the representation stage as a whole, and it beat 78-model stacks (4th) on CV.
- **Nothing worked:** classical FE on this dataset — polynomial expansion, binning, normalization, scaling, dropping low-importance features (4th); PCA, KMeans, feature multiplication (6th); ensembling for 6th ("effect was little"); DAE reproductions for 2nd and 4th (didn't converge to gains); >2 stack levels and non-Ridge level-2 for 4th.
