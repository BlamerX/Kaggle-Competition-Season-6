## TPSDEC21 — Tabular Playground Series - Dec 2021
- Task: Tabular (Multiclass Classification) | Metric: Categorization Accuracy | Problem: Practice multiclass classification
  (forest cover type, 7 classes)
- Kaggle display title: "Tabular Playground Series - Dec 2021"
- Competition: https://www.kaggle.com/c/tabular-playground-series-dec-2021
- Writeups covered: 1 of 1
- Score ladder: 2nd `0.9707 (his public score mid-competition, explicitly "that time my score was 0.9707"); final private accuracy not
  stated` · the copied public blends he chased sat at `0.9709 -> 0.9711`, i.e. 0.0002-0.0004 above his solo model, and he states pseudo-
  label blends bought a "questionable margin of 0.002 on the public Leaderboard".
- Only one writeup is indexed for this competition, so no agreement claim is possible anywhere in the consensus block (rule 10).

### TPSDEC21-02 · 2nd · sergiosaharovskiy (SSS / Sergey Saharovskiy) · LB not stated/not stated (0.9707 public mid-comp) · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/298304
- **Status:** FETCHED (first try, `/c/` discussion form, 18,910 B; writeup slug `sergey-saharovskiy-2-place-winning-solution`, dated
  May 16, 2023 on the page, citation line dated 2022). Comments needed a second route: the jina render collapsed one thread as
  "5 more replies", so the page was re-read in the real browser (`navigate_page` + `take_snapshot`) to recover those five replies.
- **TL;DR:** A core Keras-style NN soft-voted over its 10 best runs, then blended with six public notebooks minus anything pseudo-labeled -
  2nd place with 9 blend members and no meta-learner.
- Everything he built from the original Covertype dataset failed: "scored extremely well on CV but failed on the test data".
- His own summary of why the medal came: "if I did not avoid 'blend path until the end', I would not make it to the top. The core model
  really made the difference."
- **Architecture:** FLAT · stages=1 · l1=9 members (3 × his own core NN at 0.9707, 1 × mlanhenke's Keras NN, 3 × kaaveland's XGBoost,
  2 × ambrosm's "Eliminate cover type 4") · l2=none (no meta-learner named anywhere on the page) · novel=none · mod=public-oof ·
  topo=core NN ×10 best runs -> soft voting (breakthrough 1) -> same architecture, different batch_sizes -> soft voting (breakthrough 2)
  -> 3 of those models + 6 public-notebook models → flat blend -> submission
  - The submission is an average of member probabilities: he defines "stacked" as "a.k.a soft voting" for his own pool, and the final step
    is described as a blend of member models with no fitted blender, so `FLAT`, not `STACK2` (ruling 1).
  - The 10-performer and batch-size soft votes are averaging inside ONE architecture, so per ruling 4 they never count as a level -
    `stages=1` even though two averaging steps precede the blend.
  - `mod=public-oof`: 6 of the 9 members are other people's public notebooks, taken as-is. He states "There is no public notebook besides
    the one I posted. There are small changes to it which can reconstruct the final solution."
  - Not `PSEUDO`: he tried pseudo-labeling with his own models, it degraded them, and he stripped pseudo-label material out of the blends he
    copied ("I removed anything related to pseudo labels").
- **Setup:** rows/cols and split sizes: not stated. Classes: forest cover type, with `Soil_Type*` one-hot columns and an `Id` column named
  via his drop list. Slots: not stated.
- Structural quirk he exploited: the synthetic test set is close enough to public blends that copying a proven blend was the front-runner
  strategy - "I looked closely at 0.9709->0.9711 opened blends" - and he beat them by removing their pseudo-label component.
- His second structural observation: "30 people on top of me copied the same notebook" and later "another tens of people ahead of me with
  score 0.9709->0.9711 copying the same blend".
- **Features:** exactly three survived his FE program, all named in the body:
  - `Euclidian distance` (applied before chryzal's FE notebook appeared - "Before that I had already applied Euclidian distance for my model").
  - `Manhattan distance`.
  - `Aspect fix` (correcting the `Aspect` column; the fix's formula is not given on the page).
  - Aggregation stance: "it seemed that feature engineering was justified, but I could not agree with @chryzal on aggregated features" -
    he rejected them by argument; the page does not say he benchmarked them.
  - Dropped columns, verbatim: `col_drop=['Id', 'Soil_Type7', 'Soil_Type15']` - stated as a parenthetical on the soft-voting breakthrough,
    with no reason given on the page.
  - Failed FE families (see Failed): feature clipping, soil masks, and all "bring it closer to the original dataset" runs.
  - Target encoding, frequency encoding, GP/generated features: not stated. Source silent.
  - Comment admission that bounds the whole FE effort: "I put a lot of efforts into the original dataset ... Unfortunately, nothing worked
    except couple added features. I believe it is what happens when you work with generated data."
- **Models:** core model = a deep NN whose architecture exists only as a linked screenshot
  (https://monosnap.com/file/U1Jl800CSowTfIJytaI7ZB9TNWK2ze); nothing about layer counts or widths is written on the page.
- Named facts about it: `SeLU` activation ("I did not believe in the SeLU from the very beginning"), varied `batch_sizes` used as the
  diversity device, and his logged knobs from the comment reply: `lr, wd, plateau_factor, plateau_patience, batch, epochs, early_stopping`
  (i.e. weight decay + a ReduceLROnPlateau-style scheduler + early stopping). Values: not stated.
- His verdict on model choice: "I tried different models architectures and got convinced that almost all of my models converged
  identically."
- Pool construction: two selection rounds - (1) soft voting of his 10 best performers, (2) same architecture with different `batch_sizes`,
  soft voting of the best performers again. Fold count per model: not stated, though his logger records `fold` and `nfold`.
- Blend members taken from public notebooks (verbatim from the body):
  - 3 models of 0.9707 (his best performer)
  - 1 model @mlanhenke - `tps-12-g-res-variable-selection-nn-keras`
  - 3 models @kaaveland - `TPS202112 - Reasonable XGBoost model`
  - 2 models @ambrosm - `TPSDEC21-12 Eliminate cover type 4!`
- Library versions, seeds (beyond the `seed` logger field), XGBoost hyperparameters of the borrowed members: not stated.
- **CV:** fold structure implied by his logger header (`fold`, `nfold`, `train_loss/valid_loss`, `train_acc/valid_acc`); the number of folds
  and repeats is not stated, and no CV accuracy number appears on the page.
- CV-vs-LB gap: he reports a sign flip rather than a number - the original-dataset feature runs "scored extremely well on CV but failed on
  the test data", which is why he stopped trusting his own FE CV signal.
- Trust verdict, paraphrased from the same line plus the comment: CV was reliable for model selection inside one architecture (his soft-vote
  breakthroughs) and unreliable for feature provenance borrowed from the real Covertype data.
- Public score he does state: 0.9707 while the copied blends were at 0.9709-0.9711.
- **Ensembling:** flat/soft voting, twice over, no weights published.
  - Inner: 10 best runs of the core NN averaged; then again with different batch sizes.
  - Outer (the submission): 3 own models + 6 external notebook models. Per-member weights: not stated. Whether the 9 are equally weighted:
    not stated.
  - Shared-OOF usage: 6 of 9 members come from other people's public notebooks; he adds "small changes" to his own posted notebook to
    reconstruct the final solution rather than publishing the blend code.
  - Deliberate exclusion criterion for the pool: anything pseudo-label-derived was removed, which is the one design decision he quantifies
    by consequence (his score went from 0.9707 toward the 0.9709-0.9711 band, final number unreported).
- **Post-processing:** not stated on this page. One blend member carries a class-level rule in its own title
  (`TPSDEC21-12 Eliminate cover type 4!`), but the author never describes what that notebook does, so no post-processing step can be
  credited to him from this source.
- **Gains:** two qualitative "breakthroughs", both unnumbered:
  1. Soft voting of his 10 best NN performers: "and it was a breakthrough".
  2. Same architecture, different `batch_sizes`, soft voting of the best performers: "another breakthrough".
- Final blend closed the 0.0002-0.0004 gap between his 0.9707 and the copied 0.9709-0.9711 blends (arithmetic on his stated numbers; his
  finished score is not printed).
- Value of pseudo-label blends he refused: he prices them at "a questionable margin of 0.002 on the public Leaderboard" in exchange for
  adding "4%+ mislabeled data" to train.
- Feature-side: only 3 of his FE experiments survived to add anything (Euclidian distance, Manhattan distance, Aspect fix); none is
  quantified.
- **Failed:** Highest-value anti-knowledge in the file, and almost all of it is FE:
  - "Bringing the competition dataset as closer to the original dataset" (UCI Covertype): "scored extremely well on CV but failed on the test
    data" - "Tens of experiments failed."
  - Extra feature clipping: failed.
  - Soil masks (real-world logic that some cover types never grow on particular soils): failed.
  - Aggregated features in the style of @chryzal's FE notebook: he "could not agree" with them; no benchmark claimed.
  - Pseudo-labeling with his own models: "I quickly realized that my models performance only deteriorates from it. Since it worked for
    someone, but proved not working for me, I decided to stay away from it."
  - Original-dataset domain knowledge as an edge, overall: "nothing worked except couple added features".
  - Commenter-side corroboration of the pseudo-label failure (@thariqnugrohotomo, 396th, on this page): a train-split "virtual-test-set"
    pseudo-label loop showed "an increase of 0.001, but when submitting the LB score is dropping instead".
- **Comments:** page header "32 Comments". The proxy render carries 25 named comment blocks (6 of them Topic Author replies), 1 block whose
  account name is gone ("Congrats! You got some incredible insights in this write-up!"), 1 "This comment has been deleted." position and a
  collapsed marker reading "5 more replies"; the browser re-read expands that thread and adds those 5 blocks, of which 1 is a Topic Author
  reply - 25 + 1 + 1 + 5 = the header's 32, every position accounted for on the two routes. All author-only technical
  content, from both routes:
  - Experiment bookkeeping (reply to @upgradedtotoro, 595th): the experiment tree is drawn with the python `graphviz` library; logs live in
    spreadsheets; best and last model per fold are saved in a per-experiment directory; the logger header for this competition was
    `[fold, run_epoch, train_loss, valid_loss, train_acc, valid_acc, seed, nfold, lr, wd, plateau_factor, plateau_patience, batch, epochs,
    early_stopping]` - this is the only enumeration of his tuning knobs anywhere in the file.
  - Provenance of that workflow (reply to @cv13j0, 104th, from the collapsed thread): the tree-of-experiments idea is inspired by
    @philippsinger, whose YouTube interview he links at https://youtu.be/OenmJTdF0-M?t=1395.
  - @kaggleqrdl (170th), also in the recovered thread: the real cost of this style is deciding "when to parameterize via new function
    parameters so you can re-execute prior experiments without changing code and when to just fiddle with parameters directly", and he
    biases toward parameterizing for optuna compatibility.
  - Public notebook policy (reply to @kaggleqrdl): "There is no public notebook besides the one I posted. There are small changes to it which
    can reconstruct the final solution." - i.e. the 9-member blend is not fully published.
  - His counter-argument to pseudo-labeling (same reply): "how bad we are ready to compromise our train set (add 4%+ mislabeled data) in
    order to have a questionable margin of 0.002 on the public Leaderboard. What if it is not pseudolabels themselves, but great combination
    of models made the difference for the author?"
  - Credit correction from a blend member's author: @kaaveland (129th) - "You have linked to my xgboost notebook, but credited remekkinas";
    SSS apologizes and updates the post, adding "Your xgboost model has made the difference. It is exceptionally elegant." So the XGBoost
    block is the credited-strongest external member.
  - @kaaveland on why copying wins TPS: his own booster notebook "scored top 10" on a desktop machine and he hesitated to publish it, because
    "you can be sure that if you aren't beating it, there are going to be anywhere from 10 to 100 people ahead of you who might not even
    have looked at the problem."
  - @kaggleqrdl's probing aside, relevant to the sibling Nov-2021 file: leaderboard probing of uncertain labels ("just submit -1 for all
    classes in doubt and probe the leaderboard") plus his papers thread at
    https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/297960.
  - Rank badges of commenters on this page (context): 6th (Andrew Schleiss), 24th, 48th, 94th (Adam Wurdits), 95th, 104th,
    129th (kaaveland), 170th, 189th, 348th, 396th, 595th.
- **Compute:** no wall-clock, GPU model, RAM, or cost is stated. Implicit: 10+ NN runs × 2 selection rounds × per-fold checkpoints, run
  outside Kaggle notebooks (he saves models to experiment directories and logs to spreadsheets); the borrowed XGBoost block was itself
  written on a desktop machine per @kaaveland's comment. He abandoned the follow-up: "A job offer has come in and drained me of energy".
- **Artifacts:** (verbatim, not fetched)
  - https://archive.ics.uci.edu/ml/datasets/Covertype (the original dataset he worked against)
  - https://www.kaggle.com/chryzal/features-engineering-for-you (the FE notebook he partly disagreed with)
  - https://monosnap.com/file/U1Jl800CSowTfIJytaI7ZB9TNWK2ze (screenshot of his best-performing architecture)
  - https://monosnap.com/file/vr5d5J8Ho4YazWcKV2m3ex6tqlADiH ("Tree of experiment small version")
  - https://www.kaggle.com/mlanhenke/tps-12-g-res-variable-selection-nn-keras (1 blend member)
  - https://www.kaggle.com/kaaveland/tps202112-reasonable-xgboost-model (3 blend members)
  - https://www.kaggle.com/ambrosm/tpsdec21-12-eliminate-cover-type-4 (2 blend members)
  - http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html (in @thariqnugrohotomo's comment)
  - https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/297960 (in @kaggleqrdl's comment)
  - https://youtu.be/OenmJTdF0-M?t=1395 (philippsinger workflow interview, in the browser-recovered collapsed thread)
  - https://www.kaggle.com/sergiosaharovskiy, https://www.kaggle.com/chryzal, https://www.kaggle.com/mlanhenke,
    https://www.kaggle.com/kaaveland, https://www.kaggle.com/ambrosm, https://www.kaggle.com/philippsinger,
    https://www.kaggle.com/aayushpoddar, https://www.kaggle.com/upgradedtotoro, https://www.kaggle.com/cv13j0,
    https://www.kaggle.com/adamwurdits, https://www.kaggle.com/dryanma, https://www.kaggle.com/datastrophy,
    https://www.kaggle.com/samuelcortinhas, https://www.kaggle.com/saurabhbagchi, https://www.kaggle.com/sandhyakrishnan02,
    https://www.kaggle.com/thariqnugrohotomo, https://www.kaggle.com/balavashan, https://www.kaggle.com/satoshiss,
    https://www.kaggle.com/sisharaneranjana, https://www.kaggle.com/kaggleqrdl, https://www.kaggle.com/slythe,
    https://www.kaggle.com/jack232126
  - https://www.kaggle.com/competitions/tabular-playground-series-dec-2021
  - https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/writeups/sergey-saharovskiy-2-place-winning-solution
    (citation line)
  - https://www.kaggle.com/competitions/28012/images/header (competition header image; no `/images/thumbnail` URL appears in any render of
    this page)
  - https://storage.googleapis.com/kaggle-avatars/thumbnails/621084-kg.jpeg, .../6259210-kg.png, .../5488710-kg.jpg, .../9052057-kg.jpg
    (avatar assets surfaced by the browser render)
- **Lesson:** On a fully synthetic multiclass board the winning move is to abandon domain-motivated feature engineering after CV lies to you
  twice, keep one architecture and average it over batch sizes, and then blend six public notebooks minus their pseudo-label component.

## TPSDEC21 — consensus recipe
Single-entry board: rule 10 forbids any "agreed on (n of 1)" claim, so every line below names the one entry that states it and no line
asserts corroboration.
- **Architecture distribution:** 2nd `FLAT` (9 blend members: 3 own core-NN soft-votes + 6 public-notebook models; no meta-learner, no
  fitted weights on the page).
  - **winner topology available from this file: `FLAT`** — with only one indexed writeup, the board's 1st place is unrecorded here, so no
    rank-vs-architecture inference about the winner is possible from this file.
  - Depth was never on the table in this entry: two averaging steps precede the blend, but both are inside a single architecture (ruling 4),
    so the record is a one-stage flat ensemble of nine members.
- **New architectures at the board:** none. The only models named are his own `SeLU`-activated deep NN (architecture published as a
  screenshot, not as a named design), @mlanhenke's Keras NN, @kaaveland's XGBoost, and @ambrosm's cover-type-4 notebook. `novel=none`.
  - The interesting 2021-era content is the opposite of a new model: the 9-member blend is a plain average, and every novel-sounding device
    he tried (domain-driven soil masks, original-dataset alignment, pseudo-labeling) is reported as failing.
- **What the one entry states about features (TPSDEC21-02):** 3 survivors named - `Euclidian distance`, `Manhattan distance`, `Aspect fix`;
  3 dropped columns named verbatim - `col_drop=['Id', 'Soil_Type7', 'Soil_Type15']`; aggregated features rejected by argument against
  @chryzal's notebook; no target encoding, frequency encoding, or generated feature appears anywhere in the
  set. Exact formulas for the two distances and the Aspect fix: source silent.
- **What the one entry states about validation (TPSDEC21-02):** fold-based training with `train_acc/valid_acc` logging (`nfold` value not
  published), and a documented CV-vs-test sign flip on borrowed-domain features.
- **What the one entry states about ensembling (TPSDEC21-02):** soft voting twice inside the core NN (10 best performers, then
  different-`batch_size` runs), then a flat blend of 9 members with no published weights, minus anything pseudo-label-derived.
- **Divergences:** unmeasurable on a one-entry board. The only contrast available is internal to the entry: the blend members he copied
  (0.9709-0.9711, pseudo-label based) beat his solo model (0.9707) until he stripped pseudo-labels out and kept the model diversity -
  2nd place. External corroboration is a commenter's own experiment, not an entry: +0.001 local pseudo-label gain that dropped on LB
  (@thariqnugrohotomo, 396th).
- **Highest-leverage single trick:** average the same architecture across batch sizes and keep only the best performers (TPSDEC21-02,
  "Same architecture but different batch_sizes -> soft voting of the best performers ... another breakthrough") - the entry's second
  named breakthrough, obtained without adding a single feature or a second model family.
- **Nothing worked (per the single entry):** original-Covertype-domain feature work (clipping, soil masks, dataset alignment - "Tens of
  experiments failed", CV rose while the test score fell), and pseudo-labeling with his own models ("my models performance only deteriorates
  from it").
- **Nothing worked (per that entry's comments, attributed to commenters and therefore not counted as agreement):** a virtual-test-set
  pseudo-label loop (+0.001 locally, LB dropped, 396th); publishing nothing and copying (kaaveland's 10-to-100-people-ahead observation is a
  warning about the copy-the-notebook meta-game, which the author himself documents as the reason he stopped building solo).
