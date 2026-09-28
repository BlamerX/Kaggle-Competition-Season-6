## TPSSEP22 — Tabular Playground Series - Sep 2022
- Task: Tabular (Time Series / Forecasting) | Metric: SMAPE | Problem: Practice forecasting on playground time series
- Kaggle display title: "Tabular Playground Series - Sep 2022"
- Competition: https://www.kaggle.com/c/tabular-playground-series-sep-2022
- Writeups covered: 2 of 2
- Score ladder: 2nd `not stated/not stated (rank only)` · 3rd `not stated/not stated (rank only)`
- Neither page publishes a single SMAPE value: 2nd's page carries no metric number at all (only his
  "IDEA THAT WORKED (1/1300)" heading) and 3rd's carries none either, so all deltas in this file are rank-based, not
  score-based. Index metric (SMAPE, lower better) is used for sign convention only.
- **Series granularity caveat for the whole file:** neither entry states whether its models are **per-series** (one model per
  item-store combination) or **global** (one model with series as a feature). This is a real gap in the record, not a
  `not stated` placeholder — see the per-entry Setup fields.

### TPSSEP22-02 · 2nd · akmalmir (Akmal Azzam o'g'li) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2022/discussion/356746
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 5.7 KB; body + all 8 comments rendered)
- **TL;DR:** He did not build a forecaster; he took the 5 best public notebooks plus 2 of his own and combined them with the
  **Boltzmann Ensemble** weighting scheme from TPS-Jan-2022 (whose author, John Mitchell, had placed 40th there).
- He reports this as his only working idea out of many attempts: "IDEA THAT WORKED (1/1300)".
- **Architecture:** FLAT · stages=1 · l1=7 prediction sets (5 public notebooks + 2 own high-scoring notebooks) · l2=none ·
  novel=Boltzmann-Ensemble · mod=public-oof ·
  topo=7 forecast sets [5 public + 2 own] → Boltzmann partition-function weights over member scores → one blended forecast
  - Why `FLAT` and not `STACK2` (ruling 1): Boltzmann ensemble weights are computed in closed form from each member's
    validation score through a softmax-style partition function — the author of the scheme confirms the mechanism in his
    comment: "it is exactly the same mathematics as the partition functions used in ststistical thermodynamics". No
    estimator is trained on the OOF matrix, so this is weight selection, i.e. `FLAT`.
  - `novel=Boltzmann-Ensemble` names the **combiner**, not a base learner: no novel tabular architecture appears in this
    entry. It is machine-parseable as a bare name per the format's `novel=` rule.
  - The submitted artifact is the blended forecast, so the primary tag is the blend (`FLAT`), not `SINGLE` — 3rd place's GAM
    is one of the members being weighted (he thanks "@paddykb … GAM model" by name).
- **Setup:** rows/cols, series count, horizon, submission slots: not stated.
- **Per-series vs global: not stated** — the page lists member notebooks but never says whether their forecasts come from one
  model per item-store series or from a global model; the Boltzmann weighting granularity (per series vs one weight per
  member overall) is likewise not stated.
- Structural quirk exploited: none named; the stated strategy is reuse of the strongest public work plus a physics-derived
  combiner.
- **Features:** none — this entry builds no features at all; every feature/term decision is inside the borrowed notebooks.
- **Models:** members are named by notebook author, not by configuration:
  - @juhjoo — "TPS(Sep) Ridge+Lasso+Linear+Elastic"
  - @adaubas — "Eda-and-linear-regression-by-cv"
  - @paddykb — "GAM model" (this is TPSSEP22-03's own 3rd-place architecture)
  - @tatudoug — "Genetic_programming"
  - @cabaxiom — "his amazing kernels" (thanked separately; not explicitly identified as one of the 5 members)
  - + "2 own high scoring notebooks" — titles not given on the page.
  - Only 4 members are unambiguously model notebooks; the identity of the 5th public notebook cannot be settled from this
    page, so the 5-vs-4 question is left as stated ambiguity rather than resolved by naming cabaxiom.
- Per-member hyperparameters, forecast horizons, fold counts: not stated (they live in the linked notebooks).
- Boltzmann temperature / energy scaling parameter: not stated (only the source notebook is linked).
- **CV:** no validation scheme is described on the page — splitter, folds, repeats, stratification, seeds: all not stated.
- The member named "…by cv" (@adaubas) implies CV-fitted linear models, but that describes a member, not his own protocol.
- CV-vs-LB gap: not computable (neither side published).
- Author's trust verdict: not stated.
- **Ensembling:** the whole solution. Members (7 prediction sets) → Boltzmann Ensemble weights derived from member scores;
  weights, temperature, and the number of rows/series the weights are applied to: not stated.
- Shared-OOF usage: 5 of the 7 members are community predictions (`mod=public-oof`); their OOF files are not described.
- Source of the combiner: John Mitchell's Boltzmann Ensemble notebook from TPS-Jan-2022, where he placed 40th — the author
  frames the transfer as the idea that worked.
- **Post-processing:** not stated (no clipping, rounding, or level shift is mentioned).
- **Gains:** No SMAPE delta is published, and he publishes no member-by-member scores.
- The only ranking fact available in this file: his combined pool of 7 finished **2nd**, one place ahead of a single member
  of his own pool (paddykb's GAM, 3rd) — which is the strongest evidence on this board that the combiner, not the members,
  bought the place.
- **Failed:** Only one failure is stated, and it is unnamed: the heading "IDEA THAT WORKED (1/1300)" implies ~1299 rejected
  attempts whose content is not described. No model, feature, or blender is named as a dead end on the page.
- **Comments:** 8 comments (counted). Technical content, all from the scheme's author and the topic author:
  - Akmal asks John Mitchell whether the notebook "is from statistical mechanics?"; **John Mitchell** (@jbomitchell, the
    Boltzmann Ensemble's creator) confirms: "Yes, it is exactly the same mathematics as the partition functions used in
    ststistical thermodynamics." — this exchange is what licenses the `FLAT` (closed-form weights) reading above.
  - John Mitchell: "Congratulalations, I'm glad my notebook was helpful."
  - Author replies to @ottpocket ("We missed your awesome code solutions!") and @oscarm524 (80th) with thanks; Will, Oscar
    Aguilar congratulate. No numbers appear in any comment.
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/akmalmir/tps-09-2022-boltzmann-ensemble# (his submission notebook, printed with the trailing
    anchor on the page)
  - https://www.kaggle.com/code/jbomitchell/boltzmann-ensemble-kaggle-stores/notebook?scriptVersionId=86589280 (the original
    Boltzmann Ensemble + explanation)
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/writeups/akmal-azzam-o-g-li-2rd-place-solution-boltzmann-en
    (citation line)
  - https://www.kaggle.com/competitions/33109/images/thumbnail
  - https://www.kaggle.com/akmalmir · https://www.kaggle.com/jbomitchell · https://www.kaggle.com/juhjoo ·
    https://www.kaggle.com/adaubas · https://www.kaggle.com/paddykb · https://www.kaggle.com/tatudoug ·
    https://www.kaggle.com/CABAXIOM · https://www.kaggle.com/ashleychow · https://www.kaggle.com/ottpocket ·
    https://www.kaggle.com/willcramptonn · https://www.kaggle.com/oscarm524
  - Named without URLs in the body: the 5 member notebooks (juhjoo / adaubas / paddykb / tatudoug / cabaxiom) are credited by
    handle only; he thanks the organizers (@ashleychow) and "everyone who shared they notebooks".
- **Lesson:** A closed-form score-based combiner over the community's best forecasts beat a hand-built single model by one
  place — so when a public notebook is already podium-quality, spend the week on how to weight it, not on rebuilding it.

### TPSSEP22-03 · 3rd · paddykb (Patrick Blackwill) · LB not stated/not stated · CV not stated

- **Link:** https://www.kaggle.com/c/tabular-playground-series-sep-2022/discussion/356643
- **Status:** FETCHED (ladder step 1: `/c/` + `X-Return-Format: markdown`, 3.5 KB; body complete, both comments rendered;
  the discussion id resolves to the writeup page `patrick-blackwill-3rd-place-solution`)
- **TL;DR:** One generalized additive model with splines, "No tricks, no ensemble", validated by leaving 3-month chunks out;
  and his claim that ranks 3rd through 7th are all variations of the same GAM.
- The final submission differs from his notebook GAM by exactly one term: a spline added for Christmas-into-New-Year.
- **Architecture:** SINGLE · stages=1 · l1=1 GAM (one model, its coefficients refit per CV fold) · l2=none · novel=none ·
  mod=none ·
  topo=history years → GAM (smooth terms for the recurring patterns + spline for Christmas-to-New-Year) → coefficients
  projected forward onto the forecast window → one submission, no blending
  - Rule 11 does **not** apply to this entry, and the file should not be read as a hand-built deterministic formula: the GAM
    **fits** its smooth terms on the training years, so a fitted estimator is present. The fitted components are named:
    the pattern/behaviour smooth terms learned from the training years, plus the added Christmas→New-Year spline.
  - One stage, no blender ("No tricks, no ensemble" verbatim); the submission is the model's own forecast, so `SINGLE`
    rather than `SEED` — no seed or fold averaging is described.
  - "Projected forward based on strong assumptions about which of those behaviours continues" = the model is fitted on past
    years and extrapolated, not retrained on the forecast window; no test-time loop, so not `PSEUDO`.
- **Setup:** rows/cols, series count, forecast horizon, submission slots: not stated on the page.
- **Per-series vs global: not stated** — the post never says whether the GAM is fitted per item-store series or once with
  series identity as a term; his linked notebook is not quoted here (rule 5: artifacts are not fetched).
- Structural quirk exploited (stated): the year's calendar effects — specifically the **Christmas into New Year** block — are
  real and repeatable, so they deserve their own spline rather than being absorbed by the seasonal terms.
- **Features:** named time-structure terms only, no column list:
  - "the patterns and behaviours from the training years" (i.e. the recurring seasonal/dow/month effects the GAM learns).
  - a spline term added in the final version to "capture _Christmas into New year_".
  - Exact smooth-term definitions, degrees of freedom, basis type, and any lag/date features: not stated (they live in the
    linked notebook).
- **Models:** GAM (library and package not named on the page).
- Author's own description of the model's role: it learns behaviour from the training years and projects it forward under
  stated assumptions about which behaviours continue.
- Hyperparameters, spline degrees, smoothing parameters, seeds: not stated.
- **CV:** "leave 3 month chunks out" — a blocked holdout of whole 3-month blocks, chosen "to get a stableish CV".
- Fold count, repeats, exact block boundaries, seeds: not stated.
- CV score: not published; CV-vs-LB gap not computable.
- Author's trust verdict (implicit): the 3-month-block CV is what he relied on, and he attributes his placement partly to
  luck — "blind luck being on holiday for the last week so I couldn't tinker and do something silly".
- **Ensembling:** none — "No tricks, no ensemble" (verbatim). Number of models averaged: 1.
- Shared-OOF usage: none mentioned on the page (counted).
- **Post-processing:** the only submission-level change is architectural (the extra Christmas→New-Year spline), not a
  calibration/clipping/scaling step. No rounding or threshold stated.
- **Gains:** No numeric delta is published for anything on this page.
- The one stated improvement step: adding the Christmas-into-New-Year spline is "the only difference in the final
  submission" versus his notebook version — magnitude not stated.
- Rank-level claim, flagged by the author as inference not fact: "3rd and I suspect also 4th, 5th, 6th & 7th (based on the
  public score -- apologies if it is not the case) are a variation on this GAM" — i.e. he believes one architecture occupied
  ranks 3-7.
- **Failed:** No failed model, feature, or blender is named on the page (counted body + both comments).
- Self-identified risk instead of a dead end: projecting learned behaviour forward rests on "strong assumptions about which
  of those behaviours continues" — the assumption he says he deliberately stopped testing because he was away.
- **Comments:** 2 comments (counted). No technical content; the exchange is between 2nd and 3rd place:
  - Akmal (@akmalmir, "2nd in this Competition", the author of TPSSEP22-02): "Congratulations Patrick, i really wished you
    to climb top 3 almost a year, now it seems we do it together!"
  - paddykb: "Well played Akmal!"
  - Material use of this thread: it confirms both medals were contested by these two, and that 2nd knows 3rd's method well
    enough to have used his notebook as a pool member (see TPSSEP22-02).
- **Compute:** not stated.
- **Artifacts:** (verbatim; not fetched)
  - https://www.kaggle.com/code/paddykb/tps-2022-09-compare-to-best-public-notebook (the GAM notebook his rank band is based on)
  - https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/writeups/patrick-blackwill-3rd-place-solution
    (citation line)
  - https://www.kaggle.com/competitions/33109/images/thumbnail
  - https://www.kaggle.com/paddykb · https://www.kaggle.com/akmalmir
- **Lesson:** On a calendar-driven forecasting board, one additive model with the right named seasonal terms reaches the
  podium — but you must choose the blocked CV first (whole month chunks), because a random split here validates nothing.

## TPSSEP22 — consensus recipe
- **Architecture distribution:** 2nd `FLAT` (Boltzmann-Ensemble weights over 7 prediction sets: 5 public + 2 own) ·
  3rd `SINGLE` (one GAM, "no tricks, no ensemble").
  - **winner topology: `FLAT`** — 2nd submits a weighted blend; the single model (3rd's GAM) is one of its members.
  - Only two entries exist on this board, so this file supports no wider claim than "1 blend + 1 single split the two medals".
- **New architectures at the board:** `Boltzmann-Ensemble` (2nd place) — a statistical-mechanics partition-function weighting
  of member forecasts, imported from TPS-Jan-2022 where its creator placed 40th; confirmed by the scheme's author in the
  comments. It is the combiner, not a base learner.
  - `GAM` at 3rd is the classic additive-model path (`novel=none`): a generalized additive model is a stock statistical
    choice, and the novelty here is the **Christmas→New-Year spline** term, not the architecture.
  - A genetic-programming notebook is present but only as a **pool member** inside 2nd's blend (@tatudoug), not as a
    standalone architecture; no score is attached to it on the page.
- **Agreed on (2 of 2 — TPSSEP22-02, TPSSEP22-03):** public notebooks are load-bearing, not optional. 2nd's pool is 5 of 7
  community members, including 3rd's GAM; 3rd's own rank band (3rd-7th, his inference from public scores) is described as
  "a variation on this GAM", i.e. variations of one public notebook.
- **Agreed on (2 of 2 — TPSSEP22-02, TPSSEP22-03):** nothing about their numbers is published — zero SMAPE values, zero
  CV scores, zero member scores on either page, so no delta in this file can be quoted and 3rd's/2nd's ranking is the only
  available signal.
- **Agreed on (2 of 2 — TPSSEP22-02, TPSSEP22-03):** no fitted-parameter values appear on either page (2nd: no temperature,
  no weights; 3rd: no smoothing parameters, spline degrees, or term list). Replication requires opening the linked
  notebooks; these entries alone are not sufficient to rebuild the solutions.
- **Not agreed (and this is the actual split of the board):** validation discipline. 3rd states a deliberate blocked scheme
  ("leave 3 month chunks out … to get a stableish CV"); 2nd states no validation scheme at all, and none is inferable from
  his page. 2 of 2 cannot be claimed for any CV practice because only one page discusses CV.
- **Divergences:** the top medal and the next one went to opposite philosophies — 2nd = 7 borrowed/self-made forecast sets
  plus a physics-derived weighting; 3rd = 1 model, explicitly no ensemble. 3rd's claim that the whole 3rd-7th band is the
  same GAM means the marginal value of blending over the best GAM was one place, not a class of solutions.
- Divergence in what they credit: 3rd credits a validation choice and one seasonal term; 2nd credits an idea-transfer
  ("IDEA THAT WORKED (1/1300)") and the community's notebooks, and both attribute their placement partly to luck
  (3rd: "blind luck being on holiday"; 2nd: "1/1300").
- **Highest-leverage single trick:** for the single-model path — the blocked **3-month-chunk CV** plus a dedicated
  Christmas-to-New-Year spline (TPSSEP22-03): the CV is what makes a spline addition a decision rather than leaderboard
  noise. For the blend path — reusing the 5 best public notebooks and computing Boltzmann weights from their scores
  (TPSSEP22-02), which is a two-hour artifact, not a month of modeling.
- **Nothing worked:** nothing is named as failing on either page (both checked: 2nd names no dead model, feature or blender;
  3rd names none either), so this competition contributes no negative knowledge.
  - The closest thing to a recorded failure is 2nd's own framing that roughly 1299 of his 1300 ideas did not work, without
    describing any of them — and 3rd's warning that his forward projection depends on "strong assumptions about which of
    those behaviours continues", which he left untested because he was on holiday.
