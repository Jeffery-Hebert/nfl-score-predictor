# Project Context and Established Findings

Read this fully before proposing any new feature or architectural change. It exists so past work is never silently repeated or contradicted.

## Current Architecture (As of Last Session)

- **Repo:** `nfl-score-predictor`, Python + pandas/sklearn, walk-forward evaluation harness at `src/validate/walk_forward.py` (now with per-fold progress logging and elapsed-time reporting — added after a session where a Gaussian Process run's duration was hard to judge without it).
- **Production model stack:** `src/models/linear.py`, `poisson_glm.py`, `gaussian_process.py`, feeding `stacking.py` (a Ridge meta-model). `BASE_MODELS = ["linear", "poisson", "gp"]` in `stacking.py` — this was already reduced from 13 original candidate models before the sessions covered here; the operator confirmed this reduction was intentional and evidence-based, though the original comparison evidence itself was not directly reviewed in these sessions.
- **Unused-but-retained models** (in `src/models/unused/`): `random_forest.py`, `xgboost_model.py`, `catboost_model.py`, `lightgbm_model.py`, `logistic.py`, `mlp.py`, `bayesian_hierarchical.py`, `rnn_lstm.py`, `monte_carlo.py`.
- **Feature set (`src/models/common.py` FEATURE_COLS):** 16 columns — home/away pregame team-level rolling EPA, success rate, rest days, prior games played. Built via `src/features/build_rolling_features.py` (EWM, halflife = 17 weeks / 119 days, leakage-safe via `shift(1)`) → `build_game_features.py` → `model_table.parquet`.
- **Finale-week masking:** `is_finale_week()` (week 17 for 2019–2020, week 18 for 2021+) masks finale-week stats from contributing to future EWM averages, preventing rested-starters blowouts from contaminating next season's early-week features. Confirmed via schema check (`game_type` column cleanly separates REG/WC/DIV/CON/SB) and a synthetic leakage test. **Measured effect: no statistically significant change to accuracy** (kept anyway — theoretically sound, zero cost, harmless). Applied in *both* `build_rolling_features.py` and `build_drive_rolling_features.py`. **Correction (2026-09-14):** an earlier revision of this file called the drive-level version an "incomplete fix" because the `.where()` step looked absent. A previous pass through this document then over-corrected, claiming commit `2dc3f67` had resolved it. Both were wrong. `2dc3f67` did add `.where(~finale_mask)`, but the same commit made `add_pregame_rolling_drive_features` read `season`/`week`, which `main()` never merged in from schedules (`drive_stats.parquet` carries neither). The stage therefore **crashed with `KeyError: 'season'` on every run from `2dc3f67` onward**, and `team_drive_rolling_features.parquet` on disk was left stale from before that commit. Nothing detected it: there was no leakage test for the drive path, and its only consumer (Monte Carlo) was not being run. Found by `src/features/build_all.py` on first execution. Fixed by adding `season`/`week` to the schedules merge, and covered now by `tests/test_drive_rolling_leakage.py` (4 tests, mutation-verified). Lesson: a code-reading pass confirmed the fix was present but could not confirm the stage could *execute* — running it was what found the bug.
- **Pipeline runner + provenance:** `python -m src.features.build_all` runs all 9 feature stages in verified dependency order (~51s end to end on the operator's machine, from an existing `data/raw/` snapshot) and writes `data/processed/_manifest.json` recording git sha, per-stage timing, row counts and output hashes. `tests/test_pipeline_freshness.py` fails if any output is older than an input or has been modified out of band. Added after `model_table.parquet` was found carrying a timestamp 94 minutes older than the table it derives from. Feature builds are cheap; the expensive step is GP model fitting, not feature construction.
- **Baseline out-of-sample performance (Linear/Poisson/GP on current feature set):** home_rmse ≈ 9.52–9.54, away_rmse ≈ 9.24–9.25 (walk-forward, min_train_seasons=2, ~1,426 test games spanning 2021–2026).

## Closed Null Results — Do Not Re-Propose Without New Evidence

All of the following were built, leakage-tested, and evaluated via standalone falsifiable experiments (never merged into production feature set) with bootstrap significance testing where the effect size warranted it:

1. **Starting QB identity features** (`build_qb_rolling_features.py`, EWM QB-level EPA/completion%/success rate, merged by identified starter). Tested on Linear Regression: slightly worse RMSE (+0.05), not bootstrapped but consistent with the mechanism below. Leading hypothesis: redundant with team-level EPA (QB performance already substantially drives team EPA), not adding independent information. **87.9% starter-identification coverage** — no live data source exists yet for future/undetermined starters; this remains a blocker for ever using this in production regardless of accuracy findings.

2. **Pass/rush EPA decomposition** (splitting blended `off_epa_per_play`/`def_epa_per_play_allowed` into pass-only/rush-only components, replacing not adding). Tested on Linear, Poisson, RF: consistently worse. **Confirmed statistically real (not noise) via bootstrap on Linear** (home_rmse delta +0.037, 95% CI [+0.0058, +0.0685], excludes zero). Leading hypothesis: splitting a full-game average into pass-only/rush-only subsets roughly halves the effective play-count per component, increasing per-game measurement noise more than the added granularity helps.

3. **CPOE (completion % over expectation)**, added alongside blended EPA. Tested on Linear, Poisson: consistently worse (+0.03 RMSE range). Not bootstrap-tested individually but directionally consistent with #2. Leading hypothesis: CPOE measures accuracy relative to difficulty, not points — may correlate only weakly with the actual prediction target.

4. **Pass/rush split + CPOE combined.** Tested on Linear, Poisson, RF: consistently worse, RF result confirmed as NOT statistically distinguishable from noise via bootstrap (home CI [-0.0577,+0.0440], away CI [-0.0513,+0.0388]).

5. **Opponent-adjusted EPA ratings** (`build_adjusted_ratings.py` — joint ridge regression, team-offense and opponent-defense indicator variables, refit at every walk-forward week-cutoff using only strictly prior games, recency-weighted sample weights matching the existing 17-week halflife). Sanity-checked: team rankings (top/bottom 5 offenses, 2025) were football-plausible. **Tested on Linear, Poisson, RF, XGBoost — no statistically confirmed improvement on any model.** RF's away-score result was the closest to significance (95% CI [-0.0631, +0.0003] — barely misses, upper bound essentially touching zero) but does not clear the pre-registered bar. XGBoost's own baseline (home_rmse 9.593) was notably worse than every other model's baseline tested this session, consistent with (but not proof of) the original decision to exclude it from the production stack.

**Aggregate conclusion from this line of investigation:** five distinct, well-motivated feature ideas, tested across four model architectures with proper leakage safety and statistical rigor, produced zero confirmed improvements and one confirmed (small) regression. This suggests granular decomposition/adjustment of the *existing* box-score/EPA data has limited remaining headroom for these model classes — not that football context doesn't matter, but that these specific cuts of already-available information don't clear the bar. Before retrying variations in this family, have a specific, new reason to expect a different result.

## A5 Finding (2026-09-14): The Stack Does Not Earn Its Place

Measured on a clean full-pipeline rebuild, all models re-run on identical
features. Reproduce with `python -m src.experiments.test_stack_vs_base_models`.
Paired bootstrap, 10,000 resamples, resampling **games** (so a resampled game
contributes both its home and away error; within-game error correlation is
0.02). Pooled home+away RMSE.

**Promotion gate -- does each model beat `baseline.py`?** (n=1426, 2021-2026)

| model | delta vs baseline | 95% CI | verdict |
|---|---|---|---|
| poisson | -0.0613 | [-0.1179, -0.0037] | **beats baseline (real)** |
| gp | -0.0593 | [-0.1197, -0.0001] | beats baseline, but the CI upper bound is essentially touching zero |
| linear | -0.0504 | [-0.1189, +0.0175] | **not distinguishable from noise** |

**Does the stack beat its own base models?** (n=1141, stacking's own window)

Ranked pooled RMSE: poisson 9.2748 < gp 9.2819 < **stacking 9.2975** < linear
9.2977 < baseline 9.3285.

| comparison | delta | 95% CI | verdict |
|---|---|---|---|
| stacking vs poisson | +0.0230 | [-0.0115, +0.0586] | noise, but P(stacking better) = only 10.2% |
| stacking vs baseline | -0.0303 | [-0.0993, +0.0398] | **not distinguishable from noise** |

**Conclusion:** Poisson GLM alone clears the project's promotion gate. The
three-model Ridge stack does not -- it cannot be distinguished from the
rule-based baseline, while its own best input can. Stacking is not adding
signal here; it is averaging away the signal Poisson has. Linear does not clear
the gate either, and GP clears it only marginally while costing ~26 minutes per
walk-forward pass (14.18s/fold x 111 folds) versus Poisson's ~10s.

**Recommendation (not yet actioned -- architectural change, needs operator
sign-off):** ship Poisson GLM as the production model and retire the stack, or
at minimum drop Linear and GP from `BASE_MODELS`. This would cut a full
prediction run from ~27 minutes to seconds. Counter-argument worth weighing
before acting: a stack is more robust to one base model degrading over time,
and this is a single 1141-game sample. If the stack is kept, it should be
re-tested with base models that are genuinely decorrelated -- Linear, Poisson
and GP on identical features are near-duplicates of each other, which is the
likely mechanism for the stack's failure to add anything.

**Also settled here:** the long-open gap "the original evidence for reducing
stacking.py from 13 to 3 base models has not been directly reviewed" is now
partly moot -- the question is no longer which 3, but whether the stack should
exist at all.

## Active Accuracy Baseline (2026-09-14)

Every model re-run on the clean A3 rebuild, identical features, one raw
snapshot. Reproduce with `python -m src.validate.model_scoreboard --against baseline`.
Gaussian Process was run separately (26 min); everything else is under 4 minutes.

**Measured runtimes** (operator's 8-core machine, weekly walk-forward, 111 folds):

| model | runtime | | model | runtime |
|---|---|---|---|---|
| baseline | ~1s | | catboost | 27s |
| logistic | 9s | | rnn_lstm | 25s (season-level) |
| lightgbm | 12s | | bayesian_hier | 115s (season-level, ADVI) |
| linear | 7s | | mlp | 207s |
| montecarlo | 10s | | random_forest | 214s |
| poisson | 22s | | **gaussian_process** | **1573s (26 min)** |
| xgboost | 25s | | | |

Feature rebuild is 51s for all 9 stages. GP alone costs more than every other
model combined, by roughly 3x.

**Full set, n=1426 (2021-2026).** margin/total are the betting-relevant
metrics; slope is calibration (1.0 = perfect, >1 = compressed toward the mean).

| model | home | away | mean | margin | total | h_bias | slope | vs baseline (bootstrap) |
|---|---|---|---|---|---|---|---|---|
| poisson | 9.523 | 9.238 | 9.380 | 13.129 | 13.404 | +0.41 | 1.128 | **BETTER** [-0.117, -0.006] |
| gp | 9.516 | 9.250 | 9.383 | 13.140 | 13.400 | +0.09 | 1.051 | **BETTER** [-0.120, -0.000] |
| linear | 9.543 | 9.240 | 9.391 | 13.136 | 13.429 | +0.29 | 1.028 | noise |
| catboost | 9.555 | 9.302 | 9.429 | 13.238 | 13.432 | +0.03 | 1.152 | noise |
| rf | 9.560 | 9.303 | 9.432 | 13.234 | 13.444 | -0.03 | 0.888 | noise |
| **baseline** | 9.597 | 9.286 | 9.442 | 13.258 | 13.450 | -0.47 | 1.226 | -- |
| xgb | 9.593 | 9.315 | 9.454 | 13.264 | 13.477 | +0.00 | 0.897 | noise |
| lgbm | 9.571 | 9.346 | 9.459 | 13.277 | 13.477 | -0.02 | 0.927 | noise |
| logistic | 9.706 | 9.389 | 9.548 | 13.255 | 13.749 | +0.20 | 1.517 | **WORSE** |
| bayesian | 9.784 | 9.370 | 9.577 | 13.216 | 13.870 | -1.97 | 1.526 | **WORSE** |
| montecarlo | 9.804 | 9.377 | 9.590 | 13.660 | 13.472 | -1.31 | 0.992 | **WORSE** |
| rnn | 10.021 | 9.719 | 9.870 | 14.215 | 13.700 | +0.12 | 1.722 | **WORSE** |
| mlp | 10.087 | 9.701 | 9.894 | 13.445 | 14.523 | +1.95 | 0.581 | **WORSE** |

**Conclusions.**

1. Only Poisson and GP clear the promotion gate, and GP's CI upper bound is
   -0.0004 -- it barely clears, for 70x Poisson's runtime. Poisson GLM is the
   defensible production choice.
2. Five models are *significantly worse* than a no-ML rule-based blend:
   logistic, bayesian, montecarlo, rnn, mlp. Retaining them in
   `src/models/unused/` is correct; promoting any of them would be a
   regression, and this is now measured rather than assumed.
3. The GBM trio (rf/xgb/lgbm/catboost) sits indistinguishable from baseline.
   Consistent with the original decision to exclude them.
4. **Calibration finds things RMSE cannot.** Bayesian carries a -1.97 home bias
   and MLP a +1.95 bias, both invisible in a ranking by RMSE. MLP's slope of
   0.581 means wildly over-dispersed predictions; RNN's 1.722 means heavily
   compressed. Crucially the rule-based **baseline itself is compressed
   (slope 1.226)** -- so a model can match it on RMSE while being far more
   useful for derived spreads.
5. **Nuance on the A5 "retire the stack" recommendation.** On the common
   1141-game set Poisson still leads on accuracy (margin 12.921 vs stacking
   12.953), so the accuracy conclusion stands. But stacking has the best
   calibration slope of any model (0.986 vs Poisson's 1.152). If the stack is
   kept, calibration -- not accuracy -- is the defensible reason, and that
   argument should be made explicitly rather than assumed.

**Caveat:** on the 1141-game common set nothing clears the gate, including
Poisson. The significant results above depend on the full 1426-game set. These
effects are near the resolution limit of the available data.

## Data Source Survey (2026-09-15)

What is available free via `nflreadpy`, and what the operator has ruled in or out.

**Ruled OUT by the operator, with reasons worth preserving:**

- **Weather.** Training on ACTUAL weather and predicting on FORECAST weather is
  a train/serve mismatch: the model would learn a relationship with truth and
  then be fed a noisy estimate of it. Confirmed empirically -- `temp` and `wind`
  are 0% populated before kickoff, so there is no forecast in this data at all.
  Only usable if a genuine forecast feed is added, and even then the historical
  training rows would need forecasts, not actuals, to match serve conditions.
  `roof` (dome/outdoors) IS 84% pre-game and is static stadium context rather
  than weather, so it remains available if wanted.
- **Betting market data.** `spread_line`, `total_line`, moneylines and odds are
  all present in schedules.parquet and must NEVER enter training. Permitted use
  is side-by-side comparison only. `src/features/build_advanced_team_stats.py`
  keeps a FORBIDDEN_MARKET_COLS list and prints what it is refusing to read;
  note that pbp also carries `vegas_wp`/`vegas_wpa`, which are market-derived
  and must not be used -- the project uses nflfastR's `wp`, modelled from game
  state (score, time, field position, timeouts) only.
- **Referee assignments.** `load_officials` exists but is post-hoc. Real
  assignments leak Tuesday via unofficial sources (FootballZebras); an official
  pre-game feed does not exist. Weak expected signal for the scraping cost and
  fragility. Not pursued.

**Ruled IN and verified feasible -- injuries.**
`nflreadpy.load_injuries()` covers 2019-2026 (40,386 rows) with official
`report_status` (Out / Doubtful / Questionable), position, player id, and a
`date_modified` timestamp. Leakage-checked against kickoff:

  - 99.94% of rows published BEFORE kickoff
  - median lead time 49.4 hours (the Friday report before a Sunday game)
  - only 22 rows of 34,119 land after kickoff -- filter on
    `date_modified < kickoff` rather than trusting the week key
  - a QB is ruled Out in 240 games, 10.8% of the sample

This closes the long-standing "no injury/inactive data source" gap in this
document. NOT yet built into features -- see the feature-saturation finding
below, which says the estimator must be fixed before more features are added.

**Also available, unexplored:** `load_depth_charts` (a candidate answer to the
open "live starting QB" problem), `load_snap_counts` (would let injuries be
weighted by a player's recent snap share, so a starter out counts and a
4th-stringer does not), `load_nextgen_stats`, `load_pfr_advstats`,
`load_participation` (personnel groupings), `load_ftn_charting`.

## The Model Was Feature-Saturated, Not Feature-Starved (2026-09-15)

The most useful finding of this line of work, and it inverts the working
assumption that the model needed more information.

**What was tried.** `src/features/build_advanced_team_stats.py` builds four
families of metrics from play-by-play already on disk, each with a football
reason rather than a "throw it in" reason:

  PACE/VOLUME     plays, drives, plays per drive, seconds per play, no-huddle
                  rate, pass rate over expected. Points = efficiency x
                  POSSESSIONS, and the production feature set carried no volume
                  term at all -- a genuine structural gap for predicting a
                  score rather than a margin.
  COMPETITIVE     EPA recomputed over plays with 0.20 < wp < 0.80, stripping
                  prevent-defense garbage time out of the efficiency estimate.
                  Uses nflfastR `wp` (game state); never `vegas_wp`.
  TURNOVER LUCK   fumbles forced (persists) separated from fumbles recovered
                  (~coin flip), so a model can weight skill and discount luck.
  SPECIAL TEAMS   field-goal conversion -- absent from the feature set entirely.

**Result: every family made Linear worse, monotonically with feature count.**

  base 9.4110 | +PACE 9.4359 | +COMPETITIVE 9.4362 | +SPECIAL 9.4348
  +TURNOVER 9.4553 | +PACE+COMPETITIVE 9.4603 | +ALL (54 feats) 9.5302

That is an overfitting signature, not evidence the football is wrong.

**Diagnosis.** `src/models/linear.py` was sklearn `LinearRegression` -- ordinary
least squares, NO regularization -- on 18 correlated features with ~1,900
training rows. Meanwhile `PoissonRegressor` defaults to `alpha=1.0` and is
already regularized, and degraded far less under the same features (+PACE
9.4104 vs Linear's 9.4359). That asymmetry is the tell, and it also explains
why Poisson had been quietly beating Linear on the benchmark all along.

**Confirmed by bootstrap** (5,000 resamples over games, pooled home+away):

| variant | delta vs production OLS | 95% CI | verdict |
|---|---|---|---|
| Ridge a=1, base | -0.0047 | [-0.0094, -0.0004] | **BETTER (real)** |
| Ridge a=10, base | -0.0108 | [-0.0232, +0.0014] | noise |
| Ridge a=100, base | -0.0141 | [-0.0336, +0.0051] | noise |
| RidgeCV, base | -0.0067 | [-0.0279, +0.0142] | noise |
| **OLS, base+advanced** | **+0.1193** | **[+0.0585, +0.1800]** | **WORSE (real)** |
| RidgeCV, base+advanced | +0.0217 | [-0.0222, +0.0641] | noise |

Note the statistical subtlety: a=1 clears significance because it barely
changes the predictions, so the paired differences are small and consistent.
a=100 has the larger effect but a wider interval. Significance here measures
reliability of the difference, not which alpha is best.

**Adopted:** ridge in place of OLS in `src/models/linear.py`. Justified on two
grounds -- OLS is the wrong estimator for p=18, n~1900 with correlated
features regardless of scoreboard, and every alpha tested improved on it.

**Deliberately NOT adopted: alpha = 100**, despite scoring best of everything
tried (9.3971, nearly matching Poisson). That number is the test-set optimum,
and taking it would be tuning a hyperparameter on the test set -- exactly the
error walk-forward evaluation exists to prevent. Production uses `RidgeCV` with
a `TimeSeriesSplit` inside each training fold, which scores worse (9.4045) and
is the only defensible choice. If anyone later "improves" this by hardcoding
alpha=100, that is a regression in method even though the number will look
better.

**Rejected: all four advanced-metric families.** Under regularization they stop
being actively harmful but still add nothing (RidgeCV+advanced: +0.0217, CI
spans zero). The information is either already carried by the existing
efficiency features or too noisy at this sample size. The builder is retained
at `src/features/build_advanced_team_stats.py` for future use; it is NOT part
of `build_all.py` and nothing in production reads it.

**Consequence for sequencing.** Injuries are the largest genuine information
gap and are now verified feasible, but adding them to a saturated feature set
would produce a misleading null result and burn the idea. The estimator had to
be fixed first. Any future feature work should be evaluated under the
regularized model, and should prefer COMPRESSING information into few strong
features over adding many raw ones.

**Benchmark after adopting ridge** (n=1426, mean of home/away RMSE):
  baseline 9.4415 | linear 9.4045 (was 9.4110) | poisson 9.3920

## FIRST CONFIRMED FEATURE WIN: Injury Availability (2026-09-15)

After five documented null results, a feature cleared the promotion gate on
both production models. Recorded in full because the *sequencing* mattered as
much as the feature.

**The feature.** `src/features/build_injury_features.py` compresses the
official injury report to ONE column per team:

    injury_impact = sum over unavailable players of
                        positional_value x prior_snap_share

with Questionable at half weight. The idea that makes it work: a starting left
tackle being out and a fourth-string linebacker being out are not the same
event, and snap share is how you tell them apart. A plain count of injured
players cannot, which is likely why "injuries" felt intractable before.

**Result** (paired bootstrap, 5,000 resamples over games, pooled home+away):

| model | base | +injury_impact | delta | 95% CI | verdict |
|---|---|---|---|---|---|
| Linear (RidgeCV) | 9.4045 | 9.3687 | -0.0355 | [-0.0607, -0.0102] | **BETTER** |
| Poisson | 9.3921 | 9.3530 | -0.0388 | [-0.0655, -0.0111] | **BETTER** |

**qb_out was tested and deliberately REJECTED.** It helped on its own but both
CIs spanned zero, and adding it on top of injury_impact made things *worse*
(Poisson -0.0388 -> -0.0312). A starting QB out already dominates the impact
sum via weight 1.00 x ~1.0 snap share, so the extra column is redundant and
costs more in variance than it returns. Do not re-add it.

**Why this worked when everything else failed.** The estimator was fixed first.
Under the old unregularized OLS the model was feature-saturated and degraded
with every addition; injuries would have produced a misleading null and the
idea would have been recorded as dead. The order was: diagnose saturation ->
regularize -> compress the new information to one column -> test.

**Two bugs the builder shipped first, both of which produced plausible output:**

1. The leakage filter dropped rows with a NaT `date_modified` alongside rows
   genuinely published after kickoff. nflverse stopped populating that field in
   2025, so it silently discarded the two most recent seasons -- 6,250 rows --
   while reporting a sensible-looking count. Only 22 rows are actually late.
2. Snap share was joined on `(game_id, gsis_id)`. A player ruled Out has no
   snap row for that game, so the join failed for exactly the players the
   feature exists to measure: 0 of 289 QB-out rows matched and qb_out came out
   identically zero. Fixed with a `merge_asof` on the player's history at
   kickoff (`allow_exact_matches=False`), 97.2% match rate.

Both are covered by `tests/test_injury_leakage.py` (6 tests, mutation-verified:
flipping `allow_exact_matches` back to True fails 3 of them).

**Leakage position.** Injury reports publish a median 49 hours before kickoff.
Snap counts are post-game but are only ever read from PRIOR games via the
as-of join, never from the game being predicted.

## Estimator Repairs (2026-09-15)

Audited every estimator in the project for the defect found in linear.py.

- **linear.py** -- was `LinearRegression`, ordinary least squares, NO
  regularization. Fixed to `RidgeCV` with in-fold `TimeSeriesSplit`.
  9.4110 -> 9.4045. Genuinely broken; genuinely fixed.
- **stacking.py** -- was `Ridge(alpha=1.0)` hardcoded and never validated.
  Fixed to `RidgeCV` with the same in-fold pattern. 9.366/9.242 -> 9.361/9.236.
- **poisson_glm.py** -- `PoissonRegressor` alpha defaulted to 1.0. **Tested
  in-fold selection and REVERTED it**: mean RMSE 9.4175 against 9.3920 for the
  default, at 3.5x the runtime. The inner folds are small enough that the
  selected alpha is noisy and the adaptivity costs more than it buys. The
  default stands, now on evidence. An earlier note in this document called
  Poisson's alpha a defect of the same class as linear.py's missing
  regularization; that was an overstatement. A fixed hyperparameter that has
  been checked is fine. Do NOT "fix" this by copying linear.py's pattern.
- **gaussian_process.py** -- WhiteKernel supplies noise regularization and
  kernel hyperparameters are fitted by marginal likelihood. No defect.
- Shelved models all carry explicit regularization (depth limits, reg_alpha,
  l2_leaf_reg, weight_decay, priors). No defect.

Also: `GridSearchCV` without `n_jobs=-1` ran the inner search serially and took
the Poisson walk-forward from ~1 minute to over 13. Worth remembering before
concluding that in-fold selection is unaffordable.

**Benchmark after all of the above** (n=1426, mean of home/away RMSE):
  baseline 9.4415 | linear 9.3685 | poisson 9.3525

## Real, Unresolved Gaps (Worth Pursuing With New Data or New Direction, Not New Cuts of Old Data)

- **No injury/inactive/depth-chart data source.** This is the single most-cited real gap across every session — the model has no visibility into who is actually playing, which is the dominant driver of the QB-identity and finale-week findings above. Solving this requires a new data source, not new feature engineering on existing play-by-play.
- **Live starting-QB determination for future games** has no operational answer yet (historical backtesting used actual post-hoc starters, which isn't available before a real future game is played).
- **The original evidence for reducing `stacking.py` from 13 to 3 base models has not been directly reviewed** in these sessions; the operator confirmed the decision was intentional, but the underlying comparison data was not re-examined.
- **A more surgical "meaningful game" flag** (using standings/playoff-clinch logic instead of blanket finale-week masking) was proposed as a theoretically more precise alternative but never built — the current blanket week-based mask was chosen as the cheaper first test.

## Operational Notes

- The operator's machine (Jeff's T490s) has limited cores; Gaussian Process walk-forward runs take on the order of 20+ minutes per full pass — always add or confirm progress logging before running anything of that scale, and get explicit confirmation before running GP twice in one script (e.g., base vs. extended feature comparisons).
- `black` is the formatting standard; run `black .` after any file edit before committing.
- Recurring failure pattern to avoid: giving integration code against an *inferred* schema instead of a verified one (caused a real bug in `build_drive_rolling_features.py` once). Always request or read the actual file/schema before writing code that depends on its exact structure.
