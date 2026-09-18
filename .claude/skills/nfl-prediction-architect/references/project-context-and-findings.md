# Project Context and Established Findings

Read this fully before proposing any new feature or architectural change. It exists so past work is never silently repeated or contradicted.

## Current Architecture (As of Last Session)

- **Repo:** `nfl-score-predictor`, Python + pandas/sklearn, walk-forward evaluation harness at `src/validate/walk_forward.py` (now with per-fold progress logging and elapsed-time reporting — added after a session where a Gaussian Process run's duration was hard to judge without it).
- **Production model stack:** `src/models/linear.py`, `poisson_glm.py`, `gaussian_process.py`, feeding `stacking.py` (a Ridge meta-model). `BASE_MODELS = ["linear", "poisson", "gp"]` in `stacking.py` — this was already reduced from 13 original candidate models before the sessions covered here; the operator confirmed this reduction was intentional and evidence-based, though the original comparison evidence itself was not directly reviewed in these sessions.
- **Unused-but-retained models** (in `src/models/unused/`): `random_forest.py`, `xgboost_model.py`, `catboost_model.py`, `lightgbm_model.py`, `logistic.py`, `mlp.py`, `bayesian_hierarchical.py`, `rnn_lstm.py`, `monte_carlo.py`.
- **Feature set (`src/models/common.py` FEATURE_COLS):** 24 columns — home/away pregame team-level rolling scoring, success rate, rest days, prior games played (12 -- blended off/def EPA per play was REMOVED 2026-09-17, superseded by the split); `injury_impact` per side (2); volume-weighted shrunk pass/rush EPA splits, offence and defence, per side (8); `is_neutral_site`/`is_playoff` (2). (This line read "16 columns" until 2026-09-17; it had not been updated when injuries and the split shipped.) Built via `src/features/build_rolling_features.py` (EWM, halflife = 17 weeks / 119 days, leakage-safe via `shift(1)`) → `build_game_features.py` → `model_table.parquet`.
- **Finale-week masking:** `is_finale_week()` (week 17 for 2019–2020, week 18 for 2021+) masks finale-week stats from contributing to future EWM averages, preventing rested-starters blowouts from contaminating next season's early-week features. Confirmed via schema check (`game_type` column cleanly separates REG/WC/DIV/CON/SB) and a synthetic leakage test. **Measured effect: no statistically significant change to accuracy** (kept anyway — theoretically sound, zero cost, harmless). Applied in *both* `build_rolling_features.py` and `build_drive_rolling_features.py`. **Correction (2026-09-14):** an earlier revision of this file called the drive-level version an "incomplete fix" because the `.where()` step looked absent. A previous pass through this document then over-corrected, claiming commit `2dc3f67` had resolved it. Both were wrong. `2dc3f67` did add `.where(~finale_mask)`, but the same commit made `add_pregame_rolling_drive_features` read `season`/`week`, which `main()` never merged in from schedules (`drive_stats.parquet` carries neither). The stage therefore **crashed with `KeyError: 'season'` on every run from `2dc3f67` onward**, and `team_drive_rolling_features.parquet` on disk was left stale from before that commit. Nothing detected it: there was no leakage test for the drive path, and its only consumer (Monte Carlo) was not being run. Found by `src/features/build_all.py` on first execution. Fixed by adding `season`/`week` to the schedules merge, and covered now by `tests/test_drive_rolling_leakage.py` (4 tests, mutation-verified). Lesson: a code-reading pass confirmed the fix was present but could not confirm the stage could *execute* — running it was what found the bug.
- **Pipeline runner + provenance:** `python -m src.features.build_all` runs all 11 feature stages in verified dependency order (~47s end to end on the operator's machine, from an existing `data/raw/` snapshot) and writes `data/processed/_manifest.json` recording git sha, per-stage timing, row counts and output hashes. `tests/test_pipeline_freshness.py` fails if any output is older than an input or has been modified out of band. Added after `model_table.parquet` was found carrying a timestamp 94 minutes older than the table it derives from. Feature builds are cheap; the expensive step is GP model fitting, not feature construction.
- **Baseline out-of-sample performance (Linear/Poisson/GP on current feature set):** home_rmse ≈ 9.52–9.54, away_rmse ≈ 9.24–9.25 (walk-forward, min_train_seasons=2, ~1,426 test games spanning 2021–2026).

## Closed Null Results — Do Not Re-Propose Without New Evidence

All of the following were built, leakage-tested, and evaluated via standalone falsifiable experiments (never merged into production feature set) with bootstrap significance testing where the effect size warranted it:

1. **Starting QB identity features** (`build_qb_rolling_features.py`, EWM QB-level EPA/completion%/success rate, merged by identified starter). Tested on Linear Regression: slightly worse RMSE (+0.05), not bootstrapped but consistent with the mechanism below. Leading hypothesis: redundant with team-level EPA (QB performance already substantially drives team EPA), not adding independent information. **87.9% starter-identification coverage** — no live data source exists yet for future/undetermined starters; this remains a blocker for ever using this in production regardless of accuracy findings.

2. **Pass/rush EPA decomposition** (splitting blended `off_epa_per_play`/`def_epa_per_play_allowed` into pass-only/rush-only components, replacing not adding). Tested on Linear, Poisson, RF: consistently worse. **Confirmed statistically real (not noise) via bootstrap on Linear** (home_rmse delta +0.037, 95% CI [+0.0058, +0.0685], excludes zero). Leading hypothesis: splitting a full-game average into pass-only/rush-only subsets roughly halves the effective play-count per component, increasing per-game measurement noise more than the added granularity helps. **REOPENED AND SUPERSEDED 2026-09-17 — this entry is no longer a closed null result. See "Pass/Rush Split v2" below. The rejection above was measured on unregularized OLS, the estimator this project diagnosed as broken the following day, against a 16-column base with no `injury_impact`. A volume-weighted, shrunk rebuild is now IN PRODUCTION.**

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

Feature rebuild is ~47s for all 11 stages. GP alone costs more than every other
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

## The Week-2 Total Gap: Investigated, Mostly Not A Bug (2026-09-15)

Week 2 2026 predictions averaged 47.35 total points against a market line of
45.31 -- a 2.04-point gap that looked like a calibration fault. It largely is
not. Recorded in full because three plausible fixes were built and all three
failed, and that is expensive to rediscover.

**What the gap is NOT.** The model is well calibrated on totals across the
backtest: +0.34 (linear), +0.31 (poisson). And the Week 2 market total is
unremarkable -- 45.31 against a 45.21 eight-year average. Neither side is doing
anything strange in aggregate.

**What actually happened.** 2026 Week 1 produced 49.44 points a game against
2025's 45.96. The model partially projected that forward off a 16-game sample;
the market did not.

**The mechanism is real.** Team form decays on CALENDAR DAYS at a 17-week
half-life. Across a ~200-day offseason, last season's games fall to
0.5^(200/119) ~= 0.31 weight while a Week 1 game seven days back holds ~0.96 --
so one game outweighs a full prior season roughly 3-4x. The decay treats a
summer as though 200 days of football happened.

**Corroborated in the error analysis:** weeks 1-3 are the worst-calibrated
stretch of the season. Total bias there is +1.14 (linear) and +0.93 (poisson)
against +0.31 overall, and weeks 1-3 RMSE is 9.4876 against 9.3640 for weeks 4+.

**Three fixes tested. All rejected.**

1. `test_offseason_decay.py` -- COMPRESSED offseason (a summer counts as 45
   days of forgetting, not 200). Helps weeks 1-3 slightly (9.4887 -> 9.4655)
   and hurts weeks 4+ more (9.3635 -> 9.4113). Overall **WORSE**: +0.0355
   linear (CI [+0.0032, +0.0682]), +0.0375 poisson (CI [+0.0042, +0.0711]).
   Mechanism of failure: compressing the summer makes last season linger all
   year, not just in September.

2. `test_offseason_decay.py` -- decay by GAMES PLAYED rather than days. Same
   shape, larger damage: +0.0656 linear, +0.0692 poisson, both CIs excluding
   zero. **WORSE.**

3. `test_early_season_bias.py` -- a second bias offset measured on historical
   weeks 1-3 only, applied when predicting an early-season game. Noise on RMSE
   (+0.0028 all games, +0.0148 on weeks 1-3, both CIs spanning zero) and it
   made the weeks 1-3 total bias **worse**, +1.141 -> +1.387. Mechanism of
   failure: historical early-season games are dominated by 2019-2021, the
   empty-stadium era with high away scoring and no home-field edge, so the
   offset is estimated from exactly the wrong football. The thin sample was
   flagged as a risk before running (~336 games, SE ~0.5 against a ~1.0
   effect); it turned out worse than thin, it was biased.

**Conclusion.** The early-season weakness is real and measurable but is not
fixable by re-weighting history -- every scheme that helps September costs more
in October through January. The remaining gap is genuine disagreement between
a model that reacts to Week 1 and a market that does not. Which of them is
right is an empirical question that Week 2's results will start to answer.

**If revisited, do NOT re-try the three above.** The promising untested
direction is shrinking early-season features toward a PRIOR-SEASON team rating
rather than toward recent games -- that keeps team identity while refusing to
over-read one game, and is a different mechanism from anything tried here.

## Pass/Rush Split v2: A Reopened Null Result, Shipped Without Clearing The Gate (2026-09-17)

The first idea in this document's closed-null-results list has been rebuilt and
promoted. Recorded in full, because it is the first time a "closed" finding here
turned out to be an artifact of the measurement rather than of the football --
and because it shipped under an explicit operator override rather than on
evidence.

**Why it was reopened.** Two defects in the original test, both verifiable from
git rather than from memory:

1. `src/experiments/test_pass_rush_split_only.py` uses
   `sklearn.linear_model.LinearRegression` -- ordinary least squares, no
   regularization. It was committed in `81c48d9` (2026-09-14). The feature
   saturation diagnosis landed the NEXT DAY in `3222086` (2026-09-15) and
   concluded that this exact estimator is unfit for p~20 / n~1900 with
   correlated inputs, degrading monotonically as features are added. The
   pass/rush verdict was taken with an instrument this project subsequently
   declared broken, and the experiment added four net columns to it.
2. The base it was compared against was the 16-column set. `injury_impact` did
   not exist yet.

The recorded diagnosis -- that splitting halves the effective play count and
raises measurement noise -- was CORRECT. v1 simply did nothing about it.

**How much noise, exactly.** Measured by variance decomposition over 224
team-seasons (`src/experiments/tune_split_shrinkage.py`):

| split | sigma^2/play | var(true talent) | k (plays) |
|---|---|---|---|
| off pass EPA/play | 2.548 | 0.01123 | 227 |
| off rush EPA/play | 1.029 | 0.00347 | 297 |
| def pass EPA/play allowed | 2.545 | 0.00599 | 425 |
| def rush EPA/play allowed | 1.033 | 0.00192 | 538 |

k is the play count at which a team's own rate and the league prior deserve
equal weight. A team sees ~36 pass and ~26 rush plays a game, so after four
games its own passing number has earned roughly a QUARTER of the weight, and
after a full season under two thirds. v1 handed the raw four-game number to the
model at face value. Defence needs markedly more shrinkage than offence, which
matches the independently measured persistence gap (split-half r 0.33 vs 0.53).

**What v2 does differently** (`src/features/build_split_efficiency.py`):

- **volume weighting** -- the estimate is a recency-weighted rate per PLAY,
  not a recency-weighted mean of per-game rates. A 38-carry game now outweighs a
  9-carry game because it is more evidence. v1's estimator is the special case
  where every play count is 1.
- **empirical-Bayes shrinkage** toward a recency-weighted LEAGUE rate, sized by
  the k values above and by the effective play count actually behind each
  number. The league prior is itself computed strictly from prior games, so it
  tracks the era without seeing the future.
- **the blend is REPLACED, not kept.** This changed within the same day and the
  reversal is worth recording. The split first shipped with blended
  `pregame_off_epa_per_play` / `pregame_def_epa_per_play` retained alongside it,
  on the argument that the blend is the lower-variance measurement (~62 plays
  against ~36 and ~26) and that discarding it was v1's mistake. The operator
  overruled that on duplication grounds: the blend and the split describe the
  same efficiency at different grains, correlating 0.86 (pass) and 0.58 (rush)
  on offence, 0.78 and 0.45 on defence, and an additive model handed two
  descriptions of one quantity splits the credit between them.

  Measured cost of removing it: none, and measured tightly enough to say so.
  Paired bootstrap, blend-present vs blend-absent, both carrying the split:

  | model | delta | 95% CI | P(better) |
  |---|---|---|---|
  | Poisson | -0.0000 | [-0.0136, +0.0134] | 50.0% |
  | RidgeCV | +0.0019 | [-0.0053, +0.0091] | 30.6% |

  These are the two TIGHTEST intervals in the whole experiment -- +/-0.009 on
  Ridge against +/-0.035 for the base comparisons -- because the two feature
  sets differ only by four columns that are 0.86-correlated with columns already
  present. So this is not the usual "cannot resolve it at n=1976" null. It is a
  positive finding that the blend contributed nothing: Poisson's delta is
  exactly zero to four decimals, and Ridge's worst case is under a hundredth of
  a point of RMSE.

  The variance argument that motivated keeping it is now handled where it
  arises -- inside the estimator, by volume weighting and shrinkage -- rather
  than by carrying a coarser second copy in the feature set.
  `tests/test_feature_cols.py` asserts the blend is ABSENT and that each
  side/unit is described by exactly two EPA columns, so it cannot drift back.

**Result** (paired bootstrap, 5,000 resamples over games, pooled home+away,
n=1976 completed games, production estimators):

| comparison | base | v2 | delta | 95% CI | verdict |
|---|---|---|---|---|---|
| RidgeCV, base -> base+split | 9.3861 | 9.3742 | -0.0121 | [-0.0343, +0.0105] | noise, P(better) 86.0% |
| Poisson, base -> base+split | 9.3708 | 9.3668 | -0.0041 | [-0.0352, +0.0269] | noise, P(better) 59.6% |
| OLS, v1's design (blend REPLACED) | 9.3943 | 9.3851 | -0.0094 | [-0.0500, +0.0325] | noise |

**Read this carefully, because it says three separate things.**

1. v2 does NOT clear the promotion gate. Both CIs span zero. It is not a
   confirmed improvement and must not be described as one.
2. v1's confirmed REGRESSION is gone. The sign has flipped from +0.037
   (CI excluding zero) to -0.012. The damage was real and has been removed.
3. The third row is the interesting one: v1's own design, on v1's own
   estimator, no longer reproduces v1's regression either (-0.0094, spans
   zero). So part of the original result was never about the pass/rush split
   at all -- it was the old feature set and the old estimator interacting. The
   C1-C8 fixes, the ridge switch and `injury_impact` changed the ground the
   comparison stood on.

**Promotion status: SHIPPED BY OPERATOR OVERRIDE, not by evidence.** The
operator directed promotion regardless of the bootstrap outcome, and it is in
`FEATURE_COLS`. This is a deliberate exception to the project's own promotion
gate and is logged as such so nobody later mistakes it for a confirmed win. The
defensible reading: it is free (no measured harm, 1.4s of build time), it is
directionally positive on both models, and P(better) = 86% on Ridge is
suggestive but short of the bar.

**If it is ever reassessed**, the question to ask is not "does it help" -- that
has been asked and the answer is "cannot tell at n=1976." Ask instead whether
the MATCHUP interaction helps, which is the part additive models structurally
cannot use: a pass-efficient offence against a pass-vulnerable defence is worth
more than either number alone, and Linear/Poisson can only add them. That is an
untried mechanism, not a re-run of this one.

**Do NOT re-try:** re-adding blended EPA next to the split (tried and removed
2026-09-17, zero measured benefit, and a test now pins it out); adding raw
unshrunk split columns (that is v1 with extra steps); collapsing the four k
constants to one number (defence genuinely needs more shrinkage than offence and
a test asserts it).

## When To Run The Weekly Prediction (2026-09-17)

Discovered by `TestUpcomingWeekIsPredictable` on its first run, which is the
reason that test exists.

**The finding.** `report_status` on the official injury report -- the Out /
Doubtful / Questionable designation that `injury_impact` is built from -- is not
populated until the FRIDAY report before Sunday games. Earlier in the week the
rows exist and carry practice participation, but the status column is mostly
null:

| week | rows | with report_status |
|---|---|---|
| 2026 wk1 (settled) | 182 | 61 |
| 2026 wk2, pulled Thursday | 194 | **6** |
| 2025 wk2 (settled) | 256 | 112 |

So a Wednesday or Thursday prediction runs with `injury_impact` at roughly 5-10%
of its eventual signal. It is the only confirmed feature win in this project and
its only genuine leading indicator, so that is not a small loss.

**Worse, it fails silently.** Every model imputes missing inputs with the
training-fold mean. A feature that is null or zero for the upcoming week does
not raise -- every team receives the same value, the forecast still looks like a
football score, and the feature has simply stopped contributing. On 2026-09-17
`injury_impact` was 0.0 for all 32 week-2 teams because the injury pull predated
that week's report entirely, and nothing in the suite reported it until
`TestUpcomingWeekIsPredictable` was written. `assert_training_is_current` cannot
see this: it validates SCORES, never feature inputs.

**Consequence for operations.** Four cloud routines now run the whole cadence,
and the split between them matters:

| routine | cron (UTC) | local | what it does |
|---|---|---|---|
| full retrain + benchmark | `0 11 * * 3` | Wed 06:00 CT | walk-forward backtest, scoreboard, full test suite. ~1 hour, GP is 30 min of it. Validates; forecasts nothing. |
| prediction, TNF | `0 18 * * 4` | Thu 13:00 CT | pull, rebuild, predict Thursday's game |
| prediction, main slate | `0 11 * * 0` | Sun 06:00 CT | pull, rebuild, predict Sunday |
| prediction, MNF | `0 18 * * 1` | Mon 13:00 CT | pull, rebuild, predict Monday night |

Ids: `trig_01WCWe1gGxdj3HcNkgCeYs93` (Wed), `trig_014S6uMD3zyXm38YoAtqYSBk` (Thu),
`trig_01WhT9vopskUP7qWehv3jmuX` (Sun), `trig_01GabWqUtpb92XaR3irsoHsP` (Mon).

The distinction worth holding onto: **there is no saved model artifact.**
`predict_week` fits from scratch on every run, so each prediction slot is
already a retrain -- it just takes 40 seconds because it fits once rather than
111 times. What Wednesday adds is not a fresher model but VALIDATION: the
walk-forward backtest, the bootstrap against baseline, and the full test suite.
That is the thing that silently went stale, and the thing a bad forecast would
otherwise take weeks to reveal.

Each prediction slot trains on everything available at the moment it runs (the
cutoff is the first PENDING kickoff, not the week's first game), so the Monday
run does see the weekend's results. No intra-week retrain is configured beyond
that because none is needed.

`config.yaml` still documents a Wednesday retrain and the README's weekly loop
still says Tuesday; neither is read by code, and the Wednesday line is now
accidentally correct for the benchmark job.

**Two caveats on that schedule, neither fixed:**

1. **DST.** The cron is UTC, so after the November DST change it fires at 05:00
   local rather than 06:00. Same defect as the comment in
   `.github/workflows/weekly.yml`. Harmless here -- an hour earlier is still
   after the Friday report -- but it will drift.
2. **It marks the week backfilled.** `build_report.week_payload` computes
   `backfilled = generated_at > min(gameday)`, which is WEEK-level. Any week with
   a Thursday night game is therefore flagged backfilled by a Sunday re-run, even
   though 15 of its 16 games have not kicked off. The flag is coarser than the
   thing it is trying to describe. Making it per-game would be the honest fix and
   has not been done.

**Open question worth an experiment, not yet run:** whether a Sunday-morning
prediction actually scores better than a Thursday one. The mechanism is obvious
and the feature is proven, but the size of the gain is unmeasured. It would need
both to be generated for the same weeks and compared over a season.

## Seven Relational / Recency Experiments (2026-09-18)

A single session testing every remaining idea on the "how we represent what we
already know" axis, plus reviving two closed results. One win, six nulls, and
three corrections to my own reasoning that are worth more than most of the
numbers.

All of it is in `src/experiments/`; NOTHING was promoted. 99 new tests.

### The headline table

Control is production's own model_table. Paired bootstrap, 5,000 resamples,
n=1440. Both production estimators, because they disagree in instructive ways.

| arm | Poisson delta | Ridge delta | verdict |
|---|---|---|---|
| exp 8w half-life | +0.0575 | +0.0557 | WORSE, both |
| exp 12w | +0.0187 | +0.0170 | WORSE, both |
| **exp 17w (= production)** | +0.0004 | -0.0002 | control |
| exp 26w | +0.0103 | +0.0103 | noise |
| exp 39w | +0.0444 | +0.0393 | WORSE, both |
| per-stat half-lives | +0.0202 | +0.0134 | noise, both negative |
| two-timescale 3w/26w@40% | +0.0011 | -0.0007 | dead tie |
| two-timescale 2w/34w@30% | +0.0215 | +0.0180 | noise, both negative |
| two-timescale 2.5w/17w@60% | +0.0198 | +0.0157 | noise, both negative |
| + matchup interactions | +0.0151 | +0.0062 | WORSE on Poisson |
| + adjusted ratings v2 | +0.0253 | -0.0008 | WORSE on Poisson |
| + roster continuity | +0.0486 | +0.0120 | WORSE on Poisson |
| + all three | +0.0878 | +0.0110 | WORSE on Poisson |

### 1. The 17-week half-life survives a re-sweep. Close the question.

It was chosen before injuries, before ridge, before shrinkage, and shrinkage in
particular interacts with it. It is still the optimum, on both models, and the
curve is cleanly unimodal. The re-sweep also validated the harness: the
rebuilt-at-17w arm reproduces production to within 0.0004 RMSE, so every other
delta is attributable to its arm rather than to reimplementation drift.

### 2. Per-stat half-lives: no, and the reasoning that motivated them was wrong.

I proposed setting them from split-half r, which ranges 0.23 to 0.70 across the
stats. That is the wrong statistic. **Split-half r conflates trait drift with
measurement noise, and the two want OPPOSITE fixes** -- an unstable trait wants
a short window, a noisy measurement wants a long one, because averaging kills
noise. Shrinkage already handles the noise half.

The statistic that actually sets a half-life is autocorrelation by LAG. Measured:

    off_success_rate   lag1 0.22  lag4 0.22  lag8 0.20  lag16 0.23
    off_pass_epa       lag1 0.12  lag4 0.10  lag8 0.13  lag16 0.09

**Flat.** Within a season these traits do not decay at all; the correlation is
low because one game is a noisy measurement, not because the team changed.
Fitted per-game half-lives land between 30 and 485 GAMES. The real decay is at
the season boundary, where rosters turn over -- and a calendar-day exponential
already applies it (a ~200-day summer costs 0.5^(200/119) = 0.31).

This single measurement explains three separate null results at once: this one,
tune_halflife's rejection of 4 and 8 weeks, and test_offseason_decay's rejection
of compressing the summer.

### 3. Two-timescale decay: no. The best mixture is a dead tie.

The operator's stated shape -- last 2-3 weeks dominant, long tail behind -- is
`two 2.5w/17w@60%` and it is +0.0198 / +0.0157. The best mixture found
(3w/26w@40%) is -0.0007 to +0.0011: a coin flip with production at P(better)
43-55%. A single exponential genuinely cannot be both steep and long-tailed, so
the family was worth testing; it simply has nothing to buy here, for the reason
in #2.

### 4. Anchoring is a no-op for exponentials. I was wrong about this.

I claimed production's `.shift(1)` mis-anchors the weights at the previous
game's date. It does, and it does not matter: for a single exponential,

    w_i = 0.5^((A - t_i)/H) = 0.5^(A/H) * 0.5^(-t_i/H)

and the 0.5^(A/H) factor is identical across observations, so it cancels in a
normalised mean. Moving the anchor rescales every weight equally. It becomes
real only for a MIXTURE, where two components pick up two different constants.

### 5. Matchup interactions: no, but the mechanism was confirmed.

Four products pairing each offence against the defence it will actually face.
The prediction was that they should help RIDGE more than POISSON, because a
Poisson log link ALREADY multiplies feature effects (E[y] = exp(b0)*exp(b1x1)*...)
while Ridge can only add. Measured: Ridge +0.0062, Poisson +0.0151.

Directionally right -- it hurts Ridge less than half as much -- but "less
harmful" is not "helps". Poisson's existing multiplicativity is likely part of
why it leads this benchmark, and that is worth knowing even though the feature
failed.

### 6. Opponent-adjusted ratings v2: the closest thing to a positive.

v1 adjusted a BLEND of all plays, unweighted by volume, tested as a replacement.
v2 adjusts pass and rush separately, weights each row by its play count, and is
tested as an addition. On Ridge it is **-0.0008, P(better) 54%** -- a genuine
dead heat rather than the clear loss v1 was -- and it improves calibration slope
1.115 -> 1.098. On Poisson it is +0.0253, clearly worse.

Sanity-checked: the ratings recover known strengths at r > 0.9 on synthetic
round-robins, and on real data BUF tops adjusted pass offence with MIN and SEA
the best pass defences.

### 7. Roster continuity: no on accuracy, real on calibration.

Three measures from snap counts -- cross-season carryover, within-season line-up
stability, snap-weighted tenure. The ONLY arm carrying genuinely new information
rather than a re-cut of play-by-play.

On Ridge it moves calibration slope **1.115 -> 1.023** (and 1.016 with
everything), at an RMSE cost inside noise (+0.0120, P 23%). Slope above 1 means
compressed predictions, which flattens every derived spread, so that is a real
representational gain RMSE cannot see -- exactly the case for promoting
something that misses the significance bar.

**It does not survive the synthesis.** Poisson already sits at slope 1.008 with
BETTER RMSE (9.367 vs Ridge's 9.376), and every arm makes Poisson worse on both
axes. The calibration gain exists only on a model that is already dominated, so
there is no configuration here that beats production on either measure.

Also note the saturation signature returning: Poisson's slope degrades
monotonically 1.008 -> 0.948 -> 0.869 -> 0.798 as columns are added.

**A leakage trap found in the build, worth recording.** `snap_weighted_tenure`
originally weighted by the CURRENT game's line-up. That both leaks and is
unservable -- on Wednesday nobody knows who will take snaps on Sunday -- and it
is precisely what shelved the QB-identity feature. It now uses the most recent
PRIOR line-up.

### 8. THE ONE WIN: the drive model, revived. 9.608 -> 9.478.

`src/models/unused/monte_carlo.py` was one of five models SIGNIFICANTLY WORSE
than a no-ML rule (+0.1514 vs baseline, CI excluding zero).
`src/experiments/drive_model_v2.py` is **+0.0191, CI [-0.0126, +0.0514] --
statistically indistinguishable from baseline.** Margin RMSE 13.670 -> 13.245,
home bias -1.34 -> -0.13, and calibration slope 1.013 makes it the
second-best-calibrated model in the project behind only Poisson.

Five defects, in order of size:

  1. **The Monte Carlo estimated a closed form.** `draws @ POINTS` then `.mean()`
     approximates n * (p . points), which is exact. 2,000 samples added pure
     noise to every prediction for zero information. The sampler is retained for
     DISTRIBUTIONS, which is a drive model's real advantage, but is out of the
     point prediction's path.
  2. **No home-field advantage existed at all.** Found by measuring v2, not by
     reading v1: drive-outcome rates carry no home/away split, so the model had
     no mechanism to produce one. It predicted a home margin of -0.06 against an
     actual +2.23.
  3. **Arithmetic blending ignored the league baseline.** Replaced with log5
     (odds-ratio), which is what makes it genuinely relational.
  4. **Defensive points went to nobody.** `opp_touchdown` and `safety` scored 0
     for both sides. build_drive_stats fixed this for its own columns and the
     simulator never picked it up.
  5. **Possessions treated as independent** when football alternates them.

**The sequencing lesson is the valuable part.** Fixing defects 1-4 alone made
RMSE WORSE, 9.608 -> 9.756, because they widened a distribution still centred in
the wrong place. v1's predictions had sd 2.60 against an actual 10.04 -- so
compressed that its missing home-field term barely showed up in the error.
Correcting structure without correcting the centre turns four genuine
improvements into a worse number.

The log5 blend also needed shrinking toward the baseline (it AMPLIFIES
deviations, and the inputs are ~11 noisy possessions a game) -- the same
correction that rescued the pass/rush split. The weight is chosen INSIDE each
training fold: the test-set optimum is 0.6 for 9.469, in-fold selection gives
9.478, and taking the 0.6 would be the alpha=100 error this project already
refused once.

Betting is unchanged: ATS 49.4%, identical to v1, still unprofitable. Better
scores did not become better bets.

### Follow-up (same day): four more drive-model ideas, one survived

Each was screened by MEASURING the thing that would have to be true, before
building. Three failed that screen and were not built. Recording the screens,
because they are reusable and two of them corrected claims I had made out loud.

**SURVIVED -- three context corrections the drive model was missing.**
9.478 -> 9.4635, and against baseline it moves from +0.0191 to **+0.0047, CI
[-0.0329, +0.0411]**. Away bias **+1.098 -> +0.094**, calibration slope 0.986.
All three were already settled elsewhere in the project and this model simply
never picked them up:

  - `recent_residual_offset`, the scoring-drift correction every other
    production model applies. The project had already measured it taking the
    others from +0.78 to +0.06; it did the same here.
  - C3 neutral sites: a Super Bowl has no home team, and v2 was both averaging
    those games into the home-field estimate and then applying the estimate to
    them.
  - C5 overtime: halved when fitting the home edge, as elsewhere.

`drive_model_table` carries none of the context columns needed for the last two,
so `load_table()` joins them from schedules rather than changing production.

**FIELD POSITION -- built, tested and measured in the model. Null, but the
first screen of it was bad and the correction matters more than the verdict.**

I screened this out on a regression, called it "~0.2 points of signal", and was
challenged on it. The challenge was right: the screen conflated THREE questions
that have different answers, and reporting only the third as though it settled
all of them was wrong.

  1. DOES FIELD POSITION MATTER PER DRIVE? Enormously. Across deciles, 3.32
     expected points per drive from the best starting position down to 1.36 from
     the worst -- a 1.96 spread, monotone. (The 3.23 figure quoted earlier uses
     fixed-width buckets whose top one holds 730 of 42,917 drives; the decile
     number is the representative one.) Measured WITHIN team-season, so team
     quality cannot confound it, the slope is -0.0355 points per drive per yard.

  2. DO TEAMS DIFFER? Yes, meaningfully. Team-season averages run from starting
     at their own 33.3 (BUF 2024) to their own 25.6 (CAR 2023), a 7.7-yard
     spread worth about 3 points a game.

  3. IS IT KNOWABLE IN ADVANCE? Barely, and this is the only one that bears on a
     forecast:

         prior FP -> future FP         r = +0.108
         prior FP -> future points     r = -0.112
         prior TD rate -> future points r = +0.262

     Field position barely predicts itself. It is produced by opponent punting,
     turnovers, touchbacks and penalties -- largely luck and opponent, not a team
     trait.

Also worth having asked and I had not: is it INCREMENTAL over the drive-outcome
rates the model already carries? Contemporaneously, yes and substantially --
R^2 0.7793 -> 0.8101, +0.0308. Out of sample against PRIOR rates, +0.0014.

**Settled in the model rather than by regression**
(`src/experiments/test_field_position_value.py`): field position enters as an
additive expected-points adjustment at the measured slope, with the offence's
expected start blended from its own prior field position and the opponent's
prior field position allowed. Result: **+0.0147, CI [-0.0039, +0.0343],
P(better) 6%**, and calibration slope degrades 0.986 -> 0.919.

So the verdict stands, but the reason is specifically (3) and nothing else. The
mechanism is real and large; it is simply not knowable far enough ahead.
`src/experiments/field_position.py` and its 15 tests are kept so this does not
need rebuilding, and so the three questions stay separated. One test pins
r=0.108 explicitly and fails if it ever rises above 0.30 -- if that happens, the
model test is worth re-running.

**The extraction trap, which produced a plausible and entirely wrong answer on
my first attempt:** take the first play of a drive and you pick up KICKOFFS at
the 35, which piles half of all drives into one bucket and flattens the curve.
The first SCRIMMAGE play is the drive's real start, identified by `down` being
populated. The wrong version still produced a monotone curve from 4.2 EP down to
1.4 and nothing about it looked wrong. Verified now against nflverse's own
`drive_start_yard_line` string, 100% agreement, asserted on real data.

**REJECTED ON MEASUREMENT -- the possession model.** I had claimed the channel
was dead because predicted drive counts have sd 0.59 against an actual 1.65.
That comparison was wrong: the actual figure includes within-game noise. The
PREDICTABLE spread is the team-season one, sd 0.66 drives (persistence 0.372),
so a predicted sd of 0.59 is about right. The channel is not dead; it is
correctly sized.

**REJECTED ON METHOD -- blending with Poisson.** A fixed-weight probe showed
9.3679 -> 9.3555 at 80/20, holding across 20-40% drive weight. That was
test-set optimism. Selecting the weight INSIDE each training fold gives
**-0.0007, CI [-0.0222, +0.0211], P(better) 53%** -- a dead tie -- and a ridge
meta-model is +0.0068. The reason is visible in the chosen weights: the selector
spreads them from 0.0 to 0.6 across folds rather than converging, so the optimum
is not stable enough to be learned. Error correlation with Poisson is 0.980,
the lowest pairing in the project but still very high.

**The reusable lesson across all four.** The screen that decided each one was
persistence, not effect size. An effect can be large per unit and worthless as a
feature if teams do not differ on it, or if the differences do not repeat. Ask
"how much do teams differ, and does that difference persist" BEFORE building,
and most ideas answer themselves in ten minutes.

### What NOT to re-try

- Any single-exponential half-life other than 17 (swept twice now, unimodal).
- Per-stat half-lives, and more importantly do not motivate anything from
  split-half r -- use autocorrelation by lag.
- Steeper recency in any form. The within-season autocorrelation is flat; there
  is no decay to capture.
- Matchup products on Poisson specifically. Its log link already multiplies.
- Adding any of these to Poisson at all. Every arm made it worse on both RMSE
  and calibration.

### The one genuinely open thread

`drive_model_v2` at 9.478 with slope 1.013 is not a production candidate on its
own -- Poisson is better on both. But it is the only model in the project built
on a DIFFERENT mechanism (drive outcomes rather than per-play EPA) that is now
competitive, and the A5 finding blamed the stack's failure on its base models
being near-duplicates of each other. A stack of Poisson + drive_model_v2 pairs
two genuinely decorrelated predictors for the first time. Untested.

## Real, Unresolved Gaps (Worth Pursuing With New Data or New Direction, Not New Cuts of Old Data)

- **No injury/inactive/depth-chart data source.** This is the single most-cited real gap across every session — the model has no visibility into who is actually playing, which is the dominant driver of the QB-identity and finale-week findings above. Solving this requires a new data source, not new feature engineering on existing play-by-play.
- **Live starting-QB determination for future games** has no operational answer yet (historical backtesting used actual post-hoc starters, which isn't available before a real future game is played).
- **The original evidence for reducing `stacking.py` from 13 to 3 base models has not been directly reviewed** in these sessions; the operator confirmed the decision was intentional, but the underlying comparison data was not re-examined.
- **A more surgical "meaningful game" flag** (using standings/playoff-clinch logic instead of blanket finale-week masking) was proposed as a theoretically more precise alternative but never built — the current blanket week-based mask was chosen as the cheaper first test.

## Operational Notes

- The operator's machine (Jeff's T490s) has limited cores; Gaussian Process walk-forward runs take on the order of 20+ minutes per full pass — always add or confirm progress logging before running anything of that scale, and get explicit confirmation before running GP twice in one script (e.g., base vs. extended feature comparisons).
- `black` is the formatting standard; run `black .` after any file edit before committing.
- Recurring failure pattern to avoid: giving integration code against an *inferred* schema instead of a verified one (caused a real bug in `build_drive_rolling_features.py` once). Always request or read the actual file/schema before writing code that depends on its exact structure.
