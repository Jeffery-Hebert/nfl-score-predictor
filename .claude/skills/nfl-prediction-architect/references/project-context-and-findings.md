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

## Real, Unresolved Gaps (Worth Pursuing With New Data or New Direction, Not New Cuts of Old Data)

- **No injury/inactive/depth-chart data source.** This is the single most-cited real gap across every session — the model has no visibility into who is actually playing, which is the dominant driver of the QB-identity and finale-week findings above. Solving this requires a new data source, not new feature engineering on existing play-by-play.
- **Live starting-QB determination for future games** has no operational answer yet (historical backtesting used actual post-hoc starters, which isn't available before a real future game is played).
- **The original evidence for reducing `stacking.py` from 13 to 3 base models has not been directly reviewed** in these sessions; the operator confirmed the decision was intentional, but the underlying comparison data was not re-examined.
- **A more surgical "meaningful game" flag** (using standings/playoff-clinch logic instead of blanket finale-week masking) was proposed as a theoretically more precise alternative but never built — the current blanket week-based mask was chosen as the cheaper first test.

## Operational Notes

- The operator's machine (Jeff's T490s) has limited cores; Gaussian Process walk-forward runs take on the order of 20+ minutes per full pass — always add or confirm progress logging before running anything of that scale, and get explicit confirmation before running GP twice in one script (e.g., base vs. extended feature comparisons).
- `black` is the formatting standard; run `black .` after any file edit before committing.
- Recurring failure pattern to avoid: giving integration code against an *inferred* schema instead of a verified one (caused a real bug in `build_drive_rolling_features.py` once). Always request or read the actual file/schema before writing code that depends on its exact structure.
