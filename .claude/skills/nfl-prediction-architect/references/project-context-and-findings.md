# Project Context and Established Findings

Read this fully before proposing any new feature or architectural change. It exists so past work is never silently repeated or contradicted.

## Current Architecture (As of Last Session)

- **Repo:** `nfl-score-predictor`, Python + pandas/sklearn, walk-forward evaluation harness at `src/validate/walk_forward.py` (now with per-fold progress logging and elapsed-time reporting — added after a session where a Gaussian Process run's duration was hard to judge without it).
- **Production model stack:** `src/models/linear.py`, `poisson_glm.py`, `gaussian_process.py`, feeding `stacking.py` (a Ridge meta-model). `BASE_MODELS = ["linear", "poisson", "gp"]` in `stacking.py` — this was already reduced from 13 original candidate models before the sessions covered here; the operator confirmed this reduction was intentional and evidence-based, though the original comparison evidence itself was not directly reviewed in these sessions.
- **Unused-but-retained models** (in `src/models/unused/`): `random_forest.py`, `xgboost_model.py`, `catboost_model.py`, `lightgbm_model.py`, `logistic.py`, `mlp.py`, `bayesian_hierarchical.py`, `rnn_lstm.py`, `monte_carlo.py`.
- **Feature set (`src/models/common.py` FEATURE_COLS):** 16 columns — home/away pregame team-level rolling EPA, success rate, rest days, prior games played. Built via `src/features/build_rolling_features.py` (EWM, halflife = 17 weeks / 119 days, leakage-safe via `shift(1)`) → `build_game_features.py` → `model_table.parquet`.
- **Finale-week masking:** `is_finale_week()` (week 17 for 2019–2020, week 18 for 2021+) masks finale-week stats from contributing to future EWM averages, preventing rested-starters blowouts from contaminating next season's early-week features. Confirmed via schema check (`game_type` column cleanly separates REG/WC/DIV/CON/SB) and a synthetic leakage test. **Measured effect: no statistically significant change to accuracy** (kept anyway — theoretically sound, zero cost, harmless). Applied in *both* `build_rolling_features.py` and `build_drive_rolling_features.py` — an earlier revision of this file listed the drive-level version as an incomplete fix; that was resolved in commit `2dc3f67`, where the `.where(~finale_mask)` step was added alongside the mask ([`build_drive_rolling_features.py:55`](../../../../src/features/build_drive_rolling_features.py)). Verified against the file on 2026-09-14. Do not re-open.
- **Baseline out-of-sample performance (Linear/Poisson/GP on current feature set):** home_rmse ≈ 9.52–9.54, away_rmse ≈ 9.24–9.25 (walk-forward, min_train_seasons=2, ~1,426 test games spanning 2021–2026).

## Closed Null Results — Do Not Re-Propose Without New Evidence

All of the following were built, leakage-tested, and evaluated via standalone falsifiable experiments (never merged into production feature set) with bootstrap significance testing where the effect size warranted it:

1. **Starting QB identity features** (`build_qb_rolling_features.py`, EWM QB-level EPA/completion%/success rate, merged by identified starter). Tested on Linear Regression: slightly worse RMSE (+0.05), not bootstrapped but consistent with the mechanism below. Leading hypothesis: redundant with team-level EPA (QB performance already substantially drives team EPA), not adding independent information. **87.9% starter-identification coverage** — no live data source exists yet for future/undetermined starters; this remains a blocker for ever using this in production regardless of accuracy findings.

2. **Pass/rush EPA decomposition** (splitting blended `off_epa_per_play`/`def_epa_per_play_allowed` into pass-only/rush-only components, replacing not adding). Tested on Linear, Poisson, RF: consistently worse. **Confirmed statistically real (not noise) via bootstrap on Linear** (home_rmse delta +0.037, 95% CI [+0.0058, +0.0685], excludes zero). Leading hypothesis: splitting a full-game average into pass-only/rush-only subsets roughly halves the effective play-count per component, increasing per-game measurement noise more than the added granularity helps.

3. **CPOE (completion % over expectation)**, added alongside blended EPA. Tested on Linear, Poisson: consistently worse (+0.03 RMSE range). Not bootstrap-tested individually but directionally consistent with #2. Leading hypothesis: CPOE measures accuracy relative to difficulty, not points — may correlate only weakly with the actual prediction target.

4. **Pass/rush split + CPOE combined.** Tested on Linear, Poisson, RF: consistently worse, RF result confirmed as NOT statistically distinguishable from noise via bootstrap (home CI [-0.0577,+0.0440], away CI [-0.0513,+0.0388]).

5. **Opponent-adjusted EPA ratings** (`build_adjusted_ratings.py` — joint ridge regression, team-offense and opponent-defense indicator variables, refit at every walk-forward week-cutoff using only strictly prior games, recency-weighted sample weights matching the existing 17-week halflife). Sanity-checked: team rankings (top/bottom 5 offenses, 2025) were football-plausible. **Tested on Linear, Poisson, RF, XGBoost — no statistically confirmed improvement on any model.** RF's away-score result was the closest to significance (95% CI [-0.0631, +0.0003] — barely misses, upper bound essentially touching zero) but does not clear the pre-registered bar. XGBoost's own baseline (home_rmse 9.593) was notably worse than every other model's baseline tested this session, consistent with (but not proof of) the original decision to exclude it from the production stack.

**Aggregate conclusion from this line of investigation:** five distinct, well-motivated feature ideas, tested across four model architectures with proper leakage safety and statistical rigor, produced zero confirmed improvements and one confirmed (small) regression. This suggests granular decomposition/adjustment of the *existing* box-score/EPA data has limited remaining headroom for these model classes — not that football context doesn't matter, but that these specific cuts of already-available information don't clear the bar. Before retrying variations in this family, have a specific, new reason to expect a different result.

## Real, Unresolved Gaps (Worth Pursuing With New Data or New Direction, Not New Cuts of Old Data)

- **No injury/inactive/depth-chart data source.** This is the single most-cited real gap across every session — the model has no visibility into who is actually playing, which is the dominant driver of the QB-identity and finale-week findings above. Solving this requires a new data source, not new feature engineering on existing play-by-play.
- **Live starting-QB determination for future games** has no operational answer yet (historical backtesting used actual post-hoc starters, which isn't available before a real future game is played).
- **The original evidence for reducing `stacking.py` from 13 to 3 base models has not been directly reviewed** in these sessions; the operator confirmed the decision was intentional, but the underlying comparison data was not re-examined.
- **A more surgical "meaningful game" flag** (using standings/playoff-clinch logic instead of blanket finale-week masking) was proposed as a theoretically more precise alternative but never built — the current blanket week-based mask was chosen as the cheaper first test.

## Operational Notes

- The operator's machine (Jeff's T490s) has limited cores; Gaussian Process walk-forward runs take on the order of 20+ minutes per full pass — always add or confirm progress logging before running anything of that scale, and get explicit confirmation before running GP twice in one script (e.g., base vs. extended feature comparisons).
- `black` is the formatting standard; run `black .` after any file edit before committing.
- Recurring failure pattern to avoid: giving integration code against an *inferred* schema instead of a verified one (caused a real bug in `build_drive_rolling_features.py` once). Always request or read the actual file/schema before writing code that depends on its exact structure.
