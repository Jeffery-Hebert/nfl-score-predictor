# Football Domain Knowledge Reference

Use this to generate football-literate hypotheses about *why* a feature might matter — not as a substitute for testing. Every idea sourced from this file still goes through the standard falsifiable-experiment gate in SKILL.md.

## Personnel Groupings and Formations

- **Personnel packages** (11, 12, 21, etc.) describe RB/TE counts on the field (e.g., "11 personnel" = 1 RB, 1 TE, 3 WR). Offensive personnel tendency correlates with play-calling tendency (pass vs. run) and can proxy for game script.
- **Formation** (shotgun vs. under center, spread vs. bunch) affects play-action rate, time-to-throw, and sack risk. Shotgun-heavy offenses generally pass more and face different pressure profiles.
- **Nickel/Dime defensive packages** (5-6 DBs) signal a defense preparing for pass-heavy opponents; **base 4-3/3-4** signals run-stopping priority. A defense's personnel mix, if available, could in principle refine matchup-specific adjustment beyond simple opponent-adjusted EPA — untested in this project as of the last session.

## Positional Impact Hierarchy (Rough Order of Marginal Value to Score)

1. **Quarterback** — largest single-position driver of offensive output; already indirectly captured via team-level offensive EPA. Direct QB-identity features were tested and found to hurt Linear Regression (see project-context-and-findings.md) — likely due to redundancy with team-level EPA, not because QB doesn't matter.
2. **Offensive line** — pass protection and run blocking; no direct proxy currently in the pipeline. Sack rate and pressure rate (if in play-by-play) are a cheap, untested lever.
3. **Skill positions (WR/RB/TE)** — talent depth matters but is harder to isolate from scheme/QB effects at a team-aggregate level.
4. **Defensive front seven** — pass rush and run stuffing; **defensive backs** — coverage quality, directly tied to opponent CPOE and explosive-play rate allowed.
5. **Special teams** — field position swings (punt/kickoff return, field goal accuracy) are a real but typically small point-swing factor; the project's drive-stats pipeline already captures field-goal and punt rates.

## Injury and Rest Impact

- **Starter injuries at QB** cause the largest single-game variance shock of any position — and are exactly the kind of information this project's models cannot see (no injury/depth-chart feature source is wired in as of the last session). This is the most-flagged, still-unresolved gap in the pipeline.
- **Rest advantage** (bye week, Thursday-to-Sunday short weeks, international-game logistics) has modest, well-documented effects — already captured via `rest_days` in the current feature set.
- **Late-season "nothing to play for" games** (locked seeding, eliminated teams resting starters) are a distinct phenomenon from generic injury risk — this project's finale-week masking fix addresses feature *contamination* from these games but does NOT fix prediction accuracy *on* these games, since the model still lacks any signal about who's actually playing. Treat any Week 17/18 prediction as inherently higher-variance regardless of model improvements elsewhere.

## Morale, Motivation, and Situational Factors

- **Divisional rivalry games** tend to be closer than model-implied point spreads suggest (familiarity, defensive scheme knowledge) — a real, studied effect in football analytics circles, untested in this project.
- **Must-win/elimination games** and **playoff-clinched "letdown" spots** are motivation-driven variance sources distinct from pure rest — a more precise "meaningful game" flag (using standings/clinching logic) is a more surgical alternative to blanket finale-week masking, flagged in project history as a viable but unbuilt follow-up.
- **Coaching change / midseason firing** effects on performance are documented in football analytics literature but have no representation in this pipeline.

## Matchup and Opponent-Adjustment Concepts

- Raw per-team efficiency stats (EPA/play, success rate) are **not** strength-of-schedule adjusted unless explicitly computed that way — this was a real, confirmed gap in this project's original feature set, partially addressed via the opponent-adjusted ratings system (joint ridge regression across team-offense/defense indicators) built and tested in a prior session — see project-context-and-findings.md for its measured (null) result on Linear/Poisson/RF/XGBoost.
- **Explosive plays** (20+ yard gains) disproportionately drive scoring variance relative to their play-count share — a common and well-supported analytics finding, untested as an explicit feature in this pipeline.
