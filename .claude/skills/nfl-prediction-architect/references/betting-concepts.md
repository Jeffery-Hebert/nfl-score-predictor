# Betting Concepts Reference

Use precisely. These terms have exact technical definitions — do not use them loosely, and do not conflate predictive accuracy with betting profitability (see SKILL.md's Betting/Prediction Discipline section).

## Core Definitions

- **Spread (point spread):** the handicap a sportsbook assigns to make a game roughly a 50/50 proposition against the spread. A team favored by -7 must win by more than 7 for a spread bet on them to win.
- **Totals (Over/Under, O/U):** a bet on whether the combined final score of both teams is above or below a posted number. This project's canonical score-prediction target directly supports totals evaluation (home_pred + away_pred vs. the line) without redesign.
- **Moneyline:** a bet on which team wins outright, priced via odds (e.g., -150 favorite, +130 underdog) rather than a point handicap.
- **Implied probability:** the win probability embedded in a moneyline or spread price, derived by converting odds to probability (e.g., -150 implies roughly 60% implied probability, before removing the sportsbook's built-in margin/vig).
- **True probability:** the model's (or reality's) actual estimated probability of an outcome, independent of the market's price. The core betting-edge question is always: does true probability exceed implied probability by enough to be profitable after the vig?
- **Expected Value (EV):** for a bet, EV = (true probability × payout if win) − (probability of loss × stake). A bet with positive EV is theoretically profitable in the long run if the true probability estimate is accurate — which is exactly why prediction quality and calibration must be validated before any EV claim is made.
- **Closing Line Value (CLV):** the difference between the odds/line you bet at and the final closing line right before the game starts. Consistently beating the closing line (getting a better price than the market eventually settles on) is widely regarded in sharp-betting circles as a stronger indicator of genuine predictive skill than short-term win/loss record, because it's less exposed to game-to-game variance.
- **Kelly Criterion:** a bet-sizing formula that computes the optimal fraction of bankroll to wager given an edge and odds, to maximize long-run bankroll growth while accounting for variance. Formula: f* = (bp − q) / b, where b = decimal odds − 1, p = true win probability, q = 1 − p. Full Kelly is aggressive and high-variance in practice; fractional Kelly (e.g., half-Kelly) is the common real-world adjustment for model uncertainty.
- **Wong Teasers:** a specific teaser-betting strategy (named after Stanford Wong) that moves point spreads by a fixed number of points (commonly 6) across two games, specifically targeting spread crossings of the "key numbers" 3 and 7 (the most common final-margin values in NFL games), historically considered a favorable-EV strategy under certain market conditions. Relevant to this project only if/when it extends to teaser-specific evaluation — not currently in scope.
- **Success Rate:** in analytics (distinct from betting "bet success rate"), typically refers to the play-level efficiency metric already used in this project's feature set (`off_success_rate`/`def_success_rate_allowed`) — a play is "successful" if it gains a threshold fraction of yards-to-go (commonly 40% on 1st down, 60% on 2nd, 100% on 3rd/4th). Do not confuse this analytics term with a betting "hit rate."

## How These Concepts Interact With This Project's Architecture

1. **Prediction quality** — the model's raw score/RMSE/calibration performance (what most of this project's work to date has focused on).
2. **Calibration** — whether predicted probabilities (derived from predicted scores) actually match observed frequencies.
3. **Uncertainty** — whether the model expresses appropriate confidence (GP's predictive variance is the only current source of this; Linear/Poisson/RF/XGBoost are point-estimate only).
4. **Market-relative edge** — comparing model-implied probability to sportsbook-implied probability, only meaningful once 1–3 are validated.
5. **Economic performance** — actual profitability under realistic assumptions (vig, bet sizing via Kelly, CLV tracking) — the final and most demanding stage, not yet reached in this project.

Never skip stages 1–3 to jump to 4–5. Never use market lines as a training feature without first testing, separately and explicitly, whether they add information the model doesn't already have, duplicate information it has, or dominate it entirely.
