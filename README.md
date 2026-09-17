# nfl-score-predictor

A from-scratch system that tries to predict the final score of NFL games —
for example, guessing "Chiefs 27, Bills 24" before the game is played.

This README assumes **no machine learning experience and no deep football
knowledge**. Everything is defined as it comes up. If you know SQL or dbt,
look for the **`For dbt/SQL folks`** boxes — they map each idea onto something
you already use.

---

## 1. What problem is this solving?

Given two teams and a date, predict how many points each will score.

That's it. Every other question people care about — who wins, what the point
difference is, whether the total lands over or under a number — can be derived
from a predicted score pair. That's why the score is the target rather than
just "who wins."

### How good is it, honestly?

The headline number is **RMSE** (root mean squared error). Think of it as
"typically how many points off are we, with big misses punished extra."

Current accuracy, measured across 1,440 real games from 2021–2026:

| What | Typical error per team's score |
|---|---|
| Always guess the league average | ~9.93 points |
| Simple rule, no machine learning | 9.46 points |
| **Our best model (Poisson)** | **9.37 points** |
| Las Vegas betting markets | ~9.10 points |

Two honest takeaways:

1. **NFL games are mostly unpredictable.** The gap between "guess the average
   every time" and "professional betting markets with millions of dollars
   behind them" is only about 0.8 points. Most of what decides a football game
   is randomness — a tipped pass, a fumble bouncing the right way, a kicker's
   bad afternoon. No model fixes that.
2. **We are behind the market, and that's the real scoreboard.** Vegas has
   information we don't (late injury news, betting flow). Closing that gap is
   the point of the project.

> **For dbt/SQL folks:** RMSE is just
> `SQRT(AVG(POWER(actual - predicted, 2)))`. Squaring before averaging is what
> makes one 20-point miss hurt more than four 5-point misses.

---

## 2. Vocabulary you'll need

### Football terms

**Drive** — one team's continuous possession of the ball, from getting it to
losing it (by scoring, punting, or turning it over). A typical game has about
11 drives per team. Points come from drives, so drives are the natural unit.

**EPA (Expected Points Added)** — the single most useful stat in modern
football analysis. Every game situation has an expected point value based on
history: "1st down on your own 25" is worth about 0.9 points on average. If a
play moves you to a situation worth 2.1 points, that play's EPA is +1.2.

Why it beats yards: a 4-yard gain on 3rd-and-2 (keeps your drive alive) and a
4-yard gain on 3rd-and-15 (ends it) are identical in yards and opposite in
value. EPA knows the difference.

**Success rate** — the share of plays with positive EPA. EPA measures *how
much*; success rate measures *how often*. A team with one huge play and
nine bad ones has good EPA and terrible success rate.

**Home field advantage** — home teams win more. Worth roughly 2 points.

**Rest days** — days since a team's last game. Usually 7. Sometimes 4 (Thursday
games) or 14 (after a bye week).

### Machine learning terms

**Feature** — an input to the model. "How well has the home team's offense
been playing lately" is a feature. Features are the columns; the score is what
we predict.

**Training vs. testing** — you fit the model on past games (training), then
check it on games it has never seen (testing). Testing on games you trained on
is like grading your own homework with the answer key open.

**Leakage** — accidentally letting the model see information it couldn't have
had at prediction time. This is *the* cardinal sin here and most of this
project's engineering exists to prevent it. See section 4.

**Baseline** — a deliberately simple method you must beat. Ours blends each
team's recent scoring with the opponent's recent scoring allowed, plus home
field. No machine learning at all. If a fancy model can't beat that, the
fancy model isn't adding anything.

**Overfitting** — the model memorizes quirks of past games instead of learning
real patterns, so it looks great on data it has seen and fails on new games.
Like memorizing answers to last year's exam.

**Regularization** — the standard cure for overfitting: penalize the model for
relying too heavily on any one input. See section 7 — this turned out to be
the single most important fix in the project.

---

## 3. How the system works

Data flows one direction, in stages. Nothing loops back.

```
 STEP 1: DOWNLOAD           free public NFL data
 ─────────────────────────────────────────────────────────────
   schedules   →  who played whom, when, final scores
   play-by-play →  every play of every game since 2019 (~343k rows)
   injuries    →  the official weekly injury report
   snap counts →  what share of plays each player was on the field for

                            ↓

 STEP 2: SUMMARIZE          one row per team per game
 ─────────────────────────────────────────────────────────────
   "In game X, Kansas City averaged +0.12 EPA per play on offense,
    allowed -0.03 on defense, had 11 drives, 7 rest days..."

                            ↓

 STEP 3: LOOK BACKWARD ONLY  ← the critical step
 ─────────────────────────────────────────────────────────────
   For each game, summarize how each team had been playing
   BEFORE that game. Recent games count more than old ones.

                            ↓

 STEP 4: ONE ROW PER GAME   the modeling table
 ─────────────────────────────────────────────────────────────
   home team's recent form | away team's recent form | actual score

                            ↓

 STEP 5: PREDICT AND SCORE
 ─────────────────────────────────────────────────────────────
   Train on past games, predict future ones, measure the error.
```

> **For dbt/SQL folks:** this is exactly a staged dbt project.
> Step 1 is your `raw` sources. Step 2 is `staging` — light cleanup, one grain
> change. Steps 3–4 are `intermediate` and `marts`. Each stage is a script
> that reads Parquet files and writes one Parquet file, the way a dbt model
> reads refs and writes a table. `src/features/build_all.py` is the DAG
> runner — it knows the dependency order and refuses to continue if a stage
> fails. Parquet is just columnar storage; think "a table on disk."

### Why "recent games count more"

A team in week 12 is not the team it was in week 1 — players get injured,
schemes change. So older games are weighted down using **exponential decay**
with a **half-life of 17 weeks**: a game 17 weeks ago counts half as much as
last week's game, a game 34 weeks ago a quarter as much, and so on.

The 17-week value was originally a guess. It has since been tested across
values from 4 to 52 weeks, and 17 really is the best — see section 7.

---

## 4. Leakage: the rule everything else serves

**The rule:** to predict a game, the model may only use information that
genuinely existed before that game kicked off.

This sounds obvious and is extremely easy to violate by accident. Real examples
from this project:

- Computing a team's season EPA average and attaching it to every game that
  season — including games that helped produce that average. The model would
  "know" how the season turned out.
- Filling in missing values using the average of *all* data, including future
  games. A tiny leak, and it inflates accuracy.
- Using a stat published on Monday to predict Sunday's game.

The consequence is always the same: the model looks excellent in testing and
fails in reality, because reality doesn't hand you the future.

### How we prevent it

**Walk-forward testing.** Never test on a random sample of games. Instead:

```
Train on 2019–2020 ───────────► predict week 1 of 2021
Train on 2019–2020 + wk 1 ────► predict week 2 of 2021
Train on 2019–2020 + wks 1–2 ─► predict week 3 of 2021
... and so on, 111 times
```

Each prediction is made knowing only what was actually knowable at the time.
It is slower and it scores worse than a random split — that's the point. The
random split was lying.

**Tests that try to break it.** `tests/` contains hand-built fake teams where
the right answer is known. For example: a team scores 10, 20, 30, then 40
points. The "recent form" value attached to game 4 must land between 10 and 30.
If it's 40, the model has seen the game it's predicting, and the test fails.

We also verify these tests actually work by deliberately breaking the real
code and confirming the tests catch it. A test that can never fail is worse
than no test, because it creates false confidence.

> **For dbt/SQL folks:** leakage is a join that ignores effective dates —
> joining a fact to a slowly-changing dimension without `WHERE valid_from <=
> event_date`. The walk-forward harness is a window function with an explicit
> frame: only rows strictly before the current one. The leakage tests are dbt
> tests, except asserting temporal correctness rather than uniqueness.

---

## 5. Running it

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

**Step 1 — download the data** (slow the first time; play-by-play is ~130 MB):

```bash
python src/ingest/pull_schedules.py
python src/ingest/pull_pbp.py
python src/ingest/pull_injuries.py
```

**Step 2 — build all the features** (about 50 seconds, all 11 stages in order):

```bash
python -m src.features.build_all
```

**Step 3 — check it worked:**

```bash
pytest tests/ -v
```

**Step 4 — train and score a model:**

```bash
python -m src.models.baseline           # the no-ML benchmark, ~3 seconds
python -m src.models.linear             # ~2.5 minutes
python -m src.models.poisson_glm        # ~1 minute
python -m src.models.gaussian_process   # ~30 minutes, currently the best
python -m src.models.stacking           # ~1 minute, needs the three above
```

**Compare everything at once:**

```bash
python -m src.validate.model_scoreboard --against baseline --common-games
```

### The weekly loop, once you're set up

Every Tuesday, after the previous week's games are in the books:

```bash
python src/ingest/pull_schedules.py      # last week's final scores
python src/ingest/pull_pbp.py
python src/ingest/pull_injuries.py
python -m src.features.build_all         # ~30s
python -m src.predict.predict_week --html   # ~40s: predicts the new week,
                                            # grades the old one, rebuilds the page
firefox data/predictions/index.html
```

Step 4 is the only slow part and the only one that touches a model. If you just
want last week's predictions graded against the results that landed, skip it and
run `python -m src.predict.build_report` instead — a second, no refitting.

---

## 5b. Predicting games that haven't been played

Everything above scores the past. This predicts the future.

```bash
python -m src.predict.predict_week --html
```

That fits every model on all completed games, predicts the next unplayed week,
prints a table, and writes:

```
data/predictions/2026_wk02.parquet   this week's numbers, one file per week
data/predictions/index.html          the ledger: every week, graded
```

Pick a specific week with `--season 2026 --week 2`. Add `--json out.json` if you
want the raw numbers somewhere else.

### The ledger

The page is not a snapshot of one week. It reads **every** prediction file on
disk and shows them all, opening on the most recent:

- a week that hasn't been played yet shows the four model predictions next to
  the market's implied score;
- a week that has been played shows the same predictions with the **final score
  beside them**, how far off each model was, and whether it picked the winner;
- the tab strip along the top pages through every week on file, each tab
  carrying its own record (`12–4 · 7.62 MAE`). Arrow keys work too.

Nothing about a prediction is ever rewritten when results arrive. Each week's
parquet is written once, before kickoff, and read back unchanged — the grades
are computed fresh at render time from the final scores. That is the whole
point of keeping the parquet files in git: they are a dated record of what the
model claimed *beforehand*, which is the only kind of forecast worth counting.

Weeks that were predicted after the games were played are labelled **backfilled**
on the page. The fit still never sees a game that hasn't kicked off, so no result
from that week reaches the model — but two softer advantages remain that a real
pre-kickoff forecast doesn't have:

1. nobody was stopping you from regenerating them until they looked good;
2. the model's settings (the recency half-life, the ridge penalty) were chosen on
   a dataset that already contained those weeks.

Neither is leakage in the strict sense, and both are enough to make a backfilled
record flatter than a live one. The 2025 season on this page is backfilled. The
weeks predicted from 2026 Week 2 onward are not, and those are the ones to judge
the model by.

Rebuild the page any time new results land, without re-predicting anything:

```bash
python -m src.predict.build_report
```

That takes about a second, because it only re-reads files. Re-running
`predict_week` is the slow part (it refits every model, roughly 40 seconds) and
is only needed when you want a *new* week predicted.

### Viewing it

Open the file. No server, no build step, no internet needed beyond the web fonts
(it falls back to system fonts offline):

```bash
firefox data/predictions/index.html
```

or `xdg-open` on Linux, `open` on macOS, or double-click it in a file manager.
`python -m src.predict.build_report --open` builds it and launches your browser
in one step. Everything is baked into the one file, so you can email it or copy
it to a phone and it still works.

### It will refuse to run on stale data

Predicting Week 2 from a table that never received Week 1 produces confident,
wrong numbers and no error message, so `predict_week` treats that as a failure
rather than a warning. It checks **completeness, not the calendar**: it asks
whether any game that has already been played is missing from the training
table, and tells you which fix you need —

- the score exists in `schedules.parquet` but not in the model table → rebuild
  (`python -m src.features.build_all`);
- a game has kicked off and has no score anywhere → re-pull first.

(The earlier version of this check compared dates and refused anything older
than three weeks. That was wrong in the worst possible place: every Week 1 sits
about 210 days after the previous Super Bowl, so a perfectly current table looked
seven months stale and the guard would have blocked opening weekend outright.
`tests/test_prediction_freshness.py` now pins that case down.)

All predicted scores are rounded to one decimal place. The models' typical
error is over nine points, so anything finer would be noise dressed as
precision.

Note: scripts that import from other project files must be run with `-m` from
the repo root (`python -m src.models.linear`), not as a file path.

---

## 6. Project layout

```
src/
  ingest/      downloads raw data. Run these first.
  features/    turns raw data into model inputs. build_all.py runs them in order.
  models/      the prediction models themselves
    unused/    models that were built, measured, and shelved — kept on purpose
  predict/     forecasts for games that haven't happened, and the ledger page
  validate/    the scoring harness, error analysis, calibration
  experiments/ one-off tests of "would this idea help?" — never touched by production
tests/         correctness and leakage gates
data/
  raw/         downloaded data (not in git)
  processed/   built features (not in git)
  predictions/ one parquet per predicted week — IN git on purpose, as a dated
               record of what was claimed before kickoff. index.html is not.
```

Two conventions worth knowing:

**Failed ideas are kept, not deleted.** `src/models/unused/` holds nine models
that were properly built and measured and did not earn a place. Deleting them
would mean someone rebuilds them in a year. Their docstrings say what happened.

**Experiments never touch production.** Testing a new idea means writing a
standalone script in `src/experiments/` that reads production data and writes
nothing back. Production changes only after the experiment shows the idea works.

---

## 7. What we've actually learned

These are measured results, not opinions. All use the same 1,426-game test set.

### The model was drowning in features, not starving for them

The intuition "add more football knowledge, get better predictions" is wrong
here. We built four new families of advanced stats — pace of play, efficiency
with blowouts excluded, turnover luck, kicking — and **every single one made
the model worse**, with the damage growing as more were added.

The cause wasn't bad football. The linear model was using an estimator with no
regularization on 18 inputs and only ~1,900 training games. It was memorizing
noise. Switching to a regularized version (**ridge regression**) fixed it.

**The lesson:** with limited data, fewer strong inputs beat many weak ones.

### Injuries help — the first feature that ever did

The model had no idea who was actually playing. Adding that helped, but *how*
it was added mattered enormously.

A plain count of injured players is useless — losing a starting left tackle and
losing a fourth-string linebacker both count as "1." Instead we compute a
single number per team:

```
injury_impact = Σ (how valuable the position is  ×  how much that player
                   had actually been playing)
```

That one column improved both models with statistical confidence — the first
feature in the project's history to do so.

We also tested a separate "starting QB is out" flag. It made things **worse**
when added alongside, because a missing QB already dominates the impact number.
Two features fighting over the same signal is worse than one clean feature.

### Ideas that were tested and rejected

Recorded so nobody rebuilds them:

| Idea | Outcome |
|---|---|
| Splitting efficiency into passing vs. rushing | Worse on the first attempt; **rebuilt and shipped** — see below |
| CPOE (a quarterback accuracy stat) | Worse |
| Adjusting stats for opponent strength | No effect |
| Quarterback-specific historical stats | No effect |
| Pace, turnover luck, special teams | Worse |
| Weather | Not usable — see below |

**The pass/rush split came back.** It is the one rejected idea that has since
been rebuilt, and the story is worth keeping because the original rejection was
partly an artifact of *how* it was measured.

The first attempt replaced blended efficiency with pass-only and rush-only
numbers and measured a real regression. But it was measured with the
unregularized model — the estimator this project diagnosed as broken the
very next day — and it did nothing about the problem it had itself identified:
splitting a team's plays in two roughly halves the evidence behind each number.
A team throws about 36 times and runs about 26 times a game, against 62 plays
combined, and the old code treated a 9-carry game and a 38-carry game as equal
evidence.

The rebuild fixes both. Each number is now a rate per *play* rather than an
average of per-game averages, so a heavy workload counts for more. And each is
pulled toward the league average in proportion to how little evidence stands
behind it — measured, not guessed: a team's own passing number earns only about
a quarter of the weight after four games, and a bit under two-thirds after a
full season. Defences are pulled harder than offences, because defensive
performance is measurably less repeatable.

Like the first attempt, the split **replaces** the combined efficiency numbers
rather than sitting beside them. Keeping both was tried for about an hour and
dropped: the combined number and the split describe the same thing at different
levels of detail, and handing a model two versions of one measurement is asking
it to divide the credit between them. Removing the combined columns cost
nothing measurable — the models scored identically to three decimal places with
and without them, which is itself the clearest evidence they were redundant.

Result: **the regression is gone and a small improvement appears, but it is
still inside the noise band** (Ridge −0.012, 95% CI [−0.034, +0.011]). It was
shipped anyway as a deliberate call. Honest summary: it no longer hurts, it
probably helps slightly, and the project cannot prove it.

**Weather deserves explaining.** Historical weather is what *actually
happened*. But to predict a future game you'd only have a *forecast*, which is
often wrong. Training on truth and predicting from guesses is a mismatch that
makes the model worse, not better. (We also confirmed the data source has no
pre-game forecasts at all.)

**Betting odds are deliberately excluded** from the model's inputs. They're
extremely predictive — they contain everything the market knows — which is
exactly the problem. A model that has learned to copy Vegas has learned
nothing, and can never beat Vegas. Odds are used only as a scoreboard.

---

## 8. Testing philosophy

294 tests in five layers:

1. **Data contracts** — is the downloaded data shaped correctly? (Exactly 32
   teams, no duplicate plays, scores non-negative.)
2. **Leakage gates** — hand-built examples with known answers, proving no
   future information reaches the model.
3. **Freshness gates** — every built file must be newer than the files it was
   built from. This caught a real bug where results were being compared against
   a stale table for two days.
4. **Completeness gates** — the mirror image of leakage: does the model see
   everything it *should*? A model that is leakage-free but silently missing
   last week's games runs cleanly and is useless. This layer also covers the
   ledger's grading — the sign of every error, and what counts as a correct
   pick.
5. **Statistical honesty** — improvements smaller than ~0.05 RMSE must pass a
   **bootstrap test** before being believed.

> **What's a bootstrap test?** If a change improves error from 9.40 to 9.38, is
> that real or luck? We re-draw the 1,426 games at random (with repeats) 5,000
> times and re-measure. If the improvement holds up across nearly all 5,000
> redraws, it's real. If it flips sign depending on which games we drew, it was
> noise. Most promising-looking improvements in this project turned out to be
> noise, which is precisely why the test exists.

---

## 9. Current status and what's next

**Working:** the full data pipeline, leakage-safe evaluation, four trained
models, a live prediction step for games that haven't been played, and a ledger
page that grades those predictions once the results land.

**Known gaps:**
- No live starting-quarterback source for future games. Historical rows know who
  actually played; on Wednesday you don't. `load_depth_charts` is the obvious
  place to look and hasn't been tried.
- The three-model ensemble does not beat the best single model, and its
  calibration is *worse* than its own inputs — the meta-model refits on
  predictions that have already had the bias correction applied to them, and
  over-corrects. Either feed it uncorrected inputs or drop it. Not yet done.
- Predicted totals ran about 2 points high in 2026 Week 1. Three fixes were
  tested and all three were rejected (they each made the scores worse); the
  mechanism is written up in `src/experiments/` so nobody re-tries them blind.
- Against the spread the model is at coin-flip, not profitable. See below.

**Current benchmark** (1,440 games, 2021–2026, RMSE per team's score — the
number every future change is measured against):

| Model | Error | Beats the no-ML baseline? |
|---|---|---|
| Poisson GLM | **9.371** | yes, confirmed by bootstrap |
| Gaussian Process | 9.376 | yes, confirmed by bootstrap |
| Ridge regression | 9.385 | not distinguishable from noise |
| Rule-based baseline | 9.458 | — |

(On the 1,155 games all sixteen models share — the only fair head-to-head — the
order is GP 9.277, Poisson 9.279, Ridge 9.290, stack 9.293, baseline 9.350.)

Home-score bias, which was the last thing fixed, now sits at +0.06 to +0.09
points across the active models. It was +0.78 before the correction.

**Is it ready to bet with? No.** Beating the market requires about 52.4% against
the spread to cover the vig; the model is at 51.0%, which is inside the range you
would expect from a coin flip over this many games. It predicts *scores*
respectably and it does not predict *market inefficiency* at all. Those are
different jobs, and only the first one is going well.
