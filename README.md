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

Current accuracy, measured across 1,473 real games from 2021 through 2026 Week 4's
Thursday game (re-benchmarked 2026-10-04):

| What | Typical error per team's score |
|---|---|
| Always guess the league average | 9.95 points |
| Simple rule, no machine learning | 9.47 points |
| **Our best model (the composite: Poisson + GP + Ridge, averaged)** | **9.36 points** |
| Las Vegas closing line | 9.12 points |

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

### The one rule above accuracy: correct before accurate

**If logic is broken, we fix it — always — even if the fix makes the model
less accurate.** "Broken" means a calculation that is wrong, that uses
information it could not have had yet, or that does not measure what its name
says it measures.

Why: an accuracy score is only as honest as the math behind it. If a model
scores better *because of* a mistake, it is not a better model — we just no
longer know what we are measuring. If your calculator adds wrong and happens to
give you a higher test score, you still fix the calculator.

What happens when we find broken logic:

1. **Fix it.** No vote and no accuracy test decides this. The accuracy tests
   in section 8 decide whether a *new idea* gets in; they never decide whether
   broken logic stays broken.
2. **Measure what the fix did** — better or worse — and write it down here, so a
   number that moves is never a surprise.
3. **Re-forecast every game that hasn't kicked off.** A forecast for a game that
   has already started is never changed.
4. **If the broken version was accidentally useful**, that idea can come back
   only as its own honestly named, correctly built input — and then it has to
   pass the normal tests like any other new idea.

Real example: in October 2026, "offensive success rate" turned out to include
kickoffs and extra points (section 7). It was fixed because it was wrong. It
happened to make the model slightly more accurate, but it would have been fixed
if it had made it worse.

> **For dbt/SQL folks:** if a model named `net_revenue` turned out to be adding
> refunds instead of subtracting them, you would fix the SQL even though this
> quarter's dashboard number drops. A metric that looks better because it is
> wrong is still wrong — and every report built on it was quietly wrong too.

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

**Success rate** — the share of a team's snaps (passes and runs) with
positive EPA. EPA measures *how much*; success rate measures *how often*. A
team with one huge play and nine bad ones has good EPA and terrible success
rate.

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
   play-by-play →  every play of every game since 2019 (~350k rows)
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

### It runs itself, on GitHub

Nothing has to be run by hand. `.github/workflows/pipeline.yml` runs the whole
pipeline on GitHub's machines on a schedule: pull the latest data, rebuild every
feature table, run every test and data check, refit the live models, forecast
every game whose final injury report is out, grade the ledger, and **commit any
forecast that changed** back to this repository. No secrets, no person, no
Claude. The data is public, and the workflow's own token makes the commit.

| When (UTC) | What it does |
|---|---|
| Every day, 21:37 | Shortly after that day's 4pm ET final injury report (90 minutes after in daylight time, 37 in winter), forecasts the slate it covers: Wednesday → the Thursday game, Thursday → Saturday games, Friday → the Sunday slate, Saturday → Monday night |
| Every day, 13:37 | Refreshes those forecasts on the overnight data. Monday's run refits the Monday night game on Sunday's results. |
| Sunday, 11:37 | Re-forecasts the Sunday and Monday games on the latest data. Meant to run before the 9:30am ET London kickoffs, but GitHub starts it hours late (below), so London games are covered by Saturday's runs |
| Tuesday, 13:37 | Also grades the week and re-runs **every** model's walk-forward backtest plus the model report |

A game is never forecast before its final injury report (below), or within 15
minutes of its kickoff, so every forecast is committed before its game starts.
Two runs a day is deliberate: GitHub starts scheduled runs late -- every one of
the first 19 (2026-09-25 to 10-03) began 2.2 to 6.6 hours after its cron time,
3.6 hours typically -- and can skip them, so every game needs several chances.
A Thursday night game is the tight one: its final report comes Wednesday
afternoon, and nflverse has published it more than a day late. A run that
finds nothing ready, or reproduces forecasts already on file, commits nothing.
Between seasons each run stops after one schedule check.

- **Results:** forecasts land in `data/predictions/` as commits titled
  `Forecast: <season> week <N> (automated)`. Each run's page under the
  **Actions** tab shows the current week's forecasts, and the leaderboard on
  Tuesdays. The ledger page, model report and logs are attached as downloads.
- **Failures** are emailed by GitHub, and nothing half-finished is committed.
- **By hand:** Actions → pipeline → *Run workflow* (options: backtest, publish,
  forecast early), or `gh workflow run pipeline.yml -f backtest=true`.
- **Pause:** Actions → pipeline → ⋯ → *Disable workflow*.

Everything below runs the same code on your own machine.

Requires Python 3.12. Run everything from the repo root.

```bash
pip install -r requirements.txt
```

### The weekly run — one command

```bash
python -m src.weekly
```

That does, in order, stopping at the first failure with the fix spelled out:

1. **pull** fresh schedules, play-by-play, injuries and snap counts from nflverse,
   and check they are complete and agree with each other (every finished game has a
   score, every scored game has plays, and play-by-play's final score matches the
   schedule's);
2. **build** every feature table in dependency order (~50 seconds);
3. **test** everything: every unit test and leakage gate, and every check on the
   built data (joins, completeness, next week populated);
4. **forecast** every game that has not kicked off *and whose final injury report
   is in the data* (see below);
5. **rebuild the ledger page**, grading every finished week.

`--backtest` adds the full re-benchmark before step 4. `--only-in-season` makes
the run a no-op unless a game is within 2 days back or 9 ahead. The scheduled
workflow passes both, as appropriate.

On GitHub the workflow commits the forecasts itself. Run locally, record them
before kickoff, so the ledger stays a dated, pre-registered record:

```bash
git add data/predictions/*.parquet && git commit -m "Forecast: 2026 week 3"
```

### When to run it: the injury report

**The injury data has to be pulled shortly before game time.** The model's only
leading indicator is `injury_impact` — who is out, weighted by how much they
play — and it is built from each player's game status (Out / Doubtful /
Questionable). Teams only assign those statuses in the **final** injury report
before each game:

| Game day | Final report | Run the forecast |
|---|---|---|
| Thursday | Wednesday afternoon | Thursday afternoon |
| Sunday | Friday afternoon | Sunday morning |
| Monday | Saturday afternoon | Sunday morning |
| Saturday | Thursday afternoon | Friday or Saturday morning |

Earlier in the week the report is nearly empty (on the Thursday of 2026 week 3,
8 of 259 rows had a status, all for the two Thursday-night teams). A forecast
made then quietly treats every team as fully healthy. Nothing errors; the
numbers just look like football scores.

So **`predict_week` refuses to forecast a game until its final report is in the
data.** It checks that the injuries were pulled after the report was due, and
that the report actually shows up for that slate. Held-back games are listed
with the time their report is due. The scheduled workflow's daily runs follow
that rhythm automatically. By hand, it would be:

- **Thursday afternoon:** `python -m src.weekly` forecasts the Thursday game.
- **Sunday morning:** `python -m src.weekly` forecasts the Sunday and Monday games.

Running it at any other time is safe. It forecasts only what is ready, and it
never touches a forecast for a game that has kicked off or that it did not
re-forecast. If you truly need an early number, `--allow-unsettled-injuries`
forecasts everything and tags those games *pre-injury-report* on the ledger.

### The pieces, one at a time

```bash
python -m src.ingest.pull_all              # pull + validate (or --check-only)
python -m src.features.build_all           # all 11 feature stages, in order
pytest tests/ -m "not backtest_artifacts"   # every test that needs no backtest
python -m src.predict.predict_week --dry-run   # what is ready to forecast, and why
python -m src.predict.predict_week            # forecast what is ready (+ rebuilds the ledger)
python -m src.predict.build_report         # re-grade the ledger only (1 second)
```

### Re-benchmarking every model

```bash
python -m src.models.run_all               # every model's walk-forward, in parallel
python -m src.validate.model_report        # the deep evaluation (see section 9)
python -m src.validate.model_scoreboard --common-games
python -m src.validate.error_analysis
```

Each of the three takes `--season 2026` to judge one season on its own (the
report then goes to `data/processed/reports/season_2026/`). And for the
forecasts that were actually published, rather than the backtest:

```bash
python -m src.validate.live_report --season 2026
```

That grades every record in `data/predictions/` against the results and the
market: accuracy next to the market's own implied score, each model's spread and
total picks won-lost-push (against the line when forecast and against the close),
and closing-line value -- how far the line moved toward each pick after it was
published, the standard test of whether a forecast knows something the market
doesn't. Backfilled forecasts are graded but kept out of that last one.

`run_all` runs all 17 backtests (every production, shelved and experimental
model plus the stack and the composite) in dependency order, 4 at a time, with
a log per model in `data/processed/logs/`. It takes about 40 minutes, most of
it the Gaussian Process. Each backtest records a hash of the data it was
scored on. Tests flag a backtest as stale only when the played games it read
have actually changed, not whenever a file is re-saved.

### Which models forecast live

Set in `config.yaml` under `live:`. That's the individual models and the
composite (an equal-weight average of the members, computed before rounding).
Any model in `src/models/registry.py` that can forecast an unplayed game is
allowed there. The config is checked before anything is fitted.

---

## 5b. Predicting games that haven't been played

Everything above scores the past. This predicts the future.

```bash
python -m src.predict.predict_week
```

That fits the live models on every completed game before the first game it is
forecasting, predicts every game whose final injury report is in, prints a
table, rebuilds the ledger page, and writes:

```
data/predictions/2026_wk03.parquet   this week's forecasts, one file per week
data/predictions/index.html          the ledger: every week, graded
```

Pick a specific week with `--season 2026 --week 3`. Add `--json out.json` if you
want the raw numbers somewhere else. Each row records when it was forecast,
which models made it, whether its injury report was final, and the code version.

### The ledger

The page is not a snapshot of one week. It reads **every** prediction file on
disk and shows them all, opening on the most recent:

- a week that hasn't been played yet shows the live models' predictions, then
  four betting columns for the combined forecast:

  | Column | Reads | Means |
  |---|---|---|
  | Implied score | `19.0–24.5` | the score that line works out to, read like the forecasts |
  | Market | `GB −5.5 · O/U 43.5` | the line when the forecast was made: GB favoured by 5.5, 43.5 points |
  | Model line | `GB −3.6 · 45.6 pts` | the forecast in the same terms: GB by 3.6, 45.6 points |
  | Spread pick | `ATL +5.5 · edge 1.9` | the side the forecast takes against that line, and the gap in points |
  | Total pick | `Over 43.5 · edge 2.1` | likewise for the total |

  Hovering a pick gives the rule for any other line ("take ATL if GB is
  favoured by more than 3.6"). Played games mark each pick won, lost or push,
  and the header keeps the running record. These are the model's picks, not
  advice: in the backtest they have won about half the time;
- a week that has been played shows the same predictions with the **final score
  beside them**, how far off each model was, and whether it picked the winner;
- the tab strip along the top pages through every week on file, each tab
  carrying its own record (`12–4 · 7.62 MAE`). Arrow keys work too.

Nothing about a prediction is ever rewritten when results arrive. Each week's
parquet is written once, before kickoff, and read back unchanged — the grades
are computed fresh at render time from the final scores. That is the whole
point of keeping the parquet files in git: they are a dated record of what the
model claimed *beforehand*, which is the only kind of forecast worth counting.

Individual games predicted after they kicked off are labelled **predicted late**
on the page, and a week says how many of its games that applies to. The fit still never sees a game that hasn't kicked off, so no result
from that week reaches the model — but two softer advantages remain that a real
pre-kickoff forecast doesn't have:

1. nobody was stopping you from regenerating them until they looked good;
2. the model's settings (the recency half-life, the ridge penalty) were chosen on
   a dataset that already contained those weeks.

Neither is leakage in the strict sense, and both are enough to make a late
forecast flatter than a live one.

The label is per **game**, not per week, because a week is written several
times as its slates come up: each game is first forecast by the first run after
its own final injury report is in the data (usually the day before it, for a
Sunday game), and every later run until 15 minutes before its kickoff forecasts
it again. A re-forecast that comes out identical leaves the stored row, and its
original time, alone; one that moved -- Monday night's game refitted on
Sunday's results, say -- replaces it. Once a game kicks off its forecast is
frozen for good. So a week is routinely part pre-registered and part not, and
saying which games were late beats condemning the whole slate. (Until
2026-10-04 this paragraph said each game was forecast about six hours before
kickoff and then frozen; neither was how the code works.)

**The page always matches the records.** `data/predictions/index.html` is
tracked in git next to the records and committed with them by the pipeline, so
a `git pull` brings a page that shows exactly what the records say. (Until
2026-09-27 it was a local file only a local run rebuilt: a pull brought Sunday's
forecasts in as *final* while the page still tagged them *pre-injury-report*.)
Three guarantees keep it honest:

- it embeds a fingerprint of every record it was built from, and is checked
  against them game by game (forecasts and injury-report status) before it is
  written, so a page that disagrees with the records is never produced;
- it contains no build time, so the same records and results always give the
  same file, and the committed page changes only when they do;
- CI fails any push where the committed page and records disagree.

Rebuild it any time new results land, or check whether the copy you have is
current, without re-predicting anything:

```bash
python -m src.predict.build_report          # rebuild (about a second)
python -m src.predict.build_report --check  # exit 1 if it does not match the records
```

`predict_week` rebuilds it automatically whenever it writes a record. Running
it is the slow part (it refits every model, roughly 40 seconds) and is only
needed when you want a *new* week predicted.

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
.github/workflows/
  pipeline.yml the scheduled pipeline: runs src/weekly.py, commits forecasts
  tests.yml    unit tests and leakage gates on every push
src/
  weekly.py    the weekly run: pull -> build -> test -> forecast -> ledger
  config.py    reads and validates the live-model block of config.yaml
  schedule.py  kickoff times (one implementation, used everywhere)
  provenance.py  content hashes and git state for every artefact
  ingest/      downloads raw data; pull_all.py pulls everything and validates it
  features/    turns raw data into model inputs. build_all.py runs them in order.
  models/      the prediction models themselves
    registry.py  every model, described once (run_all, predict_week use it)
    run_all.py   every walk-forward backtest, in parallel
    composite.py the live composite, backtested exactly as published
    unused/    models that were built, measured, and shelved — kept on purpose
  predict/     forecasts for games that haven't happened, and the ledger page
    injury_readiness.py  is a game's final injury report in the data yet?
    summarize.py         forecast tables for the workflow's page and commits
  validate/    the scoring harness, error analysis, calibration, model report
  experiments/ one-off tests of "would this idea help?" — never touched by production
tests/         correctness and leakage gates
data/
  raw/         downloaded data (not in git); _pull_manifest.json says when
  processed/   built features, backtests, logs and reports (not in git)
  predictions/ one parquet per predicted week — IN git on purpose, as a dated
               record of what was claimed before kickoff — and index.html, the
               ledger page, verified against those records on every build
```

Three conventions worth knowing:

**Broken logic is fixed, not voted on.** Section 1's rule: anything that
calculates the wrong thing, or doesn't measure what its name says, is fixed even
if the model gets less accurate. The fix is still measured, and the result is
written down whichever way it goes.

**Failed ideas are kept, not deleted.** `src/models/unused/` holds nine models
that were properly built and measured and did not earn a place. Deleting them
would mean someone rebuilds them in a year. Their docstrings say what happened.

**Experiments never touch production.** Testing a new idea means writing a
standalone script in `src/experiments/` that reads production data and writes
nothing back. A *new idea* reaches production only after the experiment shows it
works. A *fix* to broken logic doesn't wait for that — the experiment only
records what the fix changed. Experiments that rebuild a feature table use
production's own assembler (`build_game_features.assemble`), so they can't
quietly drift from it.

---

## 7. What we've actually learned

These are measured results, not opinions. Each was measured on the walk-forward test set of its day: every game from 2021 on that had been played by then (1,426 to 1,472 games).

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

### Success rate measures the offence now, not the kicking team

Two of the model's inputs are each team's recent **success rate**, on offence
and on defence. Until October 2026 it was taken over *every* play — kickoffs,
punts, extra points, field goals and kneel-downs included, 23% of the plays.
Those aren't neutral: an extra point "succeeds" 94% of the time, so a team that
scored a lot of touchdowns looked more *efficient*, double-counting the points
number right next to it, and a team kneeling out a win looked less so.

Both rates are now measured on snaps from scrimmage only — passes, sacks, runs
and scrambles. Three definitions went through the full walk-forward against
production itself (`src/experiments/test_success_rate_definition.py`). Change in
typical error per team score; negative is better:

| Definition | Ridge | Poisson | GP | Combined (published) |
|---|---|---|---|---|
| **Scrimmage snaps only** — shipped | −0.009 | −0.007 | −0.010 | **−0.008** |
| Also counting snaps wiped out by a penalty (nflfastR's convention) | −0.006 | −0.004 | −0.007 | −0.005 |

Every model happened to improve, and the margin error with it, though no
version clears the bootstrap test on its own (the published model is better in
87% of redraws). **That is not why it shipped.** The old number measured the
wrong thing, so under section 1's rule it would have been fixed even if every
model had got worse. The table is here so you can see what the fix changed. The
experiment chose *which* correct definition to use; it never decided *whether*
to fix it.

### Five more fixes (October 4, 2026) -- each made accuracy a hair worse

A line-by-line audit found logic that did not do what its name or its
documentation said. All of it was fixed for that reason alone (section 1):

1. **Recent form decayed on the wrong clock.** The "half-life of 17 weeks" was
   computed with a pandas setting that skips the decay across masked rows (the
   rested-starters finale, unplayed games), so an older game kept more weight
   than its age allows. 88% of rows moved, by 0.03 points on average.
2. **Missing injury data counted as a healthy team.** 53 played team-games
   have no injury report in the data, or no earlier snap counts to say who
   matters (2019's opening week, most of the 2023 playoffs). They were scored as
   "nobody of note was out"; they are now marked unknown, and the models fill in
   the average.
3. **Two players were counted twice** in one 2024 week (listed Questionable,
   then Out); the game-day report now stands alone.
4. **The ledger showed week 2 as if it used the final injury report.** Fifteen
   of its sixteen forecasts were made a day or more before that report was due.
   They are now tagged *pre-injury-report*.
5. **A network error could silently drop this season's data.** The download
   step treated any failure on the newest season as "not published yet", even
   in October. It now stops the run once the season has been underway a week.

Fixes 1-3 change model inputs. Measured together on the same 1,473 games, change
in typical error per team score (positive is worse):

| Ridge | Poisson | GP | Combined (published) |
|---|---|---|---|
| +0.002 | +0.002 | +0.001 | **+0.002** (95% interval −0.000 to +0.004) |

Inside the noise, slightly worse on every model, and shipped anyway: the old
numbers measured something other than what they said.

### Ideas that were tested and rejected

Recorded so nobody rebuilds them:

| Idea | Outcome |
|---|---|
| Splitting efficiency into passing vs. rushing | Worse on the first attempt; **rebuilt and shipped** — see below |
| CPOE (a quarterback accuracy stat) | Worse |
| Adjusting stats for opponent strength | No effect |
| Quarterback-specific historical stats | No effect |
| Pace, turnover luck, special teams | Worse |
| Penalties (yards committed, EPA lost on flagged plays) | Worse -- mostly noise (see below) |
| Travel: distance, time zones crossed, body-clock kickoff time, road streaks, international trips | Worse -- patterns don't carry from season to season |
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

**Penalties are mostly officiating and chance.** No feature uses them directly;
they reach the model only through points and success rate, since the pass/rush
efficiency numbers leave out the ~75% of flags that wipe a play out. Screened
2026-09-28: a team's penalty yards in odd-numbered games barely predict its
even-numbered games (split-half r = 0.11; penalty EPA 0.11-0.18), and
correcting the backtest with leakage-safe rolling penalty numbers made every
later season worse (+0.016 to +0.025 RMSE, intervals clear of zero). A
residual screen rather than the full walk-forward experiment, but with that
little signal there is nothing for a model to find.

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

About 790 tests (run `pytest --co -q` for the current count), in five layers.
CI runs every test that does not need `data/` on each push; the rest run
locally after a build.

1. **Data contracts** — is the downloaded data shaped correctly? (Exactly 32
   teams, no duplicate plays, scores non-negative.)
2. **Leakage gates** — hand-built examples with known answers, proving no
   future information reaches the model.
3. **Freshness gates** — every built file must have been built from the inputs
   as they are now, checked by content fingerprint (an earlier version compared
   file times, and cried wolf on every re-save). This caught a real bug where
   results were being compared against a stale table for two days.
4. **Completeness gates** — the mirror image of leakage: does the model see
   everything it *should*? A model that is leakage-free but silently missing
   last week's games runs cleanly and is useless. This layer also covers the
   ledger's grading — the sign of every error, and what counts as a correct
   pick.
5. **Statistical honesty** — improvements smaller than ~0.05 RMSE must pass a
   **bootstrap test** before being believed. This is how a *new idea* earns its
   place. It is never used to decide whether to fix broken logic: a fix ships
   even if it scores worse (section 1), and its result is recorded either way.

> **What's a bootstrap test?** If a change improves error from 9.40 to 9.38, is
> that real or luck? We re-draw the test set's *weeks* at random (with repeats)
> 5,000 times and re-measure -- whole weeks, because games in the same week share
> one fitted model and one scoring environment, so they are not independent
> draws. If the improvement holds up across nearly all 5,000
> redraws, it's real. If it flips sign depending on which games we drew, it was
> noise. Most promising-looking improvements in this project turned out to be
> noise, which is precisely why the test exists.

---

## 9. Current status and what's next

**Working:** the full data pipeline with validated pulls, leakage-safe
evaluation, 16 benchmarked models plus a composite, a live forecast step that
waits for each game's final injury report, a one-command weekly run, and a
ledger page that grades every forecast once the results land.

**Current benchmark** (live models and baseline re-run 2026-10-04 on 1,473 games
from 2021 through 2026 Week 4's Thursday game, after the fixes of that day --
section 7; every other model last re-run 2026-10-02 on 1,472 games, before
them. RMSE per team's score, pooled home and away):

| Rank | Model | RMSE | Beats the no-ML baseline?* |
|---|---|---|---|
| 1 | **Composite** (Poisson + GP + Ridge, equal weights) — live | **9.363** | yes |
| 2 | Poisson GLM — live | 9.366 | yes |
| 3 | Gaussian Process — live | 9.369 | yes |
| 4 | Ridge regression — live | 9.374 | yes |
| 5 | Bayesian hierarchical | 9.410 | no (noise) |
| 6 | CatBoost | 9.426 | no |
| 7 | Random forest | 9.441 | no |
| 8 | XGBoost | 9.443 | no |
| 9 | LightGBM | 9.457 | no |
| 10 | Rule-based baseline | 9.469 | — |
| 11–16 | Drive model v2, drive chain, logistic, RNN, Monte Carlo v1, MLP | 9.48–9.90 | no; Monte Carlo and MLP are significantly worse |

\*Paired bootstrap that resamples whole weeks. Games in one week share a model
and a scoring environment, so resampling single games gives intervals that are
too narrow. The ridge stack is scored on 2022 onward only (1,187 games, 9.282 on
its own games) and is not ranked against models scored on more.

What the deeper report says (2026-10-02 run, all in
`data/processed/reports/model_report.html`):

- **The live models are near-duplicates.** Their errors correlate at 0.997–0.9996,
  which is why averaging them gains only a few thousandths of a point. Fitted
  weights do no better than equal ones.
- **The closing line already contains everything the model knows.** Regress the
  real margin on the closing spread AND the model's margin and the model's weight
  is 0.14, 95% interval −0.09 to +0.38 -- indistinguishable from zero (totals:
  −0.15, −0.46 to +0.15). Blending the model into the line does not reduce error.
- **Totals lean high, and most in prime time.** The median total error is +1.3
  points and 54% of games finish under the model's total; rare blowouts pull the
  *mean* back to about +0.2. Prime-time totals are over-predicted by more (Monday
  night +2.3, Sunday night +1.4). The model's total is an average, and scores are
  skewed, so against a betting total (which sits near the median) its total pick
  is the Over 60% of the time while only 48% of games go over.
- **Two teams are mis-rated by every model**, beyond the noise: Buffalo is
  under-rated (margin about 2.9 points too low) and Tennessee over-rated (2.7).
- **The market is still ahead:** closing-line margin error 12.67 against the
  composite's 13.02.

**Is it ready to bet with? No.** Beating the market requires about 52.4% against
the spread to cover the vig. The composite is at 50.6% (95% interval 48.2–53.1%),
which is a coin flip. It predicts *scores* respectably and it does not predict
*market inefficiency* at all. Those are different jobs, and only the first one is
going well.

**Known gaps:**
- No live starting-quarterback source for future games. Historical rows know who
  actually played; on Wednesday you don't. `load_depth_charts` is the obvious
  place to look and hasn't been tried.
- A kickoff-window feature (prime time) is the cheapest candidate for the
  prime-time totals bias above. It is untested and needs a walk-forward
  experiment before anything ships.
- The roster-continuity experiment must be re-run: its builder had a bug (fixed
  2026-09-24) that corrupted the features it was measured with.
- ~~The Sunday cloud routine cannot push its forecasts.~~ Replaced 2026-09-24 by
  the GitHub Actions pipeline (section 5), which needs no outside access.
