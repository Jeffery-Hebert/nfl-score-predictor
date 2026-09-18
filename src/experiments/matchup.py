"""
src/experiments/matchup.py

Explicit offense-versus-defense interaction terms.

EXPERIMENTAL. Writes nothing; production is untouched.

--------------------------------------------------------------------- the gap

Ridge is additive. It computes

    home_score = b0 + b1*(home passing) + b2*(away pass defence) + ...

so an elite passing attack against a leaky secondary gets `high + high`. The two
effects ADD and can never compound. If the matchup is worth more than the sum of
its parts -- the usual football claim -- Ridge has no way to say so.

Poisson already does, which is easy to miss and may be why it leads the
benchmark. A log link means

    E[score] = exp(b0 + b1x1 + b2x2 + ...) = exp(b0)*exp(b1x1)*exp(b2x2)*...

so feature effects MULTIPLY on the predicted score. The GP can represent
interactions too, implicitly, through its kernel.

That yields a falsifiable prediction rather than a hope: explicit product terms
should help RIDGE materially more than POISSON, because Poisson is already
getting the multiplicative structure for free. If both improve equally, the
terms are doing something other than what this module claims, and if Poisson
improves more, the story is wrong.

--------------------------------------------------------------- what is added

Four columns, one per (side, play type). Each pairs an offence against the
defence that will actually face it:

    home_pass_matchup = home_off_pass_epa  x  away_def_pass_epa_allowed
    home_rush_matchup = home_off_rush_epa  x  away_def_rush_epa_allowed
    away_pass_matchup = away_off_pass_epa  x  home_def_pass_epa_allowed
    away_rush_matchup = away_off_rush_epa  x  home_def_rush_epa_allowed

Note the pairing is deliberate and the obvious alternative is wrong: a home
passing attack meets the AWAY pass defence, not the home one. Getting that
backwards produces a column that is perfectly well-behaved, correlates with
everything, and means nothing -- so tests/test_matchup_terms.py checks the
orientation directly rather than trusting the naming.

ON CENTERING. The products are formed from the raw shrunk rates, which sit near
zero by construction (league EPA/play is ~+0.02 passing, ~-0.02 rushing) but are
not exactly centered. For an UNCENTERED product,

    x*y = (xbar+dx)(ybar+dy) = xbar*ybar + xbar*dy + ybar*dx + dx*dy

the first three terms are a constant and two linear terms, all of which the
model already carries as main effects. Only dx*dy is new, and a linear model can
separate it because the main effects are present. Centering per fold would be
tidier; it would also mean the column depends on the training split, which is a
larger complication than the problem deserves.

BOUNDED BY DESIGN. The inputs are shrunk toward the league mean, so the products
inherit that -- a thin-sample team cannot manufacture a large interaction from
two noisy extremes. That is the specific failure mode a raw-EPA interaction
would have had, and the reason this is worth trying now rather than before the
split shipped.
"""

from __future__ import annotations

import pandas as pd

# (output column, offence column, defence column facing it)
MATCHUP_TERMS = [
    (
        "home_pass_matchup",
        "home_pregame_off_pass_epa_shrunk",
        "away_pregame_def_pass_epa_allowed_shrunk",
    ),
    (
        "home_rush_matchup",
        "home_pregame_off_rush_epa_shrunk",
        "away_pregame_def_rush_epa_allowed_shrunk",
    ),
    (
        "away_pass_matchup",
        "away_pregame_off_pass_epa_shrunk",
        "home_pregame_def_pass_epa_allowed_shrunk",
    ),
    (
        "away_rush_matchup",
        "away_pregame_off_rush_epa_shrunk",
        "home_pregame_def_rush_epa_allowed_shrunk",
    ),
]

MATCHUP_COLS = [c for c, _, _ in MATCHUP_TERMS]

# A second, more speculative arm: does an injury hurt more against a strong
# opponent? Kept separate so it cannot quietly ride along on the first arm's
# result -- this project has been burned by bundled features before.
INJURY_TERMS = [
    (
        "home_injury_vs_opp_pass",
        "home_injury_impact",
        "away_pregame_off_pass_epa_shrunk",
    ),
    (
        "away_injury_vs_opp_pass",
        "away_injury_impact",
        "home_pregame_off_pass_epa_shrunk",
    ),
]

INJURY_COLS = [c for c, _, _ in INJURY_TERMS]


def add_matchup_terms(df: pd.DataFrame, terms=None) -> pd.DataFrame:
    """Return a copy of `df` with the interaction columns appended.

    Raises on a missing input rather than silently producing NaN: a matchup
    column full of NaN would be imputed to the training mean by every model
    here, which means it would quietly become a constant and contribute nothing
    while still looking present in the feature list.
    """
    terms = MATCHUP_TERMS if terms is None else terms
    out = df.copy()
    for name, off_col, def_col in terms:
        missing = [c for c in (off_col, def_col) if c not in out.columns]
        if missing:
            raise KeyError(
                f"cannot build {name}: {missing} absent from the table. "
                "Interaction terms require the shrunk split features."
            )
        out[name] = out[off_col] * out[def_col]
    return out
