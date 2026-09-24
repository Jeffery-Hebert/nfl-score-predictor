"""
src/experiments/continuity.py

Roster continuity: a measurable proxy for the things people mean by "chemistry".

EXPERIMENTAL. Writes nothing; production is untouched.

--------------------------------------------------------------------- the idea

Chemistry, morale and cohesion are not in any feed and never will be. But one
component of them IS observable: whether the eleven players on the field have
played together before. A unit that returned intact from last season is not the
same proposition as one rebuilt from free agents and replacements, and the box
score does not distinguish them until the results arrive.

This is the only item in this line of work that adds genuinely NEW information
rather than re-cutting play-by-play. Every previous attempt at the latter
produced a null (five documented), which is itself the argument for trying the
former.

Three measures, each from snap counts alone:

  roster_carryover     share of THIS season's snaps so far taken by players who
                       also took snaps for this team LAST season. Cross-season
                       continuity -- the "did the band stay together" number.

  lineup_stability     average overlap between the snap-weighted personnel of
                       consecutive prior games. Within-season churn, which is
                       what injuries and experimentation actually look like week
                       to week. Distinct from carryover: a team can be entirely
                       returning players and still be shuffling its line-up.

  snap_weighted_tenure mean number of prior games these contributors have
                       played, weighted by how much they are playing now. An
                       experience proxy. Censored at 2019 because that is where
                       the data starts, so it understates veterans -- which is
                       fine for a within-era comparison and stated here rather
                       than discovered later.

--------------------------------------------------------------------- leakage

Snap counts are POST-game data. They are only ever read from games that have
already finished, and every measure for game N is computed from games strictly
before N. That is the same position the injury features occupy: post-game data
is safe precisely when it is used to describe the past.

The trap, and it is the one that made qb_out identically zero the first time
around: a player who is unavailable for game N has no snap row for game N. Any
measure that asks "who is on the roster" by looking at the current game's snaps
is both leaking and wrong. Everything here asks only about PRIOR games.

Week 1 of a season has no prior games in that season, so carryover and stability
are undefined there and are returned as NaN rather than as a fabricated default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNAPS = "data/raw/snap_counts.parquet"
SCHEDULES = "data/raw/schedules.parquet"

CONTINUITY_COLS = [
    "roster_carryover",
    "lineup_stability",
    "snap_weighted_tenure",
]

# A player counts as a contributor to a game if he took at least this share of
# either offensive or defensive snaps. Below it he is a special-teamer or a
# late-game substitute, and including him would let the tail of the roster swamp
# the signal from the starters.
CONTRIBUTOR_THRESHOLD = 0.10


def _load_snaps() -> pd.DataFrame:
    s = pd.read_parquet(SNAPS)
    s["role_share"] = s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    s = s[s["role_share"] >= CONTRIBUTOR_THRESHOLD].copy()
    s["team"] = s["team"].replace({"OAK": "LV"})
    sched = pd.read_parquet(SCHEDULES)[["game_id", "gameday"]]
    s = s.merge(sched, on="game_id", how="left")
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s.dropna(subset=["gameday", "pfr_player_id"])


def _weighted_jaccard(a: dict, b: dict) -> float:
    """Overlap between two snap-weighted line-ups, in [0, 1].

    sum(min) / sum(max) over the union of players. Two identical line-ups give
    1.0; two with no one in common give 0.0. Weighting by snap share means
    swapping a starter costs far more than swapping a rotational player, which
    is the distinction a plain set overlap would miss.
    """
    if not a or not b:
        return np.nan
    keys = set(a) | set(b)
    lo = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    hi = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return lo / hi if hi > 0 else np.nan


def build_continuity(snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game_id, team) with the three continuity measures.

    Every value is computed from that team's games STRICTLY BEFORE the row's
    own game.
    """
    s = _load_snaps() if snaps is None else snaps
    rows = []

    for team, tg in s.groupby("team"):
        # Ordered list of that team's games, each with its snap-weighted line-up.
        games = (
            tg.groupby(["game_id", "season", "gameday"])
            .apply(
                lambda g: dict(zip(g["pfr_player_id"], g["role_share"])),
                include_groups=False,
            )
            .reset_index(name="lineup")
            .sort_values("gameday")
            .reset_index(drop=True)
        )

        # Career appearances per player, accumulated as we walk forward, so a
        # row only ever sees counts from before it.
        appearances: dict[str, int] = {}
        # Which players took snaps for this team in each season, for carryover.
        by_season: dict[int, set] = {}

        for i, row in games.iterrows():
            season = int(row["season"])
            prior = games.iloc[:i]
            prior_this_season = prior[prior["season"] == season]
            # This game's own line-up. Used ONLY to update the running history
            # after the row's values have been computed -- never to compute them.
            lu = row["lineup"]

            # --- carryover: prior snaps THIS season taken by players who were
            # here LAST season.
            carry = np.nan
            last_season_squad = by_season.get(season - 1)
            if len(prior_this_season) and last_season_squad:
                total = 0.0
                returning = 0.0
                # NOT `lu`: that name holds THIS game's line-up for the history
                # update below. Reusing it here (the original code did) left it
                # pointing at the previous game, so from the second season on
                # each game entered the history one game late, week 1 counted
                # twice and the season's last game never counted -- tenure and
                # carryover were both wrong (fixed 2026-09-24; see the tests).
                for prior_lu in prior_this_season["lineup"]:
                    for pid, share in prior_lu.items():
                        total += share
                        if pid in last_season_squad:
                            returning += share
                carry = returning / total if total > 0 else np.nan

            # --- stability: mean overlap between consecutive prior line-ups
            stab = np.nan
            if len(prior_this_season) >= 2:
                lus = list(prior_this_season["lineup"])
                pairs = [
                    _weighted_jaccard(lus[j], lus[j + 1]) for j in range(len(lus) - 1)
                ]
                pairs = [p for p in pairs if np.isfinite(p)]
                stab = float(np.mean(pairs)) if pairs else np.nan

            # --- tenure: how experienced the likely contributors are.
            #
            # Weighted by the MOST RECENT PRIOR line-up, not this game's. Using
            # this game's would be a second kind of leak and, worse, one that
            # cannot be served live: on Wednesday nobody knows who will take
            # snaps on Sunday. That is exactly what shelved the QB-identity
            # feature -- it worked historically and had no pre-game source.
            # "Whoever played last week" is both knowable in advance and a good
            # approximation, since lineup_stability above measures the overlap
            # at ~0.74.
            tenure = np.nan
            if len(prior) and appearances:
                recent_lu = prior.iloc[-1]["lineup"]
                den = sum(recent_lu.values())
                if den > 0:
                    tenure = (
                        sum(
                            share * appearances.get(pid, 0)
                            for pid, share in recent_lu.items()
                        )
                        / den
                    )

            rows.append(
                {
                    "game_id": row["game_id"],
                    "team": team,
                    "roster_carryover": carry,
                    "lineup_stability": stab,
                    "snap_weighted_tenure": tenure,
                }
            )

            # Only NOW does this game become history.
            for pid in lu:
                appearances[pid] = appearances.get(pid, 0) + 1
            by_season.setdefault(season, set()).update(lu.keys())

    return pd.DataFrame(rows)


def add_continuity(df: pd.DataFrame, continuity: pd.DataFrame) -> pd.DataFrame:
    """Join continuity onto a game-level table as home_/away_ columns."""
    out = df.copy()
    for side in ("home", "away"):
        cols = continuity[["game_id", "team"] + CONTINUITY_COLS].rename(
            columns={
                **{c: f"{side}_{c}" for c in CONTINUITY_COLS},
                "team": f"{side}_team",
            }
        )
        out = out.merge(cols, on=["game_id", f"{side}_team"], how="left")
    return out


def main():
    c = build_continuity()
    print(f"Built continuity for {len(c)} team-game rows\n")
    for col in CONTINUITY_COLS:
        v = c[col].dropna()
        print(
            f"  {col:22s} n={len(v):5d}  mean {v.mean():7.3f}  sd {v.std():6.3f}  "
            f"range [{v.min():.3f}, {v.max():.3f}]  null {c[col].isna().mean():.1%}"
        )


if __name__ == "__main__":
    main()
