"""
B3: the feature-name contract.

BASE_FEATURE_COLS (per-team naming) and FEATURE_COLS (per-game home_/away_
naming) must stay consistent with each other and with what the feature
pipeline actually produces. These were two hand-maintained lists in two files
until 2026-09-14; this pins the contract that replaced them.

Run: pytest tests/test_feature_cols.py -v
"""

import pytest

from src.models.common import (
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    GAME_FEATURE_COLS,
    INJURY_FEATURE_COLS,
    SPLIT_FEATURE_COLS,
)


def test_feature_cols_is_base_prefixed_home_then_away_plus_game_level():
    expected = (
        [f"home_{c}" for c in BASE_FEATURE_COLS]
        + [f"away_{c}" for c in BASE_FEATURE_COLS]
        + [f"home_{c}" for c in INJURY_FEATURE_COLS]
        + [f"away_{c}" for c in INJURY_FEATURE_COLS]
        + [f"home_{c}" for c in SPLIT_FEATURE_COLS]
        + [f"away_{c}" for c in SPLIT_FEATURE_COLS]
        + GAME_FEATURE_COLS
    )
    assert FEATURE_COLS == expected


def test_blended_epa_is_not_carried_alongside_the_split():
    """The split REPLACES blended EPA; the two must not both be present.

    Blended off/def EPA per play and the pass/rush split describe the same
    efficiency at different grains -- the blend correlates 0.86 with the shrunk
    pass split and 0.58 with the rush split (0.78 / 0.45 on defence). Carrying
    both puts two descriptions of one quantity in front of an additive model.

    The split shipped WITH the blend on 2026-09-17 and the blend was removed the
    same day by operator decision. This pins that decision so the blend cannot
    drift back in as a well-meaning 'the split is noisy, keep the stable one
    too' -- the noise is handled by shrinkage inside build_split_efficiency.py.
    """
    for side in ("home", "away"):
        assert f"{side}_pregame_off_epa_per_play" not in FEATURE_COLS
        assert f"{side}_pregame_def_epa_per_play" not in FEATURE_COLS
        # ...and the split that replaced it is actually there.
        assert f"{side}_pregame_off_pass_epa_shrunk" in FEATURE_COLS
        assert f"{side}_pregame_off_rush_epa_shrunk" in FEATURE_COLS
        assert f"{side}_pregame_def_pass_epa_allowed_shrunk" in FEATURE_COLS
        assert f"{side}_pregame_def_rush_epa_allowed_shrunk" in FEATURE_COLS


def test_efficiency_is_described_exactly_once_per_side_per_unit():
    """Generalises the above: for each side and each unit, count how many EPA
    efficiency columns describe it. Two is duplication, zero is a missing
    signal."""
    for side in ("home", "away"):
        for unit in ("off", "def"):
            epa_cols = [
                c
                for c in FEATURE_COLS
                if c.startswith(f"{side}_pregame_{unit}") and "epa" in c
            ]
            assert len(epa_cols) == 2, (
                f"{side}/{unit} is described by {len(epa_cols)} EPA columns "
                f"({epa_cols}) -- expected exactly the pass and rush split"
            )


def test_split_columns_are_the_shrunk_ones_not_the_raw_rolling_ones():
    """build_rolling_features.py still emits pregame_off_pass_epa_per_play and
    friends -- the v1, game-equal-weighted, unshrunk versions kept only for the
    archived experiments. Those must never be what the models train on."""
    raw_v1 = {
        "pregame_off_pass_epa_per_play",
        "pregame_off_rush_epa_per_play",
        "pregame_def_pass_epa_per_play_allowed",
        "pregame_def_rush_epa_per_play_allowed",
    }
    for col in FEATURE_COLS:
        stripped = col.replace("home_", "", 1).replace("away_", "", 1)
        assert stripped not in raw_v1, (
            f"{col} is the unshrunk v1 column from build_rolling_features.py, "
            "not the volume-weighted one from build_split_efficiency.py"
        )


def test_game_level_features_are_not_sided():
    """is_neutral_site / is_playoff describe the fixture, not a team, so they
    must not carry a home_/away_ prefix."""
    for c in GAME_FEATURE_COLS:
        assert not c.startswith(("home_", "away_")), f"{c} should not be sided"
        assert c in FEATURE_COLS


def test_overtime_is_never_a_feature():
    """C5 guard. went_to_ot is 0% populated before kickoff; using it as an input
    would be target leakage. It is a training-side sample weight only."""
    from src.models.common import OT_TRAINING_COL

    assert OT_TRAINING_COL not in FEATURE_COLS
    assert not any("overtime" in c or "went_to_ot" in c for c in FEATURE_COLS)


def test_home_and_away_sides_are_symmetric():
    """Every home_ feature needs its away_ twin, or the model sees one side of
    the matchup in more detail than the other."""
    home = {c[len("home_") :] for c in FEATURE_COLS if c.startswith("home_")}
    away = {c[len("away_") :] for c in FEATURE_COLS if c.startswith("away_")}
    assert home == away, f"asymmetric: {home ^ away}"


def test_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert len(BASE_FEATURE_COLS) == len(set(BASE_FEATURE_COLS))


def test_build_game_features_uses_the_shared_list():
    """It must import the list, not redeclare it -- the drift this replaced."""
    from src.features import build_game_features as bgf

    assert bgf.FEATURE_COLS is BASE_FEATURE_COLS, (
        "build_game_features.py has stopped using the shared list from "
        "src/models/common.py -- the two definitions can now drift apart again"
    )


@pytest.mark.parametrize("col", FEATURE_COLS)
def test_every_feature_is_pregame_or_a_known_non_stat(col):
    """Guards against a post-game stat entering the model feature set."""
    stripped = col.replace("home_", "", 1).replace("away_", "", 1)
    # rest_days and prior_games_played come from the schedule, not from play
    # data; is_neutral_site/is_playoff are fixture context. All four are known
    # before kickoff.
    # All known before kickoff: rest_days and prior_games_played come from the
    # schedule, is_neutral_site/is_playoff are fixture context, and
    # injury_impact comes from the official injury report, which publishes a
    # median 49 hours before kickoff (see build_injury_features.py).
    allowed_non_pregame = {
        "rest_days",
        "prior_games_played",
        "is_neutral_site",
        "is_playoff",
        *INJURY_FEATURE_COLS,
    }
    assert (
        stripped.startswith("pregame_") or stripped in allowed_non_pregame
    ), f"{col} is neither a pregame_ stat nor a known schedule-derived column"
