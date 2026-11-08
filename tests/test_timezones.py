"""
The project's one time zone: a canonical name, resolved from the pipeline's own
copy of the time-zone database.

Why. Every scheduled run from 2026-10-30 to 2026-11-08 -- nineteen of them --
died on its first line of real work: ZoneInfoNotFoundError for "US/Eastern".
GitHub's ubuntu-latest had moved to Ubuntu 26.04, whose tzdata package no
longer carries the legacy US/* aliases, and pandas 3 installs the tzdata Python
package only on Windows, so on Linux zoneinfo read the OS database. Nothing in
the repository changed; the floor under it did, and a week of forecasts went
unmade while every unit test stayed green.

Two rules, either of which alone would have prevented it:
  1. the name is canonical -- a Zone in the tz database, not a Link (alias) --
     and no code names an alias anywhere;
  2. it resolves with the OS database hidden, from the pinned tzdata package,
     so a run never depends on what the runner image happens to ship.

Run: pytest tests/test_timezones.py -v
"""

import ast
import zoneinfo
from importlib import resources
from pathlib import Path

import pytest

from src.schedule import EASTERN

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tz_database():
    """Zones and links, from the tzdata package's own source summary
    (tzdata.zi: 'Z <name> ...' for a zone, 'L <target> <alias>' for a link)."""
    text = (resources.files("tzdata") / "zoneinfo" / "tzdata.zi").read_text()
    zones, links = set(), {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Z":
            zones.add(parts[1])
        elif len(parts) == 3 and parts[0] == "L":
            links[parts[2]] = parts[1]
    return zones, links


def test_the_zone_is_canonical(tz_database):
    zones, links = tz_database
    assert EASTERN not in links, f"{EASTERN} is an alias of {links[EASTERN]}"
    assert EASTERN in zones


def test_it_resolves_without_the_os_database():
    # An empty search path leaves zoneinfo nothing but the tzdata package --
    # what a runner image without the name looks like. Missing package or
    # missing name, this raises ZoneInfoNotFoundError as the pipeline did.
    zoneinfo.reset_tzpath(to=[])
    try:
        zone = zoneinfo.ZoneInfo.no_cache(EASTERN)
    finally:
        zoneinfo.reset_tzpath()
    assert zone.key == EASTERN


def python_strings(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def test_no_code_names_a_time_zone_alias(tz_database):
    _, links = tz_database
    # Area/Location aliases only. The bare ones collide with ordinary strings
    # -- "GB" is Green Bay here, not Europe/London -- and the one that matters,
    # "UTC", pandas maps to datetime.timezone.utc without asking the database.
    aliases = {name for name in links if "/" in name}
    found = sorted(
        f"{path.relative_to(ROOT)}: {s!r} (use {links[s]!r})"
        for folder in ("src", "tests")
        for path in (ROOT / folder).rglob("*.py")
        for s in python_strings(path)
        if s in aliases
    )
    assert not found, "legacy time-zone aliases:\n" + "\n".join(found)
