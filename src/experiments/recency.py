"""
src/experiments/recency.py

Recency weighting, generalised. Shared machinery for the experiments that
question how this project ages its evidence.

EXPERIMENTAL. Nothing in production imports this. Production still uses the
single 17-week exponential in build_rolling_features.py.

--------------------------------------------------------------- why generalise

Three separate complaints about the current scheme, all of which reduce to the
same limitation -- there is exactly one decay rate and it is applied to
everything:

  1. PER-STAT RATES. Persistence varies threefold across the stats we keep
     (split-half r over 224 team-seasons):

         off success rate            0.695
         off pass EPA/play           0.510
         off rush EPA/play           0.415
         def rush EPA/play allowed   0.294
         def pass EPA/play allowed   0.227

     A trait that repeats at 0.70 and one that repeats at 0.23 have no business
     sharing a memory length. The stable one deserves a long look-back; the
     noisy one deserves a short one, because old noise is still noise.

  2. SHAPE. `tune_halflife.py` swept 4 to 52 weeks and found 17 optimal, so
     "weight recent games more" has been tested as a SINGLE exponential and
     lost. But a single exponential cannot be both steep near the present and
     long-tailed; it has one parameter. A two-timescale mixture can, and that
     family has never been tried:

         w = a * 0.5^(age/H_fast) + (1-a) * 0.5^(age/H_slow)

     which is the natural way to say "recent form AND underlying talent" rather
     than blending them into one rate.

  3. ANCHORING -- and a correction. It looked like production had a defect
     here: it computes a time-based EWM at every row and then calls .shift(1),
     which is POSITIONAL, so the weights are anchored at the PREVIOUS game's
     date rather than the date of the game being predicted.

     That turns out not to matter, for a reason worth writing down. A single
     exponential is SHIFT-INVARIANT under a normalised mean:

         w_i = 0.5^((A - t_i)/H) = 0.5^(A/H) * 0.5^(-t_i/H)

     The 0.5^(A/H) factor is the same for every observation, so it cancels top
     and bottom. Moving the anchor rescales every weight identically and the
     weighted mean does not move at all. Production's .shift(1) is harmless.

     It stops being harmless the moment the kernel is a MIXTURE. Two
     exponentials pick up two different constants, 0.5^(A/H_fast) and
     0.5^(A/H_slow), which do not cancel -- so the fast/slow balance depends on
     where you measure age from. For TwoTimescale the anchor is a real choice,
     and "as of the game being predicted" is the defensible one. Hence the
     parameter: it is a no-op for Exponential and load-bearing for mixtures.

------------------------------------------------------------------ correctness

The weighted mean is computed as an explicit n x n matrix rather than an
incremental accumulator. A team plays ~140 games, so the matrix is ~20k entries
and the cost is irrelevant, while the strictly-prior rule becomes a visible
triangular mask instead of an off-by-one waiting to happen. build_split_efficiency
uses accumulators because it walks the whole league; here it would buy nothing.

WHAT THIS COMPUTES, EXACTLY: the kernel-weighted mean of unmasked prior
observations, weighted by real elapsed time. That is pandas' `ewm(...,
ignore_na=False)`, and tests assert the equivalence.

Production passes `ignore_na=TRUE`, which is a subtly different convention --
NaN rows change how the remaining weights normalise. On real data the two differ
on 88% of rows by a mean of 0.012 (max 0.31) against a team_score sd of ~9.9, so
roughly a thousandth of a standard deviation. Immaterial, but not zero, and it
is the reason the experiments here use production's OWN function as the control
arm rather than this one with an Exponential. A control that is 0.012 away from
production is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------- kernels


@dataclass(frozen=True)
class Exponential:
    """The production scheme: one half-life, applied to everything."""

    halflife_days: float

    def __call__(self, age_days: np.ndarray) -> np.ndarray:
        return 0.5 ** (np.clip(age_days, 0.0, None) / self.halflife_days)

    def describe(self) -> str:
        return f"exp(H={self.halflife_days / 7:.0f}w)"


@dataclass(frozen=True)
class TwoTimescale:
    """A fast component for current form plus a slow one for underlying class.

    weight_fast is the share of total mass on the fast component AT AGE ZERO.
    The two components are each normalised to 1.0 at age zero, so the mixture
    starts at 1.0 and decays faster than either component alone at first, then
    flattens onto the slow tail. That is the shape a single exponential cannot
    produce and the reason this family is worth testing separately.
    """

    halflife_fast_days: float
    halflife_slow_days: float
    weight_fast: float

    def __post_init__(self):
        if not 0.0 <= self.weight_fast <= 1.0:
            raise ValueError(f"weight_fast must be in [0,1], got {self.weight_fast}")
        if self.halflife_fast_days >= self.halflife_slow_days:
            raise ValueError(
                "halflife_fast_days must be shorter than halflife_slow_days; got "
                f"{self.halflife_fast_days} >= {self.halflife_slow_days}. A mixture "
                "where the 'fast' component decays slower is just a confusing "
                "single exponential."
            )

    def __call__(self, age_days: np.ndarray) -> np.ndarray:
        a = np.clip(age_days, 0.0, None)
        fast = 0.5 ** (a / self.halflife_fast_days)
        slow = 0.5 ** (a / self.halflife_slow_days)
        return self.weight_fast * fast + (1.0 - self.weight_fast) * slow

    def describe(self) -> str:
        return (
            f"two({self.halflife_fast_days / 7:.0f}w/"
            f"{self.halflife_slow_days / 7:.0f}w@{self.weight_fast:.0%})"
        )


# --------------------------------------------------------------- weighted mean


def prior_weighted_mean(
    values: np.ndarray,
    times: np.ndarray,
    kernel,
    include: np.ndarray | None = None,
    anchor: str = "current",
) -> np.ndarray:
    """Kernel-weighted mean over STRICTLY PRIOR observations, per row.

    values   the stat, NaN where unobserved
    times    datetime64, assumed sorted ascending (one team's games)
    kernel   callable age_days -> weight
    include  which rows may CONTRIBUTE to later rows. Rows excluded here still
             receive a value; they just never feed forward. This is how
             finale-week masking works.
    anchor   "current"  weights are measured from the row's own date -- form as
                        of the game being predicted.
             "previous" weights are measured from the previous row's date.
                        Reproduces production (a time-based ewm followed by a
                        positional shift). Kept so the two can be compared
                        without also changing the implementation.

    Returns NaN where a row has no prior evidence at all.
    """
    if anchor not in ("current", "previous"):
        raise ValueError(f"anchor must be 'current' or 'previous', got {anchor!r}")

    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    v = np.asarray(values, dtype=float)
    t_days = (
        np.asarray(times, dtype="datetime64[ns]")
        - np.asarray(times, "datetime64[ns]")[0]
    ) / np.timedelta64(1, "D")
    t_days = t_days.astype(float)

    usable = np.isfinite(v)
    if include is not None:
        usable = usable & np.asarray(include, dtype=bool)

    # Anchor time per row.
    if anchor == "current":
        anchors = t_days
    else:
        anchors = np.empty(n)
        anchors[0] = t_days[0]
        anchors[1:] = t_days[:-1]

    # ages[k, i] = how old observation i is, seen from row k's anchor.
    ages = anchors[:, None] - t_days[None, :]

    # Strictly prior BY TIME, not by position.
    #
    # For one team's schedule the two agree -- a team plays once a day -- but
    # this function is also used across the whole league, where a dozen games
    # share a Sunday. A positional mask would let the earlier-SORTED of two
    # simultaneous games feed the later-sorted one, which is a leak that depends
    # on nothing but sort order. Comparing timestamps is correct in both cases.
    prior = t_days[None, :] < t_days[:, None]
    mask = prior & usable[None, :]

    w = np.where(mask, kernel(ages), 0.0)
    denom = w.sum(axis=1)
    numer = w @ np.nan_to_num(v, nan=0.0)

    nonzero = denom > 0
    out[nonzero] = numer[nonzero] / denom[nonzero]
    return out


# ------------------------------------------------------------ per-stat config


@dataclass(frozen=True)
class RecencyPlan:
    """Which kernel governs which stat.

    `default` applies to anything not named in `per_stat`. A plan with an empty
    `per_stat` and an Exponential default is the production scheme, which makes
    it the natural control in every comparison.
    """

    default: object
    per_stat: dict | None = None
    anchor: str = "current"

    def kernel_for(self, stat: str):
        if self.per_stat and stat in self.per_stat:
            return self.per_stat[stat]
        return self.default

    def describe(self) -> str:
        base = f"{self.default.describe()} anchor={self.anchor}"
        if self.per_stat:
            base += f" +{len(self.per_stat)} per-stat"
        return base


PRODUCTION_HALFLIFE_DAYS = 17 * 7


def production_plan() -> RecencyPlan:
    """Exactly what production does today, for use as the control arm."""
    return RecencyPlan(
        default=Exponential(PRODUCTION_HALFLIFE_DAYS), per_stat=None, anchor="previous"
    )


def prior_weighted_rate(
    numerator: np.ndarray,
    denominator: np.ndarray,
    times: np.ndarray,
    kernel,
    include: np.ndarray | None = None,
    anchor: str = "current",
):
    """Kernel-weighted RATE over strictly prior observations, plus its effective
    denominator.

    The volume-weighted counterpart of prior_weighted_mean, for quantities that
    are a ratio rather than an average of averages:

        rate = sum(w * numerator) / sum(w * denominator)

    This is what the pass/rush splits need -- a rate per PLAY, so a 38-carry
    game outweighs a 9-carry game. prior_weighted_mean would give them equal say.

    Returns (rate, n_eff). n_eff is the recency-weighted denominator actually
    behind each estimate, which is what shrinkage is sized by. It is 0.0 where
    there is no prior evidence, never NaN, so it is always safe to compare.
    """
    n = len(numerator)
    rate = np.full(n, np.nan)
    n_eff = np.zeros(n)
    if n == 0:
        return rate, n_eff

    num = np.nan_to_num(np.asarray(numerator, float), nan=0.0)
    den = np.nan_to_num(np.asarray(denominator, float), nan=0.0)
    t = np.asarray(times, dtype="datetime64[ns]")
    t_days = ((t - t[0]) / np.timedelta64(1, "D")).astype(float)

    usable = np.ones(n, dtype=bool) if include is None else np.asarray(include, bool)

    anchors = (
        t_days if anchor == "current" else np.concatenate([[t_days[0]], t_days[:-1]])
    )
    ages = anchors[:, None] - t_days[None, :]

    # Strictly prior BY TIME -- see prior_weighted_mean. This function is used
    # for the league-wide prior, where simultaneous games must not inform each
    # other, so a positional mask would leak.
    mask = (t_days[None, :] < t_days[:, None]) & usable[None, :]
    w = np.where(mask, kernel(ages), 0.0)

    wn = w @ num
    wd = w @ den
    nonzero = wd > 0
    rate[nonzero] = wn[nonzero] / wd[nonzero]
    n_eff = wd
    return rate, n_eff


def shrink_to_prior(team_rate, n_eff, prior_rate, k: float):
    """Pull a rate toward a prior in proportion to thin evidence.

    Identical to build_split_efficiency.shrink; duplicated here so the
    experimental path has no import dependency on a production module whose
    behaviour it is trying to vary.
    """
    team_rate = np.where(np.isnan(team_rate), 0.0, team_rate)
    return (n_eff * team_rate + k * prior_rate) / (n_eff + k)
