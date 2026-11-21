"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Round 2: a COUNT model. The earn rate of a trophy is a ratio,
    rate = earned / players_tracked,
and this program forecasts the two counts separately instead of fitting
the ratio itself, because the two counts move for different reasons:

  * players_tracked only changes when Exophase refreshes the game's
    counts: it sits flat between refreshes and jumps at a refresh, as
    the flood of newly tracked players is counted at once. The program
    detects those jumps in the case history (tracked count changing
    between reads), measures their cadence and size, and projects the
    tracked count to the target from that. With no jump ever seen, the
    snapshot is stale: the page has been unchanged since the first read,
    so a refresh is treated as due, with a prior interval and jump size.
  * earned only changes at a refresh too (the page is frozen between
    refreshes), but it grows as players actually progress through the
    game, trophy by trophy. Each trophy's earned count is fitted on a
    log-time axis over the recent history, and the fitted move from the
    last count is shrunk toward flat: a slope is cheap to believe over
    a short horizon and expensive over a week. The result is capped by
    a ceiling that depends on the trophy's kind, read from its own name
    and description (a chapter trophy can plausibly reach tens of
    percent in a week, a grind or collectible trophy cannot, and the
    platinum / 100%-completion trophies stay well under 1%).

The median rate is earned / tracked at the target. The spread, on the
logit scale, combines the earned fit's prediction error with the
refresh uncertainty (how many refreshes before the target, and how big
each jump is), calibrated from the refreshes actually observed in the
history. When few refreshes are expected before the target, the forecast
is a mixture: with some probability no refresh comes and the rate stays
where the last read put it; otherwise it moves to the count model's
value. The five quantiles are the mixture's quantiles then; with many
refreshes expected, the count model's value with a normal spread.

One trophy is special: the one every tracked player holds (earned ==
tracked in every read). It has survived dilution at 100% before, so it
is forecast in a tight band just under 100%, not by the count fit.

Python 3.13, standard library only, no network, no randomness: two runs
of the same case write the same bytes. Reads every *.in file in the
input directory and writes one *.out file per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# How far back the trend fits look (hours before the checkpoint).
FIT_WINDOW_HOURS = 48.0
# A tracked count may not be projected more than this many times its last
# value: a guard against a wild fit, not a forecast.
MAX_TRACKED_GROWTH = 100.0
# The jump projection only ever spans this many refreshes: jump sizes
# shrink as the tracked base grows, so compounding one median jump over a
# whole week would explode. The trend fit carries long horizons.
MAX_JUMP_SPAN = 3.0
# Below this many expected refreshes the forecast is a mixture (a refresh
# may or may not come); above it, one normal around the count model.
MIXTURE_MAX_REFRESHES = 2.5
# Prior refresh interval (hours) and median jump (log units) when the
# history has never shown a refresh: Exophase refreshes a hot game a few
# times a day, and a launch-night refresh reveals several times the
# pre-release tracked base.
PRIOR_INTERVAL_HOURS = 4.0
PRIOR_JUMP_LOG = 1.3862943611198906  # ln(4)
PRIOR_JUMP_SD = 0.7
# Spread floors and caps (logit units).
SIGMA_FLOOR = 0.45
SIGMA_CAP = 2.0
TREND_SIGMA_FLOOR = 0.25
# Per-refresh log-variance of the earned count that the tracked jump does
# not explain (0.3 log units): earned and tracked jump together at a
# refresh, so the ratio moves less than either count alone.
EARNED_JUMP_VAR = 0.09


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y > 35.0:
        return 100.0
    if y < -35.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """Round to 4 decimals, halves up, as JavaScript's Math.round does."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Standard normal quantile by bisection on the erf CDF."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def least_squares(points):
    """Fit y = a + b*x. Returns (a, b, residual_sd, n, x_mean, sxx)."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0, 0, 0.0, 0.0)
    x_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    sxx = sum((p[0] - x_mean) ** 2 for p in points)
    if n < 3 or sxx < 1e-9:
        return (y_mean, 0.0, 0.0, n, x_mean, sxx)
    sxy = sum((p[0] - x_mean) * (p[1] - y_mean) for p in points)
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    resid = math.sqrt(
        sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
        / (n - 2))
    return (intercept, slope, resid, n, x_mean, sxx)


# (kind, ceiling percent, prior earned growth per refresh) per trophy,
# from its own name and description. Ceilings cap the count fit, never
# set it: a trophy the data says is slow stays slow. The prior growth is
# used only when no refresh has ever been observed, so the earned count
# has no measured growth at all.
KINDS = [
    ("platinum", 0.6, 1.02),
    ("completion", 0.5, 1.02),
    ("chapter1", 45.0, 1.30),
    ("chapter2", 35.0, 1.25),
    ("chapter3", 28.0, 1.20),
    ("chapter4", 22.0, 1.15),
    ("final-mission", 10.0, 1.15),
    ("collectible", 10.0, 1.05),
    ("grind", 8.0, 1.05),
    ("mission", 15.0, 1.20),
    ("activity", 18.0, 1.10),
    ("misc", 12.0, 1.10),
]


def classify(trophy):
    """Return (kind, ceiling percent, prior earned growth per refresh)."""
    desc = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return KINDS[0]
    if "100% completion" in desc:
        return KINDS[1]
    for chapter, ceiling, growth in (
            ("chapter 1", 45.0, 1.30), ("chapter 2", 35.0, 1.25),
            ("chapter 3", 28.0, 1.20), ("chapter 4", 22.0, 1.15)):
        if "complete " + chapter in desc:
            return (chapter.replace(" ", ""), ceiling, growth)
    if "complete the final mission" in desc:
        return KINDS[6]
    if ("collectible card" in desc or "waink" in desc
            or "discover 100 vehicles" in desc
            or "explore all of leonida" in desc):
        return KINDS[7]
    if ("250 stunts" in desc or "50 challenges" in desc
            or "20 different races" in desc or "spend $" in desc
            or "earn $250,000" in desc or "purchase all garage" in desc
            or "register 10 vehicles" in desc or "gold medal" in desc):
        return KINDS[8]
    if desc.startswith("complete"):
        return KINDS[9]
    if ("win " in desc or "beat " in desc or "try each type" in desc
            or "meditate" in desc or "hunt " in desc):
        return KINDS[10]
    return KINDS[11]


def read_history(case):
    """Turn the case history into [(time, tracked, [earned...])].

    Earned counts are rebuilt from the printed rates: rate * tracked /
    100. The rates are printed to 2 decimals, so a rebuilt count is the
    true integer count to within a fraction of an earner.
    """
    trophies = case.get("trophies") or []
    reads = []
    for read in case.get("history") or []:
        tracked = read.get("players_tracked")
        rates = read.get("rates") or []
        if tracked is None or tracked <= 0 or not rates:
            continue
        try:
            when = parse_time(read["at"])
        except (KeyError, ValueError):
            continue
        earned = []
        for i in range(len(trophies)):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                earned.append(None)
            else:
                earned.append(rate / 100.0 * tracked)
        reads.append((when, tracked, earned))
    return reads


def refresh_stats(reads, checkpoint, horizon_hours):
    """Measure Exophase's refreshes from the tracked-count jumps.

    Returns (expected refreshes before target, P(no refresh), median
    log-jump, log-jump sd). A jump is a read whose tracked count differs
    from the previous read: the page is frozen between refreshes, so the
    count only moves at one.
    """
    jumps = []
    for prev, cur in zip(reads, reads[1:]):
        if cur[1] != prev[1]:
            jumps.append((cur[0], math.log(cur[1] / prev[1])))
    if jumps:
        gaps = [(b[0] - a[0]).total_seconds() / 3600.0
                for a, b in zip(jumps, jumps[1:])]
        interval = median_of(gaps) if gaps else None
        sizes = [size for _, size in jumps]
        jump = median_of(sizes)
        if len(sizes) >= 2:
            mean = sum(sizes) / len(sizes)
            sd = math.sqrt(sum((v - mean) ** 2 for v in sizes)
                           / (len(sizes) - 1))
        else:
            sd = 0.5
        sd = clamp(sd, 0.15, 1.0)
        if interval:
            expected = horizon_hours / interval
        else:
            expected = horizon_hours / PRIOR_INTERVAL_HOURS
        return (expected, math.exp(-expected), jump, sd)

    # No refresh has ever been seen: the snapshot is at least as old as
    # the recording, and a refresh on a roughly regular schedule is due.
    # The time left before it is uniform on what is left of one prior
    # interval after the age already observed.
    age = (checkpoint - reads[0][0]).total_seconds() / 3600.0
    remaining = max(PRIOR_INTERVAL_HOURS - age, 0.5)
    if horizon_hours < remaining:
        p_none = 1.0 - horizon_hours / remaining
        expected = (1.0 - p_none) * (
            1.0 + (horizon_hours - remaining / 2.0)
            / PRIOR_INTERVAL_HOURS)
    else:
        p_none = 0.0
        expected = 1.0 + (horizon_hours - remaining / 2.0) \
            / PRIOR_INTERVAL_HOURS
    return (expected, p_none, PRIOR_JUMP_LOG, PRIOR_JUMP_SD)


def project_tracked(reads, release, target, expected_refreshes, jump_log):
    """Project the tracked count at the target.

    A log-time trend fit carries the projection; the next few refreshes
    put a floor under it, because a refresh can only add tracked players.
    """
    now = reads[-1][1]
    points = []
    for when, tracked, _ in reads:
        hours = (when - release).total_seconds() / 3600.0
        points.append((max(0.0, hours), math.log(tracked)))
    newest = points[-1][0]
    window = [(math.log1p(h), y) for h, y in points
              if newest - h <= FIT_WINDOW_HOURS] or \
        [(math.log1p(h), y) for h, y in points]
    intercept, slope, _, n, _, _ = least_squares(window)
    target_hours = (target - release).total_seconds() / 3600.0
    fitted = math.exp(intercept + slope * math.log1p(max(0.0, target_hours)))
    # Shrink the fitted move from the last count, not the slope: over a
    # week of extrapolation a small slope change is a huge level change.
    shrink = n / (n + 3.0) if n else 0.0
    trend = math.exp(math.log(now)
                     + shrink * (math.log(max(fitted, 1.0)) - math.log(now)))
    trend = clamp(trend, now, now * MAX_TRACKED_GROWTH)
    span = min(expected_refreshes, MAX_JUMP_SPAN)
    if span > 0:
        floor = now * math.exp(jump_log * span)
        return max(trend, floor)
    return trend


def project_earned(index, reads, release, target, kind, ceiling,
                   tracked_target, has_refresh, expected_refreshes):
    """Project one trophy's earned count at the target.

    Returns (earned_target, fit_sigma). The fit runs on the distinct
    earned values only (the count is a step function that moves at
    refreshes), on a log-time axis over the recent window, with the
    fitted move shrunk toward flat. Without any observed refresh there
    is no measured growth, so a kind-based prior growth per refresh
    stands in.
    """
    _, _, prior_growth = kind
    distinct = []
    last = None
    for when, _, earned in reads:
        value = earned[index]
        if value is None:
            continue
        last = value
        hours = (when - release).total_seconds() / 3600.0
        logged = math.log(max(value, 0.5))
        if not distinct or distinct[-1][1] != logged:
            distinct.append((max(0.0, hours), logged))
    if last is None:
        last = 1.0
    target_hours = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, target_hours))
    if not has_refresh:
        grown = last * (prior_growth ** expected_refreshes) \
            if expected_refreshes > 0 else last
        return (clamp(grown, last, tracked_target), 0.0)
    newest = distinct[-1][0] if distinct else 0.0
    window = [(math.log1p(h), y) for h, y in distinct
              if newest - h <= FIT_WINDOW_HOURS] or \
        [(math.log1p(h), y) for h, y in distinct]
    intercept, slope, resid, n, x_mean, sxx = least_squares(window)
    fitted = math.exp(intercept + slope * x_target)
    shrink = n / (n + 3.0) if n else 0.0
    grown = math.exp(math.log(max(last, 0.5))
                     + shrink * (math.log(max(fitted, 0.5))
                                 - math.log(max(last, 0.5))))
    grown = clamp(grown, last, tracked_target)
    grown = min(grown, ceiling / 100.0 * tracked_target)
    if n >= 3 and sxx > 1e-9:
        pred = 1.0 + 1.0 / n + (x_target - x_mean) ** 2 / sxx
        fit_sigma = min(resid * math.sqrt(pred), 1.2)
    else:
        fit_sigma = 0.0
    return (grown, fit_sigma)


def mixture_quantiles(levels, weight_none, median_none, median_refresh,
                      sigma_refresh):
    """Quantiles of a two-part forecast on the logit scale.

    With weight weight_none the rate stays at median_none (no refresh
    before the target: the page is frozen, so the band is narrow); the
    rest of the weight sits on a normal around median_refresh with
    spread sigma_refresh. Each quantile is found by bisection on the
    mixture CDF.
    """
    mu_none = logit(median_none)
    mu_refresh = logit(median_refresh)
    out = []
    for level in levels:
        lo, hi = -12.0, 12.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            cdf = weight_none * norm_cdf((mid - mu_none) / 0.15) \
                + (1.0 - weight_none) * norm_cdf(
                    (mid - mu_refresh) / sigma_refresh)
            if cdf < level:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2.0)
    return out


def forecast_trophy(trophy, index, reads, release, target, tracked_target,
                    expected_refreshes, p_none, jump_sd, has_refresh,
                    levels, z_values):
    kind, ceiling, _ = classify(trophy)
    now_tracked = reads[-1][1]
    last_earned = None
    last_rate = None
    for _, tracked, earned in reads:
        if earned[index] is not None:
            last_earned = earned[index]
            last_rate = earned[index] / tracked * 100.0
    if last_earned is None:
        last_earned = 1.0
        last_rate = 1.0

    # The trophy every tracked player holds: earned == tracked in every
    # read. It has sat at 100% through dilution before; forecast it in a
    # tight band just under 100%, not from a count fit (there is no
    # trend to fit).
    universal = now_tracked >= 100 and all(
        earned[index] is not None and abs(earned[index] - tracked) < 0.5
        for _, tracked, earned in reads)
    if universal:
        return [98.0, 99.5, 99.9, 99.98, 100.0]

    earned_target, fit_sigma = project_earned(
        index, reads, release, target, classify(trophy),
        ceiling, tracked_target, has_refresh, expected_refreshes)
    median_rate = clamp(earned_target / tracked_target * 100.0, 0.01, 99.99)

    # Spread on the logit scale: the earned fit's error, plus the
    # refresh uncertainty (how many refreshes, how big each jump), with
    # the random-walk part capped at two refreshes: past that, the trend
    # fit's own error governs.
    sigma_trend = math.sqrt(fit_sigma ** 2 + TREND_SIGMA_FLOOR ** 2)
    span = min(expected_refreshes, 2.0)
    sigma_refresh = math.sqrt(span * (jump_sd ** 2 + EARNED_JUMP_VAR))
    sigma = clamp(math.sqrt(sigma_trend ** 2 + sigma_refresh ** 2),
                  SIGMA_FLOOR, SIGMA_CAP)

    if expected_refreshes < MIXTURE_MAX_REFRESHES:
        weight_none = max(p_none, 0.05)
        logits = mixture_quantiles(
            levels, weight_none, clamp(last_rate, 0.01, 99.99),
            median_rate, max(sigma, 0.35))
        quantiles = [round4(clamp(inv_logit(v), 0.01, 100.0))
                     for v in logits]
    else:
        centre = logit(median_rate)
        quantiles = [round4(clamp(inv_logit(centre + z * sigma),
                                  0.01, 100.0))
                     for z in z_values]
    # Rounding can only tie neighbours, never invert them.
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return quantiles


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    z_values = [norm_ppf(level) for level in levels]
    reads = read_history(case)
    if not reads:
        raise ValueError(f"{name} has no history")
    release_text = (case.get("game") or {}).get("release_date") or "2026-11-19"
    try:
        release = datetime.fromisoformat(release_text).replace(
            tzinfo=timezone.utc)
    except ValueError:
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    horizon_hours = case["horizon_minutes"] / 60.0
    has_refresh = any(cur[1] != prev[1]
                      for prev, cur in zip(reads, reads[1:]))
    expected, p_none, jump_log, jump_sd = refresh_stats(
        reads, checkpoint, horizon_hours)
    tracked_target = project_tracked(
        reads, release, target, expected, jump_log)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(
                         trophy, i, reads, release, target, tracked_target,
                         expected, p_none, jump_sd, has_refresh,
                         levels, z_values)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
