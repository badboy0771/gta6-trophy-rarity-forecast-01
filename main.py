"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for each trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, scored by mean pinball loss on the logit scale.

Model, per trophy:
  1. Fit a line to logit(rate) against ln(1 + hours since release) over the
     case history's recent reads (least squares). The fit follows whatever the
     rates were actually doing: rising as players progress through the game,
     falling when a flood of newly tracked players dilutes them.
  2. Extrapolate that line to the target, with the move from the last read
     capped, and capped again at a ceiling that depends on the trophy's kind
     (read from its own name and description): a chapter trophy can plausibly
     reach tens of percent a week in, a grind or collectible trophy cannot,
     and the platinum / 100%-completion trophies stay well under 1%.
  3. Spread the five quantiles honestly. The spread combines a random-walk
     term growing with the horizon, the fit's own prediction error, and a
     fixed jump term: Exophase refreshes its counts on its own schedule, and
     one refresh can move every rate at once (the practice cases straddle
     exactly such a refresh on launch night). When the whole history is flat
     and the horizon is short, the median is nudged down, because that first
     refresh lowered every rate but the one at 100%.

Python 3.13, standard library only, no network, no randomness: two runs of
the same case write the same bytes. Reads every *.in file in the input
directory and writes one *.out file per case.
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
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Spread of a rate's move, in logit units per square-root day (the house
# baseline uses 0.5; this entry carries a little more, the launch week's
# dilution makes moves bigger than a plain random walk).
RW_SIGMA_PER_SQRT_DAY = 0.55
# Logit spread for a single Exophase refresh / dilution jump, whatever the
# horizon: one refresh can move every rate at once.
JUMP_SIGMA = 1.0
# Median nudge (logit units down) when no read has moved yet and the horizon
# is short: the first refresh after launch night lowered every rate but one.
FLAT_HISTORY_SHIFT = 0.6
FLAT_HISTORY_MAX_HORIZON_DAYS = 0.25
# Only reads this recent (before the checkpoint) feed the fit.
FIT_WINDOW_HOURS = 48.0

# (kind, ceiling in percent) per trophy, from its name and description.
# Ceilings are caps on the fit, never targets: a fit below its ceiling is
# untouched, so a trophy the data says is slow stays slow.
CHAPTER_CEILINGS = {"chapter 1": 45.0, "chapter 2": 35.0,
                    "chapter 3": 28.0, "chapter 4": 22.0}


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


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
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, error < 2e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                                + b[4]) * r + 1)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def classify(trophy):
    """Return (kind, ceiling percent) for one trophy, from its own text."""
    desc = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return ("platinum", 0.6)
    if "100% completion" in desc:
        return ("completion", 0.5)
    for chapter, ceiling in CHAPTER_CEILINGS.items():
        if "complete " + chapter in desc:
            return (chapter.replace(" ", ""), ceiling)
    if "complete the final mission" in desc:
        return ("final-mission", 10.0)
    if ("collectible card" in desc or "waink" in desc
            or "discover 100 vehicles" in desc
            or "explore all of leonida" in desc):
        return ("collectible", 10.0)
    if ("250 stunts" in desc or "50 challenges" in desc
            or "20 different races" in desc or "spend $" in desc
            or "earn $250,000" in desc or "purchase all garage" in desc
            or "register 10 vehicles" in desc or "gold medal" in desc):
        return ("grind", 8.0)
    if desc.startswith("complete"):
        return ("mission", 15.0)
    if ("win " in desc or "beat " in desc or "try each type" in desc
            or "meditate" in desc or "hunt " in desc):
        return ("activity", 18.0)
    return ("misc", 12.0)


def fit_line(points):
    """Least squares y = a + b*x over [(x, y), ...].

    Returns (intercept, slope, residual std, n, x mean, Sxx). A flat or
    tiny series gets slope 0 and residual 0: no trend to trust.
    """
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


def forecast_trophy(trophy, index, history, release, target, horizon_days,
                    zvals):
    kind, ceiling = classify(trophy)
    series = []
    last_rate = None
    flat = True
    seen = None
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        if seen is None:
            seen = rate
        elif rate != seen:
            flat = False
        last_rate = rate
        try:
            hours = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        series.append((max(0.0, hours), math.log1p(max(0.0, hours)),
                       logit(rate)))
    if last_rate is None:
        last_rate = 1.0
    y_last = logit(last_rate)

    # The trophy everyone tracked already holds: it has sat at 100% through
    # the launch-night refresh and still does. Forecast it near the top with
    # a narrow band, not a trend fit (there is no trend to fit).
    if last_rate >= 99.9:
        return [round4(v) for v in (90.0, 97.0, 99.5, 99.95, 100.0)]

    # Fit on the recent window only: the launch week's regime changes fast.
    # The log-time axis already downweights the distant past, so fit the
    # whole series when it is short and the last 48 hours of it when long.
    if series:
        newest_hours = series[-1][0]
        window = [(p[1], p[2]) for p in series
                  if newest_hours - p[0] <= FIT_WINDOW_HOURS] or \
            [(p[1], p[2]) for p in series]
    else:
        window = []
    intercept, slope, resid, n, x_mean, sxx = fit_line(window)
    # Shrink a slope the data barely supports toward flat persistence.
    slope *= n / (n + 3.0) if n else 0.0

    target_hours = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, target_hours))
    y_med = intercept + slope * x_target
    # A rate cannot run away in either direction over one horizon: dilution
    # slows as tracked-player growth saturates, and earners only accumulate.
    up = 3.0 if kind in ("chapter1", "chapter2") else 2.0
    y_med = clamp(y_med, y_last - 2.5, y_last + up)
    y_med = min(y_med, logit(ceiling))
    # No read has moved yet and the horizon is short: the first refresh is
    # coming, and it pushed every rate down.
    if flat and horizon_days <= FLAT_HISTORY_MAX_HORIZON_DAYS:
        y_med = max(y_med - FLAT_HISTORY_SHIFT, y_last - 2.5)

    sigma_rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(max(0.0, horizon_days))
    if n >= 3 and sxx > 1e-9:
        pred = 1.0 + 1.0 / n + (x_target - x_mean) ** 2 / sxx
        sigma_fit = min(resid * math.sqrt(pred), 1.5)
    else:
        sigma_fit = 0.0
    sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + JUMP_SIGMA ** 2)
    floor = 1.15 if kind in ("platinum", "completion") else 0.95
    sigma = max(sigma, floor)

    quantiles = [round4(clamp(inv_logit(y_med + z * sigma), 0.01, 100.0))
                 for z in zvals]
    # Rounding can only tie neighbours, never invert them; enforce the order.
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return quantiles


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    zvals = [Z[LEVELS.index(lv)] if lv in LEVELS else norm_ppf(lv)
             for lv in levels]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    release_text = (case.get("game") or {}).get("release_date") or "2026-11-19"
    try:
        release = datetime.fromisoformat(release_text).replace(
            tzinfo=timezone.utc)
    except ValueError:
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, i, history, release,
                                          target, horizon_days, zvals)})
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
