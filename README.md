# GTA VI Trophy Rarity Forecaster

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — a week after launch — as five
quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Entry for the make.fun competition
[gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01).

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/badboy0771/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/badboy0771/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case file
per run from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes one forecast file
per case to `/output` (or `$MAKEFUN_OUTPUT_DIR`).

For each trophy it:

1. **Fits the trend it can see.** Least squares of `logit(rate)` against
   `ln(1 + hours since release)` over the case history's recent reads, so the
   forecast follows the direction the rates were actually moving — up as
   players progress through the game, down when a flood of new tracked
   players dilutes the rates.
2. **Caps the extrapolation by trophy kind.** A chapter trophy can plausibly
   reach tens of percent a week in; a grind or collectible trophy cannot, and
   the platinum and 100%-completion trophies stay well under 1%. The kind is
   read from each trophy's own name and description, and the cap only trims
   runaway fits — it never invents growth the data does not show.
3. **Spreads honestly.** The pinball loss on the logit scale punishes both
   over- and under-confidence, so the spread combines a random-walk term that
   grows with the horizon, the fit's own prediction error, and a fixed jump
   term: Exophase refreshes its counts on its own schedule, and one refresh
   can move every rate at once (the practice cases straddle exactly such a
   refresh). When the whole history is flat and the horizon is short, the
   median is nudged down, because the first refresh after launch night
   lowered every rate but one.

Run it locally on the practice cases:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
