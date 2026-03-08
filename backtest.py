"""
backtest.py

Model accuracy backtest using this season's finished results.

For each finished match (from gameweek ~6 onwards, once there's enough form data):
  - Computes team form using ONLY results BEFORE that match date
  - Runs the Poisson model to get predicted probabilities
  - Records whether the model's top prediction was correct

No historical odds available on the free tier, so this tests ACCURACY not P&L.
But it tells you: is the model picking outcomes better than random chance?

Usage:
  python3 backtest.py
"""

import os
from datetime import datetime, timezone
from collections import defaultdict

from data_fetcher import FootballDataClient, FormCalculator, normalise_team_name
from football_value_acca import LEAGUE_CONFIGS, MarketPricer, MatchModel, Match, ScheduleContext, TeamFeatures

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Leagues to backtest (football-data.org code, our league ID)
LEAGUES = [
    ("PL",  "EPL"),
    ("PD",  "LALIGA"),
    ("BL1", "BUNDESLIGA"),
    ("SA",  "SERIEA"),
    ("FL1", "LIGUE1"),
    ("PPL", "PRIMEIRA"),
]

SEASON = 2025

# Only flag a prediction if model probability exceeds this threshold
# (simulates the confidence filter the live app uses)
MIN_MODEL_PROB = 0.45

# Minimum matches played before we start making predictions
# (need enough history for form to be meaningful)
MIN_HISTORY = 15

# ---------------------------------------------------------------------------
# Neutral fallback features (used when a team has too few matches)
# ---------------------------------------------------------------------------
_NEUTRAL = TeamFeatures(
    home_attack_long=1.0,  home_attack_short=1.0,
    home_defence_long=1.0, home_defence_short=1.0,
    away_attack_long=1.0,  away_attack_short=1.0,
    away_defence_long=1.0, away_defence_short=1.0,
    strict_validation=False,
)

_NEUTRAL_SCHED = ScheduleContext(
    rest_days=7, played_midweek=False,
    played_europe_midweek=False, between_two_europe_legs=False,
)


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "HOME"
    if away_goals > home_goals:
        return "AWAY"
    return "DRAW"


def backtest_league(
    fd: FootballDataClient,
    comp_code: str,
    league_id: str,
    model: MatchModel,
    pricer: MarketPricer,
) -> dict:
    calc = FormCalculator()

    raw = fd.get_finished_matches(comp_code, SEASON)
    all_parsed = [calc.parse_result(m) for m in raw]
    all_parsed = [p for p in all_parsed if p is not None]
    all_parsed.sort(key=lambda r: r["date"])

    if len(all_parsed) < MIN_HISTORY:
        print(f"  {league_id}: only {len(all_parsed)} results — skipping")
        return {}

    stats = defaultdict(lambda: {"flagged": 0, "correct": 0})

    for i, result in enumerate(all_parsed):
        # Only use results BEFORE this match to compute form
        prior = [r for r in all_parsed if r["date"] < result["date"]]
        if len(prior) < MIN_HISTORY:
            continue

        league_home_avg, league_away_avg = calc.league_averages(prior)

        home = result["home"]
        away = result["away"]

        home_matches = [r for r in prior if r["home"] == home or r["away"] == home]
        away_matches = [r for r in prior if r["home"] == away or r["away"] == away]

        # Skip if either team has very little history
        if len(home_matches) < 4 or len(away_matches) < 4:
            continue

        home_feat = calc.compute(home, prior, league_home_avg, league_away_avg)
        away_feat = calc.compute(away, prior, league_home_avg, league_away_avg)

        data_quality = 0.85 if league_id in ("EPL", "LALIGA") else 0.75

        match = Match(
            match_id=f"{league_id}_{i}",
            league=league_id,
            kickoff_utc=result["date"],
            home_team=home,
            away_team=away,
            home_features=home_feat,
            away_features=away_feat,
            home_schedule=_NEUTRAL_SCHED,
            away_schedule=_NEUTRAL_SCHED,
            league_data_quality=data_quality,
        )

        lam = model.lambdas(match)
        p_home, p_draw, p_away = pricer.p_1x2(lam.lam_home, lam.lam_away)
        p_over, p_under = pricer.p_over_under(lam.lam_home, lam.lam_away, 2.5)

        actual = actual_outcome(result["home_goals"], result["away_goals"])
        actual_over = (result["home_goals"] + result["away_goals"]) >= 3

        # 1X2 — flag top pick if above threshold
        best_1x2 = max([("HOME", p_home), ("DRAW", p_draw), ("AWAY", p_away)], key=lambda x: x[1])
        if best_1x2[1] >= MIN_MODEL_PROB:
            sel = best_1x2[0]
            stats[f"1X2_{sel}"]["flagged"] += 1
            if actual == sel:
                stats[f"1X2_{sel}"]["correct"] += 1

        # Over 2.5
        if p_over >= MIN_MODEL_PROB:
            stats["OVER_2.5"]["flagged"] += 1
            if actual_over:
                stats["OVER_2.5"]["correct"] += 1

        # Under 2.5
        if p_under >= MIN_MODEL_PROB:
            stats["UNDER_2.5"]["flagged"] += 1
            if not actual_over:
                stats["UNDER_2.5"]["correct"] += 1

    return dict(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fd_key = os.environ.get("FOOTBALL_DATA_KEY", "")
    if not fd_key:
        print("FOOTBALL_DATA_KEY not set.")
        raise SystemExit(1)

    fd     = FootballDataClient(fd_key)
    model  = MatchModel(LEAGUE_CONFIGS)
    pricer = MarketPricer(max_goals=10)

    print(f"\nBacktest — Season 2025/26  (min model prob: {MIN_MODEL_PROB:.0%})")
    print(f"Min history before predicting: {MIN_HISTORY} matches\n")

    grand_total = defaultdict(lambda: {"flagged": 0, "correct": 0})

    for comp_code, league_id in LEAGUES:
        print(f"Running {league_id}...")
        stats = backtest_league(fd, comp_code, league_id, model, pricer)

        if not stats:
            continue

        print(f"\n  {'Market':<12}  {'Flagged':>7}  {'Correct':>7}  {'Hit rate':>9}  {'Baseline':>9}")
        print(f"  {'':─<12}  {'':─>7}  {'':─>7}  {'':─>9}  {'':─>9}")

        # Rough baselines (league averages across Europe)
        baselines = {"1X2_HOME": "~45%", "1X2_DRAW": "~25%", "1X2_AWAY": "~30%",
                     "OVER_2.5": "~55%", "UNDER_2.5": "~45%"}

        for market, s in sorted(stats.items()):
            f, c = s["flagged"], s["correct"]
            hit = c / f if f else 0
            bl = baselines.get(market, "—")
            print(f"  {market:<12}  {f:>7}  {c:>7}  {hit:>8.1%}  {bl:>9}")
            grand_total[market]["flagged"] += f
            grand_total[market]["correct"] += c

        print()

    # Grand totals
    print("=" * 60)
    print("TOTALS ACROSS ALL LEAGUES\n")
    print(f"  {'Market':<12}  {'Flagged':>7}  {'Correct':>7}  {'Hit rate':>9}")
    print(f"  {'':─<12}  {'':─>7}  {'':─>7}  {'':─>9}")
    for market, s in sorted(grand_total.items()):
        f, c = s["flagged"], s["correct"]
        hit = c / f if f else 0
        print(f"  {market:<12}  {f:>7}  {c:>7}  {hit:>8.1%}")

    print()
    print("Baseline rates (what a coin-flip-style guess would get):")
    print("  Home win ~45%  |  Draw ~25%  |  Away win ~30%")
    print("  Over 2.5 ~55%  |  Under 2.5 ~45%")
    print()
    print("If model hit rates are consistently above baseline → model is adding value.")
    print("If hit rates match or lag baseline → model needs more work.\n")


if __name__ == "__main__":
    main()
