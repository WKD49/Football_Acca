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
import time
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
    ("ELC", "CHAMP"),
    ("PD",  "LALIGA"),
    ("BL1", "BUNDESLIGA"),
    ("SA",  "SERIEA"),
    ("FL1", "LIGUE1"),
    ("PPL", "PRIMEIRA"),
]

SEASONS = [2024, 2025]  # runs both and combines totals

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
    season: int = 2025,
) -> dict:
    calc = FormCalculator()

    raw = fd.get_finished_matches(comp_code, season)
    all_parsed = [calc.parse_result(m) for m in raw]
    all_parsed = [p for p in all_parsed if p is not None]
    all_parsed.sort(key=lambda r: r["date"])

    if len(all_parsed) < MIN_HISTORY:
        print(f"  {league_id}: only {len(all_parsed)} results — skipping")
        return {}, {}

    # Compute actual outcome frequencies from ALL results in this dataset
    n = len(all_parsed)
    actual_baselines = {
        "1X2_HOME": sum(1 for r in all_parsed if r["home_goals"] > r["away_goals"]) / n,
        "1X2_DRAW": sum(1 for r in all_parsed if r["home_goals"] == r["away_goals"]) / n,
        "1X2_AWAY": sum(1 for r in all_parsed if r["away_goals"] > r["home_goals"]) / n,
        "OVER_2.5": sum(1 for r in all_parsed if r["home_goals"] + r["away_goals"] >= 3) / n,
        "UNDER_2.5": sum(1 for r in all_parsed if r["home_goals"] + r["away_goals"] < 3) / n,
    }

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

    return dict(stats), actual_baselines


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

    season_labels = {2024: "2024/25", 2025: "2025/26"}
    baselines = {"1X2_HOME": "~45%", "1X2_DRAW": "~25%", "1X2_AWAY": "~30%",
                 "OVER_2.5": "~55%", "UNDER_2.5": "~45%"}

    print(f"\nBacktest — seasons: {', '.join(season_labels.values())}  (min model prob: {MIN_MODEL_PROB:.0%})")
    print(f"Min history before predicting: {MIN_HISTORY} matches\n")

    grand_total = defaultdict(lambda: {"flagged": 0, "correct": 0})

    for season in SEASONS:
        print(f"{'='*60}")
        print(f"SEASON {season_labels[season]}\n")

        for comp_code, league_id in LEAGUES:
            print(f"  {league_id}...")
            time.sleep(7)  # stay within 10 req/min free tier limit
            stats, actual_bl = backtest_league(fd, comp_code, league_id, model, pricer, season)

            if not stats:
                continue

            print(f"\n  {'Market':<12}  {'Flagged':>7}  {'Correct':>7}  {'Hit rate':>9}  {'Actual base':>12}  {'vs Base':>8}")
            print(f"  {'':─<12}  {'':─>7}  {'':─>7}  {'':─>9}  {'':─>12}  {'':─>8}")

            for market, s in sorted(stats.items()):
                f, c = s["flagged"], s["correct"]
                hit = c / f if f else 0
                bl_val = actual_bl.get(market, 0)
                bl = f"{bl_val:.1%}"
                diff = hit - bl_val
                diff_str = f"{diff:+.1%}"
                print(f"  {market:<12}  {f:>7}  {c:>7}  {hit:>8.1%}  {bl:>12}  {diff_str:>8}")
                grand_total[market]["flagged"] += f
                grand_total[market]["correct"] += c

            print()

    # Grand totals across both seasons
    print("=" * 60)
    print(f"TOTALS — BOTH SEASONS, ALL LEAGUES\n")
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
