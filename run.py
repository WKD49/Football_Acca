"""
run.py

End-to-end pipeline:
  1. Fetch real match data + odds from both APIs
  2. Run each fixture through the Poisson model to find value bets
  3. Build accumulators from the best value bets
  4. Print recommendations (straight acca + Yankee + Super Yankee)

Usage:
  python3 run.py
"""

import os
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, timezone
from itertools import combinations

from data_fetcher import FootballDataClient, OddsApiClient, ApiFootballClient, fetch_competition, fetch_competition_af
from football_value_acca import (
    AccaConstraints,
    BetCandidate,
    CandidateRules,
    LeagueConfig,
    LEAGUE_CONFIGS,
    Market,
    MarketPricer,
    Match,
    MatchModel,
    OddsSnapshot,
    OptimiserConfig,
    build_accas_beam_search,
    build_candidate,
)

# ---------------------------------------------------------------------------
# Configuration — tweak these to change what the model looks for
# ---------------------------------------------------------------------------

SEASON = 2025        # 2025/26 season
DAYS_AHEAD = 7       # how many days of upcoming fixtures to fetch

# Set this to your bookmaker's key so all legs come from the same place.
# Common keys: "bet365", "betway", "williamhill", "unibet", "paddypower"
# Set to None to use best available price (not suitable for accas).
# Run with BOOKMAKER = None first to see what's available, then pick one.
BOOKMAKER = "paddypower"  # also try "ladbrokes_uk"

# Value detection thresholds
# EV = (model_prob × decimal_odds) - 1. Primary filter: require 4% EV.
# min_edge is a backstop — prevents recommending bets with tiny probability edges.
RULES = CandidateRules(
    min_ev=0.04,
    min_edge=0.01,
    min_confidence=0.55,
)

# Accumulator constraints (3–5 legs, max odds tightened to avoid lottery tickets)
CONSTRAINTS = AccaConstraints(
    min_legs=3,
    max_legs=5,
    min_total_decimal_odds=8.0,    # 7/1 minimum
    soft_max_total_decimal_odds=101.0,  # 100/1 soft cap
)

# Beam search settings (higher beam_width = more thorough but slower)
OPT = OptimiserConfig(beam_width=500, expand_top_m=250)

# Competitions to fetch: (football-data.org code, our league ID, odds API sport key)
COMPETITIONS = [
    ("PL",  "EPL",        "soccer_epl"),
    ("ELC", "CHAMP",      "soccer_efl_champ"),
    ("PD",  "LALIGA",     "soccer_spain_la_liga"),
    ("PPL", "PRIMEIRA",   "soccer_portugal_primeira_liga"),
    ("BL1", "BUNDESLIGA", "soccer_germany_bundesliga"),
    ("SA",  "SERIEA",     "soccer_italy_serie_a"),
    ("FL1", "LIGUE1",     "soccer_france_ligue_one"),
]

# Leagues sourced from API-Football (not on football-data.org free tier)
# Format: (league_id, af_league_id, odds_sport_key)
AF_COMPETITIONS = [
    ("BUNDESLIGA2",  79,  "soccer_germany_bundesliga2"),
    ("SERIEB",      136,  "soccer_italy_serie_b"),
    ("LIGUE2",       62,  "soccer_france_ligue_two"),
    ("LIGAMX",      262,  "soccer_mexico_ligamx"),
    ("BRASILEIRAO",  71,  "soccer_brazil_campeonato"),
    ("BRASILEIRAO_B", 72, "soccer_brazil_serie_b"),
]

# Max individual odds for Yankee/Super Yankee selections (long shots make bad Yankees)
YANKEE_MAX_ODDS = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_selection(market: Market) -> str:
    """Turn a market into plain English."""
    if market.kind == "1X2":
        mapping = {"HOME": "Home win", "DRAW": "Draw", "AWAY": "Away win"}
        return mapping.get(market.selection, market.selection)
    if market.kind == "OVER_UNDER":
        direction = "Over" if market.selection == "OVER" else "Under"
        return f"{direction} {market.line} goals"
    return market.selection


def confidence_label(conf: float) -> str:
    if conf >= 0.80: return "Very high"
    if conf >= 0.70: return "High"
    if conf >= 0.60: return "Good"
    return "Moderate"


def win_prob_label(prob: float) -> str:
    """Express a small win probability as a 1-in-X phrase."""
    if prob <= 0:
        return "very unlikely"
    n = round(1 / prob)
    return f"roughly 1 in {n}"


def print_candidates(candidates: list) -> None:
    if not candidates:
        print("  (none found this week)")
        return
    for c in candidates:
        bookie_edge = 1 / c.odds.decimal_odds
        print(
            f"  {c.home_team} vs {c.away_team}  [{c.league}]\n"
            f"    Bet: {format_selection(c.market)}  |  "
            f"Odds: {c.odds.decimal_odds:.2f}  |  "
            f"Our estimate: {c.model_prob:.0%}  |  "
            f"Bookmaker implies: {bookie_edge:.0%}  |  "
            f"Edge: {c.edge:+.0%}  |  "
            f"Confidence: {confidence_label(c.confidence)}"
        )


def print_acca(i: int, acca) -> None:
    chance = win_prob_label(acca.win_prob)
    print(f"\n{'='*60}")
    print(f"  ACCA #{i+1}")
    print(f"  Combined odds: {acca.total_odds:.1f}/1  |  "
          f"Chance of winning: {chance}  |  "
          f"Confidence: {confidence_label(acca.avg_conf)}")
    print(f"{'='*60}")
    for leg in acca.legs:
        print(
            f"  ✦  {leg.home_team} vs {leg.away_team}  [{leg.league}]\n"
            f"       {format_selection(leg.market)}  "
            f"@ {leg.odds.decimal_odds:.2f}  "
            f"(our estimate: {leg.model_prob:.0%}, edge: {leg.edge:+.0%})"
        )
    if acca.notes:
        # Filter out internal soft-cap warnings — not useful to the punter
        external_notes = [n for n in acca.notes if "soft" not in n.lower()]
        if external_notes:
            print(f"\n  Note: " + "  |  ".join(external_notes))


def yankee_score(c) -> float:
    """Rank candidates for Yankee/Super Yankee: reward edge and confidence equally."""
    return c.edge * c.confidence


def print_coverage_bet(label: str, num_bets: int, selections: list) -> None:
    """Print a Yankee or Super Yankee recommendation."""
    print(f"\n{'='*60}")
    print(f"  {label}  —  {num_bets} bets  (£1/bet = £{num_bets} total stake)")
    print(f"{'='*60}")
    for i, c in enumerate(selections, 1):
        print(
            f"  {i}.  {c.home_team} vs {c.away_team}  [{c.league}]\n"
            f"       {format_selection(c.market)}  "
            f"@ {c.odds.decimal_odds:.2f}  "
            f"(our estimate: {c.model_prob:.0%}, edge: {c.edge:+.0%})"
        )
    odds_list = [c.odds.decimal_odds for c in selections]
    doubles = sorted([o1 * o2 for o1, o2 in combinations(odds_list, 2)], reverse=True)
    print(f"\n  If 2 win  — best double pays: {doubles[0] - 1:.0f}/1")
    if len(odds_list) >= 3:
        trebles = sorted(
            [o1 * o2 * o3 for o1, o2, o3 in combinations(odds_list, 3)], reverse=True
        )
        print(f"  If 3 win  — best treble pays: {trebles[0] - 1:.0f}/1")
    full = 1.0
    for o in odds_list:
        full *= o
    n = len(selections)
    print(f"  If all {n} win — {n}-fold pays: {full - 1:.0f}/1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fd_key      = os.environ.get("FOOTBALL_DATA_KEY", "")
    odds_key    = os.environ.get("ODDS_API_KEY", "")
    rapid_key   = os.environ.get("RAPID_API_KEY", "")

    if not fd_key or not odds_key:
        print("API keys not set. Run:")
        print("  export FOOTBALL_DATA_KEY=your_key")
        print("  export ODDS_API_KEY=your_key")
        raise SystemExit(1)

    fd   = FootballDataClient(fd_key)
    odds = OddsApiClient(odds_key)
    af   = ApiFootballClient(rapid_key) if rapid_key else None
    now  = datetime.now(timezone.utc)

    model  = MatchModel(LEAGUE_CONFIGS)
    pricer = MarketPricer(max_goals=10)

    # -----------------------------------------------------------------------
    # Step 1: fetch all fixtures + odds
    # -----------------------------------------------------------------------
    bookie_label = BOOKMAKER if BOOKMAKER else "best available (multi-bookie)"
    print(f"\nFetching fixtures for next {DAYS_AHEAD} days... [{bookie_label}]\n")

    all_fixtures: list = []
    for comp_code, league_id, sport_key in COMPETITIONS:
        try:
            results = fetch_competition(
                comp_code, league_id, fd, odds, sport_key,
                season=SEASON, days_ahead=DAYS_AHEAD,
                bookmaker_filter=BOOKMAKER,
            )
            all_fixtures.extend(results)
            fixtures_with_odds = sum(1 for _, mo in results if mo)
            print(f"  {league_id:12} {len(results)} fixtures, {fixtures_with_odds} with {bookie_label} odds")
        except Exception as e:
            print(f"  {league_id:12} ERROR: {e}")

    if af:
        for league_id, af_league_id, sport_key in AF_COMPETITIONS:
            try:
                results = fetch_competition_af(
                    league_id, af_league_id, sport_key, af, odds,
                    season=SEASON, days_ahead=DAYS_AHEAD,
                    bookmaker_filter=BOOKMAKER,
                )
                all_fixtures.extend(results)
                fixtures_with_odds = sum(1 for _, mo in results if mo)
                print(f"  {league_id:12} {len(results)} fixtures, {fixtures_with_odds} with {bookie_label} odds")
            except Exception as e:
                print(f"  {league_id:12} ERROR: {e}")
    else:
        print("  (RAPID_API_KEY not set — skipping AF leagues)")

    print(f"\n  Total: {len(all_fixtures)} fixtures\n")

    # -----------------------------------------------------------------------
    # Step 2: run model, collect value bets
    # -----------------------------------------------------------------------
    candidates: list = []
    for match, market_odds in all_fixtures:
        for market, snap in market_odds:
            c = build_candidate(match, market, snap, model, pricer, now, RULES)
            if c:
                candidates.append(c)

    print(f"Value bets identified this week: {len(candidates)}\n")
    print_candidates(candidates)

    # -----------------------------------------------------------------------
    # Step 3: Yankee + Super Yankee (shorter-odds selections only)
    # -----------------------------------------------------------------------
    yankee_pool = sorted(
        [c for c in candidates if c.odds.decimal_odds <= YANKEE_MAX_ODDS],
        key=yankee_score,
        reverse=True,
    )

    print(f"\n{'='*60}")
    print("  YANKEE / SUPER YANKEE SUGGESTIONS")
    print(f"  You win something as long as 2+ selections come in")
    print(f"  (only picks with odds under {YANKEE_MAX_ODDS:.1f} are used here)")
    print(f"{'='*60}")

    if len(yankee_pool) >= 4:
        print_coverage_bet("YANKEE", 11, yankee_pool[:4])
    else:
        print(f"\n  Not enough short-odds value bets for a Yankee (need 4, have {len(yankee_pool)}).")

    if len(yankee_pool) >= 5:
        print_coverage_bet("SUPER YANKEE (Canadian)", 26, yankee_pool[:5])
    else:
        print(f"\n  Not enough short-odds value bets for a Super Yankee (need 5, have {len(yankee_pool)}).")

    print(f"\n{'='*60}")
    print("  REMINDER: This is a mathematical model. It finds statistical")
    print("  edges but cannot guarantee results. Bet responsibly.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
