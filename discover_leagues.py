"""
discover_leagues.py

Finds football leagues that have both:
  1. Results data on API-Football (api-sports.io)
  2. Odds available on The Odds API

Run this to identify leagues worth adding to the model.
Uses 1 API-Football request (league list) + 1 Odds API request.

Usage:
    source venv/bin/activate
    python3 discover_leagues.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAPID_API_KEY = os.environ.get("RAPID_API_KEY", "")
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")

# Leagues we already have in the model — skip these
ALREADY_HAVE = {
    "England Premier League", "EFL Championship", "EFL League One", "EFL League Two",
    "La Liga", "La Liga 2",
    "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga",
    "Pro League",
    "Allsvenskan", "Eliteserien", "Superliga",
    "J-League", "A-League",
}

# Leagues worth targeting — good mix of coverage + bookmaker interest
TARGET_COUNTRIES = {
    "Netherlands", "Turkey", "Portugal", "Austria", "Czech Republic",
    "Poland", "Romania", "Greece", "Scotland", "Belgium",
    "Brazil", "Argentina", "Mexico", "USA",
}


def get_af_leagues():
    """Fetch all current leagues from API-Football."""
    if not RAPID_API_KEY:
        print("RAPID_API_KEY not set — skipping API-Football")
        return []

    resp = requests.get(
        "https://v3.football.api-sports.io/leagues",
        headers={"x-apisports-key": RAPID_API_KEY},
        params={"current": "true", "type": "League"},
        timeout=15,
    )
    if not resp.ok:
        print(f"API-Football error: {resp.status_code} {resp.text[:200]}")
        return []

    results = []
    for item in resp.json().get("response", []):
        league  = item.get("league", {})
        country = item.get("country", {})
        seasons = item.get("seasons", [])
        current = next((s for s in seasons if s.get("current")), None)
        fixtures = current.get("coverage", {}).get("fixtures", {}) if current else {}
        results.append({
            "id":            league.get("id"),
            "name":          league.get("name"),
            "country":       country.get("name"),
            "has_standings": fixtures.get("statistics_fixtures", False),
            "fixtures_count": current.get("fixtures", {}).get("played", 0) if current else 0,
        })
    return results


def get_odds_leagues():
    """Fetch all soccer leagues available on The Odds API."""
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set — skipping Odds API")
        return set()

    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports",
        params={"apiKey": ODDS_API_KEY},
        timeout=10,
    )
    if not resp.ok:
        print(f"Odds API error: {resp.status_code} {resp.text[:200]}")
        return set()

    return {s["title"] for s in resp.json() if "soccer" in s["key"]}


def main():
    print("Fetching leagues from API-Football...")
    af_leagues = get_af_leagues()
    print(f"  Found {len(af_leagues)} active leagues\n")

    print("Fetching leagues from The Odds API...")
    odds_titles = get_odds_leagues()
    print(f"  Found {len(odds_titles)} soccer markets\n")

    print("=" * 60)
    print("LEAGUES IN TARGET COUNTRIES (not already in model)")
    print("=" * 60)

    found = []
    for lg in af_leagues:
        if lg["name"] in ALREADY_HAVE:
            continue
        if lg["country"] not in TARGET_COUNTRIES:
            continue
        found.append(lg)

    found.sort(key=lambda x: (x["country"], x["name"]))

    for lg in found:
        odds_flag = "✓ odds" if lg["name"] in odds_titles else "  no odds"
        print(f"  [{lg['id']:5}] {lg['country']:20} {lg['name']:35} {odds_flag}")

    print()
    print("=" * 60)
    print("ALL OTHER LEAGUES WITH ODDS COVERAGE (sample)")
    print("=" * 60)

    others = [
        lg for lg in af_leagues
        if lg["name"] not in ALREADY_HAVE
        and lg["country"] not in TARGET_COUNTRIES
        and lg["name"] in odds_titles
    ]
    others.sort(key=lambda x: (x["country"], x["name"]))
    for lg in others[:30]:
        print(f"  [{lg['id']:5}] {lg['country']:20} {lg['name']}")

    if len(others) > 30:
        print(f"  ... and {len(others) - 30} more")


if __name__ == "__main__":
    main()
