"""
check_name_matching.py

Diagnoses team name mismatches between football-data.co.uk CSVs
and The Odds API for the new leagues.

Usage:
    python3 check_name_matching.py
"""

import os
from dotenv import load_dotenv; load_dotenv()
from data_fetcher import FootballDataCoUkClient, OddsApiClient, normalise_team_name

LEAGUES = [
    ("BUNDESLIGA2", "soccer_germany_bundesliga2"),
    ("SERIEB",      "soccer_italy_serie_b"),
    ("LIGUE2",      "soccer_france_ligue_two"),
]

odds_key = os.environ.get("ODDS_API_KEY", "")
if not odds_key:
    print("ODDS_API_KEY not set")
    exit(1)

odds = OddsApiClient(odds_key)
csv_client = FootballDataCoUkClient()

for league_id, sport_key in LEAGUES:
    print(f"\n{'='*60}")
    print(f"  {league_id}")
    print(f"{'='*60}")

    # Names from football-data.co.uk CSV
    csv_matches = csv_client.get_matches(league_id)
    csv_names = set()
    for m in csv_matches:
        csv_names.add(m["home"])
        csv_names.add(m["away"])

    # Names from Odds API
    events = odds.get_odds(sport_key)
    odds_names = set()
    for ev in events:
        odds_names.add(normalise_team_name(ev.get("home_team", "")))
        odds_names.add(normalise_team_name(ev.get("away_team", "")))

    matched   = csv_names & odds_names
    csv_only  = csv_names - odds_names
    odds_only = odds_names - csv_names

    print(f"\n  CSV has {len(csv_matches)} matches, {len(csv_names)} unique teams")
    print(f"  Odds API has {len(events)} upcoming fixtures, {len(odds_names)} unique teams")
    print(f"\n  MATCHED ({len(matched)}) — form data will work for these:")
    for n in sorted(matched):
        print(f"    ✓  {n}")
    print(f"\n  CSV ONLY ({len(csv_only)}) — in form data but not in upcoming fixtures (fine):")
    for n in sorted(csv_only):
        print(f"    ~  {n}")
    print(f"\n  ODDS ONLY ({len(odds_only)}) — upcoming fixtures with NO form data match:")
    for n in sorted(odds_only):
        print(f"    ✗  {n}")
