"""
check_apis.py

Queries both APIs to show exactly what leagues/competitions are available
with your free keys. Run this once to confirm what works.
"""

import os
from dotenv import load_dotenv; load_dotenv()
import requests

fd_key   = os.environ.get("FOOTBALL_DATA_KEY", "")
odds_key = os.environ.get("ODDS_API_KEY", "")

# ---------------------------------------------------------------------------
# football-data.org: list available competitions
# ---------------------------------------------------------------------------
if fd_key:
    print("=== football-data.org: available competitions ===")
    resp = requests.get(
        "https://api.football-data.org/v4/competitions",
        headers={"X-Auth-Token": fd_key},
        timeout=10,
    )
    if resp.ok:
        for c in resp.json().get("competitions", []):
            print(f"  {c['code']:8}  {c['name']}")
    else:
        print(f"  Error: {resp.status_code} {resp.text[:200]}")
else:
    print("FOOTBALL_DATA_KEY not set — skipping")

print()

# ---------------------------------------------------------------------------
# The Odds API: list available soccer leagues
# ---------------------------------------------------------------------------
if odds_key:
    print("=== The Odds API: available soccer leagues ===")
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports",
        params={"apiKey": odds_key},
        timeout=10,
    )
    if resp.ok:
        for s in resp.json():
            if "soccer" in s["key"]:
                print(f"  {s['key']:45}  {s['title']}")
    else:
        print(f"  Error: {resp.status_code} {resp.text[:200]}")
else:
    print("ODDS_API_KEY not set — skipping")

print()

# ---------------------------------------------------------------------------
# Show which bookmakers are available for EPL (use this to pick BOOKMAKER in run.py)
# ---------------------------------------------------------------------------
if odds_key:
    print("=== Bookmakers available for EPL ===")
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
        params={"apiKey": odds_key, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=10,
    )
    if resp.ok and resp.json():
        event = resp.json()[0]
        print(f"  (from fixture: {event.get('home_team')} vs {event.get('away_team')})")
        for b in event.get("bookmakers", []):
            print(f"  {b['key']}")
        print(f"\n  To use one of these, set BOOKMAKER = \"bet365\" (or whichever) in run.py")
