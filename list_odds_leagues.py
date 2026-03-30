"""
list_odds_leagues.py

Lists all soccer leagues available on The Odds API with your current key.
Run this to see exactly what odds coverage you have.

Usage:
    source venv/bin/activate
    python3 list_odds_leagues.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

if not ODDS_API_KEY:
    print("ODDS_API_KEY not set")
    exit(1)

resp = requests.get(
    "https://api.the-odds-api.com/v4/sports",
    params={"apiKey": ODDS_API_KEY},
    timeout=10,
)

if not resp.ok:
    print(f"Error: {resp.status_code} {resp.text[:200]}")
    exit(1)

leagues = [(s["key"], s["title"]) for s in resp.json() if "soccer" in s["key"]]
leagues.sort()

print(f"Total soccer markets: {len(leagues)}\n")
for key, title in leagues:
    print(f"  {key:50}  {title}")
