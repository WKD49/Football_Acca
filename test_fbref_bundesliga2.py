"""
test_fbref_bundesliga2.py

Tests whether we can scrape xG data from FBref for Bundesliga 2.

FBref blocks automated requests, but responds to polite ones:
  - A real browser User-Agent
  - A session cookie from visiting the page first
  - A short delay between requests

If this works, we can hand it to OpenClaw to run nightly.

Usage:
    python3 test_fbref_bundesliga2.py
"""

import time
import json
import re
import requests

# FBref Bundesliga 2 2025/26 fixtures/scores page
# comp/33 = Bundesliga 2
PAGE_URL  = "https://fbref.com/en/comps/33/2025-2026/schedule/2025-2026-2-Bundesliga-Scores-and-Fixtures"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://fbref.com/en/comps/33/2-Bundesliga-Stats",
}


def scrape():
    session = requests.Session()
    session.headers.update(HEADERS)

    print("Step 1: visiting FBref Bundesliga 2 stats page to get a session cookie...")
    r1 = session.get("https://fbref.com/en/comps/33/2-Bundesliga-Stats", timeout=15)
    print(f"  Status: {r1.status_code}")
    if r1.status_code != 200:
        print("  Failed to get session cookie. FBref may be blocking us.")
        return

    print("  Waiting 4 seconds before next request...")
    time.sleep(4)

    print(f"Step 2: fetching scores/fixtures page...")
    r2 = session.get(PAGE_URL, timeout=15)
    print(f"  Status: {r2.status_code}")
    if r2.status_code != 200:
        print(f"  Failed. Status {r2.status_code} — still blocking.")
        return

    html = r2.text

    # Look for the scores table — it contains xG columns (xG, xGA)
    # FBref buries data in HTML tables; we look for rows with xG values
    # Pattern: find all table rows that have a score (digit-digit) and xG floats
    matches_found = []

    # Find the sched table
    table_match = re.search(r'<table[^>]+id="sched_2025-2026_33_1"[^>]*>(.*?)</table>',
                            html, re.DOTALL)
    if not table_match:
        # Try alternate table id pattern
        table_match = re.search(r'<table[^>]+id="sched[^"]*"[^>]*>(.*?)</table>',
                                html, re.DOTALL)

    if not table_match:
        print("\n  Could not find the schedule table in the HTML.")
        print("  Saving first 3000 chars of HTML to fbref_debug.html for inspection...")
        with open("fbref_debug.html", "w") as f:
            f.write(html[:5000])
        print("  Check fbref_debug.html to see what FBref returned.")
        return

    print(f"\n  Found schedule table ({len(table_match.group(0))} chars)")

    # Extract rows — look for home xG and away xG columns
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_match.group(1), re.DOTALL)
    for row in rows:
        # Extract team names
        home = re.search(r'data-stat="home_team"[^>]*>.*?<a[^>]*>([^<]+)</a>', row, re.DOTALL)
        away = re.search(r'data-stat="away_team"[^>]*>.*?<a[^>]*>([^<]+)</a>', row, re.DOTALL)
        score = re.search(r'data-stat="score"[^>]*>([^<]+)</a>', row, re.DOTALL)
        home_xg = re.search(r'data-stat="home_xg"[^>]*>([0-9.]+)<', row)
        away_xg = re.search(r'data-stat="away_xg"[^>]*>([0-9.]+)<', row)

        if home and away and score and home_xg and away_xg:
            score_text = score.group(1).strip()
            parts = score_text.split("–") if "–" in score_text else score_text.split("-")
            if len(parts) == 2:
                try:
                    matches_found.append({
                        "home":     home.group(1).strip(),
                        "away":     away.group(1).strip(),
                        "home_goals": int(parts[0].strip()),
                        "away_goals": int(parts[1].strip()),
                        "home_xg":  float(home_xg.group(1)),
                        "away_xg":  float(away_xg.group(1)),
                    })
                except ValueError:
                    continue

    if matches_found:
        print(f"\n  SUCCESS — found {len(matches_found)} matches with xG data\n")
        print("  Sample (last 5):")
        for m in matches_found[-5:]:
            print(f"    {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}"
                  f"  (xG: {m['home_xg']:.2f} - {m['away_xg']:.2f})")
        with open("fbref_bundesliga2_test.json", "w") as f:
            json.dump(matches_found, f, indent=2)
        print(f"\n  Full data saved to fbref_bundesliga2_test.json")
    else:
        print("\n  Table found but no xG rows extracted.")
        print("  Saving HTML snippet to fbref_debug.html for inspection...")
        with open("fbref_debug.html", "w") as f:
            f.write(html)
        print("  Open fbref_debug.html and search for 'xg' to check the column names.")


if __name__ == "__main__":
    scrape()
