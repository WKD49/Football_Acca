"""
scrape_fbref_xg.py

Scrapes xG data from FBref using a real browser (Playwright/Chromium).
FBref can't block this because it looks identical to a human visit.

Saves results to .cache/fbref_xg_{league_id}.json
The main model will pick this up automatically next run.

Supported leagues (add more by extending LEAGUES below):
    BUNDESLIGA2, SERIEB, LIGUE2

Usage:
    python3 scrape_fbref_xg.py

For OpenClaw nightly job:
    python3 /path/to/scrape_fbref_xg.py
"""

import json
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Leagues to scrape
# FBref URL structure: /en/comps/{comp_id}/{season}/schedule/{season}-{name}-Scores-and-Fixtures
# ---------------------------------------------------------------------------
LEAGUES = [
    {
        "id":       "BUNDESLIGA2",
        "comp_id":  33,
        "season":   "2025-2026",
        "slug":     "2-Bundesliga-Scores-and-Fixtures",
    },
    {
        "id":       "SERIEB",
        "comp_id":  18,
        "season":   "2025-2026",
        "slug":     "Serie-B-Scores-and-Fixtures",
    },
    {
        "id":       "LIGUE2",
        "comp_id":  60,
        "season":   "2025-2026",
        "slug":     "Ligue-2-Scores-and-Fixtures",
    },
]

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def scrape_league(page, league: dict) -> list:
    url = (
        f"https://fbref.com/en/comps/{league['comp_id']}"
        f"/{league['season']}/schedule"
        f"/{league['season']}-{league['slug']}"
    )
    print(f"\n  Fetching: {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    # Wait for the schedule table element specifically
    try:
        page.wait_for_selector("table", timeout=15000)
    except Exception:
        pass

    time.sleep(3)
    html = page.content()

    # Always save a debug snapshot for the first league so we can inspect it
    debug_path = Path(f"fbref_debug_{league['id']}.html")
    debug_path.write_text(html)
    print(f"  Page title: {page.title()}")
    print(f"  HTML length: {len(html)} chars — saved to {debug_path}")

    if 'data-stat="home_xg"' not in html:
        print(f"  WARNING: xG columns not found in page HTML")
        return []

    matches = []
    # Find rows with xG data
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    for row in rows:
        home    = re.search(r'data-stat="home_team"[^>]*>.*?<a[^>]*>([^<]+)</a>', row, re.DOTALL)
        away    = re.search(r'data-stat="away_team"[^>]*>.*?<a[^>]*>([^<]+)</a>', row, re.DOTALL)
        score   = re.search(r'data-stat="score"[^>]*><a[^>]*>([0-9]+)[–\-]([0-9]+)</a>', row)
        home_xg = re.search(r'data-stat="home_xg"[^>]*>\s*([0-9]+\.[0-9]+)', row)
        away_xg = re.search(r'data-stat="away_xg"[^>]*>\s*([0-9]+\.[0-9]+)', row)

        if home and away and score and home_xg and away_xg:
            matches.append({
                "home":       home.group(1).strip(),
                "away":       away.group(1).strip(),
                "home_goals": int(score.group(1)),
                "away_goals": int(score.group(2)),
                "home_xg":    float(home_xg.group(1)),
                "away_xg":    float(away_xg.group(1)),
            })

    return matches


def main():
    print("FBref xG scraper — using real browser (Playwright/Chromium)")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            locale="en-GB",
        )
        page = context.new_page()

        # Visit FBref homepage first to get cookies — like a human would
        print("\nVisiting FBref homepage to establish session...")
        page.goto("https://fbref.com/en/", wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)

        for league in LEAGUES:
            print(f"\nScraping {league['id']}...")
            try:
                matches = scrape_league(page, league)
                if matches:
                    cache_path = CACHE_DIR / f"fbref_xg_{league['id']}.json"
                    cache_path.write_text(json.dumps({"matches": matches}))
                    print(f"  ✓  {len(matches)} matches saved to {cache_path}")
                    # Show a couple of samples
                    for m in matches[-3:]:
                        print(f"     {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}"
                              f"  xG: {m['home_xg']:.2f}-{m['away_xg']:.2f}")
                else:
                    print(f"  ✗  No matches found — check the URL or table structure")
            except Exception as e:
                print(f"  ✗  Error: {e}")

            # Be polite between leagues
            time.sleep(4)

        browser.close()

    print("\n" + "=" * 60)
    print("Done. Run python3 run.py or streamlit run app.py to use the new xG data.")


if __name__ == "__main__":
    main()
