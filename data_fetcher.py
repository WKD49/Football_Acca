"""
data_fetcher.py

Pulls real match results and odds into football_value_acca.py data structures.

Primary free sources:
  - football-data.org  : match results / form  (EPL, Championship, La Liga, Primeira Liga)
  - the-odds-api.com   : live bookmaker odds   (most leagues above + Belgium, Scandinavia, etc.)

Setup:
  1. Sign up (free, instant, no card) at:
       https://www.football-data.org/client/register
       https://the-odds-api.com/
  2. Set environment variables before running:
       export FOOTBALL_DATA_KEY=your_key_here
       export ODDS_API_KEY=your_key_here
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from football_value_acca import (
    LEAGUE_CONFIGS,
    Market,
    Match,
    OddsSnapshot,
    ScheduleContext,
    TeamFeatures,
)


# ---------------------------------------------------------------------------
# League / competition mappings
# ---------------------------------------------------------------------------

# football-data.org competition codes -> our internal league IDs
FD_COMPETITION_MAP: Dict[str, str] = {
    "PL":  "EPL",
    "ELC": "CHAMP",
    "PD":  "LALIGA",
    "PPL": "PRIMEIRA",
}

# The Odds API sport keys -> our internal league IDs
# Keys confirmed against live API response March 2026
ODDS_SPORT_MAP: Dict[str, str] = {
    "soccer_epl":                    "EPL",
    "soccer_efl_champ":              "CHAMP",
    "soccer_spain_la_liga":          "LALIGA",
    "soccer_portugal_primeira_liga": "PRIMEIRA",
    "soccer_belgium_first_div":      "PRO_LEAGUE",
    "soccer_sweden_allsvenskan":     "ALLSVENSKAN",
    # Norway (soccer_norway_eliteserien) and Denmark (soccer_denmark_superliga)
    # are NOT available on The Odds API free tier — omitted.
    "soccer_japan_j_league":         "JLEAGUE",
    "soccer_australia_aleague":      "ALEAGUE",
    "soccer_brazil_campeonato":      "BRASILEIRAO",  # bonus — on both APIs
    "soccer_germany_bundesliga2":    "BUNDESLIGA2",
    "soccer_italy_serie_b":          "SERIEB",
    "soccer_france_ligue_two":       "LIGUE2",
    "soccer_mexico_ligamx":          "LIGAMX",
    "soccer_brazil_serie_b":         "BRASILEIRAO_B",
}

# European competition names (used to detect Europe midweek penalty flag)
EUROPEAN_COMPETITIONS = {
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
}


# ---------------------------------------------------------------------------
# Team name normalisation
# Different APIs use different names for the same club. Add entries here as needed.
# ---------------------------------------------------------------------------

_NAME_OVERRIDES: Dict[str, str] = {
    # football-data.org names -> canonical
    "Arsenal FC":                "Arsenal",
    # API-Football (RapidAPI) names -> canonical
    "Bayern München":            "Bayern Munich",
    "Internazionale":            "Inter Milan",
    "Paris Saint-Germain":       "PSG",
    "Olympique de Marseille":    "Olympique de Marseille",
    "Olympique Lyonnais":        "Olympique Lyonnais",
    "AS Monaco":                 "Monaco",
    "LOSC Lille":                "Lille",
    "Stade Rennais":             "Rennes",
    "Eintracht Frankfurt":       "Eintracht Frankfurt",
    "VfB Stuttgart":             "Stuttgart",
    "TSG Hoffenheim":            "Hoffenheim",
    "Sport-Club Freiburg":       "Freiburg",
    "1. FSV Mainz 05":           "Mainz",
    "1. FC Köln":                "Koln",
    "Bayer Leverkusen":          "Bayer Leverkusen",
    "Brighton & Hove Albion FC": "Brighton",
    "Manchester City FC":        "Manchester City",
    "Manchester United FC":      "Manchester United",
    "Liverpool FC":              "Liverpool",
    "Chelsea FC":                "Chelsea",
    "Tottenham Hotspur FC":      "Tottenham",
    "Real Madrid CF":            "Real Madrid",
    "FC Barcelona":              "Barcelona",
    "Atlético de Madrid":        "Atletico Madrid",
    "Sport Lisboa e Benfica":    "Benfica",
    "FC Porto":                  "Porto",
    "Sporting CP":               "Sporting CP",
    "RSC Anderlecht":            "Anderlecht",
    "Club Brugge KV":            "Club Brugge",
    "Malmö FF":                  "Malmo FF",
    "Bodø/Glimt":                "Bodo/Glimt",
    # football-data.org long-form Spanish names -> canonical
    "Rayo Vallecano de Madrid":  "Rayo Vallecano",
    "RC Celta de Vigo":          "Celta Vigo",
    # The Odds API names -> canonical
    "Brighton and Hove Albion":  "Brighton",
    "Atlético Madrid":           "Atletico Madrid",   # accent + no "de"
    "Athletic Bilbao":           "Athletic Club",
    "Oviedo":                    "Real Oviedo",
    "Celta Vigo":                "Celta Vigo",        # already canonical, prevents prefix-strip mangling
    "AVS Futebol SAD":           "AVS",
    "AVS Futebol":               "AVS",
    "Sporting Lisbon":           "Sporting CP",
    # Understat.com names -> canonical
    "Paris Saint Germain":       "PSG",
    "Inter":                     "Inter Milan",
    "Hellas Verona":             "Verona",
    "Wolverhampton Wanderers":   "Wolves",
    "Nottingham Forest":         "Nottm Forest",
    "Leicester":                 "Leicester City",
    # ClubElo names -> canonical
    "Bayern":                    "Bayern Munich",
    "Paris SG":                  "PSG",
    "Man City":                  "Manchester City",
    "Man United":                "Manchester United",
    "Atletico":                  "Atletico Madrid",
    "Sporting":                  "Sporting CP",
    "Newcastle":                 "Newcastle United",
    "Dortmund":                  "Borussia Dortmund",
    "Leverkusen":                "Bayer Leverkusen",
    "Milan":                     "AC Milan",
    "Forest":                    "Nottm Forest",
    "Betis":                     "Real Betis",
    "Celta":                     "Celta Vigo",
}

_STRIP_SUFFIXES  = (" FC", " AFC", " CF", " SC", " FK", " IF", " BK", " SK", " KV",
                    " SAD", " SL")
_STRIP_PREFIXES  = ("FC ", "RC ", "CF ", "CD ", "UD ", "SD ", "RCD ", "CA ", "AD ", "SC ")


def normalise_team_name(raw: str) -> str:
    """Convert external API team names to the canonical names used internally."""
    if raw in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[raw]
    name = raw.strip()
    # Strip trailing suffixes (e.g. "Arsenal FC" -> "Arsenal")
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    # Strip leading prefixes (e.g. "FC Famalicão" -> "Famalicão")
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
            break
    return name


# ---------------------------------------------------------------------------
# Simple file cache
# Saves API responses as JSON so we don't burn rate limits re-fetching
# during the same session. Results are cached for 1 hour; odds for 30 mins.
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".cache")


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def _cache_load(key: str, max_age_seconds: int = 3600) -> Optional[dict]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_seconds:
        return None
    return json.loads(path.read_text())


def _cache_save(key: str, data: dict) -> None:
    _cache_path(key).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# football-data.org client
# ---------------------------------------------------------------------------

class FootballDataClient:
    """
    Wraps the football-data.org v4 REST API.

    Free tier covers: EPL (PL), Championship (ELC), La Liga (PD), Primeira Liga (PPL).
    Rate limit: 10 requests/minute.
    Docs: https://www.football-data.org/documentation/api
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers["X-Auth-Token"] = api_key

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_finished_matches(self, competition_code: str, season: int) -> List[dict]:
        """
        All FINISHED matches for a competition in a given season.
        season: the year the season starts (e.g. 2024 = 2024/25)
        """
        cache_key = f"fd_{competition_code}_{season}_finished"
        cached = _cache_load(cache_key, max_age_seconds=3600)
        if cached:
            return cached["matches"]

        data = self._get(
            f"/competitions/{competition_code}/matches",
            params={"season": season, "status": "FINISHED"},
        )
        matches = data.get("matches", [])
        _cache_save(cache_key, {"matches": matches})
        return matches

    def get_upcoming_fixtures(self, competition_code: str, days_ahead: int = 7) -> List[dict]:
        """SCHEDULED matches in the next N days."""
        now = datetime.now(timezone.utc)
        date_from = now.strftime("%Y-%m-%d")
        date_to = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        cache_key = f"fd_{competition_code}_upcoming_{date_from}"
        cached = _cache_load(cache_key, max_age_seconds=1800)
        if cached:
            return cached["matches"]

        data = self._get(
            f"/competitions/{competition_code}/matches",
            params={"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
        )
        matches = data.get("matches", [])
        _cache_save(cache_key, {"matches": matches})
        return matches


# ---------------------------------------------------------------------------
# The Odds API client
# ---------------------------------------------------------------------------

class OddsApiClient:
    """
    Wraps the-odds-api.com v4 API.

    Free tier: 500 requests/month (each call to get_odds() uses 1 request).
    Covers most leagues in ODDS_SPORT_MAP above.
    Docs: https://the-odds-api.com/liveapi/guides/v4/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_odds(
        self,
        sport_key: str,
        regions: str = "uk",
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
    ) -> List[dict]:
        """
        Fetch current pre-match odds for a league.
        sport_key: one of the values in ODDS_SPORT_MAP (e.g. "soccer_england_premier_league")
        regions:   "uk" returns UK bookmakers (Bet365, Paddy Power, William Hill, etc.)
        markets:   "h2h" = 1X2, "totals" = over/under
        """
        cache_key = f"odds_{sport_key}_{regions}"
        cached = _cache_load(cache_key, max_age_seconds=1800)
        if cached:
            return cached["odds"]

        resp = requests.get(
            f"{self.BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        _cache_save(cache_key, {"odds": data})
        return data


# ---------------------------------------------------------------------------
# API-Football client (api-sports.io direct)
# Covers Bundesliga, Serie A, Ligue 1 — leagues not on football-data.org free tier.
# Free tier: 100 requests/day. Each league fetch = 1 request.
# Sign up: https://www.api-sports.io/
# ---------------------------------------------------------------------------

# API-Football league IDs -> our internal league IDs
AF_LEAGUE_MAP: Dict[int, str] = {
    78:  "BUNDESLIGA",
    79:  "BUNDESLIGA2",
    135: "SERIEA",
    136: "SERIEB",
    61:  "LIGUE1",
    62:  "LIGUE2",
    262: "LIGAMX",
    72:  "BRASILEIRAO_B",
}


class ApiFootballClient:
    """
    Wraps the API-Football v3 REST API (api-sports.io direct endpoint).

    Free tier: 100 requests/day.
    Docs: https://www.api-football.com/documentation-v3
    """

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": api_key,
        })

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_finished_matches(self, league_id: int, season: int) -> List[dict]:
        """
        All finished matches for a league in a given season.
        league_id: API-Football league ID (e.g. 78 = Bundesliga)
        season: start year of the season (e.g. 2025 = 2025/26)
        """
        cache_key = f"af_{league_id}_{season}_finished"
        cached = _cache_load(cache_key, max_age_seconds=3600)
        if cached:
            return cached["matches"]

        data = self._get(
            "/fixtures",
            params={"league": league_id, "season": season, "status": "FT"},
        )
        matches = data.get("response", [])
        _cache_save(cache_key, {"matches": matches})
        return matches

    def get_recently_finished(self, league_id: int, season: int, days_back: int = 35) -> List[dict]:
        """Finished matches in the last N days — much smaller than the full season dump."""
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to   = now.strftime("%Y-%m-%d")

        cache_key = f"af_{league_id}_{season}_recent_{date_from}"
        cached = _cache_load(cache_key, max_age_seconds=3600)
        if cached:
            return cached["matches"]

        data = self._get(
            "/fixtures",
            params={"league": league_id, "season": season,
                    "from": date_from, "to": date_to, "status": "FT"},
        )
        matches = data.get("response", [])
        _cache_save(cache_key, {"matches": matches})
        return matches

    def get_upcoming_fixtures(self, league_id: int, season: int, days_ahead: int = 21) -> List[dict]:
        """Upcoming (not-started) fixtures for a league, within the next N days."""
        now = datetime.now(timezone.utc)
        date_from = now.strftime("%Y-%m-%d")
        date_to   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        cache_key = f"af_{league_id}_{season}_upcoming_{date_from}"
        cached = _cache_load(cache_key, max_age_seconds=1800)
        if cached:
            return cached["fixtures"]

        data = self._get(
            "/fixtures",
            params={"league": league_id, "season": season,
                    "from": date_from, "to": date_to, "status": "NS"},
        )
        fixtures = data.get("response", [])
        _cache_save(cache_key, {"fixtures": fixtures})
        return fixtures

    def parse_result(self, raw: dict) -> Optional[dict]:
        """Convert an API-Football fixture dict into our internal format."""
        try:
            hg = raw["goals"]["home"]
            ag = raw["goals"]["away"]
            if hg is None or ag is None:
                return None
            return {
                "date":       datetime.fromisoformat(
                                  raw["fixture"]["date"].replace("Z", "+00:00")
                              ),
                "home":       normalise_team_name(raw["teams"]["home"]["name"]),
                "away":       normalise_team_name(raw["teams"]["away"]["name"]),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "competition": "",
            }
        except (KeyError, TypeError, ValueError):
            return None


def fetch_api_football_results(
    af_client: ApiFootballClient,
    season: int,
) -> List[dict]:
    """
    Fetch finished domestic results for Bundesliga, Serie A, and Ligue 1
    from API-Football and return them in the same parsed format used by
    FormCalculator — ready to pass as extra_results to fetch_euro_competition().

    The free plan only covers up to season 2024. If the requested season
    returns no data, we automatically fall back to the previous season.
    """
    all_results: List[dict] = []
    for league_id in AF_LEAGUE_MAP:
        try:
            raw_matches = af_client.get_finished_matches(league_id, season)
            # Free plan may not cover the current season — fall back one year
            if not raw_matches:
                raw_matches = af_client.get_finished_matches(league_id, season - 1)
            for raw in raw_matches:
                parsed = af_client.parse_result(raw)
                if parsed:
                    all_results.append(parsed)
        except Exception:
            pass  # If one league fails, keep going with the others
    return all_results


# ---------------------------------------------------------------------------
# Club Elo client
# Fetches cross-league Elo ratings from clubelo.com (free, no API key).
# Used in the European model to compare teams from different leagues.
# ---------------------------------------------------------------------------

class EloClient:
    """
    Fetches club Elo ratings from api.clubelo.com.

    Plain English: Elo is a single number that captures a club's overall
    strength across all competitions. Unlike league form (which is relative
    to league average), Elo lets us compare a Bundesliga team directly with
    a Portuguese team — essential for European knockout fixtures.

    Cached for 24 hours (ratings only update after each match day).
    """

    BASE_URL = "http://api.clubelo.com"

    def fetch_ratings(self, date_str: Optional[str] = None) -> Dict[str, float]:
        """
        Return {canonical_team_name: elo} for all clubs on the given date.
        date_str: "YYYY-MM-DD" (defaults to today)
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        cache_key = f"clubelo_{date_str}"
        cached = _cache_load(cache_key, max_age_seconds=86400)  # 24 hours
        if cached:
            return cached["ratings"]

        try:
            r = requests.get(f"{self.BASE_URL}/{date_str}", timeout=10)
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text))
            ratings: Dict[str, float] = {}
            for row in reader:
                try:
                    name = normalise_team_name(row["Club"])
                    ratings[name] = float(row["Elo"])
                except (KeyError, ValueError):
                    continue
        except Exception:
            return {}

        _cache_save(cache_key, {"ratings": ratings})
        return ratings


def _apply_elo_factors(
    home_tf: "TeamFeatures",
    away_tf: "TeamFeatures",
    home_elo: float,
    away_elo: float,
    alpha: float = 0.4,
) -> Tuple["TeamFeatures", "TeamFeatures"]:
    """
    Adjust TeamFeatures attack multipliers based on the Elo difference.

    Plain English: if Real Madrid (Elo 1956) faces Galatasaray (Elo 1650),
    Real Madrid's attack is boosted ~5% and Galatasaray's is cut ~5%.
    This corrects for the fact that league-relative form cannot compare
    teams across different countries.

    alpha controls sensitivity: 0.4 gives ~2–6% adjustment for typical
    CL matchups (100–300 Elo point gaps). Capped at ±15%.
    """
    avg_elo = (home_elo + away_elo) / 2.0
    home_factor = max(0.85, min(1.15, (home_elo / avg_elo) ** alpha))
    away_factor = max(0.85, min(1.15, (away_elo / avg_elo) ** alpha))

    def scale_tf(tf: "TeamFeatures", factor: float) -> "TeamFeatures":
        from football_value_acca import TeamFeatures as TF
        return TF(
            home_attack_long=  tf.home_attack_long  * factor,
            home_attack_short= tf.home_attack_short * factor,
            home_defence_long= tf.home_defence_long,
            home_defence_short=tf.home_defence_short,
            away_attack_long=  tf.away_attack_long  * factor,
            away_attack_short= tf.away_attack_short * factor,
            away_defence_long= tf.away_defence_long,
            away_defence_short=tf.away_defence_short,
            strict_validation=False,
        )

    return scale_tf(home_tf, home_factor), scale_tf(away_tf, away_factor)


# ---------------------------------------------------------------------------
# Understat xG client
# Scrapes expected-goals data from understat.com (free, no API key).
# Supports EPL, La Liga, Bundesliga, Serie A, Ligue 1.
# Cache TTL: 6 hours (data only changes when matches finish).
# ---------------------------------------------------------------------------

class UnderstatClient:
    """
    Fetches xG (expected goals) data by scraping understat.com.

    Plain English: xG is a measure of how many goals a team *should* have
    scored based on the quality of their chances. Blending xG with actual
    goals gives the model a more stable picture of a team's true strength —
    a team that scores 3 goals from 1 xG got lucky; a team with 3 xG that
    scored 1 was unlucky. The blend is 65% xG + 35% actual goals.

    Understat covers EPL, La Liga, Bundesliga, Serie A, and Ligue 1.
    Championship and Primeira Liga fall back to goals only.
    """

    LEAGUE_MAP: Dict[str, str] = {
        "EPL":       "EPL",
        "LALIGA":    "La_liga",
        "BUNDESLIGA":"Bundesliga",
        "SERIEA":    "Serie_A",
        "LIGUE1":    "Ligue_1",
    }

    _UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    def get_matches(self, league_id: str, season: int) -> List[dict]:
        """
        Return finished matches with xG for the given league/season.
        Each dict: {home, away, home_goals, away_goals, home_xg, away_xg}.
        Returns [] if the league is not supported or the request fails.
        """
        understat_key = self.LEAGUE_MAP.get(league_id)
        if not understat_key:
            return []

        cache_key = f"understat_{league_id}_{season}"
        cached = _cache_load(cache_key, max_age_seconds=21600)  # 6 hours
        if cached:
            return cached["matches"]

        # Understat requires a session cookie obtained by visiting the league page
        # before it will serve the JSON data endpoint.
        try:
            session = requests.Session()
            session.headers["User-Agent"] = self._UA
            session.get(f"https://understat.com/league/{understat_key}/{season}", timeout=15)
            r = session.get(
                f"https://understat.com/getLeagueData/{understat_key}/{season}",
                timeout=20,
                headers={
                    "Referer": f"https://understat.com/league/{understat_key}/{season}",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            r.raise_for_status()
            raw = json.loads(r.text)
            entries = raw.get("dates", [])
        except Exception:
            return []

        matches: List[dict] = []
        for entry in entries:
            if not entry.get("isResult"):
                continue
            try:
                home_name = normalise_team_name(entry["h"]["title"])
                away_name = normalise_team_name(entry["a"]["title"])
                matches.append({
                    "home":       home_name,
                    "away":       away_name,
                    "home_goals": int(entry["goals"]["h"]),
                    "away_goals": int(entry["goals"]["a"]),
                    "home_xg":    float(entry["xG"]["h"]),
                    "away_xg":    float(entry["xG"]["a"]),
                })
            except (KeyError, TypeError, ValueError):
                continue

        _cache_save(cache_key, {"matches": matches})
        return matches


def _blend_xg(parsed: List[dict], xg_matches: List[dict]) -> List[dict]:
    """
    Replace home_goals/away_goals with 0.65*xG + 0.35*actual for any
    match that has xG data. Unmatched matches are left unchanged.
    """
    if not xg_matches:
        return parsed
    xg_lookup = {(m["home"], m["away"]): (m["home_xg"], m["away_xg"]) for m in xg_matches}
    enriched = []
    for p in parsed:
        key = (p["home"], p["away"])
        if key in xg_lookup:
            hxg, axg = xg_lookup[key]
            enriched.append({
                **p,
                "home_goals": round(0.65 * hxg + 0.35 * p["home_goals"], 3),
                "away_goals": round(0.65 * axg + 0.35 * p["away_goals"], 3),
            })
        else:
            enriched.append(p)
    return enriched


# ---------------------------------------------------------------------------
# Form calculator: raw match results -> TeamFeatures multipliers
# ---------------------------------------------------------------------------

class FormCalculator:
    """
    Converts a season's match results into the league-normalised multipliers
    that TeamFeatures expects.

    Plain English: for each team, we look at how many goals they score/concede
    compared to the league average, split by whether they're at home or away.
    A multiplier of 1.15 means 15% above average; 0.90 means 10% below.
    """

    def __init__(self, long_half_life: float = 10.0, short_half_life: float = 3.0):
        """
        Exponentially weighted form decay.

        long_half_life  — matches ago at which a result has half the weight of
                          the most recent game (feeds _long fields, ~season-wide view)
        short_half_life — same but for the fast-decay component (~recent 3–5 games)

        Replaces the old hard 25/6-match windows. Benefits:
          · no cliff-edge at the window boundary
          · manager changes and form collapses automatically reflected
          · the short component gives more weight to the last 1–2 games
        """
        self.long_half_life  = long_half_life
        self.short_half_life = short_half_life

    def parse_result(self, raw: dict) -> Optional[dict]:
        """
        Convert a football-data.org match dict into a simple internal format.
        Returns None if the match isn't finished or has no score.
        """
        score = raw.get("score", {})
        ft = score.get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            return None
        return {
            "date":       datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            "home":       normalise_team_name(raw["homeTeam"]["name"]),
            "away":       normalise_team_name(raw["awayTeam"]["name"]),
            "home_goals": int(hg),
            "away_goals": int(ag),
            "competition": raw.get("competition", {}).get("name", ""),
        }

    def league_averages(self, results: List[dict]) -> Tuple[float, float]:
        """
        Returns (avg_home_goals_per_match, avg_away_goals_per_match).
        Used to normalise each team's figures against the league baseline.
        """
        if not results:
            return 1.3, 1.0
        return (
            sum(r["home_goals"] for r in results) / len(results),
            sum(r["away_goals"] for r in results) / len(results),
        )

    def compute(
        self,
        team: str,
        all_results: List[dict],
        league_home_avg: float,
        league_away_avg: float,
        strict_validation: bool = False,
    ) -> TeamFeatures:
        """
        Compute TeamFeatures for one team.

        all_results: every parsed match in the league this season
        league_home_avg / league_away_avg: from league_averages() above
        """
        home_matches = sorted(
            [r for r in all_results if r["home"] == team],
            key=lambda r: r["date"], reverse=True,
        )
        away_matches = sorted(
            [r for r in all_results if r["away"] == team],
            key=lambda r: r["date"], reverse=True,
        )

        def ewma(matches: List[dict], scored: bool, half_life: float) -> float:
            """
            Exponentially weighted average of goals, most-recent-first.
            weight_i = alpha^i  where  alpha = 0.5^(1/half_life)
            Returns raw goals per match (normalised to a multiplier by caller).
            """
            if not matches:
                return 1.0  # will become 1.0/league_avg ≈ 1.0 below
            alpha = 0.5 ** (1.0 / half_life)
            total_w = 0.0
            total_g = 0.0
            for i, m in enumerate(matches):
                w = alpha ** i
                if m["home"] == team:
                    g = m["home_goals"] if scored else m["away_goals"]
                else:
                    g = m["away_goals"] if scored else m["home_goals"]
                total_g += w * g
                total_w += w
            return total_g / total_w if total_w > 0 else 1.0

        def ratio(team_avg: float, league_avg: float) -> float:
            return team_avg / league_avg if league_avg > 0 else 1.0

        return TeamFeatures(
            home_attack_long=  ratio(ewma(home_matches, scored=True,  half_life=self.long_half_life),  league_home_avg),
            home_attack_short= ratio(ewma(home_matches, scored=True,  half_life=self.short_half_life), league_home_avg),
            home_defence_long= ratio(ewma(home_matches, scored=False, half_life=self.long_half_life),  league_away_avg),
            home_defence_short=ratio(ewma(home_matches, scored=False, half_life=self.short_half_life), league_away_avg),
            away_attack_long=  ratio(ewma(away_matches, scored=True,  half_life=self.long_half_life),  league_away_avg),
            away_attack_short= ratio(ewma(away_matches, scored=True,  half_life=self.short_half_life), league_away_avg),
            away_defence_long= ratio(ewma(away_matches, scored=False, half_life=self.long_half_life),  league_home_avg),
            away_defence_short=ratio(ewma(away_matches, scored=False, half_life=self.short_half_life), league_home_avg),
            strict_validation=strict_validation,
        )


# ---------------------------------------------------------------------------
# Schedule context builder
# ---------------------------------------------------------------------------

def build_schedule_context(
    team: str,
    fixture_date: datetime,
    all_results: List[dict],
) -> ScheduleContext:
    """
    Derive rest days, midweek flags, and Europe flags from match history.
    Tip: for more accurate rest days, pass results from ALL competitions,
    not just the league (so cup + European matches count towards fatigue).
    """
    prior = sorted(
        [r for r in all_results
         if (r["home"] == team or r["away"] == team) and r["date"] < fixture_date],
        key=lambda r: r["date"], reverse=True,
    )

    if not prior:
        return ScheduleContext(
            rest_days=7, played_midweek=False,
            played_europe_midweek=False, between_two_europe_legs=False,
        )

    last = prior[0]
    rest_days = max(0, (fixture_date - last["date"]).days)
    played_midweek = last["date"].weekday() in (1, 2, 3)  # Tue/Wed/Thu
    played_europe_midweek = played_midweek and last.get("competition", "") in EUROPEAN_COMPETITIONS

    # Check if there's a European match coming up shortly after this fixture
    future_europe = [
        r for r in all_results
        if (r["home"] == team or r["away"] == team)
        and r["date"] > fixture_date
        and r.get("competition", "") in EUROPEAN_COMPETITIONS
    ]
    days_to_next_europe = (
        min(r["date"] for r in future_europe) - fixture_date
    ).days if future_europe else 99

    return ScheduleContext(
        rest_days=rest_days,
        played_midweek=played_midweek,
        played_europe_midweek=played_europe_midweek,
        between_two_europe_legs=played_europe_midweek and days_to_next_europe <= 4,
    )


# ---------------------------------------------------------------------------
# Head-to-head helper
# ---------------------------------------------------------------------------

def compute_h2h_edge(home: str, away: str, results: List[dict], n: int = 6) -> float:
    """
    Return a lam_home multiplier based on the last N H2H meetings.
    Range: ~0.96–1.04. Returns 1.0 (neutral) if fewer than 3 meetings found.

    Counts wins/losses for the upcoming home team across all past meetings,
    regardless of which side they were on in each individual game.
    """
    h2h = [
        r for r in results
        if (r["home"] == home and r["away"] == away)
        or (r["home"] == away and r["away"] == home)
    ]
    h2h = sorted(h2h, key=lambda r: r["date"])[-n:]

    if len(h2h) < 3:
        return 1.0

    home_wins = 0
    for r in h2h:
        if r["home"] == home:
            hg, ag = r["home_goals"], r["away_goals"]
        else:
            hg, ag = r["away_goals"], r["home_goals"]
        if hg > ag:
            home_wins += 1

    home_rate = home_wins / len(h2h)
    # Centre on 0.33 (typical home win rate); max ±4% effect
    edge = max(-0.04, min(0.04, (home_rate - 0.33) * 0.12))
    return 1.0 + edge


# ---------------------------------------------------------------------------
# Odds parser: The Odds API event -> OddsSnapshot
# ---------------------------------------------------------------------------

def best_odds_for_selection(
    odds_event: dict,
    market_kind: str,
    selection: str,
    line: Optional[float] = None,
    bookmaker_filter: Optional[str] = None,
) -> Optional[OddsSnapshot]:
    """
    Find decimal odds for a given market/selection.

    bookmaker_filter: if set (e.g. "bet365"), only use that bookmaker.
                      If None, takes the best price across all bookmakers.
                      If the filtered bookmaker doesn't offer this market,
                      returns None so the fixture is skipped for that acca.
    """
    now = datetime.now(timezone.utc)
    home_team = normalise_team_name(odds_event.get("home_team", ""))
    away_team = normalise_team_name(odds_event.get("away_team", ""))

    best_price: Optional[float] = None
    best_book = "unknown"

    for bookmaker in odds_event.get("bookmakers", []):
        if bookmaker_filter and bookmaker["key"] != bookmaker_filter:
            continue
        for market in bookmaker.get("markets", []):

            if market_kind == "1X2" and market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    name = normalise_team_name(outcome["name"])
                    price = float(outcome["price"])
                    match = (
                        (selection == "HOME" and name == home_team) or
                        (selection == "AWAY" and name == away_team) or
                        (selection == "DRAW" and outcome["name"] == "Draw")
                    )
                    if match and (best_price is None or price > best_price):
                        best_price, best_book = price, bookmaker["key"]

            elif market_kind == "OVER_UNDER" and market["key"] == "totals":
                for outcome in market["outcomes"]:
                    point = outcome.get("point", line)
                    if point is None or abs(float(point) - line) > 0.01:
                        continue
                    price = float(outcome["price"])
                    match = (
                        (selection == "OVER"  and outcome["name"] == "Over") or
                        (selection == "UNDER" and outcome["name"] == "Under")
                    )
                    if match and (best_price is None or price > best_price):
                        best_price, best_book = price, bookmaker["key"]

    if best_price is None:
        return None
    return OddsSnapshot(decimal_odds=best_price, timestamp_utc=now, bookmaker=best_book)


# ---------------------------------------------------------------------------
# High-level: pull everything together for a competition
# ---------------------------------------------------------------------------

def fetch_competition(
    competition_code: str,
    league_id: str,
    fd_client: FootballDataClient,
    odds_client: OddsApiClient,
    odds_sport_key: str,
    season: int,
    days_ahead: int = 7,
    bookmaker_filter: Optional[str] = None,
) -> List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]]:
    """
    Full pipeline for one competition:
      1. Fetch finished results -> compute form (TeamFeatures)
      2. Fetch upcoming fixtures
      3. Fetch current odds
      4. Assemble Match objects + market/odds pairs

    Returns a list of (Match, [(Market, OddsSnapshot), ...]) ready to feed
    into build_candidate() in football_value_acca.py.
    """
    calc = FormCalculator()

    # Step 1: results -> form (current season)
    raw_finished = fd_client.get_finished_matches(competition_code, season)
    parsed = [calc.parse_result(m) for m in raw_finished]
    parsed = [p for p in parsed if p is not None]

    # Enrich with xG where Understat has coverage (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
    xg_data = UnderstatClient().get_matches(league_id, season)
    parsed = _blend_xg(parsed, xg_data)

    league_home_avg, league_away_avg = calc.league_averages(parsed)

    # Also fetch previous season for H2H (cached separately, no extra cost per run)
    raw_prev = fd_client.get_finished_matches(competition_code, season - 1)
    parsed_prev = [calc.parse_result(m) for m in raw_prev]
    h2h_pool = parsed + [p for p in parsed_prev if p is not None]

    # Step 2: upcoming fixtures
    raw_upcoming = fd_client.get_upcoming_fixtures(competition_code, days_ahead)

    # Step 3: odds
    odds_events = odds_client.get_odds(odds_sport_key)
    odds_lookup: Dict[Tuple[str, str], dict] = {}
    for ev in odds_events:
        h = normalise_team_name(ev.get("home_team", ""))
        a = normalise_team_name(ev.get("away_team", ""))
        odds_lookup[(h, a)] = ev

    # Step 4: assemble
    output: List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]] = []
    data_quality = 0.85 if league_id in ("EPL", "LALIGA") else 0.75

    for raw in raw_upcoming:
        home = normalise_team_name(raw["homeTeam"]["name"])
        away = normalise_team_name(raw["awayTeam"]["name"])
        try:
            kickoff = datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue

        home_features = calc.compute(home, parsed, league_home_avg, league_away_avg)
        away_features = calc.compute(away, parsed, league_home_avg, league_away_avg)
        home_sched = build_schedule_context(home, kickoff, parsed)
        away_sched = build_schedule_context(away, kickoff, parsed)

        match = Match(
            match_id=str(raw.get("id", f"{league_id}_{home}_{away}")),
            league=league_id,
            kickoff_utc=kickoff,
            home_team=home,
            away_team=away,
            home_features=home_features,
            away_features=away_features,
            home_schedule=home_sched,
            away_schedule=away_sched,
            h2h_home_edge=compute_h2h_edge(home, away, h2h_pool),
            league_data_quality=data_quality,
        )

        # Markets + odds (filtered to bookmaker_filter if set)
        market_odds: List[Tuple[Market, OddsSnapshot]] = []
        ev = odds_lookup.get((home, away))
        if ev:
            for sel in ("HOME", "DRAW", "AWAY"):
                snap = best_odds_for_selection(ev, "1X2", sel, bookmaker_filter=bookmaker_filter)
                if snap:
                    market_odds.append((Market(kind="1X2", selection=sel), snap))
            for sel in ("OVER", "UNDER"):
                snap = best_odds_for_selection(ev, "OVER_UNDER", sel, line=2.5, bookmaker_filter=bookmaker_filter)
                # Fall back to best available if chosen bookmaker doesn't offer totals
                if snap is None and bookmaker_filter:
                    snap = best_odds_for_selection(ev, "OVER_UNDER", sel, line=2.5, bookmaker_filter=None)
                if snap:
                    market_odds.append((Market(kind="OVER_UNDER", line=2.5, selection=sel), snap))

        output.append((match, market_odds))

    return output


# ---------------------------------------------------------------------------
# European-only fetch
# Fixtures + odds from The Odds API; form from football-data.org if available.
# ---------------------------------------------------------------------------

# CL/EL competition codes on football-data.org (finished matches only —
# upcoming scheduled fixtures require a paid tier)
_EURO_FD_CODES: Dict[str, str] = {
    "UCL": "CL",
    "UEL": "EL",
}

# 25/6 window — same as domestic, but the pool includes both domestic
# league results and CL/EL league-phase results combined.
_EURO_CALC = FormCalculator()  # uses default half-lives (10 long, 3 short)

_NEUTRAL_FEATURES = TeamFeatures(
    home_attack_long=1.0,  home_attack_short=1.0,
    home_defence_long=1.0, home_defence_short=1.0,
    away_attack_long=1.0,  away_attack_short=1.0,
    away_defence_long=1.0, away_defence_short=1.0,
    strict_validation=False,
)

# European knockout: fatigue/schedule flags not relevant — always neutral.
_EURO_SCHED = ScheduleContext(
    rest_days=7, played_midweek=False,
    played_europe_midweek=False, between_two_europe_legs=False,
)


def fetch_euro_competition(
    league_id: str,
    odds_client: OddsApiClient,
    odds_sport_key: str,
    season: int,
    fd_client: Optional[FootballDataClient] = None,
    extra_results: Optional[List[dict]] = None,
    bookmaker_filter: Optional[str] = None,
    af_client: Optional[ApiFootballClient] = None,
    af_league_id: Optional[int] = None,
) -> List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]]:
    """
    Fetch European knockout fixtures and odds.

    Fixtures + odds: The Odds API.

    Form data: CL/EL league-phase results from football-data.org (if available)
    combined with domestic league results passed in via extra_results.
    The merged pool is used with a 25-match window — the same as the domestic
    model — giving each team a full picture of recent form across competitions.

    Schedule fatigue flags are not applied — all games here are European.
    """
    # ── CL/EL form from football-data.org ──────────────────────────────────
    euro_parsed: List[dict] = []
    fd_code = _EURO_FD_CODES.get(league_id)
    if fd_client and fd_code:
        try:
            raw_finished = fd_client.get_finished_matches(fd_code, season)
            euro_parsed = [_EURO_CALC.parse_result(m) for m in raw_finished]
            euro_parsed = [p for p in euro_parsed if p is not None]
        except Exception:
            euro_parsed = []

    # ── Merge with domestic results, giving European games 1.1× weight ──────
    # Scaling goals by 1.1 means each CL/EL result has slightly more influence
    # on a team's computed form ratio than a domestic match.
    euro_weighted = [
        {**r, "home_goals": r["home_goals"] * 1.1, "away_goals": r["away_goals"] * 1.1}
        for r in euro_parsed
    ]
    parsed = euro_weighted + (extra_results or [])
    league_home_avg, league_away_avg = _EURO_CALC.league_averages(parsed)
    has_form = bool(parsed)

    # ── Odds lookup from The Odds API (keyed by normalised team names) ─────
    odds_events = odds_client.get_odds(odds_sport_key)
    odds_lookup: Dict[Tuple[str, str], dict] = {}
    for ev in odds_events:
        h = normalise_team_name(ev.get("home_team", ""))
        a = normalise_team_name(ev.get("away_team", ""))
        odds_lookup[(h, a)] = ev

    output: List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]] = []

    # ── Club Elo ratings (cross-league strength calibration) ─────────────
    elo_ratings: Dict[str, float] = EloClient().fetch_ratings()

    # ── Fixture source: API-Football (complete CL/EL coverage) if available,
    #    otherwise fall back to The Odds API events ─────────────────────────
    af_upcoming: List[dict] = []
    if af_client and af_league_id:
        try:
            af_upcoming = af_client.get_upcoming_fixtures(af_league_id, season)
        except Exception:
            af_upcoming = []

    def _build_match_and_odds(home: str, away: str, kickoff: datetime, match_id: str):
        if has_form:
            hf = _EURO_CALC.compute(home, parsed, league_home_avg, league_away_avg)
            af_ = _EURO_CALC.compute(away, parsed, league_home_avg, league_away_avg)
            dq = 0.72
        else:
            hf = _NEUTRAL_FEATURES
            af_ = _NEUTRAL_FEATURES
            dq = 0.60

        # Apply Elo-based cross-league strength adjustment if both teams are rated
        home_elo = elo_ratings.get(home)
        away_elo = elo_ratings.get(away)
        if home_elo and away_elo:
            hf, af_ = _apply_elo_factors(hf, af_, home_elo, away_elo)

        m = Match(
            match_id=match_id,
            league=league_id,
            kickoff_utc=kickoff,
            home_team=home,
            away_team=away,
            home_features=hf,
            away_features=af_,
            home_schedule=_EURO_SCHED,
            away_schedule=_EURO_SCHED,
            league_data_quality=dq,
        )

        ev = odds_lookup.get((home, away))
        mo: List[Tuple[Market, OddsSnapshot]] = []
        if ev:
            for sel in ("HOME", "DRAW", "AWAY"):
                snap = best_odds_for_selection(ev, "1X2", sel, bookmaker_filter=bookmaker_filter)
                if snap:
                    mo.append((Market(kind="1X2", selection=sel), snap))
            for sel in ("OVER", "UNDER"):
                snap = best_odds_for_selection(ev, "OVER_UNDER", sel, line=2.5, bookmaker_filter=bookmaker_filter)
                if snap is None and bookmaker_filter:
                    snap = best_odds_for_selection(ev, "OVER_UNDER", sel, line=2.5, bookmaker_filter=None)
                if snap:
                    mo.append((Market(kind="OVER_UNDER", line=2.5, selection=sel), snap))
        return m, mo

    if af_upcoming:
        seen: set = set()
        for raw in af_upcoming:
            try:
                home = normalise_team_name(raw["teams"]["home"]["name"])
                away = normalise_team_name(raw["teams"]["away"]["name"])
                kickoff = datetime.fromisoformat(
                    raw["fixture"]["date"].replace("Z", "+00:00")
                )
                match_id = str(raw["fixture"]["id"])
            except (KeyError, ValueError):
                continue
            key = (home, away)
            if key in seen:
                continue
            seen.add(key)
            output.append(_build_match_and_odds(home, away, kickoff, match_id))
    else:
        # Fallback: use The Odds API events as fixture source
        for ev in odds_events:
            home = normalise_team_name(ev.get("home_team", ""))
            away = normalise_team_name(ev.get("away_team", ""))
            try:
                kickoff = datetime.fromisoformat(
                    ev.get("commence_time", "").replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                continue
            output.append(_build_match_and_odds(home, away, kickoff,
                                                ev.get("id", f"{league_id}_{home}_{away}")))

    return output


# ---------------------------------------------------------------------------
# Auto-detect first leg scores for European knockout second legs
# ---------------------------------------------------------------------------

# API-Football league IDs for European competitions
_AF_EURO_LEAGUE_IDS = {
    2: "UCL",  # UEFA Champions League
    3: "UEL",  # UEFA Europa League
}


def fetch_euro_first_leg_scores(
    upcoming_fixtures: list,
    fd_client: "FootballDataClient",
    season: int,
    af_client: Optional["ApiFootballClient"] = None,
    days_lookback: int = 35,
) -> Dict[str, Tuple[int, int]]:
    """
    Auto-detect first leg scores for upcoming 2nd-leg CL/EL fixtures.

    Uses API-Football as primary source (full knockout round coverage), with
    football-data.org as fallback. Looks at recently finished CL/EL matches
    (within the last `days_lookback` days) and checks whether the teams in any
    upcoming fixture played each other with home/away reversed.

    Score is expressed from the perspective of the 2nd-leg home team:
      (goals the 2nd-leg home team scored in leg 1,
       goals the 2nd-leg away team scored in leg 1)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    finished: List[dict] = []

    # Primary: API-Football — fetch only the last 35 days to avoid season-dump truncation
    if af_client:
        for league_id in _AF_EURO_LEAGUE_IDS:
            try:
                raw_matches = af_client.get_recently_finished(league_id, season, days_back=days_lookback)
                if not raw_matches:
                    raw_matches = af_client.get_recently_finished(league_id, season - 1, days_back=days_lookback)
                for raw in raw_matches:
                    parsed = af_client.parse_result(raw)
                    if parsed and parsed["date"] >= cutoff:
                        finished.append(parsed)
            except Exception:
                pass

    # Fallback: football-data.org (free tier has patchy knockout coverage)
    if not finished:
        calc = FormCalculator()
        for fd_code in ("CL", "EL"):
            try:
                raw = fd_client.get_finished_matches(fd_code, season)
                for m in raw:
                    parsed = calc.parse_result(m)
                    if parsed and parsed["date"] >= cutoff:
                        finished.append(parsed)
            except Exception:
                pass

    # Build lookup: (home_team, away_team) -> (home_goals, away_goals)
    results_lookup: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for r in finished:
        results_lookup[(r["home"], r["away"])] = (r["home_goals"], r["away_goals"])

    # For each upcoming fixture (A home, B away), look for finished (B home, A away)
    first_leg_scores: Dict[str, Tuple[int, int]] = {}
    for match, _ in upcoming_fixtures:
        home = match.home_team
        away = match.away_team
        leg1 = results_lookup.get((away, home))
        if leg1:
            # leg1[0] = goals 'away' (current) scored as home in leg 1
            # leg1[1] = goals 'home' (current) scored as away in leg 1
            first_leg_scores[f"{home} vs {away}"] = (leg1[1], leg1[0])

    return first_leg_scores


# ---------------------------------------------------------------------------
# Sanity-check runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fd_key   = os.environ.get("FOOTBALL_DATA_KEY", "")
    odds_key = os.environ.get("ODDS_API_KEY", "")

    if not fd_key or not odds_key:
        print("Both API keys are required. Set them as environment variables:")
        print("  export FOOTBALL_DATA_KEY=your_key_here")
        print("  export ODDS_API_KEY=your_key_here")
        print("")
        print("Sign up (free, no card needed) at:")
        print("  https://www.football-data.org/client/register")
        print("  https://the-odds-api.com/")
        raise SystemExit(1)

    fd    = FootballDataClient(fd_key)
    odds  = OddsApiClient(odds_key)

    # season=2025 = the 2025/26 season (started Aug 2025)
    competitions = [
        ("PL",  "EPL",      "soccer_epl"),
        ("ELC", "CHAMP",    "soccer_efl_champ"),
        ("PD",  "LALIGA",   "soccer_spain_la_liga"),
        ("PPL", "PRIMEIRA", "soccer_portugal_primeira_liga"),
    ]

    all_matches: List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]] = []

    for comp_code, league_id, sport_key in competitions:
        print(f"\nFetching {league_id}...")
        try:
            results = fetch_competition(comp_code, league_id, fd, odds, sport_key, season=2025)
            all_matches.extend(results)
            print(f"  {len(results)} upcoming fixtures with form data")
            for m, mo in results[:2]:
                print(f"  {m.home_team} vs {m.away_team} | {m.kickoff_utc.date()} | {len(mo)} markets")
        except requests.HTTPError as e:
            print(f"  HTTP error: {e}")

    print(f"\nTotal fixtures ready for model: {len(all_matches)}")

