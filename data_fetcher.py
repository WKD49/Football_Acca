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

import json
import os
import time
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

    def __init__(self, long_window: int = 25, short_window: int = 6):
        self.long_window = long_window
        self.short_window = short_window

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

        def avg(matches: List[dict], scored: bool, window: int) -> float:
            subset = matches[:window]
            if not subset:
                return 1.0
            goals = [
                (m["home_goals"] if m["home"] == team else m["away_goals"]) if scored
                else (m["away_goals"] if m["home"] == team else m["home_goals"])
                for m in subset
            ]
            return sum(goals) / len(goals)

        def ratio(team_avg: float, league_avg: float) -> float:
            return team_avg / league_avg if league_avg > 0 else 1.0

        return TeamFeatures(
            home_attack_long=  ratio(avg(home_matches, scored=True,  window=self.long_window),  league_home_avg),
            home_attack_short= ratio(avg(home_matches, scored=True,  window=self.short_window), league_home_avg),
            home_defence_long= ratio(avg(home_matches, scored=False, window=self.long_window),  league_away_avg),
            home_defence_short=ratio(avg(home_matches, scored=False, window=self.short_window), league_away_avg),
            away_attack_long=  ratio(avg(away_matches, scored=True,  window=self.long_window),  league_away_avg),
            away_attack_short= ratio(avg(away_matches, scored=True,  window=self.short_window), league_away_avg),
            away_defence_long= ratio(avg(away_matches, scored=False, window=self.long_window),  league_home_avg),
            away_defence_short=ratio(avg(away_matches, scored=False, window=self.short_window), league_home_avg),
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

    # Step 1: results -> form
    raw_finished = fd_client.get_finished_matches(competition_code, season)
    parsed = [calc.parse_result(m) for m in raw_finished]
    parsed = [p for p in parsed if p is not None]
    league_home_avg, league_away_avg = calc.league_averages(parsed)

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
_EURO_CALC = FormCalculator(long_window=25, short_window=6)

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

    # ── Fixtures + odds from The Odds API ──────────────────────────────────
    odds_events = odds_client.get_odds(odds_sport_key)
    output: List[Tuple[Match, List[Tuple[Market, OddsSnapshot]]]] = []

    for ev in odds_events:
        home = normalise_team_name(ev.get("home_team", ""))
        away = normalise_team_name(ev.get("away_team", ""))

        try:
            kickoff = datetime.fromisoformat(
                ev.get("commence_time", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            continue

        if has_form:
            home_features = _EURO_CALC.compute(home, parsed, league_home_avg, league_away_avg)
            away_features = _EURO_CALC.compute(away, parsed, league_home_avg, league_away_avg)
            data_quality = 0.72
        else:
            home_features = _NEUTRAL_FEATURES
            away_features = _NEUTRAL_FEATURES
            data_quality = 0.60

        match = Match(
            match_id=ev.get("id", f"{league_id}_{home}_{away}"),
            league=league_id,
            kickoff_utc=kickoff,
            home_team=home,
            away_team=away,
            home_features=home_features,
            away_features=away_features,
            home_schedule=_EURO_SCHED,
            away_schedule=_EURO_SCHED,
            league_data_quality=data_quality,
        )

        market_odds: List[Tuple[Market, OddsSnapshot]] = []
        for sel in ("HOME", "DRAW", "AWAY"):
            snap = best_odds_for_selection(ev, "1X2", sel, bookmaker_filter=bookmaker_filter)
            if snap:
                market_odds.append((Market(kind="1X2", selection=sel), snap))
        for sel in ("OVER", "UNDER"):
            snap = best_odds_for_selection(ev, "OVER_UNDER", sel, line=2.5, bookmaker_filter=bookmaker_filter)
            if snap:
                market_odds.append((Market(kind="OVER_UNDER", line=2.5, selection=sel), snap))

        output.append((match, market_odds))

    return output


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

