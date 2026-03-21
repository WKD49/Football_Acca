"""
Calibration tracker.

Each time the app runs:
  1. log_predictions()  — saves this week's candidates to .cache/prediction_log.json
  2. update_outcomes()  — matches logged predictions against newly finished results
  3. update_clv()       — updates closing odds for bets not yet kicked off

Summary stats (Brier score, accuracy by league, CLV) are read by app.py
and displayed in a Calibration tab.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOG_FILE = Path(".cache/prediction_log.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load() -> List[Dict[str, Any]]:
    if not _LOG_FILE.exists():
        return []
    try:
        return json.loads(_LOG_FILE.read_text())
    except Exception:
        return []


def _save(entries: List[Dict[str, Any]]) -> None:
    _LOG_FILE.parent.mkdir(exist_ok=True)
    _LOG_FILE.write_text(json.dumps(entries, default=str, indent=2))


def _entry_key(entry: Dict[str, Any]) -> str:
    return f"{entry['match_id']}::{entry['market']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_predictions(candidates: list, now_utc: Optional[datetime] = None) -> int:
    """
    Save candidates to the log. Skips any already logged (same match + market).
    Returns the number of new entries added.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    existing = _load()
    existing_keys = {_entry_key(e) for e in existing}

    new_entries = []
    for c in candidates:
        market_str = _market_str(c.market)
        key = f"{c.match_id}::{market_str}"
        if key in existing_keys:
            continue
        new_entries.append({
            "match_id":       c.match_id,
            "league":         c.league,
            "home_team":      c.home_team,
            "away_team":      c.away_team,
            "kickoff_utc":    c.kickoff_utc.isoformat(),
            "market":         market_str,
            "model_prob":     round(c.model_prob, 4),
            "bookie_odds":    round(c.odds.decimal_odds, 3),
            "bookie_implied": round(1 / c.odds.decimal_odds, 4),
            "edge":           round(c.edge, 4),
            "ev":             round(c.ev, 4),
            "confidence":     round(c.confidence, 3),
            "logged_at":      now_utc.isoformat(),
            "closing_odds":   None,   # updated before kickoff on next run
            "outcome":        None,   # True = won, False = lost, None = pending
            "outcome_home_goals": None,
            "outcome_away_goals": None,
        })
        existing_keys.add(key)

    if new_entries:
        _save(existing + new_entries)

    return len(new_entries)


def update_outcomes(parsed_results: list, now_utc: Optional[datetime] = None) -> int:
    """
    Match logged predictions against finished results.
    Returns the number of entries updated.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    entries = _load()
    if not entries:
        return 0

    # Build lookup: (home, away) -> result dict
    results_by_teams: Dict[tuple, dict] = {}
    for r in parsed_results:
        key = (r["home"], r["away"])
        results_by_teams[key] = r

    updated = 0
    for entry in entries:
        if entry["outcome"] is not None:
            continue  # already resolved

        kickoff = datetime.fromisoformat(entry["kickoff_utc"])
        if kickoff > now_utc:
            continue  # not played yet

        result = results_by_teams.get((entry["home_team"], entry["away_team"]))
        if result is None:
            continue  # result not in this run's data

        hg = result["home_goals"]
        ag = result["away_goals"]
        entry["outcome_home_goals"] = hg
        entry["outcome_away_goals"] = ag
        entry["outcome"] = _did_win(entry["market"], hg, ag)
        updated += 1

    if updated:
        _save(entries)

    return updated


def update_clv(candidates: list, now_utc: Optional[datetime] = None) -> int:
    """
    For bets not yet kicked off, update closing_odds with the current odds.
    This approximates CLV — the final snapshot before kickoff.
    Returns the number of entries updated.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    entries = _load()
    if not entries:
        return 0

    # Build lookup from current candidates
    current_odds: Dict[str, float] = {}
    for c in candidates:
        current_odds[f"{c.match_id}::{_market_str(c.market)}"] = c.odds.decimal_odds

    updated = 0
    for entry in entries:
        if entry["outcome"] is not None:
            continue  # already settled
        kickoff = datetime.fromisoformat(entry["kickoff_utc"])
        if kickoff <= now_utc:
            continue  # already kicked off
        key = _entry_key(entry)
        if key in current_odds:
            entry["closing_odds"] = round(current_odds[key], 3)
            updated += 1

    if updated:
        _save(entries)

    return updated


def get_summary() -> Dict[str, Any]:
    """
    Return calibration stats over all resolved predictions.
    """
    entries = _load()
    resolved = [e for e in entries if e["outcome"] is not None]

    if not resolved:
        return {"total": 0, "resolved": 0}

    total    = len(entries)
    wins     = sum(1 for e in resolved if e["outcome"])
    n        = len(resolved)
    win_rate = wins / n

    # Brier score: lower is better; 0.25 = random
    brier = sum((e["model_prob"] - (1 if e["outcome"] else 0)) ** 2 for e in resolved) / n

    # Log loss
    eps = 1e-9
    log_loss = -sum(
        math.log(e["model_prob"] + eps) if e["outcome"] else math.log(1 - e["model_prob"] + eps)
        for e in resolved
    ) / n

    # CLV: bets where we have closing odds
    clv_entries = [e for e in resolved if e.get("closing_odds")]
    clv_mean = None
    if clv_entries:
        # Positive CLV = our odds were better than closing (we had value)
        clv_mean = sum(e["bookie_odds"] - e["closing_odds"] for e in clv_entries) / len(clv_entries)

    # By league
    by_league: Dict[str, Dict[str, int]] = {}
    for e in resolved:
        lg = e["league"]
        if lg not in by_league:
            by_league[lg] = {"wins": 0, "total": 0}
        by_league[lg]["total"] += 1
        if e["outcome"]:
            by_league[lg]["wins"] += 1

    # Calibration buckets (model_prob bands vs actual win rate)
    buckets: Dict[str, Dict[str, int]] = {}
    for e in resolved:
        band = f"{int(e['model_prob'] * 10) * 10}–{int(e['model_prob'] * 10) * 10 + 10}%"
        if band not in buckets:
            buckets[band] = {"wins": 0, "total": 0}
        buckets[band]["total"] += 1
        if e["outcome"]:
            buckets[band]["wins"] += 1

    return {
        "total":      total,
        "resolved":   n,
        "pending":    total - n,
        "wins":       wins,
        "win_rate":   win_rate,
        "brier":      brier,
        "log_loss":   log_loss,
        "clv_mean":   clv_mean,
        "clv_n":      len(clv_entries),
        "by_league":  by_league,
        "buckets":    buckets,
        "entries":    entries,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _market_str(market) -> str:
    if market.kind == "1X2":
        return f"1X2:{market.selection}"
    if market.kind == "OVER_UNDER":
        return f"O/U{market.line}:{market.selection}"
    return str(market)


def _did_win(market_str: str, hg: int, ag: int) -> bool:
    if market_str.startswith("1X2:"):
        sel = market_str.split(":")[1]
        if sel == "HOME":  return hg > ag
        if sel == "AWAY":  return ag > hg
        if sel == "DRAW":  return hg == ag
    if market_str.startswith("O/U"):
        # e.g. "O/U2.5:OVER"
        parts = market_str.replace("O/U", "").split(":")
        line = float(parts[0])
        sel  = parts[1]
        total = hg + ag
        if sel == "OVER":  return total > line
        if sel == "UNDER": return total < line
    return False
