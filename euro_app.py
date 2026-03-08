"""
euro_app.py

Streamlit interface for Champions League & Europa League knockout fixtures.

Run with:  streamlit run euro_app.py

This is a SEPARATE model from the domestic app — it applies:
  • League strength normalisation (Bundesliga ≠ Ligue 1)
  • Elite club European pedigree bonuses
  • Two-leg aggregate context (enter first leg score to adjust)
"""

import os
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, Optional, Tuple

import streamlit as st

from data_fetcher import FootballDataClient, OddsApiClient, fetch_euro_competition
from football_value_acca import (
    AccaConstraints, CandidateRules, LEAGUE_CONFIGS,
    MarketPricer, OptimiserConfig,
    build_accas_beam_search, build_candidate,
)
from euro_model import (
    ELITE_CLUBS, EuroMatchModel, FirstLegResult, TEAM_DOMESTIC_LEAGUE,
)
from run import (
    BOOKMAKER, SEASON, YANKEE_MAX_ODDS,
    confidence_label, format_selection, win_prob_label, yankee_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EURO_DAYS_AHEAD = 14   # knockout fixtures announced further in advance than domestic

EURO_COMPETITIONS = [
    ("CL", "UCL", "soccer_uefa_champs_league"),
    ("EL", "UEL", "soccer_uefa_europa_league"),
]

EURO_RULES = CandidateRules(
    min_edge=0.03,
    min_confidence=0.58,
    require_positive_ev=True,
)

EURO_CONSTRAINTS = AccaConstraints(
    min_legs=2,
    max_legs=4,
    min_total_decimal_odds=4.0,
    soft_max_total_decimal_odds=40.0,
)

EURO_OPT = OptimiserConfig(beam_width=300, expand_top_m=150)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="European Football Advisor", page_icon="🏆", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 0.75rem !important; padding-bottom: 1rem !important; }
  .stMarkdown p { margin-bottom: 0.15rem !important; }
  hr { margin: 0.4rem 0 !important; border-color: #e0e0e0; }
  .streamlit-expanderContent { padding: 0.4rem 0.75rem !important; }
  [data-testid="metric-container"] { padding: 0.3rem 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

CONF_ICON = {"Very high": "🟢", "High": "🟡", "Good": "🟠", "Moderate": "⚪"}

def conf_badge(conf: float) -> str:
    label = confidence_label(conf)
    return f"{CONF_ICON[label]} {label}"

def pedigree_label(team: str) -> str:
    p = ELITE_CLUBS.get(team)
    if not p:
        return ""
    stars = "⭐" * min(round((p.attack_bonus + p.defence_bonus) / 0.04), 5)
    return f" {stars}"


# ---------------------------------------------------------------------------
# Data fetching (cached 30 min)
# ---------------------------------------------------------------------------
# Domestic competitions whose results feed the European form window.
# These cover the clubs most likely to appear in CL/EL knockout rounds.
_DOMESTIC_FD_CODES = ["PL", "PD", "BL1", "SA", "FL1", "PPL"]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_euro_fixtures(fd_key: str, odds_key: str):
    fd   = FootballDataClient(fd_key)
    odds = OddsApiClient(odds_key)

    # ── Pre-fetch domestic results for form enrichment ─────────────────────
    from data_fetcher import FormCalculator as _FC
    _calc = _FC()
    domestic_parsed: list = []
    for fd_code in _DOMESTIC_FD_CODES:
        try:
            raw = fd.get_finished_matches(fd_code, SEASON)
            domestic_parsed += [r for r in (_calc.parse_result(m) for m in raw) if r]
        except Exception:
            pass

    # ── Fetch European fixtures + odds ─────────────────────────────────────
    fixture_counts: dict = {}
    all_fixtures:   list = []

    for _comp_code, league_id, sport_key in EURO_COMPETITIONS:
        try:
            results = fetch_euro_competition(
                league_id, odds, sport_key,
                season=SEASON,
                fd_client=fd,
                extra_results=domestic_parsed,
                bookmaker_filter=BOOKMAKER,
            )
            all_fixtures.extend(results)
            with_odds = sum(1 for _, mo in results if mo)
            fixture_counts[league_id] = (len(results), with_odds)
        except Exception:
            fixture_counts[league_id] = (0, 0)

    return fixture_counts, all_fixtures


def run_model(
    all_fixtures: list,
    first_leg_scores: Dict[str, FirstLegResult],
) -> Tuple[list, list, list]:
    """Run EuroMatchModel over all fixtures and return candidates, accas, yankee_pool."""
    now    = datetime.now(timezone.utc)
    # EuroMatchModel.lambdas() is overridden — European adjustments applied automatically
    model  = EuroMatchModel(LEAGUE_CONFIGS, TEAM_DOMESTIC_LEAGUE, first_leg_scores)
    pricer = MarketPricer(max_goals=10)

    candidates = []
    for match, market_odds in all_fixtures:
        for market, snap in market_odds:
            c = build_candidate(match, market, snap, model, pricer, now, EURO_RULES)
            if c:
                candidates.append(c)

    accas = []
    if len(candidates) >= EURO_CONSTRAINTS.min_legs:
        accas = build_accas_beam_search(candidates, EURO_CONSTRAINTS, EURO_OPT)

    yankee_pool = sorted(
        [c for c in candidates if c.odds.decimal_odds <= YANKEE_MAX_ODDS],
        key=yankee_score,
        reverse=True,
    )

    return candidates, accas, yankee_pool


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🏆 European Football Advisor")
st.caption("Champions League & Europa League — Knockout rounds")

fd_key   = os.environ.get("FOOTBALL_DATA_KEY", "")
odds_key = os.environ.get("ODDS_API_KEY", "")

if not fd_key or not odds_key:
    st.error("API keys not set. Run: export FOOTBALL_DATA_KEY=... and export ODDS_API_KEY=...")
    st.stop()

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.caption(f"Bookmaker: **{BOOKMAKER}**  ·  Next **{EURO_DAYS_AHEAD}** days  ·  Season 25/26")
with col_btn:
    if st.button("🔄 Refresh"):
        fetch_euro_fixtures.clear()
        st.rerun()

st.info(
    "⚠️  Cross-league comparisons are inherently less precise than domestic predictions. "
    "The model applies league strength and pedigree adjustments, but treat edges here "
    "with more caution than domestic picks."
)

# ---------------------------------------------------------------------------
# Fetch fixtures
# ---------------------------------------------------------------------------
with st.spinner("Fetching European fixtures and odds..."):
    fixture_counts, all_fixtures = fetch_euro_fixtures(fd_key, odds_key)

total_fixtures  = sum(v[0] for v in fixture_counts.values())
total_with_odds = sum(v[1] for v in fixture_counts.values())

m1, m2, m3 = st.columns(3)
m1.metric("Fixtures", total_fixtures)
m2.metric("With odds", total_with_odds)
m3.metric("Competitions", len(fixture_counts))

with st.expander("Competition breakdown"):
    for league, (total, with_odds) in fixture_counts.items():
        label = "Champions League" if league == "UCL" else "Europa League"
        st.write(f"**{label}** — {total} fixtures, {with_odds} with {BOOKMAKER} odds")

st.divider()

# ---------------------------------------------------------------------------
# First leg scores (user input)
# ---------------------------------------------------------------------------
st.subheader("⚽ First leg scores  (optional)")
st.caption(
    "If any of the fixtures below are second legs, enter the first leg score. "
    "Leave blank for first legs or if you don't know."
)

first_leg_scores: Dict[str, FirstLegResult] = {}

if all_fixtures:
    for match, _ in all_fixtures:
        comp_label = "CL" if match.league == "UCL" else "EL"
        label = f"{match.home_team} vs {match.away_team} ({comp_label})"
        home_ped = pedigree_label(match.home_team)
        away_ped = pedigree_label(match.away_team)
        col_label, col_h, col_dash, col_a = st.columns([4, 1, 0.3, 1])
        with col_label:
            st.markdown(
                f"**{match.home_team}**{home_ped} vs **{match.away_team}**{away_ped} `{comp_label}`"
            )
        with col_h:
            h = st.number_input("Home", min_value=0, max_value=20, value=0,
                                key=f"h_{match.match_id}", label_visibility="collapsed")
        with col_dash:
            st.markdown("<div style='padding-top:6px;text-align:center'>–</div>",
                        unsafe_allow_html=True)
        with col_a:
            a = st.number_input("Away", min_value=0, max_value=20, value=0,
                                key=f"a_{match.match_id}", label_visibility="collapsed")

        # Only record if user changed from default (non-zero total or deliberate 0-0)
        if h > 0 or a > 0:
            first_leg_scores[f"{match.home_team} vs {match.away_team}"] = FirstLegResult(
                home_scored=h, away_scored=a
            )

st.divider()

# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
candidates, accas, yankee_pool = run_model(all_fixtures, first_leg_scores)

st.subheader(f"📋 Value bets found: {len(candidates)}")

if not candidates:
    st.info(
        "No value bets found. European odds often appear 3–4 days before midweek fixtures. "
        "Try again Tuesday/Wednesday for that week's games."
    )
else:
    for c in candidates:
        bookie_implied = 1 / c.odds.decimal_odds
        comp_label = "Champions League" if c.league == "UCL" else "Europa League"
        home_ped = pedigree_label(c.home_team)
        away_ped = pedigree_label(c.away_team)
        fl = first_leg_scores.get(f"{c.home_team} vs {c.away_team}")
        fl_note = f"  *(2nd leg — first leg: {fl.home_scored}–{fl.away_scored})*" if fl else ""
        st.markdown(
            f"**{c.home_team}**{home_ped} vs **{c.away_team}**{away_ped} `{comp_label}`{fl_note}  \n"
            f"{format_selection(c.market)} &nbsp;·&nbsp; "
            f"Odds **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
            f"Our estimate: **{c.model_prob:.0%}** &nbsp;·&nbsp; "
            f"Bookie implies: {bookie_implied:.0%} &nbsp;·&nbsp; "
            f"Edge: **{c.edge:+.0%}** &nbsp;·&nbsp; "
            f"{conf_badge(c.confidence)}"
        )

st.divider()

# ---------------------------------------------------------------------------
# High-scoring games (Over/Under 2.5 value bets only)
# ---------------------------------------------------------------------------
st.subheader("⚽ High-scoring game picks  (Over/Under 2.5 goals)")

over_under = [c for c in candidates if c.market.kind == "OVER_UNDER"]
if not over_under:
    st.info(
        "Paddy Power doesn't provide Over/Under odds for CL/EL through the odds feed. "
        "Check their site manually for Over 2.5 prices on any fixtures the model flags."
    )
else:
    for c in over_under:
        bookie_implied = 1 / c.odds.decimal_odds
        comp_label = "Champions League" if c.league == "UCL" else "Europa League"
        st.markdown(
            f"**{c.home_team} vs {c.away_team}** `{comp_label}` &nbsp;·&nbsp; "
            f"{format_selection(c.market)} &nbsp;·&nbsp; "
            f"Odds **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
            f"Our estimate: **{c.model_prob:.0%}** &nbsp;·&nbsp; "
            f"Bookie implies: {bookie_implied:.0%} &nbsp;·&nbsp; "
            f"Edge: **{c.edge:+.0%}** &nbsp;·&nbsp; "
            f"{conf_badge(c.confidence)}"
        )

st.divider()

# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
st.subheader(f"🎯 Accumulator suggestions  ({EURO_CONSTRAINTS.min_legs}–{EURO_CONSTRAINTS.max_legs} legs)")

if not accas:
    st.info("Not enough value bets to build an accumulator this week.")
else:
    for i, acca in enumerate(accas[:5]):
        header = (
            f"Acca #{i+1}  —  **{acca.total_odds:.0f}/1**  "
            f"({win_prob_label(acca.win_prob)})  —  "
            f"Confidence: {confidence_label(acca.avg_conf)}"
        )
        with st.expander(header, expanded=(i == 0)):
            for leg in acca.legs:
                comp_label = "CL" if leg.league == "UCL" else "EL"
                st.markdown(
                    f"✦ **{leg.home_team} vs {leg.away_team}** `{comp_label}` &nbsp;·&nbsp; "
                    f"{format_selection(leg.market)} "
                    f"@ **{leg.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
                    f"Our estimate: {leg.model_prob:.0%} &nbsp;·&nbsp; "
                    f"Edge: {leg.edge:+.0%}"
                )

st.divider()

# ---------------------------------------------------------------------------
# Yankee / doubles
# ---------------------------------------------------------------------------
st.subheader("🃏 Yankee & doubles")
st.caption(f"Only picks with odds under {YANKEE_MAX_ODDS:.1f} included.")


def show_coverage_bet(label: str, num_bets: int, selections: list) -> None:
    with st.expander(f"{label}  —  {num_bets} bets  (£1/bet = £{num_bets} stake)", expanded=True):
        for i, c in enumerate(selections, 1):
            comp_label = "CL" if c.league == "UCL" else "EL"
            st.markdown(
                f"**{i}. {c.home_team} vs {c.away_team}** `{comp_label}`  \n"
                f"{format_selection(c.market)} "
                f"@ **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
                f"Our estimate: {c.model_prob:.0%} &nbsp;·&nbsp; "
                f"Edge: {c.edge:+.0%}"
            )
        odds_list = [c.odds.decimal_odds for c in selections]
        doubles = sorted([a * b for a, b in combinations(odds_list, 2)], reverse=True)
        st.markdown(f"**If 2 win** — best double pays: **{doubles[0] - 1:.0f}/1**")
        if len(odds_list) >= 3:
            trebles = sorted(
                [a * b * c for a, b, c in combinations(odds_list, 3)], reverse=True
            )
            st.markdown(f"**If 3 win** — best treble pays: **{trebles[0] - 1:.0f}/1**")
        full = 1.0
        for o in odds_list:
            full *= o
        n = len(selections)
        st.markdown(f"**If all {n} win** — {n}-fold pays: **{full - 1:.0f}/1**")


if len(yankee_pool) >= 4:
    show_coverage_bet("Yankee", 11, yankee_pool[:4])
else:
    st.info(f"Not enough short-odds picks for a Yankee (need 4, found {len(yankee_pool)}).")

if len(yankee_pool) >= 5:
    show_coverage_bet("Super Yankee (Canadian)", 26, yankee_pool[:5])

st.divider()
st.caption(
    "Separate model from domestic app — applies league strength, pedigree bonuses, and "
    "two-leg context. Still a mathematical model: no guarantees. Bet responsibly."
)
