"""
app.py

Streamlit web interface for the Football Acca Advisor.

Run with:  streamlit run app.py
"""

import os
from datetime import datetime, timezone
from itertools import combinations

import streamlit as st

from data_fetcher import FootballDataClient, OddsApiClient, fetch_competition
from football_value_acca import (
    LEAGUE_CONFIGS, MarketPricer, MatchModel,
    build_accas_beam_search, build_candidate,
)
from run import (
    BOOKMAKER, COMPETITIONS, CONSTRAINTS, DAYS_AHEAD, OPT,
    RULES, SEASON, YANKEE_MAX_ODDS,
    confidence_label, format_selection, win_prob_label, yankee_score,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Football Acca Advisor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
  /* Reduce top padding on the main container */
  .block-container { padding-top: 0.75rem !important; padding-bottom: 1rem !important; }
  /* Tighten gaps between markdown elements */
  .stMarkdown p { margin-bottom: 0.15rem !important; }
  /* Thinner horizontal rules */
  hr { margin: 0.4rem 0 !important; border-color: #e0e0e0; }
  /* Shrink expander padding */
  .streamlit-expanderContent { padding: 0.4rem 0.75rem !important; }
  /* Tighten metric cards */
  [data-testid="metric-container"] { padding: 0.3rem 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

CONF_ICON = {
    "Very high": "🟢",
    "High":      "🟡",
    "Good":      "🟠",
    "Moderate":  "⚪",
}

def conf_badge(conf: float) -> str:
    label = confidence_label(conf)
    return f"{CONF_ICON[label]} {label}"


# ---------------------------------------------------------------------------
# Data fetching — cached for 30 minutes so page interactions don't re-fetch
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all(fd_key: str, odds_key: str):
    fd    = FootballDataClient(fd_key)
    odds  = OddsApiClient(odds_key)
    now   = datetime.now(timezone.utc)
    model  = MatchModel(LEAGUE_CONFIGS)
    pricer = MarketPricer(max_goals=10)

    fixture_counts: dict = {}
    all_fixtures:   list = []

    for comp_code, league_id, sport_key in COMPETITIONS:
        try:
            results = fetch_competition(
                comp_code, league_id, fd, odds, sport_key,
                season=SEASON, days_ahead=DAYS_AHEAD,
                bookmaker_filter=BOOKMAKER,
            )
            all_fixtures.extend(results)
            with_odds = sum(1 for _, mo in results if mo)
            fixture_counts[league_id] = (len(results), with_odds)
        except Exception as e:
            fixture_counts[league_id] = (0, 0)

    candidates: list = []
    for match, market_odds in all_fixtures:
        for market, snap in market_odds:
            c = build_candidate(match, market, snap, model, pricer, now, RULES)
            if c:
                candidates.append(c)

    accas: list = []
    if len(candidates) >= CONSTRAINTS.min_legs:
        accas = build_accas_beam_search(candidates, CONSTRAINTS, OPT)

    yankee_pool = sorted(
        [c for c in candidates if c.odds.decimal_odds <= YANKEE_MAX_ODDS],
        key=yankee_score,
        reverse=True,
    )

    return fixture_counts, candidates, accas, yankee_pool, now


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("⚽ Football Acca Advisor")

fd_key   = os.environ.get("FOOTBALL_DATA_KEY", "")
odds_key = os.environ.get("ODDS_API_KEY", "")

if not fd_key or not odds_key:
    st.error("API keys not set. In your terminal run:\n\n"
             "  export FOOTBALL_DATA_KEY=your_key\n"
             "  export ODDS_API_KEY=your_key\n\n"
             "Then restart the app.")
    st.stop()

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.caption(f"Bookmaker: **{BOOKMAKER}**  ·  Next **{DAYS_AHEAD}** days  ·  Season 25/26")
with col_btn:
    if st.button("🔄 Refresh"):
        fetch_all.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
with st.spinner("Fetching fixtures and odds..."):
    fixture_counts, candidates, accas, yankee_pool, fetched_at = fetch_all(fd_key, odds_key)

st.caption(f"Last updated: {fetched_at.strftime('%d %b %Y  %H:%M')} UTC")

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
total_fixtures  = sum(v[0] for v in fixture_counts.values())
total_with_odds = sum(v[1] for v in fixture_counts.values())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Leagues", len(fixture_counts))
m2.metric("Fixtures", total_fixtures)
m3.metric("With odds", total_with_odds)
m4.metric("Value bets found", len(candidates))

with st.expander("League breakdown"):
    for league, (total, with_odds) in fixture_counts.items():
        st.write(f"**{league}** — {total} fixtures, {with_odds} with {BOOKMAKER} odds")

# ---------------------------------------------------------------------------
# Value bets
# ---------------------------------------------------------------------------
st.subheader("📋 Value bets this week")

if not candidates:
    st.info("No value bets found this week. Try again closer to the weekend when more fixtures have odds.")
else:
    for c in candidates:
        bookie_implied = 1 / c.odds.decimal_odds
        st.markdown(
            f"**{c.home_team} vs {c.away_team}** `{c.league}` &nbsp;·&nbsp; "
            f"{format_selection(c.market)} &nbsp;·&nbsp; "
            f"Odds **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
            f"Our estimate: **{c.model_prob:.0%}** &nbsp;·&nbsp; "
            f"Bookie implies: {bookie_implied:.0%} &nbsp;·&nbsp; "
            f"Edge: **{c.edge:+.0%}** &nbsp;·&nbsp; "
            f"{conf_badge(c.confidence)}"
        )

st.divider()

# ---------------------------------------------------------------------------
# Straight accumulators
# ---------------------------------------------------------------------------
st.subheader(f"🎯 Accumulator suggestions  ({CONSTRAINTS.min_legs}–{CONSTRAINTS.max_legs} legs, all must win)")

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
                st.markdown(
                    f"✦ **{leg.home_team} vs {leg.away_team}** `{leg.league}` &nbsp;·&nbsp; "
                    f"{format_selection(leg.market)} "
                    f"@ **{leg.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
                    f"Our estimate: {leg.model_prob:.0%} &nbsp;·&nbsp; "
                    f"Edge: {leg.edge:+.0%}"
                )

# ---------------------------------------------------------------------------
# Yankee / Super Yankee
# ---------------------------------------------------------------------------
st.subheader("🃏 Yankee & Super Yankee suggestions")
st.caption(
    f"You win something as long as 2 or more selections come in. "
    f"Only picks with odds under {YANKEE_MAX_ODDS:.1f} are included here."
)


def show_coverage_bet(label: str, num_bets: int, selections: list) -> None:
    header = f"{label}  —  {num_bets} bets  (£1 per bet = £{num_bets} total stake)"
    with st.expander(header, expanded=True):
        for i, c in enumerate(selections, 1):
            st.markdown(
                f"**{i}. {c.home_team} vs {c.away_team}** &nbsp; `{c.league}`  \n"
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
    st.info(
        f"Not enough short-odds value bets for a Yankee this week "
        f"(need 4, found {len(yankee_pool)})."
    )

if len(yankee_pool) >= 5:
    show_coverage_bet("Super Yankee (Canadian)", 26, yankee_pool[:5])
else:
    st.info(
        f"Not enough short-odds value bets for a Super Yankee this week "
        f"(need 5, found {len(yankee_pool)})."
    )

st.divider()
st.caption(
    "This is a mathematical model. It finds statistical edges but cannot guarantee results. "
    "Bet responsibly."
)
