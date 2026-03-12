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
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, Optional, Tuple

import streamlit as st

from data_fetcher import (
    FootballDataClient, OddsApiClient, fetch_euro_competition,
    ApiFootballClient, fetch_api_football_results,
)
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

EURO_DAYS_AHEAD = 14

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
_DOMESTIC_FD_CODES = ["PL", "PD", "BL1", "SA", "FL1", "PPL"]

# Leagues where real form data is available on the free football-data.org tier.
# Teams from other leagues (e.g. Bundesliga, Serie A, Ligue 1) get neutral/average
# form features — treat their predictions with extra caution.
_COVERED_LEAGUES_BASE     = {"EPL", "CHAMP", "LALIGA", "PRIMEIRA"}
_COVERED_LEAGUES_WITH_AF  = {"EPL", "CHAMP", "LALIGA", "PRIMEIRA", "BUNDESLIGA", "SERIEA", "LIGUE1"}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_euro_fixtures(fd_key: str, odds_key: str, af_key: str = ""):
    fd   = FootballDataClient(fd_key)
    odds = OddsApiClient(odds_key)

    from data_fetcher import FormCalculator as _FC
    _calc = _FC()
    domestic_parsed: list = []
    for fd_code in _DOMESTIC_FD_CODES:
        try:
            raw = fd.get_finished_matches(fd_code, SEASON)
            domestic_parsed += [r for r in (_calc.parse_result(m) for m in raw) if r]
        except Exception:
            pass

    # Extra domestic results from API-Football (Bundesliga, Serie A, Ligue 1)
    if af_key:
        af = ApiFootballClient(af_key)
        domestic_parsed += fetch_api_football_results(af, SEASON)

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
) -> Tuple[list, list, list, list]:
    """Run EuroMatchModel and return candidates, accas, yankee_pool, goal_predictions."""
    now    = datetime.now(timezone.utc)
    model  = EuroMatchModel(LEAGUE_CONFIGS, TEAM_DOMESTIC_LEAGUE, first_leg_scores)
    pricer = MarketPricer(max_goals=10)

    candidates = []
    for match, market_odds in all_fixtures:
        for market, snap in market_odds:
            if snap.decimal_odds <= 1.0:
                continue
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

    # Raw goal predictions — all fixtures, no edge filter
    goal_predictions = []
    for match, market_odds in all_fixtures:
        lam = model.lambdas(match)
        p_over, _ = pricer.p_over_under(lam.lam_home, lam.lam_away, 2.5)
        over_snap = next(
            (snap for mkt, snap in market_odds
             if mkt.kind == "OVER_UNDER" and mkt.selection == "OVER"),
            None,
        )
        comp_label = "Champions League" if match.league == "UCL" else "Europa League"
        goal_predictions.append({
            "home":       match.home_team,
            "away":       match.away_team,
            "league":     comp_label,
            "kickoff":    match.kickoff_utc,
            "lam_home":   lam.lam_home,
            "lam_away":   lam.lam_away,
            "pred_total": lam.lam_home + lam.lam_away,
            "p_over_25":  p_over,
            "over_odds":  over_snap.decimal_odds if over_snap else None,
            "over_book":  over_snap.bookmaker if over_snap else None,
        })
    goal_predictions.sort(key=lambda x: x["pred_total"], reverse=True)

    # Debug breakdowns — one entry per fixture
    debug_rows = []
    for match, _ in all_fixtures:
        d = model.debug_lambdas(match)
        comp_label = "CL" if match.league == "UCL" else "EL"
        debug_rows.append({"match": match, "comp": comp_label, **d})

    return candidates, accas, yankee_pool, goal_predictions, debug_rows


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🏆 European Football Advisor")
st.caption("Champions League & Europa League — Knockout rounds")

fd_key   = os.environ.get("FOOTBALL_DATA_KEY", "")
odds_key = os.environ.get("ODDS_API_KEY", "")
af_key   = os.environ.get("RAPID_API_KEY", "")
_COVERED_LEAGUES = _COVERED_LEAGUES_WITH_AF if af_key else _COVERED_LEAGUES_BASE

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
    fixture_counts, all_fixtures = fetch_euro_fixtures(fd_key, odds_key, af_key)

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

        if h > 0 or a > 0:
            first_leg_scores[f"{match.home_team} vs {match.away_team}"] = FirstLegResult(
                home_scored=h, away_scored=a
            )

st.divider()

# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
candidates, accas, yankee_pool, goal_predictions, debug_rows = run_model(all_fixtures, first_leg_scores)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_value, tab_goals, tab_debug = st.tabs(["📋 Value Bets & Accas", "⚽ High-Scoring Games", "🔍 Model Breakdown"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Value bets, accas, Yankees
# ═══════════════════════════════════════════════════════════════════════════
with tab_value:

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
            # Flag imperfect data
            home_league = TEAM_DOMESTIC_LEAGUE.get(c.home_team, "")
            away_league = TEAM_DOMESTIC_LEAGUE.get(c.away_team, "")
            missing = []
            if home_league and home_league not in _COVERED_LEAGUES:
                missing.append(f"{c.home_team} ({home_league})")
            if away_league and away_league not in _COVERED_LEAGUES:
                missing.append(f"{c.away_team} ({away_league})")
            if missing:
                st.warning(f"⚠️ No form data for {', '.join(missing)} — prediction based on league average only.")
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

    # Accumulators
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

    # Yankee / doubles
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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — High-Scoring Games (raw predictions, no edge filter)
# ═══════════════════════════════════════════════════════════════════════════
with tab_goals:
    st.subheader("⚽ Predicted goal totals — all fixtures")
    st.caption(
        "Every fixture ranked by the model's expected total goals. "
        "No edge filter — you decide what looks interesting. "
        "Over 2.5 odds are best available from any bookmaker."
    )

    if not goal_predictions:
        st.info("No fixtures loaded yet.")
    else:
        for g in goal_predictions:
            kickoff_str = g["kickoff"].strftime("%-d %b %H:%M")
            pred = g["pred_total"]

            if pred >= 3.0:
                indicator = "🔴"
            elif pred >= 2.6:
                indicator = "🟠"
            elif pred >= 2.2:
                indicator = "🟡"
            else:
                indicator = "⚪"

            p_over_str = f"{g['p_over_25']:.0%}"

            if g["over_odds"]:
                odds_str = f"Over 2.5 @ **{g['over_odds']:.2f}** ({g['over_book']})"
            else:
                odds_str = "Over 2.5 odds not available"

            st.markdown(
                f"{indicator} **{g['home']} vs {g['away']}** `{g['league']}` "
                f"&nbsp;·&nbsp; {kickoff_str} "
                f"&nbsp;·&nbsp; Pred goals: **{pred:.2f}** "
                f"&nbsp;·&nbsp; P(over 2.5): **{p_over_str}** "
                f"&nbsp;·&nbsp; {odds_str}"
            )

    st.divider()
    st.caption(
        "🔴 ≥ 3.0 predicted goals &nbsp;·&nbsp; "
        "🟠 2.6–3.0 &nbsp;·&nbsp; "
        "🟡 2.2–2.6 &nbsp;·&nbsp; "
        "⚪ < 2.2"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Breakdown (debug)
# ═══════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.subheader("🔍 How the model builds each prediction")
    st.caption(
        "Expected goals (xG) at each stage of the calculation. "
        "Higher xG = more goals expected. The team with higher final xG is favoured to win."
    )

    if not debug_rows:
        st.info("No fixtures loaded.")
    else:
        for d in debug_rows:
            match = d["match"]
            home_ped = d["home_ped"]
            away_ped = d["away_ped"]
            header = f"**{match.home_team}** vs **{match.away_team}** `{d['comp']}`"
            with st.expander(header):

                # No-data warnings
                home_no_data = d["home_league"] not in _COVERED_LEAGUES
                away_no_data = d["away_league"] not in _COVERED_LEAGUES
                if home_no_data or away_no_data:
                    missing = []
                    if home_no_data:
                        missing.append(f"{match.home_team} ({d['home_league']})")
                    if away_no_data:
                        missing.append(f"{match.away_team} ({d['away_league']})")
                    st.warning(
                        f"⚠️ No real form data for: {', '.join(missing)}. "
                        f"The model is using league average — treat this prediction with caution."
                    )

                # Step 1: Form (base)
                st.markdown("**Step 1 — Recent domestic form**")
                c1, c2 = st.columns(2)
                c1.metric(f"{match.home_team} xG", f"{d['base_lam_home']:.2f}")
                c2.metric(f"{match.away_team} xG", f"{d['base_lam_away']:.2f}")

                # Step 2: League strength
                st.markdown(
                    f"**Step 2 — League strength** "
                    f"({d['home_league']} {d['home_str']:.2f} vs {d['away_league']} {d['away_str']:.2f})"
                )
                h_delta = d['league_lam_home'] - d['base_lam_home']
                a_delta = d['league_lam_away'] - d['base_lam_away']
                c1, c2 = st.columns(2)
                c1.metric(f"{match.home_team} xG", f"{d['league_lam_home']:.2f}",
                          delta=f"{h_delta:+.2f}", delta_color="normal")
                c2.metric(f"{match.away_team} xG", f"{d['league_lam_away']:.2f}",
                          delta=f"{a_delta:+.2f}", delta_color="normal")

                # Step 3: Pedigree
                home_ped_str = f"+{home_ped.attack_bonus:.0%} atk / -{home_ped.defence_bonus:.0%} opp" if home_ped else "no bonus"
                away_ped_str = f"+{away_ped.attack_bonus:.0%} atk / -{away_ped.defence_bonus:.0%} opp" if away_ped else "no bonus"
                st.markdown(
                    f"**Step 3 — Pedigree**  \n"
                    f"{match.home_team}: {home_ped_str}  \n"
                    f"{match.away_team}: {away_ped_str}"
                )
                h_delta2 = d['final_lam_home'] - d['league_lam_home']
                a_delta2 = d['final_lam_away'] - d['league_lam_away']
                c1, c2 = st.columns(2)
                c1.metric(f"{match.home_team} final xG", f"{d['final_lam_home']:.2f}",
                          delta=f"{h_delta2:+.2f}", delta_color="normal")
                c2.metric(f"{match.away_team} final xG", f"{d['final_lam_away']:.2f}",
                          delta=f"{a_delta2:+.2f}", delta_color="normal")

                # Verdict
                if d['final_lam_home'] > d['final_lam_away'] * 1.1:
                    verdict = f"➡️ Model favours **{match.home_team}** (home)"
                elif d['final_lam_away'] > d['final_lam_home'] * 1.1:
                    verdict = f"➡️ Model favours **{match.away_team}** (away)"
                else:
                    verdict = "➡️ Model sees this as roughly even"
                st.info(verdict)
