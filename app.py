"""
app.py

Streamlit web interface for the Football Acca Advisor.

Run with:  streamlit run app.py
"""

import os
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, timezone
from itertools import combinations

import streamlit as st

from data_fetcher import FootballDataClient, OddsApiClient, fetch_competition, FormCalculator
from football_value_acca import (
    LEAGUE_CONFIGS, MarketPricer, MatchModel,
    build_accas_beam_search, build_candidate,
)
from run import (
    BOOKMAKER, COMPETITIONS, CONSTRAINTS, DAYS_AHEAD, OPT,
    RULES, SEASON, YANKEE_MAX_ODDS,
    confidence_label, format_selection, win_prob_label, yankee_score,
)
import calibration

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Football Acca Advisor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 0.75rem !important; padding-bottom: 1rem !important; }
  .stMarkdown p { margin-bottom: 0.15rem !important; }
  hr { margin: 0.4rem 0 !important; border-color: #e0e0e0; }
  .streamlit-expanderContent { padding: 0.4rem 0.75rem !important; }
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
# Data fetching — cached for 30 minutes
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
        except Exception:
            fixture_counts[league_id] = (0, 0)

    # Value bet candidates (edge-filtered)
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

    # Raw goal predictions — all fixtures, no edge filter
    goal_predictions: list = []
    for match, market_odds in all_fixtures:
        lam = model.lambdas(match)
        p_over, _ = pricer.p_over_under(lam.lam_home, lam.lam_away, 2.5)
        over_snap = next(
            (snap for mkt, snap in market_odds
             if mkt.kind == "OVER_UNDER" and mkt.selection == "OVER"),
            None,
        )
        goal_predictions.append({
            "home":       match.home_team,
            "away":       match.away_team,
            "league":     match.league,
            "kickoff":    match.kickoff_utc,
            "lam_home":   lam.lam_home,
            "lam_away":   lam.lam_away,
            "pred_total": lam.lam_home + lam.lam_away,
            "p_over_25":  p_over,
            "over_odds":  over_snap.decimal_odds if over_snap else None,
            "over_book":  over_snap.bookmaker if over_snap else None,
        })
    goal_predictions.sort(key=lambda x: x["pred_total"], reverse=True)

    # Collect all finished results (from cache) for calibration outcome checking
    calc = FormCalculator()
    all_parsed: list = []
    for comp_code, league_id, sport_key in COMPETITIONS:
        try:
            raw = fd.get_finished_matches(comp_code, SEASON)
            all_parsed.extend(r for r in (calc.parse_result(m) for m in raw) if r)
        except Exception:
            pass


    return fixture_counts, candidates, accas, yankee_pool, goal_predictions, all_parsed, now


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
    fixture_counts, candidates, accas, yankee_pool, goal_predictions, all_parsed, fetched_at = fetch_all(fd_key, odds_key)

st.caption(f"Last updated: {fetched_at.strftime('%d %b %Y  %H:%M')} UTC")

# Log this run's predictions and update CLV on any pending bets
_now = datetime.now(timezone.utc)
_new = calibration.log_predictions(candidates, _now)
calibration.update_clv(candidates, _now)
calibration.update_outcomes(all_parsed, _now)

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
# Tabs
# ---------------------------------------------------------------------------
tab_value, tab_goals, tab_odds, tab_cal = st.tabs(["📋 Value Bets & Accas", "⚽ High-Scoring Games", "🔢 Odds Translator", "📊 Calibration"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Value bets, accas, Yankees
# ═══════════════════════════════════════════════════════════════════════════
with tab_value:

    # ── Singles & Doubles (primary recommendation) ───────────────────────────
    st.subheader("⭐ Top picks — Singles & Doubles")
    st.caption(
        "These are the model's highest expected-value bets. "
        "Singles and doubles give you the best risk/reward ratio — "
        "variance stays manageable and the edge compounds cleanly."
    )

    if not candidates:
        st.info("No value bets found this week. Try again closer to the weekend when more fixtures have odds.")
    else:
        top = candidates[:6]
        for c in top:
            bookie_implied = 1 / c.odds.decimal_odds
            st.markdown(
                f"**{c.home_team} vs {c.away_team}** `{c.league}` &nbsp;·&nbsp; "
                f"{format_selection(c.market)} &nbsp;·&nbsp; "
                f"Odds **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
                f"Our estimate: **{c.model_prob:.0%}** &nbsp;·&nbsp; "
                f"Bookie implies: {bookie_implied:.0%} &nbsp;·&nbsp; "
                f"EV: **{c.ev:+.0%}** &nbsp;·&nbsp; "
                f"{conf_badge(c.confidence)}"
            )

        # Best doubles from the top picks
        if len(top) >= 2:
            st.markdown("##### Best doubles")
            from itertools import combinations as _comb
            doubles = sorted(
                [(a, b, a.ev + b.ev, a.odds.decimal_odds * b.odds.decimal_odds)
                 for a, b in _comb(top, 2)
                 if a.home_team != b.home_team and a.away_team != b.away_team],
                key=lambda x: x[2], reverse=True
            )
            for a, b, combined_ev, combined_odds in doubles[:3]:
                st.markdown(
                    f"✦ **{a.home_team} vs {a.away_team}** {format_selection(a.market)} @ {a.odds.decimal_odds:.2f}"
                    f"  +  **{b.home_team} vs {b.away_team}** {format_selection(b.market)} @ {b.odds.decimal_odds:.2f}"
                    f" &nbsp;·&nbsp; Double odds: **{combined_odds:.2f}** &nbsp;·&nbsp; Combined EV: **{combined_ev:+.0%}**"
                )

    st.divider()

    # ── Accumulators (secondary — higher variance) ───────────────────────────
    st.subheader(f"🎯 Accumulators  ({CONSTRAINTS.min_legs}–{CONSTRAINTS.max_legs} legs)")
    st.caption(
        "⚠️ Accumulators compound variance fast. All legs must win. "
        "Treat these as entertainment — size stakes accordingly."
    )
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
                        f"EV: {leg.ev:+.0%}"
                    )

    # Yankee / Super Yankee
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

    # Subs bench — next 3 picks after the Super Yankee slots
    bench = yankee_pool[5:8]
    if bench:
        st.markdown("#### 🪑 Subs bench")
        st.caption("Next best picks by edge × confidence — swap any of the above out for these if you prefer.")
        for c in bench:
            st.markdown(
                f"**{c.home_team} vs {c.away_team}** &nbsp; `{c.league}` &nbsp;·&nbsp; "
                f"{format_selection(c.market)} "
                f"@ **{c.odds.decimal_odds:.2f}** &nbsp;·&nbsp; "
                f"Our estimate: {c.model_prob:.0%} &nbsp;·&nbsp; "
                f"Edge: {c.edge:+.0%} &nbsp;·&nbsp; "
                f"{conf_badge(c.confidence)}"
            )

    st.divider()
    st.caption(
        "This is a mathematical model. It finds statistical edges but cannot guarantee results. "
        "Bet responsibly."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — High-Scoring Games (raw predictions, no edge filter)
# ═══════════════════════════════════════════════════════════════════════════
with tab_goals:
    st.subheader("⚽ Predicted goal totals — all fixtures")
    st.caption(
        "Every fixture ranked by the model's expected total goals (home + away). "
        "No edge filter applied — you decide what looks interesting. "
        "Over 2.5 odds are best available price from any bookmaker."
    )

    if not goal_predictions:
        st.info("No fixtures loaded yet.")
    else:
        for g in goal_predictions:
            kickoff_str = g["kickoff"].strftime("%-d %b %H:%M")
            pred = g["pred_total"]

            # Colour-code by predicted total
            if pred >= 3.0:
                indicator = "🔴"   # very likely high-scoring
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
# TAB 3 — Odds Translator
# ═══════════════════════════════════════════════════════════════════════════
with tab_odds:
    st.subheader("🔢 Odds Translator")
    st.caption(
        "Convert between fractional (6/1), decimal (7.0), and implied probability (14.3%). "
        "Use this to compare what the model says against what your bookie is showing."
    )

    st.markdown("#### Enter any one value — the others will calculate automatically")

    col1, col2, col3 = st.columns(3)

    with col1:
        frac_input = st.text_input("Fractional odds (e.g. 6/4 or 10/11)", placeholder="e.g. 6/4")
    with col2:
        dec_input = st.text_input("Decimal odds (e.g. 2.50)", placeholder="e.g. 2.50")
    with col3:
        prob_input = st.text_input("Implied probability % (e.g. 40)", placeholder="e.g. 40")

    # Parse whichever field was filled in
    decimal = None
    source = None
    error = None

    try:
        if frac_input.strip():
            parts = frac_input.strip().split("/")
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                decimal = (num / den) + 1
                source = "fractional"
            else:
                error = "Fractional odds should be in the form 6/4"
        elif dec_input.strip():
            decimal = float(dec_input.strip())
            source = "decimal"
        elif prob_input.strip():
            prob = float(prob_input.strip().replace("%", ""))
            if 0 < prob < 100:
                decimal = 1 / (prob / 100)
                source = "probability"
            else:
                error = "Probability must be between 1 and 99"
    except ValueError:
        error = "Couldn't parse that — check your input"

    if error:
        st.error(error)
    elif decimal is not None and decimal > 1:
        implied = 1 / decimal
        if decimal >= 2:
            num = decimal - 1
            # simplify fraction roughly
            frac_str = f"{num:.2f}/1" if num != round(num) else f"{int(num)}/1"
        else:
            # odds-on: express as X/Y where Y > X
            den = 1 / (decimal - 1)
            frac_str = f"1/{den:.2f}" if den != round(den) else f"1/{int(den)}"

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Fractional", frac_str)
        r2.metric("Decimal", f"{decimal:.2f}")
        r3.metric("Implied probability", f"{implied:.1%}")

        st.divider()
        st.markdown("#### What does this mean for the model?")
        st.markdown(
            f"The bookie thinks this has a **{implied:.1%}** chance of happening.  \n"
            f"If the model shows **Our estimate** above {implied:.1%} for the same selection, "
            f"there's a **value edge** — the model thinks it's more likely than the bookie does.  \n"
            f"If the model's estimate is *below* {implied:.1%}, the bookie has already priced it tighter than the model — no value."
        )
    elif decimal is not None:
        st.error("Decimal odds must be greater than 1.0")

    st.divider()
    st.markdown("#### Quick reference table")
    ref_data = [
        ("1/4",  "1.25", "80.0%"),
        ("1/3",  "1.33", "75.0%"),
        ("4/9",  "1.44", "69.2%"),
        ("1/2",  "1.50", "66.7%"),
        ("8/13", "1.62", "61.9%"),
        ("4/6",  "1.67", "60.0%"),
        ("8/11", "1.73", "57.9%"),
        ("4/5",  "1.80", "55.6%"),
        ("10/11","1.91", "52.4%"),
        ("Evs",  "2.00", "50.0%"),
        ("6/5",  "2.20", "45.5%"),
        ("5/4",  "2.25", "44.4%"),
        ("6/4",  "2.50", "40.0%"),
        ("7/4",  "2.75", "36.4%"),
        ("2/1",  "3.00", "33.3%"),
        ("9/4",  "3.25", "30.8%"),
        ("5/2",  "3.50", "28.6%"),
        ("3/1",  "4.00", "25.0%"),
        ("4/1",  "5.00", "20.0%"),
        ("5/1",  "6.00", "16.7%"),
        ("6/1",  "7.00", "14.3%"),
        ("10/1", "11.00","9.1%"),
    ]
    st.markdown(
        "| Fractional | Decimal | Implied % |\n"
        "|---|---|---|\n" +
        "\n".join(f"| {f} | {d} | {p} |" for f, d, p in ref_data)
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Calibration
# ═══════════════════════════════════════════════════════════════════════════
with tab_cal:
    st.subheader("📊 Model Calibration")
    st.caption(
        "Tracks this model's predictions against actual results over time. "
        "Builds up automatically each week — the more weeks recorded, the more meaningful the stats."
    )

    summary = calibration.get_summary()

    if summary["resolved"] < 5:
        st.info(
            f"**{summary.get('total', 0)} predictions logged, {summary['resolved']} resolved so far.** "
            "Keep running the model each week — calibration stats become meaningful after ~20 resolved bets."
        )
        if _new:
            st.success(f"{_new} new prediction(s) logged this run.")
    else:
        n = summary["resolved"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bets tracked", summary["total"])
        c2.metric("Resolved", n)
        c3.metric("Win rate", f"{summary['win_rate']:.0%}")
        c4.metric("Brier score", f"{summary['brier']:.3f}", help="Lower is better. 0.25 = random guessing.")

        if summary["clv_mean"] is not None:
            st.metric(
                "Avg Closing Line Value",
                f"{summary['clv_mean']:+.3f}",
                help="Positive = our odds were better than closing odds (model found value before market corrected)."
            )

        st.divider()

        # Calibration buckets
        st.markdown("#### Predicted probability vs actual win rate")
        st.caption("If the model is well-calibrated, each bar should be close to its label.")
        buckets = summary["buckets"]
        if buckets:
            for band, data in sorted(buckets.items()):
                actual = data["wins"] / data["total"] if data["total"] else 0
                st.markdown(
                    f"**{band}** — predicted | actual win rate: **{actual:.0%}** "
                    f"({data['wins']}/{data['total']} bets)"
                )

        st.divider()

        # By league
        st.markdown("#### Win rate by league")
        for lg, data in sorted(summary["by_league"].items()):
            wr = data["wins"] / data["total"] if data["total"] else 0
            st.markdown(f"**{lg}** — {wr:.0%} ({data['wins']}/{data['total']})")

        st.divider()

        # Recent predictions log
        with st.expander("Full prediction log"):
            for e in sorted(summary["entries"], key=lambda x: x["kickoff_utc"], reverse=True)[:50]:
                status = "✅" if e["outcome"] is True else ("❌" if e["outcome"] is False else "⏳")
                clv = f" CLV: {e['bookie_odds'] - e['closing_odds']:+.2f}" if e.get("closing_odds") else ""
                st.markdown(
                    f"{status} **{e['home_team']} vs {e['away_team']}** `{e['league']}` "
                    f"{e['market']} @ {e['bookie_odds']:.2f} | "
                    f"model: {e['model_prob']:.0%} | EV: {e['ev']:+.0%}{clv}"
                )
