"""
football_value_acca.py

A self-contained, showable code skeleton for:
- Home/away split form (venue-specific attack/defence multipliers)
- Rest-days + Europe congestion adjustments
- Simple Poisson match model (goals -> market probabilities)
- Value detection (edge + EV)
- Accumulator builder (3–6 legs, total odds >= 21.0) using beam search
- Confidence + fragility penalties
- LLM prompt templates (Spanish-aware) for injury/news extraction

Not production-ready, but coherent end-to-end and easy to hand to a dev.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, factorial
from typing import Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds

def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))

def odds_product(legs: Iterable["BetCandidate"]) -> float:
    prod = 1.0
    for leg in legs:
        prod *= leg.odds.decimal_odds
    return prod

def prob_product(legs: Iterable["BetCandidate"]) -> float:
    prod = 1.0
    for leg in legs:
        prod *= leg.model_prob
    return prod

def ev_for_bet(model_p: float, decimal_odds: float, stake: float = 1.0) -> float:
    """
    EV for a single bet with stake, where decimal odds d pay (d-1)*stake profit on win.
    EV = p*(d-1)*stake - (1-p)*stake
    """
    return model_p * (decimal_odds - 1.0) * stake - (1.0 - model_p) * stake

def ev_for_acca(win_p: float, acca_decimal_odds: float, stake: float = 1.0) -> float:
    return win_p * (acca_decimal_odds - 1.0) * stake - (1.0 - win_p) * stake

def is_stale(odds_ts: datetime, now_utc: datetime, stale_hours: float) -> bool:
    age_hours = (now_utc - odds_ts).total_seconds() / 3600.0
    return age_hours > stale_hours


# ----------------------------
# Squad depth tiers + injury config
# ----------------------------

@dataclass(frozen=True)
class InjuryConfig:
    """
    Multipliers applied to lambdas when key players are absent.

    key_attacker_out_multiplier:
        Applied to the team's own attacking lambda.

    key_defender_out_opp_multiplier:
        Applied to the *opponent's* attacking lambda (weakened defence).

    multi_starters_out_multiplier:
        Applied to the team's lambda when 2+ starters are out.

    Values:
      - attacking multipliers should be <= 1.0
      - defensive multipliers applied to opponent should be >= 1.0
    """
    key_attacker_out_multiplier: float
    key_defender_out_opp_multiplier: float
    multi_starters_out_multiplier: float


INJURY_CONFIGS: Dict[str, InjuryConfig] = {
    "elite": InjuryConfig(
        key_attacker_out_multiplier=0.98,
        key_defender_out_opp_multiplier=1.02,
        multi_starters_out_multiplier=0.99,
    ),
    "mid": InjuryConfig(
        key_attacker_out_multiplier=0.95,
        key_defender_out_opp_multiplier=1.04,
        multi_starters_out_multiplier=0.97,
    ),
    "thin": InjuryConfig(
        key_attacker_out_multiplier=0.92,
        key_defender_out_opp_multiplier=1.07,
        multi_starters_out_multiplier=0.94,
    ),
}

LEAGUE_DEFAULT_DEPTH: Dict[str, str] = {
    "EPL":          "mid",
    "CHAMP":        "mid",
    "L1":           "thin",
    "L2":           "thin",
    "LALIGA":       "mid",
    "SEGUNDA":      "mid",
    "PRIMERA_RFEF": "thin",
    "SEGUNDA_RFEF": "thin",
    # Scandinavia
    "ALLSVENSKAN":  "mid",
    "ELITESERIEN":  "mid",
    "SUPERLIGA_DK": "mid",
    # Portugal
    "PRIMEIRA":     "mid",
    # Belgium
    "PRO_LEAGUE":   "mid",
    # Japan
    "JLEAGUE":      "mid",
    # Australia
    "ALEAGUE":      "thin",
    # Brazil
    "BRASILEIRAO":  "mid",
}

TEAM_DEPTH_OVERRIDES: Dict[str, str] = {
    # England
    "Manchester City":    "elite",
    "Arsenal":            "elite",
    "Liverpool":          "elite",
    "Chelsea":            "elite",
    "Manchester United":  "elite",
    "Tottenham":          "mid",
    # Spain
    "Real Madrid":        "elite",
    "Barcelona":          "elite",
    "Atletico Madrid":    "elite",
    "Athletic Club":      "mid",
    # Portugal (Champions League regulars)
    "Benfica":            "elite",
    "Porto":              "elite",
    "Sporting CP":        "elite",
    # Belgium (Champions League regulars)
    "Club Brugge":        "elite",
    "Anderlecht":         "mid",
    # Scandinavia (Europa League regulars)
    "Malmo FF":           "mid",
    "AIK":                "mid",
    "FC Copenhagen":      "mid",
    "Bodo/Glimt":         "mid",
    # Japan
    "Kawasaki Frontale":  "mid",
    "Vissel Kobe":        "mid",
    # Australia
    "Melbourne City":     "mid",
}

def get_injury_config(team: str, league: str) -> InjuryConfig:
    tier = TEAM_DEPTH_OVERRIDES.get(team) or LEAGUE_DEFAULT_DEPTH.get(league, "mid")
    return INJURY_CONFIGS[tier]


# ----------------------------
# Core data model
# ----------------------------

_MULTIPLIER_MIN = 0.30
_MULTIPLIER_MAX = 2.50

@dataclass(frozen=True)
class TeamFeatures:
    """
    Rolling form split by venue, for BOTH attack and defence.

    Values must be league-normalised multipliers centred around 1.0, where:
      - Attack multiplier > 1.0  => scores more than league average
      - Defence multiplier > 1.0 => concedes more than league average (worse defence)

    Recommended windows:
      - long  ~ 20–30 matches (or season-to-date EWMA)
      - short ~ 5–8 matches

    strict_validation:
      - True: raise on out-of-range values
      - False: clamp values into range (useful for early-season noise)
    """
    # Home venue
    home_attack_long: float
    home_attack_short: float
    home_defence_long: float
    home_defence_short: float

    # Away venue
    away_attack_long: float
    away_attack_short: float
    away_defence_long: float
    away_defence_short: float

    strict_validation: bool = True

    def __post_init__(self) -> None:
        fields = [
            ("home_attack_long",   self.home_attack_long),
            ("home_attack_short",  self.home_attack_short),
            ("home_defence_long",  self.home_defence_long),
            ("home_defence_short", self.home_defence_short),
            ("away_attack_long",   self.away_attack_long),
            ("away_attack_short",  self.away_attack_short),
            ("away_defence_long",  self.away_defence_long),
            ("away_defence_short", self.away_defence_short),
        ]

        bad = [(name, val) for name, val in fields if not (_MULTIPLIER_MIN <= val <= _MULTIPLIER_MAX)]
        if not bad:
            return

        if self.strict_validation:
            msg = (
                f"TeamFeatures multipliers out of expected range "
                f"[{_MULTIPLIER_MIN}, {_MULTIPLIER_MAX}]:\n"
                + "\n".join(f"  {k} = {v}" for k, v in bad)
                + "\nCheck values are league-normalised multipliers, not raw counts/rates."
            )
            raise ValueError(msg)

        # Clamp in non-strict mode (dataclass is frozen; must use object.__setattr__)
        for name, val in fields:
            object.__setattr__(self, name, clamp(val, _MULTIPLIER_MIN, _MULTIPLIER_MAX))


@dataclass(frozen=True)
class ScheduleContext:
    rest_days: int
    played_midweek: bool
    played_europe_midweek: bool
    between_two_europe_legs: bool
    midweek_travel_km: Optional[float] = None  # optional


@dataclass(frozen=True)
class Match:
    match_id: str
    league: str
    kickoff_utc: datetime

    home_team: str
    away_team: str

    home_features: TeamFeatures
    away_features: TeamFeatures

    home_schedule: ScheduleContext
    away_schedule: ScheduleContext

    # Injury/rotation inputs can be crude at MVP stage:
    home_key_attacker_out: bool = False
    home_key_defender_out: bool = False
    home_starters_out_count: int = 0

    away_key_attacker_out: bool = False
    away_key_defender_out: bool = False
    away_starters_out_count: int = 0

    # International window absences (Asian Cup, AFCON, Copa America, World Cup qualifiers, etc.)
    # Count of first-team players called up to national duty.
    # Elite squads absorb 1–2 absences; thin squads feel 3+ keenly.
    home_intl_absences: int = 0
    away_intl_absences: int = 0

    # Data quality flags (esp. lower divisions)
    injury_info_reliable: bool = True
    league_data_quality: float = 0.75  # 0..1


@dataclass(frozen=True)
class Market:
    """
    MVP market support:
      - "OVER_UNDER" with line (e.g. 2.5) and selection "OVER"/"UNDER"
      - "1X2" with selection "HOME"/"DRAW"/"AWAY"
    """
    kind: str
    line: Optional[float] = None
    selection: Optional[str] = None


@dataclass(frozen=True)
class OddsSnapshot:
    decimal_odds: float
    timestamp_utc: datetime
    bookmaker: str = "unknown"


@dataclass(frozen=True)
class BetCandidate:
    match_id: str
    league: str
    kickoff_utc: datetime

    home_team: str
    away_team: str

    market: Market
    odds: OddsSnapshot

    model_prob: float
    confidence: float
    edge: float
    ev: float

    tags: Tuple[str, ...] = ()


# ----------------------------
# League config (placeholders)
# ----------------------------

@dataclass(frozen=True)
class LeagueConfig:
    league: str
    base_goal_rate: float
    home_adv_multiplier: float
    is_lower_tier: bool = False


LEAGUE_CONFIGS: Dict[str, LeagueConfig] = {
    # England
    "EPL":          LeagueConfig("EPL",          base_goal_rate=1.35, home_adv_multiplier=1.08, is_lower_tier=False),
    "CHAMP":        LeagueConfig("CHAMP",        base_goal_rate=1.28, home_adv_multiplier=1.07, is_lower_tier=False),
    "L1":           LeagueConfig("L1",           base_goal_rate=1.30, home_adv_multiplier=1.07, is_lower_tier=True),
    "L2":           LeagueConfig("L2",           base_goal_rate=1.27, home_adv_multiplier=1.07, is_lower_tier=True),

    # Spain
    "LALIGA":       LeagueConfig("LALIGA",       base_goal_rate=1.20, home_adv_multiplier=1.07, is_lower_tier=False),
    "SEGUNDA":      LeagueConfig("SEGUNDA",      base_goal_rate=1.05, home_adv_multiplier=1.06, is_lower_tier=False),
    "PRIMERA_RFEF": LeagueConfig("PRIMERA_RFEF", base_goal_rate=1.05, home_adv_multiplier=1.06, is_lower_tier=True),
    "SEGUNDA_RFEF": LeagueConfig("SEGUNDA_RFEF", base_goal_rate=1.00, home_adv_multiplier=1.06, is_lower_tier=True),

    # Scandinavia (Apr–Nov; larger travel distances inflate home advantage)
    # base_goal_rate: ~2.56 goals/match total => 1.28 per team
    "ALLSVENSKAN":  LeagueConfig("ALLSVENSKAN",  base_goal_rate=1.28, home_adv_multiplier=1.10, is_lower_tier=False),
    "ELITESERIEN":  LeagueConfig("ELITESERIEN",  base_goal_rate=1.32, home_adv_multiplier=1.11, is_lower_tier=False),
    "SUPERLIGA_DK": LeagueConfig("SUPERLIGA_DK", base_goal_rate=1.28, home_adv_multiplier=1.09, is_lower_tier=False),

    # Portugal (~2.5 goals/match)
    "PRIMEIRA":     LeagueConfig("PRIMEIRA",     base_goal_rate=1.25, home_adv_multiplier=1.08, is_lower_tier=False),

    # Belgium (~2.65 goals/match)
    "PRO_LEAGUE":   LeagueConfig("PRO_LEAGUE",   base_goal_rate=1.32, home_adv_multiplier=1.08, is_lower_tier=False),

    # Japan (~2.5 goals/match; Feb–Dec)
    "JLEAGUE":      LeagueConfig("JLEAGUE",      base_goal_rate=1.25, home_adv_multiplier=1.07, is_lower_tier=False),

    # Australia (~2.8 goals/match; documented late-goal inefficiency)
    "ALEAGUE":      LeagueConfig("ALEAGUE",      base_goal_rate=1.40, home_adv_multiplier=1.07, is_lower_tier=False),

    # Brazil (~2.6 goals/match; soft market, on both free APIs)
    "BRASILEIRAO":  LeagueConfig("BRASILEIRAO",  base_goal_rate=1.30, home_adv_multiplier=1.09, is_lower_tier=False),

    # Germany Bundesliga (~2.9 goals/match; high-tempo, open play)
    "BUNDESLIGA":   LeagueConfig("BUNDESLIGA",   base_goal_rate=1.45, home_adv_multiplier=1.07, is_lower_tier=False),

    # Italy Serie A (~2.5 goals/match; defensive, tight margins)
    "SERIEA":       LeagueConfig("SERIEA",       base_goal_rate=1.25, home_adv_multiplier=1.06, is_lower_tier=False),

    # France Ligue 1 (~2.6 goals/match)
    "LIGUE1":       LeagueConfig("LIGUE1",       base_goal_rate=1.30, home_adv_multiplier=1.06, is_lower_tier=False),
}


# ----------------------------
# Match model -> lambdas
# ----------------------------

@dataclass(frozen=True)
class LambdaResult:
    lam_home: float
    lam_away: float
    confidence: float
    risk_flags: Tuple[str, ...]


class MatchModel:
    """
    - Home/away split form -> attack/defence multipliers
    - Rest days -> small lambda adjustment
    - Europe midweek -> small lambda penalty + confidence penalty
    - Injury/rotation -> squad-depth-aware lambda adjustments
    """

    def __init__(self, league_configs: Dict[str, LeagueConfig]):
        self.league_configs = league_configs

    def _blend(self, long_v: float, short_v: float, is_lower_tier: bool) -> float:
        w_long = 0.8 if is_lower_tier else 0.7
        w_short = 1.0 - w_long
        return w_long * long_v + w_short * short_v

    def lambdas(self, match: Match) -> LambdaResult:
        cfg = self.league_configs.get(match.league)
        if not cfg:
            raise ValueError(f"Unknown league config: {match.league}")

        risk_flags: List[str] = []

        # Form (home/away split)
        home_attack = self._blend(match.home_features.home_attack_long,
                                 match.home_features.home_attack_short,
                                 cfg.is_lower_tier)
        home_defence = self._blend(match.home_features.home_defence_long,
                                  match.home_features.home_defence_short,
                                  cfg.is_lower_tier)

        away_attack = self._blend(match.away_features.away_attack_long,
                                 match.away_features.away_attack_short,
                                 cfg.is_lower_tier)
        away_defence = self._blend(match.away_features.away_defence_long,
                                  match.away_features.away_defence_short,
                                  cfg.is_lower_tier)

        # Base lambdas
        lam_home = cfg.base_goal_rate * home_attack * away_defence * cfg.home_adv_multiplier
        lam_away = cfg.base_goal_rate * away_attack * home_defence

        # Confidence baseline
        confidence = 0.75 + 0.05 * clamp(match.league_data_quality - 0.75, -1.0, 1.0)

        if cfg.is_lower_tier:
            confidence -= 0.05
            risk_flags.append("thin_data_league")

        # Injury info reliability penalty (apply ONCE per match)
        if not match.injury_info_reliable:
            confidence -= 0.10
            risk_flags.append("injury_info_unknown")

        # Rest-days adjustment
        rest_diff = match.home_schedule.rest_days - match.away_schedule.rest_days
        rd = clamp(rest_diff, -3, 3)
        lam_home *= (1.0 + 0.02 * rd)
        lam_away *= (1.0 - 0.02 * rd)

        # Short-rest penalty
        def apply_short_rest(team: str, rest_days: int, is_lower: bool, lam: float) -> float:
            if rest_days <= 2:
                risk_flags.append(f"short_rest_{team}")
                return lam * (0.94 if is_lower else 0.96)
            return lam

        lam_home = apply_short_rest("home", match.home_schedule.rest_days, cfg.is_lower_tier, lam_home)
        lam_away = apply_short_rest("away", match.away_schedule.rest_days, cfg.is_lower_tier, lam_away)

        # Europe midweek penalty + confidence effects
        def apply_europe(team: str, sched: ScheduleContext, is_away_in_league: bool, lam: float) -> float:
            nonlocal confidence
            if sched.played_europe_midweek:
                risk_flags.append(f"europe_midweek_{team}")
                lam *= 0.97
                confidence -= 0.05
                if is_away_in_league:
                    lam *= 0.95
                    risk_flags.append(f"europe_away_penalty_{team}")
                if sched.between_two_europe_legs:
                    confidence -= 0.10
                    risk_flags.append(f"between_europe_legs_{team}")
            return lam

        lam_home = apply_europe("home", match.home_schedule, is_away_in_league=False, lam=lam_home)
        lam_away = apply_europe("away", match.away_schedule, is_away_in_league=True, lam=lam_away)

        # International window absences (Asian Cup, AFCON, Copa America, qualifiers, etc.)
        # Penalty scales with number of absentees; elite squads absorb losses better.
        def apply_intl_absences(team_label: str, team_name: str, absences: int, lam: float) -> float:
            nonlocal confidence
            if absences <= 0:
                return lam
            inj_cfg = get_injury_config(team_name, match.league)
            # Tier-based base penalty per absent player
            per_player = {"elite": 0.015, "mid": 0.025, "thin": 0.035}.get(
                TEAM_DEPTH_OVERRIDES.get(team_name) or LEAGUE_DEFAULT_DEPTH.get(match.league, "mid"),
                0.025,
            )
            total_penalty = clamp(absences * per_player, 0.0, 0.15)
            lam *= (1.0 - total_penalty)
            risk_flags.append(f"intl_absences_{team_label}:{absences}")
            if absences >= 3:
                confidence -= 0.05
            if absences >= 5:
                confidence -= 0.05
                risk_flags.append(f"severe_intl_depletion_{team_label}")
            return lam

        lam_home = apply_intl_absences("home", match.home_team, match.home_intl_absences, lam_home)
        lam_away = apply_intl_absences("away", match.away_team, match.away_intl_absences, lam_away)

        # Injury adjustments (squad-depth-aware)
        def apply_injuries(
            team_label: str,
            team_name: str,
            league: str,
            key_att_out: bool,
            key_def_out: bool,
            starters_out_count: int,
            lam_team: float,
            lam_opponent: float,
        ) -> Tuple[float, float]:
            inj_cfg = get_injury_config(team_name, league)

            if key_att_out:
                risk_flags.append(f"key_attacker_out_{team_label}")
                lam_team *= inj_cfg.key_attacker_out_multiplier

            if starters_out_count >= 2:
                risk_flags.append(f"multiple_starters_out_{team_label}")
                lam_team *= inj_cfg.multi_starters_out_multiplier

            if key_def_out:
                risk_flags.append(f"key_defender_out_{team_label}")
                lam_opponent *= inj_cfg.key_defender_out_opp_multiplier
                risk_flags.append(f"key_defender_out_{team_label}")

            return lam_team, lam_opponent

        lam_home, lam_away = apply_injuries(
            "home",
            match.home_team,
            match.league,
            match.home_key_attacker_out,
            match.home_key_defender_out,
            match.home_starters_out_count,
            lam_home,
            lam_away,
        )
        lam_away, lam_home = apply_injuries(
            "away",
            match.away_team,
            match.league,
            match.away_key_attacker_out,
            match.away_key_defender_out,
            match.away_starters_out_count,
            lam_away,
            lam_home,
        )

        # Clamp confidence + lambdas
        confidence = clamp(confidence, 0.40, 0.90)
        lam_home = clamp(lam_home, 0.2, 3.2)
        lam_away = clamp(lam_away, 0.2, 3.2)

        return LambdaResult(
            lam_home=lam_home,
            lam_away=lam_away,
            confidence=confidence,
            risk_flags=tuple(sorted(set(risk_flags))),
        )


# ----------------------------
# Pricing markets using lambdas
# ----------------------------

class MarketPricer:
    """
    - 1X2 from Poisson scoreline (truncated)
    - Over/Under from Poisson total goals
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals

    def p_1x2(self, lam_home: float, lam_away: float) -> Tuple[float, float, float]:
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for hg in range(0, self.max_goals + 1):
            ph = poisson_pmf(hg, lam_home)
            for ag in range(0, self.max_goals + 1):
                pa = poisson_pmf(ag, lam_away)
                p = ph * pa
                if hg > ag:
                    p_home += p
                elif hg == ag:
                    p_draw += p
                else:
                    p_away += p

        s = p_home + p_draw + p_away
        if s > 0:
            p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
        return p_home, p_draw, p_away

    def p_over_under(self, lam_home: float, lam_away: float, line: float) -> Tuple[float, float]:
        """
        For line=2.5:
          OVER => total goals >= 3
          UNDER => total goals <= 2
        """
        lam_total = lam_home + lam_away
        under_max = int(line // 1)  # 2.5 -> 2
        p_under = poisson_cdf(under_max, lam_total)
        p_over = 1.0 - p_under
        return p_over, p_under


# ----------------------------
# Candidate generation (value detection)
# ----------------------------

@dataclass(frozen=True)
class CandidateRules:
    min_edge: float = 0.03
    min_confidence: float = 0.55
    require_positive_ev: bool = True
    odds_stale_hours: float = 12.0


def build_candidate(
    match: Match,
    market: Market,
    odds: OddsSnapshot,
    model: MatchModel,
    pricer: MarketPricer,
    now_utc: datetime,
    rules: CandidateRules,
) -> Optional[BetCandidate]:
    lambdas = model.lambdas(match)

    conf = lambdas.confidence
    tags = list(lambdas.risk_flags)

    if is_stale(odds.timestamp_utc, now_utc, rules.odds_stale_hours):
        conf = clamp(conf - 0.10, 0.40, 0.90)
        tags.append("stale_odds_possible")

    # Model probability for the selection
    if market.kind == "1X2":
        if market.selection not in ("HOME", "DRAW", "AWAY"):
            raise ValueError("1X2 selection must be HOME/DRAW/AWAY")
        pH, pD, pA = pricer.p_1x2(lambdas.lam_home, lambdas.lam_away)
        p = {"HOME": pH, "DRAW": pD, "AWAY": pA}[market.selection]

    elif market.kind == "OVER_UNDER":
        if market.line is None or market.selection not in ("OVER", "UNDER"):
            raise ValueError("OVER_UNDER requires line and selection OVER/UNDER")
        p_over, p_under = pricer.p_over_under(lambdas.lam_home, lambdas.lam_away, market.line)
        p = p_over if market.selection == "OVER" else p_under

    else:
        raise NotImplementedError(f"Market kind not supported in MVP: {market.kind}")

    d = odds.decimal_odds
    q = implied_prob(d)

    edge = p - q
    ev = ev_for_bet(p, d, stake=1.0)

    if conf < rules.min_confidence:
        return None
    if edge < rules.min_edge:
        return None
    if rules.require_positive_ev and ev <= 0.0:
        return None

    tags.extend([
        f"match:{match.match_id}",
        f"team:{match.home_team}",
        f"team:{match.away_team}",
        f"market:{market.kind}",
        f"league:{match.league}",
    ])

    return BetCandidate(
        match_id=match.match_id,
        league=match.league,
        kickoff_utc=match.kickoff_utc,
        home_team=match.home_team,
        away_team=match.away_team,
        market=market,
        odds=odds,
        model_prob=p,
        confidence=conf,
        edge=edge,
        ev=ev,
        tags=tuple(tags),
    )


# ----------------------------
# Accumulator building (beam search)
# ----------------------------

@dataclass(frozen=True)
class AccaConstraints:
    min_legs: int = 3
    max_legs: int = 6
    min_total_decimal_odds: float = 21.0  # 20/1
    soft_max_total_decimal_odds: float = 45.0


@dataclass(frozen=True)
class OptimiserConfig:
    beam_width: int = 500
    expand_top_m: int = 250  # prune expansions for scale

    fragility_weight: float = 0.5
    repeat_team_penalty: float = 0.10
    lower_tier_cluster_penalty: float = 0.05
    league_diversity_bonus: float = 0.03
    same_match_hard_ban: bool = True


@dataclass
class AccaResult:
    legs: List[BetCandidate]
    total_odds: float
    win_prob: float
    ev: float
    avg_conf: float
    fragility_penalty: float
    score: float
    notes: List[str] = field(default_factory=list)


def acca_score(
    legs: List[BetCandidate],
    constraints: AccaConstraints,
    opt: OptimiserConfig,
    lower_tier_leagues: Optional[set] = None,
) -> AccaResult:
    lower_tier_leagues = lower_tier_leagues or {"L1", "L2", "PRIMERA_RFEF", "SEGUNDA_RFEF"}

    total_odds = odds_product(legs)
    win_p = prob_product(legs)

    ev = ev_for_acca(win_p, total_odds, stake=1.0)
    avg_conf = sum(l.confidence for l in legs) / len(legs)

    notes: List[str] = []

    match_ids = [l.match_id for l in legs]
    if opt.same_match_hard_ban and len(set(match_ids)) != len(match_ids):
        return AccaResult(
            legs=legs,
            total_odds=total_odds,
            win_prob=win_p,
            ev=ev,
            avg_conf=avg_conf,
            fragility_penalty=999.0,
            score=-999.0,
            notes=["Rejected: same match duplicated"],
        )

    # Repeated team penalty (team appears in multiple legs)
    team_appearances: Dict[str, int] = {}
    for l in legs:
        team_appearances[l.home_team] = team_appearances.get(l.home_team, 0) + 1
        team_appearances[l.away_team] = team_appearances.get(l.away_team, 0) + 1

    repeat_team_count = sum(max(0, c - 1) for c in team_appearances.values())
    same_team_pen = opt.repeat_team_penalty * repeat_team_count
    if repeat_team_count:
        notes.append(f"Repeated team exposure: {repeat_team_count} repeats")

    # Lower-tier cluster penalty
    low_tier_count = sum(1 for l in legs if l.league in lower_tier_leagues)
    lower_cluster_pen = opt.lower_tier_cluster_penalty if low_tier_count >= 3 else 0.0
    if lower_cluster_pen:
        notes.append("Lower-tier clustering penalty applied")

    # League diversity bonus
    leagues = set(l.league for l in legs)
    diversity_bonus = opt.league_diversity_bonus if len(leagues) >= 2 else 0.0

    fragility_penalty = (1.0 - avg_conf) + same_team_pen + lower_cluster_pen - diversity_bonus
    score = ev - opt.fragility_weight * fragility_penalty

    if total_odds > constraints.soft_max_total_decimal_odds:
        notes.append("Soft warning: total odds exceed recommended band")

    return AccaResult(
        legs=legs,
        total_odds=total_odds,
        win_prob=win_p,
        ev=ev,
        avg_conf=avg_conf,
        fragility_penalty=fragility_penalty,
        score=score,
        notes=notes,
    )


def build_accas_beam_search(
    candidates: List[BetCandidate],
    constraints: AccaConstraints,
    opt: OptimiserConfig,
) -> List[AccaResult]:
    if not candidates:
        return []

    def cand_key(c: BetCandidate) -> float:
        return c.ev + 0.2 * c.confidence + 0.2 * c.edge

    cands = sorted(candidates, key=cand_key, reverse=True)

    beam: List[Tuple[int, ...]] = [()]
    finished: List[AccaResult] = []

    for depth in range(1, constraints.max_legs + 1):
        next_beam: List[Tuple[int, ...]] = []

        for combo in beam:
            start_idx = (combo[-1] + 1) if combo else 0

            # Expand only top M candidates remaining to keep scale sane
            end_idx = min(len(cands), start_idx + opt.expand_top_m)

            for i in range(start_idx, end_idx):
                new_combo = combo + (i,)

                if opt.same_match_hard_ban:
                    mids = [cands[j].match_id for j in new_combo]
                    if len(mids) != len(set(mids)):
                        continue

                next_beam.append(new_combo)

        def interim_score(combo: Tuple[int, ...]) -> float:
            legs = [cands[j] for j in combo]
            total_odds = odds_product(legs)
            win_p = prob_product(legs)
            ev = ev_for_acca(win_p, total_odds, stake=1.0)
            avg_conf = sum(l.confidence for l in legs) / len(legs)
            return ev + 0.1 * avg_conf + 0.05 * sum(l.edge for l in legs)

        next_beam = sorted(next_beam, key=interim_score, reverse=True)[: opt.beam_width]
        beam = next_beam

        if depth >= constraints.min_legs:
            for combo in beam:
                legs = [cands[j] for j in combo]
                total_odds = odds_product(legs)
                if total_odds < constraints.min_total_decimal_odds:
                    continue
                res = acca_score(legs, constraints, opt)
                if res.score > -900:
                    finished.append(res)

    finished = sorted(finished, key=lambda r: r.score, reverse=True)
    return finished[:25]


# ----------------------------
# LLM prompt templates (Spanish-aware)
# ----------------------------

LLM_SYSTEM_PROMPT = """\
You extract structured football team news and injury information for an accumulator value-betting app.
Leagues include Spain and England, including lower divisions. Sources may be in Spanish.

Hard rules:
- Do NOT invent facts (no hallucinations). If unclear, mark as "unknown".
- Preserve proper nouns. You may summarise Spanish sources but do not fabricate missing details.
- Be blunt about uncertainty. Use UK English, but include key Spanish labels in parentheses where useful,
  e.g. rest days (descanso), rotation (rotación), confirmed absence (baja confirmada).

You do NOT calculate probabilities or choose bets. The app provides model outputs and selected bets.

Return:
A) short human summary
B) strict JSON following the provided schema, including confidence scores (0..1) and risk_flags.
"""

def llm_user_prompt(
    match: Match,
    snippets: List[Dict[str, str]],
    model_outputs: Dict[str, float],
    odds_snapshot: List[Dict[str, str]],
    selected_acca: List[BetCandidate],
    json_schema: str,
) -> str:
    snippet_text = "\n".join(
        f"- [{s.get('source','unknown')} | {s.get('timestamp_utc','unknown')}] {s.get('text','')}"
        for s in snippets
    )

    acca_text = "\n".join(
        f"{i+1}) {l.home_team} vs {l.away_team} | {l.market.kind} {l.market.line or ''} {l.market.selection or ''} "
        f"| odds={l.odds.decimal_odds:.2f} | model_p={l.model_prob:.3f} | conf={l.confidence:.2f}"
        for i, l in enumerate(selected_acca)
    )

    return f"""\
Match: {match.home_team} vs {match.away_team}
League: {match.league}
Kickoff (UTC): {match.kickoff_utc.isoformat()}

Raw team news snippets (with timestamps + sources):
{snippet_text}

Model outputs (do not recalculate):
{model_outputs}

Market odds snapshot (decimal, do not invent):
{odds_snapshot}

Chosen accumulator legs (already optimised):
{acca_text}

Task:
1) Extract availability updates (injuries/suspensions/rotation hints) for both teams.
2) Flag anything that makes the value signal unreliable (rotation risk, lineup unknown, thin data, stale odds).
3) Write a short risk note.

Return (A) explanation + (B) JSON.
JSON schema:
{json_schema}
"""


# ----------------------------
# Demo wiring (varied fake data)
# ----------------------------

def demo() -> None:
    now = datetime.now(timezone.utc)

    model = MatchModel(LEAGUE_CONFIGS)
    pricer = MarketPricer(max_goals=10)

    rules = CandidateRules(min_edge=0.03, min_confidence=0.55, require_positive_ev=True)
    constraints = AccaConstraints(min_legs=3, max_legs=6, min_total_decimal_odds=21.0)
    opt = OptimiserConfig(beam_width=400, expand_top_m=200)

    tf_arsenal = TeamFeatures(
        home_attack_long=1.15, home_attack_short=1.20,
        home_defence_long=0.88, home_defence_short=0.85,
        away_attack_long=1.05, away_attack_short=1.08,
        away_defence_long=0.95, away_defence_short=0.92,
    )
    tf_brighton = TeamFeatures(
        home_attack_long=1.02, home_attack_short=0.98,
        home_defence_long=1.00, home_defence_short=1.03,
        away_attack_long=0.95, away_attack_short=0.92,
        away_defence_long=1.05, away_defence_short=1.08,
    )
    tf_team_a = TeamFeatures(
        home_attack_long=1.05, home_attack_short=1.10,
        home_defence_long=1.08, home_defence_short=1.12,
        away_attack_long=0.90, away_attack_short=0.88,
        away_defence_long=1.10, away_defence_short=1.15,
    )
    tf_team_b = TeamFeatures(
        home_attack_long=0.95, home_attack_short=0.93,
        home_defence_long=1.02, home_defence_short=1.05,
        away_attack_long=0.88, away_attack_short=0.85,
        away_defence_long=1.08, away_defence_short=1.10,
    )
    tf_zaragoza = TeamFeatures(
        home_attack_long=1.08, home_attack_short=1.05,
        home_defence_long=0.95, home_defence_short=0.98,
        away_attack_long=0.98, away_attack_short=0.95,
        away_defence_long=1.02, away_defence_short=1.00,
    )
    tf_oviedo = TeamFeatures(
        home_attack_long=0.98, home_attack_short=1.00,
        home_defence_long=1.05, home_defence_short=1.02,
        away_attack_long=0.92, away_attack_short=0.90,
        away_defence_long=1.08, away_defence_short=1.05,
    )

    match1 = Match(
        match_id="EPL_001",
        league="EPL",
        kickoff_utc=now,
        home_team="Arsenal",
        away_team="Brighton",
        home_features=tf_arsenal,
        away_features=tf_brighton,
        home_schedule=ScheduleContext(rest_days=6, played_midweek=False, played_europe_midweek=False, between_two_europe_legs=False),
        away_schedule=ScheduleContext(rest_days=3, played_midweek=True, played_europe_midweek=False, between_two_europe_legs=False),
        home_key_attacker_out=True,
        injury_info_reliable=True,
        league_data_quality=0.85,
    )

    match2 = Match(
        match_id="L2_101",
        league="L2",
        kickoff_utc=now,
        home_team="TeamA",
        away_team="TeamB",
        home_features=tf_team_a,
        away_features=tf_team_b,
        home_schedule=ScheduleContext(rest_days=2, played_midweek=True, played_europe_midweek=False, between_two_europe_legs=False),
        away_schedule=ScheduleContext(rest_days=2, played_midweek=True, played_europe_midweek=False, between_two_europe_legs=False),
        home_key_attacker_out=True,
        injury_info_reliable=False,
        league_data_quality=0.60,
    )

    match3 = Match(
        match_id="SEG_055",
        league="SEGUNDA",
        kickoff_utc=now,
        home_team="Zaragoza",
        away_team="Oviedo",
        home_features=tf_zaragoza,
        away_features=tf_oviedo,
        home_schedule=ScheduleContext(rest_days=4, played_midweek=False, played_europe_midweek=False, between_two_europe_legs=False),
        away_schedule=ScheduleContext(rest_days=7, played_midweek=False, played_europe_midweek=False, between_two_europe_legs=False),
        injury_info_reliable=True,
        league_data_quality=0.75,
    )

    markets_and_odds = [
        (match1, Market(kind="OVER_UNDER", line=2.5, selection="OVER"), OddsSnapshot(1.95, now, "book1")),
        (match1, Market(kind="1X2", selection="HOME"), OddsSnapshot(1.80, now, "book1")),
        (match2, Market(kind="OVER_UNDER", line=2.5, selection="UNDER"), OddsSnapshot(2.10, now, "book2")),
        (match3, Market(kind="OVER_UNDER", line=2.5, selection="UNDER"), OddsSnapshot(1.90, now, "book3")),
        (match3, Market(kind="1X2", selection="DRAW"), OddsSnapshot(3.10, now, "book3")),
    ]

    candidates: List[BetCandidate] = []
    for m, market, odds in markets_and_odds:
        c = build_candidate(m, market, odds, model, pricer, now, rules)
        if c:
            candidates.append(c)

    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        print(
            f"- {c.match_id} {c.market.kind} {c.market.selection or ''} {c.market.line or ''} "
            f"odds={c.odds.decimal_odds:.2f} p={c.model_prob:.3f} edge={c.edge:.3f} EV={c.ev:.3f} conf={c.confidence:.2f}"
        )

    accas = build_accas_beam_search(candidates, constraints, opt)
    print("\nTop accas:")
    for i, a in enumerate(accas[:5]):
        print(
            f"\n#{i+1} score={a.score:.4f} EV={a.ev:.4f} win_p={a.win_prob:.5f} "
            f"odds={a.total_odds:.2f} avg_conf={a.avg_conf:.2f}"
        )
        for leg in a.legs:
            print(
                f"  - {leg.league} {leg.home_team} vs {leg.away_team} | "
                f"{leg.market.kind} {leg.market.selection} {leg.market.line or ''} "
                f"| odds={leg.odds.decimal_odds:.2f} p={leg.model_prob:.3f} conf={leg.confidence:.2f}"
            )
        if a.notes:
            print("  notes:", "; ".join(a.notes))

    print("\n--- Validation demo ---")
    try:
        _ = TeamFeatures(
            home_attack_long=15.0,  # raw count, not a normalised multiplier
            home_attack_short=1.0,
            home_defence_long=1.0,
            home_defence_short=1.0,
            away_attack_long=1.0,
            away_attack_short=1.0,
            away_defence_long=1.0,
            away_defence_short=1.0,
            strict_validation=True,
        )
    except ValueError as e:
        print(f"Caught expected validation error:\n{e}")


if __name__ == "__main__":
    demo()
