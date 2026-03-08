"""
euro_model.py

European knockout match model — completely separate from the domestic model.

Extends MatchModel with three layers:
  1. League strength normalisation  — adjusts for gap between EPL, Bundesliga, Ligue 1 etc.
  2. Elite club pedigree bonuses    — Real Madrid raise their game; Atletico protect their lead.
  3. Two-leg context adjustments    — trailing 2-0 changes everything.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from football_value_acca import LambdaResult, MatchModel, clamp


# ---------------------------------------------------------------------------
# 1. League strength index
#    Based on UEFA country coefficients (5-year rolling average), normalised
#    so that EURO_BASELINE = 1.10 represents the average CL participant.
# ---------------------------------------------------------------------------

LEAGUE_STRENGTH: Dict[str, float] = {
    "EPL":        1.20,
    "LALIGA":     1.18,
    "BUNDESLIGA": 1.15,
    "SERIEA":     1.10,
    "LIGUE1":     1.05,
    "PRIMEIRA":   0.95,
    "CHAMP":      0.88,
}

EURO_BASELINE = 1.10  # approximate average strength of a CL qualifier


# ---------------------------------------------------------------------------
# 2. Elite club pedigree table
#    attack_bonus  : multiplier added to the club's own expected goals
#    defence_bonus : reduction applied to the opponent's expected goals
#    Style note: defensive clubs (Atletico) get most of their bonus on defence;
#                attacking clubs (Bayern, Barca) get most on attack.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PedigreeEntry:
    attack_bonus:  float   # e.g. 0.12 => their lam × 1.12
    defence_bonus: float   # e.g. 0.08 => opponent lam × 0.92


ELITE_CLUBS: Dict[str, PedigreeEntry] = {
    # ── Attacking elite ──────────────────────────────────────────────
    "Real Madrid":               PedigreeEntry(attack_bonus=0.16, defence_bonus=0.10),
    "Barcelona":                 PedigreeEntry(attack_bonus=0.13, defence_bonus=0.08),
    "Bayern Munich":             PedigreeEntry(attack_bonus=0.13, defence_bonus=0.08),
    "Paris Saint-Germain":       PedigreeEntry(attack_bonus=0.12, defence_bonus=0.08),
    "PSG":                       PedigreeEntry(attack_bonus=0.12, defence_bonus=0.08),
    "Manchester City":           PedigreeEntry(attack_bonus=0.10, defence_bonus=0.07),
    "Liverpool":                 PedigreeEntry(attack_bonus=0.10, defence_bonus=0.07),
    "Borussia Dortmund":         PedigreeEntry(attack_bonus=0.07, defence_bonus=0.04),

    # ── Balanced elite ───────────────────────────────────────────────
    "Inter Milan":               PedigreeEntry(attack_bonus=0.07, defence_bonus=0.07),
    "Internazionale":            PedigreeEntry(attack_bonus=0.07, defence_bonus=0.07),
    "Chelsea":                   PedigreeEntry(attack_bonus=0.06, defence_bonus=0.06),
    "Arsenal":                   PedigreeEntry(attack_bonus=0.05, defence_bonus=0.05),
    "Juventus":                  PedigreeEntry(attack_bonus=0.05, defence_bonus=0.07),
    "Benfica":                   PedigreeEntry(attack_bonus=0.04, defence_bonus=0.04),
    "Porto":                     PedigreeEntry(attack_bonus=0.04, defence_bonus=0.04),
    "Bayer Leverkusen":          PedigreeEntry(attack_bonus=0.06, defence_bonus=0.05),
    "RB Leipzig":                PedigreeEntry(attack_bonus=0.05, defence_bonus=0.04),

    # ── Defensive elite (bonus weighted heavily to defence) ──────────
    "Atletico Madrid":           PedigreeEntry(attack_bonus=0.02, defence_bonus=0.05),
    "Atletico de Madrid":        PedigreeEntry(attack_bonus=0.02, defence_bonus=0.05),
}


# ---------------------------------------------------------------------------
# 3. Two-leg context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FirstLegResult:
    """
    Score from the first leg, expressed from the perspective of the CURRENT home team.
    i.e. home_scored = goals the current home team scored in their first leg match,
         away_scored = goals the current away team scored in the first leg.
    """
    home_scored: int
    away_scored: int


def _two_leg_factors(first_leg: FirstLegResult) -> Tuple[float, float]:
    """
    Return (lam_home_factor, lam_away_factor).
    Deficit = away aggregate − home aggregate.
    Positive deficit means home team is behind on aggregate.
    """
    deficit = first_leg.away_scored - first_leg.home_scored  # from home team's POV

    if deficit >= 2:
        return 1.15, 0.90   # home trailing badly — must attack
    elif deficit == 1:
        return 1.08, 0.95   # home trailing by one
    elif deficit == 0:
        return 1.02, 1.00   # level — slight home crowd boost
    elif deficit == -1:
        return 0.95, 1.05   # home leading by one — cautious
    else:
        return 0.90, 1.12   # home leading by 2+ — protect the lead


# ---------------------------------------------------------------------------
# EuroMatchModel
# ---------------------------------------------------------------------------

class EuroMatchModel(MatchModel):
    """
    Separate model for European knockout fixtures.

    Overrides lambdas() so build_candidate() works unchanged — no special
    calling convention needed in the app.

    Usage:
        model = EuroMatchModel(LEAGUE_CONFIGS, team_league_map, first_leg_scores)
        # then just use build_candidate(match, market, snap, model, ...) as normal
    """

    def __init__(
        self,
        league_configs: dict,
        team_league_map: Dict[str, str],
        first_leg_scores: Optional[Dict[str, FirstLegResult]] = None,
    ):
        super().__init__(league_configs)
        self.team_league_map  = team_league_map
        self.first_leg_scores = first_leg_scores or {}

    def lambdas(self, match) -> LambdaResult:
        """Override: apply European adjustments on top of base domestic lambdas."""
        home_league = self.team_league_map.get(match.home_team, match.league)
        away_league = self.team_league_map.get(match.away_team, match.league)
        first_leg   = self.first_leg_scores.get(f"{match.home_team} vs {match.away_team}")
        return self._euro_lambdas(match, home_league, away_league, first_leg)

    def _euro_lambdas(
        self,
        match,
        home_league: str,
        away_league: str,
        first_leg: Optional[FirstLegResult] = None,
    ) -> LambdaResult:
        """
        Compute lambdas with all three European adjustments applied on top
        of the standard domestic model output.
        """

        # ── Step 1: base lambdas from the standard Poisson model ────────────
        base = super().lambdas(match)
        lam_home   = base.lam_home
        lam_away   = base.lam_away
        confidence = base.confidence
        risk_flags = list(base.risk_flags)

        # ── Step 2: cross-league strength normalisation ──────────────────────
        home_str = LEAGUE_STRENGTH.get(home_league, EURO_BASELINE)
        away_str = LEAGUE_STRENGTH.get(away_league, EURO_BASELINE)

        if home_league != away_league:
            # Relative adjustment: team from stronger league gets a small boost.
            # Power of 0.25 keeps this conservative (not over-leveraged).
            ratio = home_str / away_str
            lam_home *= ratio ** 0.25
            lam_away *= (1 / ratio) ** 0.25
            risk_flags.append(
                f"cross_league:{home_league}({home_str:.2f})_vs_{away_league}({away_str:.2f})"
            )

        # ── Step 3: pedigree bonuses ─────────────────────────────────────────
        home_ped = ELITE_CLUBS.get(match.home_team)
        away_ped = ELITE_CLUBS.get(match.away_team)

        if home_ped:
            lam_home *= (1.0 + home_ped.attack_bonus)
            lam_away *= (1.0 - home_ped.defence_bonus)
            risk_flags.append(f"pedigree_home:{match.home_team}")

        if away_ped:
            lam_away *= (1.0 + away_ped.attack_bonus)
            lam_home *= (1.0 - away_ped.defence_bonus)
            risk_flags.append(f"pedigree_away:{match.away_team}")

        # ── Step 4: two-leg context ──────────────────────────────────────────
        if first_leg is not None:
            h_factor, a_factor = _two_leg_factors(first_leg)
            lam_home   *= h_factor
            lam_away   *= a_factor
            confidence -= 0.05   # second legs are genuinely harder to model
            agg_home = first_leg.home_scored
            agg_away = first_leg.away_scored
            risk_flags.append(f"second_leg:agg_{agg_home}-{agg_away}")

        # ── Step 5: clamp ────────────────────────────────────────────────────
        lam_home   = clamp(lam_home,   0.2, 3.5)
        lam_away   = clamp(lam_away,   0.2, 3.5)
        confidence = clamp(confidence, 0.4, 0.90)

        return LambdaResult(
            lam_home=lam_home,
            lam_away=lam_away,
            confidence=confidence,
            risk_flags=tuple(sorted(set(risk_flags))),
        )


# ---------------------------------------------------------------------------
# Team → domestic league lookup (update each season as needed)
# ---------------------------------------------------------------------------

TEAM_DOMESTIC_LEAGUE: Dict[str, str] = {
    # England
    "Arsenal":              "EPL",
    "Manchester City":      "EPL",
    "Liverpool":            "EPL",
    "Chelsea":              "EPL",
    "Tottenham Hotspur":    "EPL",
    "Manchester United":    "EPL",
    "Newcastle United":     "EPL",
    "Aston Villa":          "EPL",

    # Spain
    "Real Madrid":          "LALIGA",
    "Barcelona":            "LALIGA",
    "Atletico Madrid":      "LALIGA",
    "Atletico de Madrid":   "LALIGA",
    "Athletic Club":        "LALIGA",
    "Real Sociedad":        "LALIGA",
    "Villarreal":           "LALIGA",
    "Sevilla":              "LALIGA",

    # Germany
    "Bayern Munich":        "BUNDESLIGA",
    "Borussia Dortmund":    "BUNDESLIGA",
    "Bayer Leverkusen":     "BUNDESLIGA",
    "RB Leipzig":           "BUNDESLIGA",
    "Eintracht Frankfurt":  "BUNDESLIGA",

    # Italy
    "Inter Milan":          "SERIEA",
    "Internazionale":       "SERIEA",
    "AC Milan":             "SERIEA",
    "Juventus":             "SERIEA",
    "Napoli":               "SERIEA",
    "AS Roma":              "SERIEA",
    "Atalanta":             "SERIEA",
    "Lazio":                "SERIEA",
    "Fiorentina":           "SERIEA",

    # France
    "Paris Saint-Germain":  "LIGUE1",
    "PSG":                  "LIGUE1",
    "Olympique de Marseille": "LIGUE1",
    "Olympique Lyonnais":   "LIGUE1",
    "Monaco":               "LIGUE1",
    "Lille":                "LIGUE1",

    # Portugal
    "Benfica":              "PRIMEIRA",
    "Porto":                "PRIMEIRA",
    "Sporting CP":          "PRIMEIRA",
    "Sporting Lisbon":      "PRIMEIRA",
    "Braga":                "PRIMEIRA",
}
