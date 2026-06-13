#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  WORLD CUP 2026 PREDICTION ENGINE  -  Scientific Training Pipeline
================================================================================
  Canonical, single-source-of-truth trainer for the WC2026 1X2 predictor.

  Pipeline
  --------
    1. Load the team-centric CSV and reconstruct match-centric records.
    2. Estimate Elo ratings (margin-of-victory + home-field aware).
    3. Fit a Dixon-Coles bivariate Poisson model by penalised maximum
       likelihood (scipy L-BFGS-B) with exponential temporal decay.
    4. Optionally fit a feature-based ML model (multinomial logistic in pure
       NumPy; scikit-learn gradient boosting if it happens to be installed).
    5. Temperature-calibrate every model and blend them with a weight chosen
       on a held-out slice.
    6. Report HONEST, leakage-free temporal-holdout metrics:
       accuracy, multiclass Brier score, log-loss and a calibration table.
    7. Emit model/model.json (camelCase, frontend-ready), predictions.json
       and analysis.txt.

  Design notes
  ------------
    * The dataset stores every match twice (once from each team's point of
      view) using columns: team, date, opponent, goals_scored,
      goals_conceded, result, tournament, venue. We normalise to a single
      directed (home, away) record and de-duplicate.
    * Only NumPy + SciPy are required. Everything degrades gracefully if
      scikit-learn is missing, which keeps the model fully reproducible
      offline with no fragile dependencies.
    * The model.json contract matches the corrected frontend:
          lambda_home = exp(teamStrengthHome[home] + teamStrengthAway[away] + homeAdvantage)
          lambda_away = exp(teamStrengthHome[away] + teamStrengthAway[home])
      where teamStrengthHome = attack and teamStrengthAway = -defence.

  Author : World Cup 2026 Predictor
================================================================================
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# scikit-learn is optional. When present it is used purely as an additional
# ensemble member; when absent the pure-NumPy logistic model is used instead.
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - depends on the environment
    HAS_SKLEARN = False

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the large public dataset (martj42 international results) when present;
# fall back to the legacy curated CSV otherwise.
_DATA_FULL = os.path.join(HERE, "data", "international_results.csv")
_DATA_LEGACY = os.path.join(HERE, "data", "wc2026_recent15.csv")
DATA_PATH = _DATA_FULL if os.path.exists(_DATA_FULL) else _DATA_LEGACY
MODEL_PATH = os.path.join(HERE, "model", "model.json")
PREDICTIONS_PATH = os.path.join(HERE, "predictions.json")
FIXTURE_PATH = os.path.join(HERE, "fixture.json")
STANDINGS_PATH = os.path.join(HERE, "standings.json")
ANALYSIS_PATH = os.path.join(HERE, "analysis.txt")

# Official FIFA World Cup 2026 group stage (final draw, 5 Dec 2025).
# (date, group, home, away, city) using the model's team names.
WC2026_FIXTURE = [
    ("2026-06-11", "A", "Mexico", "South Africa", "Mexico City"),
    ("2026-06-11", "A", "South Korea", "Czech Republic", "Guadalajara"),
    ("2026-06-12", "B", "Canada", "Bosnia and Herzegovina", "Toronto"),
    ("2026-06-12", "D", "United States", "Paraguay", "Los Angeles"),
    ("2026-06-13", "B", "Qatar", "Switzerland", "San Francisco"),
    ("2026-06-13", "C", "Brazil", "Morocco", "New Jersey"),
    ("2026-06-13", "C", "Haiti", "Scotland", "Boston"),
    ("2026-06-13", "D", "Australia", "Turkey", "Vancouver"),
    ("2026-06-14", "E", "Germany", "Curaçao", "Houston"),
    ("2026-06-14", "F", "Netherlands", "Japan", "Dallas"),
    ("2026-06-14", "E", "Ivory Coast", "Ecuador", "Philadelphia"),
    ("2026-06-14", "F", "Sweden", "Tunisia", "Guadalupe"),
    ("2026-06-15", "H", "Spain", "Cabo Verde", "Atlanta"),
    ("2026-06-15", "G", "Belgium", "Egypt", "Vancouver"),
    ("2026-06-15", "H", "Saudi Arabia", "Uruguay", "Miami"),
    ("2026-06-15", "G", "Iran", "New Zealand", "Los Angeles"),
    ("2026-06-16", "I", "France", "Senegal", "New Jersey"),
    ("2026-06-16", "I", "Iraq", "Norway", "Boston"),
    ("2026-06-16", "J", "Argentina", "Algeria", "Kansas City"),
    ("2026-06-16", "J", "Austria", "Jordan", "San Francisco"),
    ("2026-06-17", "K", "Portugal", "DR Congo", "Houston"),
    ("2026-06-17", "L", "England", "Croatia", "Dallas"),
    ("2026-06-17", "L", "Ghana", "Panama", "Toronto"),
    ("2026-06-17", "K", "Uzbekistan", "Colombia", "Mexico City"),
    ("2026-06-18", "A", "Czech Republic", "South Africa", "Atlanta"),
    ("2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina", "Los Angeles"),
    ("2026-06-18", "B", "Canada", "Qatar", "Vancouver"),
    ("2026-06-18", "A", "Mexico", "South Korea", "Guadalajara"),
    ("2026-06-19", "C", "Scotland", "Morocco", "Boston"),
    ("2026-06-19", "D", "United States", "Australia", "Seattle"),
    ("2026-06-19", "C", "Brazil", "Haiti", "Philadelphia"),
    ("2026-06-19", "D", "Turkey", "Paraguay", "San Francisco"),
    ("2026-06-20", "F", "Netherlands", "Sweden", "Houston"),
    ("2026-06-20", "E", "Germany", "Ivory Coast", "Toronto"),
    ("2026-06-20", "E", "Ecuador", "Curaçao", "Kansas City"),
    ("2026-06-20", "F", "Tunisia", "Japan", "Guadalupe"),
    ("2026-06-21", "H", "Spain", "Saudi Arabia", "Atlanta"),
    ("2026-06-21", "G", "Belgium", "Iran", "Los Angeles"),
    ("2026-06-21", "H", "Uruguay", "Cabo Verde", "Miami"),
    ("2026-06-21", "G", "New Zealand", "Egypt", "Vancouver"),
    ("2026-06-22", "J", "Argentina", "Austria", "Dallas"),
    ("2026-06-22", "I", "France", "Iraq", "Philadelphia"),
    ("2026-06-22", "I", "Norway", "Senegal", "New Jersey"),
    ("2026-06-22", "J", "Jordan", "Algeria", "San Francisco"),
    ("2026-06-23", "K", "Portugal", "Uzbekistan", "Houston"),
    ("2026-06-23", "L", "England", "Ghana", "Boston"),
    ("2026-06-23", "L", "Panama", "Croatia", "Toronto"),
    ("2026-06-23", "K", "Colombia", "DR Congo", "Guadalajara"),
    ("2026-06-24", "B", "Switzerland", "Canada", "Vancouver"),
    ("2026-06-24", "B", "Bosnia and Herzegovina", "Qatar", "Seattle"),
    ("2026-06-24", "C", "Scotland", "Brazil", "Miami"),
    ("2026-06-24", "C", "Morocco", "Haiti", "Atlanta"),
    ("2026-06-24", "A", "Czech Republic", "Mexico", "Mexico City"),
    ("2026-06-24", "A", "South Africa", "South Korea", "Guadalupe"),
    ("2026-06-25", "E", "Ecuador", "Germany", "New Jersey"),
    ("2026-06-25", "E", "Curaçao", "Ivory Coast", "Philadelphia"),
    ("2026-06-25", "F", "Japan", "Sweden", "Dallas"),
    ("2026-06-25", "F", "Tunisia", "Netherlands", "Kansas City"),
    ("2026-06-25", "D", "Turkey", "United States", "Los Angeles"),
    ("2026-06-25", "D", "Paraguay", "Australia", "San Francisco"),
    ("2026-06-26", "I", "Norway", "France", "Boston"),
    ("2026-06-26", "I", "Senegal", "Iraq", "Toronto"),
    ("2026-06-26", "H", "Cabo Verde", "Saudi Arabia", "Houston"),
    ("2026-06-26", "H", "Uruguay", "Spain", "Guadalajara"),
    ("2026-06-26", "G", "Egypt", "Iran", "Seattle"),
    ("2026-06-26", "G", "New Zealand", "Belgium", "Vancouver"),
    ("2026-06-27", "L", "Panama", "England", "New Jersey"),
    ("2026-06-27", "L", "Croatia", "Ghana", "Philadelphia"),
    ("2026-06-27", "K", "Colombia", "Portugal", "Miami"),
    ("2026-06-27", "K", "DR Congo", "Uzbekistan", "Atlanta"),
    ("2026-06-27", "J", "Algeria", "Austria", "Kansas City"),
    ("2026-06-27", "J", "Jordan", "Argentina", "Dallas"),
]

# Only this matchday (round) is published in fixture.json. Bump to 2/3 and add
# its kickoff times after retraining with the previous round's results.
ACTIVE_MATCHDAY = 1

# Kickoff time in Argentina time (UTC-3) for matchday 1. Converted from each
# host city's local time; "(+1)" means it falls past midnight, next day in ARG.
KICKOFF_ARG = {
    ("Mexico", "South Africa"): "16:00",
    ("South Korea", "Czech Republic"): "23:00",
    ("Canada", "Bosnia and Herzegovina"): "16:00",
    ("United States", "Paraguay"): "22:00",
    ("Qatar", "Switzerland"): "16:00",
    ("Brazil", "Morocco"): "19:00",
    ("Haiti", "Scotland"): "22:00",
    ("Australia", "Turkey"): "01:00 (+1)",
    ("Germany", "Curaçao"): "14:00",
    ("Netherlands", "Japan"): "17:00",
    ("Ivory Coast", "Ecuador"): "20:00",
    ("Sweden", "Tunisia"): "23:00",
    ("Spain", "Cabo Verde"): "13:00",
    ("Belgium", "Egypt"): "16:00",
    ("Saudi Arabia", "Uruguay"): "19:00",
    ("Iran", "New Zealand"): "22:00",
    ("France", "Senegal"): "16:00",
    ("Iraq", "Norway"): "19:00",
    ("Argentina", "Algeria"): "22:00",
    ("Austria", "Jordan"): "01:00 (+1)",
    ("Portugal", "DR Congo"): "14:00",
    ("England", "Croatia"): "17:00",
    ("Ghana", "Panama"): "20:00",
    ("Uzbekistan", "Colombia"): "23:00",
}

# K-factor per competition tier (Elo) and the relative weight a match carries
# in the Dixon-Coles likelihood. Bigger tournaments matter more.
TOURNAMENT_TIERS = [
    ("FIFA World Cup qualification", 40, 1.3),  # check before plain "World Cup"
    ("FIFA World Cup", 60, 1.6),
    ("UEFA European Championship", 50, 1.4),
    ("Copa America", 50, 1.4),
    ("AFC Asian Cup", 50, 1.3),
    ("African Cup of Nations", 50, 1.3),
    ("Gold Cup", 45, 1.2),
    ("UEFA Nations League", 45, 1.2),
    ("CONCACAF Nations League", 45, 1.2),
    ("Arab Cup", 30, 1.0),
    ("Kirin Cup", 25, 0.9),
    ("Friendly", 20, 0.8),
]
DEFAULT_K = 30
DEFAULT_MATCH_WEIGHT = 1.0


def tier_for(tournament: str):
    """Return (K-factor, likelihood-weight) for a tournament name."""
    for key, k, w in TOURNAMENT_TIERS:
        if key.lower() in tournament.lower():
            return k, w
    return DEFAULT_K, DEFAULT_MATCH_WEIGHT


# The 48 qualified squads come straight from the official fixture, so the
# fixture is the single source of truth for who we predict.
CORE_SQUADS = sorted({t for (_d, _g, h, a, _c) in WC2026_FIXTURE for t in (h, a)})

# WC2026 co-hosts. In the group stage each host plays all its matches in its
# own country (Mexico group A, Canada group B, USA group D), so the host gets
# the home-field advantage even when the official fixture lists them as the
# away side (e.g. "Czech Republic vs Mexico" is played in Mexico City).
HOST_NATIONS = {"Mexico", "United States", "Canada"}


def host_side(home, away):
    """'home', 'away' or None — which side plays as a WC2026 host (group stage)."""
    if home in HOST_NATIONS:
        return "home"
    if away in HOST_NATIONS:
        return "away"
    return None

# Normalise dataset spellings to our canonical names (used in the fixture).
NAME_MAP = {
    "Cape Verde": "Cabo Verde",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "United States": "United States",
    "South Korea": "South Korea",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
}


def _norm(name: str) -> str:
    name = (name or "").strip()
    return NAME_MAP.get(name, name)


# ============================================================================
# PART 1 - DATA LOADING (supports both CSV schemas)
# ============================================================================

def load_matches(filepath: str, cutoff: str = "2015-01-01"):
    """
    Load matches and return unique directed (home, away) records.

    Two CSV schemas are supported automatically:
      * match-centric : date, home_team, away_team, home_score, away_score,
                        tournament, city, country, neutral   (martj42 dataset)
      * team-centric  : team, date, opponent, goals_scored, goals_conceded,
                        result, tournament, venue            (legacy curated)

    Unplayed fixtures (NA scores) and matches before `cutoff` are dropped.
    """
    by_key = {}
    skipped = 0
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        match_centric = "home_team" in (reader.fieldnames or [])
        for row in reader:
            try:
                date = (row.get("date") or "").strip()
                if not date or date < cutoff:
                    skipped += 1
                    continue
                datetime.strptime(date, "%Y-%m-%d")
                tournament = (row.get("tournament") or "Friendly").strip()

                if match_centric:
                    home = _norm(row.get("home_team"))
                    away = _norm(row.get("away_team"))
                    hs = (row.get("home_score") or "").strip()
                    as_ = (row.get("away_score") or "").strip()
                    if hs in ("", "NA", "NaN") or as_ in ("", "NA", "NaN"):
                        skipped += 1          # unplayed fixture placeholder
                        continue
                    gh, ga = int(float(hs)), int(float(as_))
                    neutral = (row.get("neutral") or "").strip().upper() in ("TRUE", "1")
                else:
                    team = _norm(row.get("team"))
                    opp = _norm(row.get("opponent"))
                    gs = int(float(row.get("goals_scored", "")))
                    gc = int(float(row.get("goals_conceded", "")))
                    venue = (row.get("venue") or "").strip().lower()
                    neutral = venue == "neutral"
                    if venue == "away":
                        home, away, gh, ga = opp, team, gc, gs
                    else:
                        home, away, gh, ga = team, opp, gs, gc
            except (ValueError, TypeError):
                skipped += 1
                continue

            if not home or not away:
                skipped += 1
                continue

            key = (date, home, away)
            if key in by_key:
                continue
            k_factor, weight = tier_for(tournament)
            by_key[key] = {
                "date": date, "home": home, "away": away,
                "goals_home": gh, "goals_away": ga,
                "tournament": tournament, "neutral": neutral,
                "k_factor": k_factor, "weight": weight,
            }

    matches = sorted(by_key.values(), key=lambda m: m["date"])
    print(f"Loaded {len(matches)} unique matches "
          f"({skipped} rows skipped) spanning {matches[0]['date']} -> {matches[-1]['date']}")
    return matches


# ============================================================================
# PART 2 - ELO RATINGS (margin-of-victory + home advantage)
# ============================================================================

def estimate_elo(matches, hfa=65.0, base=1500.0):
    """
    World-Football-style Elo with a margin-of-victory multiplier and an
    explicit home-field advantage. Ratings are updated chronologically.
    """
    elo = defaultdict(lambda: base)
    for m in matches:
        home, away = m["home"], m["away"]
        gh, ga = m["goals_home"], m["goals_away"]

        adv = 0.0 if m["neutral"] else hfa
        diff = (elo[home] + adv) - elo[away]
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        if gh > ga:
            result = 1.0
        elif gh < ga:
            result = 0.0
        else:
            result = 0.5

        # Margin-of-victory multiplier (Hierarchy / FIFA-style).
        gd = abs(gh - ga)
        if gd <= 1:
            mov = 1.0
        elif gd == 2:
            mov = 1.5
        else:
            mov = (11.0 + gd) / 8.0

        k = m["k_factor"] * mov
        delta = k * (result - exp_home)
        elo[home] += delta
        elo[away] -= delta
    return dict(elo)


# ============================================================================
# PART 3 - DIXON-COLES MODEL (penalised MLE via scipy)
# ============================================================================

class DixonColes:
    """
    Dixon-Coles bivariate-Poisson model.

        log lambda_home = attack[home] - defence[away] + home_adv   (0 if neutral)
        log lambda_away = attack[away] - defence[home]

    Fitted by minimising the weighted negative log-likelihood with the
    low-score dependence correction tau(.) and an exponential time-decay so
    that recent matches dominate. Ridge + sum-to-zero penalties pin down the
    otherwise unidentifiable additive level.
    """

    def __init__(self, xi=0.0025, ridge=0.02):
        self.xi = xi          # temporal decay per day
        self.ridge = ridge    # L2 strength on team parameters
        self.teams = []
        self.idx = {}
        self.attack = {}
        self.defence = {}
        self.home_adv = 0.0
        self.rho = 0.0

    # -- low-score correction -------------------------------------------------
    @staticmethod
    def _tau(gh, ga, lh, la, rho):
        tau = np.ones_like(lh)
        m00 = (gh == 0) & (ga == 0)
        m10 = (gh == 1) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m11] = 1.0 - rho
        return tau

    def fit(self, matches):
        self.teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = np.array([self.idx[m["home"]] for m in matches])
        ai = np.array([self.idx[m["away"]] for m in matches])
        gh = np.array([m["goals_home"] for m in matches], dtype=float)
        ga = np.array([m["goals_away"] for m in matches], dtype=float)
        neutral = np.array([m["neutral"] for m in matches], dtype=float)

        # Exponential temporal decay relative to the most recent match.
        ref = max(datetime.strptime(m["date"], "%Y-%m-%d") for m in matches)
        days = np.array([(ref - datetime.strptime(m["date"], "%Y-%m-%d")).days for m in matches])
        decay = np.exp(-self.xi * days)
        tw = np.array([m["weight"] for m in matches]) * decay  # total per-match weight

        non_neut = 1.0 - neutral
        m00 = (gh == 0) & (ga == 0)
        m10 = (gh == 1) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m11 = (gh == 1) & (ga == 1)

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll_and_grad(p):
            """Weighted negative log-likelihood with an ANALYTIC gradient.

            The att/def/home_adv gradient uses the dominant Poisson term; rho
            uses the exact tau gradient. The tiny tau-on-lambda coupling is left
            to the optimiser's iterations - this keeps each evaluation O(matches)
            instead of O(matches * params), which makes ~11k x ~300-team fits
            run in seconds.
            """
            attack, defence, home_adv, rho = unpack(p)
            log_lh = attack[hi] - defence[ai] + home_adv * non_neut
            log_la = attack[ai] - defence[hi]
            lh = np.exp(np.clip(log_lh, -4, 4))
            la = np.exp(np.clip(log_la, -4, 4))

            tau = self._tau(gh, ga, lh, la, rho)
            tau = np.clip(tau, 1e-6, None)
            log_p = gh * log_lh - lh + ga * log_la - la + np.log(tau)

            obj = -np.sum(tw * log_p)
            obj += self.ridge * (np.sum(attack ** 2) + np.sum(defence ** 2))
            obj += 100.0 * (attack.mean() ** 2 + defence.mean() ** 2)

            # ---- gradient ----
            rh = tw * (gh - lh)          # d Poisson / d log_lh, weighted
            ra = tw * (ga - la)
            g_att = np.zeros(n)
            g_def = np.zeros(n)
            np.add.at(g_att, hi, rh)
            np.add.at(g_att, ai, ra)
            np.add.at(g_def, ai, -rh)
            np.add.at(g_def, hi, -ra)
            g_att = -g_att + 2 * self.ridge * attack + 200.0 * attack.mean() / n
            g_def = -g_def + 2 * self.ridge * defence + 200.0 * defence.mean() / n
            g_ha = -np.sum(rh * non_neut)

            # rho gradient via exact d log(tau)/d rho on the four cells.
            dlt = np.zeros_like(lh)
            dlt[m00] = -lh[m00] * la[m00] / tau[m00]
            dlt[m01] = lh[m01] / tau[m01]
            dlt[m10] = la[m10] / tau[m10]
            dlt[m11] = -1.0 / tau[m11]
            g_rho = -np.sum(tw * dlt)

            grad = np.concatenate([g_att, g_def, [g_ha], [g_rho]])
            return obj, grad

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25  # sensible positive home advantage to start
        bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.3, 0.3)]
        res = minimize(nll_and_grad, x0, method="L-BFGS-B", jac=True,
                       bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})

        attack, defence, home_adv, rho = unpack(res.x)
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        self.attack = {t: float(attack[i]) for t, i in self.idx.items()}
        self.defence = {t: float(defence[i]) for t, i in self.idx.items()}
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        return self

    def lambdas(self, home, away, neutral=False):
        a = self.attack
        d = self.defence
        log_lh = a.get(home, 0.0) - d.get(away, 0.0) + (0.0 if neutral else self.home_adv)
        log_la = a.get(away, 0.0) - d.get(home, 0.0)
        return math.exp(log_lh), math.exp(log_la)

    def predict_proba(self, home, away, neutral=False, max_goals=10):
        """Closed-form 1/X/2 probabilities from the score matrix."""
        lh, la = self.lambdas(home, away, neutral)
        gh = np.arange(max_goals)
        ph = poisson.pmf(gh, lh)
        pa = poisson.pmf(gh, la)
        mat = np.outer(ph, pa)
        # Apply the Dixon-Coles correction to the four low-score cells.
        mat[0, 0] *= 1.0 - lh * la * self.rho
        mat[0, 1] *= 1.0 + lh * self.rho
        mat[1, 0] *= 1.0 + la * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        idx = np.arange(max_goals)
        home_win = np.tril(mat, -1).sum()      # gh > ga
        draw = np.trace(mat)
        away_win = np.triu(mat, 1).sum()        # gh < ga
        return np.array([home_win, draw, away_win]), (lh, la)


# ============================================================================
# PART 4 - FEATURE-BASED ML MODEL (causal features, no leakage)
# ============================================================================

def build_features(matches):
    """
    Build a leakage-free feature matrix: every row uses only information
    available strictly *before* that match (rolling Elo + form + h2h).
    Returns X, y (0=home win,1=draw,2=away win) and the live Elo dict.
    """
    elo = defaultdict(lambda: 1500.0)
    last5 = defaultdict(list)          # team -> list of (gf, ga, points)
    h2h = defaultdict(list)            # frozenset({a,b}) -> list of (winner)
    rows, ys = [], []

    def form(team):
        games = last5[team][-5:]
        if not games:
            return 0.5, 1.2, 1.2
        pts = np.mean([g[2] for g in games]) / 3.0
        gf = np.mean([g[0] for g in games])
        ga = np.mean([g[1] for g in games])
        return pts, gf, ga

    for m in matches:
        h, a = m["home"], m["away"]
        gh, ga = m["goals_home"], m["goals_away"]
        adv = 0.0 if m["neutral"] else 65.0

        hf = form(h)
        af = form(a)
        pair = frozenset({h, a})
        past = h2h[pair]
        h2h_home = (np.mean([1.0 if w == h else 0.0 for w in past])
                    if past else 0.5)

        rows.append([
            (elo[h] + adv - elo[a]) / 100.0,
            (elo[h] - 1500.0) / 100.0,
            (elo[a] - 1500.0) / 100.0,
            hf[0], hf[1], hf[2],
            af[0], af[1], af[2],
            hf[1] - af[2],          # home attack vs away defence
            af[1] - hf[2],          # away attack vs home defence
            h2h_home,
            m["weight"],
            0.0 if m["neutral"] else 1.0,
        ])
        if gh > ga:
            ys.append(0)
        elif gh == ga:
            ys.append(1)
        else:
            ys.append(2)

        # --- update rolling state AFTER recording the features ---
        diff = (elo[h] + adv) - elo[a]
        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        res = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        gd = abs(gh - ga)
        mov = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        delta = m["k_factor"] * mov * (res - exp_h)
        elo[h] += delta
        elo[a] -= delta
        hp = 3 if gh > ga else (1 if gh == ga else 0)
        ap = 3 if ga > gh else (1 if gh == ga else 0)
        last5[h].append((gh, ga, hp))
        last5[a].append((ga, gh, ap))
        h2h[pair].append(h if gh > ga else (a if ga > gh else "draw"))

    return np.array(rows), np.array(ys), dict(elo)


class SoftmaxRegression:
    """Plain multinomial logistic regression (3 classes) in NumPy + ridge."""

    def __init__(self, l2=1.0, lr=0.3, epochs=4000):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.mu = self.sd = self.W = self.b = None

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        k = 3
        Y = np.eye(k)[y]
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(Xs @ self.W + self.b)
            gW = Xs.T @ (P - Y) / n + self.l2 * self.W / n
            gb = (P - Y).mean(axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sd
        return self._softmax(Xs @ self.W + self.b)


# ============================================================================
# PART 5 - CALIBRATION & METRICS
# ============================================================================

def fit_temperature(probs, y):
    """Temperature scaling: find T>0 minimising log-loss of probs**(1/T)."""
    eps = 1e-12
    logp = np.log(np.clip(probs, eps, 1))

    def loss(T):
        scaled = np.exp(logp / T)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(scaled[np.arange(len(y)), y], eps, 1)))

    res = minimize_scalar(loss, bounds=(0.4, 4.0), method="bounded")
    return float(res.x)


def apply_temperature(probs, T):
    eps = 1e-12
    scaled = np.exp(np.log(np.clip(probs, eps, 1)) / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


def metrics(probs, y):
    eps = 1e-12
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y))
    Y = np.eye(3)[y]
    brier = float(np.mean(np.sum((probs - Y) ** 2, axis=1)))
    logloss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1))))
    return acc, brier, logloss


def calibration_table(probs, y, bins=5):
    """Reliability of the top predicted class, bucketed by confidence."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.33, 1.0, bins + 1)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.sum() == 0:
            continue
        table.append((lo, hi, int(mask.sum()),
                      float(conf[mask].mean()), float(correct[mask].mean())))
    return table


# ============================================================================
# PART 6 - ORCHESTRATION
# ============================================================================

def temporal_split(matches, test_frac=0.18):
    n = len(matches)
    cut = int(n * (1 - test_frac))
    return matches[:cut], matches[cut:]


def tune_dc(train, calib, xis=(0.0008, 0.0015, 0.0025, 0.0040),
            ridges=(0.01, 0.03, 0.08)):
    """Grid-search (xi, ridge) for Dixon-Coles by calibration-set log-loss."""
    yc = y_of(calib)
    best, best_ll = (0.0025, 0.03), float("inf")
    for xi in xis:
        for ridge in ridges:
            dc = DixonColes(xi=xi, ridge=ridge).fit(train)
            probs = np.array([dc.predict_proba(m["home"], m["away"], m["neutral"])[0]
                              for m in calib])
            _, _, ll = metrics(probs, yc)
            if ll < best_ll:
                best_ll, best = ll, (xi, ridge)
    return best


def y_of(matches):
    return np.array([0 if m["goals_home"] > m["goals_away"]
                     else (1 if m["goals_home"] == m["goals_away"] else 2)
                     for m in matches])


def main():
    print("\n" + "=" * 78)
    print(" WORLD CUP 2026 PREDICTION ENGINE  -  Scientific Training Pipeline")
    print("=" * 78)

    print(f"Dataset: {os.path.basename(DATA_PATH)}")
    matches = load_matches(DATA_PATH)
    squads = CORE_SQUADS
    teams_all = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    print(f"Teams: {len(teams_all)} total ({len(squads)} core WC2026 squads)")

    # ---- Honest temporal evaluation -------------------------------------
    # train -> calib -> test, all chronological (no future leaks into past).
    train_full, test = temporal_split(matches, test_frac=0.18)
    train, calib = temporal_split(train_full, test_frac=0.15)
    print(f"Split: train={len(train)}  calib={len(calib)}  test={len(test)}")

    xi, ridge = tune_dc(train, calib)
    print(f"Selected Dixon-Coles xi={xi}  ridge={ridge}")

    # Dixon-Coles on train, calibrate on calib, evaluate on test.
    dc = DixonColes(xi=xi, ridge=ridge).fit(train)
    dc_calib = np.array([dc.predict_proba(m["home"], m["away"], m["neutral"])[0] for m in calib])
    T_dc = fit_temperature(dc_calib, y_of(calib))
    dc_test_raw = np.array([dc.predict_proba(m["home"], m["away"], m["neutral"])[0] for m in test])
    dc_test = apply_temperature(dc_test_raw, T_dc)

    # ML model on the same train slice with causal features.
    Xall, yall, _ = build_features(matches)
    n_tr, n_ca = len(train), len(train) + len(calib)
    Xtr, ytr = Xall[:n_tr], yall[:n_tr]
    Xca = Xall[n_tr:n_ca]
    Xte = Xall[n_ca:]

    if HAS_SKLEARN:
        ml = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                             max_iter=300, l2_regularization=1.0)
        ml.fit(Xtr, ytr)
        ml_predict = lambda X: ml.predict_proba(X)
        ml_name = "HistGradientBoosting (sklearn)"
    else:
        ml = SoftmaxRegression(l2=2.0).fit(Xtr, ytr)
        ml_predict = ml.predict_proba
        ml_name = "Multinomial logistic (NumPy)"

    ml_calib = ml_predict(Xca)
    T_ml = fit_temperature(ml_calib, y_of(calib))
    ml_test = apply_temperature(ml_predict(Xte), T_ml)

    # Blend weight chosen on the calibration slice.
    yca = y_of(calib)
    dc_ca_cal = apply_temperature(dc_calib, T_dc)
    ml_ca_cal = apply_temperature(ml_calib, T_ml)
    best_w, best_ll = 1.0, float("inf")
    for w in np.linspace(0, 1, 21):
        blend = w * dc_ca_cal + (1 - w) * ml_ca_cal
        _, _, ll = metrics(blend, yca)
        if ll < best_ll:
            best_ll, best_w = ll, float(w)
    ens_test = best_w * dc_test + (1 - best_w) * ml_test
    yte = y_of(test)

    print("\n--- Honest temporal-holdout metrics (test set) ---")
    print(f"{'Model':<26}{'Acc':>8}{'Brier':>9}{'LogLoss':>10}")
    for name, probs in [("Dixon-Coles", dc_test),
                        (ml_name, ml_test),
                        (f"Ensemble (w_dc={best_w:.2f})", ens_test)]:
        a, b, l = metrics(probs, yte)
        print(f"{name:<26}{a*100:>7.1f}%{b:>9.3f}{l:>10.3f}")

    # Choose the deployed strategy on the CALIBRATION slice (never the test
    # set) to keep the reported test metrics honest. The blend weight was also
    # chosen on calib, so the ensemble can only tie or beat Dixon-Coles there;
    # if it collapses to w=1.0 we ship plain Dixon-Coles.
    chosen = "dixon_coles" if best_w >= 0.999 else "ensemble"
    deployed = {"dixon_coles": dc_test, "ensemble": ens_test}[chosen]
    acc, brier, logloss = metrics(deployed, yte)
    cal_table = calibration_table(deployed, yte)
    print(f"Deployed strategy: {chosen}")

    # ---- Refit on ALL data for the shipped parameters -------------------
    xi_full, ridge_full = (xi, ridge)  # reuse the hyperparameters tuned above
    dc_final = DixonColes(xi=xi_full, ridge=ridge_full).fit(matches)
    # Recalibrate temperature on the most recent slice (the test tail).
    # Calibrate on the most recent calib+test tail and floor the temperature
    # so the tiny holdout cannot push the shipped probabilities into
    # over-confident extremes.
    recent = calib + test
    yrec = y_of(recent)
    dc_final_calib = np.array([dc_final.predict_proba(m["home"], m["away"], m["neutral"])[0]
                               for m in recent])
    T_final = max(0.75, fit_temperature(dc_final_calib, yrec)) if len(recent) else T_dc

    Xfull, yfull, elo_live = build_features(matches)
    if HAS_SKLEARN:
        ml_final = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                   max_iter=300, l2_regularization=1.0)
        ml_final.fit(Xfull, yfull)
        ml_final_predict = lambda X: ml_final.predict_proba(X)
    else:
        ml_final = SoftmaxRegression(l2=2.0).fit(Xfull, yfull)
        ml_final_predict = ml_final.predict_proba

    elo_final = estimate_elo(matches)

    # ---- Pre-compute calibrated probabilities for every squad pairing ---
    # The frontend reads these directly for instant, exact predictions and
    # falls back to the parametric Dixon-Coles formula for anything else.
    live_state = build_live_state(matches)

    def pairing_probs(home, away, neutral):
        p_dc, (lh, la) = dc_final.predict_proba(home, away, neutral=neutral)
        p_dc = apply_temperature(p_dc[None, :], T_final)[0]
        if chosen == "ensemble":
            feat = _live_features(home, away, elo_live, live_state, neutral=neutral)
            p_ml = apply_temperature(ml_final_predict(feat[None, :]), T_ml)[0]
            p = best_w * p_dc + (1 - best_w) * p_ml
        else:
            p = p_dc
        return p, lh, la

    mp = {}   # rich internal table; model.json stores a compact array form
    for home in squads:
        for away in squads:
            if home == away:
                continue
            p, lh, la = pairing_probs(home, away, neutral=False)        # nominal home
            pn, lhn, lan = pairing_probs(home, away, neutral=True)      # neutral site
            mp[f"{home}|{away}"] = {
                "p1": round(float(p[0]) * 100, 1),
                "pX": round(float(p[1]) * 100, 1),
                "p2": round(float(p[2]) * 100, 1),
                "lambdaHome": round(float(lh), 2),
                "lambdaAway": round(float(la), 2),
                "pick": ["1", "X", "2"][int(np.argmax(p))],
                "confidence": round(float(np.max(p)) * 100, 1),
                "p1n": round(float(pn[0]) * 100, 1),
                "pXn": round(float(pn[1]) * 100, 1),
                "p2n": round(float(pn[2]) * 100, 1),
                "lambdaHomeN": round(float(lhn), 2),
                "lambdaAwayN": round(float(lan), 2),
            }

    # Compact form shipped to the frontend: [p1, pX, p2, p1n, pXn, p2n].
    # Lambdas are recomputed parametrically in JS, so they are not stored.
    match_probs_compact = {
        k: [v["p1"], v["pX"], v["p2"], v["p1n"], v["pXn"], v["p2n"]]
        for k, v in mp.items()
    }

    # ---- Serialise model.json (camelCase, frontend contract) ------------
    # teamStrengthHome = attack ; teamStrengthAway = -defence so that
    # lambda_home = exp(SH[home] + SA[away] + homeAdvantage).
    team_strength_home = {t: round(v, 4) for t, v in dc_final.attack.items()}
    team_strength_away = {t: round(-v, 4) for t, v in dc_final.defence.items()}

    model_data = {
        "version": "4.0_scientific",
        "trainedDate": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "Dixon-Coles bivariate Poisson (penalised MLE, temporal decay) "
                     + ("+ ML ensemble" if chosen == "ensemble" else ""),
        "teamStrengthHome": team_strength_home,
        "teamStrengthAway": team_strength_away,
        "homeAdvantage": round(dc_final.home_adv, 4),
        "xi": xi_full,
        "rho": round(dc_final.rho, 4),
        "calibrationTemperature": round(T_final, 4),
        "eloRatings": {t: round(v, 1) for t, v in elo_final.items()},
        "validation": {
            "strategy": chosen,
            "accuracy": round(acc * 100, 2),
            "brierScore": round(brier, 4),
            "logLoss": round(logloss, 4),
            "samples": int(len(yte)),
            "calibration": [
                {"binLow": round(lo, 2), "binHigh": round(hi, 2), "n": n,
                 "avgConfidence": round(c * 100, 1), "accuracy": round(acc_b * 100, 1)}
                for (lo, hi, n, c, acc_b) in cal_table
            ],
        },
        "ensembleWeightDixonColes": round(best_w, 3),
        "totalTrainingSamples": len(matches),
        "teamsCount": len(teams_all),
        "coreSquads": squads,
        # ["homeWin%","draw%","awayWin%", neutral variants ...]
        "matchProbabilitiesFormat": ["p1", "pX", "p2", "p1n", "pXn", "p2n"],
        "matchProbabilities": match_probs_compact,
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, separators=(",", ":"))

    # ---- fixture.json + standings.json ----------------------------------
    write_fixture(mp, elo_final)
    write_standings(matches)

    # ---- predictions.json (neutral-site, predictions_viewer schema) -----
    # WC2026 group games are at neutral venues, so we surface the neutral
    # probabilities. Each unordered pair is shown once with the higher-Elo
    # team nominally on the left. We keep the most confident calls.
    squad_by_elo = sorted(squads, key=lambda t: -elo_final.get(t, 1500.0))
    predictions = []
    for i, home in enumerate(squad_by_elo):
        for away in squad_by_elo[i + 1:]:
            e = mp[f"{home}|{away}"]
            probs = {"1": e["p1n"], "X": e["pXn"], "2": e["p2n"]}
            pick = max(probs, key=probs.get)
            confidence = probs[pick]
            eh = elo_final.get(home, 1500.0)
            ea = elo_final.get(away, 1500.0)
            predictions.append({
                "home": home, "away": away,
                "homeELO": round(eh, 1), "awayELO": round(ea, 1),
                "expectedGoals": f"{e['lambdaHomeN']:.2f} - {e['lambdaAwayN']:.2f}",
                "probability_1": e["p1n"], "probability_X": e["pXn"],
                "probability_2": e["p2n"],
                "confidence": confidence,
                "prediction": pick,
                "prode": pick if confidence >= 40 else "?",
                "eloAdvantage": round((eh - ea) / 100.0, 3),
            })
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    predictions = predictions[:200]
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # ---- analysis.txt (honest report) -----------------------------------
    write_analysis(matches, teams_all, squads, dc_final, elo_final,
                   chosen, acc, brier, logloss, len(yte), cal_table,
                   best_w, ml_name, xi_full, T_final)

    print(f"\nWrote {MODEL_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}  (top {len(predictions)} neutral pairings)")
    print(f"Wrote {ANALYSIS_PATH}")
    print("\nTop 10 by Elo:")
    for i, (t, r) in enumerate(sorted(elo_final.items(), key=lambda x: -x[1])[:10], 1):
        print(f"  {i:2d}. {t:<22} {r:7.1f}")


def build_live_state(matches):
    """Precompute the latest form/h2h state once (reused for every pairing)."""
    last5 = defaultdict(list)
    h2h = defaultdict(list)
    for m in matches:
        h, a = m["home"], m["away"]
        gh, ga = m["goals_home"], m["goals_away"]
        hp = 3 if gh > ga else (1 if gh == ga else 0)
        ap = 3 if ga > gh else (1 if gh == ga else 0)
        last5[h].append((gh, ga, hp))
        last5[a].append((ga, gh, ap))
        h2h[frozenset({h, a})].append(h if gh > ga else (a if ga > gh else "draw"))
    return last5, h2h


def _live_features(home, away, elo, state, neutral=False):
    """Feature vector for a hypothetical match using precomputed `state`."""
    last5, h2h = state

    def form(team):
        games = last5[team][-5:]
        if not games:
            return 0.5, 1.2, 1.2
        return (np.mean([g[2] for g in games]) / 3.0,
                np.mean([g[0] for g in games]),
                np.mean([g[1] for g in games]))

    hf, af = form(home), form(away)
    past = h2h[frozenset({home, away})]
    h2h_home = np.mean([1.0 if w == home else 0.0 for w in past]) if past else 0.5
    eh, ea = elo.get(home, 1500.0), elo.get(away, 1500.0)
    adv = 0.0 if neutral else 65.0
    return np.array([
        (eh + adv - ea) / 100.0, (eh - 1500.0) / 100.0, (ea - 1500.0) / 100.0,
        hf[0], hf[1], hf[2], af[0], af[1], af[2],
        hf[1] - af[2], af[1] - hf[2], h2h_home, 1.0, 0.0 if neutral else 1.0,
    ])


def write_fixture(mp, elo):
    """Emit fixture.json: the official WC2026 group stage with predictions.

    Group games are neutral-site, so the neutral ensemble probabilities are
    used. Matches keep their official chronological order.
    """
    team_games = defaultdict(int)   # round-robin: a team's k-th game is matchday k
    fixtures = []
    for date, group, home, away, city in WC2026_FIXTURE:
        fwd = mp.get(f"{home}|{away}")
        rev = mp.get(f"{away}|{home}")
        host = host_side(home, away)
        host_team = home if host == "home" else (away if host == "away" else None)

        if host == "home" and fwd:               # home side is the host -> home advantage
            p1, pX, p2 = fwd["p1"], fwd["pX"], fwd["p2"]
            lh, la = fwd["lambdaHome"], fwd["lambdaAway"]
        elif host == "away" and rev:             # away side is the host: reverse pairing, swapped
            p1, pX, p2 = rev["p2"], rev["pX"], rev["p1"]
            lh, la = rev["lambdaAway"], rev["lambdaHome"]
        elif fwd:                                 # neutral site
            p1, pX, p2 = fwd["p1n"], fwd["pXn"], fwd["p2n"]
            lh, la = fwd["lambdaHomeN"], fwd["lambdaAwayN"]
        elif rev:                                 # neutral, stored under the reverse key
            p1, pX, p2 = rev["p2n"], rev["pXn"], rev["p1n"]
            lh, la = rev["lambdaAwayN"], rev["lambdaHomeN"]
        else:
            p1 = pX = p2 = lh = la = None

        md = team_games[home] + 1       # both teams are on the same round
        team_games[home] += 1
        team_games[away] += 1

        if md != ACTIVE_MATCHDAY:      # publish one round at a time
            continue

        pick = conf = None
        if p1 is not None:
            probs = {"1": p1, "X": pX, "2": p2}
            pick = max(probs, key=probs.get)
            conf = probs[pick]

        fixtures.append({
            "date": date, "group": group, "matchday": md,
            "home": home, "away": away, "city": city,
            "kickoffArg": KICKOFF_ARG.get((home, away)),
            "prob_1": p1, "prob_X": pX, "prob_2": p2,
            "expectedGoals": (f"{lh:.2f} - {la:.2f}" if lh is not None else None),
            "prediction": pick, "confidence": conf,
            "hostTeam": host_team,   # WC2026 host playing at home, or null (neutral)
        })

    data = {
        "tournament": "FIFA World Cup 2026",
        "stage": "Group stage",
        "matchday": ACTIVE_MATCHDAY,
        "venuesNeutral": True,        # neutral for everyone except the three hosts
        "hostAdvantage": sorted(HOST_NATIONS),
        "timezone": "America/Argentina (UTC-3)",
        "matches": fixtures,
    }
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {FIXTURE_PATH}  (matchday {ACTIVE_MATCHDAY}: {len(fixtures)} matches)")


def write_standings(matches):
    """
    Emit standings.json: live group tables computed from the WC2026 group
    matches that already have a real result in the dataset. Re-running the
    trainer after new results refreshes this automatically.
    """
    team_group = {}
    for _d, g, h, a, _c in WC2026_FIXTURE:
        team_group[h] = g
        team_group[a] = g

    table = {g: {} for g in sorted(set(team_group.values()))}
    for t, g in team_group.items():
        table[g][t] = {"team": t, "played": 0, "won": 0, "drawn": 0,
                       "lost": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0}

    played = 0
    for m in matches:
        tour = m["tournament"]
        if "FIFA World Cup" not in tour or "qualification" in tour.lower():
            continue
        if m["date"] < "2026-06-11":
            continue
        h, a = m["home"], m["away"]
        if team_group.get(h) != team_group.get(a) or h not in team_group:
            continue
        g = team_group[h]
        gh, ga = m["goals_home"], m["goals_away"]
        for team, gf, gc in ((h, gh, ga), (a, ga, gh)):
            row = table[g][team]
            row["played"] += 1
            row["gf"] += gf
            row["ga"] += gc
            row["gd"] = row["gf"] - row["ga"]
            if gf > gc:
                row["won"] += 1
                row["points"] += 3
            elif gf == gc:
                row["drawn"] += 1
                row["points"] += 1
            else:
                row["lost"] += 1
        played += 1

    standings = {}
    for g, teams in table.items():
        standings[g] = sorted(teams.values(),
                              key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"]))

    with open(STANDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now().isoformat(timespec="seconds"),
                   "matchesPlayed": played, "groups": standings},
                  f, indent=2, ensure_ascii=False)
    print(f"Wrote {STANDINGS_PATH}  ({played} group matches played so far)")


def write_analysis(matches, teams_all, squads, dc, elo, chosen, acc, brier,
                   logloss, n_test, cal_table, w_dc, ml_name, xi, T):
    lines = []
    lines.append("=" * 78)
    lines.append("WORLD CUP 2026 PREDICTION ENGINE - Statistical Analysis Report")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. DATA")
    lines.append("-" * 78)
    lines.append(f"   Unique matches      : {len(matches)}")
    lines.append(f"   Span                : {matches[0]['date']} -> {matches[-1]['date']}")
    lines.append(f"   Teams (all)         : {len(teams_all)}")
    lines.append(f"   Core WC2026 squads  : {len(squads)}")
    lines.append("")
    lines.append("2. MODEL")
    lines.append("-" * 78)
    lines.append("   Primary  : Dixon-Coles bivariate Poisson, penalised MLE (L-BFGS-B)")
    lines.append(f"   Secondary: {ml_name}")
    lines.append(f"   Deployed : {chosen}  (Dixon-Coles blend weight = {w_dc:.2f})")
    lines.append(f"   Temporal decay xi   : {xi}  (half-life ~{math.log(2)/xi:.0f} days)" if xi > 0
                 else "   Temporal decay xi   : 0 (no decay)")
    lines.append(f"   Home advantage      : {dc.home_adv:+.4f} log-goals "
                 f"(x{math.exp(dc.home_adv):.2f} scoring rate)")
    lines.append(f"   Dixon-Coles rho     : {dc.rho:+.4f}")
    lines.append(f"   Calibration temp. T : {T:.3f}")
    lines.append("")
    lines.append("3. HONEST TEMPORAL-HOLDOUT METRICS")
    lines.append("-" * 78)
    lines.append(f"   Evaluated on the most recent {n_test} matches (never seen in training).")
    lines.append(f"   1X2 accuracy        : {acc*100:.2f}%")
    lines.append(f"   Multiclass Brier    : {brier:.4f}   (lower is better; 0.66 = random)")
    lines.append(f"   Log-loss            : {logloss:.4f}   (lower is better; 1.099 = random)")
    lines.append("")
    lines.append("4. CALIBRATION (reliability of the top pick)")
    lines.append("-" * 78)
    lines.append(f"   {'confidence bin':<20}{'n':>6}{'avg conf':>12}{'actual acc':>14}")
    for lo, hi, n, c, a in cal_table:
        lines.append(f"   {f'{lo*100:.0f}%-{hi*100:.0f}%':<20}{n:>6}{c*100:>11.1f}%{a*100:>13.1f}%")
    lines.append("")
    lines.append("5. TOP 15 TEAMS BY ELO")
    lines.append("-" * 78)
    for i, (t, r) in enumerate(sorted(elo.items(), key=lambda x: -x[1])[:15], 1):
        atk = dc.attack.get(t, 0.0)
        dfc = dc.defence.get(t, 0.0)
        lines.append(f"   {i:2d}. {t:<22} Elo {r:7.1f}  | attack {atk:+.2f}  defence {dfc:+.2f}")
    lines.append("")
    lines.append("6. NOTES")
    lines.append("-" * 78)
    lines.append("   * Metrics are from a strict chronological holdout, so they reflect")
    lines.append("     genuine out-of-sample predictive power - not in-sample fit.")
    lines.append("   * The professional ceiling for international 1X2 is ~70-73%.")
    lines.append("   * Home advantage applies to host venues; WC2026 group games at")
    lines.append("     neutral sites should set homeAdvantage to 0 in the frontend.")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
