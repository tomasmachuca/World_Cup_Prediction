#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value-betting backtest of the model vs the market (Pinnacle closing odds).

Honest, leakage-free design:
  1. Walk-forward margin-aware Elo over ALL international matches chronologically
     (each match is predicted using only prior matches; ratings update after).
  2. Fit an Elo -> 1X2 probability mapping (multinomial logistic on the Elo gap
     and its absolute value, so draws peak for even matchups) using ONLY matches
     before the first odds date — never on the backtest period.
  3. Backtest period = the international matches that have Pinnacle 1X2 odds.
     For each, de-vig the bookmaker into fair probabilities; if the model's
     probability for an outcome exceeds the fair probability by `edge`, bet 1
     flat unit at the bookmaker price; settle on the real result.
  4. Report ROI / yield / hit rate / bankroll curve, plus baselines, into
     backtest.json for the front-end.

Pinnacle's closing line is the sharpest public market, so beating it is very
hard — whatever the result, it's reported truthfully.
"""

import csv
import json
import os
import sys
import tempfile
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "backtest.json")
BASE_URL = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/csv"
INT_LEAGUES = {78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88}
# Elo ratings are learned from ALL international play (above), but the betting
# backtest is evaluated only on these competitions. World Cup only (league 78).
BET_LEAGUES = {78}
HFA = 55.0       # modest home-field bump (no neutral flag in this dataset)
K = 40.0         # Elo K-factor
EDGE = 0.03      # required model-vs-fair edge to place a value bet (headline)


def _data_dir():
    d = os.path.join(tempfile.gettempdir(), "wc_stats_data")
    os.makedirs(d, exist_ok=True)
    return d


def _path(name):
    for cand in (os.path.join(HERE, "data", name), os.path.join(_data_dir(), name)):
        if os.path.exists(cand):
            return cand
    dst = os.path.join(_data_dir(), name)
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", dst)
    return dst


def _rows(name):
    with open(_path(name), encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _i(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def main():
    team_name = {r["id"]: r["name"] for r in _rows("teams.csv")}

    # International matches with a result, chronological.
    matches = []
    for r in _rows("fixtures.csv"):
        if _i(r["league_id"]) in INT_LEAGUES and r.get("status") in ("FT", "AET", "PEN"):
            d = (r["date"] or "")[:10]
            if d:
                matches.append((d, r["id"], r["home_team_id"], r["away_team_id"],
                                _i(r["goals_home"]), _i(r["goals_away"]), _i(r["league_id"])))
    matches.sort()
    print(f"International matches: {len(matches)}")

    # Pinnacle (preferred) 1X2 odds by fixture.
    odds = {}
    for r in _rows("odds.csv"):
        fid = r["fixture_id"]
        try:
            o = (float(r["home_win"]), float(r["draw"]), float(r["away_win"]))
        except (ValueError, TypeError):
            continue
        if min(o) <= 1.0:
            continue
        if fid not in odds or r["bookmaker"] == "Pinnacle":
            odds[fid] = o
    first_odds_date = min((m[0] for m in matches if m[1] in odds), default=None)
    print(f"Matches with 1X2 odds: {sum(1 for m in matches if m[1] in odds)} (from {first_odds_date})")

    # ---- Walk-forward Elo: record pre-match features + outcome ----
    elo = {}
    feats, ys, rows = [], [], []   # rows: (date, fid, home, away, outcome, league)
    for d, fid, h, a, gh, ga, lid in matches:
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        diff = (eh + HFA) - ea
        outcome = 0 if gh > ga else (1 if gh == ga else 2)
        feats.append([diff / 100.0, abs(diff) / 100.0])
        ys.append(outcome)
        rows.append((d, fid, h, a, outcome, lid))
        # update Elo (margin-of-victory aware)
        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        res = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        gd = abs(gh - ga)
        mov = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        delta = K * mov * (res - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    feats = np.array(feats)
    ys = np.array(ys)

    # ---- Fit Elo -> 1X2 mapping on the pre-odds period only ----
    train_mask = np.array([r[0] < first_odds_date for r in rows])
    clf = _fit_softmax(feats[train_mask], ys[train_mask])
    print(f"Mapping trained on {train_mask.sum()} pre-odds matches")

    # ---- Backtest on matches that have odds ----
    def devig(o):
        inv = [1.0 / x for x in o]
        s = sum(inv)
        return [x / s for x in inv]

    results = {"all": _new(), "fav": _new(), "model": _new()}
    by_edge = {e: _new() for e in (0.0, 0.02, 0.05, 0.08, 0.12)}
    curve = []   # cumulative profit of the headline (EDGE) strategy
    examples = []
    # Stable comparison (doesn't depend on betting variance): how well the
    # model and the market predicted the same World Cup matches.
    pq = {"n": 0, "m_hits": 0, "mk_hits": 0, "m_brier": 0.0, "mk_brier": 0.0}

    for k in range(len(rows)):
        d, fid, h, a, outcome, lid = rows[k]
        if fid not in odds or lid not in BET_LEAGUES:
            continue
        o = odds[fid]
        p = clf(feats[k])           # model probs [home, draw, away]
        fair = devig(o)             # market fair probs

        # prediction quality (model vs market) on the same matches
        pq["n"] += 1
        y = [0.0, 0.0, 0.0]; y[outcome] = 1.0
        if int(np.argmax(p)) == outcome: pq["m_hits"] += 1
        if int(np.argmax(fair)) == outcome: pq["mk_hits"] += 1
        pq["m_brier"] += sum((p[i] - y[i]) ** 2 for i in range(3))
        pq["mk_brier"] += sum((fair[i] - y[i]) ** 2 for i in range(3))

        # baseline: bet the bookmaker favourite every match
        fav = int(np.argmin(o))
        _settle(results["fav"], fav == outcome, o[fav])
        # baseline: bet the model's top pick every match
        mp = int(np.argmax(p))
        _settle(results["model"], mp == outcome, o[mp])

        # value bets at several edge thresholds
        for e, acc in by_edge.items():
            for i in range(3):
                if p[i] > fair[i] * (1 + e) and p[i] > 0.05:
                    _settle(acc, i == outcome, o[i])

        # headline strategy (EDGE) + curve + examples
        for i in range(3):
            if p[i] > fair[i] * (1 + EDGE) and p[i] > 0.05:
                _settle(results["all"], i == outcome, o[i])
                curve.append(round(results["all"]["profit"], 2))
                if len(examples) < 8:
                    examples.append({
                        "match": f'{team_name.get(h, h)} vs {team_name.get(a, a)}',
                        "pick": ["1", "X", "2"][i],
                        "modelProb": round(p[i] * 100, 1),
                        "fairProb": round(fair[i] * 100, 1),
                        "odd": round(o[i], 2),
                        "won": bool(i == outcome),
                    })

    payload = {
        "source": "eatpizzanot/soccer-dataset · Pinnacle closing odds",
        "competition": "Copa del Mundo (FIFA World Cup 2022)",
        "fromDate": first_odds_date,
        "edgeThreshold": EDGE,
        "predictionQuality": {
            "n": pq["n"],
            "model": {"acc": round(pq["m_hits"] / pq["n"] * 100, 1) if pq["n"] else 0,
                      "brier": round(pq["m_brier"] / pq["n"], 4) if pq["n"] else 0},
            "market": {"acc": round(pq["mk_hits"] / pq["n"] * 100, 1) if pq["n"] else 0,
                       "brier": round(pq["mk_brier"] / pq["n"], 4) if pq["n"] else 0},
        },
        "headline": _summary(results["all"]),
        "baselines": {
            "betFavourite": _summary(results["fav"]),
            "betModelPick": _summary(results["model"]),
        },
        "byEdge": [{"edge": e, **_summary(acc)} for e, acc in by_edge.items()],
        "bankrollCurve": _downsample(curve, 120),
        "examples": examples,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    s = payload["headline"]
    print(f"\nWrote {OUT_PATH}")
    print(f"Headline (edge>{EDGE:.0%}): {s['bets']} bets · ROI {s['roi']}% · "
          f"hit {s['hitRate']}% · profit {s['profit']}u")
    print(f"Baseline bet-favourite: ROI {payload['baselines']['betFavourite']['roi']}% "
          f"({payload['baselines']['betFavourite']['bets']} bets)")
    return 0


# ---- tiny helpers -----------------------------------------------------------
def _new():
    return {"bets": 0, "wins": 0, "profit": 0.0}


def _settle(acc, won, odd):
    acc["bets"] += 1
    if won:
        acc["wins"] += 1
        acc["profit"] += odd - 1.0
    else:
        acc["profit"] -= 1.0


def _summary(acc):
    n = acc["bets"]
    return {
        "bets": n,
        "profit": round(acc["profit"], 2),
        "roi": round(acc["profit"] / n * 100, 2) if n else 0.0,
        "hitRate": round(acc["wins"] / n * 100, 1) if n else 0.0,
    }


def _downsample(arr, n):
    if len(arr) <= n:
        return arr
    step = len(arr) / n
    return [arr[int(i * step)] for i in range(n)]


def _fit_softmax(X, y, l2=1.0, iters=400, lr=0.5):
    """Minimal 3-class multinomial logistic (numpy), with intercept + ridge."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    n, d = Xb.shape
    W = np.zeros((d, 3))
    Y = np.eye(3)[y]
    for _ in range(iters):
        Z = Xb @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        grad = Xb.T @ (P - Y) / n + l2 * W / n
        W -= lr * grad
    def predict(x):
        xb = np.hstack([[1.0], x])
        z = xb @ W
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        return p
    return predict


if __name__ == "__main__":
    sys.exit(main())
