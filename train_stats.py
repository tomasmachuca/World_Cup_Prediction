#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expected corners & cards per national team, from international matches.

Source: eatpizzanot/soccer-dataset (fixtures.csv + match_stats.csv + teams.csv +
leagues.csv). We keep only international competitions (World Cup, Euro, Nations
League, Copa America, AFCON, Asian Cup, the qualifiers and friendlies) and only
fixtures that actually have detailed stats (shots > 0 — the dataset stores many
stat-less matches as all-zeros).

For each team we estimate a multiplicative attack/defence factor (Dixon-Coles
style) for corners and for cards (yellow+red), shrunk toward the global mean so
teams with few matches don't get extreme numbers:

    expected corners(A vs B) = muCorners * cAtt[A] * cDef[B]
    expected cards(A vs B)   = muCards   * kAtt[A] * kDef[B]

Output: stats_model.json, consumed by the front-end "Detalle analítico" panel.
The CSVs are downloaded to a temp dir if not already present locally.
"""

import csv
import json
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "stats_model.json")
BASE_URL = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/csv"
FILES = ["leagues.csv", "teams.csv", "fixtures.csv", "match_stats.csv"]

# International competition league ids (fd_code starts with INT-).
INT_LEAGUES = {78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88}

# The 48 WC2026 squads (model's canonical names).
WC_TEAMS = [
    "Czech Republic", "Mexico", "South Africa", "South Korea", "Bosnia and Herzegovina",
    "Canada", "Qatar", "Switzerland", "Brazil", "Haiti", "Morocco", "Scotland",
    "Australia", "Paraguay", "Turkey", "United States", "Germany", "Curaçao", "Ecuador",
    "Ivory Coast", "Japan", "Netherlands", "Sweden", "Tunisia", "Belgium", "Egypt", "Iran",
    "New Zealand", "Cabo Verde", "Saudi Arabia", "Spain", "Uruguay", "France", "Iraq",
    "Norway", "Senegal", "Algeria", "Argentina", "Austria", "Jordan", "Colombia",
    "DR Congo", "Portugal", "Uzbekistan", "Croatia", "England", "Ghana", "Panama",
]
# dataset spelling -> our canonical name
ALIAS = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "DR Congo",
    "USA": "United States",
}
SHRINK_K = 6.0   # matches of "prior" pulling each factor toward the mean (1.0)


def _data_dir():
    d = os.path.join(tempfile.gettempdir(), "wc_stats_data")
    os.makedirs(d, exist_ok=True)
    return d


def _path(name):
    """Prefer a local copy (./data or temp); download to temp if missing."""
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
    # team id -> canonical name (only the names we care about resolve cleanly).
    team_name = {}
    for r in _rows("teams.csv"):
        name = ALIAS.get(r["name"], r["name"])
        team_name[r["id"]] = name

    # international fixtures -> (home_id, away_id)
    intl = {}
    for r in _rows("fixtures.csv"):
        if _i(r["league_id"]) in INT_LEAGUES and r.get("status") in ("FT", "AET", "PEN"):
            intl[r["id"]] = (r["home_team_id"], r["away_team_id"])
    print(f"International fixtures: {len(intl)}")

    # accumulate per-team corners/cards for & against (only matches with real stats)
    acc = {}  # name -> dict
    used = 0

    def bump(name, cf, ca, kf, ka):
        d = acc.setdefault(name, {"m": 0, "cf": 0, "ca": 0, "kf": 0, "ka": 0})
        d["m"] += 1; d["cf"] += cf; d["ca"] += ca; d["kf"] += kf; d["ka"] += ka

    for r in _rows("match_stats.csv"):
        fx = intl.get(r["fixture_id"])
        if not fx:
            continue
        # gate: a real match has shots; all-zero rows are missing stats.
        if _i(r["home_shots_total"]) + _i(r["away_shots_total"]) <= 0:
            continue
        hid, aid = fx
        hc, ac = _i(r["home_corners"]), _i(r["away_corners"])
        hk = _i(r["home_yellow_cards"]) + _i(r["home_red_cards"])
        ak = _i(r["away_yellow_cards"]) + _i(r["away_red_cards"])
        bump(team_name.get(hid, "?"), hc, ac, hk, ak)
        bump(team_name.get(aid, "?"), ac, hc, ak, hk)
        used += 1
    print(f"Matches with detailed stats: {used}")

    # global means per team-match
    tot_m = sum(d["m"] for d in acc.values())
    mu_c = sum(d["cf"] for d in acc.values()) / tot_m
    mu_k = sum(d["kf"] for d in acc.values()) / tot_m
    print(f"Global means -> corners/team/match: {mu_c:.2f} · cards/team/match: {mu_k:.2f}")

    def factor(total, m, mu):
        # shrink the per-match rate toward the global mean, then express as ratio.
        rate = (total + SHRINK_K * mu) / (m + SHRINK_K)
        return round(rate / mu, 3)

    teams_out, covered, low = {}, 0, []
    for name in WC_TEAMS:
        d = acc.get(name)
        if not d or d["m"] == 0:
            low.append((name, 0))
            continue
        m = d["m"]
        teams_out[name] = {
            "matches": m,
            "cAtt": factor(d["cf"], m, mu_c), "cDef": factor(d["ca"], m, mu_c),
            "kAtt": factor(d["kf"], m, mu_k), "kDef": factor(d["ka"], m, mu_k),
        }
        covered += 1
        if m < 10:
            low.append((name, m))

    payload = {
        "source": "eatpizzanot/soccer-dataset (international matches)",
        "muCorners": round(mu_c, 3), "muCards": round(mu_k, 3),
        "shrinkK": SHRINK_K, "teamsCovered": covered,
        "teams": teams_out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_PATH}  ({covered}/48 WC teams covered)")
    if low:
        print("Low/zero coverage (<10 matches):")
        for n, m in sorted(low, key=lambda x: x[1]):
            print(f"  {n}: {m} matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
