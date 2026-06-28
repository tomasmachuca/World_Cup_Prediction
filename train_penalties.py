#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-team penalty-shootout strength, from martj42 shootouts.csv (every
international shootout with its winner). Used to predict who advances when a
knockout tie is level and goes to penalties.

Each team gets a shootout win rate shrunk toward 0.5 (shootouts are noisy and
many teams have few), so:
    P(A wins the shootout) = rate_A / (rate_A + rate_B)

Output: penalties.json, consumed by the front-end knockout predictor.
"""

import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "penalties.json")
URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
SHRINK_K = 4.0   # matches of "prior" pulling each rate toward 0.5

# dataset spelling -> our canonical (model) name
NAME_MAP = {
    "Cape Verde": "Cabo Verde", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Turkiye": "Turkey", "IR Iran": "Iran",
    "Korea Republic": "South Korea", "Côte d'Ivoire": "Ivory Coast",
}


def _norm(n):
    n = (n or "").strip()
    return NAME_MAP.get(n, n)


def main():
    try:
        raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR downloading shootouts.csv: {e}", file=sys.stderr)
        return 1

    played, won = defaultdict(int), defaultdict(int)
    for r in csv.DictReader(raw.splitlines()):
        h, a, w = _norm(r["home_team"]), _norm(r["away_team"]), _norm(r["winner"])
        if not h or not a:
            continue
        played[h] += 1; played[a] += 1
        if w:
            won[w] += 1

    teams = {}
    for t in played:
        p, wv = played[t], won[t]
        teams[t] = {"played": p, "won": wv,
                    "rate": round((wv + SHRINK_K * 0.5) / (p + SHRINK_K), 3)}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"source": "martj42/international_results · shootouts",
                   "shrinkK": SHRINK_K, "teams": teams}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}  ({len(teams)} teams)")
    # quick peek at the WC strong/weak shootout sides
    for t in ("Argentina", "Germany", "Croatia", "England", "Netherlands", "Spain"):
        if t in teams:
            d = teams[t]
            print(f"  {t}: {d['won']}/{d['played']} -> rate {d['rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
