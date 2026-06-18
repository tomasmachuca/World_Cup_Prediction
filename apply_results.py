#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject the live World Cup results (results.json, from football-data.org) into
the training dataset before retraining.

The martj42 feed often lags behind the live scores, so the World Cup rows in
data/international_results.csv can still be unplayed (NA) or partially filled.
This patches those rows with the real, finished scores we already have live, so
train_advanced.py learns from the latest results. Idempotent: only fills
finished matches, leaves the rest untouched.
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "international_results.csv")
RESULTS_PATH = os.path.join(HERE, "results.json")

# Live (football-data) name -> dataset (martj42) spelling, where they differ.
CSV_ALIAS = {"Cabo Verde": "Cape Verde"}


def _csv_name(n):
    return CSV_ALIAS.get(n, n)


def main():
    if not (os.path.exists(CSV_PATH) and os.path.exists(RESULTS_PATH)):
        print("apply_results: dataset or results.json missing — skipping.")
        return 0

    # frozenset({home, away}) -> {team: goals} for finished matches.
    finished = {}
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    for v in data.get("matches", {}).values():
        if v.get("status") != "FINISHED":
            continue
        h, a = _csv_name(v["home"]), _csv_name(v["away"])
        finished[frozenset({h, a})] = {h: v["homeGoals"], a: v["awayGoals"]}

    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    patched = 0
    for r in rows:
        if not (r["date"].startswith("2026-") and "World Cup" in r["tournament"]):
            continue
        fs = frozenset({r["home_team"], r["away_team"]})
        g = finished.get(fs)
        if not g or r["home_team"] not in g or r["away_team"] not in g:
            continue
        hs, as_ = str(g[r["home_team"]]), str(g[r["away_team"]])
        if (r["home_score"], r["away_score"]) != (hs, as_):
            r["home_score"], r["away_score"] = hs, as_
            patched += 1

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"apply_results: {len(finished)} finished live matches · patched {patched} dataset rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
