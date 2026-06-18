#!/usr/bin/env python3
"""
Live group standings for the FIFA World Cup 2026, pulled from football-data.org.

Reads the API token from the FOOTBALL_DATA_TOKEN environment variable, fetches
the live group tables for competition WC, and writes them to standings.json in
the same schema the front-end already consumes:

    { "updated", "matchesPlayed", "live", "source", "groups": { "A": [ rows ] } }

Run locally for a quick test:

    FOOTBALL_DATA_TOKEN=xxxxx python live_standings.py        # bash
    $env:FOOTBALL_DATA_TOKEN="xxxxx"; python live_standings.py  # PowerShell

In CI the token comes from the GitHub Actions secret of the same name. The file
is only overwritten when the API returns valid group tables, so a transient API
error never wipes the existing standings.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "WC"  # FIFA World Cup
HERE = os.path.dirname(os.path.abspath(__file__))
STANDINGS_PATH = os.path.join(HERE, "standings.json")
RESULTS_PATH = os.path.join(HERE, "results.json")

# football-data.org spellings -> the names used in fixture.json / the model.
API_NAME_MAP = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cape Verde": "Cabo Verde",
    "Cape Verde Islands": "Cabo Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
}


def _norm_team(name):
    name = (name or "").strip()
    return API_NAME_MAP.get(name, name)


def _get(path, token):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        headers={"X-Auth-Token": token, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _group_letter(raw):
    """'GROUP_A' / 'Group A' / 'A' -> 'A'. Returns None for knockout rows."""
    if not raw:
        return None
    s = str(raw).upper().replace("GROUP", "").strip(" _-")
    return s if len(s) == 1 and s.isalpha() else None


def build_standings(token):
    data = _get(f"/competitions/{COMPETITION}/standings", token)
    groups = {}

    for block in data.get("standings", []):
        # Only the combined (TOTAL) group-stage tables, not HOME/AWAY splits.
        if block.get("type") and block.get("type") != "TOTAL":
            continue
        letter = _group_letter(block.get("group"))
        if not letter:
            continue

        rows = []
        for r in block.get("table", []):
            team = _norm_team((r.get("team") or {}).get("name") or (r.get("team") or {}).get("shortName") or "?")
            won = r.get("won", 0)
            drawn = r.get("draw", r.get("drawn", 0))
            lost = r.get("lost", 0)
            gf = r.get("goalsFor", 0)
            ga = r.get("goalsAgainst", 0)
            rows.append({
                "team": team,
                "position": r.get("position", len(rows) + 1),
                "played": r.get("playedGames", won + drawn + lost),
                "won": won,
                "drawn": drawn,
                "lost": lost,
                "gf": gf,
                "ga": ga,
                "gd": r.get("goalDifference", gf - ga),
                "points": r.get("points", won * 3 + drawn),
            })
        if rows:
            # Keep football-data's official order (it applies the WC2026 rules:
            # head-to-head BEFORE overall goal difference). Don't re-sort here.
            rows.sort(key=lambda x: x["position"])
            groups[letter] = rows

    return groups


def count_played(token):
    """Number of finished group-stage matches (best-effort; 0 if unavailable)."""
    try:
        data = _get(f"/competitions/{COMPETITION}/matches?status=FINISHED", token)
        played = 0
        for m in data.get("matches", []):
            stage = (m.get("stage") or "").upper()
            if "GROUP" in stage:
                played += 1
        return played
    except Exception:
        return None


def _outcome(gh, ga):
    """1 = home win, X = draw, 2 = away win."""
    return "1" if gh > ga else ("X" if gh == ga else "2")


def build_results(token):
    """
    Match results keyed by the unordered, alphabetically-sorted team pair, so
    the front-end can line each fixture row up regardless of home/away order:

        "Mexico|South Africa": {
            "home": "Mexico", "away": "South Africa",
            "homeGoals": 2, "awayGoals": 0, "outcome": "1",
            "status": "FINISHED", "live": false
        }

    Includes in-play matches (with the running score) so the Fixture can show a
    live score; the tick/cross is only meaningful once status is FINISHED.
    """
    data = _get(f"/competitions/{COMPETITION}/matches", token)
    out = {}
    for m in data.get("matches", []):
        status = (m.get("status") or "").upper()
        ft = (m.get("score") or {}).get("fullTime") or {}
        gh, ga = ft.get("home"), ft.get("away")
        if gh is None or ga is None:
            continue  # not started / no score yet
        home = _norm_team((m.get("homeTeam") or {}).get("name"))
        away = _norm_team((m.get("awayTeam") or {}).get("name"))
        if not home or not away:
            continue
        key = "|".join(sorted((home, away)))
        out[key] = {
            "home": home, "away": away,
            "homeGoals": gh, "awayGoals": ga,
            "outcome": _outcome(gh, ga),
            "status": status,
            "live": status in ("IN_PLAY", "PAUSED", "LIVE"),
        }
    return out


def main():
    token = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    if not token:
        # Soft skip so the scheduled workflow stays green until the secret is
        # added — no live data yet, but nothing breaks.
        print("FOOTBALL_DATA_TOKEN not set — skipping live update (configure the secret).")
        return 0

    try:
        groups = build_standings(token)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print(f"ERROR: football-data.org HTTP {e.code}: {body}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not fetch standings: {e}", file=sys.stderr)
        return 2

    if not groups:
        print("No group tables returned yet — leaving standings.json untouched.")
        return 0

    played = count_played(token)
    if played is None:
        played = sum(max(r["played"] for r in rows) for rows in groups.values())

    payload = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matchesPlayed": played,
        "live": True,
        "source": "football-data.org",
        "groups": {g: groups[g] for g in sorted(groups)},
    }
    with open(STANDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {STANDINGS_PATH}  ({len(groups)} groups, {played} matches played)")

    # ---- results.json: real scores so the Fixture can show tick/cross -------
    # Always written (even empty) so the workflow's `git add` never fails.
    try:
        results = build_results(token)
    except Exception as e:  # noqa: BLE001 — never fail the whole run over results
        print(f"WARN: could not build results.json: {e}", file=sys.stderr)
        results = {}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "football-data.org",
            "matches": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {RESULTS_PATH}  ({len(results)} matches with a score)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
