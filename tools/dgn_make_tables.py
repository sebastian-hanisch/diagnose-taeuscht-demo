"""Erzeugt die eingefrorenen Läufe `dgn_tables.json` (Zählungen aus Simulationen) und `dgn_exact.json` (exakte Gegenzahlen).

    python tools/dgn_make_tables.py            # alle Zellen neu rechnen und beide Dateien schreiben (einige Minuten)
    python tools/dgn_make_tables.py --check    # eine Auswahl neu rechnen und mit den Dateien vergleichen (schnell)
    python tools/dgn_make_tables.py --check --lang   # alle Zellen neu rechnen und vergleichen

Die Läufe sind „vorgerechnet“: 200 Wiederholungen mit je vier Ketten je Zelle, Zufallsstrom SplitMix64 (`dgn_rng.py`, plattformfest), feste Zufalls-Seeds.
Der Vergleich in `--check` verlangt für Zählungen eine Abweichung von höchstens 2 (Rundung in R-Dach nahe 1,01 kann je Plattform eine Wiederholung kippen)
und für exakte Größen eine relative Abweichung von höchstens 1e-6."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dgn_constants as C  # noqa: E402
import dgn_scenario as S  # noqa: E402

COUNT_BAND = 2
EXACT_REL = 1e-6


def _spec_dict(spec) -> dict:
    return {"id": spec.id, "vehicle": spec.vehicle, "a": spec.a, "b": spec.b, "N": spec.N, "seed": spec.seed, "reps": spec.reps,
            "kinds": list(spec.kinds), "burn": spec.burn}


def build(which: str = "all", progress=print) -> tuple[dict, dict]:
    """which: 'all' oder 'quick' (kleine Auswahl für --check)."""
    quick = which == "quick"
    tables = {"meta": {"reps": C.REPS, "chains": C.CHAINS, "rng": "SplitMix64 (dgn_rng.py)", "rhat_limit": C.RHAT_LIMIT, "ess_limit": C.ESS_LIMIT},
              "cells": {}, "windows": {}, "starts": {}, "mcells": {}, "rates": {}}
    exact = {"cells": {}, "windows": {}, "starts": {}, "rates": {}}
    t0 = time.time()
    cells = S.CELLS
    if quick:
        cells = tuple(c for c in cells if c.id in ("gauss_D10_s1_18101", "gauss_D10_s2.4_18112", "ring_n12_l50_18201", "gauss_D0_s1_18114"))
    for spec in cells:
        counts, ex_ = S.run_cell(spec)
        tables["cells"][spec.id] = {**_spec_dict(spec), "counts": counts}
        exact["cells"][spec.id] = ex_
        progress(f"  Zelle {spec.id}: kein Wechsel {counts.get('no_switch')}, täuschend {counts.get('deceiving')}  ({time.time() - t0:.0f} s)")
    if not quick:
        for spec in S.WINDOWS:
            rows, sw = S.run_window(spec)
            tables["windows"][spec.id] = {**_spec_dict(spec), "Ns": list(spec.Ns), "rows": rows, "switches": sw}
            exact["windows"][spec.id] = S.exact_window(spec)
            progress(f"  Fenster {spec.id}: {len(rows)} Längen, {sw['count']} Wechsel gezählt  ({time.time() - t0:.0f} s)")
    models = S.START_MODELS[:1] if quick else S.START_MODELS
    for vehicle, a, b, seed in models:
        rows = []
        pats = S.START_PATTERNS[:3] if quick else S.START_PATTERNS
        for k, (name, kinds, burn) in enumerate(pats):
            rows.append(S.run_start((vehicle, a, b), name, kinds, burn, seed + k))
        tables["starts"][vehicle] = {"model": [vehicle, a, b], "rows": rows}
        progress(f"  Starts {vehicle}: {len(rows)} Zeilen  ({time.time() - t0:.0f} s)")
    if not quick:
        for spec in S.MCELLS:
            row = S.run_start((spec.vehicle, spec.a, spec.b), spec.id, spec.kinds, spec.burn, spec.seed, reps=spec.reps, N=spec.N)
            tables["mcells"][spec.id] = row
            progress(f"  M = {row['chains']}: selbe Seite {row['same_side']} von {row['reps']}  ({time.time() - t0:.0f} s)")
        for spec in S.SWITCH_CELLS:
            model = spec.model()
            path = S.ch.simulate(model, spec.kinds, spec.N, spec.seed, reps=spec.reps)
            tables["rates"][spec.id] = {**_spec_dict(spec), "switches": S.count_switches(model, path)}
            exact["rates"][spec.id] = {"rate_exact": S.ex.switch_rate(model), "E_T": S.ex.killed_from_A(model).mean_exit(),
                                       "t_rel": S.ex.relaxation(model.P, model.pi)[1]}
            progress(f"  Wechselrate {spec.id}: {tables['rates'][spec.id]['switches']['count']} Wechsel gezählt  ({time.time() - t0:.0f} s)")
    return tables, exact


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def compare(old, new, where="", count_band=COUNT_BAND, rel=EXACT_REL) -> list[str]:
    """Unterschiede zwischen gespeicherten und neu gerechneten Daten (nur Schlüssel, die in `new` vorkommen)."""
    bad = []
    if isinstance(new, dict):
        for k, v in new.items():
            if k not in old:
                bad.append(f"{where}/{k}: fehlt in der Datei")
            else:
                bad += compare(old[k], v, f"{where}/{k}", count_band, rel)
    elif isinstance(new, list):
        if len(old) != len(new):
            bad.append(f"{where}: Länge {len(old)} gegen {len(new)}")
        else:
            for i, (o, n) in enumerate(zip(old, new)):
                bad += compare(o, n, f"{where}[{i}]", count_band, rel)
    elif isinstance(new, bool) or new is None or isinstance(new, str):
        if old != new:
            bad.append(f"{where}: {old!r} gegen {new!r}")
    elif isinstance(new, int):
        if abs(old - new) > count_band:
            bad.append(f"{where}: {old} gegen {new}")
    else:
        if abs(old - new) > rel * max(1.0, abs(new)):
            bad.append(f"{where}: {old} gegen {new}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="neu rechnen und mit den Dateien vergleichen, nichts schreiben")
    ap.add_argument("--lang", action="store_true", help="mit --check: alle Zellen statt einer Auswahl")
    args = ap.parse_args(argv)
    if args.check:
        tables, exact = build("all" if args.lang else "quick")
        old_t = json.loads((ROOT / C.TABLES_FILE).read_text(encoding="utf-8"))
        old_e = json.loads((ROOT / C.EXACT_FILE).read_text(encoding="utf-8"))
        bad = compare(old_t, tables) + compare(old_e, exact, count_band=0)
        if bad:
            print("ABWEICHUNGEN:\n  " + "\n  ".join(bad[:40]))
            return 1
        print("OK: die neu gerechneten Zellen stimmen mit den Dateien überein.")
        return 0
    tables, exact = build("all")
    _dump(tables, ROOT / C.TABLES_FILE)
    _dump(exact, ROOT / C.EXACT_FILE)
    print(f"geschrieben: {C.TABLES_FILE}, {C.EXACT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
