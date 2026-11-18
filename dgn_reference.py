"""Lädt die eingefrorenen Läufe (`dgn_tables.json`: Zählungen aus Simulationen, `dgn_exact.json`: exakte Gegenzahlen) und stellt sie je Zelle bereit.

Beide Dateien erzeugt `tools/dgn_make_tables.py` (vorgerechnet: 200 Wiederholungen mit je vier Ketten je Zelle, feste Zufalls-Seeds); sie sind Messungen,
keine Live-Rechnung der App."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import dgn_constants as C
import dgn_scenario as S

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def tables() -> dict:
    return json.loads((ROOT / C.TABLES_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def exact() -> dict:
    return json.loads((ROOT / C.EXACT_FILE).read_text(encoding="utf-8"))


def cell_list(vehicle: str) -> list[dict]:
    """Zellen der Tabelle (Kettenlänge 2000, Start im Gipfel A) eines Modells: Festlegung, Zählungen und exakte Größen in einem Datensatz."""
    t, e = tables()["cells"], exact()["cells"]
    order = {c.id: i for i, c in enumerate(S.CELLS)}        # Reihenfolge der Zellenliste (die JSON-Datei ist nach Schlüsseln sortiert)
    out = [{**rec, "exact": e[cid]} for cid, rec in t.items() if rec["vehicle"] == vehicle]
    return sorted(out, key=lambda r: order[r["id"]])


def cell_label(rec: dict) -> str:
    from dgn_format import dez, ganz
    if rec["vehicle"] == "gauss":
        if rec["a"] == 0:
            core = f"Gaußmischung D = 0 (nur ein Gipfel), σ = {dez(rec['b'], 1)}"
        else:
            core = f"Gaußmischung D = {rec['a']:g}, σ = {dez(rec['b'], 1)}"
    else:
        core = f"Ring n = {int(rec['a'])}, λ = {rec['b']:g}"
    return f"{core}, N = {ganz(rec['N'])} (Zufalls-Seed {rec['seed']})"


def window(vehicle: str) -> dict:
    """Fensterzelle eines Modells: Zählungen je Kettenlänge und exakte Gegenzahlen, zusammengeführt."""
    for wid, rec in tables()["windows"].items():
        if rec["vehicle"] == vehicle:
            ex_ = exact()["windows"][wid]
            rows = [{**r, **ex_["rows"][i]} for i, r in enumerate(rec["rows"])]
            return {"id": wid, "spec": {k: rec[k] for k in ("vehicle", "a", "b", "seed", "reps")}, "rows": rows, "switches": rec["switches"],
                    "E_T": ex_["E_T"], "t_rel": ex_["t_rel"], "rate_exact": ex_["rate_exact"]}
    raise KeyError(vehicle)


def starts(vehicle: str) -> list[dict]:
    return tables()["starts"][vehicle]["rows"]


def mcells() -> list[dict]:
    """Gestreute Starts mit 2, 4 und 8 Ketten (je 1000 Wiederholungen), nach Kettenzahl geordnet."""
    return sorted(tables()["mcells"].values(), key=lambda r: r["chains"])


def rate_cell() -> dict:
    """Wechselrate bei großer Schrittweite (Gaußmischung D = 6, σ = 2,4): Zählung und exakte Rate."""
    (rid, rec), = tables()["rates"].items()
    return {**rec, "exact": exact()["rates"][rid]}


def find_cell(vehicle: str, a: float, b: float, N: int = 2000) -> dict | None:
    """Eine eingefrorene Zelle, deren Einstellung genau der gefragten entspricht (erste in der Liste), sonst None."""
    for rec in cell_list(vehicle):
        if rec["a"] == a and rec["b"] == b and rec["N"] == N:
            return rec
    return None
