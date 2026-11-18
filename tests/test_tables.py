"""Eingefrorene Läufe (`dgn_tables.json`, `dgn_exact.json`): Aufbau, Zählungs-Invarianten, exakte Gegenzahlen gegen die Live-Rechnung, Neurechnung einzelner Zellen im Band.

Die Neurechnung ist plattformfest (SplitMix64, nur IEEE-Rechnung im Pfad); für Zählungen gilt trotzdem ein Band von 2 (R-Dach nahe 1,01 kann je Plattform
eine Wiederholung kippen), für exakte Größen eine relative Toleranz von 1e-6 (`tools/dgn_make_tables.compare`)."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import dgn_constants as C  # noqa: E402
import dgn_evaluation as E  # noqa: E402
import dgn_exact as ex  # noqa: E402
import dgn_make_tables as T  # noqa: E402
import dgn_reference as R  # noqa: E402
import dgn_scenario as S  # noqa: E402

TABLES = R.tables()
EXACT = R.exact()


def test_files_cover_every_cell_window_start_and_rate():
    assert set(TABLES["cells"]) == {c.id for c in S.CELLS} == set(EXACT["cells"])
    assert set(TABLES["windows"]) == {w.id for w in S.WINDOWS} == set(EXACT["windows"])
    assert set(TABLES["mcells"]) == {m.id for m in S.MCELLS}
    assert set(TABLES["rates"]) == {r.id for r in S.SWITCH_CELLS} == set(EXACT["rates"])
    assert set(TABLES["starts"]) == {"gauss", "ring"} and all(len(v["rows"]) == len(S.START_PATTERNS) for v in TABLES["starts"].values())


def test_every_cell_has_200_repetitions_with_four_chains_and_a_fixed_seed():
    for rec in TABLES["cells"].values():
        assert rec["reps"] == 200 and rec["counts"]["reps"] == 200 and len(rec["kinds"]) == 4 and isinstance(rec["seed"], int)
    assert len({rec["seed"] for rec in TABLES["cells"].values()}) == len(TABLES["cells"])


def test_count_invariants_in_every_cell_and_window_row():
    rows = [r["counts"] for r in TABLES["cells"].values()] + [r for w in TABLES["windows"].values() for r in w["rows"]]
    for c in rows:
        reps = c["reps"]
        assert c["rhat_lt"] + c["rhat_ge"] + c["rhat_nan"] == reps
        if "deceiving" in c:
            assert c["deceiving"] <= c["no_switch"] and c["deceiving"] <= c["rhat_lt"] and c["today"] <= c["deceiving"] <= reps
            assert c["deceiving"] >= c["no_switch"] + c["rhat_lt"] - reps       # Überschneidung mindestens so groß
        assert 0.0 <= c["share_B_mean"] <= 1.0


def test_no_switch_counts_are_within_four_standard_errors_of_the_exact_expectation_in_all_rows():
    """Zählung gegen 200 p^4 (exakte Überlebenswahrscheinlichkeit): alle Zellen und alle Fensterzeilen."""
    n = 0
    for cid, rec in TABLES["cells"].items():
        c, e = rec["counts"], EXACT["cells"][cid]
        if "no_switch" in c:
            n += 1
            assert abs(c["no_switch"] - e["expected"]) <= 4 * max(e["sd"], 1.0), cid
    for wid, w in TABLES["windows"].items():
        for r, e in zip(w["rows"], EXACT["windows"][wid]["rows"]):
            n += 1
            assert r["N"] == e["N"] and abs(r["no_switch"] - e["expected"]) <= 4 * max(e["sd"], 1.0), (wid, r["N"])
    assert n == 13 + 10 + 9      # 13 Zellen mit Gipfeln (14 minus der Kontrolle D = 0), 19 Fensterzeilen


def test_deception_is_bounded_by_the_exact_no_switch_probability():
    """täuschend <= kein Wechsel <= 200 p^4 + 4 sd in jeder Zelle (die Täuschung ist durch die exakte Wechselwahrscheinlichkeit begrenzt)."""
    for cid, rec in TABLES["cells"].items():
        c, e = rec["counts"], EXACT["cells"][cid]
        if "deceiving" in c:
            assert c["deceiving"] <= e["expected"] + 4 * max(e["sd"], 1.0), cid


def test_switch_counts_match_the_exact_rate_within_four_over_sqrt_of_the_switches():
    items = [(w["switches"], EXACT["windows"][wid]["rate_exact"]) for wid, w in TABLES["windows"].items()]
    items += [(r["switches"], EXACT["rates"][rid]["rate_exact"]) for rid, r in TABLES["rates"].items()]
    assert len(items) == 3
    for sw, rate in items:
        assert abs(sw["rate"] / rate - 1) <= 4 / math.sqrt(sw["count"]), (sw, rate)


@pytest.mark.parametrize("cid", ["gauss_D10_s1_18101", "gauss_D8_s2.4_18113", "gauss_D6_s1_18111", "ring_n12_l50_18201", "ring_n12_l20_18203"])
def test_exact_numbers_in_the_file_equal_the_live_computation(cid):
    spec = next(c for c in S.CELLS if c.id == cid)
    live = S.exact_for(spec.model(), spec.N, spec.reps)
    assert T.compare(EXACT["cells"][cid], live, count_band=0, rel=1e-6) == []


@pytest.mark.parametrize("wid", ["window_gauss_D8_s1"])
def test_exact_window_numbers_equal_the_live_computation(wid):
    spec = next(w for w in S.WINDOWS if w.id == wid)
    assert T.compare(EXACT["windows"][wid], S.exact_window(spec), count_band=0, rel=1e-6) == []


@pytest.mark.parametrize("cid", ["gauss_D10_s1_18101", "ring_n12_l50_18201"])
def test_recomputing_a_cell_gives_the_counts_of_the_file_within_the_band(cid):
    spec = next(c for c in S.CELLS if c.id == cid)
    counts, _ = S.run_cell(spec)
    assert T.compare(TABLES["cells"][cid]["counts"], counts) == []


def test_recomputing_a_start_row_gives_the_file_within_the_band():
    name, kinds, burn = S.START_PATTERNS[1]
    vehicle, a, b, seed = S.START_MODELS[0]
    row = S.run_start((vehicle, a, b), name, kinds, burn, seed + 1)
    assert T.compare(TABLES["starts"]["gauss"]["rows"][1], row) == []


def test_compare_reports_differences():
    assert T.compare({"a": 1, "b": 1.0}, {"a": 5, "b": 1.0})
    assert not T.compare({"a": 1, "b": 1.0}, {"a": 3, "b": 1.0 + 1e-9})
    assert T.compare({"a": 1}, {"b": 1})


def test_selected_cells_equal_the_live_ones_in_the_app_lookup():
    rec = R.find_cell("gauss", 10.0, 1.0, 2000)
    a = E.analyse(E.Settings(N=2000))
    assert a.facts.frozen == (rec["counts"]["deceiving"], 200, rec["seed"])
    assert R.find_cell("gauss", 9.0, 1.0, 2000) is None
    assert C.REPS == 200
