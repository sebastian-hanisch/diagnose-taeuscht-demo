"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import copy

import pytest

import dgn_evaluation as E


def _facts(**over):
    base = dict(vehicle="Gaußmischung", has_peaks=True, N=2000, start="alle_A", rhat=1.002, ess=930.0, rhat_ok=True, today=True, undefined=False,
                seen_A=True, seen_B=False, switches=(0, 0, 0, 0), share_B=0.005, ess_exact=4.2, p_all=0.957, frozen=(192, 200, 18101))
    base.update(over)
    return E.Facts(**base)


def test_deception_names_rhat_ess_share_exact_ess_and_counts():
    kind, text = E.verdict_run(_facts())
    assert kind == "error"
    assert "R̂ = 1,00" in text and "ESS_bulk 930" in text and "kein Wechsel" in text
    assert "0,005" in text and "der exakte 0,5" in text and "4,2" in text
    assert "gibt hier Entwarnung" in text
    assert "95,7 %" in text and "191,4 von 200" in text
    assert "täuschend in 192 von 200" in text and "Zufalls-Seed 18101" in text


def test_deception_without_frozen_cell_or_exact_probability_omits_those_sentences():
    kind, text = E.verdict_run(_facts(frozen=None, p_all=None))
    assert kind == "error" and "vorgerechneten" not in text and "Nach der exakten Rechnung" not in text


def test_today_rule_sentence_depends_on_the_ess():
    _, text = E.verdict_run(_facts(today=False, ess=250.0))
    assert "keine Entwarnung" in text and "nur 250" in text


def test_rhat_alarm_without_switch_and_single_peak_is_noise_not_a_second_peak():
    kind, text = E.verdict_run(_facts(rhat=1.2, rhat_ok=False, today=False, ess=80.0))
    assert kind == "warning" and "R̂ = 1,20" in text and "keinen zweiten Gipfel" in text and "ESS_bulk 80" in text


def test_start_in_both_peaks_gives_an_alarm_that_names_the_starts():
    kind, text = E.verdict_run(_facts(rhat=3.4, rhat_ok=False, today=False, seen_B=True))
    assert kind == "info" and "R̂ = 3,40" in text and "Startpunkte beide Gipfel überdecken" in text


def test_alarm_after_switches_names_the_number_of_switches_and_the_share():
    kind, text = E.verdict_run(_facts(rhat=1.5, rhat_ok=False, today=False, seen_B=True, switches=(2, 1, 0, 3), share_B=0.31))
    assert kind == "info" and "6 Wechsel zusammen" in text and "0,310" in text and "exakt 0,5" in text


def test_both_peaks_visited_and_rhat_fine_is_a_success():
    kind, text = E.verdict_run(_facts(seen_B=True, switches=(1, 2, 1, 1), share_B=0.48, ess=700.0))
    assert kind == "success" and "5 Wechsel zusammen" in text and "0,480" in text and "stimmt die Entwarnung" in text


def test_control_without_second_peak():
    kind, text = E.verdict_run(_facts(has_peaks=False, ess=972.0, ess_exact=966.0, share_B=0.5, switches=(), p_all=None, frozen=None))
    assert kind == "success" and "Kontrolle" in text and "ESS_bulk 972" in text and "exakt ESS 966" in text
    kind, text = E.verdict_run(_facts(has_peaks=False, rhat=1.03, rhat_ok=False, ess=50.0, ess_exact=60.0, switches=(), p_all=None, frozen=None))
    assert kind == "info" and "R̂ = 1,03" in text and "60" in text


CELL = {
    "vehicle": "gauss", "a": 10, "b": 1.0, "N": 2000, "seed": 18101, "kinds": ["A"] * 4,
    "counts": {"reps": 200, "no_switch": 195, "rhat_lt": 191, "deceiving": 191, "today": 190, "ess_mean": 939.0, "n_eff_mse": 1.01},
    "exact": {"expected": 191.4, "sd": 2.9, "ess_stationary": 4.19},
}


def test_cell_verdict_thresholds_and_numbers():
    kind, text = E.verdict_cell(CELL)
    assert kind == "error"
    assert "kein Wechsel in 195 von 200" in text and "191,4 ± 2,9" in text and "R̂ < 1,01 in 191 von 200" in text
    assert "täuschend (beides) in 191 von 200" in text and "in 190 von 200" in text and "939" in text and "4,2" in text
    c = copy.deepcopy(CELL)
    c["counts"]["deceiving"] = 100                       # genau die Hälfte: Grenze inklusive
    assert E.verdict_cell(c)[0] == "error"
    c["counts"]["deceiving"] = 99
    assert E.verdict_cell(c)[0] == "warning"
    c["counts"]["deceiving"] = 0
    kind, text = E.verdict_cell(c)
    assert kind == "success" and "Keine einzige Täuschung in 200 Wiederholungen" in text and "keine Obergrenze" in text


def test_cell_verdict_for_the_control_cell():
    c = {"vehicle": "gauss", "a": 0, "b": 1.0, "N": 2000, "seed": 1, "kinds": ["A"] * 4,
         "counts": {"reps": 200, "rhat_lt": 199, "ess_mean": 975.0, "N": 2000}, "exact": {"ess_stationary": 965.99}}
    kind, text = E.verdict_cell(c)
    assert kind == "info" and "R̂ < 1,01 in 199 von 200" in text and "975" in text and "966" in text and "Kontrolle" in text


def _window(rows, E_T=5000.0, reps=200):
    return {"rows": rows, "E_T": E_T, "spec": {"reps": reps}, "switches": {"count": 10000, "steps": 40000000, "rate": 2.5e-4}, "rate_exact": 2.5e-4}


def _row(N, dec, alarm, share, lt=0):
    return {"N": N, "deceiving": dec, "alarm": alarm, "share_B_mean": share, "rhat_lt": lt}


def test_window_verdict_names_peak_alarm_stretch_and_last_row():
    rows = [_row(100, 7, 193, 0.017), _row(1000, 70, 130, 0.104), _row(10000, 0, 200, 0.377), _row(20000, 0, 200, 0.421), _row(100000, 0, 193, 0.479, lt=7)]
    kind, text = E.verdict_window(_window(rows))
    assert kind == "info"
    assert "bei N = 1000 (0,20-fache" in text and "E[T] = 5000" in text and "täuschend in 70 von 200" in text
    assert "Von N = 10 000 (2,00 E[T]) bis N = 20 000 meldet R̂ ≥ 1,01 in 200 von 200" in text and "noch 0,377 statt 0,5" in text
    assert "Bei N = 100 000 (20,0 E[T])" in text and "0,479" in text and "R̂ < 1,01 in 7 von 200" in text


def test_window_verdict_without_a_full_alarm_row():
    kind, text = E.verdict_window(_window([_row(100, 2, 150, 0.01), _row(1000, 4, 190, 0.02)]))
    assert "meldet R̂ ≥ 1,01 in 200 von 200" not in text and "täuschend in 4 von 200" in text


def test_rate_verdict_band():
    kind, text = E.verdict_rate(_window([]))
    assert kind == "success" and "2,50·10⁻⁴" in text and "alle 4000 Schritte" in text and "10 000 Wechsel" in text and "40 000 000 Schritte" in text
    assert "Verhältnis gezählt zu exakt 1,000" in text and "innerhalb von 4/√Wechsel = 0,040" in text
    w = _window([])
    w["switches"] = {"count": 10000, "steps": 40000000, "rate": 2.5e-4 * 1.05}
    kind, text = E.verdict_rate(w)
    assert kind == "warning" and "außerhalb" in text


def _srow(name, burn, ge, same, ok, reps=200):
    return {"name": name, "burn": burn, "reps": reps, "rhat_ge": ge, "rhat_lt": reps - ge, "same_side": same, "same_and_rhat_lt": ok}


def test_starts_verdict_counts_scattered_rows_after_burn_in():
    rows = [_srow("alle vier in A", 0, 16, 187, 183), _srow("drei in A, eine in B", 0, 200, 0, 0), _srow("zwei in A, zwei in B", 0, 200, 0, 0),
            _srow("gestreut, sd 5", 0, 175, 25, 25), _srow("gestreut, sd 5", 500, 179, 21, 21),
            _srow("gestreut, sd 10", 0, 174, 26, 25), _srow("gestreut, sd 10", 500, 180, 20, 20)]
    kind, text = E.verdict_starts(rows)
    assert kind == "info" and "R̂ ≥ 1,01 in 200 von 200 Wiederholungen (zwei und zwei: 200 von 200)" in text
    assert "in 41 von 400 Wiederholungen (10,2 %)" in text and "± 1,7 Prozentpunkte" in text and "R̂ < 1,01 in 41 dieser 41" in text


def test_formula_verdict_is_two_to_the_one_minus_m():
    for M, val in ((2, "0,5000"), (4, "0,1250"), (8, "0,0078")):
        kind, text = E.verdict_formula(M)
        assert f"= {val}" in text and f"2^(1 − {M})" in text
    assert "mindestens vier Ketten" in E.verdict_formula(2)[1] and "Erwartung für den Start" in E.verdict_formula(4)[1]
    assert E.same_side_probability(3) == 0.25


def test_what_helps_rows_have_three_columns_and_measured_numbers():
    rows = E.what_helps_rows()
    assert [r["Maßnahme"] for r in rows][:2] == ["Starts in beiden Gipfeln", "Längere Ketten"]
    assert all(set(r) == {"Maßnahme", "Befund", "Zahl"} for r in rows)
    assert any(r["Befund"] == "nicht gemessen" for r in rows)
    assert "200 von 200" in rows[0]["Zahl"]


def test_what_helps_longer_chains_row_names_both_multiples_and_the_198_of_200_at_the_ends():
    import dgn_reference as ref
    w = ref.window("gauss")
    by = {r["N"]: r for r in w["rows"]}
    assert by[50000]["alarm"] == 198 and by[100000]["alarm"] == 198 and by[10000]["alarm"] == 200 and by[20000]["alarm"] == 200
    row = E.what_helps_rows()[1]
    assert "Zwei- und Vierfachen" in row["Befund"] and "1,96 und 3,92" in row["Befund"] and "198 von 200" in row["Befund"]
    assert "Zweifachen von" not in row["Befund"]
    row3 = E.what_helps_rows()[2]
    assert row3["Maßnahme"].startswith("Heutige Regel") and "R̂-Schwelle allein ist nicht gemessen" in row3["Befund"]
