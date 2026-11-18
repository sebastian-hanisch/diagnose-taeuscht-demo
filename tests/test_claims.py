"""Claims: jede Zahl der README steht hier und wird gegen die eingefrorenen Läufe, die exakte Rechnung oder das Handbeispiel gehalten.

Je Test wird die Zahl aus den Daten berechnet, im selben Format wie in der README geschrieben und in der README gesucht; stimmen Daten und Text nicht überein, schlägt der Test fehl.
Zählungen sind Zählungen aus den eingefrorenen Läufen (feste Zufalls-Seeds, 200 Wiederholungen); `tests/test_tables.py` rechnet einzelne Zellen neu. Exakte Größen: relativ 1e-6."""

import re
from pathlib import Path

import pytest

import dgn_chain as ch
import dgn_evaluation as E
import dgn_exact as ex
import dgn_reference as R
from dgn_format import dez, ganz, potenz

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
T, X = R.tables(), R.exact()


def cell(cid):
    return T["cells"][cid]["counts"], X["cells"][cid]


def slash(vals):
    return " / ".join(str(v) for v in vals)


def present(s):
    assert s in README, f"nicht in der README: {s!r}"


G10 = ["gauss_D10_s1_18101", "gauss_D10_s1_18102", "gauss_D10_s1_18103"]


def test_c1_deception_in_the_gaussian_mixture_three_seeds():
    cs = [cell(c)[0] for c in G10]
    e = cell(G10[0])[1]
    present(f"täuschend **{slash(c['deceiving'] for c in cs)}** von je 200")
    present(f"kein Wechsel {slash(c['no_switch'] for c in cs)}, exakt erwartet {dez(e['expected'], 1)} ± {dez(e['sd'], 1)}")
    present(f"heute {slash(c['today'] for c in cs)}; rangnormalisiertes R̂ < 1,01 in {slash(c['rhat_rank_lt'] for c in cs)}")
    present(f"heute {slash(c['today'] for c in cs)} gegen täuschend {slash(c['deceiving'] for c in cs)}; mit rangnormalisiertem R̂ täuschend {slash(c['deceiving_rank'] for c in cs)}")
    present(f"in **{slash(c['deceiving'] for c in cs)} von je 200**")
    present(f"ändert daran nichts ({slash(c['today'] for c in cs)})")
    assert all(c["reps"] == 200 for c in cs) and all(c["today"] == c["deceiving"] for c in cs)
    assert 186 <= min(c["deceiving"] for c in cs) and max(c["deceiving"] for c in cs) <= 194      # Band der drei Zufalls-Seeds


def test_c2_ess_and_mse_ess():
    cs = [cell(c)[0] for c in G10]
    e = cell(G10[0])[1]
    present(f"angezeigte ESS_bulk im Mittel {slash(ganz(c['ess_mean']) for c in cs)}, exakt für eine stationäre Kette gleicher Länge {dez(e['ess_stationary'], 1)}")
    present(f"N_eff,MSE {slash(dez(c['n_eff_mse'], 2) for c in cs)} aus 8000 Zügen")
    present(f"im Mittel {slash(dez(c['share_B_mean'], 3) for c in cs)} statt 0,5")
    present("(drei Zufalls-Seeds) „R̂ < 1,01“")
    present("(1,01 / 1,03 / 1,00)")
    assert all(0.5 <= c["n_eff_mse"] <= 1.1 for c in cs) and e["ess_stationary"] == pytest.approx(4.19, abs=0.005)
    assert all(c["share_B_mean"] < 0.01 for c in cs)


def test_c3_distance_and_step_width():
    c, e = cell("gauss_D8_s1_18110")
    present(f"kein Wechsel {c['no_switch']} (exakt erwartet {dez(e['expected'], 1)} ± {dez(e['sd'], 1)}), täuschend {c['deceiving']}, ESS {ganz(c['ess_mean'])} gegen exakt {dez(e['ess_stationary'], 1)}, N_eff,MSE {dez(c['n_eff_mse'], 2)}")
    for cid in ("gauss_D6_s1_18111", "gauss_D10_s2.4_18112", "gauss_D8_s2.4_18113"):
        cc, ee = cell(cid)
        assert cc["deceiving"] == 0 and cc["no_switch"] == 0 and ee["expected"] < 0.1
    present("täuschend 0 von je 200 (exakte Erwartung für „kein Wechsel“ unter 0,1 von 200")
    present(f"bei D = 8, σ = 2,4 meldet R̂ < 1,01 trotz Wechseln in {cell('gauss_D8_s2.4_18113')[0]['rhat_lt']} von 200")


def test_c4_ring():
    r12 = ("ring_n12_l50_18201", "ring_n12_l50_18202")
    r16 = ("ring_n16_l100_18204", "ring_n16_l100_18205")
    for pair, bold in ((r12, "**"), (r16, "")):
        cs = [cell(c)[0] for c in pair]
        e = cell(pair[0])[1]
        present(f"täuschend {bold}{slash(c['deceiving'] for c in cs)}{bold}, heute {slash(c['today'] for c in cs)}, ESS {slash(ganz(c['ess_mean']) for c in cs)} gegen exakt {dez(e['ess_stationary'], 2)}, "
                f"kein Wechsel {slash(c['no_switch'] for c in cs)} (exakt erwartet {dez(e['expected'], 1)}" + (f" ± {dez(e['sd'], 1)})" if bold else ")")
                + (f", N_eff,MSE {slash(dez(c['n_eff_mse'], 2) for c in cs)}." if bold else ""))
    ids = r12 + r16
    all_dec = [cell(c)[0]["deceiving"] for c in ids]
    today = [cell(c)[0]["today"] for c in ids]
    ess = [cell(c)[0]["ess_mean"] for c in ids]
    present(f"{min(all_dec)} bis {max(all_dec)} von 200")
    present(f"({min(today)} bis {max(today)} von 200 statt")
    present(f"ESS {round(min(ess))} bis {round(max(ess))} gegen 4,0 bis 4,2 exakt")
    rhat_ge = [cell(c)[0]["rhat_ge"] for c in ids]
    present(f"in {min(rhat_ge)} bis {max(rhat_ge)} von 200 Wiederholungen anschlägt")      # R̂ ≥ 1,01 allein, auch ohne Wechsel
    assert max(today) < min(all_dec)               # die heutige Regel schlägt im Ring seltener an als R̂ allein
    assert 4.0 <= min(cell(c)[1]["ess_stationary"] for c in ids) and max(cell(c)[1]["ess_stationary"] for c in ids) < 4.25


def test_c5_controls():
    c, e = cell("gauss_D0_s1_18114")
    present(f"R̂ < 1,01 in {c['rhat_lt']} von 200, ESS_bulk {ganz(c['ess_mean'])} gegen exakt {ganz(e['ess_stationary'])}")
    c, e = cell("ring_n16_l1_18206")
    present(f"R̂ < 1,01 nur in {c['rhat_lt']} von 200, ESS {ganz(c['ess_mean'])} gegen exakt {ganz(e['ess_stationary'])}")
    assert "deceiving" in c and c["no_switch"] == 0


def test_c6_windows():
    w = R.window("gauss")
    rows = w["rows"]
    present(f"E[T] = {ganz(w['E_T'])}, Zufalls-Seed {w['spec']['seed']}")
    peak_n = max(rows, key=lambda r: r["deceiving"])["N"]
    present("täuschend " + " / ".join(f"**{r['deceiving']}**" if r["N"] == peak_n else str(r["deceiving"]) for r in rows))
    present("bei N = " + " / ".join(ganz(r["N"]) for r in rows))
    by = {r["N"]: r for r in rows}
    assert by[10000]["alarm"] == 200 and by[20000]["alarm"] == 200 and by[10000]["N"] / w["E_T"] == pytest.approx(1.96, abs=0.005) and by[20000]["N"] / w["E_T"] == pytest.approx(3.92, abs=0.005)
    present(f"der geschätzte Anteil Gipfel B ist {dez(by[10000]['share_B_mean'], 3)} und {dez(by[20000]['share_B_mean'], 3)}")
    present(f"bei N = 100 000 {dez(by[100000]['share_B_mean'], 3)} und R̂ < 1,01 in {by[100000]['rhat_lt']} von 200")
    peak = max(rows, key=lambda r: r["deceiving"])
    assert peak["N"] == 1000 and peak["N"] / w["E_T"] == pytest.approx(0.196, abs=0.001)
    assert by[100]["alarm"] == 198 and by[100000]["alarm"] == 198
    present("(Fenster: 198 von 200 bei N = 100, 198 von 200 bei N = 100 000)")
    wr = R.window("ring")
    by = {r["N"]: r for r in wr["rows"]}
    peak = max(wr["rows"], key=lambda r: r["deceiving"])
    present(f"E[T] = {ganz(wr['E_T'])}, 18302")
    present(f"größte Zahl täuschend {peak['deceiving']} von 200 bei N = {ganz(peak['N'])}")
    present(f"Anteil Gipfel B {dez(by[2000]['share_B_mean'], 3)} bei N = 2000 und {dez(by[50000]['share_B_mean'], 3)} bei N = 50 000")
    assert wr["spec"]["seed"] == 18302 and w["spec"]["seed"] == 18301


def test_c7_switch_rate():
    w, wr, rc = R.window("gauss"), R.window("ring"), R.rate_cell()
    sw = w["switches"]
    ratio = sw["rate"] / w["rate_exact"]
    band = 4 / sw["count"] ** 0.5
    present(f"{potenz(w['rate_exact'], 3).replace('1,96', '1,96')} je Schritt (ein Wechsel alle {ganz(1 / w['rate_exact'])} Schritte); gezählt {ganz(sw['count'])} Wechsel in {ganz(sw['steps'])} Schritten, Verhältnis {dez(ratio, 3)} (Band 4/√Wechsel = {dez(band, 3)})")
    s2 = rc["switches"]
    present(f"4,02·10⁻² je Schritt, gezählt {ganz(s2['count'])} Wechsel in {ganz(s2['steps'])} Schritten, Verhältnis {dez(s2['rate'] / rc['exact']['rate_exact'], 3)}")
    assert potenz(rc["exact"]["rate_exact"], 3) == "4,02·10⁻²"
    s3 = wr["switches"]
    present(f"{potenz(wr['rate_exact'], 3)}, gezählt {ganz(s3['count'])} in {ganz(s3['steps'])}, Verhältnis {dez(s3['rate'] / wr['rate_exact'], 3)}")
    assert abs(ratio - 1) <= band
    present("1/Rate = 5090")
    assert round(1 / w["rate_exact"]) == 5090


def test_c8_starts():
    rows = {(r["name"], r["burn"]): r for r in R.starts("gauss")}
    ring = {(r["name"], r["burn"]): r for r in R.starts("ring")}
    assert rows[("drei in A, eine in B", 0)]["rhat_ge"] == 200 and rows[("zwei in A, zwei in B", 0)]["rhat_ge"] == 200
    assert ring[("drei in A, eine in B", 0)]["rhat_ge"] == 200 and ring[("zwei in A, zwei in B", 0)]["rhat_ge"] == 200
    present("R̂ ≥ 1,01 in **200 von 200** (Ring n = 12, λ = 50: ebenso 200 von 200)")
    s5, s10 = rows[("gestreut, sd 5", 500)], rows[("gestreut, sd 10", 500)]
    tot = s5["same_side"] + s10["same_side"]
    present(f"in {s5['same_side']} und {s10['same_side']} von je 200 ({tot} von 400, erwartet 2^(1−4) = 12,5 % ± 1,7 Prozentpunkte Standardfehler gegen gemessene {dez(100 * tot / 400, 1)} %)")
    present(f"R̂ < 1,01 in {s5['same_and_rhat_lt'] + s10['same_and_rhat_lt']} dieser {tot}")
    present(f"{tot} von 400 ({dez(100 * tot / 400, 1)} %) statt 48 von 400 (12,0 %)")
    se = (0.125 * 0.875 / 400) ** 0.5
    assert 1.1 < (0.125 - tot / 400) / se < 1.3 and round(se * 100, 1) == 1.7
    present("rund 1,2 Standardfehler darunter")
    present("Gemessen ist nur: Starts in beiden Gipfeln lösen den Alarm aus (200 von 200 in den vier Zellen)")


def test_c9_number_of_chains():
    m = {r["chains"]: r for r in R.mcells()}
    present(f"selbe Seite in {m[2]['same_side']} von 1000 bei M = 2 (Erwartung 2^(1−2) = 0,5), {m[4]['same_side']} bei M = 4 (0,125) und {m[8]['same_side']} bei M = 8 (2^(1−8) = 0,0078)")
    present(f"({m[8]['same_side']} von 1000, Formel 7,8), ebenso M = 2 ({m[2]['same_side']} von 1000, Formel 500)")
    assert all(r["reps"] == 1000 for r in m.values())
    for M, r in m.items():
        p = 2.0 ** (1 - M)
        assert abs(r["same_side"] - 1000 * p) <= 4 * (1000 * p * (1 - p)) ** 0.5, M           # Binomialband um die Formel
    assert E.same_side_probability(8) == 2 ** -7 and dez(E.same_side_probability(8), 4) == "0,0078"


def test_c10_exact_quantities():
    pairs = (("gauss_D8_s1_18110", "D = 8, σ = 1: 2559 und 5107"), ("gauss_D10_s1_18101", "D = 10, σ = 1: 91 306 und 182 560"),
             ("ring_n12_l50_18201", "Ring n = 12, λ = 50: 16 206 und 26 920"), ("ring_n12_l20_18203", "n = 12, λ = 20: 2887 und 5369"),
             ("ring_n16_l100_18204", "n = 16, λ = 100: 83 208 und 138 287"))
    for cid, text in pairs:
        e = X["cells"][cid]
        assert f"{ganz(e['t_rel'])} und {ganz(e['E_T'])}" in text
        present(text)
    assert ch.get_grid(8, 1.0).K == 221      # Voraussetzung: dasselbe Gitter wie die Zelle


def test_c11_hand_example(ring4):
    kl = ex.killed_from_A(ring4)
    gap, trel = ex.relaxation(ring4.P, ring4.pi)
    assert ring4.n_states == 7 and 1 + 4 * 10 + 2 * 100 == 241
    assert kl.mean_exit() == pytest.approx(2761 / 5, rel=1e-12) and 2761 / 5 == 552.2
    assert 1 - kl.surv(3) == pytest.approx(5 / 10648, rel=1e-9)
    assert dez(kl.surv(100) ** 4, 4) == "0,4981" and dez(kl.surv(1000) ** 4, 4) == "0,0007"
    assert dez(trel, 1) == "282,6"
    ess, tau = ex.exact_ess(ring4.P, ring4.pi, ring4.value, 8)
    assert dez(tau, 5) == "7,80943" and dez(ess, 3) == "4,098"
    for s in ("7 Belegungen, Z = 241", "E[T] = 2761/5 = 552,2 Schritte", "P(Wechsel) = 5/10 648", "mit Wahrscheinlichkeit 0,4981", "in 1000 Schritten mit 0,0007",
              "erwarteter Anteil des Startgipfels A aus 1000 Schritten 0,6384", "t_rel = 282,6", "τ₈ = 7,80943 und ESS 4,098"):
        present(s)
    present("erwarteter Anteil des Startgipfels A aus 1000 Schritten 0,6384 (Anteil Gipfel B also 0,3616 statt 0,5)")
    # Anteil des Startgipfels A (nicht Gipfel B): Erwartung des geschätzten Anteils aus 1000 Schritten
    d = ring4.start_A_distribution.copy()
    acc = 0.0
    for _ in range(1000):
        d = d @ ring4.P
        acc += float(d @ (1 - ring4.indicator_B))
    assert dez(acc / 1000, 4) == "0,6384" and dez(1 - acc / 1000, 4) == "0,3616"     # Anteil Gipfel B = 1 - Anteil A


def test_c12_oracle_rows():
    n = 0
    for cid, rec in T["cells"].items():
        if "no_switch" in rec["counts"]:
            e = X["cells"][cid]
            assert abs(rec["counts"]["no_switch"] - e["expected"]) <= 4 * max(e["sd"], 1.0)
            n += 1
    for wid, w in T["windows"].items():
        for r, e in zip(w["rows"], X["windows"][wid]["rows"]):
            assert abs(r["no_switch"] - e["expected"]) <= 4 * max(e["sd"], 1.0)
            n += 1
    assert n == 32
    present("in allen 32 Zeilen (13 Zellen, 19 Fensterzeilen)")


def test_every_number_group_in_the_table_rows_names_a_test_that_exists():
    tests_in_file = set(re.findall(r"^def (test_c\d+)_", Path(__file__).read_text(encoding="utf-8"), flags=re.M))
    named = set(re.findall(r"`(test_c\d+)`", README))
    assert named and named <= tests_in_file, named - tests_in_file


def test_the_model_disclaimer_is_the_same_in_app_and_readme():
    import dgn_texts as T_
    assert T_.MODEL_DISCLAIMER in " ".join(README.split())


def test_c1_rank_normalised_rhat_changes_one_of_three_counts_by_one():
    dec = [cell(c)[0]["deceiving"] for c in G10]
    rank = [cell(c)[0]["rhat_rank_lt"] for c in G10]
    assert [r - d for r, d in zip(rank, dec)] == [0, 0, 1]
    present("(im dritten Lauf R̂ < 1,01 in 195 statt 194 von 200, täuschend weiterhin 194)")
