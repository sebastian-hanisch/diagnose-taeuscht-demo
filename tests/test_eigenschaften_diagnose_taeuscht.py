"""Eigenschaftstests: Aussagen, die für das Verfahren gelten müssen, ohne dass man das Ergebnis im Einzelnen kennt (ORACLE-PLAYBOOK §5.2).
Je Eigenschaft 20 bis 50 Instanzen mit festem Zufalls-Seed."""

import numpy as np
import pytest

import dgn_chain as ch
import dgn_diagnostics as dg
import dgn_evaluation as E
import dgn_exact as ex


def test_same_seed_gives_the_same_output_and_another_seed_a_different_one():
    s = E.Settings(N=500, seed=3)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.values, a2.values) and a1.facts == a2.facts
    assert not np.array_equal(a1.values, E.analyse(E.Settings(N=500, seed=4)).values)


def test_result_does_not_depend_on_the_order_of_independent_runs(grid_d8):
    reps = ch.simulate(grid_d8, ("A",) * 4, 200, 9, reps=6)
    reversed_order = [ch.simulate(grid_d8, ("A",) * 4, 200, 9, reps=1, rep0=r)[0] for r in (5, 4, 3, 2, 1, 0)]
    assert all(np.array_equal(reps[r], reversed_order[5 - r]) for r in range(6))


def test_survival_probability_falls_with_the_number_of_steps_for_many_models():
    for D in (6.0, 8.0, 10.0):
        for sig in (1.0, 1.7):
            kl = ex.killed_from_A(ch.get_grid(D, sig))
            s = [kl.surv(N) for N in (1, 10, 100, 1000, 10000)]
            assert all(a >= b - 1e-12 for a, b in zip(s, s[1:])), (D, sig)
    for n, lam in ((6, 2.0), (8, 10.0), (10, 5.0)):
        kl = ex.killed_from_A(ch.get_ring(n, lam))
        s = [kl.surv(N) for N in (1, 10, 100, 1000, 10000)]
        assert all(a >= b - 1e-12 for a, b in zip(s, s[1:])), (n, lam)


def test_longer_runs_never_make_a_switch_less_likely():
    """Gepaarte Läufe: der Pfad bis N ist ein Präfix des Pfades bis 2N; ein Wechsel bis N bleibt ein Wechsel bis 2N."""
    m = ch.get_grid(6.0, 1.0)
    short = ch.simulate(m, ("A",) * 4, 300, 21, reps=40)
    long_ = ch.simulate(m, ("A",) * 4, 600, 21, reps=40)
    assert np.array_equal(short, long_[:, :, :300])
    assert (ch.first_switch_from_A(m, short) <= 300).sum() <= (ch.first_switch_from_A(m, long_) <= 600).sum()


@pytest.mark.parametrize("vehicle,kw", [("gauss", dict(D=8, sigma=1.0)), ("ring", dict(ring_n=12, lam=50))])
def test_one_chain_in_the_other_peak_raises_rhat_in_every_one_of_40_runs(vehicle, kw):
    """Gaußmischung D = 8, sigma = 1 und Ring n = 12, lambda = 50 (N = 400): R-Dach >= 1,01 in 40 von 40 Wiederholungen mit Start 3 + 1."""
    model = E.model_of(E.Settings(vehicle="Hard-Core-Ring" if vehicle == "ring" else "Gaußmischung", **kw))
    path = ch.simulate(model, ("A", "A", "A", "B"), 400, 41, reps=40)
    bad = [r for r in range(40) if not dg.rhat_split(model.to_value(path[r])) >= 1.01]
    assert not bad, bad


def test_swapping_the_peaks_swaps_the_roles_for_the_ring():
    """Spiegelung: die Kette im Gipfel B hat dieselbe Wahrscheinlichkeit „kein Wechsel“ wie die im Gipfel A (Symmetrie s -> -s durch Drehen um einen Platz)."""
    m = ch.get_ring(8, 10.0)
    a_exit = ex.killed_from_A(m).mean_exit()
    d = np.zeros(m.n_states)
    d[m.state_index(m.ODD)] = 1.0
    b_exit = ex.Killed(m.P, m.pi, ex.absorbing_A(m), d).mean_exit()
    assert a_exit == pytest.approx(b_exit, rel=1e-9)


def test_peak_share_estimate_is_one_half_when_the_runs_start_in_both_peaks_and_mix(grid_d8):
    """Bei kurzer Schrittweite bleiben die Ketten stecken: Start 2 + 2 gibt den Anteil Gipfel B nahe 1/2 (exakt 1/2 in der Erwartung, Symmetrie)."""
    g = ch.get_grid(8.0, 1.0)
    path = ch.simulate(g, ("A", "A", "B", "B"), 300, 55, reps=60)
    share = g.to_indicator_B(path).reshape(60, -1).mean(axis=1)
    assert abs(share.mean() - 0.5) < 0.05
