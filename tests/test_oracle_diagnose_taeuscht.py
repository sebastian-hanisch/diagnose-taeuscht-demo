"""Orakel-Tests: die Kernrechnungen der Demo gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

(a) Überlebenswahrscheinlichkeit „kein Wechsel“: Eigenzerlegung gegen Matrixpotenz (tests/test_exact.py) und gegen den Anteil der simulierten Wiederholungen
    (Binomialband, vier Standardabweichungen), für vier Ketten p^4.
(b) Exakte ESS einer stationären Kette: gegen die Varianz des Gesamtmittels vieler stationär gestarteter Ketten (Var(Mittel) = Var(f) / ESS) und gegen die
    Summe der Autokorrelationen aus Matrixpotenzen (tests/test_exact.py) und die AR(1)-Formel (tests/test_diagnostics.py).
(c) Gitterziel: Symmetrie, Stationarität, Detailed Balance (tests/test_chain.py).
(d) Gipfelanteil: Mittel der Simulation gegen die exakte Erwartung aus der Übergangsmatrix.
(e) Handbeispiel in Brüchen gegen die Ring-Klasse der Demo (tests/test_handbeispiel.py).
Die Läufe sind klein (je unter 10 Sekunden); große Läufe mit mehr Wiederholungen gehören in den Scratch-Ordner."""

import math

import numpy as np
import pytest

import dgn_chain as ch
import dgn_exact as ex
import dgn_scenario as S


def _binomial_band(p_all, reps):
    return reps * p_all, 4 * max(math.sqrt(reps * p_all * (1 - p_all)), 1.0)


@pytest.mark.parametrize("model_fn,N,seed", [
    (lambda: ch.get_grid(8.0, 1.0), 1000, 11),
    (lambda: ch.get_grid(8.0, 1.0), 3000, 12),
    (lambda: ch.get_ring(8, 5.0), 300, 13),
    (lambda: ch.get_ring(10, 20.0), 1500, 14),
])
def test_count_of_runs_without_switch_matches_the_exact_expectation(model_fn, N, seed):
    """Vier Ketten, 200 Wiederholungen: die Zahl der Wiederholungen ohne Wechsel liegt in vier Standardabweichungen um 200 p^4."""
    m = model_fn()
    reps = 200
    path = ch.simulate(m, ("A",) * 4, N, seed, reps=reps)
    nosw = int((S.first_switch_all(m, path) > N).all(axis=1).sum())
    p = ex.killed_from_A(m).surv(N)
    mean, tol = _binomial_band(p ** 4, reps)
    assert abs(nosw - mean) <= tol, (nosw, mean, tol)


def test_exact_ess_equals_the_variance_of_the_pooled_mean_of_stationary_chains():
    """Ring n = 8, lambda = 1: pi ist gleichverteilt, der Start „gestreut“ (gleichverteilt) ist stationär. Var(Gesamtmittel von s) = Var_pi(s) / ESS."""
    m = ch.get_ring(8, 1.0)
    N, reps = 200, 500
    path = ch.simulate(m, ("S5",) * 4, N, 77, reps=reps)
    means = path.astype(float).reshape(reps, -1).mean(axis=1)
    var_f = float((m.pi * m.value ** 2).sum())
    ess = ex.exact_ess(m.P, m.pi, m.value, N)[0]
    assert abs(means.mean()) < 4 * math.sqrt(var_f / ess / reps)          # E_pi s = 0
    ratio = float(means.var(ddof=1)) / (var_f / ess)
    assert 0.75 < ratio < 1.25, ratio                                   # Standardfehler der Varianz bei 500 Wiederholungen: 6,3 %


def test_estimated_peak_share_matches_the_exact_expectation():
    """Ring n = 6, lambda = 10, Start A, 150 Schritte: Mittel des geschätzten Anteils (Gipfel B) über 600 Wiederholungen gleich der exakten Erwartung."""
    m = ch.get_ring(6, 10.0)
    N, reps = 150, 600
    d = m.start_A_distribution.copy()
    acc = 0.0
    for _ in range(N):
        d = d @ m.P
        acc += float(d @ m.indicator_B)
    exact = acc / N
    path = ch.simulate(m, ("A",) * 4, N, 5, reps=reps)
    share = m.to_indicator_B(path).reshape(reps, -1).mean(axis=1)
    se = share.std(ddof=1) / math.sqrt(reps)
    assert abs(share.mean() - exact) < 4 * se, (share.mean(), exact, se)


def test_grid_estimated_peak_share_matches_the_exact_expectation():
    m = ch.get_grid(6.0, 1.0)
    N, reps = 400, 400
    d = m.start_A_distribution.copy()
    acc = 0.0
    for _ in range(N):
        d = d @ m.P
        acc += float(d @ m.indicator_B)
    exact = acc / N
    path = ch.simulate(m, ("A",) * 4, N, 6, reps=reps)
    share = m.to_indicator_B(path).reshape(reps, -1).mean(axis=1)
    se = share.std(ddof=1) / math.sqrt(reps)
    assert abs(share.mean() - exact) < 4 * se, (share.mean(), exact, se)


def test_switch_count_matches_the_exact_rate_in_a_band():
    """Ring n = 6, lambda = 3: gezählte Wechsel in 200 Ketten à 5000 Schritten gegen die exakte Rate (GTH), Band 4 / sqrt(Wechsel)."""
    m = ch.get_ring(6, 3.0)
    path = ch.simulate(m, ("A",) * 4, 5000, 31, reps=50)
    sw = S.count_switches(m, path)
    assert sw["count"] > 1000
    assert abs(sw["rate"] / ex.switch_rate(m) - 1) < 4 / math.sqrt(sw["count"])
