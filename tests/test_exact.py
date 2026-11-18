"""Exakte Größen: Überlebenswahrscheinlichkeit gegen Matrixpotenz, E[T] gegen lineares System, ESS gegen die Summe der Autokorrelationen aus Matrixpotenzen,
GTH gegen ein lineares System, Wechselrate gegen eine zweite Aufstellung der erweiterten Kette."""

import numpy as np
import pytest

import dgn_chain as ch
import dgn_exact as ex


@pytest.mark.parametrize("model_fn", [lambda: ch.get_grid(6.0, 1.0), lambda: ch.get_ring(8, 5.0)])
def test_survival_equals_matrix_power_and_mean_exit_equals_linear_solve(model_fn):
    m = model_fn()
    kl = ex.killed_from_A(m)
    for N in (1, 2, 10, 100, 400):
        assert kl.surv(N) == pytest.approx(kl.surv_matrix_power(N), abs=1e-9)
    assert kl.mean_exit() == pytest.approx(kl.mean_exit_solve(), rel=1e-8)


def test_survival_decreases_with_the_number_of_steps_and_is_one_before_the_shortest_path(ring4):
    kl = ex.killed_from_A(ring4)
    s = [kl.surv(N) for N in range(1, 60)]
    assert all(a >= b - 1e-15 for a, b in zip(s, s[1:]))
    assert s[0] == pytest.approx(1.0, abs=1e-12) and s[1] == pytest.approx(1.0, abs=1e-12)   # kürzester Weg A -> {0} -> leer -> {1}: 3 Schritte
    assert s[2] < 1.0


def test_exact_ess_equals_the_sum_of_autocorrelations_from_matrix_powers(ring4):
    P, pi, f = ring4.P, ring4.pi, ring4.value
    for N in (8, 50, 300):
        var = float((pi * f * f).sum() - (pi * f).sum() ** 2)
        rho = []
        Pk = np.eye(len(pi))
        for k in range(1, N):
            Pk = Pk @ P
            rho.append(float((pi * f * (Pk @ f)).sum() - (pi * f).sum() ** 2) / var)
        tau = 1 + 2 * sum((1 - k / N) * r for k, r in zip(range(1, N), rho))
        assert ex.exact_ess(P, pi, f, N)[0] == pytest.approx(4 * N / tau, rel=1e-9)


@pytest.mark.parametrize("lam", [-0.5, 0.3, 0.99, 1 - 1e-8, 1 - 1e-5])
def test_geometric_weighted_sum_equals_the_direct_sum(lam):
    N = 700
    k = np.arange(1, N)
    direct = float(((1 - k / N) * lam ** k).sum())
    assert ex._geometric_weighted_sum(np.array([lam]), N)[0] == pytest.approx(direct, rel=1e-8, abs=1e-8)


def test_gth_two_state_chain_and_agreement_with_linear_solve(ring4):
    a, b = 0.2, 0.05
    pi = ex.gth_stationary(np.array([[1 - a, a], [b, 1 - b]]))
    assert pi == pytest.approx([b / (a + b), a / (a + b)], rel=1e-14)
    P = ring4.P
    n = len(P)
    A = np.vstack([(np.eye(n) - P).T[:-1], np.ones(n)])
    rhs = np.zeros(n)
    rhs[-1] = 1
    assert ex.gth_stationary(P) == pytest.approx(np.linalg.solve(A, rhs), abs=1e-12)
    assert ex.gth_stationary(P) == pytest.approx(ring4.pi, abs=1e-12)


def _switch_rate_second_way(model):
    """Zweite Aufstellung der Wechselrate: Zustände (i, Marke) einzeln mit Schleifen, stationäre Verteilung über ein lineares System."""
    P, side, thr = model.P, model.side, model.thr
    K = len(side)
    inB, inA = side <= -thr + 1e-9, side >= thr - 1e-9
    nodes = [(i, "A") for i in range(K) if not inB[i]] + [(i, "B") for i in range(K) if not inA[i]]
    pos = {nd: j for j, nd in enumerate(nodes)}
    E = np.zeros((len(nodes), len(nodes)))
    flow = np.zeros(len(nodes))
    for (i, lab), j in pos.items():
        for t in range(K):
            if P[i, t] == 0:
                continue
            new = "B" if inB[t] else ("A" if inA[t] else lab)
            E[j, pos[(t, new)]] += P[i, t]
            if new != lab:
                flow[j] += P[i, t]
    n = len(nodes)
    A = np.vstack([(np.eye(n) - E).T[:-1], np.ones(n)])
    rhs = np.zeros(n)
    rhs[-1] = 1
    mu = np.linalg.solve(A, rhs)
    return float(mu @ flow)


@pytest.mark.parametrize("model_fn", [lambda: ch.get_ring(4, 10.0), lambda: ch.get_ring(6, 3.0), lambda: ch.get_grid(6.0, 2.0)])
def test_switch_rate_equals_second_computation(model_fn):
    m = model_fn()
    assert ex.switch_rate(m) == pytest.approx(_switch_rate_second_way(m), rel=1e-7)


def test_relaxation_time_of_the_hand_example(ring4):
    """Lücke 1 - lambda_2 = g/n mit g = 4 q^2 / ((1 + 2 q) + sqrt(1 + 4 q - 4 q^2)), q = 1/(1 + lambda): t_rel = 282,6 Schritte bei n = 4, lambda = 10."""
    q = 1 / 11
    g = 4 * q * q / ((1 + 2 * q) + (1 + 4 * q - 4 * q * q) ** 0.5)
    gap, trel = ex.relaxation(ring4.P, ring4.pi)
    assert gap == pytest.approx(g / 4, rel=1e-12)
    assert trel == pytest.approx(4 / g, rel=1e-9)
    assert round(trel, 1) == 282.6


def test_expected_exit_time_is_about_two_relaxation_times_for_the_hand_example(ring4):
    kl = ex.killed_from_A(ring4)
    assert kl.mean_exit() == pytest.approx(2761 / 5, rel=1e-12)


def test_grid_values_of_the_standard_case():
    """D = 8, sigma = 1 auf dem Gitter mit Abstand 0,1: Relaxationszeit 2559 (gerundet); D = 10: 91 306."""
    assert round(ex.relaxation(ch.get_grid(8, 1.0).P, ch.get_grid(8, 1.0).pi)[1]) == 2559
    assert round(ex.relaxation(ch.get_grid(10, 1.0).P, ch.get_grid(10, 1.0).pi)[1]) == 91306
