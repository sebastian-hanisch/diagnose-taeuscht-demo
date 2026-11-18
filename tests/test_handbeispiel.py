"""Handbeispiel: Hard-Core-Ring mit n = 4 Plätzen, lambda = 10 (7 Zustände), alle Werte von Hand oder in Brüchen nachgerechnet und gegen die Einheiten der Demo geprüft.

Größe s = belegte gerade minus belegte ungerade Plätze; A-Seite s >= 1, B-Seite s <= -1 (Schwelle ceil(4/4) = 1). Das Ziel ist symmetrisch."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import _ref
import dgn_chain as ch
import dgn_diagnostics as dg
import dgn_exact as ex

STATES, PF, PIF = _ref.ring_fraction_matrix(4, 10)
IDX = {m: i for i, m in enumerate(STATES)}
EVEN, ODD = 0b0101, 0b1010
SS = [bin(m & EVEN).count("1") - bin(m & ODD).count("1") for m in STATES]


def _vec_mat(v, M):
    return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(len(M[0]))]


def test_states_and_target():
    assert len(STATES) == 7
    assert sum(PIF) == 1 and PIF[IDX[EVEN]] == F(100, 241) and PIF[IDX[ODD]] == F(100, 241)
    assert PIF[IDX[0]] == F(1, 241) and all(PIF[IDX[1 << v]] == F(10, 241) for v in range(4))        # Z = 1 + 4 * 10 + 2 * 100 = 241
    side_a = sum(PIF[i] for i in range(7) if SS[i] >= 1)
    side_b = sum(PIF[i] for i in range(7) if SS[i] <= -1)
    assert side_a == side_b == F(120, 241)


def test_hand_matrix_entries_and_balance():
    iA, i0, e0 = IDX[EVEN], IDX[0], IDX[1]
    assert PF[iA][IDX[1]] == F(1, 44) and PF[iA][IDX[4]] == F(1, 44) and PF[iA][iA] == F(21, 22)
    assert PF[e0][i0] == F(1, 44) and PF[e0][iA] == F(10, 44) and PF[e0][e0] == F(33, 44)
    assert all(PF[i0][IDX[1 << v]] == F(10, 44) for v in range(4)) and PF[i0][i0] == F(4, 44)
    assert all(sum(row) == 1 for row in PF)
    assert _vec_mat(PIF, PF) == PIF
    assert all(PIF[i] * PF[i][j] == PIF[j] * PF[j][i] for i in range(7) for j in range(7))


def test_float_ring_equals_the_fraction_matrix(ring4):
    assert [int(m) for m in ring4.S] == STATES
    for i in range(7):
        for j in range(7):
            assert ring4.P[i, j] == pytest.approx(float(PF[i][j]), abs=1e-15)
    assert ring4.pi == pytest.approx([float(x) for x in PIF], abs=1e-15)


def test_mean_time_to_the_other_side_by_hand_and_exactly(ring4):
    """Von Hand: h_A = 22 + h_E; 11 h_E = 44 + 10 h_A + h_0; 10 h_0 = 11 + 5 h_E; daraus h_A = 2761/5 = 552,2."""
    live = [i for i in range(7) if SS[i] > -1]
    A = [[(F(1) if a == b else F(0)) - PF[live[a]][live[b]] for b in range(len(live))] for a in range(len(live))]
    h = dict(zip(live, _ref.solve_fraction(A, [F(1)] * len(live))))
    hA, hE, h0 = h[IDX[EVEN]], h[IDX[1]], h[IDX[0]]
    assert hA == F(2761, 5) and hE == F(2651, 5) and h0 == F(1331, 5)
    assert hA == 22 + hE and 11 * hE == 44 + 10 * hA + h0 and 10 * h0 == 11 + 5 * hE
    assert ex.killed_from_A(ring4).mean_exit() == pytest.approx(float(hA), rel=1e-12)


def test_probability_of_no_switch(ring4):
    """In 1 und 2 Schritten unmöglich; in 3 Schritten 4 * (1/44) (1/44) (10/44) = 5/10 648; vier Ketten: p^4 = 0,4981 bei 100 Schritten."""
    kl = ex.killed_from_A(ring4)
    assert kl.surv(1) == pytest.approx(1.0) and kl.surv(2) == pytest.approx(1.0)
    assert 1 - kl.surv(3) == pytest.approx(5 / 10648, rel=1e-9)
    assert round(kl.surv(100), 4) == 0.8401 and round(kl.surv(300), 4) == 0.5828 and round(kl.surv(1000), 4) == 0.1621
    assert round(kl.surv(100) ** 4, 4) == 0.4981 and round(kl.surv(1000) ** 4, 4) == 0.0007
    assert abs(kl.surv(300) - math.exp(-300 / (2761 / 5))) < 0.02            # der Wechsel ist nahezu exponentialverteilt


def test_expected_peak_share_after_n_steps(ring4):
    """Erwarteter Anteil des Startgipfels A aus den ersten N Schritten vom Start A: 1 (N = 1), 0,99495 (10), 0,6384 (1000); Anteil Gipfel B = 1 minus dieser Wert."""
    I = ring4.indicator_B
    d = np.zeros(7)
    d[ring4.state_index(ring4.EVEN)] = 1.0
    acc, got = 0.0, {}
    for t in range(1, 1001):
        d = d @ ring4.P
        acc += float(d @ (1 - I))             # Anteil A = 1 - Anteil B
        if t in (1, 10, 1000):
            got[t] = acc / t
    assert got[1] == pytest.approx(1.0) and round(got[10], 5) == 0.99495 and round(got[1000], 4) == 0.6384
    assert float(ring4.pi @ ring4.indicator_B) == pytest.approx(0.5, abs=1e-15)    # E_pi I_B = 1/2 (Symmetrie)


def test_exact_ess_of_stationary_chains_from_fractions_and_spectral_sum(ring4):
    """tau_8 = 7,80943 und ESS = 4,098 bei vier Ketten à 8 Schritte (Brüche gleich Spektralzerlegung); 6,9 / 17,0 / 60,5 bei vier Ketten à 500 / 2000 / 8000 Schritte."""
    f = [F(x) for x in SS]
    var = sum(PIF[i] * f[i] ** 2 for i in range(7))
    assert var == F(840, 241)
    N, g, rho = 8, f[:], {}
    for k in range(1, N):
        g = [sum(PF[i][j] * g[j] for j in range(7)) for i in range(7)]
        rho[k] = sum(PIF[i] * f[i] * g[i] for i in range(7)) / var
    tau = 1 + 2 * sum((1 - F(k, N)) * rho[k] for k in range(1, N))
    assert float(tau) == pytest.approx(7.80943, abs=1e-5)
    ess, tau_spec = ex.exact_ess(ring4.P, ring4.pi, ring4.value, N)
    assert tau_spec == pytest.approx(float(tau), abs=1e-10) and round(ess, 3) == 4.098
    assert [round(ex.exact_ess(ring4.P, ring4.pi, ring4.value, n)[0], 1) for n in (500, 2000, 8000)] == [6.9, 17.0, 60.5]


def test_rhat_of_two_chains_in_the_same_peak_by_hand():
    ch2 = np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 1.0]])
    assert dg.rhat_split(ch2) == pytest.approx(1.0, abs=1e-12)          # W = B = var+ = 1/8
    assert (ch2 > 0).mean() == 1.0                                       # alle Werte auf der A-Seite, geschätzter Gipfelanteil 1 statt 120/241


def test_exact_distribution_of_the_rhat_verdict_for_two_short_chains():
    """Zwei Ketten à 4 Schritte aus A, geteilt in vier Hälften zu 2: R-Dach nicht definiert (0/0) mit Wahrscheinlichkeit 0,6896, R-Dach < 1,01 mit 0,1353.
    Aufzählung aller Werte-Folgen in Ganzzahlen (Nenner 44^4 je Kette), die Bedingung 10000 (4 sum S^2 - (sum S)^2) < 31206 sum D^2 gegen die Funktion geprüft."""
    P44 = [[int(PF[a][b] * 44) for b in range(7)] for a in range(7)]
    paths = {(IDX[EVEN], ()): 1}
    for _ in range(4):
        nxt = {}
        for (st, seq), num in paths.items():
            for b in range(7):
                if P44[st][b]:
                    key = (b, seq + (SS[b],))
                    nxt[key] = nxt.get(key, 0) + num * P44[st][b]
        paths = nxt
    seqs = {}
    for (st, seq), num in paths.items():
        seqs[seq] = seqs.get(seq, 0) + num
    keys = list(seqs)
    assert len(keys) == 35
    nums = np.array([seqs[k] for k in keys], dtype=np.int64)
    arr = np.array(keys, dtype=np.int64)
    S1, S2 = arr[:, 0] + arr[:, 1], arr[:, 2] + arr[:, 3]
    D1, D2 = (arr[:, 0] - arr[:, 1]) ** 2, (arr[:, 2] - arr[:, 3]) ** 2
    sumS = S1[:, None] + S2[:, None] + S1[None, :] + S2[None, :]
    sumS2 = S1[:, None] ** 2 + S2[:, None] ** 2 + S1[None, :] ** 2 + S2[None, :] ** 2
    sumD = D1[:, None] + D2[:, None] + D1[None, :] + D2[None, :]
    pr = nums[:, None] * nums[None, :]
    lhs, rhs = 10000 * (4 * sumS2 - sumS ** 2), 31206 * sumD
    undefined = (sumD == 0) & (lhs == 0)
    ok = (sumD > 0) & (lhs < rhs)
    tot = 44 ** 8
    assert round(int(pr[undefined].sum()) / tot, 4) == 0.6896
    assert round(int(pr[ok].sum()) / tot, 4) == 0.1353
    rng = np.random.default_rng(8)
    for _ in range(600):
        i, j = int(rng.integers(35)), int(rng.integers(35))
        r = dg.rhat_split(np.array([arr[i], arr[j]], dtype=float))
        if math.isnan(r):
            assert undefined[i, j]
        elif not math.isinf(r):
            assert ok[i, j] == (r < 1.01)
