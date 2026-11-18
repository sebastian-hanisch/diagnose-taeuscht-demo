"""Ketten: die vektorisierte Simulation gibt bitgleich dieselben Pfade wie eine rein sequentielle Fassung über SplitMix64 (Orakel für den Zufallsstrom),
Zielverteilungen, Übergangsmatrizen und Startarten."""

import numpy as np
import pytest

import _ref
import dgn_chain as ch
import dgn_rng


def test_numpy_state_derivation_equals_integer_derivation():
    st = ch.stream_states(18101, 3, 4, rep0=2)
    for r in range(3):
        for c in range(4):
            assert int(st[r * 4 + c]) == _ref.derive_state(18101, 2 + r, c)


def test_draws_equal_sequential_splitmix_draws():
    st = np.array([123456789, 987654321], dtype=np.uint64)
    d = ch.draws(st, 5, 7)
    for j, s in enumerate(st):
        rng = dgn_rng.SplitMix64(int(s))
        seq = [rng.next_u64() for _ in range(12)]
        assert [int(x) for x in d[:, j]] == seq[5:12]


@pytest.mark.parametrize("kind", ["A", "B", "S10"])
def test_grid_batch_equals_scalar_reference(grid_d8, kind):
    N = 1500   # mehr als ein Block von 1000 Schritten
    path = ch.simulate(grid_d8, (kind, "A", "B"), N, 4242, reps=2)
    for rep in range(2):
        for c, k in enumerate((kind, "A", "B")):
            ref = _ref.scalar_grid_path(grid_d8, k, _ref.derive_state(4242, rep, c), N)
            assert np.array_equal(path[rep, c], ref), (rep, c)


@pytest.mark.parametrize("kind", ["A", "B", "S5"])
def test_ring_batch_equals_scalar_reference(ring12, kind):
    N = 1500
    path = ch.simulate(ring12, (kind, "A"), N, 99, reps=2)
    for rep in range(2):
        for c, k in enumerate((kind, "A")):
            ref = _ref.scalar_ring_path(ring12, k, _ref.derive_state(99, rep, c), N)
            assert np.array_equal(path[rep, c], ref), (rep, c)


def test_a_repetition_does_not_depend_on_the_other_repetitions(grid_d8):
    """Je Kette ein eigener Strom: Wiederholung 3 allein (rep0 = 3) gleicht Wiederholung 3 eines Laufs mit fünf Wiederholungen."""
    big = ch.simulate(grid_d8, ("A",) * 4, 300, 7, reps=5)
    one = ch.simulate(grid_d8, ("A",) * 4, 300, 7, reps=1, rep0=3)
    assert np.array_equal(big[3], one[0])


def test_grid_target_is_symmetric_normalised_and_stationary(grid_d8):
    g = grid_d8
    assert abs(g.pi.sum() - 1) < 1e-12
    assert abs(g.pi[g.x > 1e-9].sum() - g.pi[g.x < -1e-9].sum()) < 1e-12
    assert np.abs(g.P.sum(1) - 1).max() < 1e-12
    assert np.abs(g.pi @ g.P - g.pi).max() < 1e-12
    flow = g.pi[:, None] * g.P
    assert np.abs(flow - flow.T).max() < 1e-15      # Detailed Balance


def test_grid_size_and_threshold():
    g = ch.get_grid(10, 1.0)
    assert g.K == 241 and g.kmax == 50
    assert ch.get_grid(10, 2.4).kmax == 120          # 5 * 2,4 / 0,1 = 120 (nicht 121 durch Rundung)
    assert g.thr == 2.5
    assert abs(g.x[0] + 12.0) < 1e-12 and abs(g.x[-1] - 12.0) < 1e-12


def test_ring_states_are_lucas_numbers_and_stationary(ring4):
    assert ring4.n_states == 7
    assert [ch.get_ring(n, 1.0).n_states for n in (4, 6, 8, 10, 12)] == [7, 18, 47, 123, 322]   # Lucas-Zahlen L_n
    assert np.abs(ring4.P.sum(1) - 1).max() < 1e-14
    assert np.abs(ring4.pi @ ring4.P - ring4.pi).max() < 1e-14
    flow = ring4.pi[:, None] * ring4.P
    assert np.abs(flow - flow.T).max() < 1e-15


def test_ring_peak_states_and_threshold(ring4):
    assert ring4.s_state[ring4.state_index(ring4.EVEN)] == 2 and ring4.s_state[ring4.state_index(ring4.ODD)] == -2
    assert ring4.thr == 1 and ch.get_ring(12, 50.0).thr == 3 and ch.get_ring(16, 100.0).thr == 4


def test_ring_rejects_odd_n():
    with pytest.raises(ValueError):
        ch.Ring(5, 2.0)


def test_start_cdf_is_a_distribution_and_peaks_are_where_expected(grid_d8):
    for kind in ("A", "B", "S5", "S10"):
        c = grid_d8.start_cdf(kind)
        assert c[-1] == 1.0 and np.all(np.diff(c) >= 0)
    assert grid_d8.start_distribution("A")[grid_d8.x < 0].sum() > 0.9999
    assert grid_d8.start_distribution("B")[grid_d8.x > 0].sum() > 0.9999


def test_scattered_start_is_symmetric(grid_d8):
    d = grid_d8.start_distribution("S10")
    assert abs(d[grid_d8.x > 1e-9].sum() - d[grid_d8.x < -1e-9].sum()) < 1e-12


def test_start_patterns():
    assert ch.start_pattern("alle_A") == ("A",) * 4
    assert ch.start_pattern("eine_B") == ("A", "A", "A", "B")
    assert ch.start_pattern("zwei_zwei") == ("A", "A", "B", "B")
    assert ch.start_pattern("gestreut") == ("S10",) * 4


def test_first_true():
    m = np.array([[False, False, True, True], [False, False, False, False]])
    assert list(ch.first_true(m)) == [3, 5]
