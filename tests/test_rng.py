"""Zufallsquelle: SplitMix64 ist ganzzahlig und plattformfest; bekannte Ausgabe, Gleichverteilung, gleiche Zufalls-Seeds geben gleiche Folgen."""

import dgn_rng


def test_known_output_for_seed_zero():
    """Referenzwert der veröffentlichten SplitMix64-Fassung (Zustand 0): erste Ausgabe 0xE220A8397B1DCDAF."""
    r = dgn_rng.SplitMix64(0)
    assert r.next_u64() == 0xE220A8397B1DCDAF


def test_same_seed_same_sequence_and_different_seed_differs():
    a, b, c = dgn_rng.SplitMix64(7), dgn_rng.SplitMix64(7), dgn_rng.SplitMix64(8)
    sa = [a.random() for _ in range(20)]
    assert sa == [b.random() for _ in range(20)]
    assert sa != [c.random() for _ in range(20)]


def test_random_is_in_unit_interval_with_mean_one_half():
    r = dgn_rng.SplitMix64(123)
    xs = [r.random() for _ in range(20000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert abs(sum(xs) / len(xs) - 0.5) < 4 * (1 / 12 / 20000) ** 0.5


def test_below_covers_range():
    r = dgn_rng.SplitMix64(5)
    vals = {r.below(6) for _ in range(500)}
    assert vals == set(range(6))
