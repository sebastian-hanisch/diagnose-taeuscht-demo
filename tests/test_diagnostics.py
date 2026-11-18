"""Diagnostik: R-Dach und ESS gegen Handwerte, exakte AR(1)-ESS und die Formel der Rangnormalisierung; Randfälle (konstante Ketten, zu kurze Ketten)."""

import math

import numpy as np
import pytest
from scipy import stats

import _ref
import dgn_diagnostics as dg


def test_classic_rhat_hand_example():
    """Zwei Ketten (0,2,0,2) und (1,3,1,3): W = 4/3, B = 2, var+ = 1,5, R-Dach = sqrt(1,125)."""
    ch = np.array([[0.0, 2.0, 0.0, 2.0], [1.0, 3.0, 1.0, 3.0]])
    assert dg.rhat_classic(ch) == pytest.approx(math.sqrt(1.125), abs=1e-12)


def test_split_rhat_of_two_chains_in_the_same_peak_is_exactly_one():
    """(2,2,2,2) und (2,2,2,1), geteilt in (2,2),(2,2),(2,2),(2,1): W = B = 1/8, var+ = 1/8, R-Dach = 1 (von Hand)."""
    ch = np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 1.0]])
    assert dg.rhat_split(ch) == pytest.approx(1.0, abs=1e-12)


def test_constant_chains_are_reported_not_passed_on_as_one():
    same = np.full((2, 4), 2.0)
    assert math.isnan(dg.rhat_split(same))
    assert dg.diagnose(np.full((4, 40), 2.0))["undefined"] is True
    different = np.array([[1.0] * 4, [2.0] * 4])
    assert math.isinf(dg.rhat_split(different))


def test_too_short_chains_are_undefined():
    assert math.isnan(dg.rhat_split(np.arange(12.0).reshape(4, 3)))
    assert math.isnan(dg.ess_bulk(np.arange(12.0).reshape(4, 3)))


def test_one_chain_in_the_other_peak_gives_a_large_rhat():
    rng = np.random.default_rng(1)
    ch = rng.standard_normal((4, 400))
    ch[3] += 10
    assert dg.rhat_split(ch) > 1.5
    assert dg.rhat_split(rng.standard_normal((4, 400))) < 1.05


def test_rank_normalize_matches_the_formula():
    x = np.array([[3.0, 1.0, 2.0, 2.0], [5.0, 4.0, 6.0, 0.5]])
    r = stats.rankdata(x.ravel(), method="average")
    expect = stats.norm.ppf((r - 3 / 8) / (x.size + 1 / 4)).reshape(x.shape)
    assert np.allclose(dg.rank_normalize(x), expect, atol=1e-12)


@pytest.mark.parametrize("phi", [0.5, 0.9])
def test_ess_matches_exact_ar1_ess_in_a_band(phi):
    """Eichung an AR(1): Mittel der ESS-Schätzungen aus 60 Wiederholungen liegt innerhalb von 10 % der exakten ESS (Orakel: geschlossene Summe)."""
    N, M, reps = 1000, 4, 60
    rng = np.random.default_rng(2026)
    s = math.sqrt(1 - phi ** 2)
    ests = []
    for _ in range(reps):
        x = np.empty((M, N))
        x[:, 0] = rng.standard_normal(M)
        eps = rng.standard_normal((M, N))
        for t in range(1, N):
            x[:, t] = phi * x[:, t - 1] + s * eps[:, t]
        ests.append(dg.ess_multi(x))
    ratio = float(np.mean(ests)) / _ref.ar1_ess(phi, N, M)
    assert 0.9 < ratio < 1.1, ratio


def test_ess_of_independent_draws_is_close_to_the_number_of_draws():
    rng = np.random.default_rng(3)
    ess = [dg.ess_bulk(rng.standard_normal((4, 500))) for _ in range(20)]
    assert 0.8 < np.mean(ess) / 2000 < 1.3


def test_rhat_is_invariant_under_renumbering_shift_and_positive_scale():
    rng = np.random.default_rng(5)
    for _ in range(30):
        ch = rng.standard_normal((4, 60)) + rng.standard_normal((4, 1)) * rng.uniform(0, 2)
        base = dg.rhat_split(ch)
        assert dg.rhat_split(ch[rng.permutation(4)]) == pytest.approx(base, rel=1e-12)
        assert dg.rhat_split(ch + 17.5) == pytest.approx(base, rel=1e-9)
        assert dg.rhat_split(ch * 3.2) == pytest.approx(base, rel=1e-9)


def test_ess_does_not_depend_on_the_order_of_the_chains():
    rng = np.random.default_rng(6)
    for _ in range(10):
        ch = np.cumsum(rng.standard_normal((4, 200)), axis=1) * 0.1 + rng.standard_normal((4, 200))
        assert dg.ess_bulk(ch[[2, 0, 3, 1]]) == pytest.approx(dg.ess_bulk(ch), rel=1e-9)


def test_rank_rhat_is_invariant_under_monotone_transformations():
    rng = np.random.default_rng(7)
    ch = rng.standard_normal((4, 100)) + np.array([[0.0], [0.3], [0.0], [0.6]])
    assert dg.rhat_rank(np.exp(ch)) == pytest.approx(dg.rhat_rank(ch), rel=1e-9)


def test_today_rule_needs_rhat_ess_and_four_chains():
    rng = np.random.default_rng(8)
    ch = rng.standard_normal((4, 2000))
    d = dg.diagnose(ch)
    assert d["rhat_ok"] and d["ess"] >= 400 and d["today"]
    assert not dg.diagnose(ch[:2])["today"]                      # zwei Ketten genügen der heutigen Regel nicht
    short = dg.diagnose(rng.standard_normal((4, 60)))            # ESS unter 400 bei nur 240 Zügen
    assert not short["today"]


def test_ess_matches_the_independent_closed_form_for_a_white_noise_cap():
    """Bei weißem Rauschen darf tau nicht unter 1/log10(S) fallen (Obergrenze der ESS): ESS <= S * log10(S)."""
    rng = np.random.default_rng(9)
    ch = rng.standard_normal((4, 300))
    assert dg.ess_multi(ch) <= 1200 * math.log10(1200) + 1e-9
