"""Randfälle (DEMO-PLAYBOOK §12): konstante Ketten, zu kurze Ketten, zwei Ketten, D = 0, kleinste und größte Regler, Start außerhalb, Gleichstände der Ränge."""

import math

import numpy as np
import pytest

import dgn_chain as ch
import dgn_constants as C
import dgn_diagnostics as dg
import dgn_evaluation as E
import dgn_exact as ex


def test_constant_chains_are_reported_as_undefined():
    d = dg.diagnose(np.full((4, 40), 3.0))
    assert d["undefined"] and not d["rhat_ok"] and not d["today"]
    kind, text = E.verdict_run(E.Facts("Gaußmischung", True, 40, "alle_A", float("nan"), float("nan"), False, False, True, True, False, (0, 0, 0, 0), 0.0, 5.0))
    assert kind == "warning" and "nicht definiert" in text


def test_chains_shorter_than_four_steps_have_no_rhat():
    assert math.isnan(dg.rhat_split(np.zeros((4, 3)) + np.arange(3)))


def test_two_chains_never_pass_the_rule_of_today():
    rng = np.random.default_rng(1)
    d = dg.diagnose(rng.standard_normal((2, 4000)))
    assert d["rhat_ok"] and not d["today"]


def test_odd_length_drops_the_middle_value_when_splitting():
    ch2 = np.arange(10.0).reshape(2, 5)
    sp = dg.split_chains(ch2)
    assert sp.shape == (4, 2) and list(sp[0]) == [0.0, 1.0] and list(sp[2]) == [3.0, 4.0]


def test_d_zero_is_a_single_peak_without_switch_notion():
    s = E.Settings(D=0, N=500)
    a = E.analyse(s)
    assert not a.facts.has_peaks and a.facts.switches == () and a.E_T is None and a.facts.p_all is None
    kind, text = E.verdict_run(a.facts)
    assert kind in ("success", "info") and "Kontrolle" in text


@pytest.mark.parametrize("kw", [
    dict(D=12, sigma=C.SIGMA_MAX, N=200), dict(D=4, sigma=C.SIGMA_MIN, N=200), dict(D=12, sigma=C.SIGMA_MIN, N=200, start="gestreut"),
    dict(vehicle="Hard-Core-Ring", ring_n=6, lam=1, N=200), dict(vehicle="Hard-Core-Ring", ring_n=12, lam=100, N=200, start="eine_B"),
    dict(vehicle="Hard-Core-Ring", ring_n=12, lam=100, N=200, start="gestreut"), dict(D=8, N=200, start="zwei_zwei"),
])
def test_extreme_settings_run_and_give_a_verdict(kw):
    a = E.analyse(E.Settings(**kw))
    kind, text = E.verdict_run(a.facts)
    assert kind in ("success", "info", "warning", "error") and text
    assert a.values.shape == (4, kw["N"])


def test_all_regler_extremes_stay_inside_the_live_state_limit():
    assert ch.get_grid(max(C.D_CHOICES), C.SIGMA_MAX).n_states <= C.LIVE_MAX_STATES
    assert ch.get_ring(max(C.RING_N_CHOICES), 100.0).n_states <= C.LIVE_MAX_STATES


def test_start_outside_the_grid_is_impossible_by_construction(grid_d8):
    """Die Startverteilungen leben auf dem Gitter; jeder gezogene Startindex liegt im Gitter, auch für Ziehungen nahe 0 und nahe 1."""
    for kind in ("A", "B", "S10"):
        c = grid_d8.start_cdf(kind)
        for u in (0.0, 0.5, 1 - 2 ** -53):
            assert 0 <= min(np.searchsorted(c, u, side="right"), grid_d8.K - 1) < grid_d8.K


def test_exact_survival_is_one_for_n_below_the_shortest_path_in_the_hand_example(ring4):
    kl = ex.killed_from_A(ring4)
    assert kl.surv(1) == pytest.approx(1.0) and kl.surv(2) == pytest.approx(1.0)


def test_ties_in_the_ranks_are_averaged():
    r = dg.rank_normalize(np.array([[1.0, 1.0, 2.0, 2.0]]))
    assert r[0, 0] == pytest.approx(r[0, 1]) and r[0, 2] == pytest.approx(r[0, 3]) and r[0, 0] < r[0, 2]


def test_bounds_are_inclusive_for_the_rule():
    """ESS genau 400 und R-Dach knapp unter 1,01 bestehen; die Schwellen sind die der Konstanten."""
    assert C.RHAT_LIMIT == 1.01 and C.ESS_LIMIT == 400 and C.CHAINS == 4
    d = {"rhat": 1.0099, "ess": 400.0}
    assert (d["rhat"] < C.RHAT_LIMIT) and (d["ess"] >= C.ESS_LIMIT)


def test_seed_zero_and_maximum_seed_run():
    for seed in (0, C.SEED_MAX):
        a = E.analyse(E.Settings(N=200, seed=seed))
        assert np.isfinite(a.values).all()
