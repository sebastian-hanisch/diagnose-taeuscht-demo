"""Zellen der eingefrorenen Läufe: Festlegung (Modell, Parameter, Kettenlänge, Zufalls-Seed, Startart), Auswertung je Zelle und die exakten Gegenzahlen.

Eine Zelle sind `reps` Wiederholungen mit je vier Ketten. Gezählt wird in Wiederholungen:
  kein Wechsel   keine der Ketten erreicht in den Schritten 1, ..., N das Gebiet des anderen Gipfels (Start im Gipfel A);
  R-Dach < 1,01  geteiltes R-Dach der Wiederholung unter 1,01;
  täuschend      R-Dach < 1,01 und kein Wechsel;
  heute          R-Dach < 1,01 und ESS_bulk >= 400 und kein Wechsel (Regel nach Vehtari u. a. 2021).
Die exakte Erwartung für „kein Wechsel“ ist reps * p^4 mit p = exakte Überlebenswahrscheinlichkeit einer Kette (`dgn_exact.Killed`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import dgn_chain as ch
import dgn_constants as C
import dgn_diagnostics as dg
import dgn_exact as ex

SEED_BASE = 18000


@dataclass(frozen=True)
class CellSpec:
    id: str
    vehicle: str                  # "gauss" oder "ring"
    a: float                      # Gaußmischung: Abstand D; Ring: Zahl der Plätze n
    b: float                      # Gaußmischung: Schrittweite sigma; Ring: Gewicht lambda
    N: int
    seed: int
    reps: int = C.REPS
    kinds: tuple = ("A",) * C.CHAINS
    burn: int = 0
    Ns: tuple = ()                # nur Fensterzellen: die ausgewerteten Kettenlängen (Präfixe derselben Läufe)

    def model(self):
        return model_of(self.vehicle, self.a, self.b)


def model_of(vehicle: str, a: float, b: float):
    if vehicle == "gauss":
        return ch.get_grid(a, b)
    if vehicle == "ring":
        return ch.get_ring(int(a), float(b))
    raise ValueError(vehicle)


# --- Zellenliste -----------------------------------------------------------------------------------------------------------------------------

def _g(D, sig, seed, **kw):
    return CellSpec(f"gauss_D{D:g}_s{sig:g}_{seed}", "gauss", D, sig, kw.pop("N", 2000), seed, **kw)


def _r(n, lam, seed, **kw):
    return CellSpec(f"ring_n{n}_l{lam:g}_{seed}", "ring", n, lam, kw.pop("N", 2000), seed, **kw)


CELLS = (
    _g(10, 1.0, SEED_BASE + 101), _g(10, 1.0, SEED_BASE + 102), _g(10, 1.0, SEED_BASE + 103),
    _g(8, 1.0, SEED_BASE + 110), _g(6, 1.0, SEED_BASE + 111), _g(10, 2.4, SEED_BASE + 112), _g(8, 2.4, SEED_BASE + 113),
    _g(0, 1.0, SEED_BASE + 114),
    _r(12, 50, SEED_BASE + 201), _r(12, 50, SEED_BASE + 202), _r(12, 20, SEED_BASE + 203),
    _r(16, 100, SEED_BASE + 204), _r(16, 100, SEED_BASE + 205), _r(16, 1, SEED_BASE + 206),
)
WINDOW_NS_GAUSS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
WINDOW_NS_RING = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
WINDOWS = (
    CellSpec("window_gauss_D8_s1", "gauss", 8, 1.0, max(WINDOW_NS_GAUSS), SEED_BASE + 301, Ns=WINDOW_NS_GAUSS),
    CellSpec("window_ring_n12_l20", "ring", 12, 20, max(WINDOW_NS_RING), SEED_BASE + 302, Ns=WINDOW_NS_RING),
)
START_PATTERNS = (
    ("alle vier in A", ("A", "A", "A", "A"), 0),
    ("drei in A, eine in B", ("A", "A", "A", "B"), 0),
    ("zwei in A, zwei in B", ("A", "A", "B", "B"), 0),
    ("gestreut, sd 5", ("S5",) * 4, 0), ("gestreut, sd 5", ("S5",) * 4, C.BURN),
    ("gestreut, sd 10", ("S10",) * 4, 0), ("gestreut, sd 10", ("S10",) * 4, C.BURN),
)
START_MODELS = (("gauss", 10, 1.0, SEED_BASE + 400), ("ring", 12, 50, SEED_BASE + 410))
MCELLS = tuple(CellSpec(f"streu_M{M}", "gauss", 10, 1.0, 2000, SEED_BASE + 500 + M, reps=C.M_REPS, kinds=("S10",) * M, burn=C.BURN) for M in C.M_CHAINS)
SWITCH_CELLS = (
    CellSpec("rate_gauss_D6_s2.4", "gauss", 6, 2.4, 20000, SEED_BASE + 601),
)


# --- Auswertung ------------------------------------------------------------------------------------------------------------------------------

def var_indicator(model) -> float:
    """Var_pi(I_B) des Gipfelindikators (exakt)."""
    I = model.indicator_B
    return float((model.pi * I * I).sum() - (model.pi * I).sum() ** 2)


def _chunks(R: int, size: int):
    for s in range(0, R, size):
        yield s, min(R, s + size)


def first_switch_all(model, path: np.ndarray) -> np.ndarray:
    """Erster Schritt im Gebiet B je (Wiederholung, Kette), in Stücken gerechnet (Speicher)."""
    R, M, N = path.shape
    out = np.empty((R, M), dtype=np.int64)
    step = max(1, int(4e6 // (M * N)))
    for s, e in _chunks(R, step):
        out[s:e] = ch.first_switch_from_A(model, path[s:e])
    return out


def cumulative_share_B(model, path: np.ndarray, Ns) -> np.ndarray:
    """Geschätzter Gipfelanteil I_B der ersten N Schritte aller Ketten einer Wiederholung für jedes N: Form (len(Ns), R)."""
    R, M, N = path.shape
    out = np.empty((len(Ns), R))
    step = max(1, int(4e6 // (M * N)))
    for s, e in _chunks(R, step):
        cs = np.cumsum(model.to_indicator_B(path[s:e]), axis=2)
        for k, n_ in enumerate(Ns):
            out[k, s:e] = cs[:, :, n_ - 1].sum(axis=1) / (M * n_)
    return out


def exact_expectation(model, N: int, reps: int, chains: int = C.CHAINS, killed=None) -> dict:
    """Exakte Wahrscheinlichkeit, dass eine Kette in N Schritten nicht wechselt (p), und für vier Ketten (p^4) samt Erwartung und Standardabweichung der Zählung."""
    kl = killed or ex.killed_from_A(model)
    p = kl.surv(N)
    q = p ** chains
    return {"p_one": p, "p_all": q, "expected": reps * q, "sd": math.sqrt(reps * q * (1.0 - q))}


def count_cell(model, path: np.ndarray, N: int, *, with_rank: bool = False, fs: np.ndarray | None = None, share: np.ndarray | None = None) -> dict:
    """Zählungen einer Zelle für die ersten N Schritte der Pfade (Start aller Ketten im Gipfel A)."""
    R, M, _ = path.shape
    has_peaks = model.thr > 0
    if has_peaks:
        fs = first_switch_all(model, path[:, :, :N]) if fs is None else fs
        nosw = (fs > N).all(axis=1)
    else:
        nosw = None
    rh_ok = np.zeros(R, bool)
    rk_ok = np.zeros(R, bool)
    today = np.zeros(R, bool)
    nan_ct = 0
    ess = np.empty(R)
    for r in range(R):
        v = model.to_value(path[r, :, :N])
        d = dg.diagnose(v)
        rh_ok[r], today[r], ess[r] = d["rhat_ok"], d["today"], d["ess"]
        nan_ct += int(d["undefined"])
        if with_rank:
            rk_ok[r] = dg.rhat_rank(v) < C.RHAT_LIMIT
    if share is None:
        share = cumulative_share_B(model, path[:, :, :N], [N])[0]
    out = {
        "reps": R, "N": int(N), "rhat_lt": int(rh_ok.sum()), "rhat_ge": int(R - rh_ok.sum() - nan_ct), "rhat_nan": nan_ct,
        "ess_mean": float(np.nanmean(ess)), "share_B_mean": float(share.mean()), "mse_share": float(np.mean((share - 0.5) ** 2)),
    }
    if has_peaks:
        out.update({"no_switch": int(nosw.sum()), "deceiving": int((rh_ok & nosw).sum()), "today": int((today & nosw).sum())})
    if with_rank:
        out["rhat_rank_lt"] = int(rk_ok.sum())
        if has_peaks:
            out["deceiving_rank"] = int((rk_ok & nosw).sum())
    out["n_eff_mse"] = var_indicator(model) / out["mse_share"] if out["mse_share"] > 0 else None
    return out


def run_cell(spec: CellSpec) -> tuple[dict, dict]:
    """Simulation und Zählung einer Zelle; Rückgabe (Zählungen, exakte Größen)."""
    model = spec.model()
    path = ch.simulate(model, spec.kinds, spec.N, spec.seed, reps=spec.reps)
    counts = count_cell(model, path, spec.N, with_rank=True)
    return counts, exact_for(model, spec.N, spec.reps)


def exact_for(model, N: int, reps: int, with_rate: bool = False) -> dict:
    gap, trel = ex.relaxation(model.P, model.pi)
    out = {"gap": gap, "t_rel": trel,
           "ess_stationary": ex.exact_ess(model.P, model.pi, np.asarray(model.value, dtype=float), N)[0],
           "var_indicator": var_indicator(model), "n_states": int(model.n_states)}
    if model.thr > 0:
        kl = ex.killed_from_A(model)
        out["E_T"] = kl.mean_exit()
        out.update(exact_expectation(model, N, reps, killed=kl))
        if with_rate:
            out["switch_rate"] = ex.switch_rate(model)
    return out


# --- Fenster ---------------------------------------------------------------------------------------------------------------------------------

def run_window(spec: CellSpec) -> tuple[list[dict], dict]:
    """Auswertung derselben Läufe bei mehreren Kettenlängen (Präfixe); Rückgabe (Zählungen je N, Wechselzählung über alle Läufe)."""
    model = spec.model()
    path = ch.simulate(model, spec.kinds, spec.N, spec.seed, reps=spec.reps)
    fs = first_switch_all(model, path)
    shares = cumulative_share_B(model, path, spec.Ns)
    rows = []
    for k, n_ in enumerate(spec.Ns):
        c = count_cell(model, path, n_, fs=fs, share=shares[k])
        c["alarm"] = c["reps"] - c["rhat_lt"] - c["rhat_nan"]
        rows.append(c)
    return rows, count_switches(model, path)


def exact_window(spec: CellSpec) -> dict:
    """Exakte Größen zur Fensterzelle: E[T], Relaxationszeit, Wechselrate im Gleichgewicht, je N die exakte Erwartung von „kein Wechsel“."""
    model = spec.model()
    kl = ex.killed_from_A(model)
    return {"E_T": kl.mean_exit(), "t_rel": ex.relaxation(model.P, model.pi)[1], "rate_exact": ex.switch_rate(model),
            "rows": [dict(N=int(n_), **exact_expectation(model, n_, spec.reps, killed=kl)) for n_ in spec.Ns]}


def count_switches(model, path: np.ndarray) -> dict:
    """Zählt die Wechsel mit Hysterese über alle Ketten (Start im Gipfel A): ein Wechsel ist die Ankunft im Gebiet des Gipfels, der nicht der zuletzt erreichte war."""
    R, M, N = path.shape
    total = 0
    for r in range(R):
        inB, inA = ch.region_flags(model, path[r])
        for m in range(M):
            ev = inB[m] | inA[m]
            lab = inB[m][ev]                        # True: zuletzt Gebiet B erreicht
            if lab.size:
                total += int(np.count_nonzero(lab[1:] != lab[:-1])) + int(lab[0])
    steps = R * M * N
    return {"count": total, "steps": int(steps), "rate": total / steps}


# --- Startexperiment -------------------------------------------------------------------------------------------------------------------------

def run_start(model_spec, name: str, kinds: tuple, burn: int, seed: int, reps: int = C.REPS, N: int = 2000) -> dict:
    vehicle, a, b = model_spec
    model = model_of(vehicle, a, b)
    path = ch.simulate(model, kinds, N, seed, reps=reps)
    post = path[:, :, burn:]
    R = path.shape[0]
    rh_ok = np.array([dg.rhat_split(model.to_value(post[r])) < C.RHAT_LIMIT for r in range(R)])
    rh_nan = sum(int(math.isnan(dg.rhat_split(model.to_value(post[r])))) for r in range(R))
    inB, inA = ch.region_flags(model, post)
    same = ~(inB.any(axis=2).any(axis=1)) | ~(inA.any(axis=2).any(axis=1))   # keine Kette im Gebiet B oder keine im Gebiet A
    return {"name": name, "kinds": list(kinds), "burn": burn, "seed": seed, "reps": R, "N": N, "chains": len(kinds),
            "rhat_lt": int(rh_ok.sum()), "rhat_ge": int(R - rh_ok.sum() - rh_nan), "rhat_nan": rh_nan,
            "same_side": int(same.sum()), "same_and_rhat_lt": int((same & rh_ok).sum())}


def expected_same_side(chains: int) -> float:
    """Wahrscheinlichkeit, dass `chains` unabhängig mit Wahrscheinlichkeit 1/2 gestartete Ketten alle im selben Gipfel beginnen: 2^(1 - M) (Symmetrie)."""
    return 2.0 ** (1 - chains)
