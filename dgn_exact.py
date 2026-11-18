"""Exakte Größen einer reversiblen Kette mit dichter Übergangsmatrix (Eigenzerlegung): Relaxationszeit, Überlebenswahrscheinlichkeit (kein Gipfelwechsel),
mittlere Zeit bis zum ersten Wechsel, exakte ESS einer stationär gestarteten Kette, Wechselrate im Gleichgewicht (GTH).

Herkunft: neu geschrieben; die Verfahren sind Lehrbuch (Eigenzerlegung der symmetrisierten Matrix, GTH-Algorithmus von Grassmann, Taksar und Heyman).
Zustände: `model.P` (dichte Matrix), `model.pi`, `model.side` (a_sign * Größe je Zustand), `model.thr` (Wechselschwelle)."""

from __future__ import annotations

import numpy as np

TOL = 1e-9


def sym_eig(Pm: np.ndarray, pi: np.ndarray):
    """Eigenwerte und Eigenvektoren der symmetrisierten Matrix D^(1/2) P D^(-1/2) (reversible Kette): (ev aufsteigend, U, sqrt(pi))."""
    s = np.sqrt(pi)
    Sm = (s[:, None] * Pm) / s[None, :]
    ev, U = np.linalg.eigh(0.5 * (Sm + Sm.T))
    return ev, U, s


def relaxation(Pm: np.ndarray, pi: np.ndarray) -> tuple[float, float]:
    """(Lücke 1 - lambda_2, Relaxationszeit 1 / (1 - |lambda|*)) mit |lambda|* = max(lambda_2, |lambda_min|)."""
    ev, _, _ = sym_eig(Pm, pi)
    lam2 = ev[-2]
    lam_star = max(abs(ev[0]), abs(lam2))
    return float(1.0 - lam2), float(1.0 / (1.0 - lam_star))


def _geometric_weighted_sum(lam: np.ndarray, N: int) -> np.ndarray:
    """sum_{k=1}^{N-1} (1 - k/N) lam^k je Eigenwert (geschlossene Form; für lam nahe 1 die direkte Summe)."""
    out = np.empty_like(lam)
    k = np.arange(1, N)
    w = 1.0 - k / N
    near = np.abs(1.0 - lam) < 1e-7
    lm = lam[~near]
    if lm.size:
        out[~near] = lm / (1.0 - lm) - lm * (1.0 - np.power(lm, N)) / (N * (1.0 - lm) ** 2)
    for j in np.where(near)[0]:
        out[j] = float((w * np.power(lam[j], k)).sum())
    return out


def exact_ess(Pm: np.ndarray, pi: np.ndarray, f: np.ndarray, N: int, chains: int = 4) -> tuple[float, float]:
    """Exakte ESS von `chains` unabhängigen, stationär gestarteten Ketten der Länge N für die Funktion f: chains * N / tau_N mit
    tau_N = 1 + 2 sum_{k<N} (1 - k/N) rho_k, rho_k aus der Spektralzerlegung. Rückgabe: (ESS, tau_N)."""
    ev, U, s = sym_eig(Pm, pi)
    ft = f - float((pi * f).sum())
    var = float((pi * ft * ft).sum())
    c = (s * ft) @ U
    w2 = c * c / var
    keep = (w2 > 1e-18) & (ev < 1 - 1e-12)         # die konstante Mode (Eigenwert 1) liefert nichts
    tau = 1.0 + 2.0 * float((w2[keep] * _geometric_weighted_sum(ev[keep], N)).sum())
    return chains * N / tau, tau


class Killed:
    """Kette mit absorbierender Menge (Ankunft im Gebiet des anderen Gipfels): exakte Überlebenswahrscheinlichkeit P(T > N) und E[T] aus dem Start `mu0`
    (Zeilenvektor über alle Zustände). T ist die erste Ankunft nach mindestens einem Schritt; ein Start in der Menge zählt als Wechsel zur Zeit 0."""

    def __init__(self, Pm: np.ndarray, pi: np.ndarray, absorbing: np.ndarray, mu0: np.ndarray):
        keep = ~absorbing
        Q = Pm[np.ix_(keep, keep)]
        pk = pi[keep]
        s = np.sqrt(pk)
        Qs = (s[:, None] * Q) / s[None, :]
        self.ev, self.U = np.linalg.eigh(0.5 * (Qs + Qs.T))
        m = mu0[keep]
        self.a = (m / s) @ self.U
        self.b = s @ self.U
        self.Q = Q
        self.m = m

    def surv(self, N: int) -> float:
        """P(T > N): keine Ankunft im anderen Gipfel in den Schritten 1, ..., N (Eigenzerlegung)."""
        v = float((self.a * self.b * np.power(self.ev, int(N))).sum())
        return min(1.0, max(0.0, v))

    def surv_matrix_power(self, N: int) -> float:
        """Dieselbe Größe über wiederholte Multiplikation mit Q (andere Rechnung, für Tests und kleine N)."""
        v = self.m.copy()
        for _ in range(int(N)):
            v = v @ self.Q
        return float(v.sum())

    def mean_exit(self) -> float:
        """E[T] = sum_j a_j b_j / (1 - ev_j)."""
        return float((self.a * self.b / (1.0 - self.ev)).sum())

    def mean_exit_solve(self) -> float:
        """E[T] als Lösung von (I - Q) h = 1 (anderer Rechenweg)."""
        n = self.Q.shape[0]
        h = np.linalg.solve(np.eye(n) - self.Q, np.ones(n))
        return float(self.m @ h)


def absorbing_B(model) -> np.ndarray:
    """Zustände im Gebiet des Gipfels B: a_sign * Größe <= -Schwelle."""
    return model.side <= -model.thr + TOL


def absorbing_A(model) -> np.ndarray:
    return model.side >= model.thr - TOL


def killed_from_A(model) -> Killed:
    """Überlebenswahrscheinlichkeit und E[T] für den Start im Gipfel A (Gaußmischung: N(-D/2, 1) auf dem Gitter; Ring: alle geraden Plätze belegt)."""
    return Killed(model.P, model.pi, absorbing_B(model), model.start_A_distribution)


def gth_stationary(Pm: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung einer irreduziblen Kette nach Grassmann, Taksar und Heyman (rein additiv, ohne Subtraktion: auch bei fast zerlegbaren Ketten genau)."""
    A = np.array(Pm, dtype=float, copy=True)
    n = A.shape[0]
    for k in range(n - 1, 0, -1):
        s = A[k, :k].sum()
        A[:k, k] /= s
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ A[:k, k]
    return pi / pi.sum()


def switch_rate(model) -> float:
    """Wechsel je Schritt im Gleichgewicht, beide Richtungen, mit Hysterese: ein Wechsel ist die Ankunft im Gebiet des anderen Gipfels, nachdem der
    zuletzt erreichte Gipfel der andere war. Erweiterte Kette (Zustand, zuletzt erreichter Gipfel), stationäre Verteilung per GTH, Rate = Fluss in die
    Wechsel-Ankünfte. Das ist die Wechselrate, nicht die Zeit bis zum ersten Wechsel vom Start aus (die hängt vom Start ab)."""
    Pm, side, thr = model.P, model.side, model.thr
    inB = side <= -thr + TOL
    inA = side >= thr - TOL
    K = Pm.shape[0]
    # Zustände der erweiterten Kette: (i, A) für i nicht im Gebiet B; (i, B) für i nicht im Gebiet A (die übrigen sind unerreichbar)
    idxA = np.where(~inB)[0]
    idxB = np.where(~inA)[0]
    nA = len(idxA)
    pos_A = -np.ones(K, dtype=int)
    pos_A[idxA] = np.arange(nA)
    pos_B = -np.ones(K, dtype=int)
    pos_B[idxB] = nA + np.arange(len(idxB))
    E = np.zeros((nA + len(idxB), nA + len(idxB)))
    # Marke A: die Ankunft j im Gebiet B wechselt die Marke zu B, sonst bleibt sie A (die Zielspalten sind paarweise verschieden)
    E[np.ix_(pos_A[idxA], np.where(inB, pos_B, pos_A))] = Pm[idxA]
    E[np.ix_(pos_B[idxB], np.where(inA, pos_A, pos_B))] = Pm[idxB]
    mu = gth_stationary(E)
    rate = float((mu[pos_A[idxA]] * Pm[np.ix_(idxA, np.where(inB)[0])].sum(axis=1)).sum()
                 + (mu[pos_B[idxB]] * Pm[np.ix_(idxB, np.where(inA)[0])].sum(axis=1)).sum())
    return float(rate)


def exact_summary(model, N_list, chains: int = 4) -> dict:
    """Alle exakten Größen einer Zelle: Relaxationszeit, E[T], P(kein Wechsel) einer Kette für jedes N, ESS einer stationären Kette."""
    gap, trel = relaxation(model.P, model.pi)
    kl = killed_from_A(model)
    return {
        "gap": gap,
        "t_rel": trel,
        "E_T": kl.mean_exit(),
        "p_one": {int(N): kl.surv(N) for N in N_list},
        "ess_stationary": {int(N): exact_ess(model.P, model.pi, np.asarray(model.value, dtype=float), N, chains)[0] for N in N_list},
    }
