"""Unabhängige Referenzen für die Tests: eine rein sequentielle Fassung der Ketten über `dgn_rng.SplitMix64`, das Handbeispiel in Brüchen und die exakte
AR(1)-ESS. Nichts davon benutzt die vektorisierten Funktionen der Demo."""

import bisect
from fractions import Fraction as F

import numpy as np

import dgn_rng

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def derive_state(seed, rep, chain):
    """Anfangszustand des Stroms der Kette `chain` der Wiederholung `rep` (Ganzzahlrechnung, unabhängig von der numpy-Fassung)."""
    z = _mix((((seed * GAMMA) & MASK) + (((rep + 1) * 0xD1B54A32D192ED03) & MASK)) & MASK)
    return _mix((z + (((chain + 1) * 0xA24BAED4963EE407) & MASK)) & MASK)


def scalar_grid_path(grid, kind, state0, N):
    """Eine Kette der Gaußmischung, Zug für Zug mit `SplitMix64`: Ziehung 0 Start, dann je Schritt Vorschlag und Annahme."""
    rng = dgn_rng.SplitMix64(state0)
    cdf = grid.start_cdf(kind).tolist()
    x = min(bisect.bisect_right(cdf, rng.random()), grid.K - 1)
    pi, qcdf = grid.pi.tolist(), grid.cdf.tolist()
    out = []
    for _ in range(N):
        k = bisect.bisect_right(qcdf, rng.random()) - grid.kmax
        u = rng.random()
        y = x + k
        if 0 <= y < grid.K and u * pi[x] < pi[y]:
            x = y
        out.append(x)
    return np.array(out)


def scalar_ring_path(ring, kind, state0, N):
    """Eine Kette des Rings (Ziehung 0: Start nur bei gestreutem Start), je Schritt Platz (Rest der Division) und Annahme; Rückgabe s je Schritt."""
    rng = dgn_rng.SplitMix64(state0)
    u0 = rng.next_u64()                       # Ziehung 0 ist dem Start vorbehalten (auch wenn er fest ist)
    if kind == "A":
        x = ring.EVEN
    elif kind == "B":
        x = ring.ODD
    else:
        x = int(ring.S[u0 % ring.n_states])
    out = []
    p = ring.p_occ
    for _ in range(N):
        v = rng.next_u64() % ring.n
        u = rng.random()
        bit = 1 << v
        free = (x & int(ring.nb[v])) == 0
        x = (x & ~bit) | (bit if (free and u < p) else 0)
        out.append(int(ring.stab[x]))
    return np.array(out)


# --- Handbeispiel in Brüchen ------------------------------------------------------------------------------------------------------------------

def ring_fraction_matrix(n, lam):
    """Übergangsmatrix des Hard-Core-Rings in Brüchen (Heat-bath), Zustände in aufsteigender Bitmasken-Reihenfolge; Rückgabe (Zustände, Matrix, pi)."""
    lam = F(lam)
    states = [m for m in range(1 << n) if (m & (((m << 1) | (m >> (n - 1))) & ((1 << n) - 1))) == 0]
    idx = {m: i for i, m in enumerate(states)}
    p = lam / (1 + lam)
    P = [[F(0)] * len(states) for _ in states]
    for v in range(n):
        bit = 1 << v
        nb = (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))
        for x in states:
            i = idx[x]
            if x & nb:
                P[i][i] += F(1, n)
            else:
                P[i][idx[x | bit]] += p / n
                P[i][idx[x & ~bit]] += (1 - p) / n
    w = [lam ** bin(m).count("1") for m in states]
    pi = [x / sum(w) for x in w]
    return states, P, pi


def solve_fraction(A, b):
    """Gauß-Elimination mit Brüchen: A x = b."""
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(m):
        piv = next(r for r in range(c, m) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for r in range(m):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][m] for i in range(m)]


def ar1_ess(phi, N, M=4):
    """Exakte ESS von M stationär gestarteten AR(1)-Ketten der Länge N: M N / tau_N, tau_N = 1 + 2 sum (1 - k/N) phi^k."""
    k = np.arange(1, N)
    tau = 1 + 2 * float(((1 - k / N) * phi ** k).sum())
    return M * N / tau
