"""Ketten der Demo: eine Gaußmischung als Gitterkette und der Hard-Core-Ring (Heat-bath-Kette), beide ausgedachte Modelle.

Herkunft: neu geschrieben nach dem Aufbau der Klassen `Grid` und `Ring` aus einer früheren Messreihe (Wegwerfskripte, nicht importiert);
alle Zufallszahlen kommen aus SplitMix64 (`dgn_rng.py`, ein Strom je Kette).

Zufallsstrom. SplitMix64 ist ein Zähler-Generator: die i-te Ziehung (0-basiert) eines Stroms mit dem Anfangszustand z0 ist
`mix(z0 + (i + 1) * GAMMA)`. Deshalb lassen sich alle Ziehungen eines Blocks für alle Ketten auf einmal berechnen; die Folge je Kette ist bitgleich
die der Klasse `dgn_rng.SplitMix64` (der Test `tests/test_chain.py` prüft das gegen eine eigene, rein sequentielle Fassung). Je Kette gilt:
Ziehung 0 wählt den Startzustand (nur bei zufälligem Start), Schritt t (ab 0) verbraucht die Ziehungen 2t + 1 (Vorschlag bzw. Platz) und 2t + 2 (Annahme).

Gaußmischung. Ziel 1/2 N(-D/2, 1) + 1/2 N(+D/2, 1) auf dem Gitter mit Abstand 0,1 und Bereich +-(D/2 + 7); Zufallsschritt-Metropolis mit einem
diskretisierten, bei 5 sigma abgeschnittenen Normalvorschlag; Vorschläge außerhalb des Gitters werden abgelehnt.
Hard-Core-Ring. n Plätze im Ring (n gerade), keine zwei benachbarten belegt, Gewicht lambda^k; Heat-bath: Platz gleichverteilt wählen, belegen mit
Wahrscheinlichkeit lambda/(1 + lambda), wenn beide Nachbarn frei sind, sonst leeren. Größe s = belegte gerade minus belegte ungerade Plätze.
Gipfel A: alle geraden Plätze belegt (s = n/2), Gipfel B: alle ungeraden (s = -n/2)."""

from __future__ import annotations

import math
from functools import cached_property, lru_cache

import numpy as np

import dgn_constants as C

GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_U = np.uint64
_TWO53 = float(1 << 53)
BLOCK = 1000   # Schritte je Block beim Erzeugen der Ziehungen (Speicher: 2 * BLOCK * Ketten * 8 Byte)


# --- Zufallsstrom ----------------------------------------------------------------------------------------------------------------------------

def mix64(z):
    """Ausgabefunktion von SplitMix64 auf einem uint64-Array (Überlauf wird abgeschnitten)."""
    z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
    return z ^ (z >> _U(31))


def stream_states(seed: int, reps: int, chains: int, rep0: int = 0) -> np.ndarray:
    """Anfangszustände der Ströme, Form (reps * chains,): je (Wiederholung, Kette) ein eigener, aus dem Zufalls-Seed abgeleiteter Zustand."""
    rep = (np.arange(rep0, rep0 + reps, dtype=np.uint64) + _U(1))[:, None]
    ch = (np.arange(chains, dtype=np.uint64) + _U(1))[None, :]
    z = mix64(_U((seed * GAMMA) & _MASK) + rep * _U(0xD1B54A32D192ED03))
    z = mix64(z + ch * _U(0xA24BAED4963EE407))
    return np.ascontiguousarray(z).ravel()


def draws(state0: np.ndarray, first: int, count: int) -> np.ndarray:
    """Ziehungen first, ..., first + count - 1 (0-basiert) der Ströme `state0` (C,) als uint64-Array (count, C)."""
    i = np.arange(first, first + count, dtype=np.uint64) + _U(1)
    return mix64(state0[None, :] + i[:, None] * _U(GAMMA))


def to_unit(u: np.ndarray) -> np.ndarray:
    """uint64 -> Gleitkommazahl in [0, 1) mit 53 Bit (wie `SplitMix64.random`)."""
    return (u >> _U(11)).astype(np.float64) / _TWO53


# --- Startarten ------------------------------------------------------------------------------------------------------------------------------

START_KINDS = ("A", "B", "S5", "S10")   # Gipfel A, Gipfel B, gestreut (Gaußmischung: N(0, sd^2); Ring: gleichverteilt über alle Belegungen)


def start_pattern(kind: str, chains: int = 4) -> tuple[str, ...]:
    """Startarten der Ketten einer Wiederholung für die Namen der Oberfläche (`dgn_constants.START_CHOICES`)."""
    return {
        "alle_A": ("A",) * chains,
        "eine_B": ("A",) * (chains - 1) + ("B",),
        "zwei_zwei": ("A",) * (chains // 2) + ("B",) * (chains - chains // 2),
        "gestreut": ("S10",) * chains,
    }[kind]


# --- Gaußmischung ----------------------------------------------------------------------------------------------------------------------------

class Grid:
    """Gaußmischung als Gitterkette. Gipfel A ist der linke (x < 0), Gipfel B der rechte; Größe der Diagnostik ist x."""

    a_sign = -1   # A liegt bei negativen Werten der Größe
    value_dtype = np.float64

    def __init__(self, D: float, sigma: float, h: float = C.GRID_STEP, pad: float = C.GRID_PAD):
        self.D, self.sigma, self.h = float(D), float(sigma), h
        half = D / 2 + pad
        self.K = int(round(2 * half / h)) + 1
        self.x = -half + h * np.arange(self.K)
        lw = np.logaddexp(-0.5 * (self.x - D / 2) ** 2, -0.5 * (self.x + D / 2) ** 2)
        lw -= lw.max()
        self.logpi = lw - math.log(float(np.exp(lw).sum()))
        self.pi = np.exp(self.logpi)
        self.kmax = int(math.ceil(round(C.PROPOSAL_CUT * sigma / h, 9)))
        ks = np.arange(-self.kmax, self.kmax + 1)
        q = np.exp(-0.5 * (ks * h / sigma) ** 2)
        self.ks = ks
        self.q = q / q.sum()
        self.cdf = np.cumsum(self.q)
        self.cdf[-1] = 1.0
        self.thr = D / 4          # Wechselschwelle: der Gipfel gilt als erreicht bei |x| >= D/4
        self.n_states = self.K

    @property
    def key(self):
        return ("gauss", self.D, self.sigma)

    @cached_property
    def value(self) -> np.ndarray:
        """Größe je Zustand (hier x)."""
        return self.x

    @cached_property
    def side(self) -> np.ndarray:
        """a_sign * Größe je Zustand: positiv im Gipfel A, negativ im Gipfel B."""
        return self.a_sign * self.x

    @cached_property
    def indicator_B(self) -> np.ndarray:
        """Gipfelindikator I_B je Zustand: 1 im Gipfel B (x > 0), 1/2 bei x = 0, sonst 0; E_pi I_B = 1/2 (Symmetrie)."""
        return np.where(self.x > 1e-9, 1.0, np.where(self.x < -1e-9, 0.0, 0.5))

    @cached_property
    def P(self) -> np.ndarray:
        """Übergangsmatrix (dicht, K x K)."""
        K = self.K
        Pm = np.zeros((K, K))
        src = np.arange(K)
        for j, k in enumerate(self.ks):
            dst = src + k
            ok = (dst >= 0) & (dst < K)
            a = np.minimum(1.0, self.pi[dst[ok]] / self.pi[src[ok]])
            Pm[src[ok], dst[ok]] += self.q[j] * a
        Pm[src, src] += 1.0 - Pm.sum(1)
        return Pm

    def start_cdf(self, kind: str) -> np.ndarray:
        """Verteilung des Startzustands (kumuliert) für die Startart."""
        if kind == "A":
            w = np.exp(-0.5 * (self.x + self.D / 2) ** 2)
        elif kind == "B":
            w = np.exp(-0.5 * (self.x - self.D / 2) ** 2)
        elif kind in ("S5", "S10"):
            sd = float(kind[1:])
            w = np.exp(-0.5 * (self.x / sd) ** 2)
        else:
            raise ValueError(f"unbekannte Startart {kind!r}")
        c = np.cumsum(w / w.sum())
        c[-1] = 1.0
        return c

    def start_distribution(self, kind: str) -> np.ndarray:
        c = self.start_cdf(kind)
        return np.diff(np.concatenate([[0.0], c]))

    def initial(self, kinds: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Startindizes: Ziehung 0 jeder Kette, durch die Verteilung der Startart des Gitters gelegt."""
        u = to_unit(draws(state0, 0, 1)[0])
        x0 = np.empty(len(state0), dtype=np.int64)
        for kind in sorted(set(kinds.tolist())):
            sel = kinds == kind
            x0[sel] = np.minimum(np.searchsorted(self.start_cdf(kind), u[sel], side="right"), self.K - 1)
        return x0

    def advance(self, x: np.ndarray, U: np.ndarray, out: np.ndarray) -> np.ndarray:
        """B Schritte für alle Ketten; U: (2B, C) uint64; out: (B, C), erhält den Zustandsindex nach jedem Schritt; Rückgabe: Zustand nach dem Block."""
        kk = np.searchsorted(self.cdf, to_unit(U[0::2]), side="right") - self.kmax
        uu = to_unit(U[1::2])
        pi, K = self.pi, self.K
        for t in range(out.shape[0]):
            y = x + kk[t]
            yc = np.clip(y, 0, K - 1)
            acc = (y == yc) & (uu[t] * pi[x] < pi[yc])
            x = np.where(acc, yc, x)
            out[t] = x
        return x

    out_dtype = np.int16

    def to_value(self, path: np.ndarray) -> np.ndarray:
        """Pfad (Zustandsindizes) -> Werte der Größe."""
        return self.x[path]

    def to_indicator_B(self, path: np.ndarray) -> np.ndarray:
        return self.indicator_B[path]

    @cached_property
    def start_A_distribution(self) -> np.ndarray:
        return self.start_distribution("A")


# --- Hard-Core-Ring --------------------------------------------------------------------------------------------------------------------------

def popcount(a):
    a = np.asarray(a, dtype=np.int64)
    c = np.zeros(a.shape, dtype=np.int64)
    b = a.copy()
    while b.any():
        c += b & 1
        b >>= 1
    return c


class Ring:
    """Hard-Core-Ring mit Heat-bath. Größe s = belegte gerade minus belegte ungerade Plätze; A: s > 0, B: s < 0."""

    a_sign = +1
    out_dtype = np.int8

    def __init__(self, n: int, lam: float):
        if n < 4 or n % 2:
            raise ValueError("n muss gerade und mindestens 4 sein")
        self.n, self.lam = int(n), float(lam)
        full = (1 << n) - 1
        m = np.arange(1 << n, dtype=np.int64)
        ok = (m & (((m << 1) | (m >> (n - 1))) & full)) == 0
        self.S = m[ok]                       # zulässige Belegungen (Bitmasken), aufsteigend
        self.n_states = len(self.S)
        self.EVEN = sum(1 << v for v in range(0, n, 2))
        self.ODD = sum(1 << v for v in range(1, n, 2))
        self.k = popcount(self.S)
        self.s_state = popcount(self.S & self.EVEN) - popcount(self.S & self.ODD)
        self.stab = (popcount(m & self.EVEN) - popcount(m & self.ODD)).astype(np.int8)   # s je Bitmaske (alle 2^n)
        self.nb = np.array([(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)], dtype=np.int64)
        self.thr = int(math.ceil(n / 4))     # Wechselschwelle: Gipfel erreicht bei |s| >= ceil(n/4)
        w = self.lam ** self.k.astype(float)
        self.pi = w / w.sum()
        self.p_occ = self.lam / (1.0 + self.lam)

    @property
    def key(self):
        return ("ring", self.n, self.lam)

    @property
    def value(self) -> np.ndarray:
        return self.s_state.astype(float)

    @property
    def side(self) -> np.ndarray:
        return self.a_sign * self.s_state.astype(float)

    @cached_property
    def indicator_B(self) -> np.ndarray:
        return np.where(self.s_state < 0, 1.0, np.where(self.s_state > 0, 0.0, 0.5))

    @cached_property
    def P(self) -> np.ndarray:
        n, S = self.n, self.S
        Pm = np.zeros((self.n_states, self.n_states))
        p = self.p_occ
        for v in range(n):
            bit = 1 << v
            blocked = (S & self.nb[v]) != 0
            idx = np.arange(self.n_states)
            Pm[idx[blocked], idx[blocked]] += 1.0 / n
            free = ~blocked
            jo = np.searchsorted(S, S[free] | bit)
            je = np.searchsorted(S, S[free] & ~bit)
            Pm[idx[free], jo] += p / n
            Pm[idx[free], je] += (1 - p) / n
        return Pm

    def state_index(self, mask: int) -> int:
        return int(np.searchsorted(self.S, mask))

    def start_distribution(self, kind: str) -> np.ndarray:
        d = np.zeros(self.n_states)
        if kind == "A":
            d[self.state_index(self.EVEN)] = 1.0
        elif kind == "B":
            d[self.state_index(self.ODD)] = 1.0
        elif kind in ("S5", "S10"):
            d[:] = 1.0 / self.n_states
        else:
            raise ValueError(f"unbekannte Startart {kind!r}")
        return d

    @property
    def start_A_distribution(self) -> np.ndarray:
        return self.start_distribution("A")

    def initial(self, kinds: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Startbelegungen (Bitmasken): A und B sind feste Belegungen, 'gestreut' ist gleichverteilt über alle zulässigen Belegungen (Ziehung 0)."""
        u = draws(state0, 0, 1)[0]
        x0 = np.where(kinds == "A", self.EVEN, self.ODD).astype(np.int64)
        scat = np.isin(kinds, ("S5", "S10"))
        if scat.any():
            x0[scat] = self.S[(u[scat] % _U(self.n_states)).astype(np.int64)]
        return x0

    def advance(self, x: np.ndarray, U: np.ndarray, out: np.ndarray) -> np.ndarray:
        n = self.n
        v = (U[0::2] % _U(n)).astype(np.int64)
        bit = np.left_shift(1, v)
        occ_u = to_unit(U[1::2]) < self.p_occ
        nb, stab = self.nb, self.stab
        for t in range(out.shape[0]):
            free = (x & nb[v[t]]) == 0
            x = (x & ~bit[t]) | np.where(free & occ_u[t], bit[t], 0)
            out[t] = stab[x]
        return x

    def to_value(self, path: np.ndarray) -> np.ndarray:
        return path.astype(np.float64)

    def to_indicator_B(self, path: np.ndarray) -> np.ndarray:
        return np.where(path < 0, 1.0, np.where(path > 0, 0.0, 0.5))


# --- Modelle ---------------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_grid(D: float, sigma: float) -> Grid:
    return Grid(float(D), round(float(sigma), 3))


@lru_cache(maxsize=16)
def get_ring(n: int, lam: float) -> Ring:
    return Ring(int(n), float(lam))


# --- Läufe -----------------------------------------------------------------------------------------------------------------------------------

def simulate(model, kinds: tuple[str, ...], N: int, seed: int, reps: int = 1, rep0: int = 0) -> np.ndarray:
    """Pfade (reps, chains, N): `kinds` gibt die Startart je Kette (Länge = Zahl der Ketten). Alle Ketten laufen gemeinsam vektorisiert;
    jede hat ihren eigenen SplitMix64-Strom, so dass das Ergebnis einer Wiederholung nicht von der Zahl der übrigen abhängt."""
    chains = len(kinds)
    C_ = reps * chains
    state0 = stream_states(seed, reps, chains, rep0)
    kind_arr = np.tile(np.array(kinds), reps)
    x = model.initial(kind_arr, state0)
    out_t = np.empty((N, C_), dtype=model.out_dtype)
    for b0 in range(0, N, BLOCK):
        B = min(BLOCK, N - b0)
        U = draws(state0, 1 + 2 * b0, 2 * B)
        x = model.advance(x, U, out_t[b0:b0 + B])
    return np.ascontiguousarray(out_t.T).reshape(reps, chains, N)


def region_flags(model, path: np.ndarray):
    """(im Gebiet B, im Gebiet A) je Schritt, Boolesche Arrays wie `path`. Gebiet B: a_sign * Wert <= -Schwelle, Gebiet A: a_sign * Wert >= Schwelle
    (Gaußmischung: |x| >= D/4, Ring: |s| >= ceil(n/4); mit Toleranz 1e-9 gegen Gitterrundung)."""
    side = model.a_sign * model.to_value(path)
    tol = 1e-9
    return side <= -model.thr + tol, side >= model.thr - tol


def first_true(mask: np.ndarray) -> np.ndarray:
    """Erster 1-basierter Schritt mit True entlang der letzten Achse; N + 1, wenn es nie vorkommt."""
    N = mask.shape[-1]
    return np.where(mask.any(axis=-1), mask.argmax(axis=-1) + 1, N + 1)


def first_switch_from_A(model, path: np.ndarray) -> np.ndarray:
    """Schritt der ersten Ankunft im Gebiet B je Kette (N + 1: kein Wechsel); gedacht für Ketten, die in Gipfel A gestartet sind."""
    return first_true(region_flags(model, path)[0])
