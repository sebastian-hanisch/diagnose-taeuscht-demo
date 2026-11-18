"""Diagnostik: geteiltes R-Dach und ESS (bulk) nach Vehtari, Gelman, Simpson, Carpenter und Bürkner (Bayesian Analysis 16(2), 2021).

Herkunft: eigene Fassung der Gleichungen (Formeln und Herleitung stehen in `posterior-sampling-demo`, hier nicht wiederholt); neu geschrieben, nicht aus einer
anderen Demo importiert. Eingabe ist immer ein Array `ch` der Form (Ketten, Schritte).

Randfälle, die gemeldet statt weitergereicht werden: sind alle Hälften konstant und gleich, ist R-Dach der Quotient 0/0 (`nan`); sind sie konstant, aber
verschieden, ist es unendlich. Weniger als vier Schritte je Kette geben keine zwei Werte je Hälfte: R-Dach ist dann `nan`."""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

import dgn_constants as C


def rhat_classic(ch: np.ndarray) -> float:
    """Ungeteiltes R-Dach: sqrt(var+ / W) mit W = mittlere Varianz in den Ketten, B/N = Varianz der Kettenmittel."""
    M, N = ch.shape
    if M < 2 or N < 2:
        return float("nan")
    mu = ch.mean(1)
    B = N * mu.var(ddof=1)
    W = ch.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float("inf") if B > 1e-12 else float("nan")
    varp = (N - 1) / N * W + B / N
    return float(math.sqrt(varp / W))


def split_chains(ch: np.ndarray) -> np.ndarray:
    """Jede Kette in zwei Hälften teilen; bei ungerader Länge fällt der mittlere Wert weg."""
    M, N = ch.shape
    h = N // 2
    return np.concatenate([ch[:, :h], ch[:, N - h:]], axis=0)


def rhat_split(ch: np.ndarray) -> float:
    if ch.shape[1] < 4:
        return float("nan")
    return rhat_classic(split_chains(ch))


def rank_normalize(ch: np.ndarray) -> np.ndarray:
    """Rangnormalisierung: Ränge (Gleichstände gemittelt) -> Quantile der Standardnormalverteilung ((r - 3/8) / (S + 1/4))."""
    S = ch.size
    r = stats.rankdata(ch.ravel(), method="average").reshape(ch.shape)
    return special.ndtri((r - 3.0 / 8.0) / (S + 1.0 / 4.0))


def rhat_rank(ch: np.ndarray) -> float:
    """Rangnormalisiertes R-Dach: Maximum aus dem R-Dach der rangnormalisierten und dem der gefalteten (|x - Median|) geteilten Ketten (Vehtari u. a. 2021)."""
    if ch.shape[1] < 4:
        return float("nan")
    z = rank_normalize(split_chains(ch))
    med = np.median(ch)
    zf = rank_normalize(split_chains(np.abs(ch - med)))
    return float(max(rhat_classic(z), rhat_classic(zf)))


def acov_fft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    xc = x - x.mean()
    n = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(xc, n)
    return np.fft.irfft(f * np.conj(f), n)[:N] / N


def ess_multi(ch: np.ndarray, cap: bool = True) -> float:
    """ESS aus mehreren Ketten (Gl. 10 bis 13): Autokorrelation aus W und var+, Geyer-Abbruch bei der ersten negativen Paarsumme, monotone Paarsummen."""
    M, N = ch.shape
    if N < 4:
        return float("nan")
    acov = np.array([acov_fft(c) for c in ch])
    W = ch.var(axis=1, ddof=1).mean()
    B = N * ch.mean(1).var(ddof=1) if M > 1 else 0.0
    varp = (N - 1) / N * W + B / N
    if not varp > 0:
        return float("nan")
    rho = 1.0 - (W - acov.mean(0)) / varp
    rho[0] = 1.0
    npair = N // 2
    Pt = rho[0:2 * npair:2] + rho[1:2 * npair:2]
    neg = np.where(Pt < 0)[0]
    k = int(neg[0]) if len(neg) else npair
    if k == 0:
        k = 1
    Pt = np.minimum.accumulate(Pt[:k])
    tau = -1.0 + 2.0 * Pt.sum()
    S = M * N
    if cap:
        tau = max(tau, 1.0 / math.log10(S))
    return float(S / tau) if tau > 0 else float("inf")


def ess_bulk(ch: np.ndarray) -> float:
    """ESS_bulk: ESS der rangnormalisierten, geteilten Ketten."""
    if ch.shape[1] < 4:
        return float("nan")
    return ess_multi(rank_normalize(split_chains(ch)))


def diagnose(ch: np.ndarray) -> dict:
    """R-Dach (geteilt) und ESS_bulk einer Wiederholung; `rhat_ok`: R-Dach < 1,01; `today`: heutige Regel (R-Dach < 1,01 und ESS_bulk >= 400, mindestens vier Ketten)."""
    rh = rhat_split(ch)
    es = ess_bulk(ch)
    enough = ch.shape[0] >= C.CHAINS
    rhat_ok = bool(rh < C.RHAT_LIMIT)
    return {
        "rhat": rh,
        "ess": es,
        "rhat_ok": rhat_ok,
        "today": bool(rhat_ok and es >= C.ESS_LIMIT and enough),
        "undefined": bool(math.isnan(rh)),
    }
