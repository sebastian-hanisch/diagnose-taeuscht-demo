"""Zahlen im deutschen Format: Dezimalkomma, Gruppierung mit Leerzeichen ab fünf Stellen (DIN 5008)."""

from __future__ import annotations

import math


def ganz(n: int) -> str:
    """16796 -> „16 796“, 2000 -> „2000“."""
    s = str(abs(int(round(n))))
    if len(s) >= 5:
        gruppen = []
        while s:
            gruppen.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(gruppen))
    return ("-" if n < 0 else "") + s


def dez(x: float, stellen: int = 2) -> str:
    """Dezimalzahl mit Komma; `nan` und unendlich werden als Wort geschrieben."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nicht definiert"
    if isinstance(x, float) and math.isinf(x):
        return "unendlich"
    return f"{x:.{stellen}f}".replace(".", ",")


def prozent(anteil: float, stellen: int = 1) -> str:
    return f"{dez(anteil * 100, stellen)} %"


def potenz(x: float, stellen: int = 2) -> str:
    """Kleine Zahl in der Schreibweise 1,96·10⁻⁴."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{stellen - 1}e}".split("e")
    hoch = str(abs(int(exp))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{'⁻' if int(exp) < 0 else ''}{hoch}"


def von(k: int, reps: int) -> str:
    """„191 von 200“."""
    return f"{ganz(k)} von {ganz(reps)}"
