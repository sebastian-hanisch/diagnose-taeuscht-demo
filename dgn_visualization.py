"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import dgn_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_traces(values: np.ndarray, thr: float, vehicle_is_ring: bool, kinds: tuple) -> go.Figure:
    """Die vier Ketten als Verlauf der Größe (Gaußmischung: x, Ring: s); gestrichelt die Wechselschwellen ±D/4 bzw. ±ceil(n/4)."""
    M, N = values.shape
    stride = max(1, N // C.PLOT_POINTS)
    xs = np.arange(1, N + 1)[::stride]
    fig = go.Figure()
    for c in range(M):
        label = f"Kette {c + 1} (Start {_kind_name(kinds[c])})"
        fig.add_trace(go.Scatter(x=xs, y=values[c][::stride], mode="lines", line=dict(color=C.CHAIN_COLORS[c % len(C.CHAIN_COLORS)], width=1.4), name=label,
                                 hovertemplate="Schritt %{x}: %{y:.2f}<extra>" + label + "</extra>"))
    if thr > 0:
        for sign in (1, -1):
            fig.add_hline(y=sign * thr, line=dict(color=C.COLOR_EXACT, dash="dash", width=1))
        fig.add_annotation(xref="paper", yref="y", x=1.0, y=thr, text="Wechselschwelle", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=10, color=C.COLOR_EXACT))
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="s (belegte gerade minus ungerade Plätze)" if vehicle_is_ring else "x")
    return _base(fig)


def _kind_name(kind: str) -> str:
    return {"A": "A", "B": "B", "S5": "gestreut", "S10": "gestreut"}.get(kind, kind)


def build_share(steps: np.ndarray, curve: np.ndarray) -> go.Figure:
    """Geschätzter Anteil des Gipfels B über die ersten n Schritte aller vier Ketten; gestrichelt der exakte Wert 1/2."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=curve, mode="lines", line=dict(color=C.COLOR_BLUE, width=2.5), name="geschätzter Anteil Gipfel B",
                             hovertemplate="n = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_EXACT, dash="dash", width=1.5), annotation_text="exakt 0,5", annotation_position="top left")
    fig.update_xaxes(title_text="Schritte je Kette (n)")
    fig.update_yaxes(title_text="Anteil Gipfel B", range=[-0.02, 1.02])
    return _base(fig, height=280)


def build_cell_bars(cell: dict) -> go.Figure:
    """Zählungen einer Zelle in Wiederholungen: kein Wechsel gezählt gegen exakt erwartet, R-Dach < 1,01, täuschend, heute."""
    c, e = cell["counts"], cell["exact"]
    reps = c["reps"]
    if "deceiving" not in c:
        names, vals = ["R̂ < 1,01"], [c["rhat_lt"]]
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=C.COLOR_OK, hovertemplate="%{x}: %{y} von " + str(reps) + "<extra></extra>"))
    else:
        names = ["kein Wechsel (gezählt)", "kein Wechsel (exakt erwartet)", "R̂ < 1,01", "täuschend", "heute: R̂ < 1,01 und ESS ≥ 400"]
        vals = [c["no_switch"], e["expected"], c["rhat_lt"], c["deceiving"], c["today"]]
        colors = [C.COLOR_BLUE, C.COLOR_EXACT, C.COLOR_OK, C.COLOR_DECEIVE, C.COLOR_DECEIVE]
        err = [0, e["sd"], 0, 0, 0]
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, error_y=dict(type="data", array=err, visible=True),
                               hovertemplate="%{x}: %{y:.1f} von " + str(reps) + "<extra></extra>"))
    fig.update_yaxes(title_text=f"Wiederholungen (von {reps})", range=[0, reps * 1.05])
    return _base(fig, height=300)


def build_window(w: dict) -> go.Figure:
    """Das Täuschungsfenster: Wiederholungen mit Täuschung, mit Alarm (R̂ ≥ 1,01) und die exakte Erwartung „kein Wechsel“ über die Kettenlänge N (logarithmische Achse)."""
    rows = w["rows"]
    xs = [r["N"] for r in rows]
    et = w["E_T"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["expected"] for r in rows], mode="lines", line=dict(color=C.COLOR_EXACT, dash="dash", width=2), name="kein Wechsel, exakt erwartet"))
    fig.add_trace(go.Scatter(x=xs, y=[r["alarm"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BLUE, width=2.5), name="R̂ ≥ 1,01 (Alarm)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["deceiving"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_DECEIVE, width=3), name="täuschend"))
    for frac, label in ((0.2, "0,2 E[T]"), (2.0, "2 E[T]")):
        x = frac * et
        if xs[0] <= x <= xs[-1]:
            fig.add_vline(x=x, line=dict(color=C.COLOR_EXACT, dash="dot", width=1), annotation_text=label, annotation_position="top")
    fig.update_xaxes(title_text="Schritte je Kette N", type="log")
    fig.update_yaxes(title_text=f"Wiederholungen (von {w['spec']['reps']})", range=[0, w["spec"]["reps"] * 1.05])
    return _base(fig)


def build_window_share(w: dict) -> go.Figure:
    rows = w["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["N"] for r in rows], y=[r["share_B_mean"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BLUE, width=2.5),
                             name="geschätzter Anteil Gipfel B (Mittel der Wiederholungen)"))
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_EXACT, dash="dash", width=1.5), annotation_text="exakt 0,5", annotation_position="bottom left")
    fig.update_xaxes(title_text="Schritte je Kette N", type="log")
    fig.update_yaxes(title_text="Anteil Gipfel B", range=[-0.02, 0.6])
    return _base(fig, height=260)


def build_starts(rows: list[dict]) -> go.Figure:
    """R-Dach ≥ 1,01 (Alarm) je Startart, in Wiederholungen."""
    labels = [f"{r['name']}" + (f", Verwerfen {r['burn']}" if r["burn"] else "") for r in rows]
    reps = rows[0]["reps"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["rhat_ge"] for r in rows], name="R̂ ≥ 1,01 (Alarm)", marker_color=C.COLOR_BLUE))
    fig.add_trace(go.Bar(x=labels, y=[r["same_side"] for r in rows], name="alle Ketten auf derselben Seite", marker_color=C.COLOR_DECEIVE))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Wiederholungen (von {reps})", range=[0, reps * 1.05])
    return _base(fig, height=320)


def build_formula(chains_now: int, mrows: list[dict]) -> go.Figure:
    """Die Formel 2^(1 - M) für M = 2 bis 8 und die gemessenen Anteile für M = 2, 4, 8 (je 1000 Wiederholungen, Fehlerbalken: zwei Standardfehler)."""
    ms = list(range(2, 9))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=[2.0 ** (1 - m) for m in ms], mode="lines+markers", line=dict(color=C.COLOR_EXACT, dash="dash"), name="2^(1 − M) (Formel)"))
    if mrows:
        ys, errs = [], []
        for r in mrows:
            p = r["same_side"] / r["reps"]
            ys.append(p)
            errs.append(2 * math.sqrt(max(p * (1 - p), 1e-12) / r["reps"]))
        fig.add_trace(go.Scatter(x=[r["chains"] for r in mrows], y=ys, mode="markers", marker=dict(color=C.COLOR_DECEIVE, size=10),
                                 error_y=dict(type="data", array=errs, visible=True), name="gemessen (selbe Seite nach Verwerfen)"))
    fig.add_vline(x=chains_now, line=dict(color=C.COLOR_BLUE, width=1.5, dash="dot"))
    fig.update_xaxes(title_text="Zahl der Ketten M", dtick=1)
    fig.update_yaxes(title_text="Anteil der Wiederholungen", type="log")
    return _base(fig, height=300)
