"""Auswertung eines Laufs (vier Ketten) und die Verdikte der Seite als reine Funktionen: jede Verdikt-Funktion gibt (Art, Text) zurück, die App ruft
`getattr(st, Art)(Text)`; `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an von Hand gesetzten Werten."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dgn_chain as ch
import dgn_constants as C
import dgn_diagnostics as dg
import dgn_exact as ex
import dgn_reference as ref
import dgn_scenario as S
from dgn_format import dez, ganz, potenz, prozent, von


@dataclass(frozen=True)
class Settings:
    vehicle: str = C.DEFAULT_VEHICLE
    D: int = C.DEFAULT_D
    sigma: float = C.DEFAULT_SIGMA
    ring_n: int = C.DEFAULT_RING_N
    lam: int = C.DEFAULT_LAM
    N: int = C.DEFAULT_N
    start: str = C.DEFAULT_START
    seed: int = C.DEFAULT_SEED

    @property
    def is_ring(self) -> bool:
        return self.vehicle == C.VEHICLES[1]


def model_of(s: Settings):
    if s.is_ring:
        return ch.get_ring(s.ring_n, float(s.lam))
    return ch.get_grid(float(s.D), round(float(s.sigma), 3))


@lru_cache(maxsize=32)
def _bundle(key: tuple):
    """Exakte Größen eines Modells (einmal je Modell): Relaxationszeit und die Kette mit absorbierendem anderen Gipfel."""
    model = ch.get_ring(int(key[1]), float(key[2])) if key[0] == "ring" else ch.get_grid(float(key[1]), float(key[2]))
    gap, trel = ex.relaxation(model.P, model.pi)
    out = {"t_rel": trel}
    if model.thr > 0:
        kl = ex.killed_from_A(model)
        out["killed"] = kl
        out["E_T"] = kl.mean_exit()
    return out


@lru_cache(maxsize=128)
def _ess_exact(key: tuple, N: int) -> float:
    model = ch.get_ring(int(key[1]), float(key[2])) if key[0] == "ring" else ch.get_grid(float(key[1]), float(key[2]))
    return ex.exact_ess(model.P, model.pi, np.asarray(model.value, dtype=float), N)[0]


def count_changes(in_B: np.ndarray, in_A: np.ndarray, start_kind: str) -> int:
    """Wechsel einer Kette mit Hysterese: Änderungen des zuletzt erreichten Gipfels; der Startgipfel (A oder B) zählt als erster Gipfel, bei gestreutem Start der erste erreichte."""
    ev = in_B | in_A
    lab = in_B[ev]
    if start_kind == "A":
        lab = np.concatenate([[False], lab])
    elif start_kind == "B":
        lab = np.concatenate([[True], lab])
    return int(np.count_nonzero(lab[1:] != lab[:-1])) if lab.size > 1 else 0


@dataclass(frozen=True)
class Facts:
    """Alles, was die Verdikt-Funktion des Laufs braucht (reine Zahlen)."""

    vehicle: str
    has_peaks: bool
    N: int
    start: str
    rhat: float
    ess: float
    rhat_ok: bool
    today: bool
    undefined: bool
    seen_A: bool
    seen_B: bool
    switches: tuple
    share_B: float
    ess_exact: float
    p_all: float | None = None          # exakte Wahrscheinlichkeit, dass keine der vier Ketten wechselt (nur bei Start in A und B)
    frozen: tuple | None = None         # (täuschend, Wiederholungen, Zufalls-Seed) der eingefrorenen Zelle mit derselben Einstellung


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    values: np.ndarray                  # (4, N) Größe der Diagnostik
    kinds: tuple
    facts: Facts
    t_rel: float
    E_T: float | None
    first_cross: tuple                  # je Kette: erster Schritt im Gebiet des anderen Gipfels (None: nie)
    share_curve: np.ndarray             # geschätzter Anteil Gipfel B über die ersten n Schritte aller Ketten (n = Gitter von `steps`)
    steps: np.ndarray
    thr: float


def analyse(settings: Settings) -> Analysis:
    model = model_of(settings)
    kinds = ch.start_pattern(settings.start, C.CHAINS)
    path = ch.simulate(model, kinds, settings.N, settings.seed, reps=1)[0]
    values = model.to_value(path)
    d = dg.diagnose(values)
    has_peaks = model.thr > 0
    key = model.key
    bundle = _bundle(key)
    ind = model.to_indicator_B(path)
    stride = max(1, settings.N // C.PLOT_POINTS)
    steps = np.arange(stride, settings.N + 1, stride)
    cum = np.cumsum(ind.sum(axis=0))
    share_curve = cum[steps - 1] / (C.CHAINS * steps)
    if has_peaks:
        in_B, in_A = ch.region_flags(model, path)
        switches = tuple(count_changes(in_B[c], in_A[c], kinds[c]) for c in range(C.CHAINS))
        seen_A = bool(in_A.any() or "A" in kinds)
        seen_B = bool(in_B.any() or "B" in kinds)
        first_cross = []
        for c in range(C.CHAINS):
            region = in_A[c] if kinds[c] == "B" else in_B[c]
            first_cross.append(int(region.argmax()) + 1 if (kinds[c] in ("A", "B") and region.any()) else None)
        symmetric_start = all(k in ("A", "B") for k in kinds)
        p_all = bundle["killed"].surv(settings.N) ** C.CHAINS if symmetric_start else None
    else:
        switches, seen_A, seen_B, first_cross, p_all = (), True, False, (None,) * C.CHAINS, None
    frozen = None
    if settings.start == "alle_A" and not settings.is_ring:
        rec = ref.find_cell("gauss", float(settings.D), float(settings.sigma), settings.N)
        if rec and has_peaks:
            frozen = (rec["counts"]["deceiving"], rec["reps"], rec["seed"])
    elif settings.start == "alle_A":
        rec = ref.find_cell("ring", float(settings.ring_n), float(settings.lam), settings.N)
        if rec:
            frozen = (rec["counts"]["deceiving"], rec["reps"], rec["seed"])
    facts = Facts(
        vehicle=settings.vehicle, has_peaks=has_peaks, N=settings.N, start=settings.start, rhat=d["rhat"], ess=d["ess"], rhat_ok=d["rhat_ok"],
        today=d["today"], undefined=d["undefined"], seen_A=seen_A, seen_B=seen_B, switches=switches, share_B=float(ind.mean()),
        ess_exact=_ess_exact(key, settings.N), p_all=p_all, frozen=frozen,
    )
    return Analysis(settings, values, kinds, facts, bundle["t_rel"], bundle.get("E_T"), tuple(first_cross), share_curve, steps, float(model.thr))


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------

def _today_sentence(f: Facts) -> str:
    if f.undefined:
        return ""
    if f.today:
        return f"Die heutige Regel (R̂ < 1,01 und ESS_bulk ≥ {C.ESS_LIMIT}) gibt hier Entwarnung."
    if f.rhat_ok:
        return f"Die heutige Regel (R̂ < 1,01 und ESS_bulk ≥ {C.ESS_LIMIT}) gibt hier keine Entwarnung, weil ESS_bulk nur {ganz(f.ess)} beträgt."
    return ""


def verdict_run(f: Facts) -> tuple[str, str]:
    """Verdikt des Laufs (Experiment 1) als (Art, Text)."""
    if f.undefined:
        return "warning", ("R̂ ist hier nicht definiert: alle Hälften der Ketten sind konstant und gleich (Quotient 0/0) oder es gibt weniger als vier Schritte je Kette. "
                           "Die Seite meldet das, statt 1,0 anzuzeigen.")
    rh, es = dez(f.rhat, 2), ganz(f.ess)
    if not f.has_peaks:
        if f.rhat_ok:
            return "success", (f"Kontrolle ohne zweiten Gipfel: R̂ = {rh}, ESS_bulk {es}; eine stationär gestartete Kette gleicher Länge hätte exakt ESS {dez(f.ess_exact, 0)} "
                               f"(vier Ketten × {ganz(f.N)} Schritte). Hier stimmt die Entwarnung.")
        return "info", (f"Kontrolle ohne zweiten Gipfel: R̂ = {rh} ≥ 1,01 bei ESS_bulk {es} (exakte ESS einer stationären Kette gleicher Länge: {dez(f.ess_exact, 0)}; verschiedene Maße). "
                        "Es gibt nur einen Gipfel; die Warnung ist hier mit der kleinen ESS vereinbar und zeigt keinen zweiten Gipfel.")
    both = f.seen_A and f.seen_B
    n_sw = sum(f.switches)
    share = dez(f.share_B, 3)
    if both and not f.rhat_ok:
        if n_sw == 0:
            return "info", (f"R̂ = {rh} ≥ 1,01: die Diagnostik schlägt an, weil die Startpunkte beide Gipfel überdecken. Die Ketten wechseln nicht; "
                            f"ohne diesen Start sähe sie nichts.")
        return "info", (f"R̂ = {rh} ≥ 1,01: die Diagnostik schlägt an; die Ketten haben beide Gipfel besucht ({ganz(n_sw)} Wechsel zusammen), "
                        f"sind aber noch nicht gemischt. Geschätzter Anteil Gipfel B {share}, exakt 0,5.")
    if both and f.rhat_ok:
        return "success", (f"Die Ketten haben beide Gipfel besucht ({ganz(n_sw)} Wechsel zusammen), R̂ = {rh}, ESS_bulk {es}: hier stimmt die Entwarnung. "
                           f"Geschätzter Anteil Gipfel B {share}, exakt 0,5. {_today_sentence(f)}").strip()
    if not f.rhat_ok:
        return "warning", (f"R̂ = {rh} ≥ 1,01, obwohl keine Kette den anderen Gipfel besucht hat und kein Wechsel stattfand: die Warnung ist hier mit "
                           f"der kleinen ESS_bulk {es} vereinbar und zeigt keinen zweiten Gipfel. Geschätzter Anteil Gipfel B {share}, exakt 0,5.")
    text = (f"Täuschung: R̂ = {rh}, ESS_bulk {es}, kein Wechsel; der geschätzte Anteil Gipfel B ist {share}, der exakte 0,5. "
            f"Die exakte ESS der gemessenen Größe bei einer stationären Kette gleicher Länge wäre {dez(f.ess_exact, 1)} (verschiedene Maße). {_today_sentence(f)}")
    if f.p_all is not None:
        text += f" Nach der exakten Rechnung bleiben vier solche Ketten in {dez(100 * f.p_all, 1)} % aller Wiederholungen alle im Gipfel A ({dez(200 * f.p_all, 1)} von 200)."
    if f.frozen:
        text += f" Gezählt in 200 vorgerechneten Wiederholungen derselben Einstellung (Zufalls-Seed {f.frozen[2]}): täuschend in {von(f.frozen[0], f.frozen[1])}."
    return "error", text.strip()


def verdict_cell(cell: dict) -> tuple[str, str]:
    """Verdikt einer eingefrorenen Zelle (Experiment 2): Zählungen in Wiederholungen gegen die exakte Erwartung für „kein Wechsel“."""
    c, e = cell["counts"], cell["exact"]
    reps = c["reps"]
    name = ref.cell_label(cell)
    if "deceiving" not in c:
        return "info", (f"{name}: Kontrolle ohne zweiten Gipfel. R̂ < 1,01 in {von(c['rhat_lt'], reps)} Wiederholungen; ESS_bulk im Mittel {ganz(c['ess_mean'])} gegen "
                        f"exakt {dez(e['ess_stationary'], 0)} für eine stationäre Kette gleicher Länge (vier Ketten × {ganz(c['N'])} Schritte).")
    dec = c["deceiving"]
    exp, sd = e["expected"], e["sd"]
    core = (f"{name}: kein Wechsel in {von(c['no_switch'], reps)} Wiederholungen (exakte Erwartung {dez(exp, 1)} ± {dez(sd, 1)}), R̂ < 1,01 in {von(c['rhat_lt'], reps)}, "
            f"täuschend (beides) in {von(dec, reps)}, nach heutiger Regel (zusätzlich ESS_bulk ≥ {C.ESS_LIMIT}) in {von(c['today'], reps)}. "
            f"ESS_bulk im Mittel {ganz(c['ess_mean'])} gegen exakt {dez(e['ess_stationary'], 1)} für eine stationäre Kette gleicher Länge.")
    if dec == 0:
        return "success", core + f" Keine einzige Täuschung in {ganz(reps)} Wiederholungen (Zählung, keine Obergrenze)."
    if dec >= reps / 2:
        return "error", core
    return "warning", core


def verdict_window(w: dict) -> tuple[str, str]:
    """Verdikt des Fensters (Experiment 3) aus den Zeilen je Kettenlänge: größte Täuschung, Länge mit lauter Alarmen und der Anteil dort."""
    rows = w["rows"]
    reps = w["spec"]["reps"]
    et = w["E_T"]
    peak = max(rows, key=lambda r: r["deceiving"])
    full = [r for r in rows if r["alarm"] == reps]
    last = rows[-1]
    text = (f"Größte Täuschung bei N = {ganz(peak['N'])} ({dez(peak['N'] / et, 2)}-fache mittlere Zeit bis zum ersten Wechsel, E[T] = {ganz(et)}): "
            f"täuschend in {von(peak['deceiving'], reps)} Wiederholungen. ")
    if full:
        text += (f"Von N = {ganz(full[0]['N'])} ({dez(full[0]['N'] / et, 2)} E[T]) bis N = {ganz(full[-1]['N'])} meldet R̂ ≥ 1,01 in {von(reps, reps)} Wiederholungen Alarm, "
                 f"der geschätzte Anteil Gipfel B ist bei N = {ganz(full[0]['N'])} noch {dez(full[0]['share_B_mean'], 3)} statt 0,5. ")
    text += (f"Bei N = {ganz(last['N'])} ({dez(last['N'] / et, 1)} E[T]) liegt der Anteil bei {dez(last['share_B_mean'], 3)}, R̂ < 1,01 in {von(last['rhat_lt'], reps)}.")
    return "info", text


def verdict_rate(w: dict) -> tuple[str, str]:
    """Gegenprobe der Täuschung: exakte Wechselrate im Gleichgewicht gegen die gezählten Wechsel (Experiment 3)."""
    sw = w["switches"]
    r_ex = w["rate_exact"]
    ratio = sw["rate"] / r_ex
    band = 4 / math.sqrt(sw["count"]) if sw["count"] else float("inf")
    inside = abs(ratio - 1) <= band
    text = (f"Exakte Wechselrate im Gleichgewicht: {potenz(r_ex, 3)} je Schritt, das ist im Mittel ein Wechsel alle {ganz(1 / r_ex)} Schritte. "
            f"Gezählt in den {ganz(sw['steps'])} Schritten aller Läufe: {ganz(sw['count'])} Wechsel, Verhältnis gezählt zu exakt {dez(ratio, 3)} "
            f"({'innerhalb' if inside else 'außerhalb'} von 4/√Wechsel = {dez(band, 3)}).")
    return ("success" if inside else "warning"), text


def verdict_starts(rows: list[dict], reps_note: bool = True) -> tuple[str, str]:
    """Verdikt des Startexperiments (Experiment 4) aus den Zeilen der Tabelle (Gaußmischung)."""
    by = {(r["name"], r["burn"]): r for r in rows}
    one = by[("drei in A, eine in B", 0)]
    two = by[("zwei in A, zwei in B", 0)]
    scat = [r for r in rows if r["name"].startswith("gestreut") and r["burn"] > 0]
    same, tot, ok = sum(r["same_side"] for r in scat), sum(r["reps"] for r in scat), sum(r["same_and_rhat_lt"] for r in scat)
    se = math.sqrt(0.125 * 0.875 / tot)
    return "info", (f"Eine Kette im anderen Gipfel: R̂ ≥ 1,01 in {von(one['rhat_ge'], one['reps'])} Wiederholungen (zwei und zwei: {von(two['rhat_ge'], two['reps'])}). "
                    f"Gestreute Starts: in {von(same, tot)} Wiederholungen ({prozent(same / tot, 1)}) liegen alle vier Ketten nach dem Verwerfen auf derselben Seite "
                    f"(erwartet für den Start 2^(1 − 4) = 12,5 % ± {dez(se * 100, 1)} Prozentpunkte Standardfehler), und dort sieht R̂ nichts: R̂ < 1,01 in {ganz(ok)} dieser {ganz(same)} Wiederholungen.")


def same_side_probability(chains: int) -> float:
    """2^(1 - M): Wahrscheinlichkeit, dass M unabhängig und gleichwahrscheinlich gestartete Ketten alle im selben Gipfel beginnen (Symmetrie)."""
    return S.expected_same_side(chains)


def verdict_formula(chains: int) -> tuple[str, str]:
    p = same_side_probability(chains)
    note = "mindestens vier Ketten empfiehlt die heutige Regel" if chains < C.CHAINS else "die Zahl ist die Erwartung für den Start, nicht für das Ende"
    return "info", (f"Bei M = {chains} gestreut gestarteten Ketten liegen alle im selben Gipfel mit Wahrscheinlichkeit 2^(1 − {chains}) = {dez(p, 4)} ({prozent(p, 2)}): {note}.")


def what_helps_rows() -> list[dict]:
    """Zeilen der Einordnung „Was hilft, was nicht?“: nur gemessene Zeilen nennen eine Zahl (Gaußmischung D = 10, σ = 1, N = 2000, wenn nicht anders genannt)."""
    cells = [r for r in ref.cell_list("gauss") if r["a"] == 10 and r["b"] == 1.0]
    gst = {r["name"]: r for r in ref.starts("gauss") if r["burn"] == 0}
    rst = {r["name"]: r for r in ref.starts("ring") if r["burn"] == 0}
    w = ref.window("gauss")
    reps = w["spec"]["reps"]
    full = [r for r in w["rows"] if r["alarm"] == reps]
    toda = " / ".join(str(r["counts"]["today"]) for r in cells)
    dec = " / ".join(str(r["counts"]["deceiving"]) for r in cells)
    rank = " / ".join(str(r["counts"]["deceiving_rank"]) for r in cells)
    one, two = gst["drei in A, eine in B"], gst["zwei in A, zwei in B"]
    rone, rtwo = rst["drei in A, eine in B"], rst["zwei in A, zwei in B"]
    rows = [
        {"Maßnahme": "Starts in beiden Gipfeln", "Befund": "hilft; setzt aber Wissen über die Gipfel voraus",
         "Zahl": f"R̂ ≥ 1,01 in {von(one['rhat_ge'], one['reps'])} (drei in A, eine in B) und {von(two['rhat_ge'], two['reps'])} (zwei und zwei); Ring n = 12, λ = 50: "
                 f"{von(rone['rhat_ge'], rone['reps'])} und {von(rtwo['rhat_ge'], rtwo['reps'])}"},
        {"Maßnahme": "Längere Ketten", "Befund": ((f"hilft bei rund dem Zwei- und Vierfachen von E[T] ({dez(full[0]['N'] / w['E_T'], 2)} und {dez(full[-1]['N'] / w['E_T'], 2)}), "
                                                 "bei 50 000 und 100 000 Schritten schlägt R̂ nur noch in 198 von 200 an; der Anteil bleibt noch schief; E[T] ist in der Praxis unbekannt") if full
                                                else "hilft nicht in jeder Wiederholung; der Anteil bleibt noch schief; E[T] ist in der Praxis unbekannt"),
         "Zahl": (f"Gaußmischung D = 8, σ = 1 (E[T] = {ganz(w['E_T'])}): R̂ ≥ 1,01 in {von(reps, reps)} von N = {ganz(full[0]['N'])} bis N = {ganz(full[-1]['N'])}, Anteil Gipfel B dort "
                  f"{dez(full[0]['share_B_mean'], 3)} bis {dez(full[-1]['share_B_mean'], 3)} statt 0,5") if full else "nicht in jeder Wiederholung"},
        {"Maßnahme": "Heutige Regel (zusätzlich ESS_bulk ≥ 400)", "Befund": "hilft hier nicht; eine strengere R̂-Schwelle allein ist nicht gemessen",
         "Zahl": f"heute {toda} gegen täuschend {dec} (drei Läufe, D = 10, σ = 1, N = 2000)"},
        {"Maßnahme": "Rangnormalisiertes R̂", "Befund": "hilft hier nicht", "Zahl": f"täuschend mit rangnormalisiertem R̂ {rank} gegen {dec} mit geteiltem R̂ (dieselben drei Läufe)"},
        {"Maßnahme": "Tempering", "Befund": "Abhilfe, die zwischen den Gipfeln wechselt", "Zahl": "hier nicht gerechnet (gipfel-tempering-demo)"},
        {"Maßnahme": "Hilfsdiagnosen (Wechselzähler, Anteil im Gipfel)", "Befund": "nicht gemessen", "Zahl": "–"},
    ]
    return rows
