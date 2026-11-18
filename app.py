"""Die Diagnostik sieht nicht alles: wo R̂ täuscht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Ketten, alle im selben Gipfel gestartet, melden R̂ < 1,01 und eine hohe ESS, obwohl ein Gipfel fehlt. Zwei ausgedachte Modelle (Gaußmischung auf dem
Gitter, Stellplatz-Ring), exakt nachgerechnet; ein Lauf live, die Zählungen aus vorgerechneten Läufen (`dgn_tables.json`, `dgn_exact.json`).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import dgn_constants as C
import dgn_evaluation as E
import dgn_format as F
import dgn_layout as L
import dgn_presets as P
import dgn_reference as R
import dgn_texts as T
import dgn_visualization as V
import modul_frisch

modul_frisch.aktualisieren(__file__)      # geänderte Hilfsmodule nach einem Update neu laden (Cloud behält sie sonst bis zum Neustart)

st.set_page_config(page_title="Die Diagnostik sieht nicht alles – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return E.analyse(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dgn_layout.py).
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Die Diagnostik sieht nicht alles: wo R̂ täuscht")
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/diagnose-taeuscht-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Vier Zufallswanderungen, im Fachwort **Ketten**, starten alle im selben von zwei gleich gewichtigen Gipfeln einer Verteilung. Nach einer Weile sehen sie gleich aus, und das übliche Maß für die Übereinstimmung, **R̂**, kann Entwarnung geben, obwohl keine Kette den zweiten Gipfel erreicht hat. Die Demo zählt an zwei ausgedachten Modellen, einer Mischung zweier Glockenkurven und Stellplätzen im Ring, wie oft das vorkommt: in der Mischung in 186 bis 194 von je 200 Wiederholungen (Gipfelabstand zehn Standardabweichungen, kleine Schritte, 2000 Schritte je Kette), im Ring in 51 bis 57 von 200. Der geschätzte Anteil des zweiten Gipfels liegt dann im Mittel fast bei null statt bei der Hälfte. Die Demo rechnet dazu die exakte Antwort aus und zeigt, was hilft und was nicht: Starts in beiden Gipfeln, verstreute Starts (mehr Ketten sind nur dort gemessen) und längere Läufe. Alles gilt nur in diesen Modellen.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/diagnose-taeuscht-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Diagnostik – und wo sie blind ist", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt die Einstellungen des ersten Experiments wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()

with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.caption("Die Regler des ersten Experiments stehen beim Experiment; ein Preset setzt sie.")

# --- Experiment 1: ein Lauf live -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1 – Vier Ketten in einer Mischung")
st.caption(
    "**Kernfrage 1: Melden die Ketten „fertig“, obwohl ein Gipfel fehlt?** Vier Ketten laufen live, jede mit eigenem Zufallsstrom (SplitMix64, gewählter Zufalls-Seed). "
    "Gipfel A ist links (Gaußmischung) bzw. „gerade Plätze belegt“ (Ring). Rechnet ein Lauf bis 20 000 Schritte je Kette, dauert er etwa eine Sekunde."
)
vehicle = st.radio("Modell", C.VEHICLES, key="vehicle", horizontal=True, help="Beide Modelle sind ausgedacht; es gibt keine reale Datenquelle.")
is_ring = vehicle == C.VEHICLES[1]
c1, c2, c3 = st.columns(3)
with c1:
    D = st.select_slider("Gaußmischung: Abstand D der Gipfel", options=C.D_CHOICES, key="D", disabled=is_ring,
                         help="In Standardabweichungen; 0 ist eine einzelne Standardnormale (Kontrolle ohne zweiten Gipfel).")
    sigma = st.slider("Gaußmischung: Schrittweite σ des Vorschlags", C.SIGMA_MIN, C.SIGMA_MAX, step=C.SIGMA_STEP, key="sigma", disabled=is_ring,
                      help="Standardabweichung des Normalvorschlags des Zufallsschritt-Metropolis.")
with c2:
    ring_n = st.select_slider("Ring: Zahl der Plätze n", options=C.RING_N_CHOICES, key="ring_n", disabled=not is_ring,
                              help="Gerade n; bis n = 12 wird live exakt gerechnet (n = 16 steht nur in den vorgerechneten Zellen).")
    lam = st.select_slider("Ring: Gewicht λ je belegtem Platz", options=C.LAM_CHOICES, key="lam", disabled=not is_ring,
                           help="Größeres λ macht die Gipfel (alle geraden bzw. alle ungeraden Plätze belegt) tiefer.")
with c3:
    N = st.select_slider("Schritte je Kette N", options=C.N_CHOICES, key="N", help="Es gibt kein Einbrennen: alle Schritte zählen.")
    start = st.radio("Start der Ketten", list(C.START_CHOICES), format_func=C.START_CHOICES.get, key="start",
                     help="Gestreut: Gaußmischung N(0, 10²), Ring gleichverteilt über alle Belegungen.")
c4, c5 = st.columns([2, 1])
with c4:
    seed = st.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="seed", step=1)
with c5:
    st.write("")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=P.randomize_seed)

P.sync_query_params({k: st.session_state[k] for k in ("vehicle", "D", "sigma", "ring_n", "lam", "N", "start", "seed")})

settings = E.Settings(vehicle, int(D), float(sigma), int(ring_n), int(lam), int(N), start, int(seed))
with st.spinner("Vier Ketten laufen …"):
    a = _analysis(settings)
f = a.facts

m1, m2, m3, m4 = st.columns(4)
m1.metric("R̂ (geteilt)", F.dez(f.rhat, 3), delta=("< 1,01: Entwarnung" if f.rhat_ok else "≥ 1,01: Warnung") if not f.undefined else "nicht definiert", delta_color="off")
m2.metric("ESS_bulk (angezeigt)", F.ganz(f.ess) if f.ess == f.ess else "–", delta=f"exakt, stationär: {F.dez(f.ess_exact, 1)}", delta_color="off")
m3.metric("Wechsel je Kette", " / ".join(str(x) for x in f.switches) if f.has_peaks else "–", delta="Hysterese (📐)", delta_color="off")
m4.metric("Anteil Gipfel B (geschätzt)", F.dez(f.share_B, 3), delta="exakt 0,5", delta_color="off")
if f.has_peaks and a.E_T is not None:
    exact_line = f"Exakt für dieses Modell: Relaxationszeit {F.ganz(a.t_rel)} Schritte, mittlere Zeit bis zum ersten Wechsel aus Gipfel A {F.ganz(a.E_T)} Schritte"
    if f.p_all is not None:
        exact_line += f"; vier Ketten wechseln in {F.ganz(f.N)} Schritten alle nicht mit Wahrscheinlichkeit {F.dez(100 * f.p_all, 1)} %"
    st.caption(exact_line + ".")
else:
    st.caption(f"Exakt für dieses Modell: Relaxationszeit {F.ganz(a.t_rel)} Schritte (nur ein Gipfel).")
st.plotly_chart(V.build_traces(a.values, a.thr, is_ring, a.kinds), width="stretch", key="traces_chart")
st.plotly_chart(V.build_share(a.steps, a.share_curve), width="stretch", key="share_chart")
st.caption("Die Spuren zeigen höchstens 2000 Punkte je Kette (jeder k-te Schritt); gerechnet wird mit allen Schritten. Der Anteil Gipfel B zählt die Schritte im rechten Gipfel (Gaußmischung x > 0) bzw. mit s < 0 (Ring).")

kind, text = E.verdict_run(f)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2: wie oft täuscht es ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2 – Wie oft täuscht es?")
st.caption(
    "**Kernfragen 2 und 4: Wie oft meldet die Diagnostik Entwarnung, obwohl ein Gipfel fehlt, und wie viele unabhängige Ziehungen ist die Schätzung wirklich wert?** "
    "Vorgerechnete Läufe: je Zelle 200 Wiederholungen mit je vier Ketten, alle im Gipfel A gestartet, feste Zufalls-Seeds (`dgn_tables.json`, `dgn_exact.json`); "
    "eine **Zelle** ist eine feste Einstellung (Gaußmischung mit D und σ oder Ring mit n und λ, dazu N und Zufalls-Seed), alle Zellen hier haben N = 2000 Schritte je Kette. "
    "Der Regler wählt nur aus dem Raster. Zähleinheit: Wiederholungen."
)
v2 = st.radio("Modell", C.VEHICLES, key="exp2_vehicle", horizontal=True)
cells = R.cell_list("ring" if v2 == C.VEHICLES[1] else "gauss")
rows = []
for rec in cells:
    cn, ce = rec["counts"], rec["exact"]
    rows.append({
        "Zelle": R.cell_label(rec),
        "kein Wechsel (gezählt)": cn.get("no_switch"),
        "kein Wechsel (exakt erwartet)": round(ce["expected"], 1) if "expected" in ce else None,
        "R̂ < 1,01": cn["rhat_lt"],
        "täuschend": cn.get("deceiving"),
        "heute": cn.get("today"),
        "ESS angezeigt": round(cn["ess_mean"]),
        "ESS exakt (stationär)": round(ce["ess_stationary"], 1),
        "N_eff,MSE": round(cn["n_eff_mse"], 2) if cn.get("n_eff_mse") else None,
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
labels = [R.cell_label(r) for r in cells]
choice = st.selectbox("Zelle für das Verdikt", labels, key=f"exp2_cell_{v2}")
cell = cells[labels.index(choice)]
st.plotly_chart(V.build_cell_bars(cell), width="stretch", key="cell_chart")
kind, text = E.verdict_cell(cell)
getattr(st, kind)(text)
if cell["counts"].get("n_eff_mse"):
    cn = cell["counts"]
    st.info(
        f"**Kernfrage 4:** Die Schätzung des Gipfelanteils aus {F.ganz(len(cell['kinds']) * cell['N'])} Zügen (vier Ketten × {F.ganz(cell['N'])} Schritte) ist im Mittel so viel wert wie "
        f"**{F.dez(cn['n_eff_mse'], 2)} unabhängige Ziehungen** aus π (N_eff,MSE = Var(I_B)/MSE über {F.ganz(cn['reps'])} Wiederholungen); angezeigt wird eine ESS_bulk von {F.ganz(cn['ess_mean'])}."
    )
if v2 == C.VEHICLES[0]:
    trio = [r for r in cells if r["a"] == 10 and r["b"] == 1.0]
    if len(trio) == 3:
        st.caption("Gaußmischung D = 10, σ = 1, N = 2000 in drei unabhängigen Läufen zu je 200 Wiederholungen: täuschend "
                   + " / ".join(str(r["counts"]["deceiving"]) for r in trio) + f" (erwartet höchstens {F.dez(trio[0]['exact']['expected'], 1)} ± {F.dez(trio[0]['exact']['sd'], 1)} „kein Wechsel“).")
st.caption("Die Täuschung hängt an der Schrittweite, nicht allein am Abstand der Gipfel: wo die Ketten wechseln (Schrittweite 2,4, oder Abstand 6 bei Schrittweite 1), liegt die exakte Erwartung für „kein Wechsel“ unter 0,1 von 200, und es gibt keine Täuschung (0 von 200).")

st.markdown("---")

# --- Experiment 3: Fenster und Wechselrate --------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3 – Wie lange muss man laufen?")
st.caption(
    "**Kernfrage 2 (zweiter Teil): Für welche Kettenlänge ist die Täuschung am größten?** Dieselben vorgerechneten Läufe, ausgewertet an den ersten N Schritten (Präfixe), "
    "200 Wiederholungen mit je vier Ketten. Die Marken zeigen 0,2 und 2 mittlere Zeiten bis zum ersten Wechsel E[T], die nur der Seite bekannt sind (exakt gerechnet). "
    "Gemessen ist das Fenster nur für die Gaußmischung mit D = 8, σ = 1 und den Ring mit n = 12, λ = 20; die 191 von 200 aus Experiment 2 gehören zu D = 10, das Maximum 75 nur zu D = 8."
)
v3 = st.radio("Modell", C.VEHICLES, key="exp3_vehicle", horizontal=True)
w = R.window("ring" if v3 == C.VEHICLES[1] else "gauss")
sp = w["spec"]
st.caption(f"Zelle: {'Ring n = ' + str(int(sp['a'])) + ', λ = ' + format(sp['b'], 'g') if v3 == C.VEHICLES[1] else 'Gaußmischung D = ' + format(sp['a'], 'g') + ', σ = ' + F.dez(sp['b'], 1)}, "
           f"Zufalls-Seed {sp['seed']}, E[T] = {F.ganz(w['E_T'])} Schritte, Relaxationszeit {F.ganz(w['t_rel'])} Schritte.")
st.plotly_chart(V.build_window(w), width="stretch", key="window_chart")
st.plotly_chart(V.build_window_share(w), width="stretch", key="window_share_chart")
kind, text = E.verdict_window(w)
getattr(st, kind)(text)
kind, text = E.verdict_rate(w)
getattr(st, kind)(text)
rc = R.rate_cell()
st.caption(
    f"Zum Vergleich bei großer Schrittweite (Gaußmischung D = 6, σ = 2,4): exakte Wechselrate {F.potenz(rc['exact']['rate_exact'], 3)} je Schritt; "
    f"gezählt {F.ganz(rc['switches']['count'])} Wechsel in {F.ganz(rc['switches']['steps'])} Schritten. Ob die Kette wechselt, hängt am Schritt, nicht allein am Abstand."
)

st.markdown("---")

# --- Experiment 4: Startpunkte ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4 – Was ändert der Start?")
st.caption(
    "**Kernfrage 3: Was ändern die Startpunkte?** Vorgerechnete Läufe (200 Wiederholungen je Zeile, N = 2000 Schritte je Kette; Gaußmischung D = 10, σ = 1 und Ring n = 12, λ = 50), und die Formel 2^(1 − M) live. "
    "„Gestreut“ heißt in der Gaußmischung: Startwerte aus N(0, sd²) mit sd 5 oder sd 10, im Ring gleichverteilt über alle Belegungen. "
    "„Selbe Seite“: alle vier Ketten bleiben nach dem Verwerfen (0 oder 500 Schritte) auf derselben Seite der Senke."
)
v4 = st.radio("Modell", C.VEHICLES, key="exp4_vehicle", horizontal=True)
srows = R.starts("ring" if v4 == C.VEHICLES[1] else "gauss")
st.dataframe(pd.DataFrame([{"Start": r["name"], "Verwerfen": r["burn"], "R̂ ≥ 1,01": r["rhat_ge"], "R̂ < 1,01": r["rhat_lt"], "selbe Seite": r["same_side"],
                            "selbe Seite und R̂ < 1,01": r["same_and_rhat_lt"]} for r in srows]), hide_index=True, width="stretch")
st.plotly_chart(V.build_starts(srows), width="stretch", key="starts_chart")
kind, text = E.verdict_starts(srows)
getattr(st, kind)(text)
M = st.slider("Zahl der Ketten M (gestreute Starts)", 2, 8, 4, key="exp4_M", help="Die Formel gilt für den Start (Symmetrie); vorgerechnet sind M = 2, 4 und 8.")
st.plotly_chart(V.build_formula(int(M), R.mcells()), width="stretch", key="formula_chart")
kind, text = E.verdict_formula(int(M))
getattr(st, kind)(text)
st.caption("Gemessen (Gaußmischung D = 10, σ = 1, gestreut aus N(0, 10²), nach Verwerfen von 500 Schritten, je " + F.ganz(R.mcells()[0]["reps"]) + " Wiederholungen): "
           + "; ".join(f"M = {r['chains']}: selbe Seite in {F.von(r['same_side'], r['reps'])}" for r in R.mcells()) + ". Für M = 2 empfiehlt die heutige Regel mindestens vier Ketten.")

st.markdown("---")

# --- Einordnung -------------------------------------------------------------------------------------------------------------------------------

with st.expander("Was hilft, was nicht?"):
    st.markdown("Nur gemessene Zeilen nennen eine Zahl; alles andere steht als „nicht gemessen“.")
    st.dataframe(pd.DataFrame(E.what_helps_rows()), hide_index=True, width="stretch")

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)
st.info(T.MODEL_DISCLAIMER)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/diagnose-taeuscht-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Ziel $\pi$ mit zwei getrennten Gipfeln und vier Ketten, die $\pi$ abtasten. Gesucht ist, ob die Diagnostik (R̂, ESS) einen fehlenden Gipfel bemerkt. Die Formeln von R̂ und ESS stehen in der Nachbar-Demo `posterior-sampling-demo`; hier stehen das Modell, die Wechselschwelle und die exakten Größen.

| Symbol | Bedeutung |
|---|---|
| $D$ | Abstand der beiden Gipfel der Gaußmischung (in Standardabweichungen) |
| $\sigma$ | Schrittweite des Vorschlags |
| $n,\ \lambda$ | Zahl der Plätze im Ring, Gewicht je belegtem Platz |
| $M,\ N$ | Zahl der Ketten (hier 4), Schritte je Kette |
| $\mathcal A,\ \mathcal B$ | Gebiete der beiden Gipfel (Wechselschwelle unten) |
| $T$ | Schritt der ersten Ankunft im Gebiet des anderen Gipfels |
| $p$ | Wahrscheinlichkeit, dass eine Kette in $N$ Schritten nicht wechselt |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/diagnose-taeuscht-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.FORMULAS_MD)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)
