"""Texte der Seite, die in App und README gleich lauten müssen (ein Test vergleicht sie mit der README)."""

POSTERIOR_LINK = "[`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo)"
TEMPERING_LINK = "[`gipfel-tempering-demo`](https://github.com/sebastian-hanisch/gipfel-tempering-demo)"
COUPLING_LINK = "[`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo)"
MIXING_LINK = "[`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo)"
ESS_LINK = "[`exakt-oder-abtasten-demo`](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo)"
EQUILIBRIUM_LINK = "[`gleichgewichtsauswahl-demo`](https://github.com/sebastian-hanisch/gleichgewichtsauswahl-demo)"

INTRO = f"""
Mehrere Ketten werden gestartet, und ein Maß, **R̂** („R-Dach“), sagt, ob sie fertig sind: sehen alle Ketten gleich aus, melden R̂ nahe 1 und eine hohe **ESS** (effektive Stichprobengröße)
Entwarnung. Die Diagnostik setzt aber stillschweigend voraus, dass die **Startpunkte alle Gipfel der Verteilung überdecken**. Starten alle Ketten im selben Gipfel, sehen sie auch dann gleich aus,
wenn ein zweiter Gipfel mit der halben Masse unbesucht bleibt. Die Seite zeigt das an zwei **ausgedachten** Modellen (einer Mischung zweier Gaußglocken und einem Ring von Stellplätzen), rechnet
die **exakte** Wahrscheinlichkeit, dass keine Kette wechselt, und zählt in vorgerechneten Wiederholungen, wie oft die Diagnostik trotzdem Entwarnung gibt. Die Formeln für R̂ und ESS stehen in
{POSTERIOR_LINK}; hier geht es nur um die Täuschung. Eine Fassung ohne Formeln öffnet „💬 In einfachen Worten“.

Weiter unten: vier **Experimente** (Experiment 1 ein Lauf live; die Experimente 2 bis 4 sind Zählungen aus vorgerechneten Läufen: wie oft die Diagnostik täuscht, wie lange man laufen muss, was die Startpunkte ändern; eine **Zelle** ist dabei eine feste Einstellung mit 200 Wiederholungen zu je vier Ketten), der Expander **Was hilft, was nicht?**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""

HOW = f"""
1. **Mehrere Ketten, ein Maß.** R̂ vergleicht die Streuung innerhalb der Ketten mit der zwischen den Ketten; sehen alle Ketten gleich aus, ist R̂ nahe 1. Die ESS zählt, wie viele unabhängige
   Ziehungen die Ketten wert sind. Heute gilt R̂ < 1,01 mit mindestens vier Ketten und ESS ≥ 400 (nach Vehtari u. a.); die ältere Schwelle 1,1 gilt als veraltet.
2. **Was das Maß nicht sehen kann:** Eine Kette, die nur einen Gipfel kennt, liefert keine Daten über den anderen. Sind alle Ketten im selben Gipfel gestartet und wechseln nicht, sehen sie gleich aus.
   Die Seite nennt das **Täuschung** (R̂ < 1,01 und kein Wechsel) und prüft zusätzlich die **heutige Regel** (R̂ < 1,01 und ESS ≥ 400, ebenfalls ohne Wechsel).
3. **Die Seite kennt das exakte Ziel.** Beide Modelle sind klein genug, um Übergangsmatrix, Wechselwahrscheinlichkeit und ESS exakt zu rechnen; dadurch lässt sich nachprüfen, was die Diagnostik in
   der Praxis nicht nachprüfen kann. **Beide Modelle sind ausgedacht**, es gibt keine reale Datenquelle.
"""

ASSUMPTIONS_MARKDOWN = f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Modelle sind ausgedacht** (Gaußmischung auf dem Gitter, Stellplatz-Ring), eine Dimension, vier Ketten, ohne Einbrennen, geteiltes R̂ < 1,01 | Die Zahlen gelten nur für diese Modelle und diese Einstellungen; sie sagen nichts über reale Datensätze. Höhere Dimensionen sind hier nicht gerechnet. | {POSTERIOR_LINK} (stetige Räume) |
| **Die Seite kennt das exakte Ziel** (den Gipfelanteil 0,5 und die Wechselwahrscheinlichkeit) | In der Praxis ist beides unbekannt; was dann als Hilfsdiagnose wirkt, ist hier **nicht** gemessen. Gemessen ist nur: Starts in beiden Gipfeln lösen den Alarm aus. | - |
| **Gleich große Gipfel gleicher Breite** | Ungleiche Gipfelgewichte, ungleiche Breiten und schmale Verbindungsgrate sind nicht gemessen. | - |
| **Die Zahlen sind Zählungen mit festen Zufalls-Seeds** (200 Wiederholungen je Zelle) | Eine andere Wiederholung gibt andere Zählungen; die Seite nennt Bänder (mehrere Zufalls-Seeds) und die exakte Erwartung für „kein Wechsel“. | - |
| **Die Abhilfe gehört nicht hierher** | Tempering und andere Verfahren, die zwischen Gipfeln wechseln, sind hier nicht gerechnet. | {TEMPERING_LINK} |
"""

MODEL_DISCLAIMER = ("Die Gaußmischung und der Stellplatz-Ring sind ausgedacht; in den gelesenen Quellen gibt es keine reale Messreihe, an der die Täuschung der Diagnostik gezeigt wird. "
                    "Die Zahlen gelten für diese Modelle, vier Ketten, ohne Einbrennen, mit der angegebenen Schrittweite und Kettenlänge. "
                    "Die Seite kennt das exakte Ziel und kann deshalb nachrechnen, was eine Diagnostik in der Praxis nicht kann.")

FORMULAS_MD = r"""
**Gaußmischung auf dem Gitter (vollständiges Modell).** Gitter $\mathcal G=\{-(D/2+7)+0{,}1\,k : k=0,\dots,K-1\}$ mit $K=\operatorname{round}(2(D/2+7)/0{,}1)+1$ Punkten; Ziel
$$\pi(x)=\frac{w(x)}{\sum_{y\in\mathcal G}w(y)},\qquad w(x)=\tfrac12 e^{-(x-D/2)^2/2}+\tfrac12 e^{-(x+D/2)^2/2}.$$
Vorschlag: Offset $k\in\{-k_{\max},\dots,k_{\max}\}$ mit $k_{\max}=\lceil 5\sigma/0{,}1\rceil$ und $q(k)\propto e^{-(0{,}1\,k)^2/(2\sigma^2)}$ (diskretisierte Normalverteilung, bei $5\sigma$ abgeschnitten); Vorschläge außerhalb des Gitters werden abgelehnt. Übergang für $y\ne x$:
$$P(x,y)=q\!\left(\tfrac{y-x}{0{,}1}\right)\min\{1,\ \pi(y)/\pi(x)\},\qquad P(x,x)=1-\sum_{y\ne x}P(x,y).$$
**Hard-Core-Ring (vollständiges Modell).** Zustände sind Belegungen $\eta\in\{0,1\}^n$ ($n$ gerade) ohne zwei benachbarte Einsen, $\pi(\eta)\propto\lambda^{|\eta|}$. Heat-bath: ein Platz $v$ gleichverteilt wählen; sind beide Nachbarn frei, wird $v$ mit Wahrscheinlichkeit $\lambda/(1+\lambda)$ belegt, sonst geleert; ist ein Nachbar belegt, wird $v$ geleert. Größe der Diagnostik
$$s(\eta)=\sum_{v\ \text{gerade}}\eta_v-\sum_{v\ \text{ungerade}}\eta_v .$$
**Startverteilung.** Gipfel A: Gaußmischung $x_0\sim$ diskretisiertes $N(-D/2,1)$, Ring alle geraden Plätze belegt; Gipfel B: $N(+D/2,1)$ bzw. alle ungeraden Plätze; gestreut: $N(0,\text{sd}^2)$ auf dem Gitter bzw. gleichverteilt über alle Belegungen.

**Wechsel (eine Definition für die ganze Seite).** Hysterese: der Gipfel B gilt als erreicht, sobald $x\ge D/4$ (Ring: $s\le-\lceil n/4\rceil$); der zuletzt erreichte Gipfel ändert sich erst dann. Die Zeit bis zum ersten Wechsel vom Gipfel A aus ist $T=\min\{t\ge1: X_t\in\mathcal B\}$.

**Exakte Größen.** Mit $Q$ als $P$ eingeschränkt auf die Zustände außerhalb von $\mathcal B$ und dem Startvektor $\mu_0$:
$$p=P(T>N)=\mu_0\,Q^{N}\,\mathbf 1,\qquad E[T]=\mu_0\,(I-Q)^{-1}\mathbf 1 .$$
Vier Ketten wechseln in $N$ Schritten alle nicht mit Wahrscheinlichkeit $p^4$; in 200 Wiederholungen ist die Erwartung $200\,p^4$ mit Standardabweichung $\sqrt{200\,p^4(1-p^4)}$. Relaxationszeit $t_{\mathrm{rel}}=1/(1-|\lambda|_*)$, $|\lambda|_*$ der betragsgrößte Eigenwert von $P$ außer 1.

**Gipfelanteil und MSE-ESS.** $I_B(x)=1$ für $x>0$, $\tfrac12$ für $x=0$, $0$ sonst (Ring: $s<0$, $s=0$, $s>0$); exakt $E_\pi I_B=\tfrac12$ (Symmetrie). Schätzer aus $M$ Ketten à $N$ Schritten und der Wert einer Schätzung in unabhängigen Ziehungen:
$$\hat I=\frac1{MN}\sum_{m=1}^{M}\sum_{t=1}^{N}I_B\big(X^{(m)}_t\big),\qquad N_{\mathrm{eff,MSE}}=\frac{\operatorname{Var}_\pi(I_B)}{\operatorname{MSE}},\quad \operatorname{MSE}=\frac1{R}\sum_{r=1}^{R}\big(\hat I_r-\tfrac12\big)^2 ,$$
über $R=200$ Wiederholungen.

**Exakte ESS einer stationären Kette** (für die Größe $f$ der Diagnostik, $M$ Ketten à $N$ Schritten, Autokorrelationen $\rho_k$ aus der Spektralzerlegung von $P$):
$$\mathrm{ESS}=\frac{MN}{\tau_N},\qquad \tau_N=1+2\sum_{k=1}^{N-1}\Big(1-\frac kN\Big)\rho_k .$$
**Wechselrate im Gleichgewicht.** Erweiterte Kette aus (Zustand, zuletzt erreichter Gipfel), stationäre Verteilung nach Grassmann, Taksar und Heyman (GTH); die Rate ist der Fluss in die Ankünfte, die den zuletzt erreichten Gipfel ändern. Sie unterscheidet sich von $1/E[T]$: $E[T]$ gilt für den Start im Gipfel A.

**Schwellen der Diagnostik (nur genannt, nicht hergeleitet).** Geteiltes $\hat R<1{,}01$ mit mindestens vier Ketten, $\mathrm{ESS}_{\mathrm{bulk}}\ge 400$ (Vehtari u. a. 2021); Gleichungen in `posterior-sampling-demo`.

**Literatur.** Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), Rank-normalization, folding, and localization: An improved $\hat R$ for assessing convergence of MCMC, Bayesian Analysis 16(2), arXiv 1903.08008 v5 (Schwelle 1,01, mindestens vier Ketten, Warnung bei getrennten Gipfeln); Gelman u. a., Bayesian Data Analysis, 3. Aufl., S. 285 bis 288 (Warnung, dass Startverteilung und Gipfel zusammenhängen); Vats, Knudson (2021), Revisiting the Gelman–Rubin diagnostic, Statistical Science, arXiv 1812.09384 (Schwelle 1,1 entspricht etwa 5 effektiven Stichproben je Kette); Jacob, O'Leary, Atchadé (2020), Unbiased Markov chain Monte Carlo with couplings, Abschnitt 5.1 (bimodales Beispiel, Aussage hängt an der Startverteilung).
"""
