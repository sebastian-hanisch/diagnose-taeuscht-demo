# Die Diagnostik sieht nicht alles: wo R̂ täuscht – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-diagnose-taeuscht-demo.streamlit.app/)** -->

Eine von mehreren Demos zu **MCMC: Verteilungen abtasten** (Simulation) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Demo handelt von der Frage, **woher man weiß, dass eine Kette fertig ist**,
wenn man nur ihre Ausgabe hat: mehrere Ketten, das Maß **R̂** und die **ESS**. Die Nachbar-Demos [`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo),
[`kopplung-ketten-demo`](https://github.com/sebastian-hanisch/kopplung-ketten-demo) und [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo) besitzen dafür ein **Zertifikat** (die exakte Mischzeit aus Matrixpotenzen,
die Kopplungszeit, eine Ausgabe ohne Näherung); hier gibt es nur die Ausgabe der Ketten und eine Diagnostik darauf, und die setzt voraus, dass die Startpunkte alle Gipfel überdecken. Die Abhilfe (Tempering) steht in
[`gipfel-tempering-demo`](https://github.com/sebastian-hanisch/gipfel-tempering-demo); die Formeln von R̂ und ESS und die Autokorrelationszeit stehen in [`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo)
und werden hier nicht wiederholt; die Konvention der exakten ESS steht in [`exakt-oder-abtasten-demo`](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo). Dieselbe Erfahrung ohne Diagnostik (die Simulation bleibt in einem Becken) zeigt
[`gleichgewichtsauswahl-demo`](https://github.com/sebastian-hanisch/gleichgewichtsauswahl-demo).

Die Gaußmischung und der Stellplatz-Ring sind ausgedacht; in den gelesenen Quellen gibt es keine reale Messreihe, an der die Täuschung der Diagnostik gezeigt wird. Die Zahlen gelten für diese Modelle, vier Ketten, ohne Einbrennen, mit der angegebenen Schrittweite und Kettenlänge. Die Seite kennt das exakte Ziel und kann deshalb nachrechnen, was eine Diagnostik in der Praxis nicht kann.

## Warum dieses Problem

Wer mehrere Ketten laufen lässt und R̂ < 1,01 sowie eine hohe ESS sieht, hält die Kette für fertig. **Die Falle:** R̂ vergleicht die Ketten untereinander; liegen alle im selben Gipfel, sieht es keinen Unterschied. Antwort in diesem Modell,
Zähleinheit Wiederholungen (je 200 Wiederholungen mit vier Ketten): In der Gaußmischung mit Abstand 10 Standardabweichungen, Schrittweite 1 und 2000 Schritten je Kette melden vier im linken Gipfel gestartete Ketten in **191 / 186 / 194 von je 200**
Wiederholungen (drei Zufalls-Seeds) „R̂ < 1,01“ und besuchen den rechten Gipfel nie (gemessen als: kein Schritt mit x ≥ D/4); die angezeigte ESS_bulk ist im Mittel 939 / 920 / 955, die exakte ESS der Größe x einer stationär gestarteten Kette gleicher Länge ist 4,2 (verschiedene Maße), und die Schätzung des Gipfelanteils ist so viel wert wie rund **1,0 unabhängige Ziehungen**
(1,01 / 1,03 / 1,00). Die heutige Regel (R̂ < 1,01 und ESS_bulk ≥ 400) ändert daran nichts (191 / 186 / 194). Im Stellplatz-Ring ist die Täuschung kleiner (51 bis 57 von 200 bei n = 12, λ = 50 und n = 16, λ = 100, je 2000 Schritte), weil R̂ dort schon durch die langsame Bewegung im Gipfel in 143 bis 149 von 200 Wiederholungen anschlägt, auch ohne Wechsel (Deutung); die heutige Regel schlägt zusätzlich
wegen der kleinen ESS an, nicht weil sie den zweiten Gipfel sieht. Startet auch nur eine der vier Ketten im anderen Gipfel, ist R̂ ≥ 1,01 in 200 von 200 Wiederholungen (gemessen in zwei Modellen mit nicht wechselnden Ketten). Die Täuschung hat ein Fenster: am größten bei Ketten in der Größenordnung eines Fünftels der mittleren Zeit bis zum ersten Wechsel (Maximum 75 bei N = 1000 im Raster aus zehn Längen, ein Zufalls-Seed; das Fenster ist nur für die Gaußmischung mit D = 8, σ = 1 und den Ring mit n = 12, λ = 20 gemessen, die 191 von 200 oben gehören dagegen zu D = 10),
bei Ketten von rund dem Zweifachen dieser Zeit meldet R̂ in 200 von 200 Wiederholungen einen Fehler (ebenso beim Vierfachen; gemessen bei 1,96 und 3,92 E[T]), und der geschätzte Gipfelanteil ist dann noch schief. Alles gilt in diesen Modellen, bei diesen Einstellungen; die Zähleinheit steht je Aussage dabei.
**Lehrbuchwissen** ist, dass R̂ mehrere über die Gipfel gestreute Starts voraussetzt; die Zahlen sind eigene Messungen in diesen Modellen. Nichts davon ist inhaltlich neu; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Gegenzahlen.

## Modell

* **Gaußmischung auf dem Gitter:** Ziel ½ N(−D/2, 1) + ½ N(+D/2, 1), eine Dimension, Gitterabstand 0,1, Bereich ±(D/2 + 7), Zufallsschritt-Metropolis mit einem diskretisierten, bei 5σ abgeschnittenen Normalvorschlag der Schrittweite σ; Vorschläge außerhalb des Gitters werden abgelehnt.
  Das ist eine **Näherung der stetigen Kette** (die stetige Kette ist hier nicht gerechnet); dafür ist sie exakt rechenbar (dichte Matrix, bei D = 10 mit 241 Punkten). Gipfel A ist der linke, Gipfel B der rechte.
* **Hard-Core-Ring (ausgedachtes Modell: Stellplätze im Ring, keine zwei benachbarten belegt):** n gerade, Gewicht λ^k bei k belegten Plätzen, **Heat-bath-Kette** (Platz gleichverteilt wählen, bei freien Nachbarn mit Wahrscheinlichkeit λ/(1 + λ) belegen, sonst leeren).
  Größe der Diagnostik s = belegte gerade minus belegte ungerade Plätze; Gipfel A: alle geraden Plätze belegt, Gipfel B: alle ungeraden.
* **Wechsel (eine Definition für die ganze Seite):** Hysterese. Der Gipfel B gilt als erreicht, sobald x ≥ D/4 (Ring: s ≤ −⌈n/4⌉); der zuletzt erreichte Gipfel ändert sich erst dann. Daraus zwei Größen, die jede Zahl mit ihrem Namen nennt: die **Zeit bis zum ersten Wechsel** E[T] vom Gipfel A aus
  und die **Wechselrate im Gleichgewicht** (alle Wechsel in beiden Richtungen). **Täuschend** heißt: R̂ < 1,01 und keine der vier Ketten wechselte. **Heute** heißt: R̂ < 1,01 und ESS_bulk ≥ 400 bei vier Ketten, und kein Wechsel (Regel nach Vehtari u. a.; die ältere Schwelle 1,1 gilt als veraltet).
* **Ketten × Schritte** meint vier Ketten mit N Schritten je Kette, **ohne Einbrennen**, wenn nicht anders genannt; R̂ ist das geteilte R̂ (jede Kette in zwei Hälften). Die ESS ist die exakte ESS in Zügen (Ketten × Schritte).
* Der 📐-Expander der App enthält die Formeln als vollständiges Modell (ohne Beweise); R̂ und ESS werden dort nur genannt.

## Methodik

* **Zufall** nur über den ganzzahligen SplitMix64 (`dgn_rng.py`, geprüfte Vorlage, unverändert kopiert): gleiche Zufalls-Seeds geben in den Tests dieselben Ketten (die Ziehung ist reine Ganzzahlarithmetik; dass die Ketten auf jeder Plattform bitgleich sind, ist nicht eigens geprüft, nur die Zählungen werden unten im Band von 2 neu gerechnet). Jede Kette hat einen eigenen Strom (Zähler-Generator: alle Ziehungen eines Blocks werden auf einmal berechnet);
  eine rein sequentielle Fassung in den Tests gibt bitgleich dieselben Pfade (`tests/test_chain.py`). Der Zufalls-Seed der Seite ist der Basis-Seed; die Ketten sind (Wiederholung, Kette) abgeleitet.
* **Experimente und Zellen:** Die App hat vier Experimente: Experiment 1 ist ein Lauf mit vier Ketten, Experiment 2 zählt, wie oft die Diagnostik täuscht, Experiment 3 zeigt das Fenster der Kettenlänge und die Wechselrate, Experiment 4 die Startpunkte. Eine **Zelle** ist eine Einstellung (Gaußmischung mit D und σ oder Ring mit n und λ, dazu Kettenlänge N und Zufalls-Seed) mit 200 Wiederholungen zu je vier Ketten.
* **Live** läuft Experiment 1 (vier Ketten, bis 20 000 Schritte je Kette) mit allen exakten Gegenzahlen (Eigenzerlegung der dichten Matrix: Gitter bis 261 Punkte, Ring bis n = 12 mit 322 Belegungen).
* **Vorgerechnet** sind die Zählungen der Experimente 2 bis 4 (`dgn_tables.json`, Skript `tools/dgn_make_tables.py`, 200 Wiederholungen je Zelle, feste Zufalls-Seeds) und ihre exakten Gegenzahlen (`dgn_exact.json`, darin der Ring mit n = 16: 2207 Belegungen, dichte Matrix). `python tools/dgn_make_tables.py --check` rechnet eine Auswahl neu und vergleicht
  (Zählungen im Band von 2, exakte Größen relativ 1e-6), `--check --lang` rechnet alle Zellen (einige Minuten).
* **Exakte Größen:** Überlebenswahrscheinlichkeit „kein Wechsel“ und E[T] aus der Kette mit absorbierendem Gebiet des anderen Gipfels (Eigenzerlegung); exakte ESS einer stationären Kette aus der Spektralzerlegung; Wechselrate im Gleichgewicht aus der erweiterten Kette (Zustand, zuletzt erreichter Gipfel) mit dem GTH-Algorithmus;
  Relaxationszeit t_rel = 1/(1 − |λ|*). Die Erwartung für „kein Wechsel“ in 200 Wiederholungen ist 200 · p⁴ mit p = exakte Überlebenswahrscheinlichkeit einer Kette.
* **MSE-ESS:** N_eff,MSE = Var(I_B)/MSE über 200 Wiederholungen; I_B ist der Gipfelindikator, der exakte Anteil ist ½ (Symmetrie).
* **Rechenzeit** (Messung dieser README, nicht im Test; eine CPU, Windows, Python 3.14, unter Last anderer Prozesse): Experiment 1 braucht bei 20 000 Schritten je Kette 0,5 bis 1,1 Sekunden (Gaußmischung D = 12, σ = 3 und Ring n = 12 eingeschlossen; die erste Rechnung eines neuen Modells mit Eigenzerlegung eingeschlossen), bei 2000 Schritten etwa 0,1 Sekunden; die Streamlit Cloud ist langsamer. Die Experimente 2 bis 4 lesen nur die eingefrorenen Dateien.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Wiederholungen (je vier Ketten), wenn nicht anders genannt; die Spalte „Test“ nennt den Test in `tests/test_claims.py`, der genau diese Zahl prüft (Ganzzahlen und Brüche exakt, Zählungen aus den eingefrorenen Läufen gegen die Dateien, einzelne Zellen neu gerechnet im Band von 2; exakte Größen relativ 1e-6).
Zufallszählungen sind Bänder: eine andere Wiederholung gibt andere Zählungen.

| Frage | Befund | Test |
|---|---|---|
| Täuscht R̂ in der Gaußmischung? (D = 10, σ = 1, N = 2000, Zufalls-Seeds 18101 / 18102 / 18103) | täuschend **191 / 186 / 194** von je 200; kein Wechsel 195 / 190 / 197, exakt erwartet 191,4 ± 2,9; heute 191 / 186 / 194; rangnormalisiertes R̂ < 1,01 in 191 / 186 / 195 | `test_c1` |
| Was zeigt die ESS dort? | angezeigte ESS_bulk im Mittel 939 / 920 / 955, exakt für eine stationäre Kette gleicher Länge 4,2 (vier Ketten × 2000 Schritte; die beiden ESS sind verschiedene Maße, ESS_bulk rangnormalisiert, die exakte ESS für die Größe x); N_eff,MSE 1,01 / 1,03 / 1,00 aus 8000 Zügen; geschätzter Anteil Gipfel B im Mittel 0,003 / 0,008 / 0,001 statt 0,5 | `test_c2` |
| Hängt es am Abstand und an der Schrittweite? | D = 8, σ = 1: kein Wechsel 43 (exakt erwartet 41,8 ± 5,8), täuschend 43, ESS 245 gegen exakt 5,4, N_eff,MSE 1,76. **Wo die Ketten wechseln, ist es ehrlich:** D = 6, σ = 1, D = 10, σ = 2,4 und D = 8, σ = 2,4: täuschend 0 von je 200 (exakte Erwartung für „kein Wechsel“ unter 0,1 von 200; bei D = 8, σ = 2,4 meldet R̂ < 1,01 trotz Wechseln in 4 von 200) | `test_c3` |
| Und im Ring? (N = 2000) | n = 12, λ = 50 (Zufalls-Seeds 18201 / 18202): täuschend **57 / 51**, heute 26 / 21, ESS 198 / 176 gegen exakt 4,18, kein Wechsel 175 / 179 (exakt erwartet 179,7 ± 4,3), N_eff,MSE 1,06 / 1,05. n = 16, λ = 100 (18204 / 18205): täuschend 51 / 51, heute 7 / 8, ESS 215 / 227 gegen exakt 4,04, kein Wechsel 200 / 200 (exakt erwartet 199,9) | `test_c4` |
| Kontrolle ohne zweiten Gipfel | Gaußmischung D = 0 (eine Standardnormale; σ = 1, N = 2000, nur für σ = 1 ist die exakte ESS 966): R̂ < 1,01 in 199 von 200, ESS_bulk 975 gegen exakt 966. Ring n = 16, λ = 1 (Größe s, 4 × 2000): R̂ < 1,01 nur in 1 von 200, ESS 72 gegen exakt 77: die Warnung im Ring ist dort mit der kleinen ESS vereinbar, ein Gipfelbefund ist sie nicht (kein Wechsel 0 von 200; ob die kleine ESS der Grund ist, ist nicht getrennt gemessen) | `test_c5` |
| Das Fenster (Gaußmischung D = 8, σ = 1, E[T] = 5107, Zufalls-Seed 18301) | täuschend 2 / 12 / 52 / **75** / 38 / 2 / 0 / 0 / 0 / 0 bei N = 100 / 200 / 500 / 1000 / 2000 / 5000 / 10 000 / 20 000 / 50 000 / 100 000; bei N = 10 000 und 20 000 (1,96 und 3,92 E[T]) meldet R̂ ≥ 1,01 in 200 von 200 Wiederholungen, der geschätzte Anteil Gipfel B ist 0,367 und 0,422; bei N = 100 000 0,480 und R̂ < 1,01 in 2 von 200 | `test_c6` |
| Das Fenster im Ring (n = 12, λ = 20, N = 100 bis 50 000, E[T] = 5369, 18302) | größte Zahl täuschend 5 von 200 bei N = 1000; Anteil Gipfel B 0,103 bei N = 2000 und 0,473 bei N = 50 000 | `test_c6` |
| Exakte Wechselrate im Gleichgewicht | Gaußmischung D = 8, σ = 1: 1,96·10⁻⁴ je Schritt (ein Wechsel alle 5090 Schritte); gezählt 15 502 Wechsel in 80 000 000 Schritten, Verhältnis 0,986 (Band 4/√Wechsel = 0,032). D = 6, σ = 2,4: 4,02·10⁻² je Schritt, gezählt 643 551 Wechsel in 16 000 000 Schritten, Verhältnis 1,001. Ring n = 12, λ = 20: 2,53·10⁻⁴, gezählt 10 135 in 40 000 000, Verhältnis 1,001 | `test_c7` |
| Was ändert der Start? (Gaußmischung D = 10, σ = 1, N = 2000) | drei in A, eine in B und zwei in A, zwei in B: R̂ ≥ 1,01 in **200 von 200** (Ring n = 12, λ = 50: ebenso 200 von 200). Gestreute Starts (Startwerte aus N(0, sd²) mit sd 5 und sd 10, je 200 Wiederholungen), nach Verwerfen von 500 Schritten: alle vier Ketten auf derselben Seite in 21 und 21 von je 200 (42 von 400, erwartet 2^(1−4) = 12,5 % ± 1,7 Prozentpunkte Standardfehler gegen gemessene 10,5 %), und dort R̂ < 1,01 in 41 dieser 42 | `test_c8` |
| Mehr oder weniger Ketten? (gestreut aus N(0, 10²), nach Verwerfen von 500, je 1000 Wiederholungen) | selbe Seite in 493 von 1000 bei M = 2 (Erwartung 2^(1−2) = 0,5), 128 bei M = 4 (0,125) und 9 bei M = 8 (2^(1−8) = 0,0078) | `test_c9` |
| Hilft die heutige Regel (zusätzlich ESS_bulk ≥ 400) oder das rangnormalisierte R̂? (Eine strengere R̂-Schwelle allein ist nicht gemessen.) | nein, in diesen drei Läufen: heute 191 / 186 / 194 gegen täuschend 191 / 186 / 194; mit rangnormalisiertem R̂ täuschend 191 / 186 / 194 | `test_c1` |
| Exakte Größen der Zellen | Relaxationszeit t_rel und E[T]: D = 8, σ = 1: 2559 und 5107; D = 10, σ = 1: 91 306 und 182 560; Ring n = 12, λ = 50: 16 206 und 26 920; n = 12, λ = 20: 2887 und 5369; n = 16, λ = 100: 83 208 und 138 287 | `test_c10` |
| Handbeispiel (Ring n = 4, λ = 10; Brüche) | 7 Belegungen, Z = 241; E[T] = 2761/5 = 552,2 Schritte; in 3 Schritten P(Wechsel) = 5/10 648; vier Ketten bleiben in 100 Schritten mit Wahrscheinlichkeit 0,4981 und in 1000 Schritten mit 0,0007 alle außerhalb des Gebiets B (Zustände mit s = 0 zählen nicht zur A-Seite); erwarteter Anteil des Startgipfels A aus 1000 Schritten 0,6384 (Anteil Gipfel B also 0,3616 statt 0,5); t_rel = 282,6; τ₈ = 7,80943 und ESS 4,098 bei vier Ketten à 8 Schritte | `test_c11`, `tests/test_handbeispiel.py` |
| Orakel für „kein Wechsel“ | in allen 32 Zeilen (13 Zellen, 19 Fensterzeilen) liegt die Zählung innerhalb von vier Standardfehlern der exakten Erwartung 200 · p⁴ | `test_c12` |

Messungen dieser README, die **nicht** im Test stehen: die Rechenzeiten (Abschnitt Methodik).

**Negativbefunde (bleiben stehen):** Die heutige Regel (R̂ < 1,01 und ESS ≥ 400) hilft in der Gaußmischung nicht (191 gegen 191); im Ring schlägt sie teilweise an, wegen der ESS unter 400 und nicht wegen des zweiten Gipfels (7 bis 26 von 200 statt 51 bis 57). Auch die Kontrolle zeigt: R̂ ≥ 1,01 im Ring ist dort kein Gipfelbefund.
Das rangnormalisierte R̂ ändert die Zählung der Täuschung nicht (im dritten Lauf R̂ < 1,01 in 195 statt 194 von 200, täuschend weiterhin 194).

## Befunde und Korrekturen gegenüber der ersten Rechnung

* **Die Zufallszahlen sind andere:** SplitMix64 mit eigenem Strom je Kette gibt andere Läufe als die erste Rechnung (numpy-Strom). Gaußmischung D = 10, σ = 1: täuschend 191 / 186 / 194 (erste Rechnung: 192 / 186 / 189); Ring: 51 bis 57 von 200 (erste Rechnung: 40 bis 54);
  Fenster: Maximum 75 bei N = 1000 (erste Rechnung: 70), bei N = 100 jetzt 2 (erste Rechnung: 7). Das Band der Zufalls-Seeds ist breiter, als drei Läufe vermuten ließen; die Zahlen oben sind die dieser Demo.
* **Eine Wechselschwelle:** x ≥ D/4 statt x > D/4 (erste Rechnung). E[T] ist damit bei D = 8, σ = 1 gleich 5107 statt 5112. Die Wechselrate im Gleichgewicht ist hier mit demselben Gitter und demselben abgeschnittenen Vorschlag gerechnet: 1/Rate = 5090 (die erste Rechnung nannte 5094 für ein Gitter ohne Abschneiden des Vorschlags);
  bei D = 6, σ = 2,4 gleich 4,02·10⁻² statt 3,97·10⁻² (derselbe Unterschied im Kern; die Ursache der 1,3 % Abweichung ist nicht weiter untersucht).
* **Gestreute Starts:** 42 von 400 (10,5 %) statt 48 von 400 (12,0 %) in der ersten Rechnung; erwartet sind 12,5 % ± 1,7 Prozentpunkte (Standardfehler). Die Messung liegt rund 1,2 Standardfehler darunter. „Selbe Seite“ zählt auch Ketten, die noch in der Senke hängen; die Formel 2^(1−M) ist eine Erwartung für den Start, nicht für das Ende.
* **Ketten-Zahl M = 8 ist jetzt gemessen** (9 von 1000, Formel 7,8), ebenso M = 2 (493 von 1000, Formel 500).
* **Heute-Regel im Ring** 26 / 21 und 7 / 8 von 200 (erste Rechnung: 22 / 17 und 2 / 3).
* **Der Gipfelanteil heißt überall „Anteil Gipfel B“** (der Gipfel, in dem keine Kette gestartet ist); die erste Rechnung nannte ihn für die Gaußmischung „Anteil rechts“, für den Ring „Anteil A“.
* **Der Ring n = 16 ist nicht live** (2207 Belegungen: dichte Matrix und Eigenzerlegung brauchen deutlich über zwei Sekunden), sondern in den vorgerechneten Zellen.

## Ehrliche Grenzen

* **Ausgedachte Modelle, eine Dimension.** Beide Modelle sind ausgedacht; höhere Dimensionen und die stetige Kette sind hier nicht gerechnet. Die Gitterkette ist eine Näherung.
* **Die Seite kennt das exakte Ziel.** In der Praxis ist der Gipfelanteil unbekannt; was dann als Hilfsdiagnose wirkt (Wechselzähler, Start in mehreren Gipfeln auf Verdacht), ist nicht gemessen. Gemessen ist nur: Starts in beiden Gipfeln lösen den Alarm aus (200 von 200 in den vier Zellen).
* **Vier Ketten, ohne Einbrennen, geteiltes R̂.** Mehr als vier Ketten sind nur für gestreute Starts (M = 2, 4, 8) und nur als Selbe-Seite-Zählung gemessen; Tail-ESS, geschachteltes R̂, eine strengere R̂-Schwelle allein, Ausdünnen, ungleiche Gipfelgewichte und -breiten und schmale Verbindungsgrate sind nicht gemessen.
* **„R̂ ≥ 1,01 in 200 von 200“ heißt 200 von 200 Wiederholungen in diesen Zellen**, nicht „immer“: bei sehr kurzen und bei sehr langen Ketten ist die Warnung nicht in jeder Wiederholung da (Fenster: 198 von 200 bei N = 100, 198 von 200 bei N = 100 000).
* **Ursachen sind Deutungen, wo sie nicht gemessen sind:** „die ESS bleibt im Ring unter 400“ ist gemessen (ESS 176 bis 227 gegen 4,0 bis 4,2 exakt); dass die Defektbewegung innerhalb des Gipfels der Grund ist, ist eine Deutung.
* **Zufallszählungen sind Bänder.** Die Zählungen gelten für die angegebenen Zufalls-Seeds; die Seite nennt, wo möglich, mehrere Zufalls-Seeds und die exakte Erwartung für „kein Wechsel“.

## Verifikation

* **Kern:** Ketten gegen eine rein sequentielle Fassung bitgleich (Gitter und Ring, drei Startarten); Zielverteilungen (Symmetrie, Stationarität, Detailed Balance); Handbeispiel in Brüchen (`tests/test_handbeispiel.py`).
* **Orakel** (`tests/test_oracle_diagnose_taeuscht.py`, `tests/test_exact.py`, `tests/test_diagnostics.py`): Überlebenswahrscheinlichkeit gegen Matrixpotenz und gegen den Anteil der Läufe im Binomialband; E[T] gegen lineares System; exakte ESS gegen die Summe der Autokorrelationen aus Matrixpotenzen und gegen die Varianz des Gesamtmittels stationärer Ketten;
  GTH gegen lineares System, Wechselrate gegen eine zweite Aufstellung der erweiterten Kette und gegen die gezählten Wechsel; R̂ gegen Handwerte, ESS gegen die exakte AR(1)-Formel; Gipfelanteil gegen die exakte Erwartung.
* **Eigenschaften** (`tests/test_eigenschaften_diagnose_taeuscht.py`): gleiche Zufalls-Seeds gleiche Ausgabe; unabhängig von der Reihenfolge der Läufe; Überlebenswahrscheinlichkeit fällt in N; Start im anderen Gipfel hebt R̂ in 40 von 40 Läufen; R̂ unabhängig von Umnummerierung, Verschiebung, positiver Skalierung.
* **Randfälle** (`tests/test_edge_cases.py`): konstante Ketten (R̂ nicht definiert, wird gemeldet), weniger als vier Schritte, zwei Ketten, D = 0, kleinste und größte Regler.
* **Eingefrorene Läufe** (`tests/test_tables.py`): Aufbau, Zählungs-Invarianten, jede Zählung „kein Wechsel“ gegen die exakte Erwartung, exakte Zahlen gegen die Live-Rechnung, einzelne Zellen neu gerechnet.
* **Verdikte** (`tests/test_verdicts.py`), **Presets** und **Oberfläche** (AppTest: jeder Button, jeder Regler an den Rändern, jede Startart in beiden Modellen), **Layout**, **Footer**, **Laufzeit-Importe**, **Neuladen der Hilfsmodule** (`tests/test_modul_frisch.py`), **Texte** (`tests/test_text_regeln.py`) und **Claims** (`tests/test_claims.py`).
* Die Testzahl: 210 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (vier Experimente, Einordnung, Grenzen, Formelblock) |
| `dgn_chain.py` | Gitterkette der Gaußmischung, Hard-Core-Ring, Zufallsstrom je Kette, Läufe |
| `dgn_diagnostics.py` | Geteiltes R̂, rangnormalisiertes R̂, ESS_bulk |
| `dgn_exact.py` | Exakte Größen: Überlebenswahrscheinlichkeit, E[T], ESS einer stationären Kette, GTH, Wechselrate |
| `dgn_scenario.py` | Zellen der eingefrorenen Läufe, Zählung je Zelle, Startexperiment |
| `dgn_reference.py` | Lädt `dgn_tables.json` und `dgn_exact.json` |
| `dgn_evaluation.py` | Auswertung eines Laufs und alle Verdikte als reine Funktionen, Zeilen der Einordnung |
| `dgn_constants.py` | Regler-Grenzen, Schwellen, Farben, Presets |
| `dgn_presets.py` | Permalink, Schnellstart in der Seitenleiste (aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `dgn_layout.py` | Fester Kopf (Stylesheet) |
| `dgn_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dgn_texts.py` | Texte, die App und README gleich führen (Einleitung, Grenzen, Formelblock) |
| `dgn_format.py` | Deutsche Zahlenformate |
| `dgn_rng.py` | SplitMix64 (geprüfte Vorlage, unverändert kopiert) |
| `modul_frisch.py` | Lädt geänderte Hilfsmodule nach einem Update neu (Streamlit Cloud behält sie sonst bis zum Neustart; kopiert aus einer Nachbar-Demo, mit `tests/test_modul_frisch.py`) |
| `dgn_tables.json`, `dgn_exact.json` | Eingefrorene Läufe und exakte Gegenzahlen |
| `tools/dgn_make_tables.py` | Erzeugt die beiden JSON-Dateien; `--check` rechnet eine Auswahl neu |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Orakel, Eigenschaften, Randfälle, Handbeispiel, eingefrorene Läufe, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Texte, Claims |

## Bewusst nicht umgesetzt

* **Höhere Dimensionen und die stetige Kette** (nur Verweis auf `posterior-sampling-demo`); **Tempering und andere Abhilfen** ([`gipfel-tempering-demo`](https://github.com/sebastian-hanisch/gipfel-tempering-demo)); **Tail-ESS, geschachteltes R̂, Ausdünnen**, ungleiche Gipfelgewichte und -breiten (nicht gemessen).
* **Herleitung von R̂ und ESS:** stehen in `posterior-sampling-demo`; hier nur die Täuschung.
* **PDF-Export:** die Seite zeigt Zählungen aus vorgerechneten Läufen; ein Export ergäbe keine neue Aussage.
* **Eine Hilfsdiagnose in der Praxis:** nicht gemessen, siehe Ehrliche Grenzen.

## Quellen

Nur so zitiert, wie sie gelesen sind (Stand der Quellenarbeit zu dieser Demo; Wortlaut gelesen, wenn nicht anders vermerkt); keine wörtlichen Absätze, nur Gleichungen und Befunde in eigenen Worten:

* Vehtari, Gelman, Simpson, Carpenter, Bürkner, „Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC“, Bayesian Analysis 16(2) (2021), arXiv 1903.08008 v5: Volltext, gezielt gelesen (Abschnitte 1, 4, 5, Anhang C und D). Daraus: Split-R̂, ESS, Schwelle R̂ < 1,01, mindestens vier Ketten, ESS_bulk ≥ 400; bei getrennten Gipfeln hängt die ESS an der Zahl der gefundenen Gipfel.
* Vats, Knudson, „Revisiting the Gelman–Rubin diagnostic“, Statistical Science (2021), arXiv 1812.09384 v3: Einleitung und Abschnitt 5 gelesen; die Schwelle 1,1 entspricht etwa 5 effektiven Stichproben je Kette.
* Gelman u. a., Bayesian Data Analysis, 3. Aufl., S. 285 bis 288: gelesen (R̂, Warnung, dass Startverteilung und Gipfel zusammenhängen, sinngemäß).
* Gelman, Rubin, „Inference from iterative simulation using multiple sequences“, Statistical Science 7(4) (1992): **nur die Seiten 457 und 461 gelesen** (Empfehlung überstreuter Starts, die Gleichung des Schätzers der Varianz); mehr wird hier nicht zitiert.
* Jacob, O'Leary, Atchadé, „Unbiased Markov chain Monte Carlo with couplings“, J. R. Stat. Soc. B 82(3) (2020), arXiv 1708.03625 v5: Abschnitt 5.1 gelesen (im bimodalen Beispiel hängt die Aussage an der Startverteilung).
* Nicht gelesen und nicht zitiert: Brooks, Gelman (1998).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas, SciPy und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).
