"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Modelle (ausgedacht, keine reale Datenquelle) -----------------------------------------------------------------------------------------
GRID_STEP = 0.1            # Gitterabstand der Gaußmischung
GRID_PAD = 7.0             # Gitterbereich: +-(D/2 + GRID_PAD)
PROPOSAL_CUT = 5.0         # der Normalvorschlag wird bei 5 sigma abgeschnitten
CHAINS = 4                 # vier Ketten, wie in der heutigen Empfehlung (mindestens vier)
RHAT_LIMIT = 1.01          # Split-R-Dach < 1,01 (Vehtari u. a. 2021)
ESS_LIMIT = 400            # ESS_bulk >= 400 bei vier Ketten
RHAT_OLD_LIMIT = 1.1       # veraltete Schwelle (nur als Vergleich genannt)

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------
VEHICLES = ("Gaußmischung", "Hard-Core-Ring")
D_CHOICES = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12)     # Abstand der beiden Gipfel in Standardabweichungen (0: nur ein Gipfel, Kontrolle)
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP = 0.5, 3.0, 0.1   # Schrittweite des Vorschlags
RING_N_CHOICES = (6, 8, 10, 12)                    # live exakt rechenbar bis n = 12 (322 Belegungen); n = 16 nur eingefroren
LAM_CHOICES = (1, 2, 5, 10, 20, 50, 100)           # Gewicht je belegtem Platz
N_CHOICES = (200, 500, 1000, 2000, 5000, 10000, 20000)   # Kettenlänge (Schritte je Kette)
START_CHOICES = {
    "alle_A": "alle vier im Gipfel A",
    "eine_B": "drei in A, eine in B",
    "zwei_zwei": "zwei in A, zwei in B",
    "gestreut": "gestreut",
}
SEED_MAX, DEFAULT_SEED = 999999, 8101
DEFAULT_VEHICLE, DEFAULT_D, DEFAULT_SIGMA = "Gaußmischung", 10, 1.0
DEFAULT_RING_N, DEFAULT_LAM, DEFAULT_N, DEFAULT_START = 12, 50, 2000, "alle_A"
PLOT_POINTS = 2000          # höchstens so viele Punkte je Spur im Verlauf (Schrittweite wächst mit N)
LIVE_MAX_STATES = 400       # größte Zustandszahl, die live exakt gerechnet wird (Gitter D = 12: 261, Ring n = 12: 322)

# --- Eingefrorene Läufe ----------------------------------------------------------------------------------------------------------------------
TABLES_FILE = "dgn_tables.json"
EXACT_FILE = "dgn_exact.json"
REPS = 200                  # Wiederholungen (je vier Ketten) je Zelle
M_CHAINS = (2, 4, 8)        # Kettenzahlen der gestreuten Starts
M_REPS = 1000               # Wiederholungen je Kettenzahl der gestreuten Starts
BURN = 500                  # Verwerfen der ersten Schritte im Startexperiment

# --- Farben ----------------------------------------------------------------------------------------------------------------------------------
CHAIN_COLORS = ("#2F6B65", "#c8553d", "#4c78a8", "#b08d2c")
COLOR_EXACT = "#888888"
COLOR_DECEIVE = "#c8553d"
COLOR_OK = "#2F6B65"
COLOR_BLUE = "#4c78a8"

PRESETS = {
    "Gaußmischung: vier Ketten links (D = 10, σ = 1)": {
        "vehicle": "Gaußmischung", "D": 10, "sigma": 1.0, "N": 2000, "start": "alle_A", "seed": DEFAULT_SEED},
    "Gaußmischung: Ketten wechseln (σ = 2,4)": {
        "vehicle": "Gaußmischung", "D": 10, "sigma": 2.4, "N": 2000, "start": "alle_A", "seed": DEFAULT_SEED},
    "Gaußmischung: eine Kette rechts": {
        "vehicle": "Gaußmischung", "D": 10, "sigma": 1.0, "N": 2000, "start": "eine_B", "seed": DEFAULT_SEED},
    "Ring: vier Ketten im Gipfel A (n = 12, λ = 50)": {
        "vehicle": "Hard-Core-Ring", "ring_n": 12, "lam": 50, "N": 2000, "start": "alle_A", "seed": DEFAULT_SEED},
    "Kontrolle: ohne zweiten Gipfel (D = 0)": {
        "vehicle": "Gaußmischung", "D": 0, "sigma": 1.0, "N": 2000, "start": "alle_A", "seed": DEFAULT_SEED},
    "Fenster: N = 1 000 (D = 8)": {
        "vehicle": "Gaußmischung", "D": 8, "sigma": 1.0, "N": 1000, "start": "alle_A", "seed": DEFAULT_SEED},
    "Gestreute Starts": {
        "vehicle": "Gaußmischung", "D": 10, "sigma": 1.0, "N": 2000, "start": "gestreut", "seed": DEFAULT_SEED},
}
DEFAULT_PRESET = next(iter(PRESETS))
PRESET_HELP = {
    "Gaußmischung: vier Ketten links (D = 10, σ = 1)": "Zwei Gipfel im Abstand von 10 Standardabweichungen, Schrittweite 1, vier Ketten im linken Gipfel, 2 000 Schritte je Kette. "
                                                        "Die Diagnostik meldet Entwarnung, obwohl nur ein Gipfel besucht wird.",
    "Gaußmischung: Ketten wechseln (σ = 2,4)": "Gleiche Gipfel, aber Schrittweite 2,4: die Ketten springen zwischen den Gipfeln, hier gibt es keine Täuschung.",
    "Gaußmischung: eine Kette rechts": "Eine der vier Ketten startet im anderen Gipfel: R-Dach schlägt an.",
    "Ring: vier Ketten im Gipfel A (n = 12, λ = 50)": "Der ausgedachte Stellplatz-Ring mit 12 Plätzen und Gewicht 50 je belegtem Platz: die Täuschung ist schwächer als bei der Gaußmischung.",
    "Kontrolle: ohne zweiten Gipfel (D = 0)": "Nur ein Gipfel (eine Standardnormale): die Diagnostik darf Entwarnung geben, und es stimmt.",
    "Fenster: N = 1 000 (D = 8)": "Abstand 8, 1 000 Schritte je Kette (in der Größenordnung eines Fünftels der mittleren Zeit bis zum ersten Wechsel): die Länge mit der größten Täuschung.",
    "Gestreute Starts": "Die Startpunkte kommen aus einer breiten Normalverteilung um die Mitte: alle vier landen nur manchmal im selben Gipfel.",
}
