"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben; die Regler des ersten Experiments stehen im Hauptbereich.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler
des Experiments dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dgn_constants as C
import dgn_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
# Regler der eingefrorenen Experimente: Auswahl aus einem Raster, ändern das Preset nicht
RASTER_REGLER = {"exp2_vehicle", "exp3_vehicle", "exp4_vehicle", "exp4_M", "exp2_cell_Gaußmischung", "exp2_cell_Hard-Core-Ring"}


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.main, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht im Hauptbereich")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [C.DEFAULT_PRESET]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_des_experiments_steht_in_setting_specs_und_kein_regler_in_der_seitenleiste():
    at = _app()
    assert not [w for art in REGLER_ARTEN for w in getattr(at.sidebar, art)]
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS) - RASTER_REGLER
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel


# Wert je Regler, den kein Preset benutzt; die Ring-Regler sind nur im Modell „Ring“ frei (sonst gesperrt)
UNUSED = {"D": 5, "sigma": 2.0, "ring_n": 8, "lam": 20, "N": 5000, "start": "zwei_zwei", "seed": 42, "vehicle": "Hard-Core-Ring"}
NEEDS_RING = {"ring_n", "lam", "vehicle"}


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    if state_key in NEEDS_RING:
        _regler(at, "vehicle").set_value("Hard-Core-Ring").run()
    if state_key == "vehicle":
        assert _typen(at)["Ring: vier Ketten im Gipfel A (n = 12, λ = 50)"] == "primary"     # Ring mit allen übrigen Vorgaben ist das Ring-Preset
        _regler(at, "ring_n").set_value(UNUSED["ring_n"]).run()
    else:
        _regler(at, state_key).set_value(UNUSED[state_key]).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    name = C.DEFAULT_PRESET
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_das_ring_preset_setzt_die_ring_regler_und_laesst_die_gauss_regler_stehen():
    at = _app()
    name = "Ring: vier Ketten im Gipfel A (n = 12, λ = 50)"
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert at.main.radio(key="vehicle").value == "Hard-Core-Ring"
    assert at.main.select_slider(key="ring_n").value == 12 and at.main.select_slider(key="lam").value == 50
    assert at.main.select_slider(key="D").value == C.DEFAULT_D        # der nicht genutzte Regler behält seinen Wert (kein verstecktes Zurücksetzen)
    assert at.main.select_slider(key="D").disabled and not at.main.select_slider(key="ring_n").disabled


def test_nicht_genutzte_regler_sind_gesperrt_nicht_versteckt():
    at = _app()
    assert not at.main.select_slider(key="D").disabled and at.main.select_slider(key="ring_n").disabled
    assert at.main.slider(key="sigma").disabled is False and at.main.select_slider(key="lam").disabled
