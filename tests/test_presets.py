"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Wirkung jedes Presets (kein Preset ohne Wirkung meldet eine Wirkung)."""

import pytest

import dgn_constants as C
import dgn_evaluation as E
import dgn_presets as P


def _settings(name):
    return E.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_values_of_their_regler():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert _settings(C.DEFAULT_PRESET) == E.Settings()
    assert all(P.SETTING_SPECS[k].default == v for k, v in C.PRESETS[C.DEFAULT_PRESET].items())


def test_setting_specs_cover_every_preset_key():
    assert {k for p in C.PRESETS.values() for k in p} <= set(P.SETTING_SPECS)


def test_permalink_snapping():
    assert P._snap(P.SETTING_SPECS["D"], 9.4) == 9 and P._snap(P.SETTING_SPECS["N"], 1400) == 1000 and P._snap(P.SETTING_SPECS["N"], 1600) == 2000
    assert P._snap(P.SETTING_SPECS["sigma"], 9.0) == C.SIGMA_MAX and P._snap(P.SETTING_SPECS["sigma"], 0.01) == C.SIGMA_MIN
    assert P._snap(P.SETTING_SPECS["vehicle"], "unbekannt") is None and P._snap(P.SETTING_SPECS["start"], "gestreut") == "gestreut"
    assert P._snap(P.SETTING_SPECS["seed"], -5) == 0 and P._snap(P.SETTING_SPECS["seed"], 10 ** 9) == C.SEED_MAX


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_its_verdict_matches_the_expected_line(name):
    a = E.analyse(_settings(name))
    kind, text = E.verdict_run(a.facts)
    f = a.facts
    if name.startswith("Gaußmischung: vier Ketten links"):
        assert kind == "error" and "Täuschung" in text and f.switches == (0, 0, 0, 0) and f.rhat_ok
    elif "σ = 2,4" in name:
        assert kind != "error" and "Täuschung" not in text and sum(f.switches) > 0       # die Ketten wechseln: keine Täuschung
    elif "eine Kette rechts" in name:
        assert not f.rhat_ok and f.rhat >= 1.01 and kind == "info" and "Startpunkte beide Gipfel überdecken" in text
    elif name.startswith("Ring"):
        assert f.vehicle == C.VEHICLES[1] and kind in ("error", "warning") and f.switches == (0, 0, 0, 0)
    elif name.startswith("Kontrolle"):
        assert not f.has_peaks and "Kontrolle" in text and kind in ("success", "info")
    elif name.startswith("Fenster"):
        assert f.N == 1000 and f.has_peaks and kind in ("error", "warning", "success", "info")
    elif name.startswith("Gestreute"):
        assert f.start == "gestreut" and kind in ("error", "warning", "success", "info")
    else:
        pytest.fail(f"Preset {name} ohne erwartete Verdikt-Zeile im Test")


def test_preset_with_a_single_peak_does_not_report_a_second_peak():
    a = E.analyse(_settings("Kontrolle: ohne zweiten Gipfel (D = 0)"))
    assert not a.facts.has_peaks and not a.facts.seen_B
