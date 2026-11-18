"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler des Experiments „Vier Ketten in einer Mischung“ stehen in SETTING_SPECS; ein Preset ist aktiv, wenn jeder dieser Regler dem Preset entspricht.
Die Regler der eingefrorenen Experimente (Auswahl aus einem Raster) gehören nicht dazu: ihr Wechsel ändert das Preset nicht."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import streamlit as st

import dgn_constants as C
from dgn_rng import SplitMix64


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # erlaubte Werte (Auswahlregler); Zahlen rasten auf den nächsten erlaubten Wert ein


SETTING_SPECS = {
    "vehicle": SettingSpec("v", str, C.DEFAULT_VEHICLE, choices=C.VEHICLES),
    "D": SettingSpec("D", int, C.DEFAULT_D, choices=C.D_CHOICES),
    "sigma": SettingSpec("sig", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "ring_n": SettingSpec("rn", int, C.DEFAULT_RING_N, choices=C.RING_N_CHOICES),
    "lam": SettingSpec("lam", int, C.DEFAULT_LAM, choices=C.LAM_CHOICES),
    "N": SettingSpec("N", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "start": SettingSpec("st", str, C.DEFAULT_START, choices=tuple(C.START_CHOICES)),
    "seed": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec: SettingSpec, value):
    """Wert auf die erlaubten Werte einrasten bzw. auf die Grenzen begrenzen; None, wenn nicht verwendbar."""
    if spec.choices is not None:
        if spec.caster is str:
            return value if value in spec.choices else None
        return min(spec.choices, key=lambda c: abs(c - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return round(value, 1) if spec.caster is float else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Preset laden: jeder Regler bekommt den Wert des Presets, ein im Preset nicht genannter Regler seinen Standardwert (so wie `aktives_preset` vergleicht)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), abs_tol=1e-9)
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(_same(st.session_state.get(state_key, spec.default), preset.get(state_key, spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario für das Experiment „Vier Ketten in einer Mischung“ laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    """Neuer Zufalls-Seed aus der Uhrzeit (nur für die Oberfläche; die Läufe selbst laufen über SplitMix64 mit dem gewählten Zufalls-Seed)."""
    st.session_state["seed"] = SplitMix64(time.time_ns()).below(C.SEED_MAX + 1)
