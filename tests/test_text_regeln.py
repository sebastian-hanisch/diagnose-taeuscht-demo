"""Textregeln der Seite: keine Wörter der Planung in README, App, Docstrings und Tests, echte Umlaute, „Zufalls-Seed“, keine verbotenen Wendungen.

Die Wörter stehen hier in Teilen, damit diese Datei sich nicht selbst trifft; die Zeile „Mehr zu den Demos“ des Portfolio-Footers ist ausgenommen."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", ROOT / "app.py", *sorted(ROOT.glob("dgn_*.py")), *sorted((ROOT / "tools").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]
SELF = Path(__file__).resolve()

PLANNING = [r"\bSt" + "ück\b", r"\bLin" + "ie\b", r"\bVor" + "messung", r"\bPl" + "an\b", r"\bV[123]\b", "mc" + "_v", r"Phase " + "0", "Kanten" + "abgleich"]
OTHER = {
    "Wärmebad": "Heat-bath heißt es auf dieser Seite",
    "Saatwert": "Zufalls-Seed",
    r"\btr(ä|ae)gt\b": "reicht/gilt/zeigt statt trägt",
    r"\btragen\b": "reicht/gilt/zeigt statt tragen",
    "läuft lokal": "keine Umgebungsaussage",
    r"\bSchwanz\b": "Tail",
    "geplant": "keine „geplant“-Reste",
}


def _lines(path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "Mehr zu den Demos" in line or "Mehr zur Reihe" in line:
            continue
        yield n, line


@pytest.mark.parametrize("pattern", PLANNING)
def test_no_planning_words(pattern):
    hits = [f"{p.name}:{n}: {line.strip()[:90]}" for p in FILES if p != SELF for n, line in _lines(p) if re.search(pattern, line)]
    assert not hits, hits[:5]


@pytest.mark.parametrize("pattern,hint", list(OTHER.items()))
def test_no_forbidden_phrases(pattern, hint):
    hits = [f"{p.name}:{n}: {line.strip()[:90]}" for p in FILES if p != SELF for n, line in _lines(p) if re.search(pattern, line)]
    assert not hits, (hint, hits[:5])


def test_umlauts_are_real_and_seed_is_called_zufalls_seed():
    ascii_forms = r"(waehrend|fuer|Aenderung|Loesung|Gaussmischung|Gauss-Mischung|Unabhaengig\w*|Groesse|Schaetz\w*)"
    for p in (ROOT / "README.md", ROOT / "app.py", ROOT / "dgn_texts.py", ROOT / "dgn_evaluation.py", ROOT / "dgn_constants.py"):
        assert not re.search(ascii_forms, p.read_text(encoding="utf-8")), p.name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Zufalls-Seed" in readme
    assert not re.search(r"Seed", re.sub(r"Zufalls-Seed|Basis-Seed", "", readme))


def test_readme_names_no_open_website_integration():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"(Einbindung|Website).{0,40}\boffen\b", text)
