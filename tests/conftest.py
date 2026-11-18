"""Gemeinsame Fixtures: das Handbeispiel (Hard-Core-Ring mit n = 4 Plätzen und lambda = 10), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dgn_chain as ch  # noqa: E402


@pytest.fixture(scope="session")
def ring4():
    """Handbeispiel: 4 Plätze, lambda = 10; 7 Zustände (leer, vier Einzelplätze, die Paare A = {0, 2} und B = {1, 3}), Z = 1 + 4 * 10 + 2 * 100 = 241."""
    return ch.get_ring(4, 10.0)


@pytest.fixture(scope="session")
def grid_d8():
    return ch.get_grid(8.0, 1.0)


@pytest.fixture(scope="session")
def ring12():
    return ch.get_ring(12, 50.0)
