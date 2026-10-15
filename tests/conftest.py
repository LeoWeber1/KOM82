import sys
from pathlib import Path

import pytest

# src/ auf den Pfad: Tests laufen so auch ohne "pip install -e ." (pyproject.toml).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bundestag import config  # noqa: E402

# data/xml, data/txt und data/lines sind gitignoriert (.gitignore Z. 7–9): auf einem frischen Klon scheiterten 94 Tests
# mit KeyError/FileNotFoundError, statt übersprungen zu werden. Geprüft wird jede der 67 Sitzungen,
# weil nach einem Teillauf 146–147 nur 2 vorliegen, die Tests aber u. a. 160, 203 und 211 lesen.
ROHDATEN = [d / f"{config.WAHLPERIODE}{nr:03d}{endung}" for nr in config.SITZUNGEN_VOLL
            for d, endung in ((config.XML_DIR, ".xml"), (config.TXT_DIR, ".txt"), (config.LINES_DIR, ".jsonl"))]


def pytest_collection_modifyitems(config, items):
    if all(p.exists() for p in ROHDATEN):
        return
    skip = pytest.mark.skip(reason="Rohdaten fehlen: python run_all.py lädt und extrahiert sie")
    for item in items:
        if "rohdaten" in item.keywords:
            item.add_marker(skip)
