"""tools/make_zahlen.py: deutsches Zahlenformat, Rundung und Makronamen in zahlen.tex."""
import importlib.util
import math
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _lade(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mz = _lade("make_zahlen")


def test_format_deutsch():
    f = mz.formatiere
    assert f(13662, "Beiträge") == r"13\,662"            # Tausender erst ab 5 Stellen
    assert f(6385, "Reden") == "6385"
    assert f(3706132, "Wörter") == r"3\,706\,132"
    assert f(0.958, "F1") == "0{,}96"
    assert f(0.9992, "Quote") == "0{,}999"               # 3. Stelle nur, weil 2 Stellen 1,00 zeigten
    assert f(1.0, "Quote") == "1{,}00"
    assert f(0.3249999999999999, "kappa") == "0{,}33"    # Gleitkommarest, kaufmännisch gerundet
    assert f(-2.0863, "Prozentpunkte") == r"\ensuremath{-}2{,}1"
    assert f(-0.004, "netto") == "0{,}00"                # kein Minus vor Null
    assert f(138.147, "Wörter/Minute") == "138{,}1"
    assert f(3.7e-26, "p") == r"\ensuremath{<}\,0{,}001"
    assert f(0.03, "p") == r"\ensuremath{=}\,0{,}03"
    assert f(1, "ja=1") == "ja" and f(0, "ja=1") == "nein"
    assert f(4376.97, "je Sitz") == "4377" and f(6.85, "je Sitz") == "6{,}85"
    assert f(math.nan, "Reden") == r"\textbf{[FEHLT]}"


def _kennzahlen():
    return pd.DataFrame([
        {"schluessel": "beitraegeParser", "wert": 13662, "ki_unten": None, "ki_oben": None, "einheit": "Beiträge", "quelle": "t:a"},
        {"schluessel": "redenGesamt", "wert": 6385, "ki_unten": None, "ki_oben": None, "einheit": "Reden", "quelle": "t:b"},
        {"schluessel": "redenFEinsMikro", "wert": 0.958, "ki_unten": None, "ki_oben": None, "einheit": "F1", "quelle": "t:c"},
        {"schluessel": "hEinsSPD", "wert": -2.09, "ki_unten": -4.27, "ki_oben": 0.48, "einheit": "Prozentpunkte", "quelle": "t:d"},
        {"schluessel": "f1_mikro", "wert": 0.9, "ki_unten": None, "ki_oben": None, "einheit": "F1", "quelle": "t:e"},
        {"schluessel": "hZwei2", "wert": 1, "ki_unten": None, "ki_oben": None, "einheit": "", "quelle": "t:f"},
        {"schluessel": "luecke", "wert": None, "ki_unten": None, "ki_oben": None, "einheit": "Reden", "quelle": "t:g"},
        {"schluessel": "redenGesamt", "wert": 1, "ki_unten": None, "ki_oben": None, "einheit": "Reden", "quelle": "t:h"},
    ])


def test_zahlen_tex_meldet_statt_umzubenennen(tmp_path):
    ziel = tmp_path / "zahlen.tex"
    b = mz.schreibe_tex(_kennzahlen(), ziel)
    tex = ziel.read_text(encoding="utf-8")
    assert b["ungueltig"] == ["f1_mikro", "hZwei2"] and b["doppelt"] == ["redenGesamt"]
    assert b["fehlt"] == ["luecke"] and b["ki"] == 1 and b["makros"] == 7
    assert r"\newcommand{\hEinsSPDKIu}{\ensuremath{-}4{,}3}" in tex
    assert r"\newcommand{\redenGesamt}{6385}" in tex and "fone" not in tex.lower()
