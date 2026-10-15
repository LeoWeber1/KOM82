import numpy as np
import pandas as pd
import pytest

from bundestag.sentiment_kappa import UNENTSCHIEDEN, auswerten, cohen_kappa, mehrheit


def test_kappa_identisch_eins():
    x = ["negativ", "neutral", "positiv", "negativ", "positiv", "neutral"]
    assert cohen_kappa(x, x) == pytest.approx(1.0)
    assert cohen_kappa(x, x, gewichtet=True) == pytest.approx(1.0)


def test_kappa_vierfeldertafel():
    # Tafel a=20, b=5, c=10, d=15: p_o 0,7, p_e 0,5 → κ 0,4
    a = ["ja"] * 25 + ["nein"] * 25
    b = ["ja"] * 20 + ["nein"] * 5 + ["ja"] * 10 + ["nein"] * 15
    assert cohen_kappa(a, b, kategorien=["ja", "nein"]) == pytest.approx(0.4)


def test_kappa_drei_klassen_von_hand():
    # p_o 1/2, p_e 5/16 → κ 3/11; mittlere Abstände beobachtet 3/4, erwartet 7/8 → linear 1/7
    a = ["negativ", "neutral", "positiv", "positiv"]
    b = ["neutral", "neutral", "positiv", "negativ"]
    assert cohen_kappa(a, b) == pytest.approx(3 / 11)
    assert cohen_kappa(a, b, gewichtet=True) == pytest.approx(1 / 7)


def test_linear_gleich_ungewichtet_bei_nachbarklassen():
    a = ["negativ", "neutral", "neutral", "negativ", "neutral"]
    b = ["negativ", "negativ", "neutral", "negativ", "neutral"]
    assert cohen_kappa(a, b, gewichtet=True) == pytest.approx(cohen_kappa(a, b))


def test_unentschieden_faellt_heraus():
    assert cohen_kappa(["negativ", "positiv", UNENTSCHIEDEN], ["negativ", "positiv", "neutral"]) == pytest.approx(1.0)


def test_mehrheit():
    df = pd.DataFrame({"llm1": ["negativ", "negativ", "neutral"], "llm2": ["negativ", "neutral", "neutral"],
                       "qwen": ["positiv", "positiv", "neutral"]})
    assert list(mehrheit(df)) == ["negativ", UNENTSCHIEDEN, "neutral"]


def test_auswerten_ki_umschliesst_punktwert():
    rng = np.random.default_rng(1)
    bert = np.repeat(["negativ", "neutral", "positiv"], 20)
    g = pd.DataFrame({"id": range(60), "bert": bert})
    for r in ["llm1", "llm2", "qwen"]:
        g[r] = np.where(rng.random(60) < 0.7, bert, rng.choice(["negativ", "neutral", "positiv"], 60))
    g["mehrheit"] = mehrheit(g[["llm1", "llm2", "qwen"]])
    kap, prec = auswerten(g, n_boot=200)
    assert len(kap) == 7 and kap.kappa_linear.notna().sum() == 1
    assert ((kap.kappa_ki_unten <= kap.kappa) & (kap.kappa <= kap.kappa_ki_oben)).all()
    assert prec.n.sum() == (g.mehrheit != UNENTSCHIEDEN).sum()
