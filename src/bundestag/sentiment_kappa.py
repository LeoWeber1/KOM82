"""Übereinstimmung BERT ↔ Annotatoren auf der Gold-Stichprobe. Liegt κ(BERT–Annotator-Mehrheit) unter der vorab
festgelegten Schwelle 0,3, entfällt H3 und Sentiment bleibt deskriptiv (stats_sentiment.py).

Ausgaben in data/processed für die gewählte Fassung (Standard v2): sentiment_gold.csv (ohne Text), kappa.csv (Cohen-κ und
Rohübereinstimmung je Paar mit Bootstrap-KI, linear gewichtetes κ für BERT–Mehrheit), konfusion_bert_mehrheit.csv
(Zeilen Mehrheit, Spalten BERT); die andere Fassung als Vergleich mit Dateisuffix _v1/_v2 und Schlüsselsuffix
Zweihundert/Hundertfuenfzig. Zeilen mit Quelle sentiment_kappa.py in kennzahlen.csv.
Aufruf: python -m bundestag.sentiment_kappa [--gold v1|v2]
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import PROCESSED, REFERENCE
from .metrics import kennzahl, laufzeit, schreibe_kennzahlen
from .stats import N_BOOT, SEED, ki

KATEGORIEN = ["negativ", "neutral", "positiv"]
UNENTSCHIEDEN = "unentschieden"
ANNOTATOREN = {"llm1": "sentiment_llm_1", "llm2": "sentiment_llm_2", "qwen": "sentiment_qwen"}
RATER = ["bert", *ANNOTATOREN]
PAARE = [("bert", "llm1"), ("bert", "llm2"), ("bert", "qwen"), ("bert", "mehrheit"),
         ("llm1", "llm2"), ("llm1", "qwen"), ("llm2", "qwen")]
KURZ = {"bert": "Bert", "llm1": "LlmEins", "llm2": "LlmZwei", "qwen": "Qwen", "mehrheit": "Mehrheit"}
Q = "sentiment_kappa.py"
# v2 = Produktionseinheit 150 Wörter seit der Umstellung nach dem Fenstercheck (sentiment_fenstercheck.py),
# v1 = 200 Wörter der Erstfassung.
VERSIONEN = {"v2": "_v2", "v1": ""}
SCHLUESSEL_SUFFIX = {"v1": "Zweihundert", "v2": "Hundertfuenfzig"}


def codes(labels: Sequence[str], kategorien: Sequence[str] = KATEGORIEN) -> np.ndarray:
    """Kategorie-Index je Label; Labels außerhalb der Kategorien (z. B. „unentschieden“) → −1."""
    pos = {c: i for i, c in enumerate(kategorien)}
    return np.array([pos.get(x, -1) for x in labels], dtype=int)


def tafel(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """k×k-Kontingenztafel (Zeilen a, Spalten b) aus Codes; Paare mit einem Code −1 fallen heraus."""
    ok = (a >= 0) & (b >= 0)
    return np.bincount(a[ok] * k + b[ok], minlength=k * k).reshape(k, k)


def kappa_tafel(t: np.ndarray, gewichtet: bool = False) -> float:
    """Cohen-κ = 1 − Σ w·beobachtet / Σ w·erwartet; ungewichtet w = 1 − δ, linear w = |i − j| / (k − 1)."""
    n, k = t.sum(), len(t)
    if n == 0:
        return np.nan
    i, j = np.indices((k, k))
    w = np.abs(i - j) / (k - 1) if gewichtet else (i != j).astype(float)
    nenner = (w * np.outer(t.sum(1), t.sum(0)) / n).sum()
    return float(1 - (w * t).sum() / nenner) if nenner > 0 else np.nan


def cohen_kappa(a: Sequence[str], b: Sequence[str], gewichtet: bool = False,
                kategorien: Sequence[str] = KATEGORIEN) -> float:
    """Cohen-κ zweier Label-Folgen; Reihenfolge der Kategorien = Ordnung für das linear gewichtete κ."""
    return kappa_tafel(tafel(codes(a, kategorien), codes(b, kategorien), len(kategorien)), gewichtet)


def mehrheit(labels: pd.DataFrame) -> pd.Series:
    """Label mit mindestens 2 von 3 Stimmen, bei drei verschiedenen Labels „unentschieden“."""
    def eine(z: list[str]) -> str:
        top = max(sorted(set(z)), key=z.count)
        return top if z.count(top) >= 2 else UNENTSCHIEDEN
    return labels.apply(lambda r: eine(list(r)), axis=1)


def eingaben(version: str = "v2") -> list:
    s = VERSIONEN[version]
    return [REFERENCE / f"sentiment_gold_schluessel{s}.csv"] + [REFERENCE / f"{d}{s}.csv" for d in ANNOTATOREN.values()]


def lade(version: str = "v2") -> pd.DataFrame:
    schluessel, *annotatoren = eingaben(version)
    g = pd.read_csv(schluessel)[["id", "bert_label"]].rename(columns={"bert_label": "bert"})
    for name, datei in zip(ANNOTATOREN, annotatoren):
        r = pd.read_csv(datei)[["id", "label"]].rename(columns={"label": name})
        g = g.merge(r, on="id", how="left", validate="one_to_one")
    g[RATER] = g[RATER].apply(lambda s: s.str.strip().str.lower())
    falsch = ~g[RATER].isin(KATEGORIEN).all(axis=1)
    if falsch.any():
        raise ValueError(f"Label fehlt oder unbekannt in {list(g.id[falsch])[:5]}")
    g["mehrheit"] = mehrheit(g[list(ANNOTATOREN)])
    return g


def bootstrap_tafeln(a: np.ndarray, b: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Eine k×k-Tafel je Resample (Zeilen von idx); Fenster mit Code −1 zählen in eine verworfene Überlaufzelle."""
    za, zb = a[idx], b[idx]
    zelle = np.where((za >= 0) & (zb >= 0), za * k + zb, k * k)
    versatz = np.arange(len(idx))[:, None] * (k * k + 1)
    t = np.bincount((zelle + versatz).ravel(), minlength=len(idx) * (k * k + 1)).reshape(len(idx), k * k + 1)
    return t[:, :k * k].reshape(-1, k, k)


def auswerten(g: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """kappa-Tabelle je Paar und Precision je BERT-Klasse gegen die Mehrheit, jeweils mit 95-%-Bootstrap-KI."""
    k = len(KATEGORIEN)
    # Eine Indexmatrix für alle 7 Paare (Seed 82): gepaarte Resamples, KI zwischen Paaren vergleichbar
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    zeilen, boot_bm = [], None
    for a, b in PAARE:
        ca, cb = codes(g[a]), codes(g[b])
        t, tb = tafel(ca, cb, k), bootstrap_tafeln(ca, cb, idx, k)
        roh = np.trace(tb, axis1=1, axis2=2) / tb.sum(axis=(1, 2))
        z = {"a": a, "b": b, "n": int(t.sum()),
             "kappa": kappa_tafel(t), **dict(zip(["kappa_ki_unten", "kappa_ki_oben"],
                                                 ki(np.array([kappa_tafel(x) for x in tb])))),
             "roh": np.trace(t) / t.sum(), **dict(zip(["roh_ki_unten", "roh_ki_oben"], ki(roh))),
             "kappa_linear": np.nan, "kappa_linear_ki_unten": np.nan, "kappa_linear_ki_oben": np.nan}
        if b == "mehrheit":
            boot_bm = tb
            z["kappa_linear"] = kappa_tafel(t, gewichtet=True)
            z["kappa_linear_ki_unten"], z["kappa_linear_ki_oben"] = ki(
                np.array([kappa_tafel(x, gewichtet=True) for x in tb]))
        zeilen.append(z)
    t = tafel(codes(g.bert), codes(g.mehrheit), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec_boot = np.diagonal(boot_bm, axis1=1, axis2=2) / boot_bm.sum(axis=2)
    # Nenner = Fenster mit BERT-Label c und Mehrheit (Stand 15.10.: neutral 19 statt 20, 1 Fenster „unentschieden“)
    prec = pd.DataFrame([{"bert": c, "richtig": int(t[i, i]), "n": int(t[i].sum()), "precision": t[i, i] / t[i].sum(),
                          **dict(zip(["ki_unten", "ki_oben"], ki(prec_boot[:, i])))}
                         for i, c in enumerate(KATEGORIEN)])
    return pd.DataFrame(zeilen), prec


def konfusion(g: pd.DataFrame) -> pd.DataFrame:
    """3×3, Zeilen Mehrheit, Spalten BERT; Fenster „unentschieden“ haben keine Zeile."""
    k = len(KATEGORIEN)
    t = tafel(codes(g.mehrheit), codes(g.bert), k)
    return pd.DataFrame(t, index=pd.Index(KATEGORIEN, name="mehrheit"), columns=[f"bert_{c}" for c in KATEGORIEN])


def fehlertyp(g: pd.DataFrame, konf: pd.DataFrame) -> dict:
    """Häufigste Zelle außerhalb der Diagonale mit 2 Beispiel-IDs, einstimmige Mehrheit (3 von 3) zuerst."""
    t = konf.to_numpy().copy()
    np.fill_diagonal(t, -1)
    i, j = np.unravel_index(t.argmax(), t.shape)
    m, b = KATEGORIEN[i], KATEGORIEN[j]
    f = g[(g.mehrheit == m) & (g.bert == b)].copy()
    f["einstimmig"] = (f[list(ANNOTATOREN)] == m).all(axis=1)
    beispiele = f.sort_values(["einstimmig", "id"], ascending=[False, True]).id.head(2).tolist()
    return {"mehrheit": m, "bert": b, "anzahl": int(t[i, j]), "einstimmig": int(f.einstimmig.sum()),
            "beispiele": beispiele}


def kennzahlen_kappa(kap: pd.DataFrame, prec: pd.DataFrame, g: pd.DataFrame, fehler: dict,
                     suffix: str = "") -> list[dict]:
    z = []
    for x in kap.itertuples():
        paar = "LlmLlm" if (x.a, x.b) == ("llm1", "llm2") else KURZ[x.a] + KURZ[x.b]
        z += [kennzahl(f"kappa{paar}", x.kappa, "kappa", f"{Q}:kappa", (x.kappa_ki_unten, x.kappa_ki_oben)),
              kennzahl(f"roh{paar}", x.roh, "Anteil", f"{Q}:kappa", (x.roh_ki_unten, x.roh_ki_oben))]
        if x.b == "mehrheit":
            z.append(kennzahl(f"kappa{paar}Linear", x.kappa_linear, "kappa", f"{Q}:kappa",
                              (x.kappa_linear_ki_unten, x.kappa_linear_ki_oben)))
    z += [kennzahl(f"precisionBert{x.bert.capitalize()}", x.precision, "Anteil", f"{Q}:konfusion_bert_mehrheit",
                   (x.ki_unten, x.ki_oben)) for x in prec.itertuples()]
    z += [kennzahl("fehlertypAnzahl", fehler["anzahl"], "Fenster", f"{Q}:konfusion_bert_mehrheit"),
          kennzahl("mehrheitUnentschieden", int((g.mehrheit == UNENTSCHIEDEN).sum()), "Fenster", f"{Q}:sentiment_gold"),
          kennzahl("goldFenster", len(g), "Fenster", f"{Q}:sentiment_gold")]
    return [{**zeile, "schluessel": zeile["schluessel"] + suffix} for zeile in z]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", choices=list(VERSIONEN), default="v2", help="Fassung mit den Standard-Schlüsseln")
    gewaehlt = ap.parse_args().gold
    zeilen = []
    with laufzeit("sentiment_kappa"):
        for version in [gewaehlt] + [v for v in VERSIONEN if v != gewaehlt]:
            standard = version == gewaehlt
            if not standard and not all(p.exists() for p in eingaben(version)):
                print(f"{version}: Eingaben unvollständig, kein Vergleich")
                continue
            datei = "" if standard else f"_{version}"
            g = lade(version)
            g[["id", *RATER, "mehrheit"]].to_csv(PROCESSED / f"sentiment_gold{datei}.csv", index=False)
            # Stichprobe nach BERT-Label geschichtet (je 20): BERT-Ränder sind vorgegeben,
            # κ daher nicht als Populationswert lesbar, nur als Übereinstimmung auf diesen 60 Fenstern
            kap, prec = auswerten(g)
            kap.to_csv(PROCESSED / f"kappa{datei}.csv", index=False)
            konf = konfusion(g)
            konf.to_csv(PROCESSED / f"konfusion_bert_mehrheit{datei}.csv")
            fehler = fehlertyp(g, konf)
            zeilen += kennzahlen_kappa(kap, prec, g, fehler, "" if standard else SCHLUESSEL_SUFFIX[version])
            print(f"Fassung {version}{' (Standard)' if standard else ''}:")
            print(kap.round(3).to_string(index=False))
            print(konf.to_string())
            print(prec.round(3).to_string(index=False))
            print(f"Häufigster Fehlertyp: Mehrheit {fehler['mehrheit']}, BERT {fehler['bert']}: {fehler['anzahl']} "
                  f"Fenster ({fehler['einstimmig']} einstimmig), Beispiele {', '.join(fehler['beispiele'])}")
    schreibe_kennzahlen(zeilen, Q)


if __name__ == "__main__":
    main()
