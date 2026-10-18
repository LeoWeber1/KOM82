"""Kennzahlen → LaTeX/zahlen.tex: jede Zahl im Bericht als Makro, das die Pipeline erzeugt.

1. Ergänzt data/processed/kennzahlen.csv um Validierungs-, Parser- und Laufzeit-Kennzahlen, die dort noch
   fehlen (Quelle "make_zahlen.py:<Datei>:<Spalte>"). Schlüssel anderer Skripte bleiben unangetastet.
2. Schreibt LaTeX/zahlen.tex: \\newcommand{\\<schluessel>}{<Wert>} im deutschen Format, bei Kennzahlen mit
   Konfidenzintervall zusätzlich \\<schluessel>KIu und \\<schluessel>KIo. Schlüssel, die kein gültiger
   LaTeX-Makroname sind (Ziffern, Unterstriche), werden gemeldet und übersprungen, nicht umbenannt.
   Dazu kommen die Text-Kennzahlen aus data/processed/kennzahlen_text.csv, die tools/kennzahlen_text.py vorher neu
   erzeugt (Messwerte, Literatur, LLM-, Benchmark-, Konfusionsdateien); bei gleichem Schlüssel gewinnt kennzahlen.csv,
   der Konflikt wird gemeldet.
3. Schreibt daneben die Bausteine tab_top10.tex und zustrom_redner.tex (tools/latex_bausteine.py).

Aufruf: python tools/make_zahlen.py   (letzter Schritt von run_all.py)
        python tools/make_zahlen.py --ohne-ergaenzung   (kennzahlen.csv nur lesen, z. B. nach Gold v2/LLM-Neulauf)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # wie run_all.py: läuft auch ohne "pip install -e ." (pyproject.toml)

from bundestag import config  # noqa: E402
from bundestag.metrics import KENNZAHLEN_CSV, kennzahl, schreibe_kennzahlen  # noqa: E402

ZAHLEN_TEX = config.ZAHLEN_TEX
TEXT_CSV = config.PROCESSED / "kennzahlen_text.csv"
VALIDIERUNG = config.PROCESSED / "validierung.csv"
# Stichprobe v3: Prüfung nach der Korrektur der Abstimmungserkennung, ohne die Entwicklungsfälle (validate.STICHPROBE_V3)
URTEILE = config.REFERENCE / "abstimmung_urteile_v3.csv"
# Stichprobe v4: prüft die an v3 ergänzten Beschlussart-Regeln an neuen Fällen (validate.STICHPROBE_V4)
URTEILE_V4 = config.REFERENCE / "abstimmung_urteile_v4.csv"
LAUFZEITEN = config.PROCESSED / "laufzeiten.csv"
Q = "make_zahlen.py"
MAKRONAME = re.compile(r"[A-Za-z]+")   # Steuerwort nur aus Buchstaben (TeXbook Kap. 3), an einer Ziffer endet es

# Rundungsregel je Spalte einheit (der Kopf von zahlen.tex verweist hierher): so viele Stellen, wie die Messung trägt.
# Ganze Werte (Zählgrößen wie Reden, Beiträge, Sitzungen) stehen ohne Stellen. Sonst:
#   1 Stelle  Prozent, Prozentpunkte, Wörter/Minute, Minuten/Sitz, Verhältnis, H: die Texte vergleichen
#             Abweichungen ab 0,1 Prozentpunkten (H1-Toleranz 3 pp), feiner trägt die Messung nicht.
#   1 Stelle  auch Sekunden/Block bzw. /Sitzung: Laufzeiten streuen zwischen Läufen um Zehntelsekunden
#             (Benchmark eigen 2,27–3,04 s je Sitzung, benchmark.csv).
#   2 Stellen Anteil, netto, Epsilon-Quadrat: Werte zwischen −1 und 1, 2 Stellen = 1 Prozentpunkt.
#   Gütemaße (einheit F1, kappa, Quote): 2 Stellen; 3 nur, wenn 2 Stellen 1,00 oder 0,00 zeigen würden, der Wert
#             es aber nicht ist (Partei-Übereinstimmung 99,92 % mit drei Stellen statt 1,00, keine Scheinperfektion).
#   p         unter einem Tausendstel als Schranke mit „<“, sonst 2 Stellen (3 unter 0,01); das Makro enthält < bzw. =.
#   ja=1      „ja“ / „nein“.
#   sonst     nach Größe: ab 100 keine, ab 10 eine, darunter 2 Stellen (je Sitz, Minuten, Wörter, Sekunden …).
# Mehr als 2 Stellen nie außerhalb der Gütemaße und p: bei 67 Sitzungen bzw. 60 Gold-Fenstern trägt die
# Stichprobe keine dritte Stelle. Gerundet wird kaufmännisch auf der Dezimaldarstellung (985 Tausendstel → 0,99).
FEST = {"Prozent": 1, "Prozentpunkte": 1, "Wörter/Minute": 1, "Minuten/Sitz": 1, "Verhältnis": 1, "H": 1,
        "Sekunden/Block": 1, "Sekunden/Sitzung": 1,
        "Anteil": 2, "Anteil Reden": 2, "netto": 2, "Epsilon-Quadrat": 2}
GUETEMASSE = {"F1", "kappa", "Quote"}


def _rund(v: float, n: int) -> Decimal:
    # erst 12 signifikante Stellen: Gleitkommareste wie bei κ = 13/40 (binär knapp darunter) runden sonst ab
    d = Decimal(f"{float(v):.12g}").quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
    return abs(d) if d == 0 else d   # kein „−0,0“


def stellen(v: float, einheit: str) -> int:
    if einheit in GUETEMASSE:
        r2 = _rund(v, 2)
        return 3 if abs(r2) in (Decimal(0), Decimal(1)) and abs(v) not in (0.0, 1.0) else 2
    if einheit in FEST:
        return FEST[einheit]
    if float(v).is_integer():
        return 0
    return 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2


def zahl(v: float, n: int) -> str:
    """Deutsches Format: Dezimalkomma als {,} (auch im Mathemodus ohne Zusatzabstand), Tausender ab 5 Stellen
    mit \\, (DIN 5008: 6385, aber 13\\,662), Minus als \\ensuremath{-} (ein - wäre im Text ein Bindestrich)."""
    d = _rund(v, n)
    ganz, _, rest = f"{abs(d):f}".partition(".")
    if len(ganz) >= 5:
        teile = []
        while len(ganz) > 3:
            ganz, teile = ganz[:-3], [ganz[-3:]] + teile
        ganz = "\\,".join([ganz] + teile)
    return ("\\ensuremath{-}" if d < 0 else "") + ganz + ("{,}" + rest if rest else "")


def latex_text(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s).replace("~", r"\textasciitilde{}").replace("^", r"\^{}")


def formatiere(wert, einheit: str = "", n: int | None = None) -> str:
    """Wert → LaTeX-Text; n erzwingt die Stellenzahl (für KI-Grenzen wie beim Punktwert)."""
    if wert is None or (isinstance(wert, float) and math.isnan(wert)) or wert == "":
        return r"\textbf{[FEHLT]}"
    try:
        v = float(wert)
    except (TypeError, ValueError):
        return latex_text(str(wert))
    if einheit == "ja=1":
        return "ja" if v == 1 else "nein"
    if einheit == "p" and n is None:
        if v < 0.001:
            return r"\ensuremath{<}\,0{,}001"
        return r"\ensuremath{=}\," + zahl(v, 3 if v < 0.01 else 2)
    return zahl(v, stellen(v, einheit) if n is None else n)


# Schritt 1: validate.py, parse.py und run_all.py (Laufzeiten) schreiben keine kennzahlen.csv; ohne Nachtrag fehlten
# ihre Zahlen in zahlen.tex.
def _quelle(pfad: Path, spalten: str) -> str:
    return f"{Q}:{pfad.relative_to(ROOT)}:{spalten}"


def clopper_pearson(x: int, n: int) -> tuple:
    """Exaktes 95-%-Intervall für x Treffer aus n: die Abstimmungs-Stichprobe ist zu klein für ein Normal-Intervall."""
    from scipy.stats import beta
    return (beta.ppf(0.025, x, n - x + 1) if x else 0.0), (beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def validierung_kennzahlen() -> list[dict]:
    from bundestag.validate import abstimmung_precision, abstimmung_recall, mikro
    z = []
    if VALIDIERUNG.exists():
        df = pd.read_csv(VALIDIERUNG)
        q = lambda s: _quelle(VALIDIERUNG, s)  # noqa: E731
        voll = df[~df.xml_unvollstaendig.astype(bool)]
        p, r, f = mikro(df, "reden")
        z += [kennzahl("validierungSitzungen", len(df), "Sitzungen", q("sitzung")),
              kennzahl("redenFEinsMikro", f, "F1", q("reden_tp/reden_parser/reden_gold (mikro)")),
              kennzahl("redenPrecisionMikro", p, "Quote", q("reden_tp/reden_parser (mikro)")),
              kennzahl("redenRecallMikro", r, "Quote", q("reden_tp/reden_gold (mikro)")),
              kennzahl("redenFEinsMikroXmlVoll", mikro(voll, "reden")[2], "F1",
                       q("reden_* ohne xml_unvollstaendig (mikro)")),
              kennzahl("xmlUnvollstaendig", int(df.xml_unvollstaendig.astype(bool).sum()), "Sitzungen",
                       q("xml_unvollstaendig")),
              kennzahl("redenFEinsMedian", df.reden_f1.median(), "F1", q("reden_f1 (Median)")),
              kennzahl("redenFEinsMin", df.reden_f1.min(), "F1", q("reden_f1 (Minimum)")),
              kennzahl("beitraegeFEinsMikro", mikro(df, "beitraege")[2], "F1",
                       q("beitraege_tp/beitraege_parser/beitraege_gold (mikro)")),
              kennzahl("ordnungsquoteMedian", df.ordnungsquote.median(), "Quote", q("ordnungsquote (Median)")),
              kennzahl("ordnungsquoteMin", df.ordnungsquote.min(), "Quote", q("ordnungsquote (Minimum)")),
              kennzahl("fuenfGrammRecallMedian", df.recall_5gramm.median(), "Quote", q("recall_5gramm (Median)")),
              kennzahl("fuenfGrammRecallMin", df.recall_5gramm.min(), "Quote", q("recall_5gramm (Minimum)")),
              kennzahl("tokenRecallMedian", df.recall_token.median(), "Quote", q("recall_token (Median)")),
              kennzahl("tokenRecallMin", df.recall_token.min(), "Quote", q("recall_token (Minimum)")),
              kennzahl("parteiQuote", df.partei_gleich.sum() / max(df.partei_pruefbar.sum(), 1), "Quote",
                       q("partei_gleich/partei_pruefbar (Summe)")),
              kennzahl("parteiGleich", int(df.partei_gleich.sum()), "Reden", q("partei_gleich (Summe)")),
              kennzahl("parteiPruefbar", int(df.partei_pruefbar.sum()), "Reden", q("partei_pruefbar (Summe)")),
              kennzahl("drucksachenRecall", df.drs_gefunden.sum() / max(df.drs_gold.sum(), 1), "Quote",
                       q("drs_gefunden/drs_gold (Summe)")),
              kennzahl("drucksachenGefunden", int(df.drs_gefunden.sum()), "Drucksachen", q("drs_gefunden (Summe)")),
              kennzahl("drucksachenGold", int(df.drs_gold.sum()), "Drucksachen", q("drs_gold (Summe)"))]
        if "titel_n" in df:
            # Titelabgleich gegen den XML-Aufrufblock (validate.titel_vergleich): gleich = gleiche Wortfolge ohne Satzzeichen
            tn, tg = int(df.titel_n.sum()), int(df.titel_gesetz_n.sum())
            z += [kennzahl("titelN", tn, "Gegenstände", q("titel_n (Summe)")),
                  kennzahl("titelQuote", df.titel_gleich.sum() / max(tn, 1), "Quote", q("titel_gleich/titel_n (Summe)")),
                  kennzahl("titelGesetzN", tg, "Gesetzentwürfe", q("titel_gesetz_n (Summe)")),
                  kennzahl("titelGesetzGleich", int(df.titel_gesetz_gleich.sum()), "Gesetzentwürfe",
                           q("titel_gesetz_gleich (Summe)")),
                  kennzahl("titelGesetzQuote", df.titel_gesetz_gleich.sum() / max(tg, 1), "Quote",
                           q("titel_gesetz_gleich/titel_gesetz_n (Summe)")),
                  kennzahl("kurztitelXml", int(df.kurztitel_xml.sum()), "Kurztitel", q("kurztitel_xml (Summe)")),
                  kennzahl("kurztitelGleich", int(df.kurztitel_gleich.sum()), "Kurztitel", q("kurztitel_gleich (Summe)")),
                  kennzahl("kurztitelNurParser", int(df.kurztitel_nur_parser.sum()), "Kurztitel",
                           q("kurztitel_nur_parser (Summe)"))]
    else:
        print(f"[zahlen] {VALIDIERUNG.relative_to(ROOT)} fehlt: Validierungskennzahlen übersprungen")
    a = abstimmung_precision(URTEILE)
    if a:
        qa = lambda s: _quelle(URTEILE, s)  # noqa: E731
        richtig, falsch = round(a["precision"] * a["n_true"]), round(a["fehlrate_false"] * a["n_false"])
        z += [kennzahl("abstimmungPrecision", a["precision"], "Anteil", qa("urteil bei abgestimmt=True; Clopper-Pearson"),
                       clopper_pearson(richtig, a["n_true"])),
              kennzahl("abstimmungFehlrateFalse", a["fehlrate_false"], "Anteil",
                       qa("urteil=falsch bei abgestimmt=False; Clopper-Pearson"), clopper_pearson(falsch, a["n_false"])),
              kennzahl("abstimmungRichtig", richtig, "Fälle", qa("urteil=richtig bei abgestimmt=True")),
              kennzahl("abstimmungFaelle", a["n_true"], "Fälle", qa("abgestimmt=True")),
              kennzahl("abstimmungFalsch", falsch, "Fälle", qa("urteil=falsch bei abgestimmt=False")),
              kennzahl("abstimmungFaelleFalse", a["n_false"], "Fälle", qa("abgestimmt=False"))]
        u = pd.read_csv(URTEILE)
        t = u[u.abgestimmt.astype(str) == "True"]
        z += [kennzahl("abstimmungZeileRichtig", int(((t.urteil == "richtig") & (t.zeile_ok == "ja")).sum()), "Fälle",
                       qa("urteil=richtig und zeile_ok=ja bei abgestimmt=True")),
              kennzahl("abstimmungBeschlussartRichtig", int((u.beschlussart_ok == "ja").sum()), "Fälle",
                       qa("beschlussart_ok=ja (alle Fälle, Parserstand der Stichprobe)")),
              kennzahl("abstimmungStichprobeN", len(u), "Fälle", qa("Zeilen"))]
        g = pd.read_parquet(config.PROCESSED / "gesetze.parquet", columns=["abgestimmt"])
        n_t, n_f = int(g.abgestimmt.sum()), int((~g.abgestimmt).sum())
        # Recall aus den beiden Schichten; Intervall aus den Clopper-Pearson-Grenzen beider Schichten (konservativ)
        p_lo, p_hi = clopper_pearson(richtig, a["n_true"])
        m_lo, m_hi = clopper_pearson(falsch, a["n_false"])
        z += [kennzahl("abstimmungRecallGeschaetzt", abstimmung_recall(a["precision"], a["fehlrate_false"], n_t, n_f),
                       "Anteil", qa(f"Precision × {n_t} / (Precision × {n_t} + Fehlrate × {n_f}); Grenzen aus CP"),
                       (abstimmung_recall(p_lo, m_hi, n_t, n_f), abstimmung_recall(p_hi, m_lo, n_t, n_f))),
              kennzahl("gegenstaendeAbgestimmt", n_t, "Gegenstände", _quelle(config.PROCESSED / "gesetze.parquet",
                                                                              "abgestimmt=True"))]
    if URTEILE_V4.exists():
        qv = lambda s: _quelle(URTEILE_V4, s)  # noqa: E731
        u = pd.read_csv(URTEILE_V4)
        t, f = u[u.abgestimmt.astype(str) == "True"], u[u.abgestimmt.astype(str) == "False"]
        z += [kennzahl("abstimmungVierBeschlussartRichtig", int((u.beschlussart_ok == "ja").sum()), "Fälle",
                       qv("beschlussart_ok=ja (alle Fälle, Parserstand nach den Regeln aus v3)")),
              kennzahl("abstimmungVierN", len(u), "Fälle", qv("Zeilen")),
              kennzahl("abstimmungVierRichtig", int((t.urteil == "richtig").sum()), "Fälle",
                       qv("urteil=richtig bei abgestimmt=True")),
              kennzahl("abstimmungVierFaelle", len(t), "Fälle", qv("abgestimmt=True")),
              kennzahl("abstimmungVierFalsch", int((f.urteil == "falsch").sum()), "Fälle",
                       qv("urteil=falsch bei abgestimmt=False")),
              kennzahl("abstimmungVierFaelleFalse", len(f), "Fälle", qv("abgestimmt=False"))]
    return z


def parser_kennzahlen() -> list[dict]:
    z = []
    for schluessel, datei, einheit in (("parserSitzungen", "sitzungen", "Sitzungen"),
                                       ("parserBeitraege", "beitraege", "Beiträge"),
                                       ("parserReden", "reden", "Reden"),
                                       ("parserGesetze", "gesetze", "Beratungsgegenstände"),
                                       ("parserGesetzeEinheit", "gesetze_einheit", "Gesetzentwürfe")):
        pfad = config.PROCESSED / f"{datei}.parquet"
        if pfad.exists():
            n = pd.read_parquet(pfad, columns=[]).shape[0]
            z.append(kennzahl(schluessel, n, einheit, _quelle(pfad, "Zeilen")))
    pfad = config.PROCESSED / "gesetze_einheit.parquet"
    if pfad.exists():
        # Gesetzentwürfe ohne die Einzelpläne der Haushaltswoche (wie metrics.top10_gesetze), davon mindestens einmal
        # gemeinsam mit anderen Vorlagen beraten (parse.beteiligte: gegenstaende_im_top)
        e = pd.read_parquet(pfad, columns=["titel", "gegenstaende_im_top_max"])
        e = e[~e.titel.fillna("").str.startswith("hier: Einzelplan")]
        z += [kennzahl("gesetzeOhneEinzelplan", len(e), "Gesetzentwürfe", _quelle(pfad, "titel ohne Einzelplan")),
              kennzahl("gesetzeGemeinsamBeraten", int((e.gegenstaende_im_top_max > 1).sum()), "Gesetzentwürfe",
                       _quelle(pfad, "gegenstaende_im_top_max > 1, ohne Einzelplan"))]
    pfad = config.PROCESSED / "reden.parquet"
    if pfad.exists():
        # gleiche Abgrenzung wie metrics.py zuProtokollGesamt: zu Protokoll gegeben + Erklärungen nach § 31 GO
        rt = pd.read_parquet(pfad, columns=["redetyp"]).redetyp
        z.append(kennzahl("parserZuProtokoll", int(rt.isin(["zu_protokoll", "erklaerung_go31"]).sum()), "Reden",
                          _quelle(pfad, "redetyp in zu_protokoll/erklaerung_go31")))
    return z


def laufzeit_kennzahlen() -> list[dict]:
    if not LAUFZEITEN.exists():
        print(f"[zahlen] {LAUFZEITEN.relative_to(ROOT)} fehlt (run_all.py noch nicht gelaufen): Laufzeiten übersprungen")
        return []
    with open(LAUFZEITEN, encoding="utf-8") as f:
        # ohne die eigene Zeile: sie stammt noch aus dem vorigen Lauf, dieser Lauf misst sich erst danach
        lz = [z for z in csv.DictReader(f)
              if z["status"] in ("OK", "TEIL") and z["sekunden"] and z["schritt"] != "zahlen"]
    q = _quelle(LAUFZEITEN, "sekunden")
    z = [kennzahl("laufzeit" + "".join(t.capitalize() for t in r["schritt"].split("_")), float(r["sekunden"]),
                  "Sekunden", f"{q} (schritt={r['schritt']})") for r in lz]
    gesamt = sum(float(r["sekunden"]) for r in lz)
    z += [kennzahl("laufzeitGesamt", gesamt, "Sekunden", f"{q} (Summe OK/TEIL ohne zahlen)"),
          kennzahl("laufzeitGesamtMinuten", gesamt / 60, "Minuten", f"{q} (Summe OK/TEIL ohne zahlen / 60)")]
    return z


def ergaenze_kennzahlen() -> tuple[int, int]:
    """Schreibt nur Schlüssel, die kein anderes Skript liefert; schreibe_kennzahlen ersetzt gleichnamige Zeilen
    jeder Quelle (metrics.py), daher der Filter."""
    kandidaten = validierung_kennzahlen() + parser_kennzahlen() + laufzeit_kennzahlen()
    vorhanden = set()
    if KENNZAHLEN_CSV.exists():
        k = pd.read_csv(KENNZAHLEN_CSV)
        vorhanden = set(k.schluessel[~k.quelle.astype(str).str.startswith(Q + ":")])
    neu = [z for z in kandidaten if z["schluessel"] not in vorhanden]
    schreibe_kennzahlen(neu, Q)
    return len(neu), len(kandidaten) - len(neu)


# Schritt 2: ein Makro je Kennzahl, damit tools/check_numbers.py jede Zahl im Kapiteltext gegen zahlen.tex prüfen kann.
def schreibe_tex(k: pd.DataFrame, ziel: Path = ZAHLEN_TEX) -> dict:
    zeilen = [f"% Erzeugt von tools/make_zahlen.py aus {KENNZAHLEN_CSV.relative_to(ROOT)} und {TEXT_CSV.name} am "
              f"{datetime.now():%d.%m.%Y %H:%M}; nicht von Hand ändern, sondern die Pipeline neu laufen lassen.",
              "% Rundung: Regel im Kopf von tools/make_zahlen.py. p-Makros enthalten das Relationszeichen (< oder =).",
              "% Je Zeile: Makro, dann Einheit und Quelle (Skript:Datei/Spalte) als Kommentar.", ""]
    bericht = {"makros": 0, "ki": 0, "fehlt": [], "ungueltig": [], "doppelt": []}
    belegt: set[str] = set()
    for r in k.itertuples(index=False):
        name = str(r.schluessel)
        if not MAKRONAME.fullmatch(name):
            bericht["ungueltig"].append(name)
            continue
        hat_ki = pd.notna(r.ki_unten) and pd.notna(r.ki_oben)
        namen = [name] + ([name + "KIu", name + "KIo"] if hat_ki else [])
        if belegt.intersection(namen):
            bericht["doppelt"].append(name)
            continue
        belegt.update(namen)
        einheit = "" if pd.isna(r.einheit) else str(r.einheit)
        kommentar = re.sub(r"[{}%\n]", " ", f"{einheit}; {r.quelle}")
        wert = formatiere(r.wert, einheit)
        if "[FEHLT]" in wert:
            bericht["fehlt"].append(name)
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert}}} % {kommentar}")
        bericht["makros"] += 1
        if hat_ki:
            try:
                n = stellen(float(r.wert), einheit) if einheit != "p" else 3
            except (TypeError, ValueError):
                n = 2
            for suffix, grenze in (("KIu", r.ki_unten), ("KIo", r.ki_oben)):
                zeilen.append(f"\\newcommand{{\\{name}{suffix}}}{{{formatiere(grenze, einheit, n)}}}")
                bericht["makros"] += 1
            bericht["ki"] += 1
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_suffix(".tmp")
    tmp.write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    tmp.replace(ziel)   # atomar: ein paralleler LaTeX-Lauf (main.tex Z. 55 liest zahlen.tex) sieht nie eine halbe Datei
    return bericht


def mit_text_kennzahlen(k: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt kennzahlen_text.csv neu (tools/kennzahlen_text.py) und hängt sie an; kennzahlen.csv gewinnt."""
    try:   # Import über den Pfad: tools/ ist kein Paket, die Tests laden dieses Modul ebenfalls per Dateipfad
        import importlib.util
        spec = importlib.util.spec_from_file_location("kennzahlen_text", ROOT / "tools" / "kennzahlen_text.py")
        modul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modul)
        modul.erzeuge(TEXT_CSV)
    except Exception as e:  # noqa: BLE001  zahlen.tex soll auch mit der alten kennzahlen_text.csv entstehen
        print(f"[zahlen] kennzahlen_text.py fehlgeschlagen ({type(e).__name__}: {e}), nutze vorhandene Datei")
    if not TEXT_CSV.exists():
        return k
    t = pd.read_csv(TEXT_CSV)
    konflikt = t[t.schluessel.isin(k.schluessel)]
    for r in konflikt.itertuples(index=False):
        alt = k.loc[k.schluessel == r.schluessel, "wert"].iloc[0]
        print(f"[zahlen]   Konflikt {r.schluessel}: kennzahlen.csv {alt} gewinnt gegen {TEXT_CSV.name} {r.wert}")
    print(f"[zahlen] {TEXT_CSV.relative_to(ROOT)}: {len(t) - len(konflikt)} Kennzahlen übernommen, "
          f"{len(konflikt)} Konflikte")
    return pd.concat([k, t[~t.schluessel.isin(k.schluessel)]], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohne-ergaenzung", action="store_true",
                    help="kennzahlen.csv nicht ergänzen, nur zahlen.tex schreiben")
    a = ap.parse_args(argv)
    if not a.ohne_ergaenzung:
        n_neu, n_da = ergaenze_kennzahlen()
        print(f"[zahlen] kennzahlen.csv: {n_neu} Kennzahlen von {Q} geschrieben, {n_da} schon von anderen Skripten da")
    if not KENNZAHLEN_CSV.exists():
        print(f"[zahlen] {KENNZAHLEN_CSV.relative_to(ROOT)} fehlt")
        return 1
    b = schreibe_tex(mit_text_kennzahlen(pd.read_csv(KENNZAHLEN_CSV)))
    print(f"[zahlen] {ZAHLEN_TEX.relative_to(ROOT)}: {b['makros']} Makros ({b['ki']} mit KIu/KIo), "
          f"[FEHLT] {len(b['fehlt'])}, übersprungen {len(b['ungueltig']) + len(b['doppelt'])}")
    for art, liste in (("[FEHLT]", b["fehlt"]), ("kein gültiger Makroname", b["ungueltig"]),
                       ("doppelt", b["doppelt"])):
        if liste:
            print(f"[zahlen]   {art}: {', '.join(liste[:20])}{' …' if len(liste) > 20 else ''}")
    latex_bausteine()
    return 0


def latex_bausteine() -> None:
    """tab_top10.tex und zustrom_redner.tex (tools/latex_bausteine.py) in den Ordner von zahlen.tex."""
    try:   # Import über den Pfad wie mit_text_kennzahlen
        import importlib.util
        spec = importlib.util.spec_from_file_location("latex_bausteine", ROOT / "tools" / "latex_bausteine.py")
        modul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modul)
        modul.erzeuge()
    except Exception as e:  # noqa: BLE001  zahlen.tex ist schon geschrieben, die Bausteine sind Zusatz
        print(f"[zahlen] latex_bausteine.py fehlgeschlagen ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
