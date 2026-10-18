"""Kennzahlen für Kapitel 1–4, die kein Pipeline-Skript in kennzahlen.csv schreibt → data/processed/kennzahlen_text.csv.

Liest die Ergebnisdateien bei jedem Lauf neu (benchmark.csv, konfusion_bert_mehrheit.csv, data/llm/*.csv,
validierung.csv, fehler.csv, laufzeiten.csv, laufzeiten_teil.csv, abstimmung_urteile*.csv, data/txt/20211.txt,
Konstanten im Code), damit ein Neulauf der Pipeline sie mitzieht. Zahlen, die keine ausgelieferte Datei enthält,
stehen unten als Tabellen mit Herkunft in der Spalte quelle: Messungen früherer Läufe, vorab festgelegte Schwellen,
die Zählungen der Fehlertypenanalyse und Angaben aus Guhr et al. (2020).
Spalten wie kennzahlen.csv. Bei gleichem Schlüssel gewinnt kennzahlen.csv (make_zahlen.py meldet den Konflikt).

Reihenfolge: tools/make_zahlen.py ruft dieses Skript vor dem Lesen auf. Nach Neuberechnung (Gold v2, LLM --nur-cache)
genügt daher  python tools/make_zahlen.py --ohne-ergaenzung ; allein:  python tools/kennzahlen_text.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bundestag import config  # noqa: E402
from bundestag.validate import abstimmung_precision, mikro  # noqa: E402

PROC, LLM, REF = config.PROCESSED, config.LLM_DIR, config.REFERENCE
ZIEL = PROC / "kennzahlen_text.csv"
SPALTEN = ["schluessel", "wert", "ki_unten", "ki_oben", "einheit", "quelle"]
WORT = ["Eins", "Zwei", "Drei"]
FRAKTION = {"AfD": "AfD", "BSW": "BSW", "BÜNDNIS 90/DIE GRÜNEN": "Gruene", "CDU/CSU": "Union", "Die Linke": "Linke",
            "FDP": "FDP", "SPD": "SPD", "fraktionslos": "Fraktionslos"}   # Kürzel wie in kennzahlen.csv (redenUnion …)


def k(schluessel: str, wert, einheit: str, quelle: str, ki=(None, None)) -> dict:
    return dict(zip(SPALTEN, (schluessel, wert, ki[0], ki[1], einheit, quelle)))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def camel(s: str) -> str:
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return "".join(t[:1].upper() + t[1:] for t in re.split(r"[^A-Za-z]+", s) if t)


def clopper_pearson(x: int, n: int) -> tuple:
    try:
        from scipy.stats import beta
    except ImportError:
        return None, None
    return (beta.ppf(0.025, x, n - x + 1) if x else 0.0), (beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


# Kennzahlen aus Ergebnisdateien, bei jedem Lauf neu gelesen: ein Neulauf der Pipeline zieht sie mit.
def benchmark() -> list[dict]:
    pfad = REF / "benchmark.csv"
    b = pd.read_csv(pfad)
    m = b.groupby("variante")[["token_recall", "fuenfgramm_recall", "ordnungsquote", "laufzeit_s"]].mean()
    z = [k("benchSitzungen", b.sitzung.nunique(), "Sitzungen", f"{rel(pfad)}:sitzung (Anzahl)")]
    for v, n in (("sort", "Sort"), ("columns", "Columns"), ("eigen", "Eigen"), ("pdfminer", "Pdfminer"),
                 ("pdfplumber", "Pdfplumber")):
        if v in m.index:
            q = f"{rel(pfad)}:%s (Mittel variante={v}, {int((b.variante == v).sum())} Sitzungen)"
            z += [k(f"bench{n}TokenRecall", m.at[v, "token_recall"], "Quote", q % "token_recall"),
                  k(f"bench{n}FuenfGrammRecall", m.at[v, "fuenfgramm_recall"], "Quote", q % "fuenfgramm_recall"),
                  k(f"bench{n}Ordnungsquote", m.at[v, "ordnungsquote"], "Quote", q % "ordnungsquote"),
                  k(f"bench{n}Laufzeit", m.at[v, "laufzeit_s"], "Sekunden/Sitzung", q % "laufzeit_s")]
    return z


def konfusion() -> list[dict]:
    pfad = PROC / "konfusion_bert_mehrheit.csv"
    t = pd.read_csv(pfad, index_col=0)
    kurz = {"negativ": "Neg", "neutral": "Neu", "positiv": "Pos"}
    z = []
    for zeile, a in kurz.items():   # Schlüssel konf<Mehrheit><BERT>, z. B. konfNegNeu = Mehrheit negativ, BERT neutral
        for spalte, b in kurz.items():
            z.append(k(f"konf{a}{b}", int(t.at[zeile, f"bert_{spalte}"]), "Fenster",
                       f"{rel(pfad)}:Zeile mehrheit={zeile}, Spalte bert_{spalte}"))
        z.append(k(f"konfMehrheit{a}", int(t.loc[zeile].sum()), "Fenster", f"{rel(pfad)}:Zeilensumme mehrheit={zeile}"))
        z.append(k(f"konfBert{a}", int(t[f"bert_{zeile}"].sum()), "Fenster", f"{rel(pfad)}:Spaltensumme bert_{zeile}"))
    z.append(k("konfGesamt", int(t.to_numpy().sum()), "Fenster", f"{rel(pfad)}:Summe aller Zellen"))
    return z


LLM_KENNZAHL = {  # kennzahlen_gesetze.csv: kennzahl → (Schlüssel, Einheit); Quoten rechnet llm_quoten() exakt nach
    "bloecke_gesamt": ("llmBloeckeGesamt", "Blöcke"),
    "bloecke_filter": ("llmBloeckeFilter", "Blöcke"), "bloecke_bearbeitet": ("llmBloeckeBearbeitet", "Blöcke"),
    "bloecke_offen": ("llmBloeckeOffen", "Blöcke"), "bloecke_gekuerzt": ("llmBloeckeGekuerzt", "Blöcke"),
    "bloecke_gesetzentwurf": ("llmBloeckeGesetzentwurf", "Blöcke"),
    "gesetzentwurf_abgedeckt": ("llmGesetzentwurfAbgedeckt", "Blöcke"),
    "gesetzentwurf_offen": ("llmGesetzentwurfOffen", "Blöcke"),
    "gueltig_erst_nach_wiederholung": ("llmGueltigNachWiederholung", "Blöcke"),
    "llm_sekunden_summe": ("llmSekundenSumme", "Sekunden"),
    "sekunden_je_block_mittel": ("llmSekundenJeBlockMittel", "Sekunden/Block"),
    "sekunden_je_block_median": ("llmSekundenJeBlockMedian", "Sekunden/Block"),
    "eval_count_mittel": ("llmTokenJeBlockMittel", "Token/Block"),
}


def llm_kennzahlen() -> list[dict]:
    pfad = LLM / "kennzahlen_gesetze.csv"
    z = []
    for r in pd.read_csv(pfad).itertuples(index=False):
        q = f"{rel(pfad)}:kennzahl={r.kennzahl}"
        if r.kennzahl in LLM_KENNZAHL:
            s, e = LLM_KENNZAHL[r.kennzahl]
            z.append(k(s, r.wert, e, q))
        elif r.kennzahl.startswith(("klasse_", "typ_diff_", "stufe_diff_")):
            z.append(k("llm" + camel(r.kennzahl.replace("–", "ohne")), int(r.wert), "Blöcke", q))   # – = kein Typ
        elif not r.kennzahl.startswith(("gueltig_anteil", "quote_", "laufzeit_wand_s")):   # Wandzeit: laufzeitLlmGesetze
            print(f"[text] {rel(pfad)}: {r.kennzahl} ohne Schlüssel, übersprungen")
    return z


def llm_quoten() -> list[dict]:
    # exakt aus divergenz.csv mit den Formeln von llm.py kennzahlen(): kennzahlen_gesetze.csv rundet auf 4 Stellen,
    # gerundet 97,55 % ergäbe 97,6 % statt 159/163 = 97,5 %
    pfad = LLM / "divergenz.csv"
    d = pd.read_csv(pfad)
    ja = lambda s: s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})  # noqa: E731  leer bleibt leer wie in .mean()
    g = d[ja(d.gueltig) == 1]
    q = lambda s: f"{rel(pfad)}:{s} (Formel wie llm.py kennzahlen)"  # noqa: E731
    quoten = {"llmGueltig": (100 * ja(d.gueltig).mean(), "gueltig", "gueltig_anteil"),
              "llmQuoteDrucksachen": (100 * ja(g.drs_gleich).mean(), "drs_gleich bei gueltig", "quote_drucksachen"),
              "llmQuoteTitel": (100 * (g.titel_quote >= 0.9).mean(), "titel_quote >= 0,9 bei gueltig",
                                "quote_titel_ab_0_9"),
              "llmQuoteKurztitel": (100 * ja(g.kurztitel_gleich).mean(), "kurztitel_gleich bei gueltig", "quote_kurztitel"),
              "llmQuoteTyp": (100 * ja(g.typ_gleich).mean(), "typ_gleich bei gueltig", "quote_typ"),
              "llmQuoteStufe": (100 * ja(g.stufe_gleich).mean(), "stufe_gleich bei gueltig", "quote_stufe"),
              "llmQuoteAnzahlGegenstaende": (100 * (g.n_llm == g.n_regex).mean(), "n_llm = n_regex bei gueltig",
                                             "quote_anzahl_gegenstaende")}
    kpfad = LLM / "kennzahlen_gesetze.csv"
    csv = pd.read_csv(kpfad).set_index("kennzahl").wert if kpfad.exists() else pd.Series(dtype=float)
    for s, (wert, _, kennzahl) in quoten.items():   # Gegenprobe: divergenz.csv und kennzahlen_gesetze.csv vom selben Lauf?
        if kennzahl in csv.index and abs(wert / 100 - csv[kennzahl]) > 0.0005:
            print(f"[text] {s}: divergenz.csv {wert / 100:.4f} gegen {rel(kpfad)} {csv[kennzahl]:.4f}")
    return [k(s, w, "Prozent", q(spalte)) for s, (w, spalte, _) in quoten.items()] + \
        [k("llmGueltigBloecke", int((ja(d.gueltig) == 1).sum()), "Blöcke", q("gueltig"))]


def modellvergleich() -> list[dict]:
    pfad = LLM / "modellvergleich.csv"
    mv = pd.read_csv(pfad)
    namen = {"qwen2.5:7b": "QwenSieben", "qwen2.5:3b": "QwenDrei", "deepseek-r1:7b": "DeepseekSieben"}
    z = [k("vergleichBloecke", int(mv.groupby("modell").size().max()), "Blöcke", f"{rel(pfad)}:Zeilen je modell")]
    for modell, g in mv.groupby("modell"):
        if modell not in namen:
            print(f"[text] {rel(pfad)}: Modell {modell} ohne Schlüssel, übersprungen")
            continue
        n, q = namen[modell], f"{rel(pfad)}:%s (modell={modell})"
        z += [k(f"vergleich{n}Sekunden", g.sekunden.mean(), "Sekunden/Block", q % "sekunden (Mittel)"),
              k(f"vergleich{n}Token", g.eval_count.mean(), "Token/Block", q % "eval_count (Mittel)"),
              k(f"vergleich{n}Gleich", int((g.klasse == "gleich").sum()), "Blöcke", q % "klasse=gleich"),
              # Kern gleich: Drucksachen, Art und Beratungsstufe wie beim Parser, höchstens der Titel weicht ab
              k(f"vergleich{n}KernGleich", int(g.klasse.isin(["gleich", "nur_titelvariante"]).sum()), "Blöcke",
                q % "klasse in gleich/nur_titelvariante"),
              k(f"vergleich{n}Gueltig", int(g.gueltig.astype(bool).sum()), "Blöcke", q % "gueltig")]
        if g.thinking_zeichen.max() > 0:
            z += [k(f"vergleich{n}DenkenMin", int(g.thinking_zeichen.min()), "Zeichen", q % "thinking_zeichen (Min)"),
                  k(f"vergleich{n}DenkenMax", int(g.thinking_zeichen.max()), "Zeichen", q % "thinking_zeichen (Max)")]
    return z


def schiedsrichter() -> list[dict]:
    pfad = LLM / "schiedsrichter.csv"
    s = pd.read_csv(pfad)
    q = lambda x: f"{rel(pfad)}:{x}"  # noqa: E731
    z = [k("schiedsFaelle", len(s), "Fälle", q("Zeilen"))]
    for g in ("regex", "llm", "beide_richtig", "beide_falsch"):   # alle vier Kategorien, auch wenn 0
        z.append(k("schiedsGewinner" + camel(g), int((s.gewinner == g).sum()), "Fälle", q(f"gewinner={g}")))
    for schluessel, muster in (("schiedsLlmKurztitelErfunden", "erfindet Kurztitel"),
                               ("schiedsLlmBeAbgespalten", "spaltet BE"),
                               ("schiedsRegexKlammerKurztitel", "Regex verpasst Klammer-Kurztitel")):
        z.append(k(schluessel, int(s.fehlerklasse.str.contains(muster, regex=False).sum()), "Fälle",
                   q(f"fehlerklasse enthält „{muster}“")))
    for spalte, praefix in (("divergenzklasse", "schiedsKlasse"), ("feld", "schiedsFeld")):
        for wert, n in s[spalte].value_counts().items():
            z.append(k(praefix + camel(str(wert)), int(n), "Fälle", q(f"{spalte}={wert}")))
    return z


def ki_niveau() -> list[dict]:
    datei = "src/bundestag/stats.py"   # stats.ki: Perzentil-KI, von stats_sentiment und sentiment_kappa importiert
    zeilen = (ROOT / datei).read_text(encoding="utf-8").splitlines()
    treffer = [(nr, m) for nr, t in enumerate(zeilen, 1) if (m := re.search(r"nanquantile\(x, \[(\.\d+), (\.\d+)\]\)", t))]
    if len(treffer) != 1:
        print(f"[text] kiNiveau: Quantilaufruf in {datei} {len(treffer)}× gefunden, übersprungen")
        return []
    nr, m = treffer[0]
    return [k("kiNiveau", round(100 * (float(m.group(2)) - float(m.group(1))), 6), "Prozent KI",
              f"{datei}:{nr} (Quantile {m.group(1)}/{m.group(2)})")]


def zustrom() -> list[dict]:
    # Abgrenzung wie metrics.top10_gesetze (plenarrede, Fraktion sonst Partei); Plos/parteilos = fraktionslos (Wissing)
    pfad, rpfad = PROC / "gesetz_beteiligte.parquet", PROC / "reden.parquet"
    gb = pd.read_parquet(pfad, columns=["rede_id", "drucksache", "redetyp", "sitzung", "name"])
    gb = gb[(gb.drucksache == "20/12804") & (gb.redetyp == "plenarrede")]
    gb = gb.merge(pd.read_parquet(rpfad, columns=["rede_id", "fraktion", "partei"]), on="rede_id", how="left")
    partei = {"SPD": "SPD", "CDU": "CDU/CSU", "CSU": "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
              "FDP": "FDP", "AfD": "AfD", "Die Linke": "Die Linke", "BSW": "BSW", "Plos": "fraktionslos",
              "parteilos": "fraktionslos"}
    gb["fr"] = gb.fraktion.fillna(gb.partei.map(partei)).fillna("ohne Fraktion")
    kurz = {"CDU/CSU": "Union", "SPD": "SPD", "BÜNDNIS 90/DIE GRÜNEN": "Gruene", "FDP": "FDP", "AfD": "AfD",
            "Die Linke": "Linke", "BSW": "BSW", "fraktionslos": "Fraktionslos", "ohne Fraktion": "OhneFraktion"}
    z = []
    for teil, praefix, text in ((gb, "zustromRedner", "alle Lesungen"), (gb[gb.sitzung == 211], "zustromRednerElf",
                                                                        "2./3. Beratung, sitzung=211")):
        je = teil.groupby("fr").name.nunique()
        for fr, kz in kurz.items():
            if fr in je.index or fr != "ohne Fraktion":
                z.append(k(praefix + kz, int(je.get(fr, 0)), "Personen",
                           f"{rel(pfad)} × {rel(rpfad)}:name (nunique, drucksache=20/12804, plenarrede, {text}, {fr})"))
        z.append(k(praefix + "Summe", int(je.sum()), "Personen", f"{rel(pfad)}:name (Summe je Fraktion, {text})"))
        if int(je.sum()) != teil.name.nunique():
            print(f"[text] {praefix}: Summe je Fraktion {int(je.sum())} ≠ Personen {teil.name.nunique()} (Fraktionswechsel)")
    # Kopfzeilen von LaTeX/zustrom_redner.tex: 185 = 1. Beratung (ZP 3), 211 = 2./3. Beratung (ZP 35); Kurz = Personen
    # nur mit Kurzintervention/Erwiderung (Farle 185; Banaszak, Frohnmaier 211), sonst stimmt Kopfzahl ≠ Namenliste
    ke = pd.read_parquet(pfad, columns=["drucksache", "redetyp", "sitzung", "name"])
    ke = ke[(ke.drucksache == "20/12804") & ke.redetyp.isin(["kurzintervention", "erwiderung"])]
    q = f"{rel(pfad)} × {rel(rpfad)}:drucksache=20/12804"
    for s, sitz in (("", None), ("Erste", 185), ("Elf", 211)):
        p, kk = (gb, ke) if sitz is None else (gb[gb.sitzung == sitz], ke[ke.sitzung == sitz])
        text = "alle Lesungen" if sitz is None else f"sitzung={sitz}"
        z.append(k(f"zustromRedner{s}Kurz", len(set(kk.name) - set(p.name)), "Personen",
                   f"{q}, name nur kurzintervention/erwiderung, {text}"))
        if s:
            z.append(k(f"zustromReden{s}", len(p), "Reden", f"{q}, Zeilen plenarrede, {text}"))
        if s == "Erste":
            z.append(k("zustromRednerErsteSumme", int(p.groupby("fr").name.nunique().sum()), "Personen",
                       f"{q}, name (Summe je Fraktion, plenarrede, {text})"))
    kz = PROC / "kennzahlen.csv"
    if kz.exists():
        soll = pd.read_csv(kz).set_index("schluessel").wert.get("zustromRedner")
        summe = next(x["wert"] for x in z if x["schluessel"] == "zustromRednerSumme")
        if soll is not None and int(float(soll)) != summe:
            print(f"[text] zustromRednerSumme {summe} ≠ zustromRedner {soll} (kennzahlen.csv)")
    return z


# Namentliche Abstimmung über das Zustrombegrenzungsgesetz (20211:3517 und 3519): das Ergebnis läuft über einen
# Zwischenruf („(Beifall bei der Linken)“) in die übernächste Zeile, daher sucht die Regex über Zeilengrenzen hinweg.
STIMMEN_RE = re.compile(r"Abgegebene Stimmkarten (\d+)\. Mit Ja haben (\d+) Abgeordnete gestimmt, mit Nein haben (\d+) "
                        r"Abgeordnete gestimmt\.\s+(?:\([^()\n]*\)\s*)?(\d+) Abgeordnete haben sich enthalten\. "
                        r"Der Gesetzentwurf ist abgelehnt")


def zustrom_abstimmung() -> list[dict]:
    pfad = ROOT / "data" / "txt" / "20211.txt"
    text = pfad.read_text(encoding="utf-8")
    treffer = list(STIMMEN_RE.finditer(text))
    if len(treffer) != 1:   # nur eine abgelehnte namentliche Abstimmung mit dieser Formel in Sitzung 211
        print(f"[text] zustromStimmen…: Ergebnisformel in {rel(pfad)} {len(treffer)}× gefunden, übersprungen")
        return []
    m = treffer[0]
    nr, ende = text.count("\n", 0, m.start()) + 1, text.count("\n", 0, m.end()) + 1
    werte = [int(x) for x in m.groups()]
    if werte[1] + werte[2] + werte[3] != werte[0]:
        print(f"[text] zustromStimmen: Ja + Nein + Enthaltungen ≠ abgegeben {werte}")
    q = f"{rel(pfad)} Z. {nr}–{ende}"
    return [k(s, w, "Stimmen", q) for s, w in zip(("zustromStimmenAbgegeben", "zustromStimmenJa",
                                                 "zustromStimmenNein", "zustromStimmenEnthaltungen"), werte)]


def tests() -> list[dict]:
    import importlib.util
    import subprocess
    z = [k("testsDateien", len(list((ROOT / "tests").glob("test_*.py"))), "Dateien", "tests/test_*.py (Anzahl)")]
    if importlib.util.find_spec("pytest") is None:   # Laufzeitumgebung ohne Testabhängigkeiten: nur die Dateizahl
        print("[text] pytest nicht verfügbar: testsAnzahl übersprungen")
        return z
    lauf = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    m = re.search(r"(\d+) tests? collected", lauf.stdout)
    if not m:
        print(f"[text] testsAnzahl: keine Zeile „N tests collected“ (Rückgabecode {lauf.returncode}), übersprungen")
        return z
    return z + [k("testsAnzahl", int(m.group(1)), "Tests", "pytest --collect-only -q (Zeile „N tests collected“)")]


def top_zehn() -> list[dict]:
    pfad = LLM / "top10_zusammenfassungen.csv"
    t = pd.read_csv(pfad)
    p = t.plausibel.map({True: True, False: False, "True": True, "False": False})
    q = lambda s: f"{rel(pfad)}:{s}"  # noqa: E731
    return [k("topZehnPositionen", len(t), "Positionen", q("Zeilen")),
            k("topZehnGesetze", t.drucksache.nunique(), "Gesetzentwürfe", q("drucksache (verschieden)")),
            k("topZehnPlausibelJa", int((p == True).sum()), "Positionen", q("plausibel=True")),  # noqa: E712
            k("topZehnPlausibelNein", int((p == False).sum()), "Positionen", q("plausibel=False")),  # noqa: E712
            k("topZehnPruefbar", int(p.notna().sum()), "Positionen", q("plausibel nicht leer")),
            k("topZehnOhneErwartung", int(p.isna().sum()), "Positionen", q("plausibel leer"))]


# Sentiment-Vollauf mit 150-Wort-Fenstern ohne vollen Cache (9391 Fenster aus dem Cache, 21 674 neu gerechnet), Werte
# aus der Ausgabe von sentiment.py --voll. Ein späterer Lauf trifft den vollen Cache und misst Laufzeit und Durchsatz
# nicht mehr.
Q_VOLLAUF = "Messung Sentiment-Vollauf 150 Wörter, Ausgabe sentiment.py --voll"
VOLLAUF_150 = [  # (Schlüssel, Wert, Einheit, Feld der Ausgabe)
    ("laufzeitSentimentKalt", 367, "Sekunden", "Zeile Vollauf, s"),
    ("bertFensterHundertfuenfzig", 31065, "Fenster", "Zeile Vollauf, Fenster"),
    ("bertRedenHundertfuenfzig", 10024, "Reden", "Zeile Vollauf, Reden"),
    ("bertRedenNeutralHundertfuenfzig", 7103, "Reden", "Zeile Vollauf, Reden neutral"),
    ("bertRedenNegativHundertfuenfzig", 2720, "Reden", "Zeile Vollauf, Reden negativ"),
    ("bertRedenPositivHundertfuenfzig", 201, "Reden", "Zeile Vollauf, Reden positiv"),
    ("bertPlenarredenNeutralHundertfuenfzig", 4289, "Reden", "Zeile Vollauf, Plenarreden neutral"),
    ("bertPlenarredenNegativHundertfuenfzig", 2034, "Reden", "Zeile Vollauf, Plenarreden negativ"),
    ("bertPlenarredenPositivHundertfuenfzig", 53, "Reden", "Zeile Vollauf, Plenarreden positiv"),
    ("bertFensterCacheHundertfuenfzig", 9391, "Fenster", "im Cache"),
    ("bertFensterGerechnetHundertfuenfzig", 21674, "Fenster", "zu rechnen"),
    # letzte Fortschrittszeile = Durchsatz über den ganzen gerechneten Teil
    ("bertDurchsatzHundertfuenfzig", 59.2, "Fenster/Sekunde", "letzte Fortschrittszeile"),
    ("bertTokMedianHundertfuenfzig", 207, "Tokens", "Tokens Median"),
    ("bertTokPneunundneunzigHundertfuenfzig", 275, "Tokens", "Tokens P99"),
    ("bertTokMaxHundertfuenfzig", 432, "Tokens", "Tokens max"),
    ("bertAnteilUeberTrainingHundertfuenfzig", 2.8, "Prozent", "über 256"),
]


def laufzeiten_kalt() -> list[dict]:
    # laufzeiten.csv hält für sentiment nur den Cache-Lauf (9 s) und für llm ÜBERSPRUNGEN; Kaltläufe aus der Teil-Datei
    # und VOLLAUF_150
    pfad = PROC / "laufzeiten_teil.csv"
    lz = pd.read_csv(pfad)
    z = []
    echt = lz[lz.schritt.isin(["llm_gesetze", "llm_gesetze_gesetzentwurf"])].sekunden
    if len(echt):   # Summe echter Läufe wie llm.schritt_gesetze (Fortsetzung hängt eine Zeile an)
        q = f"{rel(pfad)}:sekunden (Summe schritt=llm_gesetze/llm_gesetze_gesetzentwurf, Läufe: {len(echt)})"
        z += [k("laufzeitLlmGesetze", float(echt.sum()), "Sekunden", q),
              k("laufzeitLlmGesetzeMinuten", float(echt.sum()) / 60, "Minuten", q + " / 60")]
    for schritt, s in (("llm_zusammenfassung", "laufzeitLlmZusammenfassung"), ("llm_vergleich", "laufzeitLlmVergleich"),
                       ("llm_gesetze_cache", "laufzeitLlmCache")):
        w = lz[lz.schritt == schritt].sekunden
        if len(w):      # Zeilen kommen mehrfach vor (Wiederholungen hängen an), gilt die letzte
            z.append(k(s, float(w.iloc[-1]), "Sekunden", f"{rel(pfad)}:sekunden (schritt={schritt}, letzte Zeile)"))
    lpfad = PROC / "laufzeiten.csv"
    e = pd.read_csv(lpfad).query("schritt == 'extract' and status in ('OK', 'TEIL')").dropna(subset=["sekunden"])
    if len(e):
        sek, n = float(e.sekunden.iloc[0]), int(e.sitzungen.iloc[0])
        z += [k("laufzeitExtractGesamt", sek, "Sekunden", f"{rel(lpfad)}:sekunden (schritt=extract, = laufzeitExtract)"),
              k("laufzeitExtractJeSitzung", sek / n, "Sekunden/Sitzung", f"{rel(lpfad)}:sekunden / sitzungen (extract)")]
    return z + [k(s, w, e, f"{Q_VOLLAUF}: {feld}") for s, w, e, feld in VOLLAUF_150]


def validierung() -> list[dict]:
    pfad = PROC / "validierung.csv"
    v = pd.read_csv(pfad)
    v["xml_unvollstaendig"] = v.xml_unvollstaendig.astype(bool)
    q = lambda s: f"{rel(pfad)}:{s}"  # noqa: E731
    z = []
    for praefix, menge, text in (("schwach", v, "alle Sitzungen"), ("schwachVoll", v[~v.xml_unvollstaendig],
                                                                    "ohne xml_unvollstaendig")):
        for i, r in enumerate(menge.sort_values(["reden_f1", "sitzung"]).head(3).itertuples(index=False)):
            p, rang = f"{praefix}{WORT[i]}", f"Rang {i + 1} nach reden_f1 aufsteigend, {text}"
            z += [k(f"{p}Sitzung", int(r.sitzung), "Sitzung", q(f"sitzung ({rang})")),
                  k(f"{p}FEins", r.reden_f1, "F1", q(f"reden_f1 ({rang})")),
                  k(f"{p}Precision", r.reden_precision, "Quote", q(f"reden_precision ({rang})")),
                  k(f"{p}Recall", r.reden_recall, "Quote", q(f"reden_recall ({rang})")),
                  k(f"{p}RedenParser", int(r.reden_parser), "Reden", q(f"reden_parser ({rang})")),
                  k(f"{p}RedenGold", int(r.reden_gold), "Reden", q(f"reden_gold ({rang})")),
                  k(f"{p}XmlFehlendeTops", int(r.xml_fehlende_tops), "TOPs", q(f"xml_fehlende_tops ({rang})"))]
    for s, spalte, e in (("beitraegeFEins", "beitraege_f1", "F1"), ("parteiQuote", "partei_quote", "Quote"),
                         ("befragungFEins", "befragung_f1", "F1"), ("drucksachenRecall", "drs_recall", "Quote"),
                         ("redenPrecision", "reden_precision", "Quote"), ("redenRecall", "reden_recall", "Quote")):
        z += [k(f"{s}Median", v[spalte].median(), e, q(f"{spalte} (Median)")),
              k(f"{s}Min", v[spalte].min(), e, q(f"{spalte} (Minimum)"))]
    anl = v.recall_5gramm_anlagen.dropna()
    sz = v[v.sitzung == 160]
    z += [k("redenFEinsMittel", v.reden_f1.mean(), "F1", q("reden_f1 (Mittel)")),
          k("befragungFEinsMikro", mikro(v, "befragung")[2], "F1", q("befragung_tp/_parser/_gold (mikro)")),
          k("redenInklAnlagenFEinsMikro", mikro(v, "reden_inkl_anlagen")[2], "F1",
            q("reden_inkl_anlagen_tp/_parser/_gold (mikro)")),
          k("anlagenSitzungen", len(anl), "Sitzungen", q("recall_5gramm_anlagen (nicht leer)")),
          k("anlagenFuenfGrammRecallMin", anl.min(), "Quote", q("recall_5gramm_anlagen (Minimum)")),
          k("anlagenTokenRecallMin", v.recall_token_anlagen.min(), "Quote", q("recall_token_anlagen (Minimum)")),
          k("beginnAbweichungen", int((v.beginn_diff_min.fillna(0) != 0).sum()), "Sitzungen", q("beginn_diff_min != 0")),
          k("schlussAbweichungen", int((v.schluss_diff_min.fillna(0) != 0).sum()), "Sitzungen",
            q("schluss_diff_min != 0")),
          k("zeitAbweichungMaxMinuten", float(pd.concat([v.beginn_diff_min, v.schluss_diff_min]).abs().max()),
            "Minuten", q("beginn_diff_min/schluss_diff_min (Betrag, Maximum)")),
          k("xmlFehlendeReden", int(v.xml_fehlende_reden.sum()), "Reden", q("xml_fehlende_reden (Summe)")),
          k("xmlAnlagenReden", int(v.reden_anlagen.sum()), "Reden", q("reden_anlagen (Summe)")),
          k("xmlRedenGesamt", int(v.reden_xml.sum() + v.reden_anlagen.sum()), "Reden",
            q("reden_xml + reden_anlagen (Summe)")),
          k("xmlAnlagenAnteil", 100 * v.reden_anlagen.sum() / (v.reden_xml.sum() + v.reden_anlagen.sum()), "Prozent",
            q("reden_anlagen / (reden_xml + reden_anlagen)"))]
    if len(sz):
        z += [k("sitzungHundertsechzigRedenXml", int(sz.reden_inkl_anlagen_gold.iloc[0]), "Reden",
                q("reden_inkl_anlagen_gold (sitzung=160)")),
              k("sitzungHundertsechzigRedenParser", int(sz.reden_inkl_anlagen_parser.iloc[0]), "Reden",
                q("reden_inkl_anlagen_parser (sitzung=160)"))]
    return z


def fehler() -> list[dict]:
    # Trennung der Fehlertypenanalyse: Abweichung bei vollständigem XML = Parserfehler, bei unvollständigem XML =
    # Referenzfehler; partei_abweichung (falsche Redner-ID im XML) und zeit_abweichung (Uhrzeit PDF gegen XML) sind
    # immer Referenzfehler, daher nur die Summe
    pfad, vpfad = PROC / "fehler.csv", PROC / "validierung.csv"
    f = pd.read_csv(pfad).merge(pd.read_csv(vpfad, usecols=["sitzung", "xml_unvollstaendig"]), on="sitzung", how="left")
    unv = f.xml_unvollstaendig.fillna(False).astype(bool)
    q = f"{rel(pfad)}:art × {rel(vpfad)}:xml_unvollstaendig"
    z = [k("fehlerZeilen", len(f), "Zeilen", f"{rel(pfad)}:Zeilen")]
    for art in sorted(f.art.unique()):
        n, m = "fehler" + camel(art), f.art == art
        z.append(k(n, int(m.sum()), "Fälle", f"{rel(pfad)}:art={art}"))
        if art not in ("partei_abweichung", "zeit_abweichung"):
            z += [k(n + "Parser", int((m & ~unv).sum()), "Fälle", f"{q} (art={art}, XML vollständig)"),
                  k(n + "Referenz", int((m & unv).sum()), "Fälle", f"{q} (art={art}, XML unvollständig)")]
    return z


def abstimmung_v1() -> list[dict]:
    pfad = REF / "abstimmung_urteile_v1.csv"
    a = abstimmung_precision(pfad)
    if not a:
        return []
    richtig, falsch = round(a["precision"] * a["n_true"]), round(a["fehlrate_false"] * a["n_false"])
    q = lambda s: f"{rel(pfad)}:{s} (Stichprobe v1 auf früherem Parserstand; Clopper-Pearson)"  # noqa: E731
    return [k("abstimmungPrecisionVEins", a["precision"], "Anteil", q("urteil bei abgestimmt=True"),
              clopper_pearson(richtig, a["n_true"])),
            k("abstimmungRichtigVEins", richtig, "Fälle", q("urteil=richtig bei abgestimmt=True")),
            k("abstimmungFaelleVEins", a["n_true"], "Fälle", q("abgestimmt=True")),
            k("abstimmungFehlrateFalseVEins", a["fehlrate_false"], "Anteil", q("urteil=falsch bei abgestimmt=False"),
              clopper_pearson(falsch, a["n_false"]))]


def abstimmung_false() -> list[dict]:
    # Stichprobe v2: Stand vor der Korrektur der Abstimmungserkennung (22.10.2026); an ihren 30 Fällen wurden die Regeln
    # entwickelt, die geltende Prüfung ist Stichprobe v3 (make_zahlen.py, abstimmungPrecision usw.)
    pfad = REF / "abstimmung_urteile.csv"
    a = abstimmung_precision(pfad)
    if not a:
        return []
    richtig, falsch = round(a["precision"] * a["n_true"]), round(a["fehlrate_false"] * a["n_false"])
    q = lambda s: f"{rel(pfad)}:{s} (Stichprobe v2, Parser vor der Korrektur; Clopper-Pearson)"  # noqa: E731
    return [k("abstimmungPrecisionVZwei", a["precision"], "Anteil", q("urteil bei abgestimmt=True"),
              clopper_pearson(richtig, a["n_true"])),
            k("abstimmungRichtigVZwei", richtig, "Fälle", q("urteil=richtig bei abgestimmt=True")),
            k("abstimmungFaelleVZwei", a["n_true"], "Fälle", q("abgestimmt=True")),
            k("abstimmungFaelleFalseVZwei", a["n_false"], "Fälle", q("abgestimmt=False")),
            k("abstimmungFalschFalseVZwei", falsch, "Fälle", q("urteil=falsch bei abgestimmt=False"))]


# Fenstercheck an 1000 Reden: Fraktionsmediane mit 3 Stellen und die Wechsel von label_rede zwischen 200 und 150 Wörtern
# gibt nur sentiment_fenstercheck.py auf stdout aus (kennzahlen.csv hält die Gesamtwerte); Werte aus dieser Ausgabe.
Q_FENSTERCHECK = "Messung sentiment_fenstercheck.py an 1000 Reden"
FENSTERCHECK_MEDIANE = {  # Fraktion → (Median netto bei 200 Wörtern, bei 150 Wörtern), Reihenfolge der Ausgabe
    "AfD": (-0.486, -0.523), "BSW": (-0.515, -0.604), "BÜNDNIS 90/DIE GRÜNEN": (-0.254, -0.273),
    "CDU/CSU": (-0.294, -0.331), "Die Linke": (-0.471, -0.524), "FDP": (-0.141, -0.165), "SPD": (-0.119, -0.181),
    "fraktionslos": (-0.743, -0.646)}
FENSTERCHECK_WECHSEL = [  # (Schlüssel, Reden, label_rede bei 200 → bei 150 Wörtern)
    ("fensterCheckWechselNegativNeutral", 46, "200 negativ, 150 neutral"),
    ("fensterCheckWechselNeutralNegativ", 92, "200 neutral, 150 negativ"),
]


def fenstercheck() -> list[dict]:
    q, med, z = Q_FENSTERCHECK, {}, []
    for name, werte in FENSTERCHECK_MEDIANE.items():
        f = FRAKTION[name]
        med[f] = werte
        z += [k(f"fensterCheckMedian{f}Zweihundert", werte[0], "netto", f"{q}: Median netto_200 ({name})"),
              k(f"fensterCheckMedian{f}Hundertfuenfzig", werte[1], "netto", f"{q}: Median netto_150 ({name})")]
    for i, s in ((0, "Zweihundert"), (1, "Hundertfuenfzig")):
        z.append(k(f"fensterCheckAbstandFdpUnion{s}", med["FDP"][i] - med["Union"][i], "netto",
                   f"{q}: netto FDP minus CDU/CSU ({s.lower()} Wörter)"))
    diff = [b - a for f, (a, b) in med.items() if f != "Fraktionslos"]   # fraktionslos: nur 23 Reden
    z += [k("fensterCheckFraktionDiffMin", min(diff, key=abs), "netto",
            f"{q}: netto_150 minus netto_200, kleinster Betrag ohne fraktionslos"),
          k("fensterCheckFraktionDiffMax", max(diff, key=abs), "netto",
            f"{q}: netto_150 minus netto_200, größter Betrag ohne fraktionslos")]
    return z + [k(s, n, "Reden", f"{q}: label_rede {wechsel}") for s, n, wechsel in FENSTERCHECK_WECHSEL]


# Messwerte, die keine ausgelieferte Datei enthält: Erstlauf der Extraktion, Zwischenstände des Parsers vor einer
# Erweiterung, Sentiment-Erstfassung mit 200-Wort-Fenstern, Laufzeitmessungen und vorab festgelegte Schwellen. Ein
# Neulauf der Pipeline stellt diese Stände nicht wieder her, daher stehen die Werte fest im Skript.
MESSWERTE = [  # (Schlüssel, Wert, Einheit, Herkunft = Spalte quelle)
    ("erstlaufSitzungenTxt", 53, "Sitzungen",
     "Messung Erstlauf Extraktion: Sitzungen mit txt, Lauf brach ohne Meldung ab"),
    ("erstlaufSitzungenAbgebrochen", 14, "Sitzungen",
     "Messung Erstlauf Extraktion: fehlende Sitzungen 198 und 200–212, einzeln nachextrahiert"),
    ("koepfeGesamt", 12583, "Köpfe", "Messung Stammdaten-Abgleich: Rednerköpfe aller 67 Sitzungen, 0 ohne Treffer"),
    ("koepfeFraktionslosVorgriff", 81, "Köpfe",
     "Messung Stammdaten-Abgleich: Köpfe „(fraktionslos)“ in 146–149, der späteren Gruppe Die Linke/BSW zugeordnet"),
    ("sitzungHundertsechzigKoepfe", 277, "Köpfe",
     "Zählung Sitzung 160: Rednerköpfe, die die Rede-Regel zu 193 Reden zusammenfasst"),
    ("verklebtVorher", 284, "Wörter",
     "Messung Extraktion: Ergänzungswörter wie „Außenund“, bevor der Strich vor „und/oder“ erhalten blieb"),
    ("verklebtNachher", 1, "Wörter", "Messung Extraktion: verklebte Ergänzungswörter danach (eine URL)"),
    ("abstimmungBeschlussartNachRegeln", 73, "Fälle",
     "Messung 22.10.2026: Beschlussart richtig in Stichprobe v3 nach vier ergänzten Regeln (Sammelüberweisung, "
     "Überweisungsvorschlag, „auch dieser“, namentliche Schlussabstimmung), an denselben 75 Fällen; vorher 61"),
    ("titelGesetzQuoteVorher", 0.926, "Quote",
     "Messung Validierung 22.10.2026: Titel gleich XML-Aufrufblock bei Gesetzentwürfen (261/282), bevor parse.py den "
     "Ergänzungsstrich vor „und“ im Titel erhielt"),
    ("titelErgaenzungsstrich", 55, "Titel",
     "Messung 22.10.2026: Titel, die parse.py vor der Korrektur mit „Stromund“ statt „Strom- und“ speicherte (alle Gegenstände)"),
    ("titelErgaenzungsstrichGesetz", 17, "Titel",
     "Messung 22.10.2026: davon Titel von Gesetzentwürfen"),
    ("kurztitelGleichVorher", 58, "Kurztitel",
     "Messung Validierung 22.10.2026: Kurztitel gleich XML vor der Erweiterung auf „-Gesetz“/„Gesetz zur …“ (von 70)"),
    ("versalTrennungen", 7875, "Fälle",
     "Messung Extraktion: verbundene Versal-Trennungen GRÜ-/NEN, BÜND-/NIS, BÜNDNIS-/SES"),
    ("kopfFortsetzungen", 1947, "Köpfe",
     "Messung Extraktion: Kopf-Fortsetzungen in Times-Schrift an den Rednerkopf gehängt"),
    ("grueneKoepfeWiederhergestellt", 260, "Köpfe",
     "Messung Extraktion: umbrochene Grünen-Köpfe, die ohne Kopf-Fortsetzung fehlten"),
    ("goldRedenReihenfolgeVorher", 104, "Reden",
     "Messung Extraktion 20211: Gold-Reden in Reihenfolge zugeordnet, vor der Kopf-Fortsetzung (von 107)"),
    ("goldRedenReihenfolgeNachher", 107, "Reden",
     "Messung Extraktion 20211: Gold-Reden in Reihenfolge zugeordnet, mit Kopf-Fortsetzung (von 107)"),
    ("kurzinterventionenErstparser", 94, "Beiträge",
     "Messung erster Parser: Kurzinterventionen samt Erwiderungen, bevor die Erwiderung ein eigener Redetyp wurde"),
    ("wortzaehlungAbweichung", 0.05, "Prozent MAD",
     "Messung Wortzählung: mittlere absolute Abweichung gegen den XML-Text, 20 Plenarreden (Seed 82)"),
    ("wortzaehlungStichprobe", 20, "Reden", "Messung Wortzählung: Plenarreden der Stichprobe (Seed 82)"),
    ("drucksachenRecallVorher", 0.704, "Quote",
     "Messung Validierung: Drucksachen-Recall gegen den XML-Aufrufblock (mikro), vor dem Tagesordnungs-Nachschlag"),
    ("bertTokMaxZweihundert", 581, "Tokens",
     "Messung Sentiment-Erstfassung, 24 881 Fenster zu 200 Wörtern: längstes Fenster"),
    ("bertFensterAbgeschnittenZweihundert", 2, "Fenster",
     "Messung Sentiment-Erstfassung, 200 Wörter: Fenster über 512 Tokens (abgeschnitten)"),
    ("redenUnterFuenfzigWoerter", 912, "Reden", "Messung Sentiment-Lauf: Reden unter 50 Wörtern ohne Sentiment"),
    ("goldPool", 1500, "Fenster", "Gold-Stichprobe: Pool aus plenarrede-Fenstern (Seed 82), wie sentiment.GOLD_POOL"),
    ("bertFensterZweihundert", 24881, "Fenster", "Messung Sentiment-Erstfassung, Vollauf mit 200 Wörtern: Fenster"),
    ("bertRedenZweihundert", 10023, "Reden", "Messung Sentiment-Erstfassung, Vollauf mit 200 Wörtern: Reden"),
    ("bertVollaufSekundenZweihundert", 481, "Sekunden",
     "Messung Sentiment-Erstfassung, Vollauf mit 200 Wörtern ohne Cache: Laufzeit"),
    ("bertDurchsatzZweihundert", 48.6, "Fenster/Sekunde",
     "Messung Sentiment-Erstfassung, Vollauf mit 200 Wörtern: Durchsatz nach 2 min"),
    ("bertMindestDurchsatz", 15, "Fenster/Sekunde",
     "vorab festgelegte Zeitgrenze des BERT-Vollaufs: Abbruch unter 15 Fenster/s oder nach 60 min"),
    ("bertBatch", 16, "Fenster", "Messung Batchgröße auf M1 Pro: 8/16/32/64 → 43/45/44/41 Fenster/s"),
    ("qwenRaterSekunden", 121, "Sekunden", "Messung tools/sentiment_qwen_rater.py: 60 Gold-Fenster der Fassung v1"),
    ("zuProtokollXml", 221, "Reden", "Zählung XML: zu Protokoll gegebene Reden (Parser 365)"),
    ("zuProtokollTopFehltImXml", 149, "Reden",
     "Zählung: zu Protokoll gegebene Reden des Parsers, deren TOP im XML fehlt (von 365)"),
    ("drucksachenNachschlag", 198, "Gegenstände",
     "Zählung Parser: Gegenstände mit Drucksachen aus dem Inhaltsverzeichnis"),
    ("erstParserRedetypFehler", 53, "Beiträge",
     "Messung erster Parser: Kurzinterventionen und Erwiderungen als plenarrede"),
    ("kurzinterventionenErstParser", 60, "Beiträge",
     "Messung erster Parser: Kurzinterventionen ohne Erkennung der Ankündigungsformen (danach 70)"),
    ("erwiderungenErstParser", 34, "Beiträge",
     "Messung erster Parser: Erwiderungen ohne Erkennung der Erwiderungsformeln (danach 44)"),
    ("downloadWiederholungSekunden", 0.5, "Sekunden",
     "Messung run_all.py: download als Wiederholungslauf, alle Dateien vorhanden"),
    ("llmOrdinalEntwurfGegenstaende", 334, "Gegenstände",
     "LLM-Abgleich vor der Ordinaltitel-Erkennung: Gegenstände mit Entwurfstitel"),
    ("llmOrdinalFalschTypisiert", 61, "Gegenstände",
     "LLM-Abgleich vor der Ordinaltitel-Erkennung: Entwurfstitel als sonstiges/bericht typisiert"),
    ("llmOrdinalFehlendeGesetze", 38, "Drucksachen",
     "LLM-Abgleich vor der Ordinaltitel-Erkennung: Drucksachen, die in gesetze_einheit fehlten"),
    ("llmKurztitelErfunden", 57, "Blöcke", "LLM-Abgleich: Gesetzesnamen, die das LLM ohne Klammer-Kurztitel erfindet"),
    ("llmKurztitelRegexVerpasst", 11, "Blöcke",
     "LLM-Abgleich: Kurztitel, die der Regex verpasst (großes „-Gesetz“, nur Abkürzung)"),
    ("typFixEntwurfFalschVorher", 61, "Gegenstände",
     "Ordinaltitel-Erkennung: Entwurfstitel als sonstiges/bericht, vorher"),
    ("typFixEntwurfFalschNachher", 0, "Gegenstände",
     "Ordinaltitel-Erkennung: Entwurfstitel als sonstiges/bericht, nachher"),
    ("typFixGesetzeEinheitVorher", 221, "Gesetzentwürfe", "Ordinaltitel-Erkennung: Zeilen in gesetze_einheit, vorher"),
    ("typFixGesetzeEinheitNachher", 259, "Gesetzentwürfe",
     "Ordinaltitel-Erkennung: Zeilen in gesetze_einheit, nachher"),
    ("typFixTypAbweichendVorher", 56, "Blöcke",
     "Ordinaltitel-Erkennung: LLM-Blöcke mit abweichendem Typ, vorher (von 163)"),
    ("typFixTypAbweichendNachher", 23, "Blöcke",
     "Ordinaltitel-Erkennung: LLM-Blöcke mit abweichendem Typ, nachher (von 163)"),
    ("typFixBloecke", 163, "Blöcke", "Ordinaltitel-Erkennung: LLM-Blöcke im Abgleich (Cache-Stand)"),
    ("laufzeitExtractSeriell", 170.6, "Sekunden", "Messung extract auf M1 Pro: sequenziell, 67 Sitzungen"),
    ("laufzeitExtractParallel", 28.7, "Sekunden",
     "Messung extract auf M1 Pro: 8 Prozesse, 67 Sitzungen, Ausgabe bytegleich"),
    # extract_text.WORKER = os.cpu_count(), keine feste Zahl im Code; 8 = Kerne des Messrechners (M1 Pro)
    ("extractProzesse", 8, "Prozesse", "Messung extract auf M1 Pro: ein Prozess je Kern"),
    ("fensterWoerterErstfassung", 200, "Wörter",
     "Sentiment-Erstfassung: Fenster zu 200 Wörtern (heute sentiment.FENSTER_WOERTER)"),
    ("spearmanSchwelle", 0.9, "Schwelle",
     "vorab festgelegte Schwelle des Fenstervergleichs: 200 Wörter bleiben nur bei Spearman ≥ 0,9"),
    ("labelGleichSchwelle", 0.85, "Schwelle",
     "vorab festgelegte Schwelle des Fenstervergleichs: und Label-Übereinstimmung ≥ 0,85"),
    ("kappaSchwelle", 0.3, "Schwelle",
     "vorab festgelegte Schwelle: κ(BERT–Annotator-Mehrheit) < 0,3 → H3 entfällt, Sentiment nur deskriptiv"),
]

PARAMETER = [  # (Schlüssel, Datei, Muster mit Wert in Gruppe 1, Einheit): Wert steht im Code, nicht hier
    ("fensterWoerterProduktion", "src/bundestag/sentiment.py", r"^FENSTER_WOERTER\s*=\s*(\d+)", "Wörter"),
    ("fensterMindestWoerter", "src/bundestag/sentiment.py", r"^MIN_WOERTER\s*=\s*(\d+)", "Wörter"),
    ("sentimentBatch", "src/bundestag/sentiment.py", r"^BATCH\s*=\s*(\d+)", "Fenster"),
    ("bootstrapResamples", "src/bundestag/stats.py", r"^N_BOOT\s*=\s*(\d+)", "Resamples"),
    ("seed", "src/bundestag/stats.py", r"^SEED\s*=\s*(\d+)", "Seed"),
    ("hEinsToleranz", "src/bundestag/stats.py", r"^TOLERANZ_PP\s*=\s*([\d.]+)", "Prozentpunkte Toleranz"),
    ("kwMindestReden", "src/bundestag/stats.py", r"^MIN_REDEN_KW\s*=\s*(\d+)", "Reden je MdB"),
    ("llmTemperatur", "src/bundestag/llm.py", r'^OPT_GESETZE = \{"temperature": (\d+)', "Temperatur"),
    ("llmSeed", "src/bundestag/llm.py", r"^SEED\s*=\s*(\d+)", "Seed"),
    ("llmMaxToken", "src/bundestag/llm.py", r"^BLOCK_MAX_TOKEN\s*=\s*(\d+)", "Token je Block"),
]


def parameter() -> list[dict]:
    z = []
    for s, datei, muster, einheit in PARAMETER:
        zeilen = (ROOT / datei).read_text(encoding="utf-8").splitlines()
        treffer = [(nr, m.group(1)) for nr, t in enumerate(zeilen, 1) if (m := re.search(muster, t))]
        if len(treffer) != 1:
            print(f"[text] {s}: Muster in {datei} {len(treffer)}× gefunden, übersprungen")
            continue
        nr, w = treffer[0]
        w = float(w.replace(",", "."))
        z.append(k(s, int(w) if w.is_integer() else w, einheit, f"{datei}:{nr}"))
    return z


# Zählungen der Fehlertypenanalyse (fehler.csv mit validierung.csv, Korpus-grep, Stichprobe v2 der Abstimmungen)
FEHLERTYPEN = [  # (Schlüssel, Wert, Einheit, Herkunft)
    ("sortMischProzentMin", 51.2, "Prozent",
     "Benchmark get_text(sort=True): Zeilen mit Text beider Spalten, 20146/20185/20211 = 51,2/53,8/54,4 %"),
    ("sortMischProzentMax", 54.4, "Prozent",
     "Benchmark get_text(sort=True): Zeilen mit Text beider Spalten, 20146/20185/20211 = 51,2/53,8/54,4 %"),
    ("abstimmungHinweiszeilen", 38, "Zeilen",
     "Fehlertyp Abstimmung zu Unrecht erkannt: Hinweiszeilen auf die laufende namentliche Abstimmung (Korpus)"),
    ("abstimmungEinzelaufrufe", 346, "Zeilen",
     "Fehlertyp Abstimmung zu Unrecht erkannt: Einzelaufrufe „Zusatzpunkt N x:“ ohne „Ich rufe“ (Korpus)"),
    ("abstimmungEinzelplanZeilen", 25, "Zeilen",
     "Fehlertyp Abstimmung übersehen: Zeilen „Abstimmung über (den) Einzelplan“"),
    ("abstimmungGegenFormelZeilen", 28, "Zeilen",
     "Fehlertyp Abstimmung übersehen: Zeilen „Wer stimmt gegen die (Zurück-)Überweisung“ ohne „Wer stimmt für“"),
    ("abstimmungStrengFlagZeile", 15, "Fälle",
     "Abstimmungs-Stichprobe v2, strengere Lesart: Flag und Zeile korrekt (von 20)"),
    ("abstimmungStrengSach", 12, "Fälle", "Abstimmungs-Stichprobe v2, strengere Lesart: nur Sachabstimmungen (von 20)"),
    ("fehlertypenParser", 10, "Typen", "Fehlertypenanalyse: Typen von Parserfehlern"),
    ("fehlertypenReferenz", 3, "Typen", "Fehlertypenanalyse: Typen von Referenzfehlern (XML)"),
]

GUHR = [  # (Schlüssel, Wert, Einheit, Fundstelle)
    ("guhrKorpusGesamt", 5355043, "Beispiele", "Guhr et al. 2020, S. 1629, Tab. 1"),
    ("guhrKorpusGesamtMio", 5.355043, "Mio. Beispiele", "Guhr et al. 2020, S. 1629, Tab. 1"),
    ("guhrJeKlasse", 611267, "Beispiele", "Guhr et al. 2020, S. 1629, Kap. 3.2"),
    ("guhrKorpusBalanciert", 3 * 611267, "Beispiele", "Guhr et al. 2020, S. 1629, Kap. 3.2 (3 × 611 267)"),
    ("guhrKorpusBalanciertMio", 3 * 611267 / 1e6, "Mio. Beispiele", "Guhr et al. 2020, S. 1629, Kap. 3.2"),
    ("guhrTrainingAnteil", 70, "Prozent", "Guhr et al. 2020, S. 1629, Kap. 3.2"),
    ("guhrTrainingMio", 0.7 * 3 * 611267 / 1e6, "Mio. Beispiele",
     "Guhr et al. 2020, S. 1629, Kap. 3.2 (abgeleitet 0,7 × 1 833 801)"),
    ("guhrTrainingTokens", 256, "Tokens", "Guhr et al. 2020, S. 1631, Kap. 4.2"),
    ("guhrHotelAnteil", 65.8, "Prozent", "Guhr et al. 2020, S. 1629, Tab. 1 (abgeleitet)"),
    ("guhrNeutralLeipzigAnteil", 97.7, "Prozent", "Guhr et al. 2020, S. 1629, Tab. 1 (abgeleitet)"),
    ("guhrNegativBewertungAnteil", 95.9, "Prozent", "Guhr et al. 2020, S. 1629, Tab. 1 (abgeleitet)"),
    ("guhrNegativMitFilmAnteil", 98.4, "Prozent", "Guhr et al. 2020, S. 1629, Tab. 1 (abgeleitet)"),
    ("guhrPottsTweets", 7504, "Tweets", "Guhr et al. 2020, S. 1628, Kap. 3.1"),
    ("guhrPottsAnteil", 100 * 7504 / 5355043, "Prozent des Korpus",
     "Guhr et al. 2020, S. 1628/1629 (abgeleitet 7504 / 5 355 043)"),
    ("guhrFEinsScare", 0.9409, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsGermeval", 0.7727, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsHolidaycheck", 0.9552, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsSbZehnK", 0.6930, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsFilmstarts", 0.9062, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsPotts", 0.6423, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsEmotions", 0.9652, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsLeipzig", 0.9983, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert)"),
    ("guhrFEinsGesamt", 0.9636, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (balanciert, combined)"),
    ("guhrFEinsGesamtUnbalanciert", 0.9744, "F1", "Guhr et al. 2020, S. 1631, Tab. 3 (unbalanciert, combined)"),
    ("guhrFEinsScareOhneTraining", 0.8039, "F1", "Guhr et al. 2020, S. 1631, Tab. 4"),
    ("guhrFEinsModellkarte", 0.9639, "F1",
     "Modellkarte oliverguhr/german-sentiment-bert auf Hugging Face (Model and Data), nicht im Paper"),
]


def kappa_mensch() -> list[dict]:
    """κ gegen das eigene Urteil (tools/sentiment_mensch_kappa.py); leer, solange der Bogen nicht ausgefüllt ist."""
    pfad = PROC / "kappa_mensch.csv"
    if not pfad.exists():
        return []
    kap = pd.read_csv(pfad)
    kurz = {"bert": "Bert", "mehrheit": "Mehrheit", "llm1": "LlmEins", "llm2": "LlmZwei", "qwen": "Qwen"}
    return [k(f"kappa{kurz[r.a]}Mensch", r.kappa, "kappa", f"{rel(pfad)}:kappa (a={r.a}, b=mensch; Bootstrap-KI)",
              (r.kappa_ki_unten, r.kappa_ki_oben)) for r in kap.itertuples()]


def messwerte() -> list[dict]:
    return [k(*z) for z in MESSWERTE]


def fehlertypen() -> list[dict]:
    return [k(*z) for z in FEHLERTYPEN]


def guhr() -> list[dict]:
    return [k(*z) for z in GUHR]


QUELLEN = [benchmark, konfusion, llm_kennzahlen, llm_quoten, modellvergleich, schiedsrichter, top_zehn, laufzeiten_kalt, validierung,
           fehler, abstimmung_v1, abstimmung_false, kappa_mensch, fenstercheck, messwerte, fehlertypen, guhr, parameter,
           ki_niveau, zustrom, zustrom_abstimmung, tests]


def erzeuge(ziel: Path = ZIEL) -> pd.DataFrame:
    zeilen = []
    for f in QUELLEN:
        try:
            zeilen += f()
        except Exception as e:  # noqa: BLE001  fehlende oder umgebaute Datei: übrige Quellen trotzdem schreiben
            print(f"[text] {f.__name__} übersprungen: {type(e).__name__}: {e}")
    df = pd.DataFrame(zeilen, columns=SPALTEN)
    falsch = df.schluessel[~df.schluessel.str.fullmatch(r"[A-Za-z]+")]
    if len(falsch) or not df.schluessel.is_unique:
        raise ValueError(f"Schlüssel ungültig oder doppelt: {list(falsch)[:5]} "
                         f"{list(df.schluessel[df.schluessel.duplicated()])[:5]}")
    df["wert"] = pd.Series([int(v) if pd.notna(v) and float(v).is_integer() else v for v in df.wert],
                           index=df.index, dtype=object)
    tmp = ziel.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(ziel)
    print(f"[text] {rel(ziel)}: {len(df)} Kennzahlen")
    return df


if __name__ == "__main__":
    erzeuge()
