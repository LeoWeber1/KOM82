"""Datengetriebene LaTeX-Bausteine für Kapitel 2 → tab_top10.tex und zustrom_redner.tex neben zahlen.tex.

tab_top10.tex     Tabelle (c) aus data/processed/top10_gesetze.csv als tabularx (booktabs, footnotesize), ohne
                  table-Umgebung und caption; Einbindung: \\input{tab_top10} in der table-Umgebung des Autors.
zustrom_redner.tex  Je Sitzung ein Satz zum Zustrombegrenzungsgesetz (Drs. 20/12804): Beratungsstufe, Datum, Punkt,
                  Redner mit Plenarrede je Fraktion, Zahl der Plenarreden und die ersten drei Redner; aus
                  gesetz_beteiligte.parquet, reden.parquet und gesetze.parquet. Einbindung: \\input{zustrom_redner}.

Ausgabeordner wie zahlen.tex (config.ZAHLEN_TEX), ein Teillauf schreibt also nach data/teillauf/<von>-<bis>/LaTeX.
Aufruf: python tools/latex_bausteine.py   (tools/make_zahlen.py ruft erzeuge() nach zahlen.tex auf)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bundestag import config  # noqa: E402

ZUSTROM = "20/12804"
TITEL_MAX = 55
REDETYPEN = ("plenarrede", "kurzintervention", "erwiderung")
GRUPPEN = {"CDU/CSU": "Union", "SPD": "SPD", "BÜNDNIS 90/DIE GRÜNEN": "Grüne", "FDP": "FDP", "AfD": "AfD",
           "Die Linke": "Linke", "BSW": "BSW", "fraktionslos": "fraktionslos"}
PARTEI = {"BÜNDNIS 90/DIE GRÜNEN": "Grüne", "Die Linke": "Linke", "Plos": "parteilos", "parteilos": "parteilos"}
STUFE = {"erste": "1.~Beratung", "zweite und dritte": "2./3.~Beratung", "zweite": "2.~Beratung", "dritte": "3.~Beratung"}
# Fraktion, sonst Partei → Fraktion, Abbildung wie kennzahlen_text.zustrom() (Regierungsmitglieder tragen keine Fraktion;
# Plos/parteilos = fraktionslos), damit die Zählung je Fraktion zu den Makros \zustromRedner… passt
PARTEI_FRAKTION = {"SPD": "SPD", "CDU": "CDU/CSU", "CSU": "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
                   "FDP": "FDP", "AfD": "AfD", "Die Linke": "Die Linke", "BSW": "BSW", "Plos": "fraktionslos",
                   "parteilos": "fraktionslos"}
# (Fraktion, Kurzform im Satz, Makro-Kürzel) in der Reihenfolge von Tabelle 1 (tab:fraktionen)
FRAKTIONEN = [("SPD", "SPD", "SPD"), ("CDU/CSU", "CDU/CSU", "Union"), ("BÜNDNIS 90/DIE GRÜNEN", "Grüne", "Gruene"),
              ("FDP", "FDP", "FDP"), ("AfD", "AfD", "AfD"), ("Die Linke", "Die Linke", "Linke"), ("BSW", "BSW", "BSW"),
              ("fraktionslos", "fraktionslos", "Fraktionslos")]
BEISPIELE = 3   # genannte Personen je Sitzung: die ersten Redner mit Plenarrede in Protokollreihenfolge


def tex(s: str) -> str:
    # gleiche Ersetzungen wie make_zahlen.latex_text; Import dort ginge im Kreis, make_zahlen lädt dieses Modul
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", r"\textasciitilde{}").replace("^", r"\^{}")


def zahl(n: int) -> str:
    """Ganzzahl deutsch: Tausender erst ab 5 Stellen mit \\, (DIN 5008, wie make_zahlen.zahl)."""
    s = str(int(n))
    return s if len(s) < 5 else f"{int(n):,}".replace(",", "\\,")


def sitzungsliste(feld) -> str:
    # Läufe ab drei Sitzungen als Spanne: 20/12400 steht in 183, 184, 185, 186 (top10_gesetze.csv)
    if pd.isna(feld) or not str(feld).strip():
        return "–"
    nr = sorted(int(x) for x in str(feld).split(","))
    teile, start = [], 0
    for i in range(1, len(nr) + 1):
        if i == len(nr) or nr[i] != nr[i - 1] + 1:
            lauf = nr[start:i]
            teile += [f"{lauf[0]}–{lauf[-1]}"] if len(lauf) >= 3 else [str(x) for x in lauf]
            start = i
    return ", ".join(teile)


def kurztitel(r) -> str:
    # Kurztitel ohne „gesetz“ ist nur die Klammer am Titelende (20/12977: „Artikel 93 und 94“), dann gekürzter Titel
    k = "" if pd.isna(r.kurztitel) else str(r.kurztitel)
    if not re.search(r"gesetz", k, re.I):
        k = re.sub(r"^.*?Entwurfs? eines (?:Gesetzes )?(?:zur |zum |über (?:die |den |das )?)?", "", str(r.titel))
        k = k.split(" – ")[0]
        # Genitiv aus „Entwurfs eines Zweiten Haushaltsfinanzierungsgesetzes 2024“ (20/9999) in den Nominativ
        k = re.sub(r"^(?:(\w+)en )?(\S*gesetz)es\b", lambda m: (m[1] + "es " if m[1] else "") + m[2], k)
        k = k[:1].upper() + k[1:]
    k = re.sub(r"\bArtikel (?=\d)", "Art. ", k)
    if len(k) > TITEL_MAX and " " in k:   # ein Wort bleibt ganz: Lieferkettensorgfaltspflichtenaufhebungsgesetz hat 46
        k = k[:TITEL_MAX - 1].rsplit(" ", 1)[0] + "…"
    return k


def fraktionen(feld) -> str:
    # Spalte fraktionen_reden: „CDU/CSU 13; SPD 11; …“; Darstellung wie die Tabelle des Autors (alle / ohne …)
    da = {t.rsplit(" ", 1)[0] for t in str(feld).split("; ") if t.strip()}
    fehlt = [GRUPPEN[g] for g in GRUPPEN if g not in da]
    if not fehlt:
        return "alle"
    if len(fehlt) <= 3:
        return "ohne " + ", ".join(fehlt)
    return ", ".join(GRUPPEN.get(g, tex(g)) for g in GRUPPEN if g in da)


def tab_top10(proc: Path) -> str:
    t = pd.read_csv(proc / "top10_gesetze.csv", dtype={"sitzungen_erste_lesung": str, "sitzungen_zweite_dritte": str})
    zeilen = [f"% Erzeugt von tools/latex_bausteine.py aus {(proc / 'top10_gesetze.csv').relative_to(ROOT)}; "
              "nicht von Hand ändern. Sortierung wie metrics.top10_gesetze: Redner, dann Reden absteigend.",
              "% * Beispiel der Aufgabenstellung (Drs. 20/12804); Fraktionen: alle = alle acht Fraktionen und Gruppen.",
              "{\\footnotesize\\setlength{\\tabcolsep}{3pt}%",
              "\\begin{tabularx}{\\textwidth}{@{}rl>{\\raggedright\\arraybackslash}Xrrr"
              ">{\\raggedright\\arraybackslash}p{1.2cm}>{\\raggedright\\arraybackslash}p{1.2cm}"
              ">{\\raggedright\\arraybackslash}p{1.85cm}l@{}}",
              "\\toprule",
              # zweizeiliger Kopf: „Lesung (Sitzung)“ in einer p-Spalte von 1,2 cm brach in vier Zeilen um
              "& & & & & & \\multicolumn{2}{c}{Lesung in Sitzung} & & \\\\",
              "\\cmidrule(lr){7-8}",
              "Rang & Drucksache & Kurztitel bzw. gekürzter Titel & Redner & Reden & Vorl. & 1. & 2./3. & "
              "Fraktionen & Beschluss \\\\",
              "\\midrule"]
    for r in t.sort_values("rang").itertuples(index=False):
        drs = tex(r.drucksache) + ("\\,*" if r.drucksache == ZUSTROM else "")
        # \hspace{0pt}: TeX trennt das erste Wort einer Zelle nicht, Lieferkettensorgfaltspflichtenaufhebungsgesetz lief 86 pt über
        # Vorl.: größte Zahl gemeinsam beratener Vorlagen im TOP (parse.beteiligte); alle Redner dieser Debatte zählen mit
        zellen = [str(int(r.rang)), drs, "\\hspace{0pt}" + tex(kurztitel(r)), zahl(r.n_redner), zahl(r.n_reden),
                  zahl(r.gegenstaende_im_top_max),
                  sitzungsliste(r.sitzungen_erste_lesung), sitzungsliste(r.sitzungen_zweite_dritte),
                  fraktionen(r.fraktionen_reden), tex(r.beschlussart_final)]
        zeilen.append(" & ".join(zellen) + " \\\\")
    zeilen += ["\\bottomrule", "\\end{tabularx}}"]
    return "\n".join(zeilen) + "\n"


def partei_text(r) -> str:
    p = PARTEI.get(r.partei, r.partei) if pd.notna(r.partei) else "ohne Partei"
    teile = [tex(p)]
    if r.fraktion == "fraktionslos" and p != "parteilos":   # parteilose MdB sind fraktionslos (Spaniel, Seitz, Cotar 211)
        teile.append("fraktionslos")
    if pd.notna(r.rolle) and r.rolle != "MdB":   # nur „Bundesministerin“: voller Titel kostete 2 Zeilen, PDF 13 → 14 Seiten
        teile.append(tex(str(r.rolle).split()[0]))
    return ", ".join(teile)


def verteilung(je: pd.Series) -> str:
    """„SPD~5, CDU/CSU~5, …“ in der Reihenfolge von Tabelle 1; Gruppen ohne Redner entfallen, unbekannte stehen hinten."""
    teile = [f"{kurz}~{int(je[fr])}" for fr, kurz, _ in FRAKTIONEN if je.get(fr, 0)]
    teile += [f"{tex(fr)}~{int(n)}" for fr, n in je.items() if fr not in {f[0] for f in FRAKTIONEN}]
    return ", ".join(teile)


def beispiel(p: pd.DataFrame) -> str:
    """Die ersten BEISPIELE Redner mit Plenarrede: „Zuerst sprechen A (SPD, Bundesministerin), B (CDU) und C (Grüne)“."""
    namen = [f"{tex(x.name).replace(' ', '~')} ({partei_text(x)})"
             for x in p.drop_duplicates("name").head(BEISPIELE).itertuples(index=False)]
    if len(namen) == 1:
        return f"Es spricht {namen[0]}"
    return "Zuerst sprechen " + ", ".join(namen[:-1]) + " und " + namen[-1]


def zustrom_redner(proc: Path) -> tuple[str, dict]:
    """Ein Satz je Sitzung; zaehlung[sitzung] = (Redner mit Plenarrede, Personen nur mit Kurzintervention/Erwiderung,
    Plenarreden, {Fraktion: Redner}), zaehlung["alle"] dasselbe über alle Sitzungen (Personen je einmal)."""
    gb = pd.read_parquet(proc / "gesetz_beteiligte.parquet", columns=["rede_id", "drucksache", "redetyp"])
    gb = gb[(gb.drucksache == ZUSTROM) & gb.redetyp.isin(REDETYPEN)]
    r = pd.read_parquet(proc / "reden.parquet",
                        columns=["rede_id", "sitzung", "datum", "name", "partei", "fraktion", "rolle", "zeile_start"])
    d = gb.merge(r, on="rede_id", how="left").sort_values(["sitzung", "zeile_start"])
    d["fr"] = d.fraktion.fillna(d.partei.map(PARTEI_FRAKTION)).fillna("ohne Fraktion")
    g = pd.read_parquet(proc / "gesetze.parquet", columns=["drucksache", "sitzung", "punkt", "beratungsstufe"])
    g = g[g.drucksache == ZUSTROM].drop_duplicates("sitzung").set_index("sitzung")
    saetze, zaehlung = [], {}
    for sitzung, s in d.groupby("sitzung", sort=True):
        p = s[s.redetyp == "plenarrede"]
        je = p.groupby("fr").name.nunique()
        kopf = [pd.Timestamp(s.datum.iloc[0]).strftime("%d.%m.%Y")]
        if sitzung in g.index:
            kopf = [STUFE.get(g.at[sitzung, "beratungsstufe"], tex(g.at[sitzung, "beratungsstufe"]))] + kopf + \
                   [tex(g.at[sitzung, "punkt"]).replace("ZP ", "ZP~")]
        saetze.append(f"Sitzung~{sitzung} ({', '.join(kopf)}): {p.name.nunique()}~Redner ({verteilung(je)}) mit "
                      f"{len(p)}~Plenarreden. {beispiel(p)}.")
        zaehlung[int(sitzung)] = (int(p.name.nunique()), len(set(s.name) - set(p.name)), len(p),
                                  {fr: int(n) for fr, n in je.items()})
    p = d[d.redetyp == "plenarrede"]
    zaehlung["alle"] = (int(p.name.nunique()), len(set(d.name) - set(p.name)), len(p),
                        {fr: int(n) for fr, n in p.groupby("fr").name.nunique().items()})
    kopf = [f"% Erzeugt von tools/latex_bausteine.py aus {(proc / 'gesetz_beteiligte.parquet').relative_to(ROOT)} × "
            "reden.parquet × gesetze.parquet; nicht von Hand ändern.",
            f"% Drs. {ZUSTROM}: Redner und Plenarreden je Sitzung (redetyp plenarrede), Verteilung Fraktion sonst Partei wie "
            "kennzahlen_text.zustrom() (\\zustromRedner…), Beispiele = erste Redner nach zeile_start mit Partei zum "
            "Sitzungsdatum, Rolle nur außer MdB.",
            f"% Personen mit Plenarrede: {zaehlung['alle'][0]} über alle Sitzungen, {zaehlung['alle'][2]} Plenarreden; nur "
            f"Kurzintervention/Erwiderung: {zaehlung['alle'][1]} (nicht gezählt)."]
    return "\n".join(kopf) + "\n" + "\n\n".join(saetze) + "\n", zaehlung


def pruefe_zaehlung(zaehlung: dict, proc: Path) -> list[str]:
    """Vergleicht die Zählung des Bausteins mit den Makros aus kennzahlen.csv und kennzahlen_text.csv (zustromRed…)."""
    soll = {}
    for datei in ("kennzahlen.csv", "kennzahlen_text.csv"):
        if (proc / datei).exists():
            k = pd.read_csv(proc / datei).set_index("schluessel").wert
            soll.update({s: int(float(w)) for s, w in k.items() if str(s).startswith("zustromRed")})
    # Schlüssel → (Sitzung oder "alle", Index in zaehlung bzw. Fraktion im Verteilungs-Dict)
    paare = {"zustromRedner": ("alle", 0), "zustromRednerKurz": ("alle", 1), "zustromReden": ("alle", 2),
             "zustromRednerErsteSumme": (185, 0), "zustromRednerErsteKurz": (185, 1), "zustromRedenErste": (185, 2),
             "zustromRednerElfSumme": (211, 0), "zustromRednerElfKurz": (211, 1), "zustromRedenElf": (211, 2)}
    for fr, _, kz in FRAKTIONEN:
        paare[f"zustromRedner{kz}"], paare[f"zustromRednerElf{kz}"] = ("alle", fr), (211, fr)
    meldungen = []
    for schluessel, (teil, i) in paare.items():
        if schluessel in soll and teil in zaehlung:
            ist = zaehlung[teil][3].get(i, 0) if isinstance(i, str) else zaehlung[teil][i]
            if soll[schluessel] != ist:
                meldungen.append(f"{schluessel} {soll[schluessel]} ≠ Rednerliste {ist}")
    return meldungen


def schreibe(ziel: Path, text: str) -> None:
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(ziel)   # atomar wie zahlen.tex: main.tex kann parallel kompiliert werden


def erzeuge() -> list[Path]:
    proc, aus = config.PROCESSED, config.ZAHLEN_TEX.parent
    pfade = []
    if (proc / "top10_gesetze.csv").exists():
        schreibe(aus / "tab_top10.tex", tab_top10(proc))
        pfade.append(aus / "tab_top10.tex")
    else:
        print(f"[bausteine] {(proc / 'top10_gesetze.csv').relative_to(ROOT)} fehlt: tab_top10.tex übersprungen")
    if all((proc / f).exists() for f in ("gesetz_beteiligte.parquet", "reden.parquet", "gesetze.parquet")):
        text, zaehlung = zustrom_redner(proc)
        schreibe(aus / "zustrom_redner.tex", text)
        pfade.append(aus / "zustrom_redner.tex")
        je = ", ".join(f"Sitzung {s} {n[0]}" for s, n in zaehlung.items() if s != "alle")
        print(f"[bausteine] Zustrom-Redner (plenarrede): {je}, gesamt {zaehlung.get('alle', (0,))[0]}")
        for m in pruefe_zaehlung(zaehlung, proc):
            print(f"[bausteine]   Abweichung: {m}")
    else:
        print("[bausteine] gesetz_beteiligte/reden/gesetze.parquet fehlen: zustrom_redner.tex übersprungen")
    for p in pfade:
        print(f"[bausteine] {p.relative_to(ROOT)} geschrieben")
    return pfade


if __name__ == "__main__":
    erzeuge()
