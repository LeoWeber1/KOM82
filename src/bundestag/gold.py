"""Goldstandard aus den offiziellen XML-Protokollen (data/xml) fuer WP 20, Sitzungen 146-212.

Definitionen (abgeglichen an 20146: 177 Reden / 273 Beitraege, 20196: 148 Reden):
- Rede = jedes <rede>-Element im gesamten Dokument, inklusive der zu Protokoll gegebenen
  Reden unter <anlagen> (20146: 168 im Sitzungsverlauf + 9 in Anlagen = 177; 20196: 147 + 1).
- Beitrag = jedes p[@klasse='redner'] im gesamten Dokument (20146: 264 Sitzungsverlauf + 9 Anlagen = 273).
  Das umfasst: Redebeginn, Wiederaufnahme nach Praesidiumseinschub (20146: 216 - 177 = 39),
  Zwischenfragen/-bemerkungen anderer Redner innerhalb einer <rede> (20146: 2) und
  Frage-Antwort-Wechsel ausserhalb von <rede> in Fragestunde/Befragung (20146: 55).
  Praesidiumseinschuebe (<name>-Elemente) zaehlen NICHT; sie stehen in gold_beitraege als
  redetyp='Praesidium' mit zaehlt_als_beitrag=False (20146: 188 Zeilen; mitgezaehlt waeren es 461).
- Redetext (gold_reden.text) = nur Absaetze der Segmente des Hauptredners der <rede>;
  Praesidiums- und Zwischenfrage-Segmente sowie <kommentar> sind ausgeschlossen.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from lxml import etree

from . import config

XML_DIR = config.XML_DIR
OUT_DIR = config.GOLD_DIR

DRS_RE = re.compile(r"20/\d+")
WS_RE = re.compile(r"\s+")


def _txt(el) -> str:
    return WS_RE.sub(" ", "".join(el.itertext())).strip() if el is not None else ""


def _uhr(s: str | None) -> str | None:
    # 44 von 67 Sitzungen schreiben '9:00' statt '09:00'
    if not s:
        return None
    h, m = s.strip().split(":")[:2]
    return f"{int(h):02d}:{int(m):02d}"


def _redner(p):
    r = p.find("redner")
    n = r.find("name")
    # Fraktionsnamen enthalten NBSP ('BÜNDNIS\xa090/DIE GRÜNEN', 1149 Reden)
    g = lambda tag: WS_RE.sub(" ", n.findtext(tag) or "").strip() or None
    rolle = n.find("rolle")
    return {
        "redner_id": r.get("id"),
        "titel": g("titel"),
        "vorname": g("vorname"),
        "nachname": g("nachname"),
        "fraktion": g("fraktion"),
        "rolle": (rolle.findtext("rolle_lang") or "").strip() or None if rolle is not None else None,
    }


def _segmente(container):
    """Zerlegt die direkten Kinder in Sprechersegmente: (kopf, [p-Absaetze])."""
    segs, cur = [], None
    for c in container:
        if c.tag == "p" and c.get("klasse") == "redner":
            cur = [("redner", c), []]
            segs.append(cur)
        elif c.tag == "name":
            cur = [("name", c), []]
            segs.append(cur)
        elif c.tag == "rede":
            cur = None  # Absaetze nach einer <rede> ohne neuen Kopf gehoeren keinem Segment
        elif c.tag == "p" and cur is not None:
            cur[1].append(c)
    return segs


def _erste6(ps) -> str:
    return json.dumps([" ".join(_txt(p).split()[:6]) for p in ps], ensure_ascii=False)


def _top_titel(top) -> str:
    teile = [_txt(p) for p in top.findall("p") if (p.get("klasse") or "").startswith("T_")
             and p.get("klasse") not in ("T_Drs", "T_Ueberweisung")]
    return " ".join(teile)


def parse_sitzung(path: Path):
    root = etree.parse(str(path)).getroot()
    nr = int(root.get("sitzung-nr"))
    sitzung = 20000 + nr
    d, m, y = root.get("sitzung-datum").split(".")
    datum = f"{y}-{m}-{d}"
    start, ende = _uhr(root.get("sitzung-start-uhrzeit")), _uhr(root.get("sitzung-ende-uhrzeit"))
    zeiten = {"sitzung": sitzung, "datum": datum, "start": start, "ende": ende,
              "ende_folgetag": bool(start and ende and ende < start)}

    tops, top_titel = [], {}
    for top in root.iter("tagesordnungspunkt"):
        tid = top.get("top-id")
        titel = _top_titel(top)
        top_titel[tid] = titel
        texte = " ".join(_txt(p) for p in top.findall("p") if p.get("klasse") != "redner")
        drs = sorted(set(DRS_RE.findall(texte)), key=lambda s: int(s.split("/")[1]))
        tops.append({"sitzung": sitzung, "top_id": tid, "titel": titel,
                     "drucksachen": json.dumps(drs), "n_drucksachen": len(drs),
                     "n_reden": len(top.findall("rede"))})

    reden, beitraege = [], []

    def kontext(el):
        top = next((a for a in el.iterancestors() if a.tag == "tagesordnungspunkt"), None)
        anl = next((a for a in el.iterancestors() if a.tag == "anlagen-text"), None)
        tid = top.get("top-id") if top is not None else None
        return tid, (anl.get("anlagen-typ") if anl is not None else None)

    def beitrag_zeile(seg_nr, kopf, ps, rede_id, haupt_id, tid, anlage):
        art, el = kopf
        if art == "name":
            info = {"redner_id": None, "titel": None, "vorname": None, "nachname": None,
                    "fraktion": None, "rolle": _txt(el).rstrip(":")}
            typ = "Praesidium"
        else:
            info = _redner(el)
            if rede_id is None:
                typ = "Befragung"  # Fragestunde/Regierungsbefragung ohne <rede>, 20146: 55
            elif info["redner_id"] == haupt_id:
                typ = "Rede"
            else:
                typ = "Zwischenfrage"
        return {"sitzung": sitzung, "datum": datum, "rede_id": rede_id, "top_id": tid,
                "anlage_typ": anlage, "segment_nr": seg_nr, "redetyp": typ,
                "zaehlt_als_beitrag": art == "redner", **info,
                "n_absaetze": len(ps), "woerter": sum(len(_txt(p).split()) for p in ps),
                "erste6": _erste6(ps), "text": "\n".join(_txt(p) for p in ps)}

    for rede in root.iter("rede"):
        tid, anlage = kontext(rede)
        segs = _segmente(rede)
        erster = next(s for s in segs if s[0][0] == "redner")
        haupt = _redner(erster[0][1])
        rows = [beitrag_zeile(i, k, ps, rede.get("id"), haupt["redner_id"], tid, anlage)
                for i, (k, ps) in enumerate(segs)]
        beitraege.extend(rows)
        eigene = [p for (k, ps) in segs if k[0] == "redner"
                  and k[1].find("redner").get("id") == haupt["redner_id"] for p in ps]
        reden.append({"sitzung": sitzung, "datum": datum, "rede_id": rede.get("id"),
                      "top_id": tid, "top_titel": (top_titel.get(tid) or "")[:120] or None,
                      "anlage_typ": anlage, **haupt,
                      "n_absaetze": len(eigene),
                      "woerter": sum(len(_txt(p).split()) for p in eigene),
                      "erste6": _erste6(eigene),
                      "text": "\n".join(_txt(p) for p in eigene)})

    # Sprecherwechsel ausserhalb <rede> fehlten sonst: Frage-Antwort-Wechsel in Fragestunde/Befragung (20146: 55)
    for top in [*root.iter("tagesordnungspunkt"), *root.iter("sitzungsbeginn")]:
        tid = top.get("top-id")
        for i, (k, ps) in enumerate(_segmente(top)):
            beitraege.append(beitrag_zeile(i, k, ps, None, None, tid, None))

    return reden, beitraege, tops, zeiten


def build(xml_dir: Path = XML_DIR, out_dir: Path = OUT_DIR):
    R, B, T, Z = [], [], [], []
    for f in sorted(xml_dir.glob("*.xml")):
        if f.stem.isdigit() and int(f.stem[2:]) not in config.SITZUNGEN:
            continue  # Teillauf: data/xml hält alle 67 Sitzungen, die Referenz gilt nur für den Bereich
        r, b, t, z = parse_sitzung(f)
        R += r; B += b; T += t; Z.append(z)
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = {"gold_reden": pd.DataFrame(R), "gold_beitraege": pd.DataFrame(B),
           "gold_tops": pd.DataFrame(T), "gold_zeiten": pd.DataFrame(Z)}
    for name, df in dfs.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return dfs


if __name__ == "__main__":
    d = build()
    r, b = d["gold_reden"], d["gold_beitraege"]
    bz = b[b.zaehlt_als_beitrag]
    for s in (20146, 20196):
        print(s, "Reden", (r.sitzung == s).sum(), "Beitraege", (bz.sitzung == s).sum(),
              "inkl. Praesidium", (b.sitzung == s).sum())
    print({k: len(v) for k, v in d.items()}, "Sitzungen", r.sitzung.nunique())
