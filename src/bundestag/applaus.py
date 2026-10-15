"""Applaus-Matrix: Beifall-Klammern je Rede der Fraktion des sprechenden Redners zuordnen."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED, SITZUNGEN, TXT_DIR
from .stammdaten import FRAKTION_KOPF, KOPF_ABG, KOPF_REG, KOPF_REG_START, lookup, sitzungsdaten

FRAKTIONEN = ["SPD", "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN", "FDP", "AfD", "Die Linke", "BSW", "fraktionslos"]
PARTEI_FRAKTION = {"SPD": "SPD", "CDU": "CDU/CSU", "CSU": "CDU/CSU", "GRÜNE": "BÜNDNIS 90/DIE GRÜNEN",
                   "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN", "FDP": "FDP"}
GRENZE = 196  # Ampel bis 06.11.2024 = Sitzung 196: vor = 146–196, nach = 197–212
# Einzelereignisse für stats (H2): ohne die Datei wiederholte stats reden_und_klammern über alle 67 Textdateien
# (15,4 s laut laufzeiten_teil.csv), obwohl main dieselben Ereignisse eine Stufe vorher schon berechnet.
EREIGNISSE = PROCESSED / "applaus_ereignisse.parquet"

PRAESIDIUM = re.compile(r"^(?:Alters)?(?:Vize)?[Pp]räsident(?:in)?\b[^:()]{0,60}:\s*$")
# 8656 Beifall-Klammern tragen Redetext hinter ")" in derselben Zeile, daher kein Zeilenende-Anker
KLAMMER = re.compile(r"^\((?:[^()]|\([^()]*\))*?\)", re.M)
PARTEI = re.compile(
    r"(?P<SPD>\bSPD\b)|(?P<CDU>CDU/CSU)|(?P<GR>BÜNDNIS(?:SES)? 90/DIE GRÜNEN)|(?P<FDP>\bFDP\b)"
    r"|(?P<AFD>\bAfD\b)|(?P<LI>\bLinken\b|\bLINKEN\b|\bDIE LINKE\b|\bDie Linke\b)|(?P<BSW>\bBSW\b)"
    r"|(?P<FL>\bfraktionslos)")
GRUPPE_KEY = {"SPD": "SPD", "CDU": "CDU/CSU", "GR": "BÜNDNIS 90/DIE GRÜNEN", "FDP": "FDP", "AFD": "AfD",
              "LI": "Die Linke", "BSW": "BSW", "FL": "fraktionslos"}
STARK = re.compile(r"\bbei(?:m| der| dem| den)\b(?! Abgeordneten)")
SCHWACH = re.compile(r"Abgeordneten|\bAbg\.")
STOP = re.compile(r"\b(?:Zuruf|Zurufe|Lachen|Heiterkeit|Widerspruch|Unruhe)\b")


def parse_beifall(klammer: str) -> dict[str, bool]:
    """Fraktion -> stark (True) / nur schwach (False) für eine Klammer."""
    out: dict[str, bool] = {}
    for seg in klammer.strip("()").split(" – "):
        i = seg.find("Beifall")
        if i < 0:
            continue
        if ":" in seg[:i]:
            continue  # "Beifall" im Wortlaut eines Zurufs, nicht als Reaktion
        stop = STOP.search(seg, i)
        # nur ab "Beifall": "Lachen bei der AfD und Beifall bei der SPD" darf die AfD nicht zählen
        seg = seg[i: stop.start()] if stop else seg[i:]
        if "ganzen Hause" in seg:
            for f in FRAKTIONEN:
                out[f] = True
            continue
        events = [(m.start(), "s") for m in STARK.finditer(seg)] + [(m.start(), "w") for m in SCHWACH.finditer(seg)]
        events += [(m.start(), GRUPPE_KEY[m.lastgroup]) for m in PARTEI.finditer(seg)]
        stark = True
        for _, e in sorted(events):
            if e == "s":
                stark = True
            elif e == "w":
                stark = False
            else:
                out[e] = out.get(e, False) or stark
    return out


def _koepfe_pos(lines: list[str]):
    """(zeile, name, fraktion_kopf, ort, typ) inkl. Präsidium; Umbruch-Join wie stammdaten.koepfe."""
    skip = 0
    for i, l in enumerate(lines):
        if skip:
            skip -= 1
            continue
        if PRAESIDIUM.match(l):
            yield i, None, None, None, "praesidium"
            continue
        if not l.endswith(":") and len(l) < 120 and KOPF_REG_START.match(l):
            for j in (i + 1, i + 2):
                if j < len(lines) and lines[j].strip():
                    nxt = lines[j]
                    joined = l[:-1] + nxt if l.endswith("-") and nxt[:1].islower() else l + " " + nxt
                    if nxt.endswith(":") and len(nxt) < 60 and not KOPF_REG.match(nxt) \
                            and not KOPF_ABG.match(nxt) and KOPF_REG.match(joined):
                        l, skip = joined, j - i
                    break
        m = KOPF_ABG.match(l)
        if m:
            yield i, m["name"], m["fr"], m["ort"], "abg"
            continue
        m = KOPF_REG.match(l)
        if m:
            yield i, m["name"], m["rolle"], None, "reg"


def reden_und_klammern(txt_dir: Path = TXT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    daten = sitzungsdaten()
    reden, ereignisse = [], []
    for p in sorted(txt_dir.glob("20*.txt")):
        if not p.stem.isdigit():
            continue  # 20NNN_anlagen.txt (Anlagen, extract_text schreibt sie getrennt) würde int(p.stem[2:]) brechen
        nr = int(p.stem[2:])
        if nr not in SITZUNGEN:
            continue  # Teillauf: data/txt hält alle 67 Sitzungen, die Ergebnisse gelten nur für den Bereich
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        starts = np.cumsum([0] + [len(x) + 1 for x in lines])
        koepfe = list(_koepfe_pos(lines))
        kz = np.array([k[0] for k in koepfe])
        rede_fr = []
        for z, name, fr, ort, typ in koepfe:  # bei typ "reg" trägt fr die Rolle
            f = None
            if typ == "abg" and fr.lower() in FRAKTION_KOPF:
                f = lookup(name, daten[nr], fr, ort)["fraktion"]
            elif typ == "reg":
                res = lookup(name, daten[nr])
                f = res["fraktion"]
                # 346 SPD- und 20 CDU/CSU-Regierungsköpfe ohne Mandat (Restliste, z. B. Pistorius) liefern keine
                # Fraktion; Bundesregierung zählt zur Partei-Fraktion, Ländervertreter wie "Staatsministerin (Bayern)" bleiben ausgeschlossen
                if f is None and "Bund" in (fr or "") and "(" not in (fr or ""):
                    f = PARTEI_FRAKTION.get(res["partei"])
            rede_fr.append(f if f in FRAKTIONEN else None)
            if rede_fr[-1]:
                reden.append({"sitzung": nr, "rede": f"{nr}_{z}", "zeile": z + 1, "kopf": lines[z][:90],
                              "redner_fraktion": rede_fr[-1]})
        for m in KLAMMER.finditer(text):
            if "Beifall" not in m.group(0):
                continue
            zeile = int(np.searchsorted(starts, m.start(), side="right")) - 1
            k = int(np.searchsorted(kz, zeile, side="left")) - 1
            if k < 0 or rede_fr[k] is None:
                continue
            for af, st in parse_beifall(m.group(0)).items():
                ereignisse.append({"sitzung": nr, "rede": f"{nr}_{kz[k]}", "zeile": zeile + 1,
                                   "redner_fraktion": rede_fr[k], "applaus_fraktion": af, "stark": st,
                                   "klammer": m.group(0)[:200]})
    r, e = pd.DataFrame(reden), pd.DataFrame(ereignisse)
    for d in (r, e):
        d["phase"] = np.where(d.sitzung <= GRENZE, "vor", "nach")
    return r, e


def heatmap(reden: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    hit = ev.drop_duplicates(["rede", "applaus_fraktion"]).assign(x=1)
    rows = []
    for (ph, rf), g in reden.groupby(["phase", "redner_fraktion"]):
        h = hit[hit.rede.isin(g.rede)]
        row = {"phase": ph, "redner_fraktion": rf}
        for af in FRAKTIONEN:
            row[af] = round(h[h.applaus_fraktion == af].rede.nunique() / len(g), 4)
        row["n_reden"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "redner_fraktion"], ascending=[False, True])


def h2(reden: pd.DataFrame, ev: pd.DataFrame, n_boot: int = 1000, seed: int = 82) -> pd.DataFrame:
    rg = {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}
    richtungen = {"SPD/Grüne→FDP-Reden": ({"FDP"}, rg), "FDP→SPD/Grüne-Reden": (rg, {"FDP"})}
    rng = np.random.default_rng(seed)
    out = []
    for lab, (redner, klatscher) in richtungen.items():
        treffer = set(ev[ev.applaus_fraktion.isin(klatscher) & ev.redner_fraktion.isin(redner)].rede)
        x = {ph: reden[reden.redner_fraktion.isin(redner) & (reden.phase == ph)].rede.isin(treffer).to_numpy(float)
             for ph in ("vor", "nach")}
        bs = {ph: np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)]) for ph, v in x.items()}
        diff = bs["nach"] - bs["vor"]
        for ph in ("vor", "nach"):
            out.append({"richtung": lab, "phase": ph, "n": len(x[ph]), "rate": x[ph].mean(),
                        "ki_lo": np.quantile(bs[ph], .025), "ki_hi": np.quantile(bs[ph], .975)})
        out.append({"richtung": lab, "phase": "diff", "n": None, "rate": x["nach"].mean() - x["vor"].mean(),
                    "ki_lo": np.quantile(diff, .025), "ki_hi": np.quantile(diff, .975)})
    return pd.DataFrame(out)


def ereignisse_je_rede(ev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Beifall-Ereignisse der Rede aus reden.parquet zuordnen (letzter Beitragskopf vor der Klammer, wie in
    reden_und_klammern); Grundlage der H2-Raten je Rede und je 1000 Wörter mit Cluster-Bootstrap in stats.py.
    Ohne ev gelten die Ereignisse, die der Schritt applaus in EREIGNISSE geschrieben hat."""
    if ev is None:
        ev = pd.read_parquet(EREIGNISSE)
    b = pd.read_parquet(PROCESSED / "beitraege.parquet", columns=["sitzung", "zeile_start", "rede_id", "anlage"])
    b = b[b.anlage.isna()].sort_values("zeile_start")
    m = pd.merge_asof(ev.sort_values("zeile"), b, left_on="zeile", right_on="zeile_start", by="sitzung")
    # Klammern im Präsidiumsbeitrag (rede_id leer) gehören zu keiner Rede
    return m.dropna(subset=["rede_id"])[["sitzung", "rede_id", "zeile", "applaus_fraktion", "stark"]]


def main() -> None:
    reden, ev = reden_und_klammern()
    ev.to_parquet(EREIGNISSE, index=False)
    agg = (ev.groupby(["sitzung", "phase", "redner_fraktion", "applaus_fraktion", "stark"])
           .size().rename("anzahl").reset_index())
    agg.to_parquet(PROCESSED / "applaus.parquet", index=False)
    heatmap(reden, ev).to_csv(PROCESSED / "applaus_heatmap.csv", index=False)
    print(h2(reden, ev).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
