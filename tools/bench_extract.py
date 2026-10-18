"""Benchmark Lesereihenfolge: drei PyMuPDF-Varianten und zwei andere Bibliotheken gegen das XML derselben Sitzung.

Varianten
1. sort        page.get_text(sort=True)
2. columns     column_boxes() nach Lieder (2023), je Box get_text(clip=…, sort=True)
3. eigen       bundestag.extract_text (Zeilen nach Seite, Spalte, y)
4. pdfminer    pdfminer.six mit Standard-LAParams (Layoutanalyse ordnet Textboxen, boxes_flow=0,5)
5. pdfplumber  pdfplumber page.extract_text() ohne Layoutoptionen (Zeilen über die ganze Seitenbreite)

4 und 5 prüfen, ob eine andere verbreitete Bibliothek die Spalten von sich aus trennt; beide nur in diesem Werkzeug
(requirements.txt: "nur tools/bench_extract.py"). Kopf und Fuß schneiden 2, 4 und 5 wie column_boxes ab.

Alle Varianten: Text von "Beginn:" bis "(Schluss: …)" (Anlagen weg wie in Variante 3).
Silbentrennung wird bei 1 und 2 mit derselben join_hyphenated() entfernt wie bei 3,
damit der Vergleich nur die Lesereihenfolge misst, nicht die Nachbearbeitung.

Ausgabe: data/reference/benchmark.csv; mit --befund zusätzlich Kandidaten für
Zwischenrufe der rechten Spalte, die Variante 1 in die linke Spalte einschiebt.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bundestag import config, extract_text, validate  # noqa: E402

SITZUNGEN = [146, 185, 211]
LAEUFE = 2
BEGIN_RE = re.compile(r"Beginn: ?\d{1,2}[.:]\d{2} ?Uhr")
END_RE = re.compile(r"\(Schluss: ?\d{1,2}[.:]\d{2} ?Uhr\)")
# Kopf 72 pt, Fuß ab 790 pt bei 842 pt Seitenhöhe (config.HEADER_Y/FOOTER_Y)
HEADER_MARGIN = config.HEADER_Y
FOOTER_MARGIN = 842 - config.FOOTER_Y


# Variante 2 als Nachbau von multi_column.py aus J. X. Lieder (2023): "Extract Text From a Multi-Column Document
# Using PyMuPDF", Artifex Blog, damit der Benchmark den Ansatz der PyMuPDF-Entwickler für Mehrspaltensatz misst:
# https://artifex.com/blog/extract-text-from-a-multi-column-document-using-pymupdf-in-python
def column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True):
    paths = page.get_drawings()
    bboxes = []
    path_rects = []
    img_bboxes = []
    vert_bboxes = []

    clip = +page.rect
    clip.y1 -= footer_margin
    clip.y0 += header_margin

    def can_extend(temp, bb, bboxlist):
        for b in bboxlist:
            if not intersects_bboxes(temp, vert_bboxes) and (
                    b is None or b == bb or (temp & b).is_empty):
                continue
            return False
        return True

    def in_bbox(bb, bboxes):
        for i, bbox in enumerate(bboxes):
            if bb in bbox:
                return i + 1
        return 0

    def intersects_bboxes(bb, bboxes):
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        for i, bb in enumerate(bboxes):
            if in_bbox(bb, path_bboxes) or in_bbox(bb, img_bboxes):
                continue
            temp = +bb
            temp.x1 = width
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue
            if can_extend(temp, bb, bboxes):
                bboxes[i] = temp
        return [b for b in bboxes if b is not None]

    def clean_nblocks(nblocks):
        if len(nblocks) < 2:
            return nblocks
        for i in range(len(nblocks) - 1, 0, -1):
            if nblocks[i - 1] == nblocks[i]:
                del nblocks[i]
        y1 = nblocks[0].y1
        i0, i1 = 0, -1
        # Boxen mit gleicher Unterkante (±10) von links nach rechts
        for i in range(1, len(nblocks)):
            b1 = nblocks[i]
            if abs(b1.y1 - y1) > 10:
                if i1 > i0:
                    nblocks[i0:i1 + 1] = sorted(nblocks[i0:i1 + 1], key=lambda b: b.x0)
                y1 = b1.y1
                i0 = i
            i1 = i
        if i1 > i0:
            nblocks[i0:i1 + 1] = sorted(nblocks[i0:i1 + 1], key=lambda b: b.x0)
        return nblocks

    for p in paths:
        path_rects.append(p["rect"].irect)
    path_bboxes = path_rects
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    for item in page.get_images():
        img_bboxes.extend(page.get_image_rects(item[0]))

    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT, clip=clip)["blocks"]
    for b in blocks:
        bbox = pymupdf.IRect(b["bbox"])
        if no_image_text and in_bbox(bbox, img_bboxes):
            continue
        line0 = b["lines"][0]
        if line0["dir"] != (1, 0):
            vert_bboxes.append(bbox)
            continue
        srect = pymupdf.EMPTY_IRECT()
        for line in b["lines"]:
            lbbox = pymupdf.IRect(line["bbox"])
            text = "".join(s["text"].strip() for s in line["spans"])
            if len(text) > 1:
                srect |= lbbox
        bbox = +srect
        if not bbox.is_empty:
            bboxes.append(bbox)

    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))
    bboxes = extend_right(bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes)
    if not bboxes:
        return []

    nblocks = [bboxes[0]]
    bboxes = bboxes[1:]
    for i, bb in enumerate(bboxes):
        check = False
        for j in range(len(nblocks)):
            nbb = nblocks[j]
            if bb is None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue
            temp = bb | nbb
            check = can_extend(temp, nbb, nblocks)
            if check:
                break
        if not check:
            nblocks.append(bb)
            j = len(nblocks) - 1
            temp = nblocks[j]
        check = can_extend(temp, bb, bboxes)
        if not check:
            nblocks.append(bb)
        else:
            nblocks[j] = temp
        bboxes[i] = None

    return clean_nblocks(nblocks)


# sort und columns auf "Beginn:" bis "(Schluss: …)" kürzen: Variante 3 liest die Anlagen nicht, sonst fehlten sie nur dort.
def _cut_body(text: str) -> str:
    m0 = BEGIN_RE.search(text)
    start = m0.start() if m0 else 0
    m1 = END_RE.search(text, start)
    end = m1.end() if m1 else len(text)
    return text[start:end]


def _dehyphen(text: str) -> str:
    # join_hyphenated() sucht per Regex am Ende des wachsenden Strings -> quadratisch;
    # über ein ganzes Protokoll 41,6 s für 20146. Daher in Stücke teilen, die nur an
    # Zeilen ohne End-Strich/-Slash enden (dort fügt join_hyphenated nur " " ein).
    chunks, buf = [], []
    for ln in (x.strip() for x in text.splitlines()):
        if not ln:
            continue
        buf.append(ln)
        if not ln.endswith(("-", "/")):
            chunks.append(extract_text.join_hyphenated(buf))
            buf = []
    if buf:
        chunks.append(extract_text.join_hyphenated(buf))
    return " ".join(chunks)


def text_sort(pdf: Path) -> str:
    doc = pymupdf.open(pdf)
    raw = "\n".join(page.get_text(sort=True) for page in doc)
    return _dehyphen(_cut_body(raw))


def text_columns(pdf: Path) -> str:
    doc = pymupdf.open(pdf)
    parts = []
    for page in doc:
        for rect in column_boxes(page, footer_margin=FOOTER_MARGIN, header_margin=HEADER_MARGIN):
            parts.append(page.get_text(clip=rect, sort=True))
    return _dehyphen(_cut_body("\n".join(parts)))


def text_eigen(pdf: Path) -> str:
    lines, _ = extract_text.extract_lines(pdf)
    body = [ln for ln in lines if ln.section == "body"]
    return "\n\n".join(extract_text.to_paragraphs(body))


def text_pdfminer(pdf: Path) -> str:
    from pdfminer.high_level import extract_pages  # nur Benchmark
    from pdfminer.layout import LTTextContainer

    parts = []
    for page in extract_pages(str(pdf)):
        # pdfminer misst y von unten: Rumpf zwischen Fuß (FOOTER_MARGIN) und Kopf (Seitenhöhe - HEADER_MARGIN)
        oben, unten = page.height - HEADER_MARGIN, FOOTER_MARGIN
        parts += [el.get_text() for el in page if isinstance(el, LTTextContainer) and unten <= el.y0 and el.y1 <= oben]
    return _dehyphen(_cut_body("\n".join(parts)))


def text_pdfplumber(pdf: Path) -> str:
    import pdfplumber  # nur Benchmark

    parts = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            rumpf = page.crop((0, HEADER_MARGIN, page.width, page.height - FOOTER_MARGIN))
            parts.append(rumpf.extract_text() or "")
    return _dehyphen(_cut_body("\n".join(parts)))


VARIANTEN = {"sort": text_sort, "columns": text_columns, "eigen": text_eigen, "pdfminer": text_pdfminer,
             "pdfplumber": text_pdfplumber}


def bench(sitzungen) -> list[dict]:
    rows = []
    for nr in sitzungen:
        pdf = config.PDF_DIR / f"{config.WAHLPERIODE}{nr:03d}.pdf"
        reden, _ = validate.load_xml_reden(nr)
        for name, fn in VARIANTEN.items():
            zeiten = []
            for _ in range(LAEUFE):
                t0 = time.perf_counter()
                txt = fn(pdf)
                zeiten.append(time.perf_counter() - t0)
            toks = validate.tokens(txt)
            row = {
                "variante": name,
                "sitzung": f"{config.WAHLPERIODE}{nr:03d}",
                "token_recall": round(validate.token_recall(reden, toks)["recall_token"], 4),
                "fuenfgramm_recall": round(validate.ngram_recall(reden, toks)["recall_5gramm"], 4),
                "ordnungsquote": round(validate.ordnungsquote(reden, toks)["ordnungsquote"], 4),
                "laufzeit_s": round(sum(zeiten) / len(zeiten), 3),
            }
            rows.append(row)
            print(row, flush=True)
    return rows


def befund(nr: int, maxn: int = 5) -> None:
    """Belege für den Fehler von Variante 1: get_text(sort=True) setzt Zeilen gleicher Höhe beider Spalten
    zu einer Zeile zusammen (20185 S. 9: 'Möglichkeiten un-' + '(Beatrix von Storch [AfD]: Wow!)').
    Gesucht: linker Teil endet mit Trennstrich, rechter Teil beginnt mit '('."""
    doc = pymupdf.open(config.PDF_DIR / f"{config.WAHLPERIODE}{nr:03d}.pdf")
    n = 0
    for page in doc:
        for ln in page.get_text(sort=True).splitlines():
            parts = [x for x in re.split(r"\s{4,}", re.sub(r"\([A-D]\)", "", ln).strip()) if x]
            if len(parts) == 2 and parts[0].endswith("-") and parts[1].startswith("("):
                print(f"S. {page.number + 1}: {parts[0]} || {parts[1]}")
                n += 1
                if n >= maxn:
                    return


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--befund", type=int, nargs="*", help="Sitzungsnummern (z. B. 146) für Beispielsuche")
    args = ap.parse_args(argv)
    if args.befund is not None:
        for nr in args.befund or SITZUNGEN:
            print(f"== {nr}")
            befund(nr)
        return
    rows = bench(SITZUNGEN)
    out = config.DATA / "reference" / "benchmark.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
