"""download._fetch und download_all mit gemockter Session statt Netz: eine HTML-Wartungsseite mit HTTP 200 darf nicht
als .pdf liegen bleiben, eine schon gespeicherte wird neu geladen (Dateianfang %PDF- bzw. <?xml geprüft)."""
import requests

from bundestag import config, download

PDF = b"%PDF-1.7\n" + b"x" * 100
HTML = b"<!DOCTYPE html><html><body>Wartungsarbeiten</body></html>"


class Antwort:
    def __init__(self, status, inhalt, typ="application/pdf"):
        self.status_code, self.inhalt, self.headers = status, inhalt, {"Content-Type": typ}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, groesse):
        for i in range(0, len(self.inhalt), groesse):
            yield self.inhalt[i:i + groesse]


class Session:
    def __init__(self, *antworten):
        self.antworten, self.urls = list(antworten), []

    def get(self, url, timeout=None, stream=False):
        self.urls.append(url)
        a = self.antworten.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def _ohne_pause(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)


def test_pdf_ok(tmp_path, monkeypatch):
    _ohne_pause(monkeypatch)
    ziel = tmp_path / "20146.pdf"
    assert download._fetch(Session(Antwort(200, PDF)), "u", ziel) == (200, len(PDF))
    assert ziel.read_bytes() == PDF
    assert not (tmp_path / "20146.pdf.part").exists()


def test_html_mit_http_200_abgelehnt(tmp_path, monkeypatch):
    _ohne_pause(monkeypatch)
    ziel = tmp_path / "20146.pdf"
    try:
        download._fetch(Session(Antwort(200, HTML, "text/html")), "u", ziel)
        raise AssertionError("HTML als PDF angenommen")
    except download.FalscherInhalt as exc:
        assert "text/html" in str(exc)
    assert not ziel.exists() and not (tmp_path / "20146.pdf.part").exists()


def test_vorhandene_html_datei_wird_neu_geladen(tmp_path, monkeypatch):
    _ohne_pause(monkeypatch)
    ziel = tmp_path / "20146.pdf"
    ziel.write_bytes(HTML)
    s = Session(Antwort(200, PDF))
    assert download._fetch(s, "u", ziel) == (200, len(PDF))
    assert ziel.read_bytes() == PDF and s.urls == ["u"]


def test_gueltige_datei_wird_uebersprungen(tmp_path):
    ziel = tmp_path / "20146.pdf"
    ziel.write_bytes(PDF)
    s = Session()
    assert download._fetch(s, "u", ziel) == (200, len(PDF)) and s.urls == []


def test_netzfehler_und_fehlinhalt_im_manifest(tmp_path, monkeypatch):
    _ohne_pause(monkeypatch)
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "download_manifest.csv")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "pdf")
    monkeypatch.setattr(config, "XML_DIR", tmp_path / "xml")
    monkeypatch.setattr(config, "SITZUNGEN", range(146, 149))
    monkeypatch.setattr(download, "stammdaten_laden", lambda s: {"sitzung": "", "typ": "stammdaten", "url": "z",
                                                                  "status": 200, "bytes": 1})
    s = Session(Antwort(200, PDF), requests.ConnectionError("kein Netz"), Antwort(200, HTML, "text/html"))
    monkeypatch.setattr(download, "_session", lambda: s)
    n_fehler, n = download.download_all(with_xml=False)
    assert (n_fehler, n) == (2, 4)
    zeilen = (tmp_path / "download_manifest.csv").read_text().splitlines()[1:]
    assert [z.split(",")[3] for z in zeilen] == ["200", "0", "0", "200"]
    assert sorted(p.name for p in (tmp_path / "pdf").iterdir()) == ["20146.pdf"]
