"""Startpunkt der Pipeline: python run_all.py --help (Ablauf und Begründung der Reihenfolge in
src/bundestag/run_all.py)."""
import sys
from pathlib import Path

# src/ auf den Pfad, damit der Aufruf auch ohne "pip install -e ." (pyproject.toml) funktioniert
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bundestag.run_all import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
