"""Run directly without package installation or modifying the user's Python setup."""
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ required. Install Python 3.10 or newer.")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from jev_isaac.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
