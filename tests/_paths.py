"""Shared project paths for directly executable regression scripts."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "examples" / "simulated_data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
