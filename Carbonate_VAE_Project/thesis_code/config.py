"""Shared paths resolved from this file rather than the shell directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(ROOT)
DATA_DIR = str(ROOT / "data" / "raw" / "CT")
RESULTS_DIR = str(ROOT / "results")
CHECKPOINT_DIR = str(ROOT / "results" / "checkpoints")
DIAGRAM_DIR = ROOT / "docs" / "figures"
