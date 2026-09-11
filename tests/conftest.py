"""Make the test helpers in this directory importable as ``tests.*``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
