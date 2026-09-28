"""Make ``bandits_jev`` and the test helpers here importable as ``tests.*``
without installing the recipe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
