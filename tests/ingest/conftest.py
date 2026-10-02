"""Every ingest test also checks record accounting.

A reader that loses a span without counting it fails whichever test happened
to feed it that shape, not only the tests written about accounting.
"""

from __future__ import annotations

import pytest

from bandits.ingest import native, otlp_standard
from bandits.ingest.report import IngestReport


class _Strict(list):
    def append(self, error: str) -> None:
        raise AssertionError(error)

    def extend(self, errors) -> None:
        for error in errors:
            self.append(error)


class _CheckedReport(IngestReport):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accounting_errors = _Strict()


@pytest.fixture(autouse=True)
def _accounting_always_adds_up(monkeypatch) -> None:
    monkeypatch.setattr(otlp_standard, "IngestReport", _CheckedReport)
    monkeypatch.setattr(native, "IngestReport", _CheckedReport)
