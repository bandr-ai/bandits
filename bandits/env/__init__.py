"""bandits.env - materialize an inferred schema into a live, runnable environment.

Stage 4. Consumes a ``StateSchema`` plus a ``TaskCase`` and produces a real
SQLite store, tools reimplemented over it, an effect ledger for everything
irreversible, and an ``EnvManifest`` that states the limitations honestly.

``interface.py`` is the published session contract stage 6 replays against.
"""

from .interface import (
    BaseEnvSession,
    EnvError,
    EnvSession,
    ReadOnlyEntityError,
    SessionClosedError,
    StoreError,
    UnsupportedToolError,
)
from .ledger import EffectLedger, LedgerFrozenError
from .session import BanditsSession, build_session
from .store import Store
from .tools import ExternalRule, ReadRule, ToolRuntime, Unsupported, WriteRule

__all__ = [
    "BaseEnvSession",
    "EffectLedger",
    "EnvError",
    "EnvSession",
    "ExternalRule",
    "LedgerFrozenError",
    "ReadOnlyEntityError",
    "ReadRule",
    "SessionClosedError",
    "Store",
    "StoreError",
    "ToolRuntime",
    "BanditsSession",
    "Unsupported",
    "UnsupportedToolError",
    "WriteRule",
    "build_session",
]
