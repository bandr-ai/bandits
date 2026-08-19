"""The effect ledger.

For a lot of production agents success lives entirely in effects, not in stored
data: the refund is only half the job, the customer also has to be told. So the
ledger is half the reward signal (see ``AssertionKind.EFFECT_COUNT``).

Properties that matter:

* **ordered** -- effects keep the order they were attempted in, with the step
  index of the action that produced them,
* **append-only** -- there is no update and no delete; ``freeze()`` at session
  teardown makes even appends impossible,
* **queryable** -- by tool and by argument value, so a verifier can ask
  "how many emails went to customer 88" without string-matching a log.

Nothing here performs anything. An entry in this ledger is the record of an
effect that was deliberately *not* carried out.
"""

from __future__ import annotations

from typing import Any

from bandits.contracts import Effect, JsonObject

from .interface import EnvError


class LedgerFrozenError(EnvError):
    """Raised on append to a ledger that has been frozen. Teardown is final."""


class EffectLedger:
    """Ordered, append-only record of attempted external side effects."""

    def __init__(self) -> None:
        self._effects: list[Effect] = []
        self._frozen = False

    # -- writing -----------------------------------------------------------

    def append(self, tool: str, arguments: JsonObject, step: int) -> Effect:
        if self._frozen:
            raise LedgerFrozenError("effect ledger is frozen; the session is closed")
        effect = Effect(tool=tool, arguments=dict(arguments), step=step)
        self._effects.append(effect)
        return effect

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # -- reading -----------------------------------------------------------

    def all(self) -> tuple[Effect, ...]:
        return tuple(self._effects)

    def __iter__(self):
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._effects)

    def by_tool(self, tool: str) -> tuple[Effect, ...]:
        return tuple(e for e in self._effects if e.tool == tool)

    def count(self, tool: str | None = None) -> int:
        return len(self._effects) if tool is None else len(self.by_tool(tool))

    def by_argument(self, name: str, value: Any, *, tool: str | None = None) -> tuple[Effect, ...]:
        """Effects whose argument ``name`` equals ``value``.

        ``value`` is compared loosely across the int/str boundary only, because
        exporters stringify ids inconsistently. Pass ``tool`` to scope the query
        to one tool, e.g. ``by_argument("to_customer_id", 88, tool="send_email")``.
        """
        wanted = _candidates(value)
        out = []
        for e in self._effects:
            if tool is not None and e.tool != tool:
                continue
            if name not in e.arguments:
                continue
            if any(type(c) is type(e.arguments[name]) and c == e.arguments[name] for c in wanted):
                out.append(e)
        return tuple(out)

    def count_by_argument(self, name: str, value: Any, *, tool: str | None = None) -> int:
        return len(self.by_argument(name, value, tool=tool))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EffectLedger(n={len(self._effects)}, frozen={self._frozen})"


def _candidates(value: Any) -> list[Any]:
    out: list[Any] = [value]
    if isinstance(value, bool):
        return out
    if isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, str):
        try:
            out.append(int(value))
        except ValueError:
            pass
    return out


__all__ = ["EffectLedger", "LedgerFrozenError"]
