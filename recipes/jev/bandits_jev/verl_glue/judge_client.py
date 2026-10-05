"""How the reward manager reaches the frozen judge served on Modal."""

from __future__ import annotations

from typing import Any, Protocol


class JudgeClient(Protocol):
    async def score(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """steps -> {"judge": provenance, "results": [{"score", "reason", ...}, ...]}."""
        ...


class ModalJudgeClient:
    """Calls the deployed ``jev-step-judge`` app (see scripts/modal_jev_judge.py)."""

    def __init__(self, app_name: str = "jev-step-judge", class_name: str = "JevJudge") -> None:
        self.app_name, self.class_name = app_name, class_name
        self._judge = None

    async def score(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        if self._judge is None:
            import modal

            self._judge = modal.Cls.from_name(self.app_name, self.class_name)()
        return await self._judge.score.remote.aio(steps)
