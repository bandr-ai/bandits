"""search / open / find as veRL tools that record each step for the reward manager.

veRL builds each tool once and calls ``create``/``release`` around every call,
so a trajectory's state lives on ``agent_data`` instead. veRL passes
``agent_data`` to ``execute`` after the assistant turn is in the token stream
and before the tool reply is appended, so ``len(response_mask) - 1`` is the
last policy token of the action.
"""

from __future__ import annotations

import importlib
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from bandits_jev.search_env import Searcher, SearchSession

SESSION_ATTR = "_jev_search_session"
EVENTS_KEY = "step_events"
_searchers: dict[str, Searcher] = {}


def searcher_for(factory: str) -> Searcher:
    """One retriever per process, built from a ``package.module:function`` path."""
    if factory not in _searchers:
        module, _, name = factory.partition(":")
        _searchers[factory] = getattr(importlib.import_module(module), name)()
    return _searchers[factory]


class BrowserTool(BaseTool):
    """One browser tool; the veRL tool name (search/open/find) picks ``browser.<name>``."""

    def __init__(self, config: dict, tool_schema) -> None:
        super().__init__(config, tool_schema)
        self.judge_tool = f"browser.{self.name}"
        self.searcher = searcher_for(config["searcher_factory"])
        self.top_k = int(config.get("top_k", 10))

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs["agent_data"]
        session = getattr(agent_data, SESSION_ATTR, None)
        if session is None:
            session = SearchSession(self.searcher, top_k=self.top_k)
            setattr(agent_data, SESSION_ATTR, session)
        reply = session.execute(self.judge_tool, parameters)
        agent_data.extra_fields.setdefault(EVENTS_KEY, []).append(
            {
                "position": len(agent_data.response_mask) - 1,
                "tool": self.judge_tool,
                "action": SearchSession.action_text(parameters),
                "observation": reply,
            }
        )
        return ToolResponse(text=reply), 0.0, {}
