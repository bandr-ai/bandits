"""A small client for the hosted Bandr Rank loop, so anyone can read exactly what it does.

    client = RankClient("https://app.example", api_key)
    while (ep := client.next_episode(run_id)):
        out = run_my_agent(ep.task["input"], base_url=client.gateway_url,
                           extra_headers=client.headers(ep))
        client.report(ep, score=my_check(ep.task, out), feedback=out.summary)

Bandr, not the harness, decides which model answers an episode. Calls the harness makes with the
``x-bandr-episode`` header go to the assigned candidate; calls without it (a user simulator, a
grader) pass through untouched. Each model response carries a receipt in ``x-bandr-record-id``;
a report must cite the receipts of its own episode, so collect them with ``Receipts``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

EPISODE_HEADER = "x-bandr-episode"
RECEIPT_HEADER = "x-bandr-record-id"


@dataclass(frozen=True)
class Episode:
    id: str
    run_id: str
    phase: str
    attempt: int
    task: dict[str, Any]


@dataclass
class Receipts:
    """Collects receipts from an agent's model responses; pass ``ids`` to ``report``."""

    ids: list[str] = field(default_factory=list)

    def add(self, response: httpx.Response) -> None:
        receipt = response.headers.get(RECEIPT_HEADER)
        if receipt:
            self.ids.append(receipt)


class RankError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class RankClient:
    def __init__(self, base_url: str, api_key: str, *, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.gateway_url = f"{self.base_url}/v1"
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._http = http or httpx.Client(timeout=60.0)

    def headers(self, episode: Episode) -> dict[str, str]:
        """Headers for the agent's model client: the Bandr key and the episode."""
        return {**self._auth, EPISODE_HEADER: episode.id}

    def next_episode(
        self, run_id: str, *, poll: float = 2.0, give_up_after: float = 3600.0
    ) -> Episode | None:
        """The next episode, or None when the run is done. Waits while others must finish first."""
        deadline = time.monotonic() + give_up_after
        while True:
            r = self._http.get(f"{self.base_url}/rank/runs/{run_id}/next", headers=self._auth)
            if r.status_code == 204:
                return None
            if r.status_code == 200:
                body = r.json()
                return Episode(
                    body["id"], body["run_id"], body["phase"], body["attempt"], body["task"]
                )
            if r.status_code == 202 and time.monotonic() < deadline:
                time.sleep(poll)
                continue
            raise RankError(r.status_code, _detail(r))

    def report(
        self,
        episode: Episode,
        *,
        score: float,
        references: list[str],
        feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Report an episode's score (0 to 1) with the receipts of its model calls."""
        r = self._http.post(
            f"{self.base_url}/rank/report",
            headers=self._auth,
            json={
                "episode": episode.id,
                "score": score,
                "feedback": feedback,
                "references": references,
                "metadata": {"task": episode.task["id"], **(metadata or {})},
            },
        )
        if r.status_code != 200:
            raise RankError(r.status_code, _detail(r))
        return r.json()

    def recheck(self, run_id: str) -> dict[str, Any]:
        """After fixing a check that passed the control model, replay the control episodes."""
        r = self._http.post(f"{self.base_url}/rank/runs/{run_id}/recheck", headers=self._auth)
        if r.status_code != 200:
            raise RankError(r.status_code, _detail(r))
        return r.json()


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text
