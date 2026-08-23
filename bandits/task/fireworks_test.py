"""The provider adapter remains limited to validated task drafts."""

import json

from bandits.task import DEFAULT_MODEL, fireworks_task_enricher
from bandits.task.fireworks import _prompt_evidence
from bandits.task.enrich import TaskEnrichmentRequest


def test_fireworks_enricher_posts_schema_and_validates_response() -> None:
    seen = {}
    def transport(request, timeout):
        seen["body"] = json.loads(request.data)
        return b'{"choices":[{"message":{"content":"{\\"instruction\\":\\"Find order status.\\",\\"evidence_message_indexes\\":[0]}"}}]}'

    enrich = fireworks_task_enricher(api_key="test-key", transport=transport)
    draft = enrich(TaskEnrichmentRequest(task_id="t", trace_id="x", canonical_instruction="status?"))
    assert draft.instruction == "Find order status."
    assert seen["body"]["model"] == DEFAULT_MODEL
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["body"]["max_tokens"] == 700
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True


def test_prompt_evidence_omits_tool_payloads() -> None:
    request = TaskEnrichmentRequest(task_id="t", trace_id="x", canonical_instruction="status?", messages=(
        {"role": "tool", "content": "private database row"},
    ))
    assert _prompt_evidence(request)["messages"][0]["content"] == "[tool result omitted from task drafting]"
