"""Adapters between our canonical traces and Simia's text format (Simia's code is in simia.py)."""
from __future__ import annotations

import json

from .schema import to_sharegpt
from .simia import build_sample_text as _simia_build_sample_text
from .simia import format_reference_conversations, parse_gpt_response


def build_sample_text(trace: dict, include_system: bool = True) -> str:
    sg = to_sharegpt(trace)
    if not include_system:
        sg["system"] = ""
    return _simia_build_sample_text(sg)


def parse_simia_text(text: str) -> list[dict]:
    """Generated text -> ShareGPT turns (human / gpt / function_call / observation)."""
    return parse_gpt_response(text)["conversations"]


def reference_text(trace: dict) -> str:
    return format_reference_conversations(to_sharegpt(trace)["conversations"])


def tools_text(tools: list[dict]) -> str:
    return json.dumps(tools, ensure_ascii=False, indent=1)
