"""Realistic user personas (PPol 2605.12894, MIMESIS 2610.09484, non-collaborative users ICLR'26).

Off-the-shelf LLM users are over-cooperative and uniform ("easy mode", 2603.11245). Each persona below
changes *how* the user behaves, never *what* they ultimately want: every persona must still deliver every
fact the task needs once the agent asks for it properly.
"""
from __future__ import annotations

import random

PERSONAS: dict[str, str] = {
    "cooperative": "Clear and polite. Answers questions directly and gives requested details when asked.",
    "vague": "Opens with an underspecified request (missing key details). Clarifies only when the agent asks a specific question.",
    "impatient": "Short, terse messages. Pushes the agent to hurry, may skip details until pressed, dislikes long explanations.",
    "withholding": "Reveals one fact at a time and only when explicitly asked for that fact. Never volunteers identifiers up front.",
    "goalpost_shift": "Starts with one request, then partway through changes a detail or adds a related requirement.",
    "informal_typos": "Casual tone, lowercase, occasional typos and abbreviations, but the meaning is recoverable.",
    "frustrated": "Annoyed from earlier problems. Pushes back on refusals or delays, may ask for a manager, but stays on task.",
    "multi_request": "Bundles two related requests into one message and expects both handled.",
}

UNIVERSAL_RULES = (
    "Never invent facts beyond the user facts you were given; if asked something not in them, say you don't know. "
    "Eventually provide every fact the task needs when the agent asks for it clearly. "
    "Do not act as the agent, do not mention tools, and do not describe the system."
)


def sample_persona(rng: random.Random, weights: dict[str, float]) -> str:
    names = [n for n in weights if n in PERSONAS]
    if not names:
        raise ValueError("persona_weights has no known personas")
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
