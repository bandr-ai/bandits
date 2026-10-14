"""Rubric judge: evidence integrity + the agent's expected answer, for any agent.

One call per trace. The rubric file carries all domain knowledge: generic text plus induced domain sections
(rubrics/generic_parts_v2.md via rubric_loop induce --v2), or a hand-written rubric.
This module only builds the prompt, calls the judge model and normalizes the verdict. There is no default rubric:
judging one agent's traces with another domain's rubric would pass silently. The seed is optional (synthetic traces pass it so the
judge can check drift; real traces have none)."""
import json
import re
from functools import lru_cache
from pathlib import Path

from .config import Config
from .llm import BudgetExhausted, chat_json, get_llm
from .schema import render

FULL = 10**7
LABELS = ("good", "bad", "unclear")
INTEGRITY = ("I1_consistent", "I2_possible_results", "I2_possible", "I3_decisive_kept", "I4_same_situation")  # I2_possible: v2

OUTPUT = """Return JSON only:
{"agent": "<which agent, from its system prompt>",
 "situation": "<the situation, named as the rubric names it>",
 "integrity": {"I1_consistent": "pass|fail|n/a", "I2_possible_results": "pass|fail|n/a",
               "I3_decisive_kept": "pass|fail|n/a", "I4_same_situation": "pass|fail|n/a"},
 "expected_answer": "<the answer the agent's rules give for this situation>",
 "answer_correct": true,
 "label": "good|bad|unclear",
 "minor_issues": ["..."],
 "reason": "two sentences at most, citing message numbers"}
Use "n/a" for I3 and I4 when no seed is given."""


@lru_cache(maxsize=8)
def load_rubric(path: str | Path | None = None) -> str:
    if not path:
        raise ValueError("no rubric: set rubric_path (python -m simia_plus.rubric_loop induce --v2, or a hand-written rubric)")
    text = Path(path).read_text()
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def describe_change(variation) -> str | None:
    """The declared change as the judge sees it: the variation's name and what it changes, nothing else."""
    if not variation:
        return None
    if isinstance(variation, str):
        return variation
    keep = {k: variation[k] for k in ("name", "kind", "change", "changes", "what_changes", "dependencies") if k in variation}
    return json.dumps(keep or variation, ensure_ascii=False)


NOTES = ("<code_notes note=\"differences from the seed found by code; check each against the declared change: reject the "
         "trace only if one is an unintended defect, not when the change explains it\">\n{notes}\n</code_notes>")


def notes_block(notes: list[str] | None) -> str:
    return NOTES.format(notes="\n".join(f"- {n}" for n in notes)) if notes else ""


def build_prompt(trace: dict, rubric: str, seed: dict | None = None, change: str | None = None,
                 notes: list[str] | None = None) -> tuple[str, str]:
    # A rubric with its own output section (generic_parts_v2.md) is domain-neutral: it keeps its output and gets a
    # neutral tag. Older (v1) rubrics get the v1 output appended.
    v2 = re.search(r"^## Output$", rubric, flags=re.MULTILINE) is not None
    system = rubric if v2 else rubric + "\n\n## Output\n" + OUTPUT
    tag = "agent_system_prompt"
    parts = [f"<{tag}>\n{trace.get('system', '')}\n</{tag}>"]
    if change:
        parts.append(f"<declared_change>{change}</declared_change>")
    if seed:
        parts.append(f'<seed note="the real trace this one was varied from; not an answer key">\n'
                     f"{render(seed, max_obs_chars=FULL)}\n</seed>")
    if notes:
        parts.append(notes_block(notes))
    parts.append(f'<trace note="the trace to grade; messages cited by number">\n{render(trace, max_obs_chars=FULL)}\n</trace>')
    return system, "\n\n".join(parts)


VERDICTS = ("pass", "fail", "n/a")
REQUIRED_CHECKS = ("I1_consistent", "I3_decisive_kept", "I4_same_situation")  # plus I2 under its v1 or v2 name


def normalize(out) -> dict:
    """Fail closed: a verdict missing a required part is an evaluator error (ValueError), never a pass. "good" needs
    answer_correct true and every integrity check present with pass or n/a."""
    if not isinstance(out, dict):
        raise TypeError("rubric judge returned non-object JSON")
    integ = out.get("integrity")
    if not isinstance(integ, dict):
        raise ValueError("rubric judge: no integrity object")  # noqa: TRY004 - a bad model answer, handled as an evaluator error
    missing = [k for k in REQUIRED_CHECKS if integ.get(k) not in VERDICTS]
    if not any(integ.get(k) in VERDICTS for k in ("I2_possible", "I2_possible_results")):
        missing.append("I2_possible")
    if missing:
        raise ValueError(f"rubric judge: integrity checks missing or invalid: {missing}")
    if not isinstance(out.get("answer_correct"), bool):
        raise ValueError("rubric judge: answer_correct must be true or false")  # noqa: TRY004 - a bad model answer
    failed = [k for k in INTEGRITY if integ.get(k) == "fail"]
    label = out.get("label") if out.get("label") in LABELS else None
    # The label must agree with its own parts: an integrity failure or a wrong answer is bad, whatever the label says.
    if failed or out["answer_correct"] is False:
        label = "bad"
    elif label is None:  # v2 judges write no label: it follows from integrity and answer_correct, decided here
        label = "good"
    if "critique" not in out and ("integrity_critique" in out or "answer_critique" in out):
        out["critique"] = f"{out.get('integrity_critique', '')}\n{out.get('answer_critique', '')}".strip()
    out.update(label=label, integrity_failed=failed)
    return out


def judge_rubric(cfg: Config, trace: dict, seed: dict | None = None, rubric: str | None = None,
                 change: str | None = None, attempts: int = 2, notes: list[str] | None = None) -> dict:
    system, user = build_prompt(trace, rubric or load_rubric(), seed, change, notes)
    for i in range(attempts):  # DeepSeek occasionally loops until the token limit; one retry clears it
        try:
            return normalize(chat_json(get_llm(cfg, "judge"), system, user))
        except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
            raise
        except Exception:
            if i == attempts - 1:
                raise


def keep(result: dict) -> bool:
    """Training gate: keep only traces the rubric judge labels good."""
    return result.get("label") == "good"
