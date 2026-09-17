"""Let the RLM propose a family's verifier from its own turns.

The verifier for a family is a set of predicates over turns — ``check(turn)``
returns True when the turn is bad, judged by the reaction that followed it.
The model is shown a sample of the family's turns with the next-state judge's
verdicts beside them, and writes predicates in a REPL where it can run them
against that sample before committing. Nothing it reports about its own
predicates is trusted: every proposal is re-executed here, in an AST sandbox,
over every turn in the family, and scored against the judge.

A check survives when it fires often enough to matter and mostly fires where
the judge said −1. That is the label-free bar: the judge is the anchor the
family-specific check has to agree with, and a check that disagrees with it is
either a discovery or a mistake — which is what the human review decides.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import signal as signal_mod
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.nextstate import ARCHETYPE_VOCABULARY, Archetype, TurnJudgeRun, TurnVerdict
from bandits.verify.turns import Turn

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
PROMPT_VERSION = 1
DEFAULT_MAX_TOKENS = 16000


class ProposedCheck(BaseModel):
    """What the model returns. Plain pydantic: DSPy decodes into it."""

    name: str
    hypothesis: str
    code: str


_INSTRUCTION = textwrap.dedent(
    """\
    You are writing a verifier for one family of agent trajectories. The verifier is a
    set of Python predicates over *turns*. A turn is one action the agent took and the
    reaction that followed it. A predicate flags a BAD turn: one where the reaction shows
    the action was wrong.

    Kind of trace: {archetype}. {vocabulary}

    `turns` is a JSON string: a list of objects with keys
      trace_id, index, task, action, next_state, reactions (list of {{kind, name, text, error}}),
      observed, errored, judge (+1 / 0 / -1 / null: what a next-state judge said about the
      turn), hint (why the judge said so).
    Parse it with json.loads and study it in the REPL. Write each predicate as

        def check(turn):
            \"\"\"<one-line hypothesis: what reaction pattern marks the action as wrong>\"\"\"
            ...
            return True   # the turn is bad
            # or False when it is fine, or None when the predicate does not apply

    Rules for the code:
    - A pure function of `turn` (a dict with the keys above, minus judge and hint —
      the predicate must not read `judge` or `hint`, they will not be there).
    - No imports, no I/O, no eval/exec/open, no attribute starting with `__`.
      Builtins available: len any all sum min max sorted set list dict str int float bool
      range enumerate zip abs round isinstance tuple map filter reversed, and re_search(pattern, text).
    - Under 25 lines. Return None rather than guessing when the turn has no next_state.
    - Read the reaction. A check that only looks at the action is describing the agent's
      style, not whether it was wrong.

    Test every predicate against `turns` before returning it: it should fire on at least
    three turns, and most of the turns it fires on should carry judge == -1. Drop anything
    that fires everywhere or nowhere. Prefer specific reaction patterns over generic ones —
    "the execution log says the file was not found after the action assumed a path" beats
    "the log mentions error".

    `library` lists checks already accepted, with how they scored; do not resubmit them,
    propose what they miss. `correction` says what was wrong with your last answer, if
    anything.

    Return `checks`: a list of {{name, hypothesis, code}} objects, 3 to 8 of them.
    """
)


def instruction_for(archetype: Archetype) -> str:
    return _INSTRUCTION.format(
        archetype=archetype.value, vocabulary=ARCHETYPE_VOCABULARY[archetype]
    )


def prompt_digest(model: str) -> str:
    payload = json.dumps(
        {"version": PROMPT_VERSION, "model": model, "instruction": _INSTRUCTION}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Proposer(Protocol):
    def __call__(self, *, turns: str, library: str, correction: str) -> Any: ...


class ProposalError(RuntimeError):
    pass


def build_proposer(
    *,
    archetype: Archetype,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_iterations: int = 20,
    max_llm_calls: int = 40,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Proposer:
    """A ``dspy.RLM`` that writes and tests predicates in its REPL."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ProposalError("proposing needs the 'audit' extra: uv sync --extra audit") from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
        # LiteLLM's own retries, before the backoff below sees anything. Three
        # was not enough alongside a judge run sharing the same rate limit.
        num_retries=8,
    )

    class _Propose(dspy.Signature):
        turns: str = dspy.InputField(
            desc="JSON list of turns with judge verdicts; parse and study it"
        )
        library: str = dspy.InputField(
            desc="checks already accepted, with their scores; possibly empty"
        )
        correction: str = dspy.InputField(
            desc="empty on a first attempt; otherwise what was wrong last time"
        )
        checks: list[ProposedCheck] = dspy.OutputField()

    _Propose.__doc__ = instruction_for(archetype)
    rlm = dspy.RLM(
        _Propose, max_iters=max_iterations, max_llm_calls=max_llm_calls, sub_lm=language_model
    )

    def propose(*, turns: str, library: str, correction: str = "") -> Any:
        return _run_rlm(rlm, language_model, turns=turns, library=library, correction=correction)

    return propose


def _run_rlm(rlm: Any, language_model: Any, **kwargs: Any) -> Any:
    """Call a ``dspy.RLM`` with the retry policy shared by proposing and revising."""
    import time

    import dspy
    from dspy.clients.lm import LMRateLimitError, LMServerError
    from dspy.primitives.code_interpreter import CodeInterpreterError

    from bandits.transport import backoff_delay

    # Two kinds of retry, and only these. The Deno REPL has been seen to
    # die mid-execution on a first boot, and a rate limit mid-loop loses
    # the whole invocation — both are fixed by trying again. A model reply
    # that fails to parse is not, and gets no retry here.
    sandbox_failures = 0
    for attempt in range(6):
        try:
            with dspy.context(lm=language_model):
                return rlm(**kwargs)
        except CodeInterpreterError as exc:
            sandbox_failures += 1
            if sandbox_failures > 1:
                raise ProposalError(f"the REPL sandbox failed twice: {exc}") from exc
        except (LMRateLimitError, LMServerError) as exc:
            if attempt == 5:
                raise ProposalError(f"the model stayed unavailable: {exc}") from exc
            time.sleep(backoff_delay(attempt))
    raise ProposalError("unreachable")  # pragma: no cover


_REVISE_INSTRUCTION = textwrap.dedent(
    """\
    You previously proposed one predicate for a family of agent trajectories. A human
    reviewer looked at turns it fired on and says it is wrong in some way — a false
    positive it should not have flagged, a false negative it missed, or a hypothesis
    that does not hold up. Fix it; do not just restate the original.

    Kind of trace: {archetype}. {vocabulary}

    The predicate being revised:
        name: {name}
        hypothesis: {hypothesis}

    ```python
    {code}
    ```

    `feedback` is the reviewer's own words on what is wrong with it.

    `turns` is a JSON string: a list of the specific turns the reviewer pointed at,
    with keys trace_id, index, task, action, next_state, reactions, observed, errored,
    judge (+1 / 0 / -1 / null), hint. Parse it with json.loads and study why the check's
    verdict on each of these turns disagreed with the reviewer.

    Same contract as a first proposal:

        def check(turn):
            \"\"\"<one-line hypothesis>\"\"\"
            ...
            return True   # the turn is bad
            # or False when it is fine, or None when the predicate does not apply

    - A pure function of `turn` (minus judge and hint — they will not be there).
    - No imports, no I/O, no eval/exec/open, no attribute starting with `__`.
      Builtins available: len any all sum min max sorted set list dict str int float bool
      range enumerate zip abs round isinstance tuple map filter reversed, and re_search(pattern, text).
    - Under 25 lines.

    Test the revision against `turns` in the REPL before returning it: it should no
    longer misjudge the turns the reviewer flagged.

    Return `checks`: a list with exactly one revised {{name, hypothesis, code}} object.
    """
)


def revise_instruction_for(archetype: Archetype, *, name: str, hypothesis: str, code: str) -> str:
    return _REVISE_INSTRUCTION.format(
        archetype=archetype.value,
        vocabulary=ARCHETYPE_VOCABULARY[archetype],
        name=name,
        hypothesis=hypothesis,
        code=code,
    )


class Reviser(Protocol):
    def __call__(self, *, feedback: str, turns: str) -> Any: ...


def build_reviser(
    *,
    archetype: Archetype,
    name: str,
    hypothesis: str,
    code: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_iterations: int = 10,
    max_llm_calls: int = 20,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Reviser:
    """A ``dspy.RLM`` that revises one existing predicate against reviewer feedback."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ProposalError("revising needs the 'audit' extra: uv sync --extra audit") from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
        num_retries=8,
    )

    class _Revise(dspy.Signature):
        feedback: str = dspy.InputField(desc="what a human reviewer said is wrong with it")
        turns: str = dspy.InputField(
            desc="JSON list of the turns the reviewer pointed at, with judge verdicts"
        )
        checks: list[ProposedCheck] = dspy.OutputField()

    _Revise.__doc__ = revise_instruction_for(archetype, name=name, hypothesis=hypothesis, code=code)
    rlm = dspy.RLM(
        _Revise, max_iters=max_iterations, max_llm_calls=max_llm_calls, sub_lm=language_model
    )

    def revise(*, feedback: str, turns: str) -> Any:
        return _run_rlm(rlm, language_model, feedback=feedback, turns=turns)

    return revise


# ------------------------------------------------------------------ sandbox

_FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "input",
        "breakpoint",
        "exit",
        "quit",
    }
)
_FORBIDDEN_KEYS = frozenset({"judge", "hint"})


class RejectedCheck(ValueError):
    pass


def _check_ast(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RejectedCheck(f"syntax error: {exc}") from exc
    # Only function definitions (plus an optional docstring) may appear at
    # module level. A top-level `while True: pass` or similar runs inside
    # compile_check's exec(), before run_check's per-call alarm exists to
    # kill it, and would otherwise hang the process forever.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # A default, an annotation, or a decorator is evaluated as an
            # expression by compile_check's exec(), before this function has
            # returned and before run_check's per-call alarm exists to bound
            # it -- `def check(turn, x=sum(range(10**10))): ...` hangs the
            # process right here, with every builtin this sandbox grants.
            if node.decorator_list:
                raise RejectedCheck("a check function may not use decorators")
            if node.returns is not None:
                raise RejectedCheck("a check function may not use a return annotation")
            args = node.args
            parameters = (
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                *((args.vararg,) if args.vararg else ()),
                *((args.kwarg,) if args.kwarg else ()),
            )
            if any(arg.annotation is not None for arg in parameters):
                raise RejectedCheck("a check function's parameters may not use annotations")
            if args.defaults or any(default is not None for default in args.kw_defaults):
                raise RejectedCheck("a check function's parameters may not use default values")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        raise RejectedCheck(f"only function definitions are allowed at module level, got {type(node).__name__}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RejectedCheck("imports are not allowed")
        if isinstance(node, (ast.Try, ast.TryStar)):
            # run_check's SIGALRM timeout works by raising _Timeout() inside
            # whatever is executing when the alarm fires. A `try/except`
            # inside the check's own body can catch that exception before it
            # reaches run_check's handler -- `try: while True: pass / except:
            # pass` swallows the interrupt and loops forever, with the alarm
            # already spent. Disallowed outright: nothing a check needs to do
            # (a boolean predicate over a turn dict) requires exception
            # handling of its own.
            raise RejectedCheck("a check may not use try/except")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RejectedCheck(f"dunder attribute {node.attr!r}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise RejectedCheck(f"name {node.id!r} is not allowed")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _FORBIDDEN_KEYS
        ):
            raise RejectedCheck(f"a check may not read {node.value!r}")


def _re_search(pattern: str, text: object) -> bool:
    import re

    try:
        return re.search(pattern, str(text or ""), re.IGNORECASE) is not None
    except re.error:
        return False


_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len",
        "any",
        "all",
        "sum",
        "min",
        "max",
        "sorted",
        "set",
        "list",
        "dict",
        "str",
        "int",
        "float",
        "bool",
        "range",
        "enumerate",
        "zip",
        "abs",
        "round",
        "isinstance",
        "tuple",
        "map",
        "filter",
        "reversed",
        "True",
        "False",
        "None",
    )
    if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
}


def compile_check(code: str) -> Callable[[dict[str, Any]], bool | None]:
    _check_ast(code)
    namespace: dict[str, Any] = {"__builtins__": _BUILTINS, "re_search": _re_search}
    exec(compile(code, "<check>", "exec"), namespace)  # noqa: S102 - sandboxed above
    # `check` when the model followed the brief; otherwise the one function it
    # defined. Models name predicates after their hypothesis, and a rename is
    # not a reason to lose a check that runs.
    defined = [node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]
    name = "check" if "check" in defined else (defined[0] if len(defined) == 1 else None)
    fn = namespace.get(name) if name else None
    if not callable(fn):
        raise RejectedCheck("define exactly one function, or one named 'check'")
    return fn


class _Timeout(Exception):
    pass


def run_check(
    fn: Callable[[dict[str, Any]], bool | None],
    turns: Sequence[dict[str, Any]],
    *,
    seconds: float = 1.0,
) -> tuple[dict[tuple[str, int], bool | None], int]:
    """Run one compiled check over turn dicts. Exceptions count, never propagate.

    A turn's result is ``None`` whether the check legitimately abstained or
    raised -- a caller deciding whether a turn was *confirmed* clean should
    not treat either as confirmation (see ``apply_verifier``); only ``errors``
    (the count, for ``CheckStats``) distinguishes exceptions from a clean
    ``return None``, and no caller currently needs to know which specific
    turns they were.
    """
    results: dict[tuple[str, int], bool | None] = {}
    errors = 0

    def _alarm(signum, frame):  # noqa: ANN001
        raise _Timeout()

    try:
        old = signal_mod.signal(signal_mod.SIGALRM, _alarm)
    except ValueError:  # not the main thread
        old = None
    try:
        for turn in turns:
            key = (turn["trace_id"], turn["index"])
            if old is not None:
                signal_mod.setitimer(signal_mod.ITIMER_REAL, seconds)
            try:
                # A defensive copy: a check must not be able to poison the
                # shared payload for the checks that run after it.
                value = fn(copy.deepcopy(turn))
                results[key] = None if value is None else bool(value)
            except (_Timeout, Exception):  # noqa: BLE001
                errors += 1
                results[key] = None
            finally:
                if old is not None:
                    signal_mod.setitimer(signal_mod.ITIMER_REAL, 0)
    finally:
        if old is not None:
            signal_mod.signal(signal_mod.SIGALRM, old)
    return results, errors


# ----------------------------------------------------------------- scoring


class CheckStats(Contract):
    """How one check behaved over every turn of the family, against the judge."""

    turns: int
    fired: int
    fired_scored: int
    """Fired on turns the judge scored."""

    fired_negative: int
    fired_positive: int
    negatives: int
    """Turns the judge scored −1, whether or not the check fired."""

    errors: int = 0
    examples: tuple[tuple[str, int], ...] = ()
    """(trace_id, index) of turns it fired on, for a reviewer to open."""

    missed: tuple[tuple[str, int], ...] = ()
    """(trace_id, index) of turns the judge scored −1 that this check did NOT
    fire on — candidate false negatives, for a reviewer revising a check that
    is too narrow. A sample, same cap as ``examples``."""

    @property
    def precision(self) -> float | None:
        return None if not self.fired_scored else self.fired_negative / self.fired_scored

    @property
    def recall(self) -> float | None:
        return None if not self.negatives else self.fired_negative / self.negatives


def evaluate_check(
    code: str,
    turns: Sequence[dict[str, Any]],
    verdicts: Mapping[tuple[str, int], TurnVerdict],
    *,
    examples: int = 5,
) -> tuple[CheckStats, str | None]:
    """Compile and run one check; the second value is why it could not run."""
    try:
        fn = compile_check(code)
    except RejectedCheck as exc:
        return CheckStats(
            turns=len(turns),
            fired=0,
            fired_scored=0,
            fired_negative=0,
            fired_positive=0,
            negatives=sum(1 for v in verdicts.values() if v.score == -1),
        ), str(exc)
    results, errors = run_check(fn, turns)
    fired = fired_scored = fired_negative = fired_positive = 0
    sample: list[tuple[str, int]] = []
    missed: list[tuple[str, int]] = []
    for key, value in results.items():
        verdict = verdicts.get(key)
        if not value:
            if len(missed) < examples and verdict is not None and verdict.score == -1:
                missed.append(key)
            continue
        fired += 1
        if len(sample) < examples:
            sample.append(key)
        if verdict is None or verdict.score is None:
            continue
        fired_scored += 1
        if verdict.score < 0:
            fired_negative += 1
        elif verdict.score > 0:
            fired_positive += 1
    return CheckStats(
        turns=len(turns),
        fired=fired,
        fired_scored=fired_scored,
        fired_negative=fired_negative,
        fired_positive=fired_positive,
        negatives=sum(1 for v in verdicts.values() if v.score == -1),
        errors=errors,
        examples=tuple(sample),
        missed=tuple(missed),
    ), None


def _code_digest(code: str) -> str:
    """A content digest of a check's code, used only to detect duplicates.

    Not a record identity: two different rounds can legitimately propose the
    same code (a revision that fails to change anything, for one), and giving
    both records the same id would let a decision on one silently apply to
    the other.
    """
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def _check_id(code: str, ordinal: int) -> str:
    """A unique identity for one check record.

    ``ordinal`` is the record's position in the verifier's checks — the
    length of the tuple before it was appended — which is monotonic for a
    verifier's whole lifetime (checks are only ever appended, never
    reordered or removed), so two records never collide even when their code
    is identical.
    """
    return f"{_code_digest(code)}-{ordinal:03d}"


class FamilyCheck(Contract):
    check_id: str
    name: str
    hypothesis: str
    code: str
    code_digest: str
    stats: CheckStats
    survived: bool
    reason: str
    round_number: int = 1
    decision: Literal["pending", "accepted", "rejected", "revised"] = "pending"
    note: str = ""
    parent_check_id: str | None = None
    """The check this one was revised from, if any. A revised parent is kept
    (decision "revised") rather than deleted, so the history stays legible."""


class FamilyVerifier(Contract):
    schema_version: int = 1
    family_id: str
    archetype: Archetype
    corpus_id: str
    judge_run_id: str
    model: str
    prompt_digest: str
    checks: tuple[FamilyCheck, ...]
    proposed: int = 0
    rounds: int = 0
    failed_rounds: int = 0
    """Rounds that produced no prediction after a retry. Reported, so a verifier
    with few checks can be told apart from one that never got its second round."""

    min_fired: int = 3
    min_precision: float = 0.6
    """The acceptance bar this verifier's checks were measured against. A
    revision has to be held to the same bar as a first proposal — recorded
    here rather than left to whatever a caller happens to pass, so a verifier
    built at a stricter bar can't quietly grow a child accepted at the
    default one."""

    raw_replies: tuple[str, ...] = ()

    def accepted(self) -> tuple[FamilyCheck, ...]:
        return tuple(c for c in self.checks if c.decision == "accepted")

    def pending(self) -> tuple[FamilyCheck, ...]:
        return tuple(c for c in self.checks if c.decision == "pending")


def survival(stats: CheckStats, *, min_fired: int, min_precision: float) -> tuple[bool, str]:
    if stats.errors and stats.errors > stats.turns // 2:
        return False, f"raised on {stats.errors} of {stats.turns} turns"
    if stats.fired < min_fired:
        return False, f"fired on {stats.fired} turn(s); needs {min_fired}"
    if stats.fired > stats.turns * 0.8:
        return False, f"fired on {stats.fired} of {stats.turns} turns; flags nearly everything"
    if stats.fired_scored < min_fired:
        return False, f"fired on only {stats.fired_scored} judged turn(s); cannot be measured"
    precision = stats.precision or 0.0
    if precision < min_precision:
        return False, f"precision vs judge {precision:.2f} below {min_precision:.2f}"
    return True, f"precision vs judge {precision:.2f} over {stats.fired_scored} judged turn(s)"


def _rows(value: Any) -> list[Any]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
        except ValueError:
            return []
    if isinstance(value, dict):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else []


def parse_checks(prediction: Any) -> list[ProposedCheck]:
    raw = getattr(prediction, "checks", prediction)
    out: list[ProposedCheck] = []
    for row in _rows(raw):
        if isinstance(row, ProposedCheck):
            out.append(row)
            continue
        if isinstance(row, BaseModel):
            row = row.model_dump()
        if not isinstance(row, dict):
            continue
        code = row.get("code")
        if not isinstance(code, str) or "def " not in code:
            continue
        out.append(
            ProposedCheck(
                name=str(row.get("name") or f"check_{len(out)}").strip()[:60],
                hypothesis=str(row.get("hypothesis") or "").strip()[:400],
                code=code,
            )
        )
    return out


def turn_payload(
    turns: Sequence[Turn],
    tasks: Mapping[str, str | None],
    verdicts: Mapping[tuple[str, int], TurnVerdict],
    *,
    with_judge: bool,
    clip: int | None = None,
) -> list[dict[str, Any]]:
    """Turns as plain rows. ``clip`` shortens action and reaction text for a
    sample shown to a model: the full text is what the checks are run over,
    but a model that prints a 300-kilobyte variable to read it runs out of
    tokens before it has written a predicate."""
    rows = []
    for turn in turns:
        row = turn.as_dict(task=(tasks.get(turn.trace_id) or "")[:600] or None)
        if clip is not None:
            row["action"] = _excerpt(row["action"], clip)
            row["next_state"] = (
                None if row["next_state"] is None else _excerpt(row["next_state"], clip)
            )
            row["reactions"] = [
                {**r, "text": _excerpt(r["text"], clip // 2)} for r in row["reactions"]
            ]
        if with_judge:
            verdict = verdicts.get((turn.trace_id, turn.index))
            row["judge"] = None if verdict is None else verdict.score
            row["hint"] = "" if verdict is None else verdict.hint
        rows.append(row)
    return rows


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}…[{len(text) - limit} chars omitted]…{text[-(limit - head) :]}"


SAMPLE_CLIP = 700
"""Characters of action and reaction text per turn in the sample shown to the
proposer. Full text is used when the checks are executed."""


def sample_turns(
    turns: Sequence[Turn], verdicts: Mapping[tuple[str, int], TurnVerdict], *, size: int, seed: int
) -> list[Turn]:
    """A stratified sample: every negative first, then neutrals and positives."""
    rng = random.Random(seed)
    negative = [t for t in turns if (v := verdicts.get((t.trace_id, t.index))) and v.score == -1]
    other = [t for t in turns if t.observed and t not in negative]
    rng.shuffle(negative)
    rng.shuffle(other)
    chosen = negative[: max(1, size // 2)]
    chosen += other[: size - len(chosen)]
    if len(chosen) < size:
        chosen += negative[len(chosen) - len(other) :][: size - len(chosen)]
    return sorted(chosen, key=lambda t: (t.trace_id, t.index))


def _library_text(checks: Sequence[FamilyCheck]) -> str:
    lines = []
    for check in checks:
        if not check.survived:
            continue
        lines.append(f"- {check.name}: {check.hypothesis} [{check.reason}]")
    return "\n".join(lines)


def propose_verifier(
    turns: Sequence[Turn],
    tasks: Mapping[str, str | None],
    judge_run: TurnJudgeRun,
    judge_run_id: str,
    *,
    family_id: str,
    propose: Proposer,
    model: str = DEFAULT_MODEL,
    rounds: int = 2,
    sample: int = 120,
    seed: int = 42,
    min_fired: int = 3,
    min_precision: float = 0.6,
) -> FamilyVerifier:
    """Ask for checks, re-execute every one over the family, keep what survives."""
    verdicts = judge_run.verdict_by_key()
    full = turn_payload(turns, tasks, verdicts, with_judge=False)
    kept: list[FamilyCheck] = []
    seen: set[str] = set()
    replies: list[str] = []
    correction = ""
    proposed = 0
    failed_rounds = 0

    for round_number in range(1, rounds + 1):
        prediction = None
        # A round whose REPL died is retried once over a different sample. At
        # temperature zero the same sample produces the same code and the same
        # crash, so retrying the identical call is not a retry.
        for attempt, reseed in enumerate((0, 100 * round_number)):
            shown = sample_turns(turns, verdicts, size=sample, seed=seed + round_number + reseed)
            payload = json.dumps(
                turn_payload(shown, tasks, verdicts, with_judge=True, clip=SAMPLE_CLIP),
                default=str,
            )
            try:
                prediction = propose(
                    turns=payload, library=_library_text(kept), correction=correction
                )
                break
            except ProposalError as exc:
                replies.append(f"round {round_number} attempt {attempt + 1} failed: {exc}")
        if prediction is None:
            failed_rounds += 1
            continue
        try:
            replies.append(json.dumps(getattr(prediction, "checks", None), default=str)[:20000])
        except (TypeError, ValueError):
            replies.append(str(prediction)[:20000])
        checks = parse_checks(prediction)
        proposed += len(checks)
        problems: list[str] = []
        for check in checks:
            if check.code in seen:
                continue
            seen.add(check.code)
            stats, error = evaluate_check(check.code, full, verdicts)
            if error is not None:
                kept.append(
                    FamilyCheck(
                        check_id=_check_id(check.code, len(kept)),
                        name=check.name,
                        hypothesis=check.hypothesis,
                        code=check.code,
                        code_digest=_code_digest(check.code),
                        stats=stats,
                        survived=False,
                        reason=f"rejected: {error}",
                        round_number=round_number,
                    )
                )
                problems.append(f"{check.name}: {error}")
                continue
            survived, reason = survival(stats, min_fired=min_fired, min_precision=min_precision)
            kept.append(
                FamilyCheck(
                    check_id=_check_id(check.code, len(kept)),
                    name=check.name,
                    hypothesis=check.hypothesis,
                    code=check.code,
                    code_digest=_code_digest(check.code),
                    stats=stats,
                    survived=survived,
                    reason=reason,
                    round_number=round_number,
                )
            )
            if not survived:
                problems.append(f"{check.name}: {reason}")
        if not checks:
            problems.append(
                "no checks were returned; return a JSON list of {name, hypothesis, code}"
            )
        correction = "\n".join(problems)

    if failed_rounds == rounds:
        raise ProposalError(f"no round produced a prediction: {'; '.join(replies)[:600]}")

    return FamilyVerifier(
        family_id=family_id,
        archetype=judge_run.archetype,
        corpus_id=judge_run.corpus_id,
        judge_run_id=judge_run_id,
        model=model,
        prompt_digest=prompt_digest(model),
        checks=tuple(kept),
        proposed=proposed,
        rounds=rounds,
        failed_rounds=failed_rounds,
        min_fired=min_fired,
        min_precision=min_precision,
        raw_replies=tuple(replies),
    )


# --------------------------------------------------------------- review/apply


def decide_check(
    verifier: FamilyVerifier,
    check_id: str,
    decision: Literal["accepted", "rejected"],
    note: str = "",
) -> FamilyVerifier:
    """Record one human decision, targeting the check's code digest.

    Names are model-chosen and can collide across two different predicates;
    ``check_id`` is a digest of the code itself, so a decision can never land
    on the wrong check.
    """
    if not any(c.check_id == check_id for c in verifier.checks):
        raise ValueError(f"no check with id {check_id!r}")
    return verifier.replace(
        checks=tuple(
            c.replace(decision=decision, note=note) if c.check_id == check_id else c
            for c in verifier.checks
        )
    )


def revise_check(
    verifier: FamilyVerifier,
    check_id: str,
    feedback: str,
    counterexamples: Sequence[tuple[str, int]],
    turns: Sequence[Turn],
    tasks: Mapping[str, str | None],
    judge_run: TurnJudgeRun,
    *,
    reviser: Reviser,
    min_fired: int | None = None,
    min_precision: float | None = None,
) -> FamilyVerifier:
    """Send one check back with feedback and counterexamples; score what comes back.

    The reviewer's note alone was previously stored as inert provenance — this
    is the loop the reviewer asked for: it feeds the note and the specific
    turns it disagreed on back into another RLM round, re-executes the result
    against the whole family exactly like a first proposal, and queues it as a
    new pending check linked to the original by ``parent_check_id``. The
    original is marked "revised", not discarded, so a reviewer can see what it
    used to be.

    ``min_fired``/``min_precision`` default to the bar the verifier itself was
    built at, not a fresh default — a verifier proposed at a stricter bar must
    hold a revision to it too, or a "survived" child could pass only because
    revision quietly relaxed the threshold.
    """
    original = next((c for c in verifier.checks if c.check_id == check_id), None)
    if original is None:
        raise ValueError(f"no check with id {check_id!r}")
    if not counterexamples:
        raise ValueError("revise needs at least one counterexample turn")

    verdicts = judge_run.verdict_by_key()
    by_key = {(t.trace_id, t.index): t for t in turns}
    missing = [key for key in counterexamples if key not in by_key]
    if missing:
        raise ValueError(f"unknown turn(s): {missing}")

    shown = [by_key[key] for key in counterexamples]
    payload = json.dumps(
        turn_payload(shown, tasks, verdicts, with_judge=True, clip=SAMPLE_CLIP), default=str
    )
    prediction = reviser(feedback=feedback, turns=payload)
    revised = parse_checks(prediction)
    if not revised:
        raise ProposalError("the revision returned no check")
    proposal = revised[0]

    digest = _code_digest(proposal.code)
    existing = next((c for c in verifier.checks if c.code_digest == digest), None)
    if existing is not None:
        same = "the original" if existing.check_id == original.check_id else f"{existing.name!r}"
        raise ProposalError(f"the revision is identical to {same}; nothing changed")

    full = turn_payload(turns, tasks, verdicts, with_judge=False)
    stats, error = evaluate_check(proposal.code, full, verdicts)
    bar_fired = verifier.min_fired if min_fired is None else min_fired
    bar_precision = verifier.min_precision if min_precision is None else min_precision
    if error is not None:
        survived, reason = False, f"rejected: {error}"
    else:
        survived, reason = survival(stats, min_fired=bar_fired, min_precision=bar_precision)

    child = FamilyCheck(
        check_id=_check_id(proposal.code, len(verifier.checks)),
        name=proposal.name,
        hypothesis=proposal.hypothesis,
        code=proposal.code,
        code_digest=digest,
        stats=stats,
        survived=survived,
        reason=reason,
        round_number=original.round_number + 1,
        parent_check_id=original.check_id,
    )
    updated_original = original.replace(
        decision="revised", note=f"revised as {child.check_id}: {feedback}"[:400]
    )
    checks = tuple(
        updated_original if c.check_id == check_id else c for c in verifier.checks
    ) + (child,)
    return verifier.replace(checks=checks, proposed=verifier.proposed + 1)


class FlaggedTurn(Contract):
    index: int
    by: tuple[str, ...]
    """Check ids, and ``judge`` when the next-state judge scored it −1.

    Ids, not names: a revision can give a child check the same ``name`` as the
    check it revised, or as any other check in the family — ``check_id`` is
    the only field this family guarantees unique.
    """


class TraceScore(Contract):
    trace_id: str
    turns: int
    observed: int
    flagged: tuple[FlaggedTurn, ...] = ()
    unresolved: tuple[int, ...] = ()
    """Observed turn indices with no confirmed-clean signal from any source:
    not every applied check resolved to an actual boolean (one erroring or
    abstaining is enough — checks are OR'd, so a False from one check proves
    nothing about a turn another check couldn't evaluate), and the judge did
    not resolve it either (excluded, or produced no score). Zero signal, as
    opposed to zero signal *because nothing was wrong*. Never also in
    ``flagged``: a turn with a real flag has real evidence, whatever else
    about it failed."""

    @property
    def passes(self) -> bool:
        return self.observed > 0 and not self.flagged and not self.unresolved

    @property
    def score(self) -> float | None:
        """None, not a number, when a turn's clean status is unconfirmed.

        Counting an unresolved turn as clean in the denominator would report
        a rate over turns this verifier did not actually rate.
        """
        if not self.observed or self.unresolved:
            return None
        return max(0.0, 1.0 - len(self.flagged) / self.observed)


class VerifierScores(Contract):
    schema_version: int = 1
    verifier_id: str
    judge_run_id: str
    include_judge: bool
    checks_applied: tuple[str, ...]
    """``check_id`` of every check that ran, not ``name`` — names are not
    guaranteed unique across a family (a revision may reuse one)."""
    via_survivors: bool = False
    """True when scoring drew from every automatic survivor rather than only
    accepted checks (``score-traces --survivors``). Recorded independently of
    ``checks_applied`` so an export can't infer "reviewed" from an empty or
    coincidentally-all-accepted set of ids scored in survivor mode."""
    scores: tuple[TraceScore, ...]


def apply_verifier(
    verifier: FamilyVerifier,
    turns: Sequence[Turn],
    tasks: Mapping[str, str | None],
    judge_run: TurnJudgeRun,
    *,
    verifier_id: str,
    judge_run_id: str,
    include_judge: bool = True,
    checks: Sequence[FamilyCheck] | None = None,
    via_survivors: bool = False,
) -> VerifierScores:
    """Score every trace: a turn is flagged by any accepted check, or by the judge.

    Checks run only over observed turns. An unobserved turn — usually the
    last of an episode — has no reaction to judge it by, and a predicate that
    happens to fire on one (``return not turn["observed"]``, or anything else
    true of a turn with nothing in it) must not be allowed to flag it: that
    would score silence as if it were a failure the reaction actually showed.
    """
    applied = tuple(checks if checks is not None else verifier.accepted())
    payload = turn_payload([t for t in turns if t.observed], tasks, {}, with_judge=False)
    flags: dict[tuple[str, int], list[str]] = {}
    indefinite: set[tuple[str, int]] = set()
    """Turns where some applied check did not return an actual boolean --
    whether it raised or legitimately abstained with ``None``, both mean
    that check contributed nothing. Scoring by checks is an OR across all of
    them: a turn is only confirmed clean by the checks if *every one*
    resolved it to False (or True, which is already a flag) -- one check
    returning False while another is indefinite proves nothing, because the
    indefinite one might have been the one that would have fired.
    """
    any_check_compiled = False
    for check in applied:
        try:
            fn = compile_check(check.code)
        except RejectedCheck:
            continue
        any_check_compiled = True
        results, _errors = run_check(fn, payload)
        for key, value in results.items():
            if value is None:
                indefinite.add(key)
            if value:
                flags.setdefault(key, []).append(check.check_id)
    judge_by_key = {(v.trace_id, v.index): v for v in judge_run.verdicts}
    if include_judge:
        for verdict in judge_run.verdicts:
            if verdict.score == -1:
                flags.setdefault((verdict.trace_id, verdict.index), []).append("judge")

    def has_signal(key: tuple[str, int]) -> bool:
        """A real judge score, or every applied check resolving to an actual
        boolean: either is confirmation a turn was clean, not merely
        evidence that something looked and shrugged."""
        judged = include_judge and (verdict := judge_by_key.get(key)) is not None and (
            verdict.score is not None
        )
        checked = any_check_compiled and key not in indefinite
        return judged or checked

    by_trace: dict[str, list[Turn]] = {}
    for turn in turns:
        by_trace.setdefault(turn.trace_id, []).append(turn)
    scores = []
    for trace_id, own in by_trace.items():
        flagged = tuple(
            FlaggedTurn(index=t.index, by=tuple(flags[(trace_id, t.index)]))
            for t in own
            if (trace_id, t.index) in flags
        )
        unresolved = tuple(
            t.index
            for t in own
            if t.observed
            and (trace_id, t.index) not in flags
            and not has_signal((trace_id, t.index))
        )
        scores.append(
            TraceScore(
                trace_id=trace_id,
                turns=len(own),
                observed=sum(1 for t in own if t.observed),
                flagged=flagged,
                unresolved=unresolved,
            )
        )
    return VerifierScores(
        verifier_id=verifier_id,
        judge_run_id=judge_run_id,
        include_judge=include_judge,
        checks_applied=tuple(c.check_id for c in applied),
        via_survivors=via_survivors,
        scores=tuple(scores),
    )


def compute_verifier_id(verifier: FamilyVerifier) -> str:
    digest = hashlib.sha256(verifier.model_dump_json().encode()).hexdigest()
    return f"family-verifier-{digest[:16]}"


def save_family_verifier(verifier: FamilyVerifier, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_verifier_id(verifier),
        kind="family_verifier",
        parent_artifact_id=verifier.judge_run_id,
        payload=verifier.model_dump_json().encode(),
        summary={
            "proposed": verifier.proposed,
            "survived": sum(1 for c in verifier.checks if c.survived),
            "accepted": len(verifier.accepted()),
            "pending": len(verifier.pending()),
        },
    )


def load_family_verifier(verifier_id: str, store: DerivedStore) -> FamilyVerifier:
    return FamilyVerifier.model_validate_json(store.read_payload(verifier_id))


def compute_scores_id(scores: VerifierScores) -> str:
    digest = hashlib.sha256(scores.model_dump_json().encode()).hexdigest()
    return f"verifier-scores-{digest[:16]}"


def save_verifier_scores(scores: VerifierScores, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_scores_id(scores),
        kind="verifier_scores",
        parent_artifact_id=scores.verifier_id,
        payload=scores.model_dump_json().encode(),
        summary={
            "traces": len(scores.scores),
            "passing": sum(1 for s in scores.scores if s.passes),
            "flagged_turns": sum(len(s.flagged) for s in scores.scores),
            "unresolved_traces": sum(1 for s in scores.scores if s.unresolved),
            "unresolved_turns": sum(len(s.unresolved) for s in scores.scores),
        },
    )


def load_verifier_scores(scores_id: str, store: DerivedStore) -> VerifierScores:
    return VerifierScores.model_validate_json(store.read_payload(scores_id))
