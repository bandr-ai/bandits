"""Find the identifier-valued fields in a corpus.

The whole of stage 3 rests on one observation from PLAN.md Step 7: **IDs
repeat**. A value that the agent passes *into* a tool and that also comes back
*out of* a tool, under the same field name, is the handle on a row of some
table. Everything else - grouping, keys, foreign keys - is built on top of the
index this module produces.

This module is deliberately *loose*. It over-collects candidates (``status``
qualifies in the golden corpus, because ``update_order_status`` takes a status
in and hands one back) and leaves the tightening to :mod:`bandits.state.entities`,
which requires an identifier to actually anchor a row before it counts. Being
loose here and strict there is the safe direction: a missed identifier is a
table we never find, a spurious one is a candidate that loses a vote.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from bandits.contracts import CallStatus, InvocationPoint, JsonObject, JsonValue, TraceCorpus

#: Longest string still considered identifier-shaped. Prose (a policy blob, an
#: email body) is not an identifier, and treating it as one would let free text
#: drive table discovery.
MAX_IDENTIFIER_STR_LEN = 128

Scalar = str | int


@dataclass
class Mention:
    """One place a particular identifier value was seen."""

    trace_id: str
    step: int
    call_id: str
    tool: str
    where: str
    """``"arguments"`` or ``"response"``."""


@dataclass
class IdentifierField:
    """One field name that behaves like an identifier across the corpus."""

    name: str
    values: set[Scalar] = field(default_factory=set)
    arg_tools: set[str] = field(default_factory=set)
    response_tools: set[str] = field(default_factory=set)
    mentions: dict[Scalar, list[Mention]] = field(default_factory=lambda: defaultdict(list))

    @property
    def occurrences(self) -> int:
        return sum(len(m) for m in self.mentions.values())


@dataclass
class IdentifierIndex:
    """field name -> values -> the invocations that mention them."""

    fields: dict[str, IdentifierField] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.fields)

    def values(self, name: str) -> set[Scalar]:
        f = self.fields.get(name)
        return set(f.values) if f else set()

    def has(self, name: str) -> bool:
        return name in self.fields


def is_identifier_scalar(value: JsonValue) -> bool:
    """A value may be an identifier only if it is a short, non-boolean scalar.

    Rule: ``int`` or ``str`` only. ``bool`` is excluded even though it is an
    ``int`` in Python (``sent: true`` is a flag, not a row handle), floats are
    excluded (money and measurements are attributes, not keys), and strings
    longer than :data:`MAX_IDENTIFIER_STR_LEN` are excluded as prose.

    Insufficient evidence: none needed - this is a pure shape test.
    """
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return 0 < len(value) <= MAX_IDENTIFIER_STR_LEN
    return False


def scalar_fields(obj: JsonValue) -> dict[str, Scalar]:
    """Scalar, identifier-shaped fields of a flat JSON object.

    Only top-level scalars are returned. Nested objects and lists are *not*
    flattened: we have no evidence about whether a nested object is an embedded
    row or a value blob, and inventing a path-joined column name would be
    inventing structure. Callers that need the nested part must handle it
    explicitly (see :func:`scalar_list_fields`).
    """
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if is_identifier_scalar(v)}


def scalar_list_fields(obj: JsonValue) -> dict[str, list[Scalar]]:
    """Top-level fields whose value is a non-empty list of identifier-shaped scalars.

    These are how "list" tools answer: ``search_orders`` returns
    ``{"order_ids": [7741, 7742]}``. The list carries no row bodies, only
    references, so it is evidence of a *read*, never of an entity's fields.
    """
    out: dict[str, list[Scalar]] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if isinstance(v, list) and v and all(is_identifier_scalar(x) for x in v):
            out[k] = list(v)
    return out


def _record(
    index_fields: dict[str, IdentifierField],
    name: str,
    value: Scalar,
    where: str,
    inv: InvocationPoint,
) -> None:
    f = index_fields.get(name)
    if f is None:
        f = IdentifierField(name=name)
        index_fields[name] = f
    f.values.add(value)
    (f.arg_tools if where == "arguments" else f.response_tools).add(inv.tool)
    f.mentions[value].append(
        Mention(
            trace_id=inv.trace_id,
            step=inv.step,
            call_id=inv.call_id,
            tool=inv.tool,
            where=where,
        )
    )


def find_identifiers(corpus: TraceCorpus) -> IdentifierIndex:
    """Build the identifier index for a corpus.

    **Rule.** A field name ``F`` is an identifier field when some scalar value
    ``v`` appears as ``arguments[F]`` of one call *and* as a response value
    under the same name ``F`` in some call (the same call or another). "Under
    the same name" is what makes the repetition attributable: the same integer
    appearing as ``limit`` here and ``customer_id`` there is a coincidence, not
    a key.

    Response occurrences count when ``F`` is a top-level scalar of a response
    object, or when the value appears inside a top-level list of scalars whose
    field name is a plural form of ``F`` (``order_ids`` -> ``order_id``). The
    plural form is checked by :func:`_singular_candidates`, an explicit rule,
    not a library.

    **Evidence required.** At least one argument occurrence and one response
    occurrence of the same value. One matching value is enough to *nominate* a
    field; deciding whether it identifies a table is
    :mod:`bandits.state.entities`' job.

    **When evidence is insufficient.** A field seen only in arguments (e.g.
    ``send_email.to_customer_id``) or only in responses (e.g. ``total_cents``)
    is not nominated. We do not guess from the name: a field called ``_id`` with
    no recurrence gives us nothing to reconstruct a table from.

    Error responses are indexed too - a ``not_found`` body echoing
    ``order_id: 9999`` is still evidence that ``order_id`` is a handle - but
    :mod:`bandits.state.entities` never lets an error body contribute fields.
    """
    arg_hits: dict[str, IdentifierField] = {}
    resp_hits: dict[str, IdentifierField] = {}

    for trace in corpus.traces:
        for inv in trace.invocations:
            args: JsonObject = inv.arguments or {}
            for name, value in scalar_fields(args).items():
                _record(arg_hits, name, value, "arguments", inv)
            for name, value in scalar_fields(inv.response).items():
                _record(resp_hits, name, value, "response", inv)
            for name, values in scalar_list_fields(inv.response).items():
                for singular in _singular_candidates(name):
                    for v in values:
                        _record(resp_hits, singular, v, "response", inv)

    index = IdentifierIndex()
    for name, arg_field in arg_hits.items():
        resp_field = resp_hits.get(name)
        if resp_field is None:
            continue
        shared = arg_field.values & resp_field.values
        if not shared:
            continue
        merged = IdentifierField(
            name=name,
            values=arg_field.values | resp_field.values,
            arg_tools=arg_field.arg_tools | resp_field.arg_tools,
            response_tools=arg_field.response_tools | resp_field.response_tools,
        )
        for source in (arg_field, resp_field):
            for value, mentions in source.mentions.items():
                merged.mentions[value].extend(mentions)
        for mentions in merged.mentions.values():
            mentions.sort(key=lambda m: (m.trace_id, m.step, m.where))
        index.fields[name] = merged
    return index


def _singular_candidates(list_field_name: str) -> list[str]:
    """Singular forms a plural list field could be the plural of.

    Explicit, documented rule - no inflection library:

    * ``xs``      -> ``x``            (``order_ids`` -> ``order_id``)
    * ``xes``     -> ``x``            (``boxes``     -> ``box``)
    * ``xies``    -> ``xy``           (``policies``  -> ``policy``)
    * ``x_list``  -> ``x``            (``order_list``-> ``order``)
    * ``x_ids``   -> ``x_id``         (covered by the ``xs`` case)

    All plausible singulars are returned; the caller keeps only the ones that
    are real identifier fields, so an over-generous guess costs nothing.
    """
    n = list_field_name
    out: list[str] = []
    if n.endswith("_list"):
        out.append(n[: -len("_list")])
    if n.endswith("ies") and len(n) > 3:
        out.append(n[:-3] + "y")
    if n.endswith("es") and len(n) > 2:
        out.append(n[:-2])
    if n.endswith("s") and len(n) > 1:
        out.append(n[:-1])
    return [c for c in dict.fromkeys(out) if c]


def identifier_mentions(
    index: IdentifierIndex, name: str, value: Scalar
) -> list[Mention]:
    """Every invocation that mentioned ``value`` under identifier field ``name``."""
    f = index.fields.get(name)
    if f is None:
        return []
    return list(f.mentions.get(value, ()))


def ok_invocations(corpus: TraceCorpus) -> list[InvocationPoint]:
    """Every successful invocation in the corpus, in (trace, step) order.

    Only ``CallStatus.OK`` calls describe state. An error body is a *protocol*
    response - ``{"error": "not_found", "order_id": 9999}`` - and letting it
    contribute fields would put an ``error`` column on ``orders``.
    """
    out: list[InvocationPoint] = []
    for trace in corpus.traces:
        for inv in trace.invocations:
            if inv.status is CallStatus.OK:
                out.append(inv)
    out.sort(key=lambda i: (i.trace_id, i.step))
    return out


__all__ = [
    "IdentifierField",
    "IdentifierIndex",
    "MAX_IDENTIFIER_STR_LEN",
    "Mention",
    "Scalar",
    "find_identifiers",
    "identifier_mentions",
    "is_identifier_scalar",
    "ok_invocations",
    "scalar_fields",
    "scalar_list_fields",
]
