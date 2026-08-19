# ENVIRONMENT — <name>

<!--
TEMPLATE. `bandits.skill.scaffold.scaffold_workspace` writes a populated copy of
this file from a real ToolSurface and StateSchema. Use this blank only when you
are hand-authoring a workspace.

Format rules the parser depends on (bandits/skill/scaffold.py):
  * `### <id>`            binds the `- **Key:** value` lines under it to that id
  * `- **Key:** value`    a decision field; the value runs to end of line
  * pipe tables           addressed by column name, so columns may be reordered
  * `**TODO(human)**`     an unanswered question; NEVER treated as yes
Prose outside those three constructs is ignored by the parser. Write as much of
it as you like — this file is meant to be read.
-->

Everything stated as fact below was recovered from the traces. Everything marked
**TODO(human)** is a decision the traces cannot make.

**Review order:** tool classes first (they decide everything downstream), then the
blind-write check, then the entities.

## 1. Tool classification

`read` is answered from the rebuilt store. `write` mutates it and is what verifiers
assert on. `external` is stubbed, recorded to the effect ledger, and never performed.
`unknown` is not reimplemented at all — calling it raises rather than returning a
plausible success.

Edit the **decision** column. Leaving it equal to *inferred* accepts the inference.

| tool | inferred | decision | calls | confidence | flags |
|---|---|---|---|---|---|
| <tool> | read | **TODO(human)** | 0 | 0.00 |  |

## 2. Blind-write check — the thing our classifier cannot see

A tool is classified `write` only when the corpus shows a before/after difference on
the same row. A tool that mutates state and is **never followed by a read of what it
changed** leaves no such evidence anywhere in the corpus, so it is classified `read`.
The heuristic is not weak here, it is blind: the information is not in the data.

If a blind write ships as `read`, the rebuilt tool returns a row instead of changing
one, every verifier assertion about that change fails identically, and the task looks
impossible rather than mismodelled.

Answer for each tool. This is the highest-value five minutes in the whole procedure.

### tool:<tool>

- **Inferred:** read (<why>, <n> call(s))
- **Confirmed read-only:** **TODO(human)** (yes / no)
- **If no, what does it change:** **TODO(human)** (entity, field, and the value it sets)

## 3. External tools — recorded, never performed

- `<tool>` — stubbed; each attempt lands in the effect ledger so a verifier can assert
  it *would* have fired.

- **All irreversible tools are listed above:** **TODO(human)** (yes / no — name any tool
  that touches money, messaging, or a third party and is not here)

## 4. Reconstructed state

Entities are inferred from repeated identifiers across tool responses. Columns are the
union of every field any response ever showed.

### <entity>

- **Primary key:** <field>
- **Fields:** <field, field, ...>
- **Written by:** <tools>
- **Read by:** <tools>
- **Rows of evidence:** <n>
- **Foreign keys:** <field -> entity.field (confidence)>
- **Static snapshot:** yes — the corpus never wrote this entity and nothing
  cross-references it, so no structure could be inferred. Rows are materialized
  verbatim. We refuse to invent a table here.
- **Acceptable as a snapshot:** **TODO(human)** (yes / no — if agents must *write* this,
  say so and the pipeline needs write evidence, not a snapshot)
- **Observed write semantics:**
  - `<tool>` key=`<arg>` sets=<constants> arg_columns=<mapping>
- **Write semantics correct:** **TODO(human)** (yes / no + what is wrong)

### Unattributed tools

Tools that returned successful bodies attributable to no entity.

- **What state do these read or write:** **TODO(human)**

## 5. Fidelity

| tool | matched | rate | mismatched | unsupported |
|---|---|---|---|---|

- **Remaining gaps consciously accepted:** **TODO(human)** (name each failing tool and
  why the gap is tolerable, or send it back to schema inference)

## 6. Open questions

- **Anything the traces could not tell us:** **TODO(human)**
