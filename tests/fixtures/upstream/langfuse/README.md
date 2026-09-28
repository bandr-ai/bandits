# Upstream Langfuse trace

`agno-2025-06-11.trace.json` is an unmodified trace fixture from
[Langfuse's framework trace tests](https://github.com/langfuse/langfuse/blob/5cdcbaf6a50dc05a7c5859be821f2a07b0c7f7b0/worker/src/__tests__/chatml/framework-traces/agno-2025-06-11.trace.json).
It contains one trace with six observations and is not an private-app recording.
The repository licenses this path under MIT; its license is copied alongside
the fixture. The paired ChatML fixture is useful as a comparison but is not
ground truth for the model messages: some "successful" parses contain only
tool definitions, not message roles or content.
