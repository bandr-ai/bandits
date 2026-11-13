# Try Mint on synthetic logs

Three made-up attempts become two whole-job tasks. Two logged answers are wrong;
Mint's example generator recomputes the correct answers from the starting files.
No production data, credentials, or model calls are needed.

From the repository root, with the checkout installed (`pip install .`):

```bash
mint init examples/mint/logs /tmp/mint-demo --allow-missing-runtime
cp examples/mint/parse.py examples/mint/jobs.json examples/mint/source_notes.md /tmp/mint-demo/
mkdir -p /tmp/mint-demo/tasktypes
cp examples/mint/tasktypes/sum.py /tmp/mint-demo/tasktypes/
cd /tmp/mint-demo
mint run parse
mint run generate
mint validate --runner local
mint report
```

Use a new workspace path if `/tmp/mint-demo` already contains your own work.
Findings are optional; this example needs no checks.py. The supplied parser and
generator are specific to this fixture. For your own logs, the coding agent writes
them after inspecting the actual source.

Inspect `out/tasks/sum/sum-a/`:

```text
instruction.md            the original request
task.toml                 runtime and provenance
environment/Dockerfile    the default task environment
environment/data/numbers.txt
solution/solve.sh         generated answer oracle
tests/test.sh             verifier
tests/expected.json        expected answer, kept from the solver
```

The result is two `generated` tasks, with local execution gates passing. Local
checks exercise the verifier; they do not execute the task in a Harbor container.
For a real Harbor run, install Harbor and start Docker, then run:

```bash
mint validate --runner harbor
mint report
```

Only successful Harbor execution gates can produce `validated`. This example's
oracle writes a recomputed answer, so it proves wiring; inspect that the answer
really follows from `numbers.txt`. It is not an independent agent solve.
