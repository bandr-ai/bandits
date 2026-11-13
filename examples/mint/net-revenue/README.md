# A failed revenue calculation becomes an eval

The agent was asked to calculate net revenue from a CSV. It counted a refund as
income and returned **140**. This example rebuilds the original task and checks
the signed total: **100 passes; 140 fails**.

The trace and its starting file are synthetic and fully included. From the
repository root, with this checkout installed:

```bash
mint init examples/mint/net-revenue/logs /tmp/bandr-revenue --allow-missing-runtime
cp examples/mint/net-revenue/parse.py examples/mint/net-revenue/jobs.json examples/mint/net-revenue/source_notes.md /tmp/bandr-revenue/
mkdir -p /tmp/bandr-revenue/tasktypes
cp examples/mint/net-revenue/tasktypes/revenue.py /tmp/bandr-revenue/tasktypes/
cd /tmp/bandr-revenue
mint run parse
mint run generate
mint validate --runner local
mint report
```

Choose a fresh workspace path if `/tmp/bandr-revenue` already contains your work.
The task is under `out/tasks/revenue/net-revenue/`. Its starting environment
contains `sales.csv`; the expected answer stays in the verifier's files.

The verifier accepts 100 and 100.0, rejects 140, and rejects doing nothing. The
answer oracle writes a recomputed result, so it demonstrates wiring rather than
an independent agent solve. Local checks leave this task **generated**. To run
container execution gates, install Harbor, start Docker, then run
`mint validate --runner harbor`. Passing gates are not a substitute for checking
that the reconstructed task matches the request and source evidence.
