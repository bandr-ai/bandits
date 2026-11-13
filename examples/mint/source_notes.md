# Synthetic source notes

This entire fixture is made up and contains no production logs. Read all three
records in logs/attempts.jsonl. Each line is a complete attempt at the requested
sum job. The numbers field is the complete starting file supplied by the fixture.
The instance field groups repeated attempts; two attempts belong to sum-a.
The ended field records completion, not success. Two logged answers are wrong.
The expected answer is recomputed from the starting file.

Mint's default environment is sufficient for this synthetic job. The answer
oracle demonstrates verifier wiring; it is not a replay or an independent solve.
This fixture does not demonstrate reconstructing a production environment.
