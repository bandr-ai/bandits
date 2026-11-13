# Net-revenue source notes

This fixture is synthetic. Read the one complete record in logs/attempts.jsonl.
The agent was asked for net revenue from a supplied CSV. It returned 140 after
counting the refund as positive revenue. The input contains a 120-dollar sale
and a -20-dollar refund. Recomputing the signed sum gives 100.

The CSV is the complete starting file. The logged answer is evidence of the
failure and is not the definition of correct. The answer oracle is generated
from recomputation; local execution checks verifier wiring, not a separate agent
solve. Mint's default environment is sufficient for this synthetic file task.
