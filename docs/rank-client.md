# Bandr Rank client

`bandits.rank` is the public part of Bandr Rank: the statistics behind the ranking, a builder that
turns recorded traces into rank tasks, and a client for the hosted loop (`pip install "bandits[rank]"`).

```python
from bandits.rank.client import RankClient, Receipts

client = RankClient("https://app.bandr.ai", BANDR_KEY)
while ep := client.next_episode(run_id):          # None when every task x model x k is done
    receipts = Receipts()
    out = run_my_agent(ep.task["input"],
                       base_url=client.gateway_url,        # OpenAI-compatible
                       extra_headers=client.headers(ep),   # Bandr picks the model, not you
                       on_response=receipts.add)           # keep x-bandr-record-id from each reply
    client.report(ep, score=my_check(ep.task, out), references=receipts.ids, feedback=out.summary)
```

- Score with your own check (tests, a state query), never the agent's own success claim.
- Calls without the `x-bandr-episode` header (a user simulator, a grader) pass through unchanged.
- A report must cite receipts from its own episode; a report with none is rejected.
- If the control model that gives up scores above 0, your check is too lenient: fix it, then `client.recheck(run_id)`.
- `bandits.rank.stats` holds pass@k, bootstrap intervals, ties, Kendall and Spearman, so the hosted
  ranking can be recomputed from the per-task counts. `bandits.rank.tasks` builds tasks from traces.
