# bandits

**Bring your agent traces. Get Harbor tasks back: for SFT data and as RL environments.**

```
your traces  →  mint     →  Harbor tasks  →  SFT data
                                          →  RL environments
```

## Mint

Mint turns agent traces into runnable, verifiable Harbor tasks. It ships in the
`bandits-cli` distribution; the command and Python import are both `mint`.

```bash
pip install bandits-cli
mint --help
mint init <logs> <workspace>
mint status
```

Use `mint schema` for script contracts and `mint docs` for the bundled skill and
guides. See the [implementation plan](docs/plans/mint.md) for the workflow.
Generated tasks become `validated` only after passing execution gates in Harbor.


## Recipes

Older recipes, kept for reference:

- [`recipes/jev`](recipes/jev): train a small decision model on your labeled traces
- [`recipes/data/sft`](recipes/data/sft): generate new SFT traces from a few good seed traces
