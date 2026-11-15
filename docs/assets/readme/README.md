# README artwork

Two complementary panels keep task construction separate from downstream use.

1. **Agent logs → Mint + coding agent → Harbor tasks.** Mint constructs and checks
   packages; the coding agent supplies source-specific reconstruction code.
2. **Harbor task → model rollout → graded attempts.** Trajectories and outcome
   scores branch into evaluation, curated SFT, rejection sampling, RL/RFT, and
   preference data. Training needs an external pipeline and dataset adapters.

There is no verifier-to-traces return arrow. A verifier grades an attempt; it
neither reconstructs a task nor updates model weights. Animation restarts are
presentation repeats, not feedback loops.

Both panels preserve the original paper, folder, network, and check icons.
The network in the first panel represents the assisting coding agent, not a
trainer. Dots move forward along arrows; the downstream branches remain visible.
All rollout and training paths are schematic. The renderer performs no live
model run, training, or Harbor execution. Local fixture gate results are saved
in `mint-loop-evidence.json`; they do not substantiate downstream model results.

The CLI panel is an excerpt captured by actually running the supplied parser,
generator, local validation, and report in a temporary workspace. Its text is
also saved in [mint-checks.txt](mint-checks.txt). The build asserts that two tasks
pass local execution gates and remain generated. It does not run Harbor.

Rebuild from the repository root:

```bash
uv sync --extra dev
uv run --with pillow python scripts/render_readme.py
```

The renderer uses installed Helvetica Neue/Menlo fonts on macOS or DejaVu fonts on
Linux. Supply `--sans-font /path/to/font.ttf --mono-font /path/to/mono.ttf` to
choose other fonts. Font files are not bundled. Pillow is needed only to rebuild
these assets, not to use Bandr.

Assets:

- `mint-workflow-{dark,light}.gif`: the task-construction panel, with a shared palette to avoid flicker.
- `mint-workflow-{dark,light}.webp`: lossless versions of the same animation.
- `mint-workflow-{dark,light}.png`: static alternatives.
- `mint-uses-{dark,light}.{gif,webp,png}`: the downstream-use panel.
- `mint-checks-{dark,light}.png`: actual local CLI output excerpt.

The README offers still-image links and reduced-motion sources. Long descriptions,
commands, and results remain available as text. Design references and rationale
are recorded in [the README design notes](../../plans/readme-design.md).
