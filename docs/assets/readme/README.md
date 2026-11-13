# README artwork

The opening follows four stages:

1. **Traces → Harbor task:** the actual package-writing contract.
2. **Model attempt:** an agent reads the instruction, uses the environment, and submits output.
3. **Verifier grade:** the task verifier checks the output and produces a reward.
4. **New traces:** the attempt's `agent/trajectory.json` and `result.json` can be used in a later conversion.

The first stage uses the package produced from the
[net-revenue example](../../../examples/mint/net-revenue). The renderer asserts
the displayed paths exist. Numeric rewards are read from that example's actual
local gate results and saved in `mint-loop-evidence.json`. The model-attempt
segment is schematic: no live model solve is performed or claimed by the renderer.
The measured reward is from the local reference/wrong-answer gates.

The return path means new run traces can feed another conversion. It does not
mean Mint automatically starts a training job or changes model weights. Harbor
executes the downstream agent; Mint constructs and checks the task package.

The loop preserves the monochrome typography and blue accent. Movement and
crossfades use 25 fps timing, with reading holds. GIFs share one palette to avoid
color flicker. Static and reduced-motion versions retain the conversion scene
and show all four stages in the navigation rail.

This is a designed animation of the included synthetic fixture, not a recording
of conversion speed. Local verifier passes remain generated; no real Harbor run
or live-model result is implied by the numeric example.

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

- `mint-workflow-{dark,light}.gif`: the README loop, with a shared palette to avoid flicker.
- `mint-workflow-{dark,light}.webp`: lossless versions of the same animation.
- `mint-workflow-{dark,light}.png`: static alternatives.
- `mint-checks-{dark,light}.png`: actual local CLI output excerpt.

The README offers still-image links and reduced-motion sources. Long descriptions,
commands, and results remain available as text. Design references and rationale
are recorded in [the README design notes](../../plans/readme-design.md).
