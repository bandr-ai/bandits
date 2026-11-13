# README artwork

The opening shows the jobs captured in agent logs becoming Harbor task packages.
It follows the source request and starting file from the bundled
[net-revenue example](../../../examples/mint/net-revenue) into the real files
written by `mint.harbor.materialize`: instruction, environment, verifier, optional
oracle, and runtime/provenance configuration.

The visual uses black/white typography, a restrained blue accent, and an open
layout. File rows fade and slide into place with eased transitions at **25 fps**.
It holds the completed package for reading, then crossfades back to the opening.
There are no low-frame-rate stepped position jumps.

This is a designed animation of the included synthetic fixture, not a recording
of conversion speed or a model solve. The renderer actually builds the example,
asserts that the displayed output files exist, and checks the local verifier
gates. Local passes remain generated; no Harbor execution is implied.

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
