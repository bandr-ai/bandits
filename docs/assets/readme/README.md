# README artwork

The opening follows [one failed net-revenue job](../../../examples/mint/net-revenue).
A 120-dollar sale and a -20-dollar refund produced a logged answer of 140. The
rebuilt task accepts 100 and rejects 140. The loop shows the failure, the restored
job, and the outcome check. Only the bandr name appears in the opening artwork.

This is a designed animation of a synthetic fixture, not a recording of a model
solve or a conversion-speed claim. The renderer runs the actual task generator
and local verifier gates before producing the assets. Light and dark variants
share the story; the static alternative shows the final outcome.

The CLI panel is an excerpt captured by actually running the supplied parser,
generator, local validation, and report in a temporary workspace. Its text is
also saved in [mint-checks.txt](mint-checks.txt). The build asserts that two tasks
pass local execution gates and remain generated. It does not run Harbor.

Rebuild from the repository root:

```bash
uv sync --extra dev
uv run --with pillow python scripts/render_readme.py
```

The renderer uses installed Helvetica/Menlo fonts on macOS or DejaVu fonts on
Linux. Supply `--sans-font /path/to/font.ttf --mono-font /path/to/mono.ttf` to
choose other fonts. Font files are not bundled. Pillow is needed only to rebuild
these assets, not to use Bandr.

Assets:

- `mint-workflow-{dark,light}.webp`: looping workflow illustration.
- `mint-workflow-{dark,light}.png`: static alternatives.
- `mint-checks-{dark,light}.png`: actual local CLI output excerpt.

The README offers still-image links and reduced-motion sources. Long descriptions,
commands, and results remain available as text. Design references and rationale
are recorded in [the README design notes](../../plans/readme-design.md).
