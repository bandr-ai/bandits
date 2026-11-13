# README artwork

The workflow animation is a technical illustration of the bundled synthetic
example, not a terminal recording or a claim about conversion speed. Its first
frame contains the complete workflow; the animation highlights its stages.
Light and dark versions use the same content.

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
