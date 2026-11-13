"""Render the bandr README animation and captured example output.

Run: uv run --with pillow python scripts/render_readme.py
Font paths can be supplied with --sans-font and --mono-font. No runtime dependency.
The hook displays actual materializer outputs from the bundled net-revenue fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/readme"
SIZE = (1440, 760)
THEMES = {
    "dark": {
        "bg": "#0A0A0B",
        "panel": "#101012",
        "ink": "#F5F5F7",
        "muted": "#99999F",
        "line": "#2B2B31",
        "accent": "#8C9DFF",
    },
    "light": {
        "bg": "#FAFAFC",
        "panel": "#FFFFFF",
        "ink": "#161619",
        "muted": "#67676F",
        "line": "#DCDCE3",
        "accent": "#3448D5",
    },
}


def font_path(supplied: str | None, mono: bool) -> str:
    choices = [supplied] if supplied else []
    choices += (
        ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
        if mono
        else [
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for name in choices:
        if name and Path(name).is_file():
            return name
    raise SystemExit("Supply --sans-font and --mono-font pointing to installed font files.")


@lru_cache(maxsize=64)
def face(path: str, size: int, bold: bool = False):
    try:
        return ImageFont.truetype(
            path, size, index=1 if bold and path.endswith("HelveticaNeue.ttc") else 0
        )
    except OSError:
        return ImageFont.truetype(path, size)


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def render(theme: dict, seconds: float, sans: str, mono: str) -> Image.Image:
    """A source record becomes the actual Harbor package, with continuous eased motion."""
    from PIL import ImageColor

    im = Image.new("RGB", SIZE, theme["bg"])
    d = ImageDraw.Draw(im)

    def color(name, alpha=1.0):
        foreground, background = ImageColor.getrgb(theme[name]), ImageColor.getrgb(theme["bg"])
        return tuple(
            round(a * alpha + b * (1 - alpha)) for a, b in zip(foreground, background, strict=True)
        )

    def text(x, y, value, size=25, tone="ink", code=False, alpha=1.0, bold=False, right=False):
        d.text(
            (round(x), round(y)),
            value,
            font=face(mono if code else sans, size, bold),
            fill=color(tone, alpha),
            anchor="rt" if right else "lt",
        )

    # Editorial typography and open space, rather than a dashboard of boxes.
    d.rectangle((64, 48, 73, 57), fill=theme["accent"])
    text(88, 39, "bandr", 31, bold=True)
    text(64, 146, "Agent logs in.", 77, bold=True)
    text(64, 237, "Harbor tasks out.", 77, bold=True)
    text(68, 350, "Your tasks, reconstructed with", 28, "muted")
    text(68, 389, "instructions, environments and verifiers.", 28, "muted")

    d.line((68, 466, 616, 466), fill=theme["line"], width=1)
    text(68, 490, "attempts.jsonl", 20, "muted", code=True)
    text(68, 535, 'request   "Calculate net revenue..."', 21, code=True)
    text(68, 575, "input     sales.csv", 21, code=True)
    text(68, 615, "source    attempts.jsonl#0", 21, "muted", code=True)

    # One document, with the actual paths materialize() writes. No invented commands.
    progress = ease(seconds / 3.9)
    text(824, 140, "HARBOR TASK", 18, "muted", code=True)
    text(824, 178, "net-revenue/", 37, bold=True)
    d.line((824, 240, 1376, 240), fill=theme["line"], width=1)
    rows = [
        ("instruction.md", "the original job"),
        ("environment/", "starting state"),
        ("tests/test.sh", "outcome verifier"),
        ("solution/solve.sh", "reference answer"),
        ("task.toml", "runtime + provenance"),
    ]
    for index, (path, description) in enumerate(rows):
        amount = ease((seconds - 0.35 - index * 0.56) / 0.65)
        y = 279 + index * 72 + (1 - amount) * 13
        opacity = 0.22 + 0.78 * amount
        # Small file glyph; no faux application chrome.
        x = 827
        d.line(
            (x, y + 1, x + 12, y + 1, x + 19, y + 8, x + 19, y + 24, x, y + 24, x, y + 1),
            fill=color("muted", opacity),
            width=1,
        )
        text(863, y, path, 25, code=True, alpha=opacity)
        text(863, y + 35, description, 19, "muted", alpha=opacity)

    # A thin continuous connection, with a packet that eases into the package.
    curve = []
    for step in range(81):
        t = step / 80
        x = (1 - t) ** 3 * 630 + 3 * (1 - t) ** 2 * t * 724 + 3 * (1 - t) * t * t * 720 + t**3 * 794
        y = (1 - t) ** 3 * 564 + 3 * (1 - t) ** 2 * t * 564 + 3 * (1 - t) * t * t * 421 + t**3 * 421
        curve.append((x, y))
    d.line(curve, fill=theme["line"], width=2)
    if seconds < 3.9:
        amount = ease((seconds % 0.78) / 0.78)
        index = min(79, int(amount * 80))
        x, y = curve[index]
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=theme["accent"])
    d.line((785, 414, 794, 421, 785, 428), fill=theme["line"], width=2)

    d.line((824, 667, 1376, 667), fill=theme["line"], width=2)
    d.line((824, 667, 824 + round(552 * progress), 667), fill=theme["accent"], width=2)
    text(64, 710, "The whole job. In Harbor format.", 22, "muted")
    text(1376, 710, "Built for evals and training.", 22, "muted", right=True)
    return im


def animation(theme, sans, mono):
    """25 fps while moving, followed by a deliberate reading hold and a smooth loop."""
    frames = [render(theme, frame / 25, sans, mono) for frame in range(101)]
    durations = [40] * len(frames)
    durations[0] = 900
    durations[-1] = 2600
    # Crossfade to the opening; no sudden reset and no blank frame.
    for index in range(1, 16):
        frames.append(Image.blend(frames[100], frames[0], ease(index / 15)))
        durations.append(40)
    return frames, durations


def capture_example(
    example: Path | None = None, expected_count: int = 2
) -> tuple[list[str], dict[str, int]]:
    """Run the shipped example; fail the build if the illustrated result stops being true."""
    example = example or ROOT / "examples/mint"
    with tempfile.TemporaryDirectory(prefix="mint-readme-") as temporary:
        workspace = Path(temporary) / "workspace"
        environment = {**os.environ, "MINT_TELEMETRY_DISABLED": "1"}

        def run(*args, cwd=ROOT):
            return subprocess.run(
                [sys.executable, "-m", "mint", *args],
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        run("init", str(example / "logs"), str(workspace), "--allow-missing-runtime")
        for name in ("parse.py", "jobs.json", "source_notes.md"):
            shutil.copy(example / name, workspace / name)
        (workspace / "tasktypes").mkdir(exist_ok=True)
        for tasktype in (example / "tasktypes").glob("*.py"):
            shutil.copy(tasktype, workspace / "tasktypes" / tasktype.name)
        run("run", "parse", cwd=workspace)
        run("run", "generate", cwd=workspace)
        output = run("validate", "--runner", "local", cwd=workspace)
        run("report", cwd=workspace)
        rows = [
            json.loads(line) for line in (workspace / "out/results.jsonl").read_text().splitlines()
        ]
        assert len(rows) == expected_count
        for row in rows:
            task = workspace / "out/tasks" / row["type"] / row["task_id"]
            for name in (
                "instruction.md",
                "environment",
                "tests/test.sh",
                "solution/solve.sh",
                "task.toml",
            ):
                assert (task / name).exists(), name
        assert all(row["status"] == "generated" and row["gates"]["exec"] == "pass" for row in rows)
        excerpt = [
            line
            for line in output.splitlines()
            if line.startswith(("runner:", "level 1:", "level 2 "))
        ]
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in ("generated", "validated")
        }
        return excerpt, counts


def render_checks(theme: dict, mono: str, lines: list[str], counts: dict) -> Image.Image:
    im = Image.new("RGB", (1600, 340), theme["bg"])
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle(
        (32, 24, 1568, 315), radius=14, fill=theme["panel"], outline=theme["line"], width=2
    )

    def text(x, y, value, size, color):
        draw.text(
            (x, y), value, font=ImageFont.truetype(mono, size), fill=theme[color], anchor="lt"
        )

    text(62, 49, "Local verifier checks", 18, "muted")
    text(62, 91, "$ mint validate --runner local", 26, "accent")
    for index, value in enumerate(lines):
        text(62, 146 + index * 36, value, 23, "ink")
    text(
        62,
        275,
        f"Task statuses: generated={counts['generated']}  validated={counts['validated']}",
        21,
        "muted",
    )
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sans-font")
    parser.add_argument("--mono-font")
    args = parser.parse_args()
    sans, mono = font_path(args.sans_font, False), font_path(args.mono_font, True)
    hero_example = ROOT / "examples/mint/net-revenue"
    capture_example(hero_example, expected_count=1)
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, counts = capture_example()
    (OUT / "mint-checks.txt").write_text(
        "$ mint validate --runner local\n" + "\n".join(transcript) + "\n"
    )
    for name, theme in THEMES.items():
        frames, durations = animation(theme, sans, mono)
        poster = frames[100]
        poster.save(OUT / f"mint-workflow-{name}.png", optimize=True)
        frames[0].save(
            OUT / f"mint-workflow-{name}.webp",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            lossless=True,
            method=4,
        )
        # An actual GIF is also provided; a shared palette prevents color flicker.
        palette = poster.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        gif = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        gif[0].save(
            OUT / f"mint-workflow-{name}.gif",
            save_all=True,
            append_images=gif[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
        del frames, gif
        print(f"Rendered {name}: 25 fps transitions, WebP + GIF + static poster.")
        render_checks(theme, mono, transcript, counts).save(
            OUT / f"mint-checks-{name}.png", optimize=True
        )


if __name__ == "__main__":
    main()
