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


def render(theme: dict, seconds: float, sans: str, mono: str, evidence: dict) -> Image.Image:
    """One persistent canvas: four visible stages and a returning trace."""
    im = Image.new("RGB", SIZE, theme["bg"])
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=25, tone="ink", code=False, bold=False, centered=False):
        d.text(
            (round(x), round(y)),
            value,
            font=face(mono if code else sans, size, bold),
            fill=theme[tone],
            anchor="mt" if centered else "lt",
        )

    def arrow(points, upward=False):
        d.line(points, fill=theme["line"], width=2)
        x, y = points[-1]
        head = (x - 6, y + 8, x, y, x + 6, y + 8) if upward else (x - 8, y - 6, x, y, x - 8, y + 6)
        d.line(head, fill=theme["line"], width=2)

    d.rectangle((64, 48, 73, 57), fill=theme["accent"])
    text(88, 39, "bandr", 31, bold=True)
    text(64, 130, "Build the task. Run the model. Grade the result.", 54, bold=True)
    text(
        66,
        210,
        "Agent traces become Harbor tasks. Each new attempt leaves more traces.",
        27,
        "muted",
    )

    phase = int(seconds / 1.8) % 4
    fraction = (seconds % 1.8) / 1.8
    xs = (64, 400, 736, 1072)
    titles = ("Traces", "Harbor task", "Model attempt", "Verifier")
    for index, (x, title) in enumerate(zip(xs, titles, strict=True)):
        text(x, 308, f"0{index + 1}", 20, "accent" if phase == index else "muted", code=True)
        text(x, 350, title, 31, bold=True)
        d.line((x, 555, x + 272, 555), fill=theme["accent" if phase == index else "line"], width=2)

    # Keep all stage content visible. Motion belongs to the connections only.
    text(64, 419, "The original job", 24)
    text(64, 460, "Files + context", 24, "muted")
    text(64, 507, "attempts.jsonl", 21, "muted", code=True)

    for index, path in enumerate(("instruction.md", "environment/", "tests/test.sh", "task.toml")):
        text(400, 414 + index * 32, path, 21, code=True)

    text(736, 419, "Read. Act. Answer.", 24)
    text(736, 460, "In the task environment", 23, "muted")
    text(736, 507, f"answer.txt → {evidence['answer']}", 21, "muted", code=True)

    text(1072, 419, "tests/test.sh", 21, "muted", code=True)
    text(1072, 459, f"{evidence['reward']:.1f}", 64, "accent", bold=True)
    text(1190, 495, "reward", 23, "muted")

    connectors = []
    for start in (315, 651, 987):
        points = [(start, 368), (start + 65, 368)]
        arrow(points)
        connectors.append(points)

    # A single return line makes the loop visible even in a still frame.
    path = []

    def bezier(a, b, c, e, samples=40):
        for step in range(samples + 1):
            t = step / samples
            path.append(
                tuple(
                    (1 - t) ** 3 * a[k]
                    + 3 * (1 - t) ** 2 * t * b[k]
                    + 3 * (1 - t) * t * t * c[k]
                    + t**3 * e[k]
                    for k in (0, 1)
                )
            )

    bezier((1208, 575), (1208, 605), (1208, 615), (1168, 615))
    path.extend((1168 - step * (992 / 160), 615) for step in range(161))
    bezier((176, 615), (136, 615), (136, 605), (136, 575))
    arrow(path, upward=True)

    if phase < 3:
        start, end = connectors[phase]
        x = start[0] + (end[0] - start[0]) * ease(fraction)
        y = start[1]
    else:
        x, y = path[min(len(path) - 1, round(ease(fraction) * (len(path) - 1)))]
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=theme["accent"])
    text(720, 658, "New attempt traces feed the next conversion.", 24, "muted", centered=True)
    return im


def animation(theme, sans, mono, evidence):
    """25 fps loop, fixed layout, no scene cuts, fades or page changes."""
    frames = [render(theme, frame / 25, sans, mono, evidence) for frame in range(180)]
    return frames, [40] * len(frames)


def capture_example(
    example: Path | None = None, expected_count: int = 2
) -> tuple[list[str], dict[str, int], dict]:
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
        evidence = {}
        if expected_count == 1:
            row = rows[0]
            task = workspace / "out/tasks" / row["type"] / row["task_id"]
            expected = json.loads((task / "tests/expected.json").read_text())["expected"]
            gates = json.loads(
                (workspace / "out/.mint/gates" / f"{row['task_id']}.json").read_text()
            )
            detail = gates["detail"]
            wrong = detail["wrong"]["runs"][0]
            evidence = {
                "answer": str(expected),
                "reward": detail["oracle"]["reward"],
                "wrong": str(wrong["answer"]),
                "wrong_reward": wrong["reward"],
            }
            assert evidence["reward"] == 1.0 and evidence["wrong_reward"] == 0.0
        return excerpt, counts, evidence


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
    _, _, evidence = capture_example(hero_example, expected_count=1)
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, counts, _ = capture_example()
    (OUT / "mint-checks.txt").write_text(
        "$ mint validate --runner local\n" + "\n".join(transcript) + "\n"
    )
    (OUT / "mint-loop-evidence.json").write_text(
        json.dumps(
            {**evidence, "runner": "local", "model_attempt": "schematic; no live model run"},
            indent=2,
        )
        + "\n"
    )
    for name, theme in THEMES.items():
        frames, durations = animation(theme, sans, mono, evidence)
        poster = frames[155]
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
        print(f"Rendered {name}: 25 fps continuous flow, WebP + GIF + static poster.")
        render_checks(theme, mono, transcript, counts).save(
            OUT / f"mint-checks-{name}.png", optimize=True
        )


if __name__ == "__main__":
    main()
