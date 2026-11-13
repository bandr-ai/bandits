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
SIZE = (1440, 530)
THEMES = {
    "dark": {
        "bg": "#0A0A0B",
        "panel": "#101012",
        "ink": "#F5F5F7",
        "muted": "#99999F",
        "line": "#44444C",
        "accent": "#8C9DFF",
    },
    "light": {
        "bg": "#FAFAFC",
        "panel": "#FFFFFF",
        "ink": "#161619",
        "muted": "#67676F",
        "line": "#C6C6CF",
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
    """A visual loop: trace sheets, task folder, model network, verifier check."""
    import math

    from PIL import ImageColor

    im = Image.new("RGB", SIZE, theme["bg"])
    d = ImageDraw.Draw(im)
    phase = int(seconds / 1.8) % 4
    fraction = (seconds % 1.8) / 1.8
    centers = (200, 536, 872, 1208)

    def color(name, opacity=1.0):
        foreground, background = ImageColor.getrgb(theme[name]), ImageColor.getrgb(theme["bg"])
        return tuple(
            round(a * opacity + b * (1 - opacity))
            for a, b in zip(foreground, background, strict=True)
        )

    def line(points, tone="ink", width=3, opacity=1.0):
        d.line(points, fill=color(tone, opacity), width=width, joint="curve")

    def text(x, y, value, size, centered=False):
        d.text(
            (x, y),
            value,
            font=face(sans, size, True),
            fill=theme["ink"],
            anchor="mt" if centered else "lt",
        )

    def paper(x, y, width=94, height=122, tone="ink", opacity=1.0, contents=True):
        # Folded paper, not a text panel. Payloads use the same silhouette.
        fold = max(5, round(width * 0.23))
        points = [
            (x, y),
            (x + width - fold, y),
            (x + width, y + fold),
            (x + width, y + height),
            (x, y + height),
            (x, y),
        ]
        d.polygon(points, fill=theme["bg"])
        line(points, tone, 2 if width < 40 else 3, opacity)
        line(
            [(x + width - fold, y), (x + width - fold, y + fold), (x + width, y + fold)],
            tone,
            2,
            opacity,
        )
        if contents:
            for index, length in enumerate((0.60, 0.72, 0.43)):
                yy = y + height * 0.43 + index * height * 0.16
                line([(x + width * 0.17, yy), (x + width * length, yy)], "muted", 2, opacity)

    d.rectangle((64, 44, 72, 52), fill=theme["accent"])
    text(88, 35, "bandr", 31)

    # Traces: a stack of records, with one record in focus.
    paper(142, 155, 94, 122, "muted", 0.55)
    paper(153, 166, 94, 122, "muted", 0.75)
    paper(164, 177, 94, 122)
    if phase == 0:
        yy = 177 + 122 * 0.43
        line([(180, yy), (225, yy)], "accent", 3, 0.55 + 0.45 * math.sin(math.pi * fraction))

    # Harbor package: a folder carrying the task's instruction and assets.
    paper(485, 155, 62, 85, "muted", contents=False)
    paper(517, 163, 62, 85, "ink", contents=False)
    folder = [
        (448, 186),
        (448, 170),
        (510, 170),
        (529, 188),
        (624, 188),
        (624, 294),
        (448, 294),
        (448, 186),
    ]
    d.polygon(folder, fill=theme["bg"])
    line(folder, "accent" if phase == 1 else "ink")
    line([(448, 207), (624, 207)], "muted", 2)
    # A small package tab ties the contents together without listing filenames.
    line([(519, 246), (553, 246)], "muted", 3)

    # Model: simple connected layers, with an activation travelling through them.
    left = [(824, 196), (824, 260)]
    middle = [(872, 178), (872, 228), (872, 278)]
    right = [(920, 196), (920, 260)]
    for a in left:
        for b in middle:
            line([a, b], "line", 2)
    for a in middle:
        for b in right:
            line([a, b], "line", 2)
    for layer, nodes in enumerate((left, middle, right)):
        active = phase == 2 and abs(fraction * 3 - (layer + 0.5)) < 0.65
        for x, y in nodes:
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=theme["accent" if active else "ink"])

    # Verifier: the task's outcome check, rather than another screen of prose.
    d.ellipse((1144, 164, 1272, 292), outline=theme["line"], width=3)
    check = [(1178, 228), (1200, 250), (1240, 204)]
    line(check, "accent" if phase == 3 else "ink", 5)
    if phase == 3:
        d.arc(
            (1144, 164, 1272, 292),
            start=-90,
            end=-90 + max(1, round(360 * ease(fraction))),
            fill=theme["accent"],
            width=3,
        )

    for x, title in zip(centers, ("Traces", "Harbor task", "Model", "Verifier"), strict=True):
        text(x, 341, title, 32, centered=True)

    # Arrows sit in the gaps. The document itself moves continuously through
    # every stage, then returns along the lower path to the trace stack.
    for start, end in ((270, 436), (636, 800), (940, 1130)):
        points = [(start, 228), (end, 228)]
        line(points, "line", 2)
        line([(end - 8, 222), (end, 228), (end - 8, 234)], "line", 2)

    def curve(points, a, b, c, e, samples=40):
        for step in range(samples + 1):
            t = step / samples
            points.append(
                tuple(
                    (1 - t) ** 3 * a[k]
                    + 3 * (1 - t) ** 2 * t * b[k]
                    + 3 * (1 - t) * t * t * c[k]
                    + t**3 * e[k]
                    for k in (0, 1)
                )
            )

    # Leave the check on its right edge, then run below the labels.
    path = []
    curve(path, (1208, 228), (1248, 228), (1272, 248), (1296, 292), 14)
    visible = []
    curve(visible, (1296, 292), (1340, 360), (1324, 449), (1164, 449))
    visible.extend((1164 - step * (920 / 160), 449) for step in range(161))
    curve(visible, (244, 449), (200, 449), (200, 420), (200, 395))
    line(visible, "line", 2)
    line([(194, 403), (200, 395), (206, 403)], "line", 2)
    path.extend(visible)
    if phase < 3:
        progress = (phase + ease(fraction)) / 3
        x = 250 + (1208 - 250) * progress
        y = 228
    else:
        x, y = path[min(len(path) - 1, round(ease(fraction) * (len(path) - 1)))]
    paper(round(x) - 9, round(y) - 12, 18, 24, "accent", contents=False)
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
