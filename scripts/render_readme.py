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


def render(
    theme: dict, seconds: float, sans: str, mono: str, phase: int = 0, evidence: dict | None = None
) -> Image.Image:
    """Conversion, downstream model attempt, grading, and the next trace."""
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
    evidence = evidence or {"answer": "100", "reward": 1.0, "wrong": "140", "wrong_reward": 0.0}
    headings = [
        ("Agent logs in.", "Harbor tasks out."),
        ("A Harbor task.", "A model attempts it."),
        ("Model output.", "Verifier grade."),
        ("A new attempt.", "A new trace."),
    ]
    subtitles = [
        ("Your tasks, reconstructed with", "instructions, environments and verifiers."),
        ("The agent gets the original job", "and works in the task environment."),
        ("The task's verifier checks the outcome", "and assigns a reward."),
        ("The run records what the model did.", "Use its traces for the next conversion."),
    ]
    for y, value in zip((146, 237), headings[phase], strict=True):
        text(64, y, value, 65 if phase == 1 else 77, bold=True)
    for y, value in zip((350, 389), subtitles[phase], strict=True):
        text(68, y, value, 28, "muted")

    previews = [
        (
            "attempts.jsonl",
            'request   "Calculate net revenue..."',
            "input     sales.csv",
            "source    attempts.jsonl#0",
        ),
        (
            "net-revenue/",
            "job       Calculate net revenue",
            "files     sales.csv",
            "runtime   Harbor",
        ),
        (
            "answer.txt",
            f"answer    {evidence['answer']}",
            "checked by the task's verifier",
            "tests/test.sh",
        ),
        (
            "agent/trajectory.json",
            "request   the original job",
            "steps     the model's actions",
            "result    verifier reward",
        ),
    ]
    d.line((68, 466, 616, 466), fill=theme["line"], width=1)
    for index, value in enumerate(previews[phase]):
        text(
            68,
            (490, 535, 575, 615)[index],
            value,
            20 if index == 0 else 21,
            "muted" if index in (0, 3) else "ink",
            code=True,
        )

    if phase == 0:
        # One document, with the actual paths materialize() writes. No invented commands.
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

    elif phase == 1:
        text(824, 140, "MODEL ATTEMPT", 18, "muted", code=True)
        text(824, 178, "A fresh agent run", 37, bold=True)
        d.line((824, 240, 1376, 240), fill=theme["line"], width=1)
        actions = [
            ("Read the instruction", "instruction.md"),
            ("Work in the environment", "sales.csv + tools"),
            ("Submit an answer", f"answer.txt  →  {evidence['answer']}"),
        ]
        for index, (label, detail) in enumerate(actions):
            y = 287 + index * 105
            text(824, y, label, 28)
            text(824, y + 43, detail, 23, "muted", code=True)
    elif phase == 2:
        text(824, 140, "TASK VERIFIER", 18, "muted", code=True)
        text(824, 178, "tests/test.sh", 37, bold=True)
        d.line((824, 240, 1376, 240), fill=theme["line"], width=1)
        text(824, 285, "answer", 22, "muted", code=True)
        text(1130, 285, "reward", 22, "muted", code=True)
        text(824, 332, evidence["answer"], 80, bold=True)
        text(1130, 332, f"{evidence['reward']:.1f}", 80, "accent", bold=True)
        text(824, 456, "Expected outcome matched.", 29)
        text(
            824,
            517,
            f"Wrong answer {evidence['wrong']} → reward {evidence['wrong_reward']:.1f}",
            23,
            "muted",
            code=True,
        )
        text(824, 584, "The verifier grades the result.", 24, "muted")
    else:
        text(824, 140, "RUN OUTPUT", 18, "muted", code=True)
        text(824, 178, "New attempt traces", 37, bold=True)
        d.line((824, 240, 1376, 240), fill=theme["line"], width=1)
        text(824, 295, "agent/trajectory.json", 27, code=True)
        text(824, 338, "The model's steps and observations", 24, "muted")
        text(824, 415, "result.json", 27, code=True)
        text(824, 458, "The verifier's result", 24, "muted")
        text(824, 564, "Traces for the next conversion.", 28)

    # A thin continuous connection, with a packet that eases into the package.
    curve = []
    for step in range(81):
        t = step / 80
        x = (1 - t) ** 3 * 630 + 3 * (1 - t) ** 2 * t * 724 + 3 * (1 - t) * t * t * 720 + t**3 * 794
        y = (1 - t) ** 3 * 564 + 3 * (1 - t) ** 2 * t * 564 + 3 * (1 - t) * t * t * 421 + t**3 * 421
        curve.append((x, y))
    d.line(curve, fill=theme["line"], width=2)
    if phase == 0 and seconds < 3.9 or phase == 3:
        amount = ease(seconds / 1.2) if phase == 3 else ease((seconds % 0.78) / 0.78)
        index = min(79, int(amount * 80))
        if phase == 3:
            index = 79 - index
        x, y = curve[index]
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=theme["accent"])
    arrow = (639, 557, 630, 564, 639, 571) if phase == 3 else (785, 414, 794, 421, 785, 428)
    d.line(arrow, fill=theme["line"], width=2)

    labels = ["01  Traces to task", "02  Model attempt", "03  Verifier grade", "04  New traces"]
    for index, label in enumerate(labels):
        x = 64 + index * 336
        d.line((x, 686, x + 300, 686), fill=theme["accent" if index == phase else "line"], width=2)
        text(x, 710, label, 20, "ink" if index == phase else "muted")
    return im


def animation(theme, sans, mono, evidence):
    """Four readable scenes, 25 fps eased transitions, and a returning trace."""
    frames = [render(theme, frame / 25, sans, mono, evidence=evidence) for frame in range(101)]
    durations = [40] * len(frames)
    durations[0], durations[-1] = 700, 1000

    def transition(target):
        previous = frames[-1]
        for index in range(1, 16):
            frames.append(Image.blend(previous, target, ease(index / 15)))
            durations.append(40)

    for phase in (1, 2, 3):
        target = render(theme, 0, sans, mono, phase, evidence)
        transition(target)
        durations[-1] = 2100 if phase < 3 else 700
        if phase == 3:
            for index in range(1, 33):
                frames.append(render(theme, index / 25, sans, mono, phase, evidence))
                durations.append(40)
            durations[-1] = 1000
    transition(frames[0])
    return frames, durations


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
