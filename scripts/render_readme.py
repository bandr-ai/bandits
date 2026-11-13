"""Render the bandr README animation and captured example output.

Run: uv run --with pillow python scripts/render_readme.py
Font paths can be supplied with --sans-font and --mono-font. No runtime dependency.
The hook uses the bundled net-revenue fixture and verifies it with the local runner.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/readme"
SIZE = (1600, 770)
THEMES = {
    "dark": {
        "bg": "#0D1117",
        "panel": "#151B23",
        "ink": "#F1EEE8",
        "muted": "#9DA7B3",
        "line": "#303945",
        "accent": "#FF693C",
        "soft": "#2B211F",
        "pass": "#9BD6B5",
    },
    "light": {
        "bg": "#F6F5F0",
        "panel": "#FFFFFF",
        "ink": "#202832",
        "muted": "#616E7C",
        "line": "#D9DDD9",
        "accent": "#C4411B",
        "soft": "#FBEBE4",
        "pass": "#286447",
    },
}


def font_path(supplied: str | None, mono: bool) -> str:
    choices = [supplied] if supplied else []
    choices += (
        ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
        if mono
        else [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for name in choices:
        if name and Path(name).is_file():
            return name
    raise SystemExit("Supply --sans-font and --mono-font pointing to installed font files.")


def render(theme: dict, stage: int, pulse: int, sans: str, mono: str, record: dict) -> Image.Image:
    """Show the failed job becoming a reusable eval, one idea at a time."""
    im = Image.new("RGB", SIZE, theme["bg"])
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=28, color="ink", code=False, right=False):
        font = ImageFont.truetype(mono if code else sans, size)
        d.text((x, y), value, font=font, fill=theme[color], anchor="rt" if right else "lt")

    def line(points, color="line", width=2):
        d.line(points, fill=theme[color], width=width)

    text(60, 35, "bandr", 34, "accent")
    text(60, 100, "A failed run becomes a test for the next agent.", 51)
    text(
        60,
        169,
        "The same job. The original starting data. A verifier for the outcome.",
        27,
        "muted",
    )

    for x, active in ((60, stage == 0), (900, stage > 0)):
        d.rounded_rectangle(
            (x, 239, x + 640, 616),
            radius=16,
            fill=theme["panel"],
            outline=theme["accent" if active else "line"],
            width=2,
        )
    text(92, 267, "THE FAILED RUN", 20, "muted", code=True)
    text(932, 267, "THE REUSABLE EVAL", 20, "muted", code=True)
    text(92, 308, "Calculate net revenue.", 32)
    text(932, 308, "Calculate net revenue.", 32)

    text(92, 367, "sale", 28, "muted", code=True)
    text(666, 363, f"+${record['_sale']}", 35, code=True, right=True)
    text(92, 412, "refund", 28, "muted", code=True)
    text(666, 408, f"-${abs(record['_refund'])}", 35, code=True, right=True)
    line((92, 465, 668, 465))
    text(92, 489, "Agent answer", 23, "muted")
    text(92, 530, f"${record['answer']}", 53, "accent")
    text(270, 548, "Refund counted as income.", 23, "accent")

    # Motion carries the job from its log to the task without implying an elapsed runtime.
    line((741, 430, 858, 430), "accent", 3)
    line((847, 419, 858, 430, 847, 441), "accent", 3)
    position = 746 + pulse * 17
    d.ellipse((position - 4, 426, position + 4, 434), fill=theme["ink"])

    if stage == 0:
        text(932, 367, "sales.csv", 29, code=True)
        text(932, 414, "Original starting data", 25, "muted")
        line((932, 465, 1508, 465))
        text(932, 498, "A new agent can try the job.", 27)
        text(932, 546, "A verifier checks its answer.", 25, "muted")
    elif stage == 1:
        text(932, 367, "sales.csv", 27, code=True)
        text(1508, 367, "restored", 24, "accent", code=True, right=True)
        text(932, 414, "instruction.md", 27, code=True)
        text(1508, 414, "same job", 24, "accent", code=True, right=True)
        line((932, 465, 1508, 465))
        text(932, 498, "tests/test.sh", 27, code=True)
        text(932, 546, "Check that refunds reduce revenue.", 24, "muted")
    else:
        text(932, 370, f"${record['_expected']}", 54, "pass")
        text(1508, 388, "PASS", 27, "pass", code=True, right=True)
        text(932, 446, f"${record['answer']}", 54, "accent")
        text(1508, 464, "FAIL", 27, "accent", code=True, right=True)
        line((932, 523, 1508, 523))
        text(932, 555, "Let the next agent find its own solution.", 23, "muted")

    line((60, 658, 1540, 658))
    notes = [
        "A log records the failure. bandr makes it repeatable.",
        "Restore the starting data. Build a test for the right outcome.",
        "Use the task to evaluate agents or run training rollouts.",
    ]
    text(60, 695, notes[stage], 29)
    for step in range(3):
        x = 1372 + step * 60
        line((x, 708, x + 38, 708), "accent" if step == stage else "line", 4)
    return im


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
    record = json.loads((hero_example / "logs/attempts.jsonl").read_text())
    entries = list(csv.DictReader(io.StringIO(record["starting_file"])))
    record["_sale"] = sum(Decimal(row["amount"]) for row in entries if row["kind"] == "sale")
    record["_refund"] = sum(Decimal(row["amount"]) for row in entries if row["kind"] == "refund")
    record["_expected"] = record["_sale"] + record["_refund"]
    assert Decimal(record["answer"]) == record["_sale"] + abs(record["_refund"])
    assert record["_expected"] != Decimal(record["answer"])
    capture_example(hero_example, expected_count=1)
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, counts = capture_example()
    (OUT / "mint-checks.txt").write_text(
        "$ mint validate --runner local\n" + "\n".join(transcript) + "\n"
    )
    for name, theme in THEMES.items():
        frames = [
            render(theme, stage, pulse, sans, mono, record)
            for stage in range(3)
            for pulse in range(6)
        ]
        frames[-1].save(OUT / f"mint-workflow-{name}.png", optimize=True)
        frames[0].save(
            OUT / f"mint-workflow-{name}.webp",
            save_all=True,
            append_images=frames[1:],
            duration=[450] * 17 + [3600],
            loop=0,
            lossless=True,
            method=6,
        )
        print(f"Rendered {name} workflow: 18 source frames, net-revenue example.")
        render_checks(theme, mono, transcript, counts).save(
            OUT / f"mint-checks-{name}.png", optimize=True
        )


if __name__ == "__main__":
    main()
