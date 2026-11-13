"""Render the README's illustrated Mint workflow (not a terminal recording).

Run: uv run --with pillow python scripts/render_readme.py
Font paths can be supplied with --sans-font and --mono-font. No runtime dependency.
The numbers and answers are read from the bundled synthetic fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/readme"
SIZE = (1600, 690)
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


def render(theme: dict, stage: int, pulse: int, sans: str, mono: str, records: list) -> Image.Image:
    im = Image.new("RGB", SIZE, theme["bg"])
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=24, color="ink", code=False, right=False):
        font = ImageFont.truetype(mono if code else sans, size)
        d.text((x, y), value, font=font, fill=theme[color], anchor="rt" if right else "lt")

    def line(coords, color="line", width=2):
        d.line(coords, fill=theme[color], width=width)

    def pill(x, y, value, width, color="muted", fill="panel", size=20):
        d.rounded_rectangle((x, y, x + width, y + 42), radius=8, fill=theme[fill])
        text(x + 14, y + 10, value, size, color, code=True)

    text(48, 33, "BANDR  /  MINT", 20, "accent", code=True)
    text(1552, 35, "ILLUSTRATED WORKFLOW", 18, "muted", code=True, right=True)
    text(48, 78, "From recorded attempts to reusable tasks.", 42)

    xs = [48, 580, 1112]
    labels = ["The source", "The reconstruction", "The output"]
    subs = ["01 / AGENT LOGS", "02 / CODING AGENT + MINT", "03 / HARBOR TASKS"]
    for i, x in enumerate(xs):
        active = min(stage, 2) == i and stage < 3
        d.rounded_rectangle(
            (x, 161, x + 440, 505),
            radius=15,
            fill=theme["panel"],
            outline=theme["accent" if active else "line"],
            width=2,
        )
        text(x + 24, 186, subs[i], 18, "accent" if active else "muted", code=True)
        text(x + 24, 222, labels[i], 30)
        line((x + 24, 267, x + 416, 267))
        if active:
            line((x + 24, 161, x + 130 + pulse * 42, 161), "accent", 4)

    for a in [506, 1038]:
        line((a, 325, a + 54, 325), "accent" if stage > (a > 600) else "line", 2)
        line((a + 47, 318, a + 54, 325, a + 47, 332), "accent", 2)

    # The actual fixture has repeat attempts with conflicting logged answers.
    a, retry, b = records
    text(72, 290, f"{a['instance']}   17 + 25", 24, code=True)
    text(88, 328, f"attempt 1  recorded: {a['answer']}", 22, "muted", code=True)
    text(88, 362, f"attempt 2  recorded: {retry['answer']}", 22, "muted", code=True)
    text(72, 411, f"{b['instance']}    3 + 4", 24, code=True)
    text(88, 449, f"attempt 1  recorded: {b['answer']}", 22, "muted", code=True)

    middle = [
        "Read the starting evidence",
        "Group repeat attempts",
        "Build the whole job",
        "Recompute expected answers",
    ]
    for i, value in enumerate(middle):
        y = 291 + i * 47
        text(604, y, f"{i + 1:02}", 19, "accent", code=True)
        text(647, y, value, 23)

    text(1136, 291, "sum-a/", 27, code=True)
    text(1136, 332, "sum-b/", 27, code=True)
    text(1136, 384, "instruction  /  environment", 20, "muted", code=True)
    text(1136, 418, "verifier  /  oracle", 20, "muted", code=True)
    pill(1136, 447, "2 tasks / generated", 300, "accent", "soft")

    text(48, 524, "LOCAL VERIFIER GATES", 18, "muted", code=True)
    checks = [
        ("oracle", "accepted"),
        ("no-op", "rejected"),
        ("wrong answer", "rejected"),
        ("alternative", "accepted"),
    ]
    for i, (label, outcome) in enumerate(checks):
        x = 48 + i * 382
        d.rounded_rectangle(
            (x, 561, x + 358, 611),
            radius=9,
            fill=theme["panel"],
            outline=theme["pass" if stage == 3 else "line"],
            width=1,
        )
        text(x + 15, 579, label, 19, "muted", code=True)
        text(
            x + 343,
            579,
            outcome if stage == 3 else "pending",
            19,
            "pass" if stage == 3 else "muted",
            code=True,
            right=True,
        )
    notes = [
        "01  Start with what was requested, the starting files, and every attempt.",
        "02  Rebuild the job. A logged answer is evidence, not ground truth.",
        "03  Package the instruction, environment, verifier, and answer oracle.",
        "04  Local checks pass. Harbor execution is still needed for validated.",
    ]
    text(48, 645, notes[stage], 21, "muted")
    return im


def capture_example() -> tuple[list[str], dict[str, int]]:
    """Run the shipped example; fail the build if the illustrated result stops being true."""
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

        run("init", str(ROOT / "examples/mint/logs"), str(workspace), "--allow-missing-runtime")
        for name in ("parse.py", "jobs.json", "source_notes.md"):
            shutil.copy(ROOT / "examples/mint" / name, workspace / name)
        (workspace / "tasktypes").mkdir(exist_ok=True)
        shutil.copy(ROOT / "examples/mint/tasktypes/sum.py", workspace / "tasktypes/sum.py")
        run("run", "parse", cwd=workspace)
        run("run", "generate", cwd=workspace)
        output = run("validate", "--runner", "local", cwd=workspace)
        run("report", cwd=workspace)
        rows = [
            json.loads(line) for line in (workspace / "out/results.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
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

    text(62, 49, "ACTUAL CLI OUTPUT / BUNDLED EXAMPLE / EXCERPT", 18, "muted")
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
    records = [
        json.loads(line)
        for line in (ROOT / "examples/mint/logs/attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3 and len({r["instance"] for r in records}) == 2
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, counts = capture_example()
    (OUT / "mint-checks.txt").write_text(
        "$ mint validate --runner local\n" + "\n".join(transcript) + "\n"
    )
    for name, theme in THEMES.items():
        frames = [
            render(theme, stage, pulse, sans, mono, records)
            for stage in range(4)
            for pulse in range(4)
        ]
        frames[-1].save(OUT / f"mint-workflow-{name}.png", optimize=True)
        frames[0].save(
            OUT / f"mint-workflow-{name}.webp",
            save_all=True,
            append_images=frames[1:],
            duration=[450] * 15 + [3600],
            loop=0,
            lossless=True,
            method=6,
        )
        print(f"Rendered {name} workflow: 16 frames, illustrated synthetic example.")
        render_checks(theme, mono, transcript, counts).save(
            OUT / f"mint-checks-{name}.png", optimize=True
        )


if __name__ == "__main__":
    main()
