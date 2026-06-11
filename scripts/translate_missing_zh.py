#!/usr/bin/env python3
"""Run batch Chinese translation only for lessons missing docs or quiz.

This script is a thin orchestrator around :mod:`translate_curriculum` so we can
resume/complete all missing zh assets in one run once a translator command is
available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def iter_missing_lessons(phase_filter: int | None) -> list[str]:
    missing: list[str] = []
    for phase in sorted((ROOT / "phases").iterdir()):
        if not phase.is_dir() or not phase.name[:2].isdigit():
            continue
        phase_id = int(phase.name[:2])
        if phase_filter is not None and phase_filter != phase_id:
            continue
        for lesson in sorted(phase.iterdir()):
            if not lesson.is_dir():
                continue
            en = lesson / "docs" / "en.md"
            if not en.is_file():
                continue
            zh = lesson / "docs" / "zh.md"
            qzh = lesson / "quiz.zh.json"
            if not zh.is_file() or not qzh.is_file():
                missing.append(f"{phase.name}/{lesson.name}")
    return missing


def run_translation(missing: list[str], phase: int | None, translator_cmd: str, dry_run: bool, force: bool) -> int:
    translator = str((SCRIPTS / "translate_curriculum.py"))
    status = 0
    for item in missing:
        phase_name, lesson_name = item.split("/", 1)
        cmd = [
            sys.executable,
            translator,
            "--lang",
            "zh",
            "--phase",
            str(phase if phase is not None else int(phase_name[:2])),
            "--lesson",
            lesson_name,
            "--translator-cmd",
            translator_cmd,
        ]
        if force:
            cmd.append("--force")
        if dry_run:
            cmd.append("--dry-run")
            print("DRY:", " ".join(cmd))
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {item}")
            status = 1
    return status


def phase_summary(missing: list[str]) -> dict[int, int]:
    buckets: dict[int, int] = {}
    for item in missing:
        phase_no = int(item[:2])
        buckets[phase_no] = buckets.get(phase_no, 0) + 1
    return buckets


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translator-cmd", required=True, help="Command that reads stdin and writes translated stdout.")
    parser.add_argument("--phase", type=int, default=None, help="Only translate this phase (00-19).")
    parser.add_argument("--force", action="store_true", help="Overwrite already-existing localized files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args(argv)

    missing = iter_missing_lessons(args.phase)
    if not missing:
        print("No missing zh files found for selected scope.")
        return 0

    total = len(missing)
    print(f"Missing zh lessons: {total}")
    for phase_no, count in sorted(phase_summary(missing).items()):
        print(f"  phase {phase_no:02d}: {count}")

    if args.phase is not None:
        return run_translation(missing, args.phase, args.translator_cmd, args.dry_run, args.force)

    # Default path: translate all missing lessons phase by phase.
    phases = sorted({int(item[:2]) for item in missing})
    for phase_no in phases:
        phase_missing = [x for x in missing if x.startswith(f"{phase_no:02d}-")]
        print(f"\n[phase {phase_no:02d}] translate {len(phase_missing)} lessons")
        status = run_translation(phase_missing, phase_no, args.translator_cmd, args.dry_run, args.force)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
