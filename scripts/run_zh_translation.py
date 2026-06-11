#!/usr/bin/env python3
"""Batch-run zh translation for all missing lessons using OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
MISSING_SCRIPT = ROOT / "scripts" / "translate_missing_zh.py"
TRANSLATOR_SCRIPT = ROOT / "scripts" / "openai_translator.py"
CURRICULUM = ROOT / "scripts" / "translate_curriculum.py"


def build_translator_cmd(args: argparse.Namespace) -> str:
    cmd = [
        sys.executable,
        str(TRANSLATOR_SCRIPT),
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--api-base",
        args.api_base,
        "--timeout",
        str(args.timeout),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def _iter_lessons(phase: int | None, include_missing_only: bool) -> list[tuple[str, str]]:
    lessons: list[tuple[str, str]] = []
    for phase_dir in sorted((ROOT / "phases").iterdir()):
        if not phase_dir.is_dir() or not phase_dir.name[:2].isdigit():
            continue
        if phase is not None and int(phase_dir.name[:2]) != phase:
            continue
        for lesson_dir in sorted(phase_dir.iterdir()):
            if not lesson_dir.is_dir() or not lesson_dir.name[:2].isdigit():
                continue
            en = lesson_dir / "docs" / "en.md"
            if not en.is_file():
                continue
            zh = lesson_dir / "docs" / "zh.md"
            if include_missing_only and zh.exists():
                quiz = lesson_dir / "quiz.zh.json"
                quiz_en = lesson_dir / "quiz.json"
                if quiz.exists() and quiz_en.exists():
                    continue
            lessons.append((phase_dir.name, lesson_dir.name))
    return lessons


def run_translate_curriculum(
    lesson_paths: Iterable[tuple[str, str]],
    translator_cmd: str,
    force: bool,
    dry_run: bool,
) -> int:
    code = 0
    for lesson_phase, lesson_name in lesson_paths:
        cmd = [
            sys.executable,
            str(CURRICULUM),
            "--lang",
            "zh",
            "--phase",
            lesson_phase[:2],
            "--lesson",
            lesson_name,
            "--translator-cmd",
            translator_cmd,
        ]
        if force:
            cmd.append("--force")
        if dry_run:
            cmd.append("--dry-run")
        print(f"RUN: {' '.join(cmd)}")
        if not dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                code = result.returncode
    return code


def run_all(
    phase: int | None,
    api_key: str,
    model: str,
    api_base: str,
    timeout: int,
    force: bool,
    dry_run: bool,
    translate_all: bool,
) -> int:
    if translate_all:
        lessons = _iter_lessons(phase, include_missing_only=False)
        return run_translate_curriculum(
            lessons,
            build_translator_cmd(argparse.Namespace(api_key=api_key, model=model, api_base=api_base, timeout=timeout)),
            force=force,
            dry_run=dry_run,
        )

    base_cmd = [
        sys.executable,
        str(MISSING_SCRIPT),
        "--translator-cmd",
        build_translator_cmd(argparse.Namespace(api_key=api_key, model=model, api_base=api_base, timeout=timeout)),
    ]
    if phase is not None:
        base_cmd.extend(["--phase", f"{phase:02d}"])
    if force:
        base_cmd.append("--force")
    if dry_run:
        base_cmd.append("--dry-run")
    proc = subprocess.run(base_cmd, text=True)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-4o-mini", help="Model id to call through chat/completions")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="Provider-compatible API base URL")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--phase", type=int, default=None, help="Translate only this phase number")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Translate all lessons with docs/en.md, not only those missing zh files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing zh files if present")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        sys.stderr.write("OPENAI_API_KEY is required (--api-key or env)\n")
        raise SystemExit(2)
    raise SystemExit(
        run_all(
            args.phase,
            args.api_key,
            args.model,
            args.api_base,
            args.timeout,
            args.force,
            args.dry_run,
            args.all,
        )
    )
