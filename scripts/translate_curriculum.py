#!/usr/bin/env python3
"""Translate lesson docs and quizzes through a pluggable command.

This script intentionally uses only the Python standard library. It does not
ship an API client. Instead, pass a command that reads text from stdin and
writes translated text to stdout:

  python scripts/translate_curriculum.py --phase 0 --lesson 01-dev-environment \
    --translator-cmd "my-translator --to zh-CN"

The command is called separately for protected text chunks. Markdown code
blocks, inline code spans, URLs, HTML comments, and Markdown links are replaced
with placeholders before translation and restored afterward.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")

PROTECTED_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),
    re.compile(r"<!--[\s\S]*?-->"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
    re.compile(r"https?://[^\s)>\"]+"),
]

TRANSLATABLE_QUIZ_FIELDS = {"question", "options", "explanation"}


@dataclass
class LessonJob:
    lesson_dir: Path
    doc_src: Path
    doc_dst: Path
    quiz_src: Path
    quiz_dst: Path


class Protector:
    def __init__(self) -> None:
        self.values: list[str] = []

    def protect(self, text: str) -> str:
        for pattern in PROTECTED_PATTERNS:
            text = pattern.sub(self._stash, text)
        return text

    def restore(self, text: str) -> str:
        for idx, value in enumerate(self.values):
            text = text.replace(f"__AIFS_PROTECTED_{idx}__", value)
        return text

    def _stash(self, match: re.Match[str]) -> str:
        token = f"__AIFS_PROTECTED_{len(self.values)}__"
        self.values.append(match.group(0))
        return token


def run_translator(command: list[str], text: str) -> str:
    proc = subprocess.run(
        command,
        input=text,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "translator command failed with exit "
            f"{proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def translate_markdown(text: str, command: list[str]) -> str:
    protector = Protector()
    protected = protector.protect(text)
    translated = run_translator(command, protected)
    return protector.restore(translated)


def translate_quiz_value(value: object, command: list[str]) -> object:
    if isinstance(value, str):
        protector = Protector()
        protected = protector.protect(value)
        translated = run_translator(command, protected)
        return protector.restore(translated).strip()
    if isinstance(value, list):
        return [translate_quiz_value(item, command) for item in value]
    return value


def translate_quiz(text: str, command: list[str]) -> str:
    data = json.loads(text)
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError("quiz must be a list or an object with questions[]")

    for question in questions:
        if not isinstance(question, dict):
            continue
        for field in TRANSLATABLE_QUIZ_FIELDS:
            if field in question:
                question[field] = translate_quiz_value(question[field], command)

    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def iter_phase_dirs(phase_filter: int | None) -> Iterable[Path]:
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir():
            continue
        match = PHASE_DIR_RE.match(phase.name)
        if not match:
            continue
        if phase_filter is not None and int(match.group(1)) != phase_filter:
            continue
        yield phase


def iter_jobs(phase_filter: int | None, lesson_filter: str | None, lang: str) -> Iterable[LessonJob]:
    for phase in iter_phase_dirs(phase_filter):
        for lesson in sorted(phase.iterdir()):
            if not lesson.is_dir() or not LESSON_DIR_RE.match(lesson.name):
                continue
            if lesson_filter and lesson.name != lesson_filter:
                continue
            doc_src = lesson / "docs" / "en.md"
            quiz_src = lesson / "quiz.json"
            if not doc_src.is_file() and not quiz_src.is_file():
                continue
            yield LessonJob(
                lesson_dir=lesson,
                doc_src=doc_src,
                doc_dst=lesson / "docs" / f"{lang}.md",
                quiz_src=quiz_src,
                quiz_dst=lesson / f"quiz.{lang}.json",
            )


def write_if_allowed(path: Path, content: str, force: bool, dry_run: bool) -> bool:
    if path.exists() and not force:
        raise FileExistsError(f"{path.relative_to(ROOT)} exists; pass --force to overwrite")
    if dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="zh", help="target language code used in filenames (default: zh)")
    parser.add_argument("--phase", type=int, default=None, help="translate one phase number")
    parser.add_argument("--lesson", default=None, help="translate one lesson directory name")
    parser.add_argument("--translator-cmd", required=True, help="command that reads stdin and writes translated stdout")
    parser.add_argument("--force", action="store_true", help="overwrite existing localized files")
    parser.add_argument("--dry-run", action="store_true", help="show what would be written without writing files")
    args = parser.parse_args(argv)

    lang = args.lang.strip().lower()
    if not re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{2,8})?", lang):
        sys.stderr.write(f"invalid --lang value: {args.lang!r}\n")
        return 2

    command = shlex.split(args.translator_cmd)
    if not command:
        sys.stderr.write("--translator-cmd cannot be empty\n")
        return 2

    jobs = list(iter_jobs(args.phase, args.lesson, lang))
    if not jobs:
        sys.stderr.write("no matching lessons found\n")
        return 1

    wrote = 0
    for job in jobs:
        rel = job.lesson_dir.relative_to(ROOT).as_posix()
        sys.stdout.write(f"{rel}\n")
        if job.doc_src.is_file():
            doc_text = job.doc_src.read_text(encoding="utf-8")
            doc_out = translate_markdown(doc_text, command)
            if write_if_allowed(job.doc_dst, doc_out, args.force, args.dry_run):
                wrote += 1
            sys.stdout.write(f"  docs -> {job.doc_dst.relative_to(ROOT).as_posix()}\n")
        if job.quiz_src.is_file():
            quiz_text = job.quiz_src.read_text(encoding="utf-8")
            quiz_out = translate_quiz(quiz_text, command)
            if write_if_allowed(job.quiz_dst, quiz_out, args.force, args.dry_run):
                wrote += 1
            sys.stdout.write(f"  quiz -> {job.quiz_dst.relative_to(ROOT).as_posix()}\n")

    sys.stdout.write(f"{'would write' if args.dry_run else 'wrote'} {wrote} file(s)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
