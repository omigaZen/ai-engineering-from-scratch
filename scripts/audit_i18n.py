#!/usr/bin/env python3
"""Audit optional localized lesson files.

English remains canonical. By default this script validates every localized
lesson that exists and permits missing translations so rollout can happen
phase by phase. Use --require-all when a language is expected to be complete.

Checks:
  - docs/<lang>.md and quiz.<lang>.json are paired when either exists
  - localized quiz schema matches quiz.json count, stage, correct, options count
  - fenced code blocks and inline code spans are preserved exactly
  - local Markdown links resolve
  - placeholder markers are absent

Usage:
  python scripts/audit_i18n.py --lang zh
  python scripts/audit_i18n.py --lang zh --phase 0
  python scripts/audit_i18n.py --lang zh --require-all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
LESSON_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
FENCE_RE = re.compile(r"^```([^\s`]*)\s*$")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
PLACEHOLDER_RE = re.compile(
    r"TODO|TRANSLATE|PLACEHOLDER|待翻译|占位|机器翻译待审",
    re.IGNORECASE,
)


@dataclass
class Issue:
    rule: str
    lesson: str
    file: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "lesson": self.lesson,
            "file": self.file,
            "message": self.message,
        }


@dataclass
class Audit:
    lessons_checked: int = 0
    localized_lessons: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, rule: str, lesson: Path, file: Path, message: str) -> None:
        self.issues.append(
            Issue(
                rule=rule,
                lesson=lesson.relative_to(ROOT).as_posix(),
                file=file.relative_to(ROOT).as_posix(),
                message=message,
            )
        )


def iter_lesson_dirs(phase_filter: int | None) -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not PHASE_DIR_RE.match(phase.name):
            continue
        if phase_filter is not None and int(phase.name.split("-", 1)[0]) != phase_filter:
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name):
                yield lesson


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None


def load_quiz(path: Path) -> list[dict[str, object]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        questions = data.get("questions")
    else:
        questions = data
    return questions if isinstance(questions, list) else None


def extract_fenced_blocks(text: str) -> tuple[list[tuple[str, str]], bool]:
    blocks: list[tuple[str, str]] = []
    in_block = False
    lang = ""
    lines: list[str] = []
    for raw in text.splitlines():
        fence = FENCE_RE.match(raw)
        if fence:
            if in_block:
                blocks.append((lang, "\n".join(lines)))
                in_block = False
                lang = ""
                lines = []
            else:
                in_block = True
                lang = fence.group(1)
                lines = []
            continue
        if in_block:
            lines.append(raw)
    return blocks, in_block


def extract_inline_code(text: str) -> list[str]:
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    return INLINE_CODE_RE.findall(cleaned)


def check_code_parity(audit: Audit, lesson: Path, en_doc: Path, loc_doc: Path) -> None:
    en_text = read_text(en_doc)
    loc_text = read_text(loc_doc)
    if en_text is None or loc_text is None:
        return

    en_blocks, en_unclosed = extract_fenced_blocks(en_text)
    loc_blocks, loc_unclosed = extract_fenced_blocks(loc_text)
    if en_unclosed or loc_unclosed:
        audit.add("I006", lesson, loc_doc, "unclosed fenced code block")
    if en_blocks != loc_blocks:
        audit.add(
            "I004",
            lesson,
            loc_doc,
            f"fenced code blocks differ from docs/en.md (en={len(en_blocks)} localized={len(loc_blocks)})",
        )

    en_inline = extract_inline_code(en_text)
    loc_inline = extract_inline_code(loc_text)
    if en_inline != loc_inline:
        audit.add(
            "I005",
            lesson,
            loc_doc,
            f"inline code spans differ from docs/en.md (en={len(en_inline)} localized={len(loc_inline)})",
        )


def check_quiz_parity(audit: Audit, lesson: Path, en_quiz: Path, loc_quiz: Path) -> None:
    en_questions = load_quiz(en_quiz)
    loc_questions = load_quiz(loc_quiz)
    if en_questions is None:
        return
    if loc_questions is None:
        audit.add("I003", lesson, loc_quiz, "localized quiz is missing or invalid JSON")
        return
    if len(en_questions) != len(loc_questions):
        audit.add(
            "I003",
            lesson,
            loc_quiz,
            f"question count differs from quiz.json (en={len(en_questions)} localized={len(loc_questions)})",
        )
        return
    for idx, (en_q, loc_q) in enumerate(zip(en_questions, loc_questions)):
        if not isinstance(en_q, dict) or not isinstance(loc_q, dict):
            audit.add("I003", lesson, loc_quiz, f"question[{idx}] is not an object")
            continue
        for key in ("stage", "correct"):
            if en_q.get(key) != loc_q.get(key):
                audit.add(
                    "I003",
                    lesson,
                    loc_quiz,
                    f"question[{idx}].{key} differs from quiz.json",
                )
        en_options = en_q.get("options")
        loc_options = loc_q.get("options")
        if not isinstance(en_options, list) or not isinstance(loc_options, list):
            audit.add("I003", lesson, loc_quiz, f"question[{idx}] options missing")
        elif len(en_options) != len(loc_options):
            audit.add(
                "I003",
                lesson,
                loc_quiz,
                f"question[{idx}] option count differs from quiz.json",
            )


def check_links(audit: Audit, lesson: Path, doc: Path, text: str) -> None:
    seen: set[str] = set()
    for match in MD_LINK_RE.finditer(text):
        href = match.group(1).strip()
        if href in seen:
            continue
        seen.add(href)
        if href.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        if href.startswith("/"):
            target = ROOT / href.lstrip("/")
        else:
            target = (doc.parent / href).resolve()
        if not target.exists():
            audit.add("I007", lesson, doc, f"internal link does not resolve: {href!r}")


def audit_lesson(audit: Audit, lesson: Path, lang: str, require_all: bool) -> None:
    audit.lessons_checked += 1
    en_doc = lesson / "docs" / "en.md"
    en_quiz = lesson / "quiz.json"
    loc_doc = lesson / "docs" / f"{lang}.md"
    loc_quiz = lesson / f"quiz.{lang}.json"
    has_doc = loc_doc.is_file()
    has_quiz = loc_quiz.is_file()

    if require_all and en_doc.is_file() and not has_doc:
        audit.add("I001", lesson, loc_doc, "missing localized docs")
    if require_all and en_quiz.is_file() and not has_quiz:
        audit.add("I002", lesson, loc_quiz, "missing localized quiz")
    if not has_doc and not has_quiz:
        return

    audit.localized_lessons += 1
    if has_doc and not has_quiz and en_quiz.is_file():
        audit.add("I002", lesson, loc_quiz, "localized docs exist but localized quiz is missing")
    if has_quiz and not has_doc and en_doc.is_file():
        audit.add("I001", lesson, loc_doc, "localized quiz exists but localized docs are missing")

    if has_doc:
        text = read_text(loc_doc)
        if text is None:
            audit.add("I001", lesson, loc_doc, "localized docs are not valid UTF-8")
        else:
            if PLACEHOLDER_RE.search(text):
                audit.add("I008", lesson, loc_doc, "placeholder marker found")
            check_code_parity(audit, lesson, en_doc, loc_doc)
            check_links(audit, lesson, loc_doc, text)

    if has_quiz:
        raw = read_text(loc_quiz)
        if raw is None:
            audit.add("I003", lesson, loc_quiz, "localized quiz is not valid UTF-8")
        elif PLACEHOLDER_RE.search(raw):
            audit.add("I008", lesson, loc_quiz, "placeholder marker found")
        check_quiz_parity(audit, lesson, en_quiz, loc_quiz)


def render_report(audit: Audit, lang: str) -> str:
    lines = [
        f"audit_i18n.py --lang {lang} — {audit.lessons_checked} lesson(s) checked, "
        f"{audit.localized_lessons} localized, {len(audit.issues)} issue(s)",
    ]
    if audit.issues:
        lines.append("")
        for issue in audit.issues:
            lines.append(f"  [{issue.rule}] {issue.file}: {issue.message}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="zh", help="language code to audit (default: zh)")
    parser.add_argument("--phase", type=int, default=None, help="restrict to a single phase number")
    parser.add_argument("--require-all", action="store_true", help="fail when any English lesson lacks localized files")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)

    lang = args.lang.strip().lower()
    if not re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{2,8})?", lang):
        sys.stderr.write(f"invalid --lang value: {args.lang!r}\n")
        return 2

    audit = Audit()
    for lesson in iter_lesson_dirs(args.phase):
        audit_lesson(audit, lesson, lang, args.require_all)

    if args.json:
        json.dump(
            {
                "lang": lang,
                "lessons_checked": audit.lessons_checked,
                "localized_lessons": audit.localized_lessons,
                "issues": [issue.to_dict() for issue in audit.issues],
            },
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_report(audit, lang) + "\n")

    return 1 if audit.issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
