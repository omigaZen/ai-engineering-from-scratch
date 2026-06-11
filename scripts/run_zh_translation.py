#!/usr/bin/env python3
"""Batch-run zh translation for all missing lessons using OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MISSING_SCRIPT = ROOT / "scripts" / "translate_missing_zh.py"
TRANSLATOR_SCRIPT = ROOT / "scripts" / "openai_translator.py"


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


def run_all(
    phase: int | None,
    api_key: str,
    model: str,
    api_base: str,
    timeout: int,
    force: bool,
    dry_run: bool,
) -> int:
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
        )
    )
