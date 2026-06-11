#!/usr/bin/env python3
"""Translate stdin text via OpenAI-compatible chat completion API.

The script intentionally uses only the Python standard library so it can run in the
same environment as the repository tooling.

Usage:
  python scripts/openai_translator.py --model gpt-4o-mini

Input must be UTF-8 text on stdin and the translated text is written to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name, e.g. gpt-4o-mini")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="Base URL for chat completions endpoint")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--target",
        default="zh-CN",
        help="Target language (for prompt text, default zh-CN)",
    )
    parser.add_argument("--source", default="en", help="Source language label for prompt context")
    return parser.parse_args()


def request_translation(payload: dict[str, Any], api_key: str, timeout: int) -> str:
    req = urllib.request.Request(
        url=f"{args.api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        body = r.read().decode("utf-8")
    data = json.loads(body)
    return (
        data["choices"][0]["message"]["content"]
        if data.get("choices")
        else ""
    )


def call_with_retries(payload: dict[str, Any], api_key: str, timeout: int, max_retries: int) -> str:
    attempt = 0
    while True:
        try:
            return request_translation(payload, api_key, timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"translation request failed: {exc!r}")
            time.sleep(2 ** (attempt - 1))


def main() -> int:
    global args
    args = parse_args()
    if not args.api_key:
        sys.stderr.write("--api-key or OPENAI_API_KEY is required\n")
        return 2

    text = sys.stdin.read()
    if not text.strip():
        return 0

    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate content to the target language. "
                    f"Source: {args.source}. Target: {args.target}. "
                    "Return only the translated text, preserving formatting and code blocks."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Translate all user-facing text to the target language.\n"
                    "Do not add explanations.\n"
                    f'Original text ({args.source}):\n"""\n{text}\n"""'
                ),
            },
        ],
    }

    try:
        translated = call_with_retries(payload, args.api_key, args.timeout, args.max_retries)
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sys.stdout.write(translated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
