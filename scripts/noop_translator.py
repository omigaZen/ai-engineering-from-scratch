#!/usr/bin/env python3
"""No-op translator used for dry-run/validation of translation pipelines."""

import sys


def main() -> int:
    sys.stdout.write(sys.stdin.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
