#!/usr/bin/env python3
"""Unwrap hard-wrapped Markdown paragraphs and list items to one line each.

The `issues/*.md` files are hand-wrapped at ~100 columns, which is a hassle to
paste into GitHub (its issue editor doesn't reflow on edit, so every tweak
means manually re-wrapping the paragraph). This rejoins each paragraph and
each list item onto a single line, leaving headings, blank lines, and table
rows untouched. Markdown renders identically either way — single newlines
inside a paragraph collapse to a space.

Usage:
    uv run python scripts/unwrap_issues.py [FILES...]   # rewrite in place
    uv run python scripts/unwrap_issues.py --check [FILES...]  # exit 1 if any file would change

With no FILES, processes issues/*.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HEADING_RE = re.compile(r"^#{1,6}\s")
LIST_MARKER_RE = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+")


def unwrap(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    buf: str | None = None
    buf_kind: str | None = None  # "list" or "para"
    list_indent = 0

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf is not None:
            out.append(buf)
        buf = None
        buf_kind = None

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            flush()
            out.append("")
            continue

        if stripped.startswith("|"):
            flush()
            out.append(line.rstrip())
            continue

        if HEADING_RE.match(stripped):
            flush()
            buf = line.rstrip()  # a heading can itself be hard-wrapped onto the next line(s)
            buf_kind = "heading"
            continue

        marker = LIST_MARKER_RE.match(line)
        if marker:
            flush()
            buf = line.rstrip()  # keep the marker's own indentation (nesting level)
            buf_kind = "list"
            list_indent = len(marker.group(1))
            continue

        indent = len(line) - len(line.lstrip(" "))
        if buf_kind == "list" and indent > list_indent:
            buf = f"{buf} {stripped}"
            continue
        if buf_kind in ("para", "heading") and indent == 0:
            buf = f"{buf} {stripped}"
            continue

        flush()
        buf = line.rstrip()
        buf_kind = "para"

    flush()

    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="Markdown files to unwrap (default: issues/*.md)")
    parser.add_argument("--check", action="store_true", help="Don't write; exit 1 if any file would change")
    args = parser.parse_args()

    files = args.files or sorted((REPO_ROOT / "issues").glob("*.md"))
    if not files:
        print("no files to process", file=sys.stderr)
        return 1

    changed = []
    for path in files:
        original = path.read_text()
        new = unwrap(original)
        if new != original:
            changed.append(path)
            if not args.check:
                path.write_text(new)

    if args.check:
        for path in changed:
            print(f"would unwrap: {path}")
        return 1 if changed else 0

    for path in changed:
        print(f"unwrapped: {path}")
    print(f"{len(changed)}/{len(files)} file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
