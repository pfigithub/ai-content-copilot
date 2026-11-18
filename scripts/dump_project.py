"""Dump project sources into one or more UTF-8 text files for review.

Never includes: .env*, credentials, LICENSE, .gitignore, .dockerignore,
virtualenvs, caches, compiled files, or binary blobs. Safe to run at
any time; does not touch git state.

Usage:
    python scripts/dump_project.py
    python scripts/dump_project.py --split 100     # parts of <=100 KB
    python scripts/dump_project.py --out review.txt
"""

import argparse
from pathlib import Path


EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
    "node_modules", ".streamlit",
}

EXCLUDED_NAMES = {
    ".env", ".gitignore", ".dockerignore",
    "LICENSE", "LICENCE",
    ".DS_Store", "Thumbs.db",
}

EXCLUDED_EXTENSIONS = {
    ".pyc", ".pyo", ".lock",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".whl", ".exe", ".dll",
}

EXCLUDED_PREFIXES = ("full_codebase", "frontend_dump")


def _is_excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    if rel.name in EXCLUDED_NAMES:
        return True
    # Exclude .env.production etc., but keep .env.example as a template.
    if rel.name.startswith(".env.") and rel.name != ".env.example":
        return True
    if rel.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True
    if rel.name.startswith(EXCLUDED_PREFIXES):
        return True
    return False


def _collect(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and not _is_excluded(p.relative_to(root))
    ]
    return sorted(files)


def _chunk_for(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        body = "<<binary or non-UTF-8: skipped>>"
    return f"=== FILE: {rel} ===\n{body}\n\n"


def _write_single(root: Path, files: list[Path], out: Path) -> None:
    parts = [_chunk_for(root, p) for p in files]
    out.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {len(files)} files to {out.name} "
          f"({out.stat().st_size:,} bytes)")


def _write_split(root: Path, files: list[Path], out: Path, kb: int) -> None:
    limit = kb * 1024
    part_num, buf, size = 1, [], 0

    def flush() -> None:
        nonlocal part_num, buf, size
        if not buf:
            return
        name = f"{out.stem}_part{part_num}{out.suffix}"
        (root / name).write_text("".join(buf), encoding="utf-8")
        print(f"Wrote {name} ({size:,} bytes)")
        part_num, buf, size = part_num + 1, [], 0

    for path in files:
        chunk = _chunk_for(root, path)
        chunk_size = len(chunk.encode("utf-8"))
        if size + chunk_size > limit and buf:
            flush()
        buf.append(chunk)
        size += chunk_size
    flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--out", default="full_codebase.txt")
    parser.add_argument(
        "--split", type=int, default=0,
        help="Split into parts of N KB each (0 = one file)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = _collect(root)
    out = root / args.out

    if args.split <= 0:
        _write_single(root, files, out)
    else:
        _write_split(root, files, out, args.split)


if __name__ == "__main__":
    main()