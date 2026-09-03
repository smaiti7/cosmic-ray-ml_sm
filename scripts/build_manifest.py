from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_SUFFIXES = {".orig", ".rej", ".pyc"}
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".git"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.name != "MANIFEST.sha256"
        and path.suffix not in EXCLUDED_SUFFIXES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"Wrote checksums for {len(files)} GitHub-deliverable files.")


if __name__ == "__main__":
    main()
