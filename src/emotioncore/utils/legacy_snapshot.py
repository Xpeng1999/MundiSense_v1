from __future__ import annotations

import argparse
from pathlib import Path

from emotioncore.utils.common import sha256_file, write_json_atomic


def parse(path: Path) -> dict[str, tuple[str, str]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, mtime, size = line.split("\t")
        rows[name] = (mtime, size)
    return rows


def compare(before: Path, after: Path) -> dict:
    first, last = parse(before), parse(after)
    added = sorted(set(last) - set(first))
    removed = sorted(set(first) - set(last))
    changed = sorted(name for name in set(first) & set(last) if first[name] != last[name])
    return {"before_path": str(before), "before_sha256": sha256_file(before),
            "after_path": str(after), "after_sha256": sha256_file(after),
            "before_files": len(first), "after_files": len(last),
            "added_count": len(added), "removed_count": len(removed),
            "changed_existing_count": len(changed), "added_files": added,
            "removed_files": removed, "changed_existing_files": changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, compare(args.before, args.after))


if __name__ == "__main__":
    main()
