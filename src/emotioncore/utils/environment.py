from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

from emotioncore.utils.common import load_paths, write_json_atomic


def key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def compare(lock_file: Path, legacy_site_packages: Path) -> dict:
    locked = {}
    for line in lock_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        locked[key(name)] = version
    installed = {key(d.metadata["Name"]): d.version
                 for d in importlib.metadata.distributions(path=[str(legacy_site_packages)])
                 if d.metadata.get("Name")}
    mismatch = {name: {"locked": version, "installed": installed.get(name)}
                for name, version in locked.items() if installed.get(name) != version}
    return {"locked_packages": len(locked), "installed_packages": len(installed),
            "mismatch_count": len(mismatch), "mismatches": mismatch,
            "key_versions": {name: installed.get(name) for name in
                             ("torch", "transformers", "accelerate", "bitsandbytes",
                              "qwen-omni-utils", "pyyaml", "peft")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("requirements.lock.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = load_paths(args.paths)
    site = paths["legacy_root"] / ".venv/lib/python3.11/site-packages"
    write_json_atomic(args.output, compare(args.lock, site))


if __name__ == "__main__":
    main()
