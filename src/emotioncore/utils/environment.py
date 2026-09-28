from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys
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
    parser.add_argument("--verify-current", action="store_true")
    args = parser.parse_args()
    if args.verify_current:
        import torch
        from emotioncore.utils.common import sha256_file

        site = Path(sys.prefix) / "lib/python3.11/site-packages"
        receipt = compare(args.lock, site)
        dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                                          capture_output=True, text=True)
        receipt.update({"interpreter": sys.executable, "venv_prefix": sys.prefix,
                        "torch_file": torch.__file__, "cuda_available": torch.cuda.is_available(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "pip_check_exit_code": dependency_check.returncode,
                        "pip_check_output": dependency_check.stdout + dependency_check.stderr,
                        "lock_sha256": sha256_file(args.lock)})
    else:
        paths = load_paths(args.paths)
        site = paths["legacy_root"] / ".venv/lib/python3.11/site-packages"
        receipt = compare(args.lock, site)
    write_json_atomic(args.output, receipt)


if __name__ == "__main__":
    main()
