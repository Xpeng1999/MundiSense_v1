"""X02 launch provenance and process-isolated hard timeouts."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PREPROCESS_TIMEOUT_S = 120


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def git_provenance(root: Path, allow_dirty: bool = False) -> dict:
    """Reject dirty formal runs or preserve their complete tracked/untracked diff."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    status = _git(root, "status", "--porcelain=v1", "-uall").decode()
    if status and not allow_dirty:
        raise RuntimeError("X02 formal run requires a clean Git worktree; pass --allow-dirty to record its full diff")
    diff = ""
    if status:
        diff = _git(root, "diff", "--binary", "HEAD", "--").decode()
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        for raw in untracked:
            if not raw:
                continue
            path = root / os.fsdecode(raw)
            if path.is_file():
                result = subprocess.run(["git", "-C", str(root), "diff", "--no-index",
                                         "--binary", "--", "/dev/null", str(path)],
                                        capture_output=True, check=False)
                if result.returncode not in (0, 1):
                    raise RuntimeError(f"Could not capture untracked diff: {path}")
                diff += result.stdout.decode(errors="replace")
    return {"commit": commit, "dirty": bool(status), "status": status,
            "full_diff": diff if status else None}


def run_with_hard_timeout(command: list[str], timeout_s: float = PREPROCESS_TIMEOUT_S,
                          *, cwd: Path | None = None,
                          env: dict[str, str] | None = None) -> dict:
    """Kill the entire child process group, including ffmpeg descendants, on timeout."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = child.communicate()
        timed_out = True
    return {"command": command, "pid": child.pid, "returncode": child.returncode,
            "timed_out": timed_out, "timeout_s": timeout_s,
            "elapsed_s": time.monotonic() - started,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace")}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = git_provenance(args.repo, args.allow_dirty)
    receipt["command"] = sys.argv
    write_json_atomic(args.output, receipt)


if __name__ == "__main__":
    main()
