"""X02 launch provenance and process-isolated hard timeouts."""
from __future__ import annotations

import json
import os
import select
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


class PersistentJSONWorker:
    """Reuse a preprocessing child while enforcing a parent-side deadline per task."""

    def __init__(self, command: list[str], *, cwd: Path | None = None,
                 env: dict[str, str] | None = None, stderr_path: Path | None = None):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.stderr_path = stderr_path
        self.child: subprocess.Popen | None = None
        self._pending = b""
        self._stderr = None

    def _start(self) -> None:
        if self.stderr_path:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = self.stderr_path.open("ab", buffering=0)
        self.child = subprocess.Popen(self.command, cwd=self.cwd, env=self.env,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._stderr or subprocess.DEVNULL,
                                      start_new_session=True, bufsize=0)
        self._pending = b""

    def close(self) -> None:
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()
        self.child = None
        if self._stderr:
            self._stderr.close()
            self._stderr = None

    def request(self, payload: dict, timeout_s: float = PREPROCESS_TIMEOUT_S) -> dict:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.child is None or self.child.poll() is not None:
            self.close()
            self._start()
        assert self.child is not None and self.child.stdin and self.child.stdout
        started = time.monotonic()
        try:
            self.child.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            self.child.stdin.flush()
            while b"\n" not in self._pending:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Preprocessing exceeded {timeout_s} seconds")
                ready, _, _ = select.select([self.child.stdout], [], [], remaining)
                if not ready:
                    raise TimeoutError(f"Preprocessing exceeded {timeout_s} seconds")
                chunk = os.read(self.child.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"Preprocessing child exited: {self.child.poll()}")
                self._pending += chunk
                if len(self._pending) > 8 * 1024 * 1024:
                    raise RuntimeError("Preprocessing response exceeded 8 MiB")
            line, self._pending = self._pending.split(b"\n", 1)
            response = json.loads(line)
            response["elapsed_s"] = time.monotonic() - started
            return response
        except (TimeoutError, BrokenPipeError, RuntimeError, json.JSONDecodeError) as exc:
            self.close()
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "timed_out": isinstance(exc, TimeoutError),
                    "elapsed_s": time.monotonic() - started}


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
