"""Run pytest and store machine-readable summary for REPORT.md."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from emotioncore.utils.common import write_json_atomic


def main() -> None:
    command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    counts = {name: int(match.group(1)) if (match := re.search(rf"(\d+) {name}", output)) else 0
              for name in ("passed", "failed", "skipped")}
    write_json_atomic(Path("experiments/X01_benchmark/results/test_results.json"),
                      {"command": command, "exit_code": result.returncode,
                       "output": output, **counts})
    print(output, end="")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
