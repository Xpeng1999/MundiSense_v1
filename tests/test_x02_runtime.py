import subprocess
import sys
import time

import pytest

from emotioncore.x02.runtime import git_provenance, run_with_hard_timeout


def test_dirty_run_rejected_and_diff_captured(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "X02 Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "x02@example.invalid"], check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    assert git_provenance(tmp_path)["dirty"] is False
    source.write_text("value = 2\n")
    (tmp_path / "new.py").write_text("added = True\n")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        git_provenance(tmp_path)
    receipt = git_provenance(tmp_path, allow_dirty=True)
    assert receipt["dirty"] is True
    assert "value = 2" in receipt["full_diff"]
    assert "added = True" in receipt["full_diff"]


def test_hard_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path / "orphan_marker"
    grandchild = ("import pathlib,time; time.sleep(0.5); "
                  f"pathlib.Path({str(marker)!r}).write_text('orphaned')")
    parent = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(10)")
    result = run_with_hard_timeout([sys.executable, "-c", parent], timeout_s=0.15)
    assert result["timed_out"] is True
    assert result["elapsed_s"] < 2
    time.sleep(0.7)
    assert not marker.exists()
