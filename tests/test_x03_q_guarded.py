import json
import sys

import numpy as np
import pytest

from emotioncore.x03.q_pipeline import run_step
from emotioncore.x03.q_manifest import sha256
from emotioncore.x03.q_train import eval_boundaries
from emotioncore.x03.q_train_guarded import (ordered_training_rows, restore_last,
                                             save_last, segment_ends,
                                             trainable_parameters)
from emotioncore.x03.q_worst_case import select_longest


def test_guarded_order_and_boundaries_preserve_frozen_trainer():
    rows = [{"utt_id": str(i)} for i in range(19)]
    ordered = ordered_training_rows(rows, "fold1")
    assert len(ordered) == 38
    assert {r["utt_id"] for r in ordered[:19]} == {str(i) for i in range(19)}
    assert {r["utt_id"] for r in ordered[19:]} == {str(i) for i in range(19)}
    ends = segment_ends(len(rows))
    assert set(eval_boundaries(len(rows))) <= set(ends)
    assert {8, 16, 24, 32, 38} <= set(ends)
    assert ends == sorted(set(ends))


def test_guarded_launcher_retries_once_from_last(tmp_path, monkeypatch):
    monkeypatch.setattr("emotioncore.x03.q_pipeline.assert_e2_cpu", lambda: None)
    root = tmp_path / "q"
    root.mkdir()
    output = root / "fold1"
    output.mkdir()
    code = ("import pathlib,sys; p=pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1]); "
            "resume='--resume-last' in sys.argv; "
            "(p/'last.json').write_text('{}') if not resume else None; "
            "sys.exit(0 if resume else 17)")
    progress = {"steps": []}
    run_step("train_fold1", [sys.executable, "-c", code, "--output-dir", str(output)],
             root, progress, resume_training=True)
    assert [step["exit_code"] for step in progress["steps"]] == [17, 0]
    incidents = [json.loads(line) for line in (root / "incidents.jsonl").read_text().splitlines()]
    assert len(incidents) == 1
    assert incidents[0]["last_exists"] is True
    assert (root / "pipeline_logs" / "train_fold1_attempt1.log").is_file()


def test_guarded_launcher_caps_retries_at_two(tmp_path, monkeypatch):
    monkeypatch.setattr("emotioncore.x03.q_pipeline.assert_e2_cpu", lambda: None)
    root = tmp_path / "q"
    output = root / "fold1"
    output.mkdir(parents=True)
    (output / "last.json").write_text("{}")
    progress = {"steps": []}
    command = [sys.executable, "-c", "import sys;sys.exit(17)",
               "--output-dir", str(output)]
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        run_step("train_fold1", command, root, progress, resume_training=True)
    assert len(progress["steps"]) == 3
    assert len((root / "incidents.jsonl").read_text().splitlines()) == 3


def test_longest_av_ranking_uses_input_token_length(tmp_path):
    rows = []
    for uid, length in (("c", 7), ("b", 13), ("a", 13)):
        folder = tmp_path / uid
        folder.mkdir()
        np.savez(folder / "AV.npz", input_ids=np.zeros((1, length), dtype=np.int64))
        (folder / "AV.json").write_text(json.dumps({
            "status": "ok", "condition": "AV", "input_wav_sha256": "wav",
            "input_video_sha256": "video", "sha256": sha256(folder / "AV.npz")}))
        rows.append({"utt_id": uid, "wav_sha256": "wav", "video_sha256": "video"})
    ranked = select_longest(rows, tmp_path)
    assert [item["row"]["utt_id"] for item in ranked] == ["a", "b", "c"]
    assert [item["input_token_length"] for item in ranked] == [13, 13, 7]


def test_last_checkpoint_restores_adapter_optimizer_scheduler_and_gradient(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))

        def save_pretrained(self, path, safe_serialization=True):
            from pathlib import Path
            folder = Path(path)
            folder.mkdir()
            (folder / "adapter_model.safetensors").write_bytes(
                self.weight.detach().numpy().tobytes())

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model.weight.sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    model.weight.sum().backward()  # Pending accumulation at validation boundary.
    expected_weight = model.weight.detach().clone()
    expected_gradient = model.weight.grad.detach().clone()
    root = tmp_path / "fold1"
    root.mkdir()
    with (root / "train_modality_receipts.jsonl").open("w+") as records, \
            (root / "skipped_oom.jsonl").open("w+") as skipped:
        records.write("sample\n")
        pointer = save_last(root, model, optimizer, scheduler,
                            trainable_parameters(model),
                            {"examples_seen": 2, "gpu_elapsed_s": .5}, 2,
                            records, skipped, "train-sha", "val-sha")
    assert pointer["position"] == 2
    model.weight.data.fill_(99)
    model.weight.grad = None
    optimizer.param_groups[0]["lr"] = .9
    position, progress, record_bytes, skipped_bytes = restore_last(
        root, model, optimizer, scheduler, trainable_parameters(model),
        "train-sha", "val-sha")
    assert position == progress["examples_seen"] == 2
    assert record_bytes == len("sample\n") and skipped_bytes == 0
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.weight.grad, expected_gradient)
    assert optimizer.param_groups[0]["lr"] < .9
