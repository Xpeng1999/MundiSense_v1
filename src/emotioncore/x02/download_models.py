"""Pin, budget, download, and hash only the public X02 encoder files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic


MAX_DOWNLOAD_BYTES = 100_000_000_000
MODELS = {
    "A3": ("TencentGameMate/chinese-hubert-large",
           ("config.json", "preprocessor_config.json", "pytorch_model.bin")),
    "V2": ("openai/clip-vit-large-patch14",
           ("config.json", "preprocessor_config.json", "model.safetensors")),
    "T1": ("hfl/chinese-roberta-wwm-ext-large",
           ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
            "vocab.txt", "special_tokens_map.json", "added_tokens.json")),
    "T2": ("BAAI/bge-large-zh-v1.5",
           ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
            "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json",
            "modules.json", "sentence_bert_config.json")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan(endpoint: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint)
    entries = {}
    total = 0
    for encoder, (repo_id, filenames) in MODELS.items():
        info = api.model_info(repo_id, files_metadata=True)
        remote = {row.rfilename: row for row in info.siblings}
        missing = set(filenames) - set(remote)
        if missing:
            raise RuntimeError(f"Missing upstream files for {repo_id}: {sorted(missing)}")
        files = []
        for name in filenames:
            sibling = remote[name]
            if sibling.size is None:
                raise RuntimeError(f"No size for {repo_id}/{name}")
            files.append({"path": name, "bytes": sibling.size,
                          "upstream_lfs_sha256": sibling.lfs.sha256 if sibling.lfs else None,
                          "upstream_git_blob": sibling.blob_id})
            total += sibling.size
        entries[encoder] = {"repo_id": repo_id,
                            "source_url": f"https://huggingface.co/{repo_id}/tree/{info.sha}",
                            "revision": info.sha, "files": files,
                            "planned_bytes": sum(row["bytes"] for row in files)}
    if total > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"Planned downloads exceed 100 GB: {total}")
    return {"endpoint": endpoint, "total_planned_bytes": total,
            "budget_bytes": MAX_DOWNLOAD_BYTES, "models": entries}


def download(run_root: Path, desired: tuple[str, ...], endpoint: str,
             allow_dirty: bool) -> None:
    from huggingface_hub import snapshot_download

    repo = Path.cwd()
    provenance = git_provenance(repo, allow_dirty)
    run_root.mkdir(parents=True, exist_ok=True)
    current_plan = plan(endpoint)
    write_json_atomic(run_root / "download_plan.json", current_plan)
    write_json_atomic(run_root / "download_run_meta.json",
                      {"git": provenance, "command": sys.argv, "endpoint": endpoint,
                       "download_plan_sha256": sha256(run_root / "download_plan.json")})
    if shutil.disk_usage(run_root).free < current_plan["total_planned_bytes"] + 10_000_000_000:
        raise RuntimeError("Insufficient free disk for selected encoder downloads")
    receipts_path = run_root / "model_downloads.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {"models": {}}
    for encoder in desired:
        if encoder not in current_plan["models"]:
            raise ValueError(encoder)
        spec = current_plan["models"][encoder]
        target = run_root / "models" / encoder
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                          local_dir=target, allow_patterns=[row["path"] for row in spec["files"]],
                          max_workers=3, endpoint=endpoint)
        files = []
        for item in spec["files"]:
            path = target / item["path"]
            actual = sha256(path)
            if item["upstream_lfs_sha256"] and actual != item["upstream_lfs_sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {spec['repo_id']}/{item['path']}")
            files.append({**item, "local_path": str(path), "local_sha256": actual})
        receipts["models"][encoder] = {**spec, "files": files, "local_dir": str(target)}
        receipts["total_verified_bytes"] = sum(
            sum(row["bytes"] for row in model["files"])
            for model in receipts["models"].values())
        if receipts["total_verified_bytes"] > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Cumulative downloaded model budget exceeded")
        write_json_atomic(receipts_path, receipts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--encoders", nargs="+", choices=tuple(MODELS), default=tuple(MODELS))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    if args.plan_only:
        print(json.dumps(plan(args.endpoint), ensure_ascii=False, indent=2))
    else:
        download(args.run_root, tuple(args.encoders), args.endpoint, args.allow_dirty)


if __name__ == "__main__":
    main()
