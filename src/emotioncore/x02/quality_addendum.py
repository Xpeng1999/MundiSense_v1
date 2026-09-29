"""Recount CER and cache text-free EmotionTalk utterance quality statistics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

from emotioncore.x02.asr_cer import edit_distance, normalize as old_normalize, sha256
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x02.text_t2 import text_path


HELD_OUT_SPEAKERS = frozenset({"02", "13", "20", "21"})
EXPECTED_COUNT = 19_250


def normalize_nfkc_alnum(value: str) -> str:
    """Keep Unicode letters and numbers (including Han ideographs) after NFKC."""
    folded = unicodedata.normalize("NFKC", value).lower()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in {"L", "N"})


def measure(reference: str, hypothesis: str, duration_s: float) -> dict:
    if duration_s <= 0:
        raise ValueError("Audio duration must be positive")
    ref = normalize_nfkc_alnum(reference)
    hyp = normalize_nfkc_alnum(hypothesis)
    ref_chars, asr_chars = len(ref), len(hyp)
    rate = ref_chars / duration_s
    ratio = asr_chars / ref_chars if ref_chars else None
    new_edits = edit_distance(ref, hyp)
    old_ref = old_normalize(reference)
    old_hyp = old_normalize(hypothesis)
    return {"reference_chars": ref_chars, "asr_chars": asr_chars,
            "reference_chars_per_s": rate, "asr_reference_ratio": ratio,
            "cer_nfkc_alnum": new_edits / ref_chars if ref_chars else None,
            "truncation_suspect": rate > 9.0 or (ratio is not None and ratio < 0.5),
            "new_edit_distance": new_edits, "old_reference_chars": len(old_ref),
            "old_edit_distance": edit_distance(old_ref, old_hyp)}


def worker_main() -> None:
    for line in sys.stdin:
        try:
            task = json.loads(line)
            annotation = json.loads(Path(task["annotation"]).read_text(encoding="utf-8"))
            transcript_path = Path(task["transcript"])
            if sha256(transcript_path) != task["transcript_sha256"]:
                raise ValueError("ASR transcript SHA-256 differs from its receipt")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            if transcript["id"] != task["utt_id"]:
                raise ValueError("ASR transcript ID mismatch")
            reference, hypothesis = annotation["content"], transcript["text"]
            if not isinstance(reference, str) or not isinstance(hypothesis, str):
                raise ValueError("Reference and ASR content must be strings")
            result = {"status": "ok", "utt_id": task["utt_id"],
                      **measure(reference, hypothesis, task["duration_s"])}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def counter() -> dict:
    return {"total": 0, "suspect": 0, "new_reference_chars": 0,
            "new_edit_distance": 0, "old_reference_chars": 0,
            "old_edit_distance": 0, "empty_reference": 0}


def add(group: dict, measurement: dict) -> None:
    group["total"] += 1
    group["suspect"] += int(measurement["truncation_suspect"])
    group["new_reference_chars"] += measurement["reference_chars"]
    group["new_edit_distance"] += measurement["new_edit_distance"]
    group["old_reference_chars"] += measurement["old_reference_chars"]
    group["old_edit_distance"] += measurement["old_edit_distance"]
    group["empty_reference"] += int(measurement["reference_chars"] == 0)


def finish(group: dict) -> dict:
    return {**group,
            "suspect_fraction": group["suspect"] / group["total"] if group["total"] else None,
            "cer_old_nfc": (group["old_edit_distance"] / group["old_reference_chars"]
                            if group["old_reference_chars"] else None),
            "cer_nfkc_alnum": (group["new_edit_distance"] / group["new_reference_chars"]
                               if group["new_reference_chars"] else None)}


def table(title: str, groups: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", "", "| 分组 | 句数 | 疑似截断 | 比例 | 原 CER | 新 CER |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, item in sorted(groups.items()):
        lines.append(f"| {name} | {item['total']:,} | {item['suspect']:,} | "
                     f"{item['suspect_fraction']:.3f} | {item['cer_old_nfc']:.3f} | "
                     f"{item['cer_nfkc_alnum']:.3f} |")
    return lines + [""]


def render_addendum(summary: dict) -> str:
    overall = summary["overall"]
    lines = ["# X02 补充说明：CER 口径与句级质量", "",
             "原 [X02 报告](REPORT.md) 正文保持不变。本说明只更正 CER 统计口径，"
             "并提供 X03 截断敏感性分析所需的句级标记。", "",
             "新口径将参考文本和 SenseVoice 输出各自做 Unicode NFKC 归一化，"
             "只保留 Unicode 字母和数字（含汉字），字母转小写，再计算字符编辑距离。"
             "原口径为 NFC 加首尾去空白，保留标点。新口径只影响本说明的 CER 和质量标记，"
             "不追改 X02 的模型预测或原报告指标。", "",
             f"全部 {overall['total']:,} 句：原 CER {overall['cer_old_nfc']:.3f}，"
             f"新 CER {overall['cer_nfkc_alnum']:.3f}；疑似截断 "
             f"{overall['suspect']:,} 句（{overall['suspect_fraction']:.3f}）。", "",
             "`truncation_suspect` 严格按预注册规则计算：参考字数/音频秒数 > 9.0，"
             "或 ASR 字数/参考字数 < 0.5。阈值未根据结果调整。", "",
             "句级文件仅含统计量，不含任何转写文本；留在服务器："
             f"`{summary['quality_path']}`。SHA-256：`{summary['quality_sha256']}`。"
             "仓库保存 [聚合统计](results/quality_summary.json)。", ""]
    lines += table("按官方划分", summary["by_official_split"])
    lines += table("按评估池", summary["by_evaluation_cohort"])
    lines += table("按说话人", summary["by_speaker"])
    lines += table("按 M 类别", summary["by_class_M"])
    lines += ["## 解释边界", "", "这些标记提示音频片段、转写和标签可能不匹配；"
              "仅凭字数与语速不能判定单句确实被截断。X03 将在主结果之外"
              "按固定标记重算指标，不用该标记修改训练标签或调阈值。", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    started = dt.datetime.now(dt.timezone.utc)
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    if len(rows) != EXPECTED_COUNT or len({row["utt_id"] for row in rows}) != EXPECTED_COUNT:
        raise ValueError("EmotionTalk manifest is not the frozen 19,250-utterance set")
    overall = counter()
    by_split = defaultdict(counter)
    by_cohort = defaultdict(counter)
    by_speaker = defaultdict(counter)
    by_class = defaultdict(counter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = args.output_dir / "utterance_quality.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker = PersistentJSONWorker([sys.executable, "-u", "-m",
                                   "emotioncore.x02.quality_addendum", "--worker"],
                                  cwd=Path.cwd(), env=env,
                                  stderr_path=args.output_dir / "worker_errors.log")
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output_dir,
                                         prefix=".utterance_quality.", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                receipt_path = args.asr_receipts / f"{row['utt_id']}.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (receipt["status"] != "ok" or
                        receipt["input_sha256"] != row["wav"]["sha256"]):
                    raise ValueError(f"ASR receipt unavailable or mismatched: {row['utt_id']}")
                response = worker.request({
                    "utt_id": row["utt_id"],
                    "annotation": str(text_path(args.media_root, row["wav"]["path"])),
                    "transcript": receipt["transcript_path"],
                    "transcript_sha256": receipt["transcript_sha256"],
                    "duration_s": row["wav"]["duration_s"]}, args.timeout)
                if response["status"] != "ok" or response["utt_id"] != row["utt_id"]:
                    raise RuntimeError(f"Quality preprocessing failed: {row['utt_id']}: {response}")
                quality = {"utt_id": row["utt_id"],
                           "audio_duration_s": row["wav"]["duration_s"],
                           **{key: response[key] for key in
                              ("reference_chars", "reference_chars_per_s", "asr_chars",
                               "asr_reference_ratio", "cer_nfkc_alnum",
                               "truncation_suspect")}}
                stream.write(json.dumps(quality, ensure_ascii=False) + "\n")
                add(overall, response)
                add(by_split[row["official_split"]], response)
                add(by_cohort["CV_pool" if row["speaker_id"] not in HELD_OUT_SPEAKERS
                              else "held_out_speakers"], response)
                add(by_speaker[row["speaker_id"]], response)
                add(by_class[row["labels"]["M"]], response)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, quality_path)
    finally:
        worker.close()
    summary = {"normalization_old": "NFC, strip ends; punctuation retained",
               "normalization_new": "NFKC, lowercase, Unicode L/N characters only",
               "truncation_rule": "reference_chars_per_s > 9.0 OR asr_reference_ratio < 0.5",
               "quality_path": str(quality_path), "quality_sha256": sha256(quality_path),
               "manifest_path": str(args.manifest), "manifest_sha256": sha256(args.manifest),
               "overall": finish(overall),
               "by_official_split": {k: finish(v) for k, v in sorted(by_split.items())},
               "by_evaluation_cohort": {k: finish(v) for k, v in sorted(by_cohort.items())},
               "by_speaker": {k: finish(v) for k, v in sorted(by_speaker.items())},
               "by_class_M": {k: finish(v) for k, v in sorted(by_class.items())}}
    old = json.loads(args.old_cv_cer.read_text(encoding="utf-8"))
    if abs(summary["by_evaluation_cohort"]["CV_pool"]["cer_old_nfc"] -
           old["overall"]["cer_success_only"]) > 1e-12:
        raise ValueError("Recomputed original CV CER differs from frozen X02 receipt")
    write_json_atomic(args.output_dir / "summary.json", summary)
    addendum = args.output_dir / "ADDENDUM.md"
    addendum.write_text(render_addendum(summary), encoding="utf-8")
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": provenance, "command": sys.argv,
                       "started_at": started.isoformat(),
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "manifest_sha256": summary["manifest_sha256"],
                       "quality_sha256": summary["quality_sha256"],
                       "summary_sha256": sha256(args.output_dir / "summary.json"),
                       "addendum_sha256": sha256(addendum),
                       "transcripts_server_only": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--asr-receipts", type=Path)
    parser.add_argument("--old-cv-cer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main()
    else:
        for name in ("manifest", "media_root", "asr_receipts", "old_cv_cer", "output_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
