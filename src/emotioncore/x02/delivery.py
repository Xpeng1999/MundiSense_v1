"""Render X02 metrics, report, and X03 handoff from server-only receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from emotioncore.eval.metrics import percentile
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


HEAD_ENCODERS = ("A1", "A2", "A3", "V1", "V2", "V3a", "V3b")
ENCODER_SPECS = {
    "A1": ("audio", "1024 frame + 1024 utterance", "50 Hz"),
    "A2": ("audio", "3584 frame", "~25 Hz; actual frame counts in receipts"),
    "A3": ("audio", "4 × 1024 frame", "50 Hz"),
    "V1": ("visual", "1280 frame + 8 native logits", "YuNet 5 fps, up to 8 frames"),
    "V2": ("visual", "768 frame", "YuNet 5 fps, up to 8 frames"),
    "V3a": ("visual", "3584 frame", "YuNet 5 fps sample positions, whole frame 50176 pixels"),
    "V3b": ("visual", "3584 frame", "YuNet 5 fps sample positions, 448×448 face"),
    "T1": ("text", "128 token IDs, full fine-tuning", "utterance"),
    "T2": ("text", "1024 sentence vector", "utterance"),
}


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required X02 result missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def check_artifact(path: str, expected_sha: str) -> dict:
    file = Path(path)
    if sha256(file) != expected_sha:
        raise ValueError(f"Artifact SHA mismatch: {file}")
    return {"path": str(file), "sha256": expected_sha}


def choose(candidates: dict[str, dict]) -> dict:
    key = lambda item: (item[1]["macro_f1"], item[1]["uar"])
    winner = max(candidates.items(), key=key)
    return {"winner": winner[0], "selection_rule":
            "CV M macro-F1, UAR tie-break, first fixed candidate order for exact ties",
            "candidates": candidates}


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    root = args.run_root
    folds = read(args.folds)
    heads = {f"{encoder}_{head}": read(root / "heads" /
             f"{encoder}_{head}_cv" / "results.json")
             for encoder in HEAD_ENCODERS for head in ("h1", "h2")}
    t1 = read(root / "heads/T1_cv/results.json")
    t2 = read(root / "heads/T2_h1_cv/results.json")
    t2r = read(root / "heads/T2_h1_asr_replay/results.json")
    selected_heads = {}
    for encoder in HEAD_ENCODERS:
        selected_heads[encoder] = choose({
            head: {key: heads[f"{encoder}_{head}"]["metrics"]["M"][key]
                   for key in ("macro_f1", "uar")}
            for head in ("h1", "h2")})
    audio_champion = choose({encoder: selected_heads[encoder]["candidates"][
        selected_heads[encoder]["winner"]] for encoder in ("A1", "A2", "A3")})
    visual_champion = choose({encoder: selected_heads[encoder]["candidates"][
        selected_heads[encoder]["winner"]] for encoder in ("V1", "V2", "V3a", "V3b")})
    text_champion = choose({"T1": {key: t1["metrics"]["r"]["M"][key]
                                 for key in ("macro_f1", "uar")},
                            "T2": {key: t2r["metrics"]["M"][key]
                                   for key in ("macro_f1", "uar")}})
    selection = {"per_encoder_head": selected_heads,
                 "modality_champions": {"A": audio_champion,
                                        "V": visual_champion, "T_deployment_r": text_champion},
                 "note": "T champion uses ASR-input CV because deployment consumes ASR; all heads train on annotated text."}
    finals = {name: read(root / "final" / name / "results.json") for name in heads}
    finals["T1"] = read(root / "final/T1_test/results.json")
    finals["T2_h1"] = read(root / "final/T2_h1/results.json")
    finals["T2_r"] = read(root / "final/T2_h1_r/results.json")
    sims = {name: read(root / "sims_valid/eval" / name / "results.json") for name in heads}
    sims["T1"] = read(root / "sims_valid/eval/T1/results.json")
    sims["T2_h1"] = read(root / "sims_valid/eval/T2_h1/results.json")
    paired = {name: read(root / "metrics" / f"{name}.json") for name in
              ("v3_h1_b_minus_a", "v3_h2_b_minus_a", "t1_o_minus_r", "t2_o_minus_r")}
    face = read(root / "face/summary.json")
    cer = read(root / "metrics/asr_cer_cv.json")
    p1 = read(root / "probes/P1/results.json")
    p2 = read(root / "p2_probe_status.json")
    p2_result_path = root / "probes/P2/results.json"
    p2_result = read(p2_result_path) if p2_result_path.is_file() else None
    if (p2["status"] == "completed") != (p2_result is not None):
        raise ValueError("P2 status and probe result disagree")
    downloads = read(root / "model_downloads.json")
    modelscope_downloads = read(root / "modelscope_downloads.json")
    migrations = {name: read(root / f"migration_{name}.json")
                  for name in ("P2_modelscope", "V2_modelscope")}
    for name, receipt in migrations.items():
        link = root / "models" / name
        if not link.is_symlink() or link.resolve() != Path(receipt["target_path"]):
            raise ValueError(f"Model storage link differs from migration receipt: {name}")
    feature_meta = {}
    feature_runtime = {}
    for encoder in (*HEAD_ENCODERS, "T2_o", "T2_r"):
        folder = "V3" if encoder.startswith("V3") else encoder
        meta = read(root / "features" / folder / "run_meta_cv.json")
        if not meta.get("ended_at"):
            raise ValueError(f"Incomplete CV feature extraction: {encoder}")
        feature_meta[encoder] = meta
        elapsed = (dt.datetime.fromisoformat(meta["ended_at"]) -
                   dt.datetime.fromisoformat(meta["started_at"])).total_seconds()
        values = []
        for receipt_path in (root / "features" / encoder / "receipts").glob("*.json"):
            receipt = read(receipt_path)
            if receipt["status"] == "ok" and receipt.get("model_seconds") is not None:
                values.append(receipt["model_seconds"])
        feature_runtime[encoder] = {"cv_count": meta["count"],
                                    "elapsed_s": elapsed,
                                    "utterances_per_second": meta["count"] / elapsed,
                                    "model_latency_p50_s": percentile(values, 0.5) if values else None,
                                    "model_latency_p95_s": percentile(values, 0.95) if values else None,
                                    "latency_count": len(values)}
    for name, result in heads.items():
        check_artifact(result["oof_path"], result["oof_sha256"])
        check_artifact(finals[name]["prediction_path"], finals[name]["prediction_sha256"])
        check_artifact(sims[name]["prediction_path"], sims[name]["prediction_sha256"])
    check_artifact(t2["oof_path"], t2["oof_sha256"])
    check_artifact(t2r["oof_path"], t2r["oof_sha256"])
    for source in ("o", "r"):
        check_artifact(t1["oof"][source]["path"], t1["oof"][source]["sha256"])
        check_artifact(finals["T1"]["et_test"][source]["prediction_path"],
                       finals["T1"]["et_test"][source]["prediction_sha256"])
    artifact_index = {"features": {}, "predictions": {}, "run_meta": {},
                      "model_storage": {name: {"logical_path": str(root / "models" / name),
                                                "target_path": receipt["target_path"],
                                                "migration_receipt": str(root / f"migration_{name}.json")}
                                        for name, receipt in migrations.items()}}
    for encoder in (*HEAD_ENCODERS, "T2_o", "T2_r"):
        folder = "V3" if encoder.startswith("V3") else encoder
        path = root / "features" / folder / "run_meta_cv.json"
        artifact_index["features"][encoder] = {
            "receipts": str(root / "features" / encoder / "receipts"),
            "data": str(root / "features" / encoder / "data"),
            "run_meta": str(path), "run_meta_sha256": sha256(path),
            "dimension": ENCODER_SPECS[encoder.replace("_o", "").replace("_r", "")][1],
            "frame_rate": ENCODER_SPECS[encoder.replace("_o", "").replace("_r", "")][2]}
    artifact_index["features"]["T1_o"] = {
        "tokens": str(root / "tokens/T1_o_cv.npz"),
        "tokens_sha256": sha256(root / "tokens/T1_o_cv.npz")}
    artifact_index["features"]["T1_r"] = {
        "tokens": str(root / "tokens/T1_r_cv.npz"),
        "tokens_sha256": sha256(root / "tokens/T1_r_cv.npz")}
    for name in heads:
        artifact_index["predictions"][name] = {
            "oof": check_artifact(heads[name]["oof_path"], heads[name]["oof_sha256"]),
            "et_test": check_artifact(finals[name]["prediction_path"],
                                      finals[name]["prediction_sha256"]),
            "sims_valid": check_artifact(sims[name]["prediction_path"],
                                         sims[name]["prediction_sha256"])}
    artifact_index["predictions"]["T1"] = {
        source: {"oof": check_artifact(t1["oof"][source]["path"],
                                        t1["oof"][source]["sha256"]),
                 "et_test": check_artifact(finals["T1"]["et_test"][source]["prediction_path"],
                                           finals["T1"]["et_test"][source]["prediction_sha256"])}
        for source in ("o", "r")}
    artifact_index["predictions"]["T1"]["sims_valid"] = check_artifact(
        sims["T1"]["prediction_path"], sims["T1"]["prediction_sha256"])
    artifact_index["predictions"]["T2_r"] = {
        "oof": check_artifact(t2r["oof_path"], t2r["oof_sha256"]),
        "et_test": check_artifact(finals["T2_r"]["prediction_path"],
                                  finals["T2_r"]["prediction_sha256"])}
    artifact_index["predictions"]["T2_h1"] = {
        "oof": check_artifact(t2["oof_path"], t2["oof_sha256"]),
        "et_test": check_artifact(finals["T2_h1"]["prediction_path"],
                                  finals["T2_h1"]["prediction_sha256"]),
        "sims_valid": check_artifact(sims["T2_h1"]["prediction_path"],
                                     sims["T2_h1"]["prediction_sha256"])}
    all_metrics = {"selection": selection, "cv_heads": heads, "cv_T1": t1,
                   "cv_T2": t2, "cv_T2_r": t2r, "final": finals,
                   "sims_valid": sims, "paired": paired, "face": face,
                   "asr_cer": cer, "P1": p1, "P2": p2,
                   "P2_probe": p2_result,
                   "feature_runtime": feature_runtime,
                   "feature_run_meta": feature_meta,
                   "download_receipts": downloads,
                   "modelscope_download_receipts": modelscope_downloads,
                   "model_storage_migrations": migrations,
                   "folds_sha256": sha256(args.folds)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "all_metrics.json", all_metrics)
    write_json_atomic(args.output_dir / "artifact_index.json", artifact_index)
    write_json_atomic(args.output_dir / "selection.json", selection)
    report = render_report(folds, all_metrics, feature_meta, artifact_index)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    handoff = render_handoff(artifact_index, selection, root)
    (args.output_dir / "X03_HANDOFF.md").write_text(handoff, encoding="utf-8")
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": provenance, "command": ["emotioncore.x02.delivery"],
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "all_metrics_sha256": sha256(args.output_dir / "all_metrics.json"),
                       "artifact_index_sha256": sha256(args.output_dir / "artifact_index.json"),
                       "report_sha256": sha256(args.output_dir / "REPORT.md"),
                       "handoff_sha256": sha256(args.output_dir / "X03_HANDOFF.md")})


def render_report(folds: dict, data: dict, feature_meta: dict,
                  artifact_index: dict) -> str:
    sel, heads, finals, sims = (data[key] for key in
                               ("selection", "cv_heads", "final", "sims_valid"))
    face, cer, p1, p2, paired = (data[key] for key in
                                 ("face", "asr_cer", "P1", "P2", "paired"))
    lines = ["# X02 单模态强基线与模态信息量", "", "## 1. 摘要", "",
             f"- CV 池：{folds['cv_pool_count']:,} 句、{len(folds['cv_pool_speakers'])} 名说话人；ET-test：{folds['et_test_count']:,} 句。",
             f"- CV 冠军：A {sel['modality_champions']['A']['winner']}、V {sel['modality_champions']['V']['winner']}、T（ASR 输入）{sel['modality_champions']['T_deployment_r']['winner']}。",
             f"- P1 BF16 ET-val：A {f(p1['conditions']['audio']['M']['macro_f1'])}、AV {f(p1['conditions']['av']['M']['macro_f1'])} macro-F1。",
             (f"- P2 ET-val：A {f(data['P2_probe']['conditions']['audio']['M']['macro_f1'])}、"
              f"V {f(data['P2_probe']['conditions']['video']['M']['macro_f1'])}、"
              f"AV {f(data['P2_probe']['conditions']['av']['M']['macro_f1'])} macro-F1。"
              if data['P2_probe'] else f"- P2：{p2['status']}；{p2['reason']}。"),
             "- 大型 ModelScope 权重存于 H 盘，X02/models 为符号链接；概率、特征和媒体缓存留在服务器。仓库只保存指标与路径、SHA-256。", "",
             "## 2. 完成情况与偏离", "",
             ("**事实。** A1–A3、V1–V3a/b、T1–T2 的 CV、ET-test 与 SIMS-valid 结果见第 5 节；P1 已完成。"
              + ("P2 也已完成。" if data['P2_probe'] else f"P2 {p2['status']}：{p2['reason']}。")), "",
             "**解读。** P2 是零样本探针，不参与单模态冠军选择。", "",
             "## 3. CV 池与折划分统计", "",
             f"冻结文件 SHA-256：`{data['folds_sha256']}`。排除说话人：{', '.join(folds['excluded_speakers'])}。", "",
             "| 外折 | 留出说话人 | 句数 | 内验证说话人 | 句数 |", "|---:|---|---:|---|---:|"]
    for fold in folds["folds"]:
        lines.append(f"| {fold['fold']} | {', '.join(fold['outer_holdout_speakers'])} | {fold['outer_holdout_count']:,} | {', '.join(fold['inner_validation_speakers'])} | {fold['inner_validation_count']:,} |")
    lines += ["", "## 4. 人脸管线统计", "",
              "**事实。** YuNet 5 fps、IoU 轨迹、面积×出现率选目标脸、裁剪外扩 20%。",
              f"全体 {face['clips']:,} 句，无脸 {face['status_counts'].get('no_face', 0):,}（{f(face['no_face_fraction'])}），多脸 {face['multi_face_clips']:,}（{f(face['multi_face_fraction'])}）。",
              f"多脸目标/第二轨迹面积比 P10/P50/P90：{f(face['area_ratio']['p10'])}/{f(face['area_ratio']['p50'])}/{f(face['area_ratio']['p90'])}。{face['contact_count']} 个多脸句的联系表位于 `{Path(face['contact_selection']).parent / 'contact_sheets'}`。", "",
              "**解读。** 轨迹面积与出现率不能独立证明人物身份，联系表供 reviewer 目视核验。", "",
              "## 5. A / V / T 结果", "", "### 事实：CV", "",
              "| 编码器 | head / 来源 | M macro-F1 | M UAR | M WA | 原生标签 macro-F1 | 覆盖率 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in heads.items():
        m, native = result["metrics"]["M"], result["metrics"]["native"]
        lines.append(f"| {result['encoder']} | {result['head']} | {f(m['macro_f1'])} | {f(m['uar'])} | {f(m['wa'])} | {f(native['macro_f1'])} | {f(m['coverage'])} |")
    for label, result, source in (("T1", data["cv_T1"], "o"), ("T1", data["cv_T1"], "r")):
        m, native = result["metrics"][source]["M"], result["metrics"][source]["native"]
        lines.append(f"| {label} | {source} | {f(m['macro_f1'])} | {f(m['uar'])} | {f(m['wa'])} | {f(native['macro_f1'])} | {f(m['coverage'])} |")
    for label, result, source in (("T2", data["cv_T2"], "o"), ("T2", data["cv_T2_r"], "r")):
        m, native = result["metrics"]["M"], result["metrics"]["native"]
        lines.append(f"| {label} | {source} | {f(m['macro_f1'])} | {f(m['uar'])} | {f(m['wa'])} | {f(native['macro_f1'])} | {f(m['coverage'])} |")
    lines += ["", "### 事实：ET-test", "",
              "| 编码器 | head / 来源 | M macro-F1 | M UAR | M WA | 原生标签 macro-F1 | 覆盖率 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in finals.items():
        if name == "T1":
            for source in ("o", "r"):
                m = result["et_test"][source]["metrics"]["M"]
                native = result["et_test"][source]["metrics"]["native"]
                lines.append(f"| T1 | {source} | {f(m['macro_f1'])} | {f(m['uar'])} | {f(m['wa'])} | {f(native['macro_f1'])} | {f(m['coverage'])} |")
        else:
            m = result["et_test"]["M"]
            native = result["et_test"]["native"]
            lines.append(f"| {name.split('_')[0]} | {'r' if name == 'T2_r' else result['head']} | {f(m['macro_f1'])} | {f(m['uar'])} | {f(m['wa'])} | {f(native['macro_f1'])} | {f(m['coverage'])} |")
    lines += ["", "### 事实：SIMS-valid", "",
              "| 编码器 / head | Pearson | Spearman | Acc-2 非零 | 分数覆盖率 | 特征覆盖率 |", "|---|---:|---:|---:|---:|---:|"]
    for name, result in sims.items():
        m = result["metrics"]
        lines.append(f"| {name} | {f(m['pearson'])} | {f(m['spearman'])} | {f(m['acc_2_nonzero'])} | {f(m['coverage'])} | {f(result['feature_coverage'])} |")
    lines += ["", "**解读。** 主表使用 M 标签；原生标签并列展示，不做跨真值相减。所有超参由外折内部验证选出。逐类指标、混淆矩阵、按说话人结果、整群 bootstrap 区间在 `all_metrics.json`。", "",
              "## 6. 分辨率对照（V3b vs V3a）", "",
              "**事实。** 同一 head、同一 CV 句子，按 15 名说话人整群重采样 2,000 次。", "",
              "| head | b−a macro-F1 | 95% 区间 | b−a UAR |", "|---|---:|---|---:|"]
    for head in ("h1", "h2"):
        values = paired[f"v3_{head}_b_minus_a"]["left_minus_right"]
        macro, uar = values["macro_f1"], values["uar"]
        lines.append(f"| {head} | {f(macro['difference'])} | [{f(macro['lower_95'])}, {f(macro['upper_95'])}] | {f(uar['difference'])} |")
    lines += ["", "**解读。** 只比较输入分辨率与裁剪的作用；显著性判断以配对区间和联系表为依据。", "",
              "## 7. 标注转写 vs 自动识别", "",
              f"**事实。** SenseVoice CV CER={f(cer['overall']['cer_success_only'])}，成功 {cer['overall']['success']:,}/{cer['overall']['total']:,}；按说话人 CER 在 `all_metrics.json`。", "",
              "| 编码器 | o−r CV M macro-F1 | 95% 区间（说话人整群） |", "|---|---:|---|"]
    for encoder in ("t1", "t2"):
        value = paired[f"{encoder}_o_minus_r"]["left_minus_right"]["macro_f1"]
        lines.append(f"| {encoder.upper()} | {f(value['difference'])} | [{f(value['lower_95'])}, {f(value['upper_95'])}] |")
    lines += ["", "**解读。** 两个文本模型只用 `(o)` 训练，`(r)` 是相同模型的回放；部署预期以 `(r)` 为准。", "",
              "## 8. P1 / P2 探针", "",
              "| 探针 | 条件 | ET-val M macro-F1 | 对照 NF4 | 差值 95% 配对区间 |", "|---|---|---:|---:|---|"]
    for condition in ("audio", "av"):
        item = p1["conditions"][condition]
        comparison = item["bf16_minus_nf4_macro_f1"]
        low, high = comparison["left_minus_right_95"]
        lines.append(f"| P1 BF16 | {condition} | {f(item['M']['macro_f1'])} | {f(item['x01_nf4_M']['macro_f1'])} | [{f(low)}, {f(high)}] |")
    if data["P2_probe"]:
        for condition in ("audio", "video", "av"):
            item = data["P2_probe"]["conditions"][condition]
            lines.append(f"| P2 NF4 thinker | {condition} | {f(item['M']['macro_f1'])} | — | — |")
        lines += ["", f"P2：7 个首 token 两两不同；ModelScope 权重下载与加载收据见 `p2_probe_status.json`；三种条件均在 ET-val 评分。", ""]
    else:
        lines += ["", f"P2：7 个首 token 两两不同；{p2['status']}；{p2['reason']}。证据见 `p2_probe_status.json`。", ""]
    lines += [
              "## 9. 与 X01 旧 LoRA 的 ET-test 对照", "",
              "**事实。** X01 旧 LoRA 的 ET-test macro-F1：A 0.603、V 0.287、AV 0.600。X02 的同模态 ET-test 结果见第 5 节。", "",
              "**解读。** 训练数据与模型条件不同，这里只作描述性对照，不作因果归因。", "",
              "## 10. 耗时与显存", "",
              "| 特征提取 | CV 句数 | 耗时（小时） | 每秒句数 | 模型 P50/P95（秒） |", "|---|---:|---:|---:|---:|"]
    for encoder, runtime in data["feature_runtime"].items():
        if encoder == "V3b":
            continue
        label = "V3a+V3b paired" if encoder == "V3a" else encoder
        lines.append(f"| {label} | {runtime['cv_count']:,} | {runtime['elapsed_s']/3600:.2f} | {runtime['utterances_per_second']:.2f} | {f(runtime['model_latency_p50_s'])} / {f(runtime['model_latency_p95_s'])} |")
    lines += ["", "P1 条件推理：", "", "| 条件 | 延迟 P50 / P95（秒） | 峰值 CUDA MiB |", "|---|---:|---:|"]
    for condition in ("audio", "av"):
        item = p1["conditions"][condition]
        lines.append(f"| {condition} | {f(item['latency_p50_s'])} / {f(item['latency_p95_s'])} | {item['peak_cuda_allocated_mib']:.0f} |")
    if data["P2_probe"]:
        lines += ["", "P2 条件推理：", "", "| 条件 | 延迟 P50 / P95（秒） | 峰值 CUDA MiB |", "|---|---:|---:|"]
        for condition in ("audio", "video", "av"):
            item = data["P2_probe"]["conditions"][condition]
            lines.append(f"| {condition} | {f(item['latency_p50_s'])} / {f(item['latency_p95_s'])} | {item['peak_cuda_allocated_mib']:.0f} |")
    lines += ["", "## 11. 异常与未解决问题", "",
              f"- CH-SIMS valid 缺源视频 27/647；A/V 失败保留在 Acc-2 非零分母，相关系数附覆盖率。",
              ("- P2 在 32 GB 显存内用 NF4 thinker 推理；加载及运行收据见 `p2_probe_status.json`。"
               if data["P2_probe"] else f"- P2：{p2['status']}；{p2['reason']}。证据在 `p2_probe_status.json`。"),
              "- T2 曾误启动一次 ASR 输入训练，已立即终止并隔离到服务器 `invalid_runs`；正式 `(r)` 结果由 `(o)` 模型回放，原始 `(o)` OOF 概率逐项核对一致。",
              "- CH-SIMS 无可用说话人 ID，跨语料只报告相关与 Acc-2，不声称说话人独立区间。", "",
              "## 12. 给 reviewer 的问题", "",
              "1. 是否接受第 11 节的缺失媒体分母与覆盖率口径？",
              "2. T 冠军按实际部署 ASR 输入的 CV 分数选择，是否符合后续 X03 使用方式？",
              ("3. P2 零样本表现是否足以进入 X03 主干候选？"
               if data["P2_probe"] else "3. 是否需要另开 P2 32 GB 显存加载验证？"), "",
              "所有聚合结果由 `emotioncore.x02.delivery` 从服务器收据生成；特征、逐句概率与完整转写均不入库。", ""]
    return "\n".join(lines)


def render_handoff(index: dict, selection: dict, root: Path) -> str:
    lines = ["# X03 对接说明", "",
             "X02 所有缓存留在服务器。先核对 `artifact_index.json` 的路径和 SHA-256；OOF 与 ET-test/SIMS-valid 预测均为 JSONL。",
             "", "## 大型模型路径", "",
             "| 用途 | X02 逻辑路径 | H 盘实际路径 |", "|---|---|---|"]
    for name, item in index["model_storage"].items():
        lines.append(f"| {name} | `{item['logical_path']}` | `{item['target_path']}` |")
    lines += ["", "V2 主基线继续使用 `openai/clip-vit-large-patch14`；ModelScope CLIP-GmP 是不同变体，未用于 X02 主表。",
             "", "## 编码器特征", "",
             "| 编码器 | 模态 | 维度 | 帧率 / 粒度 | 收据目录 |",
             "|---|---|---|---|---|"]
    for encoder, (modality, dimension, rate) in ENCODER_SPECS.items():
        if encoder == "T1":
            folder = root / "tokens"
        else:
            folder = root / "features" / ("T2_o" if encoder == "T2" else encoder) / "receipts"
        lines.append(f"| {encoder} | {modality} | {dimension} | {rate} | `{folder}` |")
    lines += ["", "T1 `(o)/(r)` token缓存、T2 `(o)/(r)` 句向量缓存均在服务器。T1 与 T2 只用 `(o)` 训练。V3a 为整帧 50176 像素，V3b 为 448×448 人脸；两者共享视觉塔。", "",
              "## 预测索引", "",
              "| 配置 | OOF | ET-test | SIMS-valid |", "|---|---|---|---|"]
    for name, item in index["predictions"].items():
        if name == "T1":
            for source in ("o", "r"):
                val = item[source]
                lines.append(f"| T1_{source} | `{val['oof']['path']}` | `{val['et_test']['path']}` | {'`' + item['sims_valid']['path'] + '`' if source == 'o' else '—'} |")
        else:
            lines.append(f"| {name} | `{item['oof']['path']}` | `{item['et_test']['path']}` | {'`' + item['sims_valid']['path'] + '`' if item.get('sims_valid') else '—'} |")
    lines += ["", "## 选择规则", "",
              "模态冠军仅按 CV M macro-F1 选择，平分用 UAR；文字按 ASR 输入 `(r)` 的 CV 分数。每个 head 的学习率、权重衰减与 epoch 由内验证选择。选择记录见 `selection.json`。",
              "", "## 校验与边界", "",
              "- `configs/folds_v1.json` 是冻结划分；OOF 每句一次，CV 池无 02/13/20/21。",
              "- 特征收据含输入媒体 SHA-256、编码器与预处理版本、特征文件 SHA-256。",
              "- 缺脸/失败概率为均匀分布且状态单列；融合时请保留状态与覆盖率。",
              "- CH-SIMS test 封存，X03 不得从该集合读取媒体、转写或特征。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
