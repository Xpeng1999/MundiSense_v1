from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotioncore.utils.common import sha256_file


def get(results: Path, name: str):
    return json.loads((results / name).read_text(encoding="utf-8"))


def num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci(result: dict, metric: str, unit: str = "utterance") -> str:
    interval = result[f"{metric}_{unit}_ci"]
    return f"[{num(interval['lower_95'])}, {num(interval['upper_95'])}]"


def render(results: Path, output: Path) -> None:
    et = get(results, "et_audit.json")
    sims = get(results, "chsims_audit.json")
    decode = get(results, "chsims_decode_audit.json")
    incidents = get(results, "d2_incidents.json")
    d3 = get(results, "d3_consistency.json")
    v16 = get(results, "leakage_v16.json")
    correction = get(results, "legacy_correction.json")
    freeze = get(results, "benchmark_freeze.json")
    execution = get(results, "execution_config.json")
    metrics = get(results, "metrics.json")
    faces = get(results, "face_audit.json")
    environment = get(results, "environment_compare.json")
    new_environment = get(results, "new_venv_verification.json")
    legacy = get(results, "legacy_snapshot_audit.json")
    predictions = get(results, "prediction_manifest.json")
    if decode["rows"] != sims["rows"] or sum(sum(row.values()) for row in decode["counts_by_mode"].values()) != sims["rows"]:
        raise RuntimeError("Full decode audit does not cover the CH-SIMS source manifest")
    if len(predictions) != sum(x["n"] for x in metrics["et"].values()) + sum(x["n"] for x in metrics["sims_valid"].values()):
        raise RuntimeError("Prediction manifest does not cover all D2 rows")
    test_results = get(results, "test_results.json")
    run_revisions = sorted({(row["git"]["commit"], row["git"]["dirty"])
                            for row in metrics["run_meta"].values()})

    lines = ["# X01 工程基座与评测基准 v1 — 执行报告", "",
             "## 1. 摘要", "",
             f"- EmotionTalk 登记 {et['rows']:,} 条，官方各划分和各类计数差异 {len(et['official_differences'])} 格。",
             f"- CH-SIMS v2 源 CSV 有 {sims['rows']:,} 条；缺媒体 {sims['missing_count']} 条，重复路径 {sims['duplicate_path_count']} 条，完整解码失败 {sum(x.get('decode_failed', 0) for x in decode['counts_by_mode'].values())} 条。",
             f"- 旧训练对 ET-test/ET-val 四层重叠均为零；第 16 轮中文集同句 {v16['overlap_counts']['utterance']} 条、同媒体哈希 {v16['overlap_counts']['media_sha256']} 条。",
             f"- D3 完成 {d3['completed']}/{d3['required']} 条，最大 logprob 差 {num(max(x['max_abs_logprob_difference'] for x in d3['details']), 6)}；完整 D2 结果见第 6 节。",
             f"- 人脸审计中 ET-val 帧的多脸比例为 {num(faces['emotiontalk']['by_split']['val']['fractions']['2+'])}，SIMS-valid 为 {num(faces['chsims_v2']['by_split']['valid']['fractions']['2+'])}。",
             f"- 旧项目快照新增 {legacy['added_count']} 个缓存文件；已有文件变化 {legacy['changed_existing_count']}，详见第 7 节。",
             "", "## 2. 实际完成情况与偏离计划之处", "", "### 事实", "",
             f"- 已建私有仓库、Mac 和服务器克隆；D2 每个组合均有独立 `run_meta.json`。本次收录 {len(predictions):,} 份服务器逐条预测的路径与 SHA-256。",
             "- 服务器重启中断了长任务；D2 按已落盘样本 ID 续跑，人脸审计改用服务器运行目录中的逐片段原子凭证续跑。",
             "- D2 运行时的 Git 状态：" + "；".join(
                 f"`{commit[:12]}`、工作区{'有' if dirty else '无'}未提交改动"
                 for commit, dirty in run_revisions) + "；每组完整元数据在服务器 `run_meta.json`。",
             f"- 新独立 Python 3.11 虚拟环境已具备 CUDA；锁中 {environment['locked_packages']} 个包与旧安装版本逐项相同（旧环境不一致 {environment['mismatch_count']} 个，新环境不一致 {new_environment['mismatch_count']} 个）。本次 D2 在新环境配齐前已用旧解释器启动并保持同一环境，未向旧环境安装包。",
             f"- 新环境 `pip check` 返回 {new_environment['pip_check_exit_code']}：{new_environment['pip_check_output'].strip() or '无冲突'}。该依赖冲突来自所复刻的既有包组合；GPU 环境验证在 `new_venv_verification.json`。",
             f"- `pytest`：{test_results['passed']} 通过、{test_results['failed']} 失败、{test_results['skipped']} 跳过。",
             f"- 计划写 CH-SIMS v2 {sims['expected_rows_in_x01']:,} 条，源文件实际 {sims['rows']:,} 条；未删除任何源行。",
             "", "### 解读", "",
             "- CH-SIMS v2 的官方论文表格划分数相加与本地 CSV 一致，论文文字总数不同；这个源材料差异留给 reviewer 确认。",
             "- 旧项目中新出现的缓存文件违反了零变更验收项。发现后已将仍在运行的任务改为禁止写入字节码并验证无新增变化；没有删除旧目录内文件。",
             "", "## 3. 数据审计要点", "", "### 事实", "",
             "| 模态 | M 一致数/总数 | 一致率 | Fleiss κ | 5 票一致数 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for mod in ("A", "V", "T"):
        x = et["m_vs_other"][mod]
        lines.append(f"| {mod} | {x['same']}/{x['total']} | {num(x['rate'])} | {num(et['fleiss_kappa'][mod])} | {et['majority_strength'][mod]['5']} |")
    lines += [f"| M | — | — | {num(et['fleiss_kappa']['M'])} | {et['majority_strength']['M']['5']} |",
              "", "- 模态两两标签一致率：" + "；".join(
                  f"{a}/{b} {num(et['pairwise_agreement'][a][b]['rate'])}"
                  for a, b in (("M", "A"), ("M", "V"), ("M", "T"),
                               ("A", "V"), ("A", "T"), ("V", "T"))) + "。",
              "", "- 各类别一致率、全部投票强度和 19 名说话人与组的关系见 `et_audit.json`。",
              f"- CH-SIMS v2：train {sims['mode_counts']['train']:,}、valid {sims['mode_counts']['valid']:,}、test {sims['mode_counts']['test']:,}；现存媒体 `ffprobe` 失败 {sims['decode_failure_count']} 条。完整 `ffmpeg` 解码结果如下；test 只作清单、SHA-256 与解码登记，未作模型推理。",
              "", "| 划分 | 解码成功 | 缺媒体 | 探测失败 | 解码失败 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for mode in ("train", "valid", "test"):
        row = decode["counts_by_mode"][mode]
        lines.append(f"| {mode} | {row.get('ok', 0)} | {row.get('missing', 0)} | {row.get('probe_failed', 0)} | {row.get('decode_failed', 0)} |")
    lines += ["",
              "- 2-fps YuNet 画面审计（零脸／一脸／多脸为帧比例）：", "",
              "| 数据 | 片段 | 0 脸 | 1 脸 | ≥2 脸 | 抽帧失败次数（含缺媒体） |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("emotiontalk", "chsims_v2"):
        f = faces[name]
        fr = f["frames"]["fractions"]
        lines.append(f"| {name} | {f['clips']:,} | {num(fr['0'])} | {num(fr['1'])} | {num(fr['2+'])} | {f['decode_failures']} |")
    lines += ["", "### 解读", "",
              "- 脸数不等同于当前说话人的身份；CH-SIMS v2 的 `meta.csv` 无说话人 ID，不能据此声称官方划分与说话人无关。",
              f"- EmotionTalk val 的多脸帧比例为 {num(faces['emotiontalk']['by_split']['val']['fractions']['2+'])}，说话人 15 对应受审计帧为 {num(faces['emotiontalk']['by_speaker']['15']['fractions']['2+'])}；单一可见脸假设并不普遍成立。此计数不能判定多脸帧中的当前说话人。",
              "", "## 4. Benchmark v1 定义摘要及 sha256", "",
              f"- 定义文件 SHA-256：`{freeze['definition_sha256']}`。ET-test 为主评测，ET-val 留作选择和校准；SIMS-valid 评估连续效价；CH-SIMS v2 test 是封存集。",
              "- ET 主标签为 M 的七类；拒识和解析失败计入错误。SIMS-valid 的概率来自八个候选首 token 对数概率的 softmax，以预定义概率差计算 Pearson、Spearman 和排除零标签的 Acc-2，不调阈值。",
              f"- D2 配置：`VIDEO_MAX_PIXELS={execution['values']['video_max_pixels']}`、`FPS_MAX_FRAMES={execution['values']['fps_max_frames']}`、`MAX_PIXELS={execution['values']['max_pixels']}`；{execution['values']['quantization']}、{execution['values']['compute_dtype']}、{execution['values']['attention']}。配置 SHA-256：`{execution['sha256']}`。system/user 提示词取对应模态旧训练清单首行，逐字保持。",
              "", "| 清单 | 条数 | SHA-256 |", "| --- | ---: | --- |"]
    for name, entry in freeze["manifests"].items():
        lines.append(f"| {name} | {entry['rows']:,} | `{entry['sha256']}` |")
    lines += ["", "## 5. 泄漏核查与对旧结论的更正", "", "### 事实", "",
              "- ET-test 与 ET-val 相对旧 AV/A 训练清单的同句、同媒体哈希、同说话人、同对话重叠均为 0；完整核查在 `leakage_et_test.json` 和 `leakage_et_val.json`。",
              f"- 第 16 轮 252 条中文集：同句 {v16['overlap_counts']['utterance']}、同媒体哈希 {v16['overlap_counts']['media_sha256']}、同说话人 {v16['overlap_counts']['speaker']}、同对话 {v16['overlap_counts']['dialogue']}。",
              "- 旧 qwen_old 的 M 标签 macro-F1，‘干净’定义为四层均无重叠，‘泄漏’为任一层有重叠；原有脸门控拒识保留在分母。", "",
              "| 条件 | 干净 n | 干净 macro-F1 | 泄漏 n | 泄漏 macro-F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for condition in ("audio", "video", "av"):
        groups = correction["groups"][condition]
        clean, leaked = groups["clean_all_levels"], groups["leaked_any_level"]
        lines.append(f"| {condition} | {clean['n']} | {num(clean['macro_f1'])} | {leaked['n']} | {num(leaked['macro_f1'])} |")
    lines += ["", "### 解读", "",
              "- 第 16 轮中文整体分数不能作为未泄漏的独立泛化证据；干净与泄漏组的样本和拒识率也不同，表中差异不是泄漏的因果效应。",
              "", "## 6. 基线结果", "", "### 事实：ET 主标签", "",
              "表内每个指标后依次给按句／按说话人 bootstrap 95% 区间（各 2,000 次）。测试集只有四名说话人，说话人区间仅描述性。", "",
              "| 模型 | 集合 | 条件 | n | 失败 | macro-F1 [句]/[人] | UAR [句]/[人] | WA [句]/[人] | 覆盖率 | 中性误报 |",
              "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |"]
    for dataset in ("et_test", "et_val"):
        for model in ("base", "old_lora"):
            for condition in ("audio", "video", "av"):
                r = metrics["et"][f"{model}_{dataset}_{condition}"]
                def cell(metric):
                    return f"{num(r[metric])} {ci(r, metric)}/{ci(r, metric, 'speaker')}"
                lines.append(f"| {model} | {dataset} | {condition} | {r['n']:,} | {r['failures']} | {cell('macro_f1')} | {cell('uar')} | {cell('wa')} | {num(r['coverage'])} | {num(r['neutral_false_positive_rate'])} |")
    lines += ["", "### 事实：逐说话人 ET 结果", "",
              "| 模型 | 集合 | 条件 | 说话人 | n | macro-F1 | UAR | WA | 覆盖率 |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in ("et_test", "et_val"):
        for model in ("base", "old_lora"):
            for condition in ("audio", "video", "av"):
                for speaker, r in metrics["et"][f"{model}_{dataset}_{condition}"]["by_speaker"].items():
                    lines.append(f"| {model} | {dataset} | {condition} | {speaker} | {r['n']:,} | {num(r['macro_f1'])} | {num(r['uar'])} | {num(r['wa'])} | {num(r['coverage'])} |")
    lines += ["", "### 事实：A/V 原生标签（与 M 结果并列，不作相减）", "",
              "| 模型 | 集合 | 条件 | 真值 | macro-F1 | UAR | WA |",
              "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for dataset in ("et_test", "et_val"):
        for model in ("base", "old_lora"):
            for condition, truth in (("audio", "A"), ("video", "V")):
                name = f"{model}_{dataset}_{condition}"
                for label, r in (("M", metrics["et"][name]), (truth, metrics["native_labels"][name])):
                    lines.append(f"| {model} | {dataset} | {condition} | {label} | {num(r['macro_f1'])} | {num(r['uar'])} | {num(r['wa'])} |")
    lines += ["", "### 事实：八类 argmax 次要结果（`other` 计错）", "",
              "| 模型 | 集合 | 条件 | macro-F1 | UAR | WA | 有效七类输出率 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset in ("et_test", "et_val"):
        for model in ("base", "old_lora"):
            for condition in ("audio", "video", "av"):
                r = metrics["et"][f"{model}_{dataset}_{condition}"]["with_other"]
                lines.append(f"| {model} | {dataset} | {condition} | {num(r['macro_f1'])} | {num(r['uar'])} | {num(r['wa'])} | {num(r['coverage'])} |")
    lines += ["", "### 事实：SIMS-valid", "",
              "| 模型 | 条件 | n | 失败 | 覆盖率 | 相关 n | Acc-2 非零 n | Pearson [95%] | Spearman [95%] | Acc-2 非零 [95%] |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for model in ("base", "old_lora"):
        for condition in ("audio", "av"):
            r = metrics["sims_valid"][f"{model}_sims_valid_{condition}"]
            def cell(metric):
                return f"{num(r[metric])} {ci(r, metric)}" if r[metric] is not None else "—"
            lines.append(f"| {model} | {condition} | {r['n']:,} | {r['n'] - r['valid']} | {num(r['coverage'])} | {r['correlation_denominator']} | {r['nonzero_denominator']} | {cell('pearson')} | {cell('spearman')} | {cell('acc_2_nonzero')} |")
    lines += ["", "### 事实：失败原因", "",
              "| 组合 | 原因 | 条数 |", "| --- | --- | ---: |"]
    for section in ("et", "sims_valid"):
        for name, result in metrics[section].items():
            for reason, count in sorted(result["failure_reasons"].items()):
                lines.append(f"| {name} | {reason} | {count} |")
    lines += ["", "### 事实：配对差值（macro-F1）", "",
              "| 左减右 | n | 差值 | 配对 95% 区间 |", "| --- | ---: | ---: | --- |"]
    for name, r in metrics["paired"].items():
        lines.append(f"| {name} | {r['n']:,} | {num(r['difference'])} | [{num(r['left_minus_right_95'][0])}, {num(r['left_minus_right_95'][1])}] |")
    lines += ["", "SIMS-valid 同样按相同样本做 A/AV 与基座/LoRA 配对比较：", "",
              "| 左减右及指标 | n | 差值 | 配对 95% 区间 |", "| --- | ---: | ---: | --- |"]
    for name, r in metrics["paired_sims"].items():
        lines.append(f"| {name} | {r['n']:,} | {num(r['difference'])} | [{num(r['left_minus_right_95'][0])}, {num(r['left_minus_right_95'][1])}] |")
    lines += ["", "### 事实：说话人、困难类别与耗时", "",
              "- 各组合的完整七类 P/R/F1、混淆矩阵、说话人结果、A/V 原生标签结果和包含 `other` 的次要结果均在 `metrics.json`。下面列 ET-test 的重点召回：", "",
              "| 模型 | 条件 | fear | disgust | surprise | neutral |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model in ("base", "old_lora"):
        for condition in ("audio", "video", "av"):
            r = metrics["et"][f"{model}_et_test_{condition}"]["per_class"]
            lines.append(f"| {model} | {condition} | {num(r['fear']['recall'])} | {num(r['disgust']['recall'])} | {num(r['surprise']['recall'])} | {num(r['neutral']['recall'])} |")
    lines += ["", "| 模型 | 集合 | 条件 | P50 秒 | P95 秒 | 显存峰值 MiB |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for section in ("et", "sims_valid"):
        for name, r in metrics[section].items():
            model = "old_lora" if name.startswith("old_lora_") else "base"
            rest = name[len(model) + 1:]
            timing = r["runtime"]
            lines.append(f"| {model} | {rest.rsplit('_', 1)[0]} | {rest.rsplit('_', 1)[1]} | {num(timing['latency_p50_s'])} | {num(timing['latency_p95_s'])} | {num(timing['peak_cuda_allocated_mib'], 1)} |")
    lines += ["", "### 解读", "",
              "- A、V、AV 与基座／旧 LoRA 的差值只在相同样本和相同 M 标签下解释；A 与 A 标签、V 与 V 标签结果并列保留，不作跨真值相减。SIMS-valid 的情绪概率差是跨任务代理量。相关系数仅对有效评分计算，覆盖率和相关 n 同时列出；Acc-2 非零 n 保留失败样本并计错。",
              "", "## 7. 异常与未解决问题", "",
              f"- CH-SIMS v2 的 4,402／4,403 源材料差异、{sims['missing_count']} 条缺媒体和未验证说话人独立性见数据卡及 `OPEN_QUESTIONS.md`。",
              f"- D2 对单条预处理和前向计算设 {execution['per_sample_timeout_s']} 秒运行时限；超时写失败凭证并保留在指标分母，具体组合和次数见第 6 节失败原因表。",
              f"- 旧项目快照：新增 {legacy['added_count']}、既有文件 mtime/大小变化 {legacy['changed_existing_count']}、删除 {legacy['removed_count']}。新增项是 Python 缓存，已保留并停止后续写入；这项验收未通过。",
              "- 所有逐条预测只在服务器运行目录；仓库仅有逐条路径及 SHA-256 清单。"]
    for item in incidents["incidents"]:
        lines.append(f"- {item['model']} {item['dataset']} {item['condition']} 的 `{item['sample_id']}` 媒体时长 {num(item['media_duration_s'])} 秒；处理至少 {item['elapsed_lower_bound_s']} 秒无输出，`ffmpeg` 完整解码成功。已写失败凭证并断点续跑；凭证 SHA-256 为 `{item['receipt_sha256']}`。")
    lines += ["", "## 8. 给 reviewer 的问题", "",
              "1. 官方论文文字写 4,402、Table 1 和所给 CSV 为 4,403；是否以当前 4,403 行作为 Benchmark v1 的 SIMS-valid/lock-box 固定来源？",
              "2. 旧项目因 Python 自动字节码生成而新增缓存文件，已有文件未变；请决定这一只读边界偏离的验收记录方式。",
              "3. CH-SIMS v2 无说话人 ID，官方来源未确立说话人独立性；是否需要后续单独获取说话人元数据？",
              ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.output)


if __name__ == "__main__":
    main()
