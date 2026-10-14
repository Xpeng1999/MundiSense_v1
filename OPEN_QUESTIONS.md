# Open questions for reviewer

1. The supplied CH-SIMS v2 `meta.csv` contains **4,403 data rows** (2,722 train, 647 valid, 1,034 test), while X01 specifies 4,402. The [official paper's Table 1](https://arxiv.org/pdf/2209.02604) has those same split counts and itself sums to 4,403, although its prose states 4,402. No CSV row is blank and `(video_id, clip_id)` has no duplicate. We retain all 4,403 rows for audit. Please confirm whether any official row should be excluded, with its ID and source, before changing Benchmark v1.
2. The pre-run versus mid-run legacy-project snapshots showed 172 new files: 166 Python `.pyc` caches and six files under `scripts/mer_download/` (`download_mer.py`, `index.html`, `mirror_status.sh`, `progress_server.ps1`, `progress_server.py`, `supervise.sh`). The same 172 files appear in the post-restart and final snapshots; no pre-existing file's size or mtime changed and no file was removed. The origin of the six non-cache files is not established by the snapshots. X01 resumed its own jobs with `PYTHONDONTWRITEBYTECODE=1`. The instruction prohibits deleting any old-project file, so all were preserved. Reviewer should decide how to record this acceptance-criterion deviation.
3. The supplied CH-SIMS v2 metadata has no speaker identifier, and the source material checked for X01 does not establish that the official `mode` split is speaker independent. No speaker-independence claim is made; reviewer can determine whether later work should seek official speaker metadata.
4. `aqgy4_0024__00004` passed full `ffmpeg` decode but stalled in Qwen AV preprocessing for both base and old LoRA. The 300-second `SIGALRM` could not interrupt the native call, so each run was stopped, given a failed per-sample receipt, and resumed. Both failures remain in the fixed denominator. Reviewer should decide whether a future benchmark revision needs process-isolated preprocessing; X01 does not change this frozen result.

## X04a

本轮未覆盖细节及保守口径集中记录于 [experiments/X04a_data/OPEN_QUESTIONS.md](experiments/X04a_data/OPEN_QUESTIONS.md)，不扩大语料、训练或推理范围。

## X05

D1 fold 0 的历史 FMH 冲突已按修订 1 解决：A3 逐值复现通过，新增 D1_fmhx 三专家均排除 06/10/11。X05 计算、测试和执行方独立复算已完成，证据与参数包见 [交付索引](experiments/X05_calib/DELIVERY.md)，等待 reviewer 验收，PR 不合入。

H_V 的 ET-test ECE 超过 0.05；MER-test H_V 缺既有 Q-V 概率；人脸质量五分位退化。详见 [X05 问题记录](experiments/X05_calib/OPEN_QUESTIONS.md) 与 [报告第 11 节](experiments/X05_calib/REPORT.md#11-给-reviewer-的问题)。


## X07：冻结后标签读取自检停止（2026-10-14）

L1 `6d9d07c` 后自查发现 valid 指标脚本会先读取整份 meta.csv 的 label 列再筛选 valid。已在执行该脚本和任何 GPU 推理前停止；原协议及代码不变，SIMS-test 未打开。标签隔离补丁只在独立审阅副本中测试，需明确授权新的冻结版本后才可应用并恢复。详情见 [X07 停止与修正提案](experiments/X07_lockbox/STOP_RECOVERY.md)。
