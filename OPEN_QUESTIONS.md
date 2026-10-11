# Open questions for reviewer

1. The supplied CH-SIMS v2 `meta.csv` contains **4,403 data rows** (2,722 train, 647 valid, 1,034 test), while X01 specifies 4,402. The [official paper's Table 1](https://arxiv.org/pdf/2209.02604) has those same split counts and itself sums to 4,403, although its prose states 4,402. No CSV row is blank and `(video_id, clip_id)` has no duplicate. We retain all 4,403 rows for audit. Please confirm whether any official row should be excluded, with its ID and source, before changing Benchmark v1.
2. The pre-run versus mid-run legacy-project snapshots showed 172 new files: 166 Python `.pyc` caches and six files under `scripts/mer_download/` (`download_mer.py`, `index.html`, `mirror_status.sh`, `progress_server.ps1`, `progress_server.py`, `supervise.sh`). The same 172 files appear in the post-restart and final snapshots; no pre-existing file's size or mtime changed and no file was removed. The origin of the six non-cache files is not established by the snapshots. X01 resumed its own jobs with `PYTHONDONTWRITEBYTECODE=1`. The instruction prohibits deleting any old-project file, so all were preserved. Reviewer should decide how to record this acceptance-criterion deviation.
3. The supplied CH-SIMS v2 metadata has no speaker identifier, and the source material checked for X01 does not establish that the official `mode` split is speaker independent. No speaker-independence claim is made; reviewer can determine whether later work should seek official speaker metadata.
4. `aqgy4_0024__00004` passed full `ffmpeg` decode but stalled in Qwen AV preprocessing for both base and old LoRA. The 300-second `SIGALRM` could not interrupt the native call, so each run was stopped, given a failed per-sample receipt, and resumed. Both failures remain in the fixed denominator. Reviewer should decide whether a future benchmark revision needs process-isolated preprocessing; X01 does not change this frozen result.

## X04a

本轮未覆盖细节及保守口径集中记录于 [experiments/X04a_data/OPEN_QUESTIONS.md](experiments/X04a_data/OPEN_QUESTIONS.md)，不扩大语料、训练或推理范围。

## X05

D1 fold 0 重拟合使用了 FMH 全部 3,266 句，与 X05 的专家回退假设冲突。已触发泄漏门禁并停止，详见 [experiments/X05_calib/OPEN_QUESTIONS.md](experiments/X05_calib/OPEN_QUESTIONS.md)。
