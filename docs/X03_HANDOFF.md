# X03 对接说明

X02 所有缓存留在服务器。先核对 `artifact_index.json` 的路径和 SHA-256；OOF 与 ET-test/SIMS-valid 预测均为 JSONL。

## 大型模型路径

| 用途 | X02 逻辑路径 | H 盘实际路径 |
|---|---|---|
| P2_modelscope | `/home/administrator/emotion-data/emotioncore_runs/X02/models/P2_modelscope` | `/mnt/h/emotionAI/emotioncore_models/X02/P2_modelscope` |
| V2_modelscope | `/home/administrator/emotion-data/emotioncore_runs/X02/models/V2_modelscope` | `/mnt/h/emotionAI/emotioncore_models/X02/V2_modelscope` |

V2 主基线继续使用 `openai/clip-vit-large-patch14`；ModelScope CLIP-GmP 是不同变体，未用于 X02 主表。

## 编码器特征

| 编码器 | 模态 | 维度 | 帧率 / 粒度 | 收据目录 |
|---|---|---|---|---|
| A1 | audio | 1024 frame + 1024 utterance | 50 Hz | `/home/administrator/emotion-data/emotioncore_runs/X02/features/A1/receipts` |
| A2 | audio | 3584 frame | ~25 Hz; actual frame counts in receipts | `/home/administrator/emotion-data/emotioncore_runs/X02/features/A2/receipts` |
| A3 | audio | 4 × 1024 frame | 50 Hz | `/home/administrator/emotion-data/emotioncore_runs/X02/features/A3/receipts` |
| V1 | visual | 1280 frame + 8 native logits | YuNet 5 fps, up to 8 frames | `/home/administrator/emotion-data/emotioncore_runs/X02/features/V1/receipts` |
| V2 | visual | 768 frame | YuNet 5 fps, up to 8 frames | `/home/administrator/emotion-data/emotioncore_runs/X02/features/V2/receipts` |
| V3a | visual | 3584 frame | YuNet 5 fps sample positions, whole frame 50176 pixels | `/home/administrator/emotion-data/emotioncore_runs/X02/features/V3a/receipts` |
| V3b | visual | 3584 frame | YuNet 5 fps sample positions, 448×448 face | `/home/administrator/emotion-data/emotioncore_runs/X02/features/V3b/receipts` |
| T1 | text | 128 token IDs, full fine-tuning | utterance | `/home/administrator/emotion-data/emotioncore_runs/X02/tokens` |
| T2 | text | 1024 sentence vector | utterance | `/home/administrator/emotion-data/emotioncore_runs/X02/features/T2_o/receipts` |

T1 `(o)/(r)` token缓存、T2 `(o)/(r)` 句向量缓存均在服务器。T1 与 T2 只用 `(o)` 训练。V3a 为整帧 50176 像素，V3b 为 448×448 人脸；两者共享视觉塔。

## 预测索引

| 配置 | OOF | ET-test | SIMS-valid |
|---|---|---|---|
| A1_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A1_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A1_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A1_h1/sims_valid.jsonl` |
| A1_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A1_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A1_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A1_h2/sims_valid.jsonl` |
| A2_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A2_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A2_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A2_h1/sims_valid.jsonl` |
| A2_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A2_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A2_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A2_h2/sims_valid.jsonl` |
| A3_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A3_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A3_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A3_h1/sims_valid.jsonl` |
| A3_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/A3_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/A3_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/A3_h2/sims_valid.jsonl` |
| V1_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V1_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V1_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V1_h1/sims_valid.jsonl` |
| V1_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V1_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V1_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V1_h2/sims_valid.jsonl` |
| V2_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V2_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V2_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V2_h1/sims_valid.jsonl` |
| V2_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V2_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V2_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V2_h2/sims_valid.jsonl` |
| V3a_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V3a_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V3a_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V3a_h1/sims_valid.jsonl` |
| V3a_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V3a_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V3a_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V3a_h2/sims_valid.jsonl` |
| V3b_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V3b_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V3b_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V3b_h1/sims_valid.jsonl` |
| V3b_h2 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/V3b_h2_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/V3b_h2/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/V3b_h2/sims_valid.jsonl` |
| T1_o | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/T1_cv/oof_o.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/T1_test/et_test_o.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/T1/sims_valid.jsonl` |
| T1_r | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/T1_cv/oof_r.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/T1_test/et_test_r.jsonl` | — |
| T2_r | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/T2_h1_asr_replay/oof_r.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/T2_h1_r/et_test_r.jsonl` | — |
| T2_h1 | `/home/administrator/emotion-data/emotioncore_runs/X02/heads/T2_h1_cv/oof.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/final/T2_h1/et_test.jsonl` | `/home/administrator/emotion-data/emotioncore_runs/X02/sims_valid/eval/T2_h1/sims_valid.jsonl` |

## 选择规则

模态冠军仅按 CV M macro-F1 选择，平分用 UAR；文字按 ASR 输入 `(r)` 的 CV 分数。每个 head 的学习率、权重衰减与 epoch 由内验证选择。选择记录见 `selection.json`。

## 校验与边界

- `configs/folds_v1.json` 是冻结划分；OOF 每句一次，CV 池无 02/13/20/21。
- 特征收据含输入媒体 SHA-256、编码器与预处理版本、特征文件 SHA-256。
- 缺脸/失败概率为均匀分布且状态单列；融合时请保留状态与覆盖率。
- CH-SIMS test 封存，X03 不得从该集合读取媒体、转写或特征。
