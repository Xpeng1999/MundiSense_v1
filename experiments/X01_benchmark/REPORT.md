# X01 工程基座与评测基准 v1 — 执行报告

## 1. 摘要

- EmotionTalk 登记 19,250 条，官方各划分和各类计数差异 0 格。
- CH-SIMS v2 源 CSV 有 4,403 条；缺媒体 154 条，重复路径 0 条，完整解码失败 0 条。
- 旧训练对 ET-test/ET-val 四层重叠均为零；第 16 轮中文集同句 161 条、同媒体哈希 155 条。
- D3 完成 40/40 条，最大 logprob 差 0.000000；完整 D2 结果见第 6 节。
- 人脸审计中 ET-val 帧的多脸比例为 0.211，SIMS-valid 为 0.250。
- 旧项目快照新增 172 个文件（166 个 `.pyc`、6 个其他文件）；已有文件变化 0，详见第 7 节。

## 2. 实际完成情况与偏离计划之处

### 事实

- 已建私有仓库、Mac 和服务器克隆；D2 每个组合均有独立 `run_meta.json`。本次收录 25,610 份服务器逐条预测的路径与 SHA-256。
- 服务器重启中断了长任务；D2 按已落盘样本 ID 续跑，人脸审计改用服务器运行目录中的逐片段原子凭证续跑。
- D2 运行时的 Git 状态：`2a5d7d2a62a3`、工作区有未提交改动；每组完整元数据在服务器 `run_meta.json`。
- 新独立 Python 3.11 虚拟环境已具备 CUDA；锁中 296 个包与旧安装版本逐项相同（旧环境不一致 0 个，新环境不一致 0 个）。本次 D2 在新环境配齐前已用旧解释器启动并保持同一环境，未向旧环境安装包。
- 新环境 `pip check` 返回 1：gradio 5.50.0 has requirement starlette<1.0,>=0.40.0, but you have starlette 1.6.0.。该依赖冲突来自所复刻的既有包组合；GPU 环境验证在 `new_venv_verification.json`。
- `pytest`：6 通过、0 失败、0 跳过。
- 计划写 CH-SIMS v2 4,402 条，源文件实际 4,403 条；未删除任何源行。

### 解读

- CH-SIMS v2 的官方论文表格划分数相加与本地 CSV 一致，论文文字总数不同；这个源材料差异留给 reviewer 确认。
- 旧项目新增文件违反零变更验收项；`.pyc` 可能由解释器自动生成，其他文件的创建者无法从快照确认。发现后已让本任务后续进程禁止写入字节码，复核快照无进一步新增；未删除旧目录内文件。

## 3. 数据审计要点

### 事实

| 模态 | M 一致数/总数 | 一致率 | Fleiss κ | 5 票一致数 |
| --- | ---: | ---: | ---: | ---: |
| A | 13045/19250 | 0.678 | 0.789 | 12700 |
| V | 12756/19250 | 0.663 | 0.726 | 10262 |
| T | 8134/19250 | 0.423 | 0.656 | 7475 |
| M | — | — | 0.783 | 11698 |

- 模态两两标签一致率：M/A 0.678；M/V 0.663；M/T 0.423；A/V 0.568；A/T 0.464；V/T 0.355。

- 各类别一致率、全部投票强度和 19 名说话人与组的关系见 `et_audit.json`。
- CH-SIMS v2：train 2,722、valid 647、test 1,034；现存媒体 `ffprobe` 失败 0 条。完整 `ffmpeg` 解码结果如下；test 只作清单、SHA-256 与解码登记，未作模型推理。

| 划分 | 解码成功 | 缺媒体 | 探测失败 | 解码失败 |
| --- | ---: | ---: | ---: | ---: |
| train | 2635 | 87 | 0 | 0 |
| valid | 620 | 27 | 0 | 0 |
| test | 994 | 40 | 0 | 0 |

- 2-fps YuNet 画面审计（零脸／一脸／多脸为帧比例）：

| 数据 | 片段 | 0 脸 | 1 脸 | ≥2 脸 | 抽帧失败次数（含缺媒体） |
| --- | ---: | ---: | ---: | ---: | ---: |
| emotiontalk | 5,837 | 0.000 | 0.894 | 0.106 | 1335 |
| chsims_v2 | 1,647 | 0.005 | 0.677 | 0.318 | 228 |

### 解读

- 脸数不等同于当前说话人的身份；CH-SIMS v2 的 `meta.csv` 无说话人 ID，不能据此声称官方划分与说话人无关。
- EmotionTalk val 的多脸帧比例为 0.211，说话人 15 对应受审计帧为 0.751；单一可见脸假设并不普遍成立。此计数不能判定多脸帧中的当前说话人。

## 4. Benchmark v1 定义摘要及 sha256

- 定义文件 SHA-256：`584be6883eb77b9acb8b1ace86c29d66eb91a69d793de1d3d0038185e4b74c6c`。ET-test 为主评测，ET-val 留作选择和校准；SIMS-valid 评估连续效价；CH-SIMS v2 test 是封存集。
- ET 主标签为 M 的七类；拒识和解析失败计入错误。SIMS-valid 的概率来自八个候选首 token 对数概率的 softmax，以预定义概率差计算 Pearson、Spearman 和排除零标签的 Acc-2，不调阈值。
- D2 配置：`VIDEO_MAX_PIXELS=50176`、`FPS_MAX_FRAMES=8`、`MAX_PIXELS=401408`；NF4 double quantization、bfloat16、sdpa。配置 SHA-256：`3b26741197db405b73b90393755b05475ddf5cbd75aa8a2b18428d6942ebee79`。system/user 提示词取对应模态旧训练清单首行，逐字保持。

| 清单 | 条数 | SHA-256 |
| --- | ---: | --- |
| et_test | 1,929 | `9875bcec77d2b7edab6ab3ecbe875461d1b0b32b4e901564c6b45c4e33b1de16` |
| et_val | 1,908 | `088d99003bea5bba84cd5989357b64c0fe85fa4a456ab9532056cdfee9676d9e` |
| sims_valid | 647 | `3da69612eb1f7f1967c4f22254d7613f43f04283c03b5bfc1cc5ccd9241604f4` |
| sims_test_lockbox | 1,034 | `730a43e1a95f640ef181fab38126c6a401d1d6947acc7b53281c031848e93b74` |

## 5. 泄漏核查与对旧结论的更正

### 事实

- ET-test 与 ET-val 相对旧 AV/A 训练清单的同句、同媒体哈希、同说话人、同对话重叠均为 0；完整核查在 `leakage_et_test.json` 和 `leakage_et_val.json`。
- 第 16 轮 252 条中文集：同句 161、同媒体哈希 155、同说话人 168、同对话 168。
- 旧 qwen_old 的 M 标签 macro-F1，‘干净’定义为四层均无重叠，‘泄漏’为任一层有重叠；原有脸门控拒识保留在分母。

| 条件 | 干净 n | 干净 macro-F1 | 泄漏 n | 泄漏 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| audio | 84 | 0.634 | 168 | 0.615 |
| video | 84 | 0.220 | 168 | 0.294 |
| av | 84 | 0.580 | 168 | 0.607 |

### 解读

- 第 16 轮中文整体分数不能作为未泄漏的独立泛化证据；干净与泄漏组的样本和拒识率也不同，表中差异不是泄漏的因果效应。

## 6. 基线结果

### 事实：ET 主标签

表内每个指标后依次给按句／按说话人 bootstrap 95% 区间（各 2,000 次）。测试集只有四名说话人，说话人区间仅描述性。

| 模型 | 集合 | 条件 | n | 失败 | macro-F1 [句]/[人] | UAR [句]/[人] | WA [句]/[人] | 覆盖率 | 中性误报 |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| base | et_test | audio | 1,929 | 0 | 0.312 [0.293, 0.329]/[0.280, 0.330] | 0.383 [0.367, 0.400]/[0.357, 0.402] | 0.484 [0.461, 0.505]/[0.444, 0.532] | 1.000 | 0.481 |
| base | et_test | video | 1,929 | 0 | 0.269 [0.255, 0.283]/[0.214, 0.293] | 0.306 [0.290, 0.322]/[0.227, 0.342] | 0.518 [0.496, 0.540]/[0.387, 0.652] | 1.000 | 0.087 |
| base | et_test | av | 1,929 | 0 | 0.336 [0.322, 0.349]/[0.316, 0.339] | 0.411 [0.396, 0.426]/[0.395, 0.423] | 0.559 [0.538, 0.581]/[0.515, 0.623] | 1.000 | 0.318 |
| old_lora | et_test | audio | 1,929 | 0 | 0.603 [0.577, 0.629]/[0.539, 0.649] | 0.566 [0.541, 0.591]/[0.506, 0.627] | 0.682 [0.661, 0.702]/[0.604, 0.753] | 1.000 | 0.142 |
| old_lora | et_test | video | 1,929 | 0 | 0.287 [0.266, 0.307]/[0.199, 0.334] | 0.339 [0.321, 0.357]/[0.269, 0.398] | 0.485 [0.462, 0.506]/[0.415, 0.551] | 1.000 | 0.252 |
| old_lora | et_test | av | 1,929 | 0 | 0.600 [0.575, 0.624]/[0.516, 0.669] | 0.623 [0.597, 0.648]/[0.569, 0.683] | 0.645 [0.625, 0.667]/[0.619, 0.683] | 1.000 | 0.367 |
| base | et_val | audio | 1,908 | 0 | 0.354 [0.332, 0.375]/[0.308, 0.397] | 0.403 [0.384, 0.422]/[0.380, 0.418] | 0.522 [0.499, 0.544]/[0.478, 0.564] | 1.000 | 0.452 |
| base | et_val | video | 1,908 | 0 | 0.275 [0.260, 0.290]/[0.220, 0.311] | 0.313 [0.298, 0.329]/[0.268, 0.348] | 0.484 [0.460, 0.506]/[0.424, 0.552] | 1.000 | 0.144 |
| base | et_val | av | 1,908 | 0 | 0.359 [0.344, 0.374]/[0.312, 0.383] | 0.418 [0.402, 0.433]/[0.406, 0.423] | 0.593 [0.571, 0.616]/[0.550, 0.639] | 1.000 | 0.268 |
| old_lora | et_val | audio | 1,908 | 0 | 0.605 [0.577, 0.631]/[0.527, 0.683] | 0.574 [0.547, 0.601]/[0.503, 0.657] | 0.679 [0.658, 0.699]/[0.635, 0.726] | 1.000 | 0.160 |
| old_lora | et_val | video | 1,908 | 0 | 0.316 [0.297, 0.337]/[0.263, 0.356] | 0.357 [0.338, 0.377]/[0.302, 0.379] | 0.499 [0.476, 0.521]/[0.439, 0.555] | 1.000 | 0.204 |
| old_lora | et_val | av | 1,908 | 0 | 0.651 [0.624, 0.674]/[0.605, 0.698] | 0.666 [0.640, 0.691]/[0.635, 0.705] | 0.708 [0.687, 0.727]/[0.689, 0.736] | 1.000 | 0.305 |

### 事实：逐说话人 ET 结果

| 模型 | 集合 | 条件 | 说话人 | n | macro-F1 | UAR | WA | 覆盖率 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | et_test | audio | 02 | 414 | 0.276 | 0.346 | 0.437 | 1.000 |
| base | et_test | audio | 13 | 379 | 0.308 | 0.388 | 0.467 | 1.000 |
| base | et_test | audio | 20 | 525 | 0.338 | 0.389 | 0.451 | 1.000 |
| base | et_test | audio | 21 | 611 | 0.260 | 0.376 | 0.553 | 1.000 |
| base | et_test | video | 02 | 414 | 0.225 | 0.315 | 0.353 | 1.000 |
| base | et_test | video | 13 | 379 | 0.293 | 0.346 | 0.470 | 1.000 |
| base | et_test | video | 20 | 525 | 0.243 | 0.309 | 0.467 | 1.000 |
| base | et_test | video | 21 | 611 | 0.154 | 0.165 | 0.705 | 1.000 |
| base | et_test | av | 02 | 414 | 0.332 | 0.425 | 0.514 | 1.000 |
| base | et_test | av | 13 | 379 | 0.333 | 0.414 | 0.522 | 1.000 |
| base | et_test | av | 20 | 525 | 0.335 | 0.398 | 0.516 | 1.000 |
| base | et_test | av | 21 | 611 | 0.292 | 0.379 | 0.648 | 1.000 |
| old_lora | et_test | audio | 02 | 414 | 0.663 | 0.639 | 0.698 | 1.000 |
| old_lora | et_test | audio | 13 | 379 | 0.622 | 0.614 | 0.660 | 1.000 |
| old_lora | et_test | audio | 20 | 525 | 0.539 | 0.520 | 0.579 | 1.000 |
| old_lora | et_test | audio | 21 | 611 | 0.531 | 0.496 | 0.773 | 1.000 |
| old_lora | et_test | video | 02 | 414 | 0.284 | 0.379 | 0.394 | 1.000 |
| old_lora | et_test | video | 13 | 379 | 0.347 | 0.380 | 0.504 | 1.000 |
| old_lora | et_test | video | 20 | 525 | 0.227 | 0.286 | 0.432 | 1.000 |
| old_lora | et_test | video | 21 | 611 | 0.146 | 0.231 | 0.579 | 1.000 |
| old_lora | et_test | av | 02 | 414 | 0.685 | 0.704 | 0.681 | 1.000 |
| old_lora | et_test | av | 13 | 379 | 0.628 | 0.644 | 0.686 | 1.000 |
| old_lora | et_test | av | 20 | 525 | 0.557 | 0.570 | 0.623 | 1.000 |
| old_lora | et_test | av | 21 | 611 | 0.468 | 0.581 | 0.615 | 1.000 |
| base | et_val | audio | 01 | 451 | 0.384 | 0.395 | 0.574 | 1.000 |
| base | et_val | audio | 02 | 389 | 0.401 | 0.421 | 0.553 | 1.000 |
| base | et_val | audio | 13 | 566 | 0.291 | 0.383 | 0.516 | 1.000 |
| base | et_val | audio | 15 | 502 | 0.316 | 0.409 | 0.458 | 1.000 |
| base | et_val | video | 01 | 451 | 0.234 | 0.269 | 0.477 | 1.000 |
| base | et_val | video | 02 | 389 | 0.251 | 0.304 | 0.458 | 1.000 |
| base | et_val | video | 13 | 566 | 0.288 | 0.350 | 0.574 | 1.000 |
| base | et_val | video | 15 | 502 | 0.194 | 0.254 | 0.408 | 1.000 |
| base | et_val | av | 01 | 451 | 0.384 | 0.412 | 0.656 | 1.000 |
| base | et_val | av | 02 | 389 | 0.374 | 0.418 | 0.594 | 1.000 |
| base | et_val | av | 13 | 566 | 0.306 | 0.442 | 0.597 | 1.000 |
| base | et_val | av | 15 | 502 | 0.317 | 0.413 | 0.532 | 1.000 |
| old_lora | et_val | audio | 01 | 451 | 0.574 | 0.555 | 0.714 | 1.000 |
| old_lora | et_val | audio | 02 | 389 | 0.730 | 0.724 | 0.740 | 1.000 |
| old_lora | et_val | audio | 13 | 566 | 0.475 | 0.447 | 0.668 | 1.000 |
| old_lora | et_val | audio | 15 | 502 | 0.550 | 0.591 | 0.612 | 1.000 |
| old_lora | et_val | video | 01 | 451 | 0.308 | 0.351 | 0.503 | 1.000 |
| old_lora | et_val | video | 02 | 389 | 0.282 | 0.330 | 0.494 | 1.000 |
| old_lora | et_val | video | 13 | 566 | 0.317 | 0.329 | 0.569 | 1.000 |
| old_lora | et_val | video | 15 | 502 | 0.215 | 0.278 | 0.420 | 1.000 |
| old_lora | et_val | av | 01 | 451 | 0.614 | 0.636 | 0.698 | 1.000 |
| old_lora | et_val | av | 02 | 389 | 0.742 | 0.753 | 0.758 | 1.000 |
| old_lora | et_val | av | 13 | 566 | 0.580 | 0.657 | 0.703 | 1.000 |
| old_lora | et_val | av | 15 | 502 | 0.635 | 0.671 | 0.683 | 1.000 |

### 事实：A/V 原生标签（与 M 结果并列，不作相减）

| 模型 | 集合 | 条件 | 真值 | macro-F1 | UAR | WA |
| --- | --- | --- | --- | ---: | ---: | ---: |
| base | et_test | audio | M | 0.312 | 0.383 | 0.484 |
| base | et_test | audio | A | 0.332 | 0.422 | 0.522 |
| base | et_test | video | M | 0.269 | 0.306 | 0.518 |
| base | et_test | video | V | 0.280 | 0.314 | 0.557 |
| old_lora | et_test | audio | M | 0.603 | 0.566 | 0.682 |
| old_lora | et_test | audio | A | 0.653 | 0.643 | 0.744 |
| old_lora | et_test | video | M | 0.287 | 0.339 | 0.485 |
| old_lora | et_test | video | V | 0.316 | 0.378 | 0.553 |
| base | et_val | audio | M | 0.354 | 0.403 | 0.522 |
| base | et_val | audio | A | 0.394 | 0.449 | 0.573 |
| base | et_val | video | M | 0.275 | 0.313 | 0.484 |
| base | et_val | video | V | 0.279 | 0.327 | 0.472 |
| old_lora | et_val | audio | M | 0.605 | 0.574 | 0.679 |
| old_lora | et_val | audio | A | 0.693 | 0.674 | 0.778 |
| old_lora | et_val | video | M | 0.316 | 0.357 | 0.499 |
| old_lora | et_val | video | V | 0.336 | 0.376 | 0.502 |

### 事实：八类 argmax 次要结果（`other` 计错）

| 模型 | 集合 | 条件 | macro-F1 | UAR | WA | 有效七类输出率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| base | et_test | audio | 0.312 | 0.381 | 0.480 | 0.986 |
| base | et_test | video | 0.269 | 0.306 | 0.518 | 0.999 |
| base | et_test | av | 0.336 | 0.411 | 0.559 | 0.997 |
| old_lora | et_test | audio | 0.603 | 0.566 | 0.682 | 1.000 |
| old_lora | et_test | video | 0.287 | 0.339 | 0.485 | 1.000 |
| old_lora | et_test | av | 0.600 | 0.623 | 0.645 | 1.000 |
| base | et_val | audio | 0.352 | 0.399 | 0.516 | 0.985 |
| base | et_val | video | 0.275 | 0.313 | 0.484 | 0.999 |
| base | et_val | av | 0.359 | 0.417 | 0.593 | 0.997 |
| old_lora | et_val | audio | 0.605 | 0.574 | 0.679 | 1.000 |
| old_lora | et_val | video | 0.316 | 0.357 | 0.499 | 1.000 |
| old_lora | et_val | av | 0.651 | 0.666 | 0.708 | 1.000 |

### 事实：SIMS-valid

| 模型 | 条件 | n | 失败 | 覆盖率 | 相关 n | Acc-2 非零 n | Pearson [95%] | Spearman [95%] | Acc-2 非零 [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| base | audio | 647 | 27 | 0.958 | 620 | 555 | 0.737 [0.694, 0.775] | 0.752 [0.711, 0.788] | 0.796 [0.762, 0.831] |
| base | av | 647 | 28 | 0.957 | 619 | 555 | 0.751 [0.705, 0.791] | 0.752 [0.706, 0.789] | 0.804 [0.770, 0.836] |
| old_lora | audio | 647 | 27 | 0.958 | 620 | 555 | 0.723 [0.678, 0.765] | 0.725 [0.679, 0.766] | 0.802 [0.768, 0.835] |
| old_lora | av | 647 | 28 | 0.957 | 619 | 555 | 0.729 [0.682, 0.771] | 0.724 [0.677, 0.765] | 0.820 [0.787, 0.851] |

### 事实：失败原因

| 组合 | 原因 | 条数 |
| --- | --- | ---: |
| base_sims_valid_av | FileNotFoundError | 27 |
| base_sims_valid_av | PreprocessTimeout | 1 |
| base_sims_valid_audio | RuntimeError | 27 |
| old_lora_sims_valid_av | FileNotFoundError | 27 |
| old_lora_sims_valid_av | TimeoutError | 1 |
| old_lora_sims_valid_audio | RuntimeError | 27 |

- SIMS-valid 音频条件的 27 条 `RuntimeError` 来自源 mp4 缺失、无法提取 wav；两模型遇到同一批缺失片段。AV 条件的 `FileNotFoundError` 对应同一批缺失媒体，均不是额外的样本。

### 事实：配对差值（macro-F1）

| 左减右 | n | 差值 | 配对 95% 区间 |
| --- | ---: | ---: | --- |
| base_et_test_av_minus_base_et_test_audio | 1,929 | 0.024 | [0.009, 0.039] |
| base_et_test_av_minus_base_et_test_video | 1,929 | 0.066 | [0.050, 0.084] |
| base_et_test_audio_minus_base_et_test_video | 1,929 | 0.043 | [0.020, 0.065] |
| old_lora_et_test_av_minus_old_lora_et_test_audio | 1,929 | -0.003 | [-0.027, 0.020] |
| old_lora_et_test_av_minus_old_lora_et_test_video | 1,929 | 0.313 | [0.285, 0.339] |
| old_lora_et_test_audio_minus_old_lora_et_test_video | 1,929 | 0.316 | [0.286, 0.346] |
| old_lora_et_test_av_minus_base_et_test_av | 1,929 | 0.264 | [0.238, 0.289] |
| old_lora_et_test_audio_minus_base_et_test_audio | 1,929 | 0.291 | [0.264, 0.320] |
| old_lora_et_test_video_minus_base_et_test_video | 1,929 | 0.018 | [-0.003, 0.038] |
| base_et_val_av_minus_base_et_val_audio | 1,908 | 0.005 | [-0.016, 0.026] |
| base_et_val_av_minus_base_et_val_video | 1,908 | 0.084 | [0.066, 0.103] |
| base_et_val_audio_minus_base_et_val_video | 1,908 | 0.079 | [0.054, 0.105] |
| old_lora_et_val_av_minus_old_lora_et_val_audio | 1,908 | 0.045 | [0.019, 0.069] |
| old_lora_et_val_av_minus_old_lora_et_val_video | 1,908 | 0.334 | [0.304, 0.366] |
| old_lora_et_val_audio_minus_old_lora_et_val_video | 1,908 | 0.289 | [0.255, 0.322] |
| old_lora_et_val_av_minus_base_et_val_av | 1,908 | 0.292 | [0.264, 0.318] |
| old_lora_et_val_audio_minus_base_et_val_audio | 1,908 | 0.251 | [0.219, 0.280] |
| old_lora_et_val_video_minus_base_et_val_video | 1,908 | 0.041 | [0.021, 0.062] |

SIMS-valid 同样按相同样本做 A/AV 与基座/LoRA 配对比较：

| 左减右及指标 | n | 差值 | 配对 95% 区间 |
| --- | ---: | ---: | --- |
| base_sims_valid_av_minus_base_sims_valid_audio_pearson | 647 | 0.014 | [-0.019, 0.045] |
| base_sims_valid_av_minus_base_sims_valid_audio_spearman | 647 | -0.001 | [-0.033, 0.031] |
| base_sims_valid_av_minus_base_sims_valid_audio_acc_2_nonzero | 647 | 0.007 | [-0.016, 0.031] |
| old_lora_sims_valid_av_minus_old_lora_sims_valid_audio_pearson | 647 | 0.005 | [-0.028, 0.039] |
| old_lora_sims_valid_av_minus_old_lora_sims_valid_audio_spearman | 647 | -0.001 | [-0.032, 0.032] |
| old_lora_sims_valid_av_minus_old_lora_sims_valid_audio_acc_2_nonzero | 647 | 0.018 | [-0.007, 0.045] |
| old_lora_sims_valid_av_minus_base_sims_valid_av_pearson | 647 | -0.022 | [-0.054, 0.012] |
| old_lora_sims_valid_av_minus_base_sims_valid_av_spearman | 647 | -0.028 | [-0.058, 0.006] |
| old_lora_sims_valid_av_minus_base_sims_valid_av_acc_2_nonzero | 647 | 0.016 | [-0.007, 0.043] |
| old_lora_sims_valid_audio_minus_base_sims_valid_audio_pearson | 647 | -0.014 | [-0.047, 0.020] |
| old_lora_sims_valid_audio_minus_base_sims_valid_audio_spearman | 647 | -0.027 | [-0.058, 0.004] |
| old_lora_sims_valid_audio_minus_base_sims_valid_audio_acc_2_nonzero | 647 | 0.005 | [-0.023, 0.036] |

### 事实：说话人、困难类别与耗时

- 各组合的完整七类 P/R/F1、混淆矩阵、说话人结果、A/V 原生标签结果和包含 `other` 的次要结果均在 `metrics.json`。下面列 ET-test 的重点召回：

| 模型 | 条件 | fear | disgust | surprise | neutral |
| --- | --- | ---: | ---: | ---: | ---: |
| base | audio | 0.008 | 0.000 | 0.093 | 0.519 |
| base | video | 0.000 | 0.000 | 0.000 | 0.913 |
| base | av | 0.000 | 0.000 | 0.000 | 0.682 |
| old_lora | audio | 0.544 | 0.336 | 0.441 | 0.858 |
| old_lora | video | 0.032 | 0.052 | 0.273 | 0.748 |
| old_lora | av | 0.560 | 0.403 | 0.696 | 0.633 |

| 模型 | 集合 | 条件 | P50 秒 | P95 秒 | 显存峰值 MiB |
| --- | --- | --- | ---: | ---: | ---: |
| base | et_test | av | 0.976 | 1.789 | 6930.2 |
| base | et_test | audio | 0.312 | 0.347 | 6926.2 |
| base | et_test | video | 0.584 | 1.359 | 6926.2 |
| base | et_val | av | 1.015 | 1.877 | 7370.7 |
| base | et_val | audio | 0.311 | 0.363 | 7370.7 |
| base | et_val | video | 0.606 | 1.429 | 7370.7 |
| old_lora | et_test | av | 1.295 | 2.363 | 6984.4 |
| old_lora | et_test | audio | 0.414 | 0.477 | 6984.4 |
| old_lora | et_test | video | 0.596 | 1.472 | 6984.4 |
| old_lora | et_val | av | 0.873 | 1.683 | 7425.2 |
| old_lora | et_val | audio | 0.296 | 0.332 | 7425.2 |
| old_lora | et_val | video | 0.526 | 1.248 | 7425.2 |
| base | sims_valid | av | 0.929 | 1.403 | 7370.7 |
| base | sims_valid | audio | 0.430 | 0.516 | 6555.7 |
| old_lora | sims_valid | av | 0.898 | 1.416 | 7425.2 |
| old_lora | sims_valid | audio | 0.330 | 0.354 | 6609.2 |

### 解读

- A、V、AV 与基座／旧 LoRA 的差值只在相同样本和相同 M 标签下解释；A 与 A 标签、V 与 V 标签结果并列保留，不作跨真值相减。SIMS-valid 的情绪概率差是跨任务代理量。相关系数仅对有效评分计算，覆盖率和相关 n 同时列出；Acc-2 非零 n 保留失败样本并计错。

## 7. 异常与未解决问题

- CH-SIMS v2 的 4,402／4,403 源材料差异、154 条缺媒体和未验证说话人独立性见数据卡及 `OPEN_QUESTIONS.md`。
- D2 对单条预处理和前向计算设 300 秒 `SIGALRM` 时限；它无法强制中断本次底层媒体调用。同一 SIMS-valid 片段在两个模型的 AV 条件均超时，已人工停止进程、写失败凭证并断点续跑；失败保留在指标分母，具体组合和次数见第 6 节失败原因表。
- 旧项目快照：新增 172（`.pyc` 166、其他 6）、既有文件 mtime/大小变化 0、删除 0。其他新增路径：scripts/mer_download/download_mer.py；scripts/mer_download/index.html；scripts/mer_download/mirror_status.sh；scripts/mer_download/progress_server.ps1；scripts/mer_download/progress_server.py；scripts/mer_download/supervise.sh。这些文件均已保留；零变更验收项未通过。
- 所有逐条预测只在服务器运行目录；仓库仅有逐条路径及 SHA-256 清单。
- base sims_valid av 的 `aqgy4_0024__00004` 媒体时长 1.998 秒；处理至少 480 秒无输出，`ffmpeg` 完整解码成功。已写失败凭证并断点续跑；凭证 SHA-256 为 `7201174da760f828b4d55dc7d235050b41a08d9dbd68b88d12e53579c2fb4740`。
- old_lora sims_valid av 的 `aqgy4_0024__00004` 媒体时长 1.998 秒；处理至少 300 秒无输出，`ffmpeg` 完整解码成功。已写失败凭证并断点续跑；凭证 SHA-256 为 `b24649c7d64e61e519d509b46d7560b58c28a71799e535d9ad00e2279f1ffcdf`。

## 8. 给 reviewer 的问题

1. 官方论文文字写 4,402、Table 1 和所给 CSV 为 4,403；是否以当前 4,403 行作为 Benchmark v1 的 SIMS-valid/lock-box 固定来源？
2. 旧项目新增 166 个 `.pyc` 和 6 个其他文件，已有文件未变；其他文件的来源尚无法确认。请决定这一只读边界偏离的验收记录方式。
3. CH-SIMS v2 无说话人 ID，官方来源未确立说话人独立性；是否需要后续单独获取说话人元数据？
4. SIMS-valid 同一片段在两个 AV 模型的预处理都无法被 `SIGALRM` 中断；后续修订是否需要用独立进程约束预处理时间？本版已将两次失败计入分母。
