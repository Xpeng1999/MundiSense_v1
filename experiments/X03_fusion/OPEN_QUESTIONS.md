# X03 待 reviewer 确认

1. Benchmark v1 代码的 `neutral_false_positive_rate` 实际计算的是**真实中性句被预测为非中性**的比例。X03 将沿用冻结代码的这一口径设置 OP-N 的 0.15 工作点，并在报告中明确分子、分母；不擅自改成通常意义上的“非中性句误判中性”。
2. X02 缓存有 T1 的 `(o)/(r)` token IDs 和模型 checkpoint，但没有 `[CLS]` 向量。E2 将只使用已有 token 缓存和冻结 T1 checkpoint 生成 `[CLS]` 特征，不重新读取原始转写或训练 T1。该新增特征提取在 X03 留收据。
3. X02 人脸轨迹回执有 8 张裁剪帧，但没有逐帧置信度与最小脸尺寸。Q 臂将对子进程读取的现有裁剪帧再做人脸检测和预先固定的尺寸/置信度过滤；未通过的帧按固定规则处理并记录失败率，不根据验证成绩调阈值。
4. `X02_REVIEW.md` 称 P2“未尝试 4bit”，但服务器 P2 冒烟测试的 run_meta 记载 `bitsandbytes_nf4_double_quant_thinker_only`，代码使用 `load_in_4bit=True` 和 BF16 计算。该测试的 PyTorch CUDA 峰值分配仍为 58,041.6 MiB，超过物理显存，不能证明满足 X03 可选臂 P 的 31 GB 上限。
5. X02 归档使用 `git worktree move` 更新注册并原样移动 21 个工作树，原路径改为符号链接；37 个 bundle 也已移动。Git 工作树仍注册在归档目录，以保留可用性和所有文件。此方式没有解除注册，但没有悬空注册，也没有删除文件。
