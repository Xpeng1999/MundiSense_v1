# X02 待 reviewer 确认

1. CH-SIMS valid 的 647 条中，27 条源视频缺失。A/V 的这 27 条保留在 Acc-2 分母；相关系数只对有分数的样本计算，并同时报告覆盖率。T 使用 `meta.csv` 的 valid 文本，仍覆盖全部 647 条。是否沿用这一缺失样本口径？
2. T1/T2 均只用标注转写训练，并同时评测标注转写与 SenseVoice。文字模态的冠军拟按实际部署输入 `(r)` 的 CV M macro-F1 选定，ET-test 不参与选择。是否认可这一预先固定的选择口径？
3. V3a/V3b 分辨率对照将在相同 head（h1 与 h2 分别）和同一 CV 句子上计算按说话人整群的配对区间。是否将 h2 作为主要对照、h1 作为敏感性分析？目前两者并列，不用 ET-test 做选择。
4. P2 官方权重约 70.5 GB。7 个候选情绪标签的首 token 已验证互异。最初的 Hugging Face 下载端点按 8 MiB 分段估算需约 68.85 小时，因此停止该通道；按用户指令改用 ModelScope，26 个文件已下载完成。迁移时逐文件比对源文件与 H 盘副本的 SHA-256，独立下载收据也已验证两个 ModelScope 模型全部文件。计入先前失败传输的公开模型累计下载量为 91,445,807,798 字节，低于 X02 的 100 GB 上限。下一步将在 32 GB 显存内尝试 NF4 thinker 加载和 ET-val 的 A/V/AV 受限首 token 评分。若加载失败，记录实测原因再依 X02 约束跳过。
5. Qwen3-Omni 与 ModelScope CLIP-GmP 已移至服务器 H 盘 `/mnt/h/emotionAI/emotioncore_models/X02/`，X02 原路径使用符号链接；Qwen3-Omni 迁移校验通过后系统盘可用空间从约 66 GB 回升到 132 GB。ModelScope CLIP-GmP 是与 V2 主基线不同的变体，仅作备用资源，不进入主结果。H 盘是 WSL 9p 挂载；P2 在该路径的完整模型加载速度和兼容性仍需实测。
