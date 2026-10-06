# MER2026 数据卡 — X04a

| 划分 | n | 项目标签分布 | 最短秒 | P50秒 | P90秒 | 最长秒 |
| --- | --- | --- | --- | --- | --- | --- |
| MER-train 原始 | 9395 | {"neutral": 2284, "anger": 2155, "sadness": 1658, "joy": 1827, "worried": 1081, "surprise": 390} | 0.255437 | 3.08825 | 6.457 | 46.718563 |
| MER-test | 572 | {"sadness": 163, "joy": 105, "worried": 130, "surprise": 45, "anger": 29, "neutral": 100} | 1.986375 | 2.561063 | 4.177118699999999 | 10.153 |

去重后的训练集：9386 条，标签分布 {"neutral": 2280, "anger": 2151, "sadness": 1657, "joy": 1827, "worried": 1081, "surprise": 390}。

只用 MER2026，不重复加入已包含的 MER2025。训练与测试 ID 不相交；MER-test 与 MER-train 不保证说话人独立。572 条测试不参与训练、早停或任何参数选择。

视频分辨率及各分辨率计数见 results/data_registration.json（MER_train/MER_test.video_resolutions）。视频采样时长取画面流，避免附加 data 流延长容器时长；音频时长分布取 WAV 流或容器回退值。

ffprobe 注册：全部音视频探测成功；此为头部探测，不能替代全解码。

全量解码、人脸统计与随机60条多脸联系表待全量冻结 X02 管线完成后补齐。

许可证据：服务器 /mnt/h/emotion-data/MER2026/README.md，SHA `7257704eef53cac3d78b95f7b9527c8f1314f9ea71c27558e9f4a416a0e2ce08`。README 标为 CC-BY-NC-4.0，并明确限定学术研究/挑战参与；数据、标注和派生文件未经书面许可不得再分发、镜像或用于商业用途。本仓库仅公开汇总证据和产物路径/SHA。
