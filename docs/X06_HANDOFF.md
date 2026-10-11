# X06：X05 冻结参数与调用接口

本接口的最终可用状态以 X05 reviewer 验收为准。参数文件为 `configs/calibration_bundle_v1.json`，配套不可变方法文件为 `configs/x05_method_freeze.json`。部署专家全部使用 D1_fmhx；Q 使用 X04b 最终 Q_mix。模型、头、拟合集和方法文件的 SHA-256 均在参数包中。

本次交付参数包 SHA-256 为 `94a0805b9775676209306187a3880595b7e7592af70aaec8d50f3b2a7968979a`。文件已入库；交付证据索引见 [X05 DELIVERY](../experiments/X05_calib/DELIVERY.md)。

## 调用

从仓库根目录设置 `PYTHONPATH=src`。CPU 参数应用依赖现有锁定环境的 NumPy / SciPy，不会加载 Q 或专家模型。

```python
from emotioncore.x05.bundle import apply, Bundle

# 每个值是七类概率向量，顺序必须如下：
# neutral, joy, sadness, anger, fear, disgust, surprise
label, confidence, route = apply(component_probs, {
    "has_face": True,
    "face_count": 1,
    "face_quality": 0.9,
    "has_speech": True,
    "full_system": "H_ref",       # 或 H3，调用方明确选择
    "rejection_point": "90",      # "90" / "80"；None 仅用于不拒识的报告
})

# 服务进程可预加载一次，避免逐次读取 JSON：
bundle = Bundle.load()
label, confidence, route = bundle.apply(component_probs, availability)
```

`label` 是七类英文标签之一，或字符串 `"reject"`。`confidence` 是 OP-N 选中标签在中性偏置调整前的校准概率。阈值拒识时仍返回该置信度和本次使用的系统名；模态守门直接拒绝时返回 `("reject", 0.0, "reject")`。参数文件读写后逐值相同，同一输入输出确定。

## 路由与输入

| 可用性 | 路由 | 必需概率分量 |
|---|---|---|
| 有脸、有语音 | H_ref | Q_A、Q_AV、A3、V2、V1 |
| 有脸、有语音且 full_system=H3 | H3 | Q_AV、A3、V2 |
| 无脸、有语音 | H_A | Q_A、A3 |
| 有脸、无语音 | H_V | Q_V、V2、V1 |
| 无脸且无语音 | reject | 无 |
| 人脸个数 > 1 | reject，原因 multi_face | 无 |

纯守门函数 `emotioncore.x05.gate.route(has_face, face_count, face_quality, has_speech)` 返回路由类别和原因。多脸规则优先；没有引入人脸质量阈值。`face_quality` 作为传入元数据保留，不影响路由。X06 负责提供一致的实际可用性、面数和语音状态；X05 的句级统计代理不能直接充当在线 VAD 或逐帧检测器。

所有必需分量必须是有限、非负、总和大于 0 的七维概率。缺分量或非法概率会报错，接口不会悄悄替换。回退路由使用独立拟合的完整参数。默认 full_system 为 H_ref、拒识点为 OP-R90；H_ref/H3 的部署取舍仍由 X06 结合延迟评估决定。

## 拟合来源与限制

- 方法：Q_et + 原 D1 五折 OOF 留一折选择，冻结提交 `755d704`，SHA `8e29dd19beaa3401fb2103542b5333d392d89b9c3e9f819153c313e8fcbcebbe`。
- 部署融合/校准：Q_mix 两折 OOF（配原 D1 留出专家）+ FMH（配 D1_fmhx），9 名说话人。
- 中性偏置与 OP-R90/80：只用部署模型在 FMH 全部 3,266 句上的预测拟合。
- FMH 的 700 句曾用于 Q checkpoint 选择；排除这部分的描述性结果另列。FMH LOSO 只有三名说话人。
- ET-test、MER-test、SIMS-valid 仅报告，不拟合；CH-SIMS test 没有推理入口。
- MER-test 缺既有 Q-V 概率，H_V 在该语料没有本轮迁移结果。
- H_V 在 ET-test 的所选标签 ECE 为 0.0621、top-label ECE 为 0.0610，已提交 reviewer 判断；不依据目标集结果修改参数。

详细数值、ECE 超限项、专家对照及计算口径见 `experiments/X05_calib/REPORT.md` 和对应 results JSON。不要在 X06 使用目标集标签调整这份参数包。
