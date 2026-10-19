# EmotionCore by Mundiscope

**Audio-visual emotion perception. A first sense for human interaction.**

EmotionCore is Mundiscope's current research prototype for the **Sense** layer of **Sense → Remember → Express**. It brings voice and facial expression together to estimate observable emotional expression, while exposing confidence, fallback and abstention behavior.

> Research preview · October 2026 · Chinese-language, single-person scope

| Release material | Status |
| --- | --- |
| Demo film | Coming soon — after review |
| English technical report | Coming soon — after review |

## What the prototype does

- Combines a Qwen2.5-Omni-based audio-visual model with speech and facial-expression specialists. The default H3 path fuses Q_AV, A3 and V2 probabilities using frozen parameters.
- Produces seven expression categories: **neutral, joy, sadness, anger, fear, disgust and surprise**, with confidence and configured rejection rules.
- Falls back to audio when no face is available, and rejects ambiguous multi-face inputs rather than assigning an emotion to an unverified speaker.
- Segments speech and emits timestamped events through a streaming prototype evaluated on recorded-stream replay.

This estimates visible and audible expression. It is not a measurement of private thoughts or a clinical diagnosis.

## Selected results

| Evaluation | Result | Scope |
| --- | --- | --- |
| CH-SIMS v2 test sentiment | Pearson **0.792**; video-cluster 95% CI **0.756–0.829** | 976 media-available clips out of 1,034; all-clip Pearson **0.761** |
| EmotionTalk test, seven classes | Macro-F1 **0.645** | 1,929 utterances; H3 OP-N, without threshold rejection |
| Automatic speech segmentation | Macro-F1 **0.624**; coverage **87.66%** | EmotionTalk R4 dialogue-duration streams, 1,929 utterances |
| Three-hour stream replay | Utterance-to-final-output **3.11 s P50 / 11.02 s P95** | Annotated utterance end → last overlapping event output; RTX 5090 32 GB workstation |

Correlation, macro-F1 and latency describe different tasks. They are not interchangeable accuracy measures.

At the fixed OP-R90 operating point, EmotionTalk macro-F1 is **0.642** at **88.5%** coverage; rejected cases remain in the F1 denominator. CH-SIMS v2 test coverage is **62.4%** (645/1,034), with accepted-clip Pearson **0.810**. “R90” names the development-calibrated operating point, not a coverage guarantee for new domains. The EmotionTalk complete-utterance comparison does not apply the multi-face gate.

X07 results are documented on the lock-box branch; inclusion here does not merge that experiment branch. Planned VAD endpoint-to-output latency is **2.21 s P50 / 2.80 s P95**. The longer utterance-to-final-output time above includes waiting for merged/split speech segments and is not time to first feedback or a natural-user latency benchmark. The stream replay has wall-clock/media ratio **1.001**, backlog P95 **0.998 s**, and approximately **11%** process-memory growth before stabilization. These results do not establish live-device performance or long-duration deployment readiness.

For the recorded methodology and limitations, see the [system card](https://github.com/Xpeng1999/emotioncore-zh/blob/71a1f99404c80e621ba4d660333ca48b1299d31c/docs/SYSTEM_CARD.md), [X06b stream evaluation](experiments/X06b_realtime/REPORT.md), and [X07 lock-box evaluation](https://github.com/Xpeng1999/emotioncore-zh/blob/71a1f99404c80e621ba4d660333ca48b1299d31c/experiments/X07_lockbox/REPORT.md).

## The Mundiscope direction

| Stage | Direction | Status |
| --- | --- | --- |
| **Sense** | Audio-visual emotion perception with uncertainty and timestamped outputs | Current research prototype |
| **Remember** | Consent-aware interaction memory that retains useful context over time | Future direction |
| **Express** | Abstract response interfaces, leaving physical behavior to each robot | Future direction |

Mundiscope aims to provide a perception and memory layer for embodied intelligence. EmotionCore represents the current sensing milestone. Persistent memory, pose and touch sensing, robotics SDKs, hardware integration and robot behavior are not established by this preview.

## Scope and availability

Current evaluation covers Chinese-language recorded data and stream replay, within a single-person scope. Real-device and real-user performance and multi-person interaction remain to be evaluated. Visual quality, occlusion, absent media and domain shift can affect outputs and coverage.

The current research assets carry upstream non-commercial restrictions. This preview does not release training data, pretrained assets, model weights, an upload service or a commercial SDK. Rights to datasets, pretrained models and recorded footage remain with their respective owners and licensors.

Demo and report links will be added after review. No download links are active yet.

## Repository guide

| Material | Location |
| --- | --- |
| Original X01 setup, reproduction and provenance | [X01 reproduction](docs/X01_REPRODUCTION.md) |
| Frozen benchmark definition | [Benchmark v1.1](docs/BENCHMARK_v1.1.md) |
| Data sources and restrictions | [Data registry](docs/DATA_REGISTRY.md) |
| Non-technical system summary | [System card](https://github.com/Xpeng1999/emotioncore-zh/blob/71a1f99404c80e621ba4d660333ca48b1299d31c/docs/SYSTEM_CARD.md) |
| Streaming handoff and frozen configuration | [X07 handoff](docs/X07_HANDOFF.md) |
| Static project overview | [HTML source](docs/index.html) — download and open locally |

This is the research repository. Historical experiment reports retain their original protocols and review status. Run commands only within the documented environment and experiment scope.

---

**MUNDISCOPE · 洞悉尘世**  
*See into what's felt.*
