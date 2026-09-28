# CH-SIMS v2 data card — X01 audit

## Source and scope

The supplied read-only `CH-SIMSV2/meta.csv` and six `Raw-*.zip` files were registered without changing the archives. All archive paths were checked before extraction; macOS `._*` entries were excluded. The derived clips reside under `emotioncore_derived/chsims_v2/raw/Raw/`. The manifest stores clip IDs, official mode, M/T/A/V numeric labels, annotation polarity, relative media path, SHA-256, duration, and media parameters. It does not store the `text` column.

The [official CH-SIMS v2 repository](https://github.com/thuiar/ch-sims-v2) and [paper](https://arxiv.org/abs/2209.02604) describe the supervised v2 release and its modality labels. The benchmark uses the provided `mode` field, without remapping clips.

## Counts and media audit

The actual CSV contains **4,403 data rows**: 2,722 train, 647 valid, and 1,034 test. The [official paper's Table 1](https://arxiv.org/pdf/2209.02604) reports these same split counts (which sum to 4,403), while its prose says 4,402; X01's planned count followed that prose. `OPEN_QUESTIONS.md` records the discrepancy. There are no blank key rows and no repeated `(video_id, clip_id)` pairs. A direct `Raw/{video_id}/{clip_id}.mp4` lookup finds 154 missing rows and no duplicate media paths. The earlier claim of 231 duplicate paths is not reproduced under this mapping. Individual missing paths are in `results/chsims_audit.json`.

All present clips were decoded end to end with FFmpeg (`-xerror -err_detect explode`, video and audio streams to a null sink). None failed full decode. Machine-readable split counts and missing IDs are in `results/chsims_decode_audit.json`.

| Split | Full decode succeeded | Missing media | Full decode failed |
| --- | ---: | ---: | ---: |
| Train | 2,635 | 87 | 0 |
| Valid | 620 | 27 | 0 |
| Test | 994 | 40 | 0 |

The test set is a lock-box. This audit only lists it, hashes/probes and fully decodes its media, and records media status. No Qwen model is run on its clips before X07. Its separate lock-box manifest omits labels.

## Evaluation contract

The [official MMSA result table](https://github.com/thuiar/MMSA/blob/master/results/result-stat.md) distinguishes `Has0_acc_2`, `Non0_acc_2`, and `Corr` for sentiment regression. X01 fixes its own threshold-free score as `P(joy) − [P(anger) + P(disgust) + P(fear) + P(sadness)]`. `Acc-2` here uses the nonzero-label convention: exclude `label = 0`, then compare signs at zero. Pearson and Spearman correlations use the continuous source `label`. These are cross-task proxy scores for an emotion classifier, not directly comparable to trained CH-SIMS v2 sentiment models.

## Speaker and visual limitations

The supplied `meta.csv` has no speaker identifier. Neither the [official repository](https://github.com/thuiar/ch-sims-v2) nor the [paper abstract](https://arxiv.org/abs/2209.02604) establishes that its `mode` split is speaker independent. We therefore do not claim speaker independence.

The 2-fps YuNet audit covers all 647 valid clips and a seeded random sample of 1,000 train clips (1,647 clips total). It read 11,902 frames; 228 requested frame seeks failed or came from missing media. Percentages use successfully read frames. Test clips were excluded from face analysis.

| Split | Frames read | 0 faces | 1 face | ≥2 faces |
| --- | ---: | ---: | ---: | ---: |
| train | 7,245 | 0.69% | 63.20% | 36.11% |
| valid | 4,657 | 0.28% | 74.73% | 24.99% |

Face counts show how often multiple faces are visible, but do not identify the active speaker.
