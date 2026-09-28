# Benchmark v1 (frozen definition)

## Cohorts

- **ET-test**, the primary set: EmotionTalk official test groups G00003 and G00015, 1,929 utterances.
- **ET-val**: EmotionTalk official validation groups G00001 and G00012, 1,908 utterances. It is reserved for later model selection and calibration. Speakers 02 and 13 appear in both ET-val and ET-test, so this is not a speaker-independent split.
- **SIMS-valid**: CH-SIMS v2 official `valid` rows, for cross-corpus polarity/valence evaluation. Its observed row count is recorded from the supplied `meta.csv` and is not silently forced to a planned number.
- **Lock-box part 1**: CH-SIMS v2 official `test` rows. The lock-box manifest records only clip identity, relative media path, and SHA-256. No model inference on these rows before X07, when the lock-box is opened once. X04 may append a separate corpus portion under a versioned amendment.

EmotionTalk's primary truth is the Multimodal (`M`) `emotion_result` in seven classes: neutral, joy, sadness, anger, fear, disgust, surprise. Audio, Video, and Text labels are secondary. All samples remain in the evaluation denominator; there is no face-count gate.

## Seven-class metrics

Report macro-F1, unweighted average recall (UAR), weighted accuracy (WA), per-class precision/recall/F1, and a seven-by-seven confusion matrix. Neutral false-positive rate is the fraction of true-neutral utterances predicted as any non-neutral class. Coverage is the fraction with valid output. Rejection, parsing failure, and video decode failure count as errors. Report every speaker separately.

Report percentile 95% intervals from 2,000 bootstrap replicates by utterance and by resampling whole speakers. The ET test set has only four speakers, so speaker-bootstrap intervals are descriptive.

## SIMS-valid metrics, without a fitted threshold

For seven-class probabilities define `s = P(joy) - [P(anger) + P(disgust) + P(fear) + P(sadness)]`. Report Pearson and Spearman correlation between `s` and `label`. For Acc-2, exclude rows with `label = 0` and compare the signs of `s` and `label`. Do not tune thresholds or temperature on the test set.

## Model comparison and reporting

Score base Qwen2.5-Omni-7B and the old checkpoint-5265 adapter with audio (A), silent video (V), and audio-video (AV) on full ET-test and ET-val. Score A and AV on full SIMS-valid. Use the training prompts verbatim and the training media limits. Select the main prediction by argmax over the seven ET classes. Report the eight-class argmax, including `other` as an error, secondarily. Save all eight first-token log probabilities per sample. Use paired samples and 2,000 paired bootstrap replicates for condition and model differences. Report A-versus-A-label and V-versus-V-label as separate native-label analyses, never by subtracting them from M-label results.

Any change to these definitions requires Benchmark v1.1 with a stated reason. The SHA-256 of this file and of each split manifest is recorded in X01 results and report.
