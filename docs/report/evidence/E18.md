# EmotionTalk data card — X01 audit

## Source and scope

The read-only source is `Emotiontalk/README.md` and `phase1_v1/emotiontalk/{Multimodal,Audio,Video,Text}/json`. The manifest includes annotation labels, five votes per modality (emotion and confidence), the M continuous label, relative media paths, SHA-256, and media probes. It excludes text content and transcripts. The media source is the read-only processed directory.

## Official split and labels

The 19,250 M-annotation rows match every cell in the source README after mapping Happy→joy, Angry→anger, Sad→sadness, Disgusted→disgust, Fearful→fear, and Surprised→surprise. The JSON source uses the normalized spellings.

| Split | neutral | joy | sadness | anger | fear | disgust | surprise | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 5,377 | 2,986 | 919 | 2,950 | 672 | 1,142 | 1,367 | 15,413 |
| Val | 675 | 360 | 111 | 409 | 125 | 95 | 133 | 1,908 |
| Test | 801 | 246 | 123 | 339 | 125 | 134 | 161 | 1,929 |

Val groups are G00001 and G00012; test groups are G00003 and G00015. All other groups are train. The 19 recorded speakers and their group/split relationships are in `results/et_audit.json`. Speakers 02 and 13 occur in both val and test; no training speaker occurs in val or test.

| Speaker | Groups | Splits | Rows |
| --- | --- | --- | ---: |
| 01 | G00001 | val | 451 |
| 02 | G00001, G00003 | val, test | 803 |
| 05 | G00004, G00010 | train | 1,209 |
| 06 | G00005, G00011 | train | 965 |
| 07 | G00002, G00006, G00008 | train | 1,929 |
| 08 | G00002, G00010 | train | 1,205 |
| 09 | G00004, G00011 | train | 1,548 |
| 10 | G00005 | train | 553 |
| 11 | G00007, G00008 | train | 1,748 |
| 12 | G00006, G00009 | train | 1,618 |
| 13 | G00003, G00012 | val, test | 945 |
| 14 | G00007, G00009 | train | 1,776 |
| 15 | G00012 | val | 502 |
| 16 | G00013 | train | 681 |
| 17 | G00013 | train | 702 |
| 18 | G00014 | train | 728 |
| 19 | G00014 | train | 751 |
| 20 | G00015 | test | 525 |
| 21 | G00015 | test | 611 |

## Annotation agreement

Across 19,250 rows, M agrees with A on 13,045 (0.6777), V on 12,756 (0.6626), and T on 8,134 (0.4225). The complete 4×4 pairwise matrix and every pair stratified by M class are in `results/et_audit.json`. Fleiss' kappa among the five raw annotators is M 0.7832, A 0.7888, V 0.7265, and T 0.6564. Maximum vote strengths of 5/5, 4/5, 3/5, and 2/5 are recorded separately for each modality in that same JSON. A 2/5 maximum is possible with distributed votes; it is not treated as a majority.

Overall pairwise `emotion_result` agreement:

|  | M | A | V | T |
| --- | ---: | ---: | ---: | ---: |
| M | 1.0000 | 0.6777 | 0.6626 | 0.4225 |
| A | 0.6777 | 1.0000 | 0.5681 | 0.4642 |
| V | 0.6626 | 0.5681 | 1.0000 | 0.3550 |
| T | 0.4225 | 0.4642 | 0.3550 | 1.0000 |

M versus each other modality, conditioned on the M class:

| M class | M=A | M=V | M=T |
| --- | ---: | ---: | ---: |
| neutral | 0.8408 | 0.6739 | 0.5290 |
| joy | 0.5025 | 0.9092 | 0.2160 |
| sadness | 0.7042 | 0.5802 | 0.3114 |
| anger | 0.8380 | 0.7839 | 0.5446 |
| fear | 0.5054 | 0.2820 | 0.3883 |
| disgust | 0.2750 | 0.3581 | 0.1940 |
| surprise | 0.4359 | 0.3329 | 0.4431 |

Maximum same-label vote count among five annotators (count and share of 19,250):

| Modality | 2 votes | 3 votes | 4 votes | 5 votes |
| --- | ---: | ---: | ---: | ---: |
| M | 24 (0.12%) | 866 (4.50%) | 6,662 (34.61%) | 11,698 (60.77%) |
| A | 34 (0.18%) | 839 (4.36%) | 5,677 (29.49%) | 12,700 (65.97%) |
| V | 55 (0.29%) | 1,612 (8.37%) | 7,321 (38.03%) | 10,262 (53.31%) |
| T | 30 (0.16%) | 741 (3.85%) | 11,004 (57.16%) | 7,475 (38.83%) |

## Face-count audit

At 2 fps, YuNet audited all test and val clips plus 2,000 seeded random train clips (5,837 clips total, seed 20260928). It read 55,411 frames; 1,335 requested frame seeks failed or came from missing media. Percentages below use successfully read frames as their denominator. A detected face is not identified as the speaker.

### By split

| Split | Frames read | 0 faces | 1 face | ≥2 faces |
| --- | ---: | ---: | ---: | ---: |
| train | 18,653 | 0.03% | 93.09% | 6.88% |
| val | 18,222 | 0.03% | 78.88% | 21.10% |
| test | 18,536 | 0.00% | 96.04% | 3.96% |

### By speaker

| Speaker | Frames read | 0 faces | 1 face | ≥2 faces |
| --- | ---: | ---: | ---: | ---: |
| 01 | 3,949 | 0.10% | 95.90% | 4.00% |
| 02 | 8,685 | 0.01% | 97.64% | 2.35% |
| 05 | 1,369 | 0.00% | 86.12% | 13.88% |
| 06 | 1,107 | 0.00% | 94.22% | 5.78% |
| 07 | 2,049 | 0.00% | 97.12% | 2.88% |
| 08 | 1,883 | 0.21% | 92.30% | 7.49% |
| 09 | 2,312 | 0.04% | 87.28% | 12.67% |
| 10 | 470 | 0.00% | 94.89% | 5.11% |
| 11 | 2,178 | 0.00% | 89.49% | 10.51% |
| 12 | 1,967 | 0.00% | 94.66% | 5.34% |
| 13 | 8,781 | 0.00% | 96.09% | 3.91% |
| 14 | 1,908 | 0.00% | 96.49% | 3.51% |
| 15 | 4,374 | 0.00% | 24.87% | 75.13% |
| 16 | 710 | 0.00% | 98.03% | 1.97% |
| 17 | 894 | 0.00% | 99.89% | 0.11% |
| 18 | 810 | 0.00% | 99.01% | 0.99% |
| 19 | 996 | 0.00% | 91.06% | 8.94% |
| 20 | 5,190 | 0.00% | 93.51% | 6.49% |
| 21 | 5,779 | 0.00% | 95.67% | 4.33% |

Speaker 15 has the largest observed multi-face fraction in this audit. The val split also has a higher multi-face fraction than test. These counts weaken the assumption that a single visible face is always the current speaker; they do not identify which face is speaking.

## Limitations

The camera view may contain zero or multiple visible faces. The M label is an annotation target, not a clinical state. The validation and test splits share speakers 02 and 13, so their difference is not a speaker-independent generalization estimate. The test set contains only four speakers.
