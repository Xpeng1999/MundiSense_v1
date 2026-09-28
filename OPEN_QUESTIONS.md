# Open questions for reviewer

1. The supplied CH-SIMS v2 `meta.csv` contains **4,403 data rows** (2,722 train, 647 valid, 1,034 test), while X01 specifies 4,402. No row is blank and `(video_id, clip_id)` has no duplicate. We retain all 4,403 rows for audit. Please confirm whether an official row should be excluded, with its ID and source, before changing Benchmark v1.
