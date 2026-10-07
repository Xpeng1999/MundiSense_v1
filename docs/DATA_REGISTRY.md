# Dataset registry — X01

Sizes are on-disk `du -sh` observations from the read-only `emotionAI_dataset` directory on the server. Formats are representative local files. “License unverified” means X01 has not established reuse rights; it is not a claim that the data is unrestricted.

| Name | Size | Observed format | License evidence | Processed in X01 / prior phase | Project relevance |
| --- | ---: | --- | --- | --- | --- |
| Emotiontalk | 34G | TAR, JSON, MP4, WAV | README present; terms unverified | Registered in X01; phase1 media already processed | Primary Chinese seven-class benchmark |
| CH-SIMSV2 | 15G | ZIP, CSV, PKL, MP4 | Terms unverified | Extracted and registered in X01 | Cross-corpus valence check and test lock-box |
| CREMA-D | 3.6G | WAV, CSV | Local `LICENSE.txt`; scope not audited | Phase1 processed; no X01 processing | Secondary English speech emotion data |
| AFFECT-HRI | 2.2G | CSV, PDF | Local readme; terms unverified | No | Human–robot affect signals, peripheral |
| AISHELL-4 | 47G | RTTM, TextGrid, audio | Local README; terms unverified | No | Multi-speaker speech, outside X01 single-speaker scope |
| AVA-ActiveSpeaker | 817M | MP4, download records | Terms unverified | No | Active-speaker work after X01 |
| HEROES | 480M | ZIP, PDF | Local readme; terms unverified | No | Human–robot emotion context, future review |
| HRI-SENSE | 5.4G | CSV and related files | Terms unverified | No | Sensor affect data, peripheral |
| MELD | 11G | TAR.GZ | Local README; terms unverified | No | Conversational emotion, outside X01 |
| RAVDESS | 23G | ZIP, audiovisual clips | Terms unverified | No | English controlled emotion reference |
| e3_datasets | 165G | JSON, ZIP and media | Terms unverified | Existing E3 resources; no X01 processing | Legacy training resources only |
| video_emo_dataset | 11G | repository files and media | Local README; terms unverified | No | Potential video emotion data for later screening |
| voice_emo_dataset | 2.8G | Parquet, CSV, scripts | Terms unverified | No | Potential voice emotion data for later screening |

## X04a 注册补充

| 语料 | 本轮角色 | n | 数据卡 |
| --- | --- | --- | --- |
| MER2026 train | 只训练；保守文本去重后 | 9386 | DATA_CARD_mer2026.md |
| MER2026 test | 只报告，禁止选择 | 572 | DATA_CARD_mer2026.md |
| CREMA-D | 只训练 | 7442 | DATA_CARD_cremad.md |

MER 媒体及派生文件留在 H 盘；MER2025 不另加入。CH-SIMS 任意划分均不训练，test 零模型推理。Benchmark v1.1 保留原 ET/SIMS 定义全文不变。
