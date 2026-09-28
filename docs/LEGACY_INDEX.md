# Legacy experiment index (read-only)

Source: the legacy project's `当前路线与执行入口.md` as read on 2026-09-28. These directories remain in `/home/administrator/projects/emotion`; no old files were moved or modified. Conclusions are one-line summaries of the historical reports, not X01 validation claims.

| Legacy directory | One-line conclusion | Still useful in X01 or later |
| --- | --- | --- |
| `data/manifests_t0` | T0 fixed training data gates. | Swift training manifests and exact training prompts for leakage/D2 |
| `scripts` | E2/E3 established restricted first-token Qwen scoring. | `eval_zeroshot.py` reference for new scorer and D3 |
| `preflight_v2` | Repaired environment and verified the old default checkpoint; no new full retraining. | `requirements.actual.lock.txt` and environment comparison |
| `av_scene_v1` | Early scene-level checks exposed the attribution gap. | Scenario taxonomy only |
| `av_target_v2` | Target-quality comparison identified observability problems. | Diagnostic definitions |
| `av_calibration_v3` | Neutral/rejection/crop comparisons did not settle deployment calibration. | Calibration questions; no X01 threshold reuse |
| `av_transfer_v4` | Person-transfer and multi-person gating remained unresolved. | Grouped-evaluation caution |
| `av_factor_v5` | Factor isolation probed the sources of AV behavior. | Paired comparison pattern |
| `av_temporal_v6` | Temporal sampling changed the qualified-clip set. | Frame-timing audit idea |
| `av_pipeline_v7` | Offline end-to-end regression ran on 27 clips. | Resume and receipt pattern |
| `av_speed_v8` | Runtime improved with output equivalence checks. | Performance receipt pattern |
| `av_identity_v9` | Target attribution and observability still constrained coverage. | Attribution error taxonomy |
| `av_factor_v10` | Factor and sampling decomposition was diagnostic, not a model upgrade. | Paired factors |
| `av_resolve_v11` | Geometry and sampling checks narrowed ambiguity. | Media geometry checks |
| `av_evidence_v12` | Evidence duration and association continuity were measured. | Interval reporting pattern |
| `av_contract_v13` | Cross-modal contract and duplicate state handling were revised. | Output contract checks |
| `av_service_v14` | Offline result service and interruption recovery were exercised. | Atomic/resume design reference |
| `av_specialist_v15` | Specialist comparison and multi-person tests found no candidate meeting all gates. | Model-comparison receipts and old prediction formats |
| `av_attribution_v16` | 252 Chinese clips gave old Qwen A/AV comparisons, but X01 found training overlap. | Read-only predictions and D3 preprocessed fixtures |
| `av_audiofront_v17` | Fixed audio gain helped locally while binding gates still failed. | Audio input diagnostics only |
| `av_binding_v18` | New ASD improved held-out binding but did not meet precision/coverage gates. | ASD evaluation for later work |
| `av_binding_v19` | Short safe memory increased coverage but failed a session precision gate. | Binding and reset diagnostics |
| `av_input_v20` | Fixed gain brought limited additional coverage; near-mic inputs expose an audio bottleneck. | Input-quality diagnostics |
| `perception/yunet.py` | Existing face detector call pattern. | Read-only model weights and YuNet parameters for X01 face counting |

X01 reimplements only the Qwen restricted scorer and YuNet invocation needed here. The Qwen implementation passed the D3 20-audio plus 20-AV logprob and label consistency gate before full evaluation.
