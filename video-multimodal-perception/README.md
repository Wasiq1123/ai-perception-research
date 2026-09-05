# Video & Multimodal Perception

Temporal and audio perception: video anomaly detection and tracking, action recognition, and speech recognition — extending the portfolio's image-level perception work into time.

## Research Focus

Static-image perception (detection, VLM reasoning, depth) doesn't capture temporal behavior — actions, anomalies, and speech all unfold over time. This repository evaluates models built specifically for temporal or sequential input: zero-shot detectors with tracking, video-level VLM reasoning, a Kinetics-400 action-recognition transformer, and a state-of-the-art ASR model.

## Research Questions

- Out-of-the-box (no fine-tuning), which combination of zero-shot detectors, tracking, and video-VLMs is best suited to tracking objects and flagging human/mechanical anomalies in a video feed?
- Does a Kinetics-400-finetuned TimeSformer generalize to a different action-recognition dataset (UCF101) when the action classes have direct semantic counterparts?
- What is the practical latency/quality profile of Whisper Large-v3-Turbo for GPU-accelerated, timestamped transcription?

## Work Included

### Video Anomaly Detection & Tracking
- `01_video_anomaly_detection_model_comparison.py` — compares four models in their native operating modes on a common reference video: YOLOE (zero-shot detection + ByteTrack), OWLv2 (open-vocabulary detection + ByteTrack), LLaVA-NeXT-Video (video-level VQA reasoning), and Qwen3-VL (chunk-based temporal anomaly aggregation). Explicitly framed as an out-of-the-box comparison, not a production system or fine-tuning study.

### Action Recognition
- `02_timesformer_multi_video_action_recognition.py` — `facebook/timesformer-base-finetuned-k400` evaluated across multiple labeled videos and temporal clips from a UCF101 subset (classes with direct Kinetics-400 counterparts), measuring classification accuracy, top-5 accuracy, confidence, and prediction consistency — a controlled cross-dataset evaluation, not a Kinetics-400 benchmark.

### Automatic Speech Recognition
- `03_whisper_large_v3_turbo_asr_evaluation.py` — end-to-end ASR pipeline with `whisper-large-v3-turbo`: hardware-aware (CUDA/FP16 vs. CPU/FP32) inference, chunked timestamped transcription, and structured CSV export with runtime diagnostics.

## Experimental Methodology

- **Anomaly/tracking comparison:** a single reference video is evaluated across each model's *native* mode (frame-level detection+tracking vs. video-level VLM reasoning vs. chunk-based temporal aggregation) rather than forcing all models into one interface.
- **Action recognition:** the TimeSformer checkpoint is held fixed across every video/clip; a labeled UCF101 subset provides ground-truth action labels for cross-dataset evaluation against the model's Kinetics-400 training.
- **ASR:** transcription correctness is inspected against the source audio; runtime is measured under both GPU (FP16) and CPU (FP32) fallback paths.

## Results

**`01_video_anomaly_detection_model_comparison` — one reference video, four models in native mode**

| Model | Primary Capability | Processing Time (s) | Detections | Detection Density | Notes |
|---|---|---|---|---|---|
| YOLOE | Zero-shot detection + tracking | 24.5 | 38 | 0.432 | 3 unique track IDs; fastest of the four |
| OWLv2 | Open-vocab detection + tracking | 61.5 | 19 | 0.413 | 0 unique track IDs recovered in this run |
| LLaVA-NeXT-Video | Video-level VLM reasoning | 35.6 | — | — | Reported 9 anomaly labels, 100% anomaly coverage |
| Qwen3-VL | Chunk-based VLM anomaly reasoning | 3051.7 | — | — | 18 chunks, 15 flagged anomalous (83.3% anomaly-chunk rate), mean chunk latency 169.5 s |

The four models operate in genuinely different modes, so this is a qualitative capability comparison rather than a single ranked metric: YOLOE/OWLv2 give fast, frame-level detection+tracking; the video-VLMs (LLaVA-NeXT-Video, Qwen3-VL) give richer anomaly *reasoning* at far higher latency — Qwen3-VL's chunk-based analysis took over 50 minutes on this one reference video (3051.7 s total pipeline time), a real practicality constraint for any live-deployment use case.

**`02_timesformer_multi_video_action_recognition` — UCF101 subset (30 videos, 120 clips)**

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.5667 |
| Top-5 accuracy | 0.6667 |
| Macro F1 | 0.3222 |
| Temporal consistency | 0.8417 |

The Kinetics-400-finetuned TimeSformer generalizes only moderately to UCF101's overlapping action classes (57% top-1) and the low Macro F1 (0.32) relative to Top-1 accuracy suggests uneven performance across classes rather than uniform 57% accuracy on every class. Temporal consistency (0.84) shows the model's predictions are relatively stable within a video even when not perfectly correct.

**`03_whisper_large_v3_turbo_asr_evaluation`**

| Metric | Value |
|---|---|
| Files transcribed | 1 |
| Total audio duration | 18.36 s |
| Total inference time | 1.84 s |
| Real-time factor | 0.100 |

A real-time factor of 0.10 means transcription ran ~10× faster than real time on this sample — but as the notebook itself notes, this is based on a single short audio file and should not be treated as a stable throughput benchmark.

## Reproducibility

- **Environment:** Google Colab (GPU strongly recommended for all three notebooks).
- **Requirements:** `transformers`, `accelerate`, `bitsandbytes` (VLM notebook), model-specific detection/tracking dependencies (YOLOE, OWLv2, ByteTrack), audio-processing libraries for Whisper.
- **Data:** the anomaly-detection reference video and Whisper's audio asset are fetched from Hugging Face Hub within the notebooks; the UCF101 subset used for action recognition is the Hugging Face-hosted subset referenced in the script.
- **Execution:** run top-to-bottom; each notebook installs dependencies in its first cell(s).

## Research Evidence

Each experiment links directly to its canonical script above.

## Related Research

- [`vision-language-model-research`](../vision-language-model-research) — Qwen3-VL's image-level perception work extended here into chunk-based video reasoning.
- [`open-vocabulary-computer-vision`](../open-vocabulary-computer-vision) — YOLOE and OWLv2 detection methodology, extended here with ByteTrack for temporal tracking (excluded in the image-level open-vocabulary benchmarks).

## Research Direction

This repository extends the portfolio's static-perception work into the temporal domain, which is directly relevant to any embodied or robotic system that must track objects, recognize actions, or process spoken instructions over time rather than from a single frame.

## Limitations

- Anomaly detection/tracking comparison uses a single reference video — it characterizes relative model behavior on that video, not a general anomaly-detection benchmark.
- TimeSformer evaluation depends on action classes overlapping between UCF101 and Kinetics-400; classes without a direct counterpart are out of scope.
- No fine-tuning is performed anywhere in this repository — all models are evaluated strictly out-of-the-box.

## Future Work

- Evaluate the anomaly-detection comparison across multiple videos/scenarios rather than one reference clip.
- Fine-tune TimeSformer or a comparable model directly on UCF101 to separate "zero-shot cross-dataset transfer" from "fine-tuned" performance.
- Combine Whisper transcription with the VLM/anomaly pipeline for multimodal (audio+video) event detection.

## Research Takeaway

The three experiments span three distinct temporal-perception paradigms with very different cost profiles: frame-level detection+tracking (YOLOE/OWLv2) runs in under a minute, chunk-based VLM anomaly reasoning (Qwen3-VL) takes nearly an hour on the same reference video, and Whisper transcribes speech roughly 10× faster than real time. No single evaluation framework fits all three — the repository's approach of reporting each system's native operational metrics, rather than forcing a shared accuracy score, reflects that these are fundamentally different tools for different parts of a temporal perception pipeline.
