# Vision-Language Model Research

Zero-shot vision-language perception: grounded object/action/scene understanding and visual question answering, with a deliberate focus on separating *semantic* competence from *spatial* and *computational* competence.

## Research Focus

Vision-language models (VLMs) can describe, localize, and reason about images without task-specific training. This repository evaluates three model families — Qwen3-VL, Qwen2.5-VL, and BLIP-2 — as zero-shot perception systems, measuring not just whether they produce plausible-looking output but whether that output is structurally valid, spatially grounded, and computationally practical.

## Research Questions

- Can a VLM reliably produce machine-readable (schema-valid) structured output for object detection, action recognition, and scene description, and how often does validation fail?
- How consistent are a model's own bounding-box and confidence estimates when re-queried on the same image?
- Does the newer Qwen3-VL-2B improve on Qwen2.5-VL-3B for zero-shot object/action/scene understanding, or trade accuracy for size?
- For visual question answering (BLIP-2), how does answer quality (exact match, token F1) trade off against latency and peak GPU memory?

## Work Included

### Qwen3-VL-2B-Instruct
- `01_qwen3_vl_2b_zero_shot_perception_baseline.py` — baseline zero-shot pipeline: object detection, 2D bounding-box estimation, observable action/state inference, scene description, enforced JSON schema, automatic bounding-box/confidence validation, and latency/memory logging.
- `02_qwen3_vl_2b_object_action_scene_evaluation.py` — extended research-grade evaluation following the same structure as the Qwen2.5-VL benchmark, explicitly distinguishing semantic understanding, spatial grounding, action recognition, and computational efficiency as separate axes that require separate ground truth to score.

### Qwen2.5-VL-3B-Instruct
- `03_qwen2_5_vl_3b_zero_shot_perception_benchmark.py` — zero-shot object localization, action/state recognition, and scene-level understanding, with structured-output reliability, inference latency, GPU memory, and repeatability measured explicitly. JSON validity is treated as a separate axis from detection accuracy throughout.

### BLIP-2 (OPT-2.7B)
- `04_blip2_visual_question_answering_evaluation.py` — VQA-only evaluation (image captioning removed to keep the task focused), scored with Exact Match, Token F1, and VQA Accuracy against supplied reference answers, alongside per-question latency and peak VRAM.

## Experimental Methodology

- **Task design:** each notebook enforces a machine-readable JSON output schema and validates it automatically (well-formed JSON, in-range bounding boxes/confidences) before any downstream scoring.
- **Metric separation:** structured-output validity, semantic/action correctness, and efficiency (latency, memory) are reported as independent measurements — a valid JSON response is never counted as a correct detection.
- **Reference-relative metrics:** BLIP-2's Exact Match / Token F1 / VQA Accuracy are explicitly scoped as valid only relative to the supplied reference answers, not as a universal accuracy score for the model.
- **Repeatability:** where applicable, models are re-queried on the same input to check consistency of bounding boxes and confidence values.
- **Hardware:** Google Colab GPU runtime; models loaded via Hugging Face `transformers`, with `bitsandbytes` used for memory-constrained configurations.

## Results

**`02_qwen3_vl_2b_object_action_scene_evaluation` — Qwen3-VL-2B-Instruct operational scorecard**

| Action Coverage (%) | Repeatability (%) | Latency (s) | Peak VRAM (GB) |
|---|---|---|---|
| 100.0 | 100.0 | 9.892 | 1.752 |

Under a separate 4-bit NF4 quantized run (3 repeats): mean latency 9.8925 s (std 0.2839 s), mean generated tokens 88.0, mean peak VRAM 1.7519 GB, repeatability 100.0%. The model's action recognition and repeatability behavior are fully consistent across repeated queries on the same input.

**`03_qwen2_5_vl_3b_zero_shot_perception_benchmark`** — representative qualitative example: given a cat image, the model returned a valid JSON detection (`bbox_2d`, `label: "cat"`, `action: "resting"`) with a correct scene description, demonstrating the pipeline's schema and grounding working end-to-end.

**`04_blip2_visual_question_answering_evaluation` — BLIP-2 VQA (4-question sample)**

| Metric | Value |
|---|---|
| VQA Accuracy | 25.0% |
| Mean Token F1 | 39.2857% |
| Mean Semantic Similarity | 60.5306% |
| Hallucination Rate | 0.0% |
| Mean Unsupported Tokens | 1.25 |
| Mean Answer Length | 2.75 words |
| Mean Latency | 0.6525 s |
| Mean Peak VRAM | 7.7301 GB |

On this 4-question sample, BLIP-2 answered "How many cats are there?" exactly correctly, and gave a partially correct answer ("cats") to "What animal is shown?" (Token F1 0.57), with a 0% hallucination rate across all four questions — the model never introduced unsupported entities in its answers.

## Reproducibility

- **Environment:** Google Colab (GPU recommended, particularly for the 2–3B parameter VLMs).
- **Requirements:** `transformers` (latest, installed from source in some notebooks), `bitsandbytes`, `accelerate`, `qwen-vl-utils` (Qwen models), `huggingface_hub`, Pillow, matplotlib, pandas.
- **Data:** input images/questions are notebook-specific; substitute your own images and (for BLIP-2) reference Q&A pairs to reproduce the pipeline.
- **Execution:** run top-to-bottom; each notebook installs its own dependencies in its first cell.

## Research Evidence

Each experiment links directly to its canonical script above. Grounding/localization work here is complementary to, but not duplicated with, the dedicated detection models in [`open-vocabulary-computer-vision`](../open-vocabulary-computer-vision).

## Related Research

- [`open-vocabulary-computer-vision`](../open-vocabulary-computer-vision) — dedicated zero-shot detection/segmentation models (OWL-ViT, GroundingDINO, SAM2) for grounding tasks VLMs perform more generally.
- [`video-multimodal-perception`](../video-multimodal-perception) — extends VLM-style reasoning (Qwen3-VL) into the temporal/video domain for anomaly detection.
- [`depth-estimation-research`](../depth-estimation-research) — complementary geometric perception modality alongside this repository's semantic perception.

## Research Direction

This work treats VLMs as general-purpose perception front-ends and is aimed at understanding where they are reliable enough (structured output, repeatability) to serve as a perception layer for downstream systems such as robots or automated visual QA, rather than only as chat-style image describers.

## Limitations

- No independently annotated ground truth for object/action localization in the Qwen notebooks — spatial and action accuracy are explicitly flagged as unverifiable without such annotations, and only proxy/operational metrics (validity, consistency, latency) are reported with confidence.
- BLIP-2 metrics are only meaningful relative to the specific reference answers used, not as a general accuracy benchmark.
- Single-image, single-turn evaluation; no multi-turn dialogue or video input in this repository (see `video-multimodal-perception` for the latter).

## Future Work

- Add independently annotated images to convert the Qwen notebooks' proxy metrics into verified detection/action accuracy.
- Extend BLIP-2 evaluation to a larger, standard VQA benchmark for external comparability.
- Compare Qwen3-VL and Qwen2.5-VL directly on an identical image/prompt set rather than parallel but separate benchmarks.

## Research Takeaway

These experiments show that zero-shot VLMs can be operationally reliable — perfectly repeatable across queries, with full action-coverage on Qwen3-VL-2B and a working grounded-detection pipeline on Qwen2.5-VL-3B — while VQA accuracy on BLIP-2 remains modest on out-of-distribution questions even with zero hallucination. The evaluation methodology deliberately keeps structured-output validity, semantic correctness, and operational metrics (latency, repeatability, VRAM) as separate axes, since a model can score well on one without scoring well on another.
