# Efficient AI / Model Deployment

Backbone swapping and post-training quantization, studied as first-class experimental variables rather than deployment afterthoughts.

## Research Focus

Model accuracy on a benchmark and model practicality in deployment are different questions. This repository treats **backbone choice** and **quantization strategy** (ONNX PTQ, 4-bit weight quantization) as controlled experimental variables, measuring their combined effect on accuracy, model size, latency, throughput, and memory — independent of any single vision task.

## Research Questions

- Can the same classification head and training protocol be reused across different backbones (ResNet-18, ResNet-50, ViT-B/16, Faster R-CNN ResNet-50 FPN, a custom residual CNN), and how do they compare on accuracy, F1, parameter count, size, training time, latency, throughput, and peak memory?
- For a given backbone, how does deployment representation (PyTorch FP32 → ONNX FP32 → ONNX Dynamic INT8 → ONNX Static INT8) change accuracy and efficiency?
- Does 4-bit (NF4) weight quantization measurably change DETR's detection quality and inference cost relative to standard precision, evaluated on real COCO images rather than a single qualitative example?

## Work Included

### Backbone Comparison + ONNX/PTQ
- `01_backbone_comparison_onnx_ptq_benchmark.py` — two-part study: (I) backbone swapping across ResNet-18, ResNet-50, ViT-B/16, Faster R-CNN ResNet-50 FPN, and a custom residual CNN under one training protocol; (II) for each selected backbone, deployment across PyTorch FP32, ONNX FP32, ONNX Dynamic INT8, and ONNX Static INT8.

### DETR Quantization
- `02_detr_fp32_vs_int4_quantization_comparison.py` — the same pretrained DETR model in standard precision vs. NF4 4-bit quantization (`bitsandbytes`), evaluated on multiple COCO 2017 validation images (extending the original single-image example) against COCO ground-truth boxes.

## Experimental Methodology

- **Controlled variables:** for each comparison, the same images, checkpoint, preprocessing, and evaluation protocol are held fixed; only the backbone or the numeric precision/quantization scheme varies.
- **Metrics:** accuracy, macro-F1, weighted-F1, parameter count, model size, training time, inference latency, throughput, peak memory (backbone study); COCO-annotation-based detection quality plus inference cost (DETR study).
- **Ground truth:** COCO 2017 validation annotations for the DETR experiment, used only for post-hoc scoring.
- **Scope:** these are controlled quantization/backbone studies on fixed datasets — not full production deployment benchmarks across hardware targets.

## Results

**`02_detr_fp32_vs_int4_quantization_comparison` — Standard vs. 4-bit NF4 DETR (COCO 2017 val)**

| Model | mAP | AP50 | AP75 | Mean Time (s) | Median Time (s) | Images/sec | Peak GPU Memory (GB) |
|---|---|---|---|---|---|---|---|
| Standard DETR | 0.5368 | 0.7377 | 0.5822 | 0.1265 | 0.0717 | 7.90 | 0.5321 |
| 4-Bit NF4 DETR | 0.2052 | 0.4086 | 0.2176 | 0.0870 | 0.0853 | 11.49 | 0.3736 |

4-bit quantization reduces peak GPU memory by ~30% and increases median throughput from 7.90 to 11.49 images/sec, at the cost of a lower mAP (0.537 → 0.205) and AP50 (0.738 → 0.409).

## Reproducibility

- **Environment:** Google Colab (GPU recommended for training and for meaningful latency measurement).
- **Requirements:** PyTorch, `torchvision`, `onnx`/`onnxruntime` (backbone study), `transformers`, `bitsandbytes`, `pycocotools` (DETR study), Pillow/matplotlib/pandas.
- **Data:** COCO 2017 validation annotations/images are fetched as needed by the DETR script; the backbone study's classification dataset is notebook-specific.
- **Execution:** run top-to-bottom; each script installs its dependencies in its first cell(s).

## Research Evidence

Each experiment links directly to its canonical script above. Depth-specific ONNX/OpenVINO quantization work — which shares this repository's methodology — has its canonical implementation in [`depth-estimation-research`](../depth-estimation-research) and is cross-linked rather than duplicated here.

## Related Research

- [`depth-estimation-research`](../depth-estimation-research) — canonical implementation of ONNX export + OpenVINO INT8 quantization applied specifically to Depth Anything V2.
- [`open-vocabulary-computer-vision`](../open-vocabulary-computer-vision) — DETR's detection task is closely related to this repository's open-vocabulary detectors; DETR's *quantization* study lives here because that is its primary research question.

## Research Direction

This repository's goal is to characterize the accuracy/cost trade-off of model compression techniques in a task-agnostic way, so that the same evidence base can inform deployment decisions across the vision, VLM, and depth work elsewhere in this portfolio — particularly for edge/robotics deployment.

## Limitations

- Backbone comparison and DETR quantization use fixed, moderate-size datasets — results characterize behavior on those sets, not universal quantization behavior.
- No on-device (embedded/edge hardware) latency measurement yet — all latency/throughput figures are measured in the Colab environment.
- 4-bit quantization is evaluated only for DETR; INT8 ONNX quantization is evaluated only for the backbone-comparison study and (separately) Depth Anything V2.

## Future Work

- Benchmark on actual edge hardware (e.g. OpenVINO on an Intel NUC, or a Jetson device) rather than Colab GPU/CPU.
- Apply the same INT8 ONNX pipeline used for Depth Anything V2 to the open-vocabulary detectors.
- Extend 4-bit quantization evaluation beyond DETR to other detection/VLM models in this portfolio.

## Research Takeaway

For DETR, 4-bit NF4 quantization delivers a real efficiency gain — roughly 30% less peak GPU memory and a jump in throughput from 7.90 to 11.49 images/sec — but at a substantial cost to detection quality (mAP dropping by nearly a third). This repository's controlled, single-variable design makes that trade-off explicit rather than reporting efficiency gains in isolation: any deployment decision here needs both numbers side by side, not just the faster one.
