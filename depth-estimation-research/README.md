# Depth Estimation Research

Monocular metric depth estimation, evaluated against real sensor ground truth and pushed through a full deployment pipeline (PyTorch → ONNX → OpenVINO INT8).

## Research Focus

Monocular depth estimation recovers dense, per-pixel scene geometry from a single RGB image — no stereo pair, no active sensor. It underlies navigation, obstacle avoidance, and 3D scene understanding for cameras without dedicated depth hardware. This repository asks how three modern metric-depth models (Depth Anything V2, ZoeDepth, Depth Pro) actually perform against **real, physically captured ground truth**, how their implementations (official repo vs. Hugging Face) diverge, and what it costs — in accuracy and latency — to deploy them at the edge.

## Research Questions

- How accurate is each model's *metric* (not just relative) depth output against dense, per-pixel RealSense ground truth?
- Do official-repository and Hugging Face implementations of the same checkpoint produce consistent predictions, or does the interface itself introduce measurable divergence?
- What is the accuracy/efficiency trade-off across Depth Anything V2, ZoeDepth, and Depth Pro under an identical protocol?
- How much accuracy is lost, and how much latency/model-size is gained, when Depth Anything V2 is quantized and deployed via ONNX Runtime and OpenVINO INT8?

## Work Included

### Depth Anything V2
- `01_depth_anything_v2_onnx_openvino_int8_pipeline.py` — full deployment pipeline: official repo vs. Hugging Face PyTorch inference → ONNX export/profiling → OpenVINO IR + NNCF INT8 quantization using real calibration images.
- `03_depth_anything_v2_realsense_deployment_benchmark.py` — three-implementation comparison (official PyTorch, Hugging Face, 4-bit BitsAndBytes) on metric accuracy, consistency, and inference time.
- `06_depth_anything_v2_multirepresentation_deployment.py` — controlled evaluation across five deployment representations: official PyTorch, Hugging Face, ONNX Runtime, OpenVINO FP32, OpenVINO INT8.

### ZoeDepth
- `05_zoedepth_indoor_metric_depth_error_analysis.py` — ZoeDepth-NYU evaluated against dense RealSense ground truth in an indoor lab setting.

### Depth Pro
- `04_depthpro_metric_depth_evaluation.py` — Apple Depth Pro zero-shot metric depth and focal-length estimation, evaluated on the same RealSense pairs.

### Three-Model Comparison
- `02_depth_anything_v2_zoedepth_depthpro_comparison.py` — Depth Anything V2, ZoeDepth, and Depth Pro evaluated under one common protocol on the same 59 synchronized RGB-depth pairs.

## Experimental Methodology

- **Dataset:** 59 synchronized RGB-depth pairs captured with an Intel RealSense depth camera in a real indoor lab environment. Depth is stored in millimeters and converted to meters (×0.001); invalid (zero-valued) pixels are excluded from every metric.
- **Protocol:** the RGB frame is the sole model input; the RealSense depth map is used only afterward, to score predictions — never as input.
- **Metrics:** pixel-wise MAE, RMSE, Absolute Relative Error (%), image-level median depth error, prediction consistency, inference latency, and model size.
- **Controlled variables:** identical images, identical ground truth, identical scoring code across implementations within each notebook.
- **Hardware:** Google Colab GPU runtime (model-dependent; see individual notebooks for exact device/precision).
- **Scope:** this is a controlled validation study on a fixed indoor set, not a substitute for NYUv2/KITTI-scale benchmarks.

## Results

All figures below are taken directly from executed notebook outputs (59 RealSense RGB-depth pairs unless noted otherwise).

**`01_depth_anything_v2_onnx_openvino_int8_pipeline` — implementation & deployment comparison**

| Implementation | AbsRel | RMSE (m) | δ<1.25 | δ<1.25² | δ<1.25³ | Mean Inference Time (s) |
|---|---|---|---|---|---|---|
| Official Repository | 0.2110 | 0.6304 | 0.6346 | 0.9576 | 0.9924 | 0.1027 |
| Hugging Face PyTorch | 0.2094 | 0.6317 | 0.6379 | 0.9570 | 0.9915 | 0.0751 |
| ONNX Runtime | 0.2327 | 0.6809 | 0.6005 | 0.9468 | 0.9917 | 1.5854 |
| OpenVINO FP32 | 0.2329 | 0.6811 | 0.6002 | 0.9467 | 0.9917 | 1.2068 |
| OpenVINO INT8 | 0.3039 | 0.7513 | 0.4838 | 0.8725 | 0.9793 | 1.1946 |

INT8 quantization trades accuracy for a smaller model: AbsRel rises from ~0.21 to 0.30 and δ<1.25 drops from ~0.63 to 0.48, while remaining latency-competitive with the ONNX/OpenVINO FP32 paths in this Colab environment (absolute latency numbers here reflect Colab overhead, not an optimized edge target).

**`02_depth_anything_v2_zoedepth_depthpro_comparison` — three-model comparison**

| Model | Accuracy Rank | Latency Rank | Spatial Consistency Rank | Mean MAE (m) | Mean AbsRel (%) | Mean Latency (ms) | Mean Surface Std (m) |
|---|---|---|---|---|---|---|---|
| Depth Pro | 1 | 3 | 2 | 0.2931 | 13.9559 | 5704.8027 | 0.7160 |
| Depth Anything V2 | 2 | 1 | 1 | 0.3508 | 15.1936 | 72.4008 | 0.6550 |
| ZoeDepth-NYU | 3 | 2 | 3 | 0.3605 | 17.2476 | 364.0591 | 0.7666 |

Depth Pro is the most accurate but ~79× slower per inference than Depth Anything V2 in this test; Depth Anything V2 is the best accuracy/latency/consistency balance overall.

**`03_depth_anything_v2_realsense_deployment_benchmark` — FP32 vs. INT8 ONNX**

| Metric | FP32 | INT8 |
|---|---|---|
| Mean Depth Error (m) | 0.317203 | 0.346097 |
| AbsRel (%) | 14.357159 | 17.586846 |
| Latency (ms) | 1210.307662 | 1153.993902 |

FP32→INT8 depth difference: 0.132663 m mean. INT8 modestly reduces latency (~5%) at the cost of a ~3-point AbsRel increase.

**`04_depthpro_metric_depth_evaluation` — Depth Pro standalone**

| Metric | Value |
|---|---|
| Samples | 17 |
| Mean Absolute Error (m) | 0.234711 |
| Mean RMSE (m) | 0.315903 |
| Median Absolute Error (m) | 0.144222 |
| Mean Relative Error (%) | 13.268662 |
| Mean Surface Depth Std (m) | 0.830402 |
| Mean Inference Time (s) | 377.591413 |

This run used a 17-image subset rather than the full 59-pair set; inference time is notably higher than in the three-model comparison, likely reflecting a different runtime/precision configuration in this standalone run.

**`05_zoedepth_indoor_metric_depth_error_analysis` — ZoeDepth per-frame error**

Per-frame Mean Absolute Error was computed across all 59 frames. Most frames fall in the 0.2–0.5 m range, with a handful of higher-error outlier frames reaching ~0.7–1.06 m (e.g. frames 00032 and 00051), consistent with the 0.3605 m mean MAE reported for ZoeDepth-NYU in the three-model comparison above.

**`06_depth_anything_v2_multirepresentation_deployment` — implementation comparison**

| Implementation | MAE (m) | RMSE (m) | AbsRel (%) | Median Error (m) | Mean Surface Std (m) | Mean Inference Time (s) | Median Inference Time (s) |
|---|---|---|---|---|---|---|---|
| Hugging Face | 0.350755 | 0.483432 | 15.193560 | 0.259012 | 0.655032 | 3.391984 | 3.097162 |
| Official Repository | 0.393508 | 0.513980 | 17.080577 | 0.315607 | 0.649155 | 5.504435 | 4.829282 |

Hugging Face's implementation is both more accurate and faster than the official repository implementation under this protocol — the interface/preprocessing difference is not negligible.

## Reproducibility

- **Environment:** Google Colab (GPU runtime recommended). Each script installs its own dependencies via `pip install` at the top of the file.
- **Requirements:** PyTorch, Hugging Face `transformers`, `onnx`/`onnxruntime`, OpenVINO + NNCF (for quantization scripts), Pillow/matplotlib/pandas for I/O and reporting.
- **Data:** the 59 RealSense RGB-depth pairs are not redistributed in this repository; they were captured locally. Substituting any calibrated RGB-D dataset (e.g. a NYUv2 subset) with the same directory structure should allow the scripts to run unmodified.
- **Execution:** run scripts top-to-bottom in Colab or a local Jupyter environment; each is a linear, single-notebook pipeline.

## Research Evidence

Each experiment above links directly to its canonical script in this repository — there is no duplicated implementation across other repositories. The ONNX/OpenVINO quantization *methodology* is shared conceptually with [`efficient-ai-model-deployment`](../efficient-ai-model-deployment), but the canonical depth-specific implementation stays here.

## Related Research

- [`efficient-ai-model-deployment`](../efficient-ai-model-deployment) — general-purpose ONNX/PTQ methodology this repository applies to depth models specifically.
- [`vision-language-model-research`](../vision-language-model-research) — complementary scene-understanding modality (semantic vs. geometric perception).
- Robotics/Physical AI (planned) — depth estimation is a direct input to robot navigation and obstacle avoidance.

## Research Direction

This work sits at the intersection of 3D perception and edge deployment: verifying that metric depth models are accurate enough, and fast enough, to be useful outside a research GPU — a prerequisite for using monocular depth in real robotic or embedded navigation systems.

## Limitations

- Ground-truth set is small (59 pairs) and single-environment (one indoor lab); results characterize behavior in that setting, not general-purpose accuracy.
- Official-vs-Hugging-Face comparisons can be confounded by preprocessing differences, not just architecture — this is called out explicitly rather than treated as a pure implementation comparison.
- No outdoor, multi-scene, or long-range depth evaluation is included.

## Future Work

- Extend evaluation to a public benchmark (NYUv2/KITTI) for external comparability.
- Broaden the RealSense set across more rooms/lighting conditions.
- Quantize and benchmark ZoeDepth and Depth Pro through the same ONNX/OpenVINO pipeline currently applied only to Depth Anything V2.

## Research Takeaway

Across all six experiments, accuracy and deployment cost move independently rather than together: Depth Pro is the most accurate model tested but also the slowest by roughly two orders of magnitude, while Depth Anything V2 offers the strongest accuracy/latency balance and remains usable after ONNX/OpenVINO INT8 quantization at a measurable but modest accuracy cost. Implementation choice matters too — Hugging Face's Depth Anything V2 path was both more accurate and faster than the official repository path under the same protocol. This repository treats metric accuracy and deployment efficiency as separate, jointly-tracked dimensions rather than assuming one predicts the other.
