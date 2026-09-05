# AI Perception & Physical AI — Research Portfolio

This portfolio documents controlled, implementation-level research across visual perception, vision-language understanding, open-vocabulary recognition, efficient deployment, video/audio perception, and model fine-tuning. Each repository is organized around a research domain rather than a single notebook, and each investigates a specific question about accuracy, robustness, or deployment cost using a controlled experimental protocol (fixed data, fixed evaluation code, varied model/implementation).

The unifying thread is **perception for embodied and resource-constrained systems**: how well do modern perception models actually work, how do different implementations of the "same" model diverge, and what does it cost to run them outside a data-center GPU.

## Research Map

```
                    Visual & Multimodal Perception
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                        │
 Depth Estimation      Vision-Language Models     Open-Vocabulary CV
(metric depth, 3D)    (VQA, grounded reasoning)   (zero-shot detection,
        │                       │                  segmentation)
        │                       │                        │
        └───────────┬───────────┴────────────┬───────────┘
                     │                        │
           Efficient AI / Deployment    Video & Multimodal
          (ONNX, OpenVINO, PTQ,           Perception
           4-bit quantization)         (temporal action,
                     │                  anomaly, ASR)
                     │
              Deep Learning /
              Fine-Tuning Foundations
        (LoRA/QLoRA, PEFT, transfer learning,
         custom CNNs, training-protocol design)
```

Robotics/Physical AI is the intended integration point for this work (depth → 3D perception, open-vocabulary grounding → robot perception, quantization → edge deployment) and will be added once the underlying robotics source code is organized into its own repository.

## Selected Quantitative Results

A quick-glance summary of headline numbers — see each repository's README for full results, methodology, and caveats.

**Depth estimation** (59 RealSense RGB-depth pairs)

| Model | MAE | AbsRel | Mean Latency |
|---|---:|---:|---:|
| Depth Pro | 0.2931 m | 13.96% | ~5705 ms/image |
| Depth Anything V2 | 0.3508 m | 15.19% | ~72.4 ms/image |
| ZoeDepth-NYU | 0.3605 m | 17.25% | ~364.1 ms/image |

Depth Pro is the most accurate; Depth Anything V2 is by far the fastest, making it the strongest accuracy/latency balance overall.

**Open-vocabulary detection** (12-image controlled COCO benchmark)

- YOLOE: precision 1.00, F1 0.750, mean IoU 0.9206, 35.4M parameters.
- OWLv2: recall 0.733, F1 0.846 (best of the three), mean latency 740.9 ms/image.
- GroundingDINO: F1 0.741, 172.3M parameters — Official vs. Hugging Face implementations match to 2–4 decimal places.

**Video & multimodal perception** (single reference video, native-mode comparison)

| System | Processing Time | Output |
|---|---:|---|
| YOLOE + ByteTrack | 24.5 s | 38 detections, 3 track IDs |
| OWLv2 + ByteTrack | 61.5 s | 19 detections |
| LLaVA-NeXT-Video | 35.6 s | 9 anomaly labels, 100% coverage |
| Qwen3-VL (chunk-based) | ~3051.7 s | 15/18 chunks flagged anomalous |

TimeSformer (UCF101 subset): Top-1 accuracy 56.67%, Top-5 66.67%, temporal consistency 84.17%.

**Efficient deployment** (DETR, standard vs. 4-bit NF4)

| Model | mAP | AP50 | Mean Inference | Peak GPU Memory |
|---|---:|---:|---:|---:|
| Standard DETR | 53.68% | 73.77% | 0.1265 s | 0.532 GB |
| 4-Bit NF4 DETR | 20.52% | 40.86% | 0.0870 s | 0.374 GB |

4-bit quantization cuts memory ~30% and roughly doubles throughput, at a substantial accuracy cost.

**Fine-tuning**

- BERT/MRPC (native PyTorch, held-out test set): Accuracy 0.826, F1 0.8799.
- Multi-adapter LoRA: Adapter 1 (589,824 params) reaches perplexity 26.66 vs. Adapter 2's (202,752 params) 28.03.
- Custom residual CNN (Flowers, 5-class, from scratch): Test Accuracy 83.85%, Macro F1 0.8388.

## Repositories

| Repository | Research Focus |
|---|---|
| [`depth-estimation-research`](./depth-estimation-research) | Monocular metric depth estimation — Depth Anything V2, ZoeDepth, Depth Pro — evaluated against real RealSense ground truth, and deployed through ONNX/OpenVINO INT8 |
| [`vision-language-model-research`](./vision-language-model-research) | Zero-shot vision-language perception — Qwen2.5-VL, Qwen3-VL, BLIP-2 — for grounded object/action/scene understanding and VQA |
| [`open-vocabulary-computer-vision`](./open-vocabulary-computer-vision) | Zero-shot, open-vocabulary detection and segmentation — OWL-ViT, OWLv2, GroundingDINO, YOLOE, SAM2 — including official-vs-Hugging-Face implementation comparisons |
| [`efficient-ai-model-deployment`](./efficient-ai-model-deployment) | Backbone swapping, ONNX export, post-training quantization (INT8/4-bit) and their effect on accuracy, latency, and model size |
| [`video-multimodal-perception`](./video-multimodal-perception) | Temporal perception — video anomaly detection & tracking, action recognition (TimeSformer), automatic speech recognition (Whisper) |
| [`deep-learning-fine-tuning`](./deep-learning-fine-tuning) | Fine-tuning and training-protocol fundamentals — BERT/MRPC, multi-adapter LoRA, GPT-2 QLoRA, custom residual CNNs from scratch |

## Cross-Cutting Experimental Principles

These principles are applied consistently across every repository:

- **Ground truth is never fed to a model.** Where a sensor ground truth exists (e.g. RealSense depth, COCO annotations), it is used only for post-hoc scoring.
- **Same-data, varied-implementation design.** Official-repository vs. Hugging Face implementation comparisons hold the checkpoint, images, and evaluation protocol fixed and vary only the inference stack.
- **Metrics are labeled by what they actually measure.** Structured-output validity, operational latency/memory, and task accuracy are reported separately and never conflated (e.g. "valid JSON" is not treated as detection accuracy).
- **No fabricated results.** Every number reported below comes from an executed notebook run — nothing is estimated or invented.
- **Small, controlled evaluation sets, described as such.** Sample sizes (e.g. 59 RGB-depth pairs, 12 COCO images) are explicit, and results are framed as controlled validation studies rather than full public benchmarks.

## How to Read This Portfolio

Each repository README follows the same structure: Research Focus → Research Questions → Work Included (by model/method) → Experimental Methodology → Results → Reproducibility → Related Research → Limitations → Future Work. Start with a repository's README for the "what and why," then open the numbered scripts for the implementation.
