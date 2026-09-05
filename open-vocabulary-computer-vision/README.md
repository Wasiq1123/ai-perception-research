# Open-Vocabulary Computer Vision

Zero-shot, open-vocabulary object detection and segmentation — evaluated with real COCO/Oxford-IIIT annotations, and cross-checked between official-repository and Hugging Face implementations.

## Research Focus

Open-vocabulary perception lets a system detect or segment objects specified by free-text queries at inference time, rather than a fixed, pre-trained class set — important for robots and vision systems operating in unstructured, previously unseen environments. This repository evaluates four model families (OWL-ViT, GroundingDINO, YOLOE, SAM2, and OWLv2) against real annotated ground truth, and specifically investigates whether official-repository and Hugging Face implementations of the same checkpoint behave equivalently.

## Research Questions

- How well do zero-shot, text-prompted detectors perform against real COCO ground-truth boxes (precision, recall, F1, IoU), not just qualitative visualization?
- Do official-repository and Hugging Face implementations of the same model (GroundingDINO, SAM2) produce practically consistent detections/masks, or does the interface introduce measurable divergence?
- Under one controlled protocol (same images, same target concepts, same threshold), how do YOLOE, OWLv2, and GroundingDINO trade off precision/recall/F1 against latency, throughput, and model size?
- How sensitive is detection quality to the confidence threshold chosen at inference time?

## Work Included

### OWL-ViT
- `01_owlvit_zero_shot_object_detection.py` — zero-shot, text-prompted detection: text-image grounding, bounding-box extraction, confidence-threshold sensitivity analysis, and quantitative inspection (detection counts, confidence statistics, box geometry) alongside visualization.

### GroundingDINO
- `02_groundingdino_official_vs_huggingface_comparison.py` — official IDEA-Research repository vs. Hugging Face `AutoModelForZeroShotObjectDetection`, evaluated on the same COCO validation images with COCO-annotation-based TP/FP/FN, precision, recall, F1, and matched IoU, plus per-image latency and threshold-sensitivity analysis.

### YOLOE + OWLv2 + GroundingDINO
- `03_yoloe_owlv2_groundingdino_controlled_benchmark.py` — three-model controlled benchmark on the same 12 COCO validation images, four target concepts, confidence threshold, and IoU ≥ 0.50 matching rule; reports precision, recall, F1, localization quality, latency, throughput, and model size. Temporal tracking (ByteTrack) is intentionally excluded — all three models are evaluated as image-level detectors here.

### SAM2
- `04_sam2_official_vs_huggingface_comparison.py` — Official Meta repository vs. Hugging Face `Sam2Model`/`Sam2Processor`, using identical point/box prompts derived from Oxford-IIIT Pet ground-truth masks, so both implementations receive exactly the same prompt for a fair comparison.

## Experimental Methodology

- **Ground truth:** COCO 2017 validation bounding-box annotations (detection notebooks) and Oxford-IIIT Pet pixel-level masks (SAM2), used only to score predictions after inference.
- **Metrics:** TP/FP/FN, precision, recall, F1-score, matched IoU (IoU ≥ 0.50), plus per-image inference latency, throughput, and model size where relevant.
- **Controlled variables:** identical images, target concepts, confidence thresholds, and evaluation code held fixed within each notebook; only the model or implementation interface varies.
- **Scope:** small, fixed evaluation sets (e.g. 12 COCO images) — explicitly framed as controlled implementation-level studies, not full COCO benchmarks.

## Results

**`01_owlvit_zero_shot_object_detection`**

| Query | Detections | Max Confidence | Mean Confidence |
|---|---|---|---|
| "a photo of a cat" | 2 | 0.716876 | 0.712163 |
| "a remote control" | 2 | 0.257253 | 0.244701 |

OWL-ViT is confidently correct on the primary "cat" query; the "remote control" query returns detections but at much lower confidence (~0.24–0.26), illustrating the threshold-sensitivity concern the notebook is designed to probe.

**`02_groundingdino_official_vs_huggingface_comparison` — Official vs. Hugging Face, across thresholds**

| Threshold | Detections | Precision | Recall | F1 | Mean Matched IoU | Implementation |
|---|---|---|---|---|---|---|
| 0.20 | 33 | 0.6875 | 0.7333 | 0.7097 | 0.8874 | Official |
| 0.30 | 14 | 0.7857 | 0.7333 | 0.7586 | 0.8874 | Official |
| 0.35 | 12 | 0.8333 | 0.6667 | 0.7407 | 0.8987 | Official |
| 0.45 | 10 | 0.9000 | 0.6000 | 0.7200 | 0.8987 | Official |
| 0.50 | 8 | 1.0000 | 0.5333 | 0.6957 | 0.9076 | Official |
| 0.20–0.50 | (same) | (same to 4 dp) | (same) | (same) | ~same | Hugging Face |

At every threshold tested, Official-repository and Hugging Face precision/recall/F1 match to 2–4 decimal places, with only trace-level IoU differences (≤0.0002) — the two implementations are practically equivalent for GroundingDINO on this evaluation set. As expected, raising the threshold trades recall for precision (33→8 detections, recall 0.73→0.53, precision 0.69→1.00).

**`03_yoloe_owlv2_groundingdino_controlled_benchmark` — three-model comparison (12 COCO images, IoU ≥ 0.50)**

| Model | Precision | Recall | F1 | Mean IoU | Mean Latency (ms) | Images/s | Params (M) |
|---|---|---|---|---|---|---|---|
| YOLOE | 1.000 | 0.600 | 0.750 | 0.9206 | 777.3 | 1.29 | 35.4 |
| OWLv2 | 1.000 | 0.733 | 0.846 | 0.8873 | 740.9 | 1.35 | 154.9 |
| GroundingDINO | 0.833 | 0.667 | 0.741 | 0.9010 | 744.8 | 1.34 | 172.2 |

All three achieve perfect-or-near-perfect precision on this set; OWLv2 gets the best F1 (0.846) via higher recall, YOLOE is the smallest and has the tightest localization (highest IoU among the two perfect-precision models), and GroundingDINO — the largest model here — has the lowest precision of the three.

**`04_sam2_official_vs_huggingface_comparison` — Official vs. Hugging Face (Oxford-IIIT Pet)**

| Prompt | Implementation | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|
| point | Official | 0.9944 | 0.9972 | 0.9978 | 0.9966 |
| point | Hugging Face | 0.9948 | 0.9974 | 0.9979 | 0.9969 |
| box | Official | 0.9937 | 0.9968 | 0.9992 | 0.9945 |
| box | Hugging Face | 0.9931 | 0.9965 | 0.9988 | 0.9943 |

Both implementations exceed 0.99 on every metric for both prompt types — Official and Hugging Face SAM2.1 Hiera Large are effectively equivalent on this evaluation set, with differences at or below the third decimal place.

## Reproducibility

- **Environment:** Google Colab (GPU recommended for GroundingDINO/SAM2/YOLOE).
- **Requirements:** model-specific — official repositories for GroundingDINO/SAM2 (cloned within the notebook) plus Hugging Face `transformers`, `pycocotools` for COCO annotation handling, Pillow/matplotlib/pandas.
- **Data:** COCO 2017 validation images/annotations and the Oxford-IIIT Pet dataset are fetched by each notebook as needed; no dataset files are redistributed in this repository.
- **Execution:** run top-to-bottom; dependency installs are in each notebook's first cell(s).

## Research Evidence

Each experiment links directly to its canonical script above. The DETR quantization experiment originally considered for this domain lives in [`efficient-ai-model-deployment`](../efficient-ai-model-deployment) instead, since its primary research question is quantization cost, not open-vocabulary detection.

## Related Research

- [`vision-language-model-research`](../vision-language-model-research) — VLMs perform open-ended grounding more generally; this repository's models are dedicated detection/segmentation specialists.
- [`efficient-ai-model-deployment`](../efficient-ai-model-deployment) — DETR quantization experiment, methodologically related to this repository's detection focus.
- Robotics/Physical AI (planned) — open-vocabulary detection and segmentation are direct inputs to robot perception and manipulation.

## Research Direction

This work is aimed at determining which open-vocabulary detectors are trustworthy enough — in accuracy and implementation consistency — to serve as a perception front-end for downstream robotic grounding and manipulation tasks.

## Limitations

- All controlled benchmarks use small, fixed image sets (as few as 12 images); results describe behavior on those sets, not general-purpose COCO performance.
- Official-vs-Hugging-Face comparisons can be affected by preprocessing/pre- and post-processing differences, not architecture alone — called out explicitly in each notebook.
- No video or multi-frame tracking evaluation in this repository (ByteTrack is explicitly excluded here; see `video-multimodal-perception` for tracking-inclusive work).

## Future Work

- Extend the controlled YOLOE/OWLv2/GroundingDINO benchmark to the full COCO validation set.
- Add ByteTrack-based multi-frame tracking evaluation as a separate, explicit experiment.
- Apply the same official-vs-Hugging-Face consistency protocol to OWLv2 and YOLOE.

## Research Takeaway

Official-repository and Hugging Face implementations proved practically equivalent for both GroundingDINO (precision/recall/F1 matching to 2–4 decimal places across five thresholds) and SAM2 (all metrics within 0.001 of each other) — the choice of inference stack is not a meaningful accuracy risk for these two models. Across the three-model controlled benchmark, no single detector dominated: YOLOE gave the tightest localization at the smallest parameter count, OWLv2 gave the best overall F1, and GroundingDINO — the largest model — had the lowest precision of the three, underscoring that parameter count alone does not predict zero-shot detection quality.
