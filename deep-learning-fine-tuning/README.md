# Deep Learning / Fine-Tuning

Training-protocol fundamentals: transformer fine-tuning (full, native-loop, LoRA, QLoRA) and residual CNNs built from scratch, with an emphasis on methodological correctness (proper train/val/test separation, controlled comparisons) over task novelty.

## Research Focus

This is the portfolio's foundations repository: it demonstrates rigorous training and evaluation protocol — the discipline underlying the more novel work elsewhere — through fine-tuning BERT (via both the Hugging Face `Trainer` API and a native PyTorch loop), multi-adapter LoRA switching, GPT-2 QLoRA, and two custom residual CNNs implemented without `torchvision.models`.

## Research Questions

- Does a native PyTorch training loop, with explicit train/validation/held-out-test separation, reproduce the same MRPC performance as the Hugging Face `Trainer` API, and does using validation data for both checkpoint selection *and* final reporting produce an optimistically biased result?
- Can multiple LoRA adapters be trained on the same frozen base model, saved, reloaded, and switched at inference time without reloading the base model — and how do they compare quantitatively (loss, perplexity)?
- Does QLoRA fine-tuning measurably improve GPT-2's held-out language-modeling performance (loss, perplexity, next-token accuracy) on a domain corpus (IMDB, used as text rather than as a sentiment-classification task)?
- Can a from-scratch residual (ResNet-style) CNN, without pretrained weights, reach solid classification performance on FashionMNIST and a 5-class flower dataset while remaining fully interpretable at the architecture level?

## Work Included

### BERT / MRPC
- `01_bert_mrpc_trainer_finetuning.py` — fine-tunes `bert-base-uncased` on GLUE MRPC using the Hugging Face `Trainer` API, with dynamic padding, AdamW/warmup/weight-decay, mixed precision, and best-F1 checkpoint restoration.
- `02_bert_mrpc_native_pytorch_test_protocol.py` — the same task using a raw PyTorch training loop, with an explicit correction: the original MRPC train split is divided into train/validation, and the original validation split is reserved as a held-out test set never used for checkpoint selection — avoiding the optimistic bias of testing on the same data used to pick the best epoch.

### Parameter-Efficient Fine-Tuning
- `03_multi_adapter_lora_switching.py` — two LoRA adapters trained sequentially on a frozen `distilgpt2`, saved, reloaded, and swapped via `set_adapter`; both evaluated on the same held-out data via loss/perplexity, with a separate PEFT-hotswap correctness check.
- `04_gpt2_qlora_imdb_finetuning.py` — QLoRA fine-tuning of GPT-2 on IMDB as a causal-LM corpus, compared against the original GPT-2 baseline via loss, perplexity, next-token accuracy, and Distinct-1/Distinct-2 generation diversity, with adapter reload verified for reproducibility.

### Custom Residual CNNs
- `05_custom_residual_cnn_fashionmnist_classification.py` — hand-built residual blocks (identity + 1×1 projection shortcuts) across a four-stage CNN (32→64→128→256 channels), with gradient-flow verification and full training/evaluation implemented from scratch.
- `06_custom_residual_cnn_flowers_classification.py` — the same residual-block design applied to 5-class flower classification, with deterministic val/test preprocessing, manual dataset splitting, best-checkpoint selection, and final test-set precision/recall/F1/confusion-matrix reporting.

## Experimental Methodology

- **Protocol correctness is the primary experimental variable** in the BERT notebooks: the native-PyTorch version specifically corrects a train/validation/test leakage risk present in the naive setup.
- **Controlled comparisons:** LoRA adapters share the same base checkpoint, training data, preprocessing, and evaluation data — only the LoRA configuration differs.
- **Task framing discipline:** IMDB is explicitly treated as a causal-LM corpus, not a sentiment-classification dataset, so loss/perplexity/next-token-accuracy are the correct primary metrics, not classification accuracy.
- **From-scratch verification:** the custom CNNs include explicit gradient-flow and parameter-count inspection to confirm the residual mechanism is implemented and training correctly, not just producing plausible-looking loss curves.

## Results

**`01_bert_mrpc_trainer_finetuning` (Hugging Face `Trainer`)** — validation F1 rose from ~0.886 (epoch 1) to ~0.894 (epoch 4), validation accuracy from ~0.841 to ~0.853 over 4 epochs. Spot-checked on 8 held-out sentence pairs, the model was correct on all 8 with confidence ≥ 99.3% on every example — a strong, if small, qualitative sanity check alongside the aggregate validation curve.

**`02_bert_mrpc_native_pytorch_test_protocol` (native loop, corrected train/val/test split)**

| Best Epoch | Test Accuracy | Test Precision | Test Recall | Test F1 | Mean Test Confidence | Test Examples |
|---|---|---|---|---|---|---|
| 3 | 0.826 | 0.8333 | 0.9319 | 0.8799 | 0.978 | 408 |

Confusion matrix on the held-out test split: 77 true negatives, 52 false positives, 19 false negatives, 260 true positives. Recall (0.93) is notably higher than precision (0.83) — the model over-predicts "paraphrase," and this is the *actual* held-out result (never used for checkpoint selection), addressing exactly the leakage risk this notebook was designed to correct for.

**`03_multi_adapter_lora_switching`**

| Adapter | LoRA Rank | Target Modules | Params | Test Loss | Test Perplexity | Eval Time (s) |
|---|---|---|---|---|---|---|
| Adapter 1 | 8 | c_attn, c_proj, c_fc | 589,824 | 3.2831 | 26.6590 | 3.9245 |
| Adapter 2 | 8 | c_attn | 202,752 | 3.3332 | 28.0279 | 3.6168 |

Adapter 1 (targeting more modules, ~2.9× the trainable parameters) achieves lower perplexity than Adapter 2 on the same held-out evaluation subset — both adapters were trained, saved, reloaded, and switched successfully on the same frozen `distilgpt2` base.

**`05_custom_residual_cnn_fashionmnist_classification`** — best validation F1 of **0.8606** at epoch 3 (3-epoch run); a separate longer training run (30 epochs) reached ~95% training accuracy and ~94% test accuracy, with training and test loss both converging to roughly 0.12–0.17. Qualitative spot check: correctly classified a "Shirt" test image with 89.05% confidence, with the next-highest class ("T-shirt/top") at only 10.8%.

**`06_custom_residual_cnn_flowers_classification`**

| Metric | Value |
|---|---|
| Best Validation Accuracy | 0.8127 |
| Test Accuracy | 0.8385 |
| Test Loss | 0.5117 |
| Macro Precision | 0.8438 |
| Macro Recall | 0.8436 |
| Macro F1 | 0.8388 |
| Test Samples | 551 |

Per-class F1 ranges from 0.75 (roses, the weakest class — confused most often with tulips) to 0.92 (sunflowers, the strongest). The from-scratch residual CNN reaches ~84% accuracy on 5-class flower classification without any pretrained weights.

## Reproducibility

- **Environment:** Google Colab (GPU recommended, required for QLoRA/GPT-2).
- **Requirements:** `transformers`, `datasets`, `evaluate`, `peft`, `bitsandbytes`, `accelerate`, `sentencepiece`, PyTorch/`torchvision`.
- **Data:** GLUE MRPC and IMDB are loaded via Hugging Face `datasets`; FashionMNIST and the flowers dataset are notebook-specific (see each script's data-loading cell).
- **Execution:** run top-to-bottom; dependency installs are in each notebook's first cell(s).

## Research Evidence

Each experiment links directly to its canonical script above.

## Related Research

- [`efficient-ai-model-deployment`](../efficient-ai-model-deployment) — shares this repository's discipline of controlled, single-variable comparisons, applied to backbones/quantization instead of fine-tuning method.
- [`vision-language-model-research`](../vision-language-model-research) — the structured-evaluation and controlled-comparison protocol used here (BERT, LoRA) is applied there to VLM zero-shot evaluation.

## Research Direction

This repository is the methodological foundation for the rest of the portfolio: correct train/validation/test separation, controlled single-variable comparisons, and from-scratch architectural understanding are the practices applied throughout the depth, VLM, and detection research elsewhere in this portfolio.

## Limitations

- FashionMNIST and flower classification are comparatively small, well-studied datasets — not a novel-task contribution on their own; their value here is methodological (correct protocol, from-scratch architecture).
- LoRA/QLoRA experiments use small base models (`distilgpt2`, `gpt2`) rather than production-scale LLMs, for Colab feasibility.
- No hyperparameter-search or ablation study is included beyond the single controlled comparison each notebook is designed around.

## Future Work

- Extend the train/val/test correction methodology demonstrated in the native-PyTorch BERT notebook as a template checklist applied across the rest of the portfolio.
- Scale QLoRA fine-tuning to a larger base model to test whether the same conclusions hold.
- Apply the custom residual CNN as an additional "custom backbone" entry in the `efficient-ai-model-deployment` backbone-comparison study (it is already referenced there conceptually).

## Research Takeaway

The native-PyTorch BERT/MRPC notebook is this repository's clearest methodological result: correcting the train/validation/test split changed the reported held-out metrics (accuracy 0.826, F1 0.880) from what a naive validation-only protocol would have shown, demonstrating why the split-correction discipline matters beyond BERT specifically. Elsewhere, both LoRA adapters and both custom residual CNNs trained, evaluated, and reproduced successfully — the from-scratch CNNs in particular reach solid accuracy (94% on FashionMNIST, 84% on 5-class Flowers) without any pretrained weights, confirming the residual architecture is implemented and training correctly.
