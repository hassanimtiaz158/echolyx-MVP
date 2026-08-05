# Technical Design Document (TDD)
## Echolyx AI — Fan Anomaly Detection MVP

**Version:** 1.0
**Companion to:** PRD.md
**Owner:** Malik (Hasan Ali)

---

## 1. System Overview

```
Audio file (upload/mic)
        │
        ▼
Preprocessing (resample → 32kHz, mono, pad/trim to fixed length)
        │
        ▼
PANNs CNN14 backbone (pretrained on AudioSet, frozen → partially fine-tuned)
        │
        ▼
Classifier head (Linear 2048 → 2)
        │
        ▼
Softmax → { "Normal": p0, "Faulty": p1 }
        │
        ▼
Gradio UI (label + confidence)
```

## 2. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Team's existing stack (FastAPI, ML familiarity) |
| Audio I/O | librosa, soundfile | Robust loading/resampling across mp3/wav |
| Audio aug | pydub, numpy | Lightweight augmentation |
| ML framework | PyTorch | Native PANNs support, flexible fine-tuning |
| Pretrained backbone | PANNs CNN14 (`audioset_tagging_cnn`) | Strong AudioSet-pretrained embeddings; good for non-speech/mechanical sounds |
| Classical baseline (optional) | scikit-learn | Fast sanity-check baseline (RF/SVM on spectral features) |
| Experiment tracking (optional) | Weights & Biases | Clean charts for investor pitch |
| Compute | Kaggle Notebook (free GPU, T4) | Sufficient for MIMII fan subset scale, avoids local GPU need |
| Demo/UI | Gradio | Fast interactive web demo, shareable link |
| Backend (post-MVP) | FastAPI + Uvicorn | If a persistent API is needed beyond notebook demo |
| Containerization (post-MVP) | Docker | Reproducible deployment once beyond notebook stage |

## 3. Data Pipeline

### 3.1 Sources & Labeling
- **MIMII fan subset — MIMII DUE format**: `fan/{train,test}/section_XX_{source,target}_{train,test}_{normal,anomaly}_NNNN_spd_M.wav`. There are no `normal/`/`abnormal/` folders; the label is derived from the **filename** via regex (`_normal_` → 0, `_anomaly_` → 1).
- **DUE train/test split is intentionally re-pooled (not used as-is)**: MIMII DUE's `train`/`test` partition was designed for the DCASE domain-shift challenge task (train = normal-only simulation, test = held-out domain-shift evaluation) and is not meaningful for a conventional binary classifier. We re-pool all 1,200 clips (1,100 Normal + 100 Anomaly) and feed them into our own stratified 70/15/15 split, because a working binary classifier requires faulty examples in the training set.
- **Freesound clips (41 per PRD; 40 present in the provided zip)**: all treated as additional `Normal` (label 0), except `fan's frame broken.mp3`, which is excluded from training and reserved as a real-world validation sample.

### 3.2 Preprocessing
- Resample all audio to **32kHz mono** (PANNs' expected input rate).
- Fix clip length to **10 seconds** (matches MIMII clip length); pad short clips with zeros, trim long clips.
- No spectrogram precomputation needed at the dataset level — PANNs' `Cnn14` computes its own log-mel spectrogram internally from raw waveform input.

### 3.3 Splitting
- Stratified split: 70% train / 15% validation / 15% test, stratified by label to preserve class balance in each split.
- Fixed random seed (42) for reproducibility.

### 3.4 Class Imbalance Handling
- MIMII anomalous clips are a minority relative to normal clips (further skewed by folding in 40 extra normal Freesound clips).
- Mitigation 1: **class-weighted cross-entropy loss**, weights computed as `total / (2 * class_count)` per class.
- Mitigation 2: **balanced sampling** (`balance_sampling: true`, default ON): the train loader draws with replacement using inverse-class-frequency weights so the Faulty minority is present every epoch instead of being swamped by Normal.

### 3.5 Augmentation (training set only)
- Additive Gaussian noise (p=0.3)
- Random gain scaling (p=0.3)
- **Mixup** (`mixup_alpha`, default `0` = OFF): implemented and smoke-verified, but empirically **disabled by default** — with the 12:1 imbalance it dilutes the scarce fault signal (fault clips become normal-soft labels ~99% of the time). Re-enable once fault data is plentiful.
- Rationale: light augmentation to reduce overfitting given limited real anomalous samples; avoided anything that could distort the acoustic signature of the fault itself (e.g., no pitch-shifting).

## 4. Model Design

### 4.1 Backbone: PANNs CNN14
- Pretrained checkpoint: `Cnn14_mAP=0.431.pth` (trained on full AudioSet, 527 classes).
- Architecture: 6 convolutional blocks → global pooling → 2048-dim embedding → (original 527-way classifier, replaced).
- Chosen over YAMNet: PANNs trained on the broader AudioSet ontology including mechanical/motor sounds, larger capacity, PyTorch-native (fits existing stack), and more precedent in machine-sound-anomaly literature.

### 4.2 Transfer Head
- Replace original AudioSet classifier with `nn.Linear(2048, 2)` → Normal/Faulty.

### 4.3 Two-Phase Fine-Tuning
- **Phase 1** (backbone frozen): train only the new linear head. Adam, lr=1e-3, ~8 epochs. Lets the head adapt without disturbing pretrained features.
- **Phase 2** (partial unfreeze): unfreeze last conv block (`conv_block6`) + `fc1`. Adam with differential learning rates (backbone: 1e-5, head: 1e-4), ~15 epochs. Refines high-level features toward fan-specific acoustic patterns while preserving general audio representations in earlier layers.
- **Data-rich variant (current config)**: once the fault pool grows to ~400 clips (MIMII DUE + MIMII DG merged), Phase 2 runs `phase2_unfreeze_blocks: 2` (conv_block5+6+fc1) for `phase2_epochs: 25`. More fault examples make deeper fine-tuning stable, and the extra epochs give the backbone time to adapt to fault signatures. Both knobs stay config-driven (`phase2_unfreeze_blocks`, `phase2_epochs`); the TDD baseline (1 block, 15 epochs) remains the default for small data.
- Best checkpoint selected by validation accuracy, saved each time it improves.

### 4.4 Loss & Optimization
- CrossEntropyLoss with class weights (see 3.4).
- Optimizer: Adam (standard, well-behaved for fine-tuning at these learning rates).

## 5. Evaluation

- **Metrics:** accuracy, precision, recall, F1 (binary, faulty=positive class), confusion matrix, and the **predicted-label distribution** (explicit majority-class-collapse check).
- **Priority metric:** recall on the Faulty class — in a failure-prediction product, a missed fault (false negative) is more costly than a false alarm.
- **Threshold tuning:** the P(Faulty) cutoff maximizing **validation** F1 is selected, then the test split is re-scored at that cutoff and reported separately (fights the argmax collapse that class imbalance induces even with weighted loss).
- **Real-world sanity check:** run inference on the one real, non-training `fan's frame broken.mp3` clip; report result explicitly and honestly (whether correct or not) as the most credible single data point for investors.
- **Real-world hold-out set:** every clip under `broken_fan_root` (real faulty consumer-fan recordings, never in training/test splits) is scored individually and reported as a per-clip table + detection count, kept strictly separate from aggregate metrics.
- **Training curves:** loss/accuracy plots across both phases, saved as artifacts for the pitch deck.

## 6. Inference / Serving

- `predict_single_file(path)`: loads audio → preprocesses → runs through model → returns label + softmax probabilities.
- Wrapped in a Gradio `Interface`: accepts upload or microphone input, outputs a `Label` component with both class probabilities.
- Kaggle: `demo.launch(share=True)` for a temporary public link (session-bound). Post-MVP: push model + Gradio app to a Hugging Face Space for a permanent link.

## 7. Environment & Reproducibility

- Kaggle Notebook, GPU (T4) accelerator, Internet enabled (to clone PANNs repo + fetch checkpoint from Zenodo).
- Fixed seeds (Python `random`, NumPy, PyTorch) = 42.
- All paths configurable via a single Config cell (`MIMII_ROOT`, `EXTRA_NORMAL_ROOT`, sample rate, clip length, batch size).
- Artifacts persisted under `/kaggle/working/`: best model checkpoint, final model artifact (with metadata: sample rate, clip length, class names), training curve plots, confusion matrix plot.

## 8. Known Limitations (to state explicitly, not hide)

- Faulty-class data comes primarily from MIMII's specific fan units/recording conditions; may not generalize to arbitrary real-world fans or other machinery.
- Real-world faulty clips from Freesound (single sanity clip + 11-clip hold-out set) are spot-checks, not a statistically meaningful validation set.
- No fault localization/type classification (only binary Normal/Faulty).
- No real-time/streaming capability — single fixed-length clip classification only.
- Small compute budget (free-tier GPU) constrains model size/epochs; production accuracy would require more data, more compute, and likely a larger backbone or ensemble.

## 9. Post-MVP Roadmap (not in current scope, for context only)

- Expand to DCASE Task 2 / MIMII DUE for domain-shift robustness.
- Add additional machine types (pumps, valves, motors) per Echolyx AI's broader vision.
- Move from single-clip classification to continuous/streaming monitoring.
- Explore unsupervised/self-supervised anomaly detection (autoencoder reconstruction error) to reduce dependence on labeled anomalous data, since real-world faulty samples will always be the scarce class.
- Production deployment: FastAPI + Docker, proper MLOps (versioning, monitoring, retraining pipeline).
