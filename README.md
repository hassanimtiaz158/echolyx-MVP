# Echolyx AI — Fan Anomaly Detection MVP

## What this is

This is the Echolyx AI **proof of concept** for its predictive-maintenance
mission: listening to industrial equipment and catching developing faults
before they become failures. The problem we're attacking is that unplanned
machinery failure is expensive and hard to predict, while sound is an
underused, low-cost signal that often reveals trouble (rattling, grinding,
unbalance) early on — yet no simple tool lets a non-expert point a
microphone at a machine and get an immediate "healthy vs. faulty" read
(see [PRD.md](PRD.md), sections 1–2, for the full background). This repo
implements the smallest demonstrable version of that idea: a binary audio
classifier that labels fan recordings as **Normal** or **Faulty**, built
with PANNs CNN14 transfer learning per the [TDD.md](TDD.md) technical design.
It is an MVP for internal validation and investor demos — **not** a
production monitoring system (see Known limitations below).

## Architecture (per TDD)

```
Audio file (upload/mic)
        |
        v
Preprocessing (resample -> 32 kHz, mono, pad/trim to 10 s)
        |
        v
PANNs CNN14 backbone (AudioSet-pretrained, frozen -> partially fine-tuned)
        |
        v
Classifier head (Linear 2048 -> 2)  ->  softmax -> {Normal, Faulty}
        |
        v
Gradio UI (label + confidence)
```

Two-phase fine-tuning (TDD 4.3): **Phase 1** trains only the new head with a
frozen backbone (Adam, lr=1e-3, 8 epochs); **Phase 2** unfreezes
`conv_block6` + `fc1` with differential learning rates (backbone 1e-5, head
1e-4, 15 epochs). Loss is class-weighted cross-entropy
(`total / (2 * class_count)`), splits are stratified 70/15/15, seed 42.

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell); macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data

**MIMII fan subset** — real labeled normal/abnormal industrial fan sounds,
the primary source of truth for the Faulty class. Download the fan archive
from the MIMII dataset release (Zenodo record 3384388) and unzip it so the
folder layout matches what the pipeline expects:

```
data/raw/
├── mimii/
│   └── fan/
│       ├── id_00/
│       │   ├── normal/          # .wav clips -> label 0 (Normal)
│       │   └── abnormal/        # .wav clips -> label 1 (Faulty)
│       ├── id_01/
│       │   ├── normal/
│       │   └── abnormal/
│       └── id_02/  ... id_04/
└── freesound/
    ├── desk_fan.mp3             # any .mp3/.wav fan clips -> label 0 (Normal)
    ├── ceiling_fan.wav
    └── fan's frame broken.mp3   # HELD OUT — see "Sanity check" below
```

**Freesound fan clips** — ~40 real recordings of various fan types
(desk, ceiling, box, exhaust, industrial, PC) collected to diversify the
Normal class. Drop them into `data/raw/freesound/`. The single real faulty
recording, `fan's frame broken.mp3`, is **excluded from training and
evaluation splits automatically** (anything with "broken" in the name) and
reserved as the held-out real-world sanity-check sample.

> `data/raw/` and `data/processed/` are gitignored — the data is large and
> licensed; never commit it.

### 3. Download the PANNs backbone checkpoint

```bash
python -m src.models.download_checkpoint --dest checkpoints
# fetches Cnn14_mAP=0.431.pth (~308 MB, AudioSet-pretrained) from Zenodo
```

## How to run

All commands run from the repo root. Every stage reads
`configs/config.yaml` (no hard-coded paths — on Kaggle/Colab, point the
roots at `/kaggle/input` and `/kaggle/working`).

```bash
# 0. Audit the data — build the labeled manifest and print class counts
python -m src.data.collect

# 1. Train (two-phase fine-tuning, ~8 + ~15 epochs; GPU recommended)
python -m src.train --config configs/config.yaml

# 2. Evaluate — test-split metrics, plots, and the held-out sanity check
python -m src.evaluate --config configs/config.yaml

# 3. Demo — interactive Gradio UI (upload or microphone)
python app/gradio_app.py
python app/gradio_app.py --share   # temporary public link (notebook/demo only)
```

Smoke-test the whole pipeline on tiny data with
`--phase1-epochs 1 --phase2-epochs 1` on `train`.

## Where results land

| Artifact | Location | Contents |
|---|---|---|
| Best model | `checkpoints/best.pt` | Best-by-validation-accuracy weights + metadata (sample rate, clip length, class names, phase/epoch) |
| Final model | `checkpoints/final.pt` | Last epoch after Phase 2 |
| Training curves | `artifacts/training_curves.png` | Train/val loss + accuracy across both phases |
| Confusion matrix | `artifacts/confusion_matrix.png` | Test-split confusion matrix |
| Training history | `artifacts/training_history.json` | Per-epoch loss/accuracy for both phases |
| Metrics | stdout of `evaluate` | Accuracy, precision, recall, F1 (Faulty = positive class; recall is the priority metric) + full `classification_report` + **REAL-WORLD SANITY CHECK** result on `fan's frame broken.mp3`, reported separately and never mixed into aggregate metrics |

## Tests

```bash
pytest -q
```

The suite runs entirely on synthetic data — no dataset or 300 MB checkpoint
download required. Model tests build a structurally valid fake checkpoint.

## Known limitations (read before quoting this project)

This is a proof of concept, and these limitations are stated explicitly
(adapted from TDD.md sec 8 — they are features of the MVP scope, not bugs):

- **Data generality**: the Faulty class comes primarily from MIMII's
  specific fan units and recording conditions; results may not generalize to
  arbitrary real-world fans or other machinery types.
- **Sanity check, not validation**: the single real "faulty" clip from
  Freesound is a spot-check for demo credibility, not a statistically
  meaningful validation set.
- **Binary only**: no fault localization or fault-type classification —
  just Normal/Faulty.
- **No real-time/streaming**: the model classifies single fixed-length
  (10 s) clips only.
- **Compute-constrained**: free-tier GPU budgets limit model size and
  epochs; production accuracy would require more data, more compute, and
  likely a larger backbone or ensemble.
- **Not production-ready**: no deployment, monitoring, or MLOps; the Gradio
  share link is session-bound, and the whole pipeline is a research
  prototype for an investor demo, not a commercial system.

See [PRD.md](PRD.md) (non-goals, risks, open questions) and
[TDD.md](TDD.md) (sec 8 limitations, sec 9 post-MVP roadmap) for the full
context.
