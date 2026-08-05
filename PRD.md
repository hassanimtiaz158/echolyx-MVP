# Product Requirements Document (PRD)
## Echolyx AI — Fan Anomaly Detection MVP

**Version:** 1.0
**Status:** Draft for build
**Owner:** Malik (Hasan Ali)

---

## 1. Background & Vision

Echolyx AI's long-term mission is predictive maintenance for heavy machinery: listening to industrial equipment and detecting anomalous sounds before failure occurs, to prevent costly downtime and damage.

This MVP is the **proof of concept** for that mission, scoped down to the smallest demonstrable version of the core idea:

> Audio input → AI analyzes the sound → outputs **"Normal"** or **"Faulty"**

The target machine for the MVP is a **fan** (industrial/household), chosen because:
- Labeled real-world data exists (MIMII dataset: normal + anomalous industrial fan sounds).
- Fan failure modes (unbalance, clogging, bearing wear) produce clearly audible acoustic signatures.
- It generalizes conceptually to other rotating machinery (pumps, motors, compressors) for future R&D.

## 2. Problem Statement

Unplanned machinery failure is expensive and hard to predict. Maintenance today is largely scheduled or reactive, not condition-based. Sound is an underused, low-cost signal that often reveals developing faults (rattling, grinding, unbalance) before a failure becomes catastrophic. No accessible, simple tool currently lets a non-expert point a microphone at a fan and get an immediate "healthy vs. faulty" read.

## 3. Goals

### 3.1 MVP Goals (this phase)
- G1: Build a working binary audio classifier: **Normal** vs. **Faulty** fan sound.
- G2: Use real labeled data (MIMII fan subset) as the primary source of truth, supplemented with public normal-fan recordings for class diversity.
- G3: Achieve a defensible, honestly-reported accuracy/F1 on a held-out test set.
- G4: Package the model behind a live, interactive demo (Gradio) usable in an investor pitch — upload or record audio, get an instant prediction with confidence.
- G5: Produce clear documentation of methodology and limitations, suitable for technical due diligence by investors.

### 3.2 Non-Goals (explicitly out of scope for MVP)
- Not a production-grade or commercially deployable system.
- Not multi-machine (only fans, not pumps/valves/motors/etc. — that's future R&D).
- Not real-time/streaming/continuous monitoring — single-clip classification only.
- Not fault *localization/diagnosis* (e.g., "bearing wear" vs. "unbalance") — binary Normal/Faulty only.
- No edge-device or on-premise hardware deployment.

## 4. Target Users (for this proof of concept)

- **Primary:** Malik / Echolyx AI founding team — to validate the concept internally.
- **Secondary:** Potential investors — as a live demo during fundraising conversations.
- **Not yet targeted:** actual industrial end-customers (post-funding, real R&D phase).

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | System accepts an audio clip (upload or microphone recording) as input. |
| FR2 | System preprocesses audio to a fixed sample rate/length suitable for the model. |
| FR3 | System classifies the clip as **Normal** or **Faulty**, with a confidence score for each class. |
| FR4 | System is trained using transfer learning on a pretrained general-audio model (PANNs CNN14), fine-tuned on fan-specific labeled data. |
| FR5 | Training data combines the MIMII fan subset (real normal + real anomalous fan recordings) with existing Freesound fan clips (as additional "Normal" diversity). |
| FR6 | System reports evaluation metrics (accuracy, precision, recall, F1, confusion matrix) on a held-out test split. |
| FR7 | System is validated against at least one real-world, non-training "faulty" audio sample as a sanity check. |
| FR8 | A shareable interactive demo (Gradio) is available for live investor presentation. |

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Trainable within free-tier compute constraints (Kaggle/Colab free GPU — T4, ~12–16GB VRAM, session time limits). |
| NFR2 | Inference latency low enough for a live demo (a few seconds per clip). |
| NFR3 | Reproducible: fixed random seeds, documented data split, saved model checkpoint. |
| NFR4 | Codebase organized clearly enough to hand off to Claude Code for implementation and future iteration. |
| NFR5 | All data provenance and limitations honestly documented (no overstated claims to investors). |

## 7. Data Sources

1. **MIMII Dataset (fan subset)** — real labeled normal/abnormal industrial fan sounds. Primary source of truth for the Faulty class.
2. **Freesound.org fan clips (41 files, ~32 min, already collected)** — real recordings of various fan types (desk, ceiling, box, exhaust, industrial, PC). Used to diversify the Normal class. Contains exactly one real faulty example (`fan's frame broken.mp3`), reserved as a held-out sanity-check sample rather than training data.
3. *(Optional, future)* DCASE Task 2 / MIMII DUE datasets — for domain-shift robustness beyond MVP scope.

## 8. Success Metrics

- **Primary:** Test-set accuracy and F1-score meaningfully above chance (>0.80 F1 as a reasonable MVP bar, to be validated empirically — not guaranteed given data constraints).
- **Qualitative:** Live demo correctly classifies a real-world clip in front of an audience without failure/crash.
- **Investor-readiness:** A one-page methodology + limitations summary that can survive technical scrutiny.

## 9. Risks & Constraints

| Risk | Mitigation |
|------|------------|
| Real faulty data is scarce/imbalanced (MIMII anomalies are a minority class) | Class-weighted loss; stratified splits; report recall on faulty class explicitly (false negatives matter most for a failure-prediction pitch). |
| Model overfits to MIMII's specific recording conditions/background noise, not generalizable to other fans | Mix in Freesound normal-class diversity; be explicit about this limitation in the write-up; treat generalization as future R&D work, not an MVP claim. |
| Free-tier compute (Colab/Kaggle) session limits interrupt training | Checkpointing every epoch; cache extracted features; design for resumability. |
| Overstating MVP capability to investors | Explicit "known limitations" section in every deliverable; avoid claiming production-readiness or multi-machine generalization. |

## 10. Milestones

1. Data audit + collection (MIMII fan subset acquired and organized) — **done/in progress**
2. Data pipeline (loading, labeling, preprocessing, splitting) — see TDD
3. Model implementation (PANNs CNN14 transfer learning, two-phase fine-tuning) — see TDD
4. Training + evaluation on Kaggle (GPU) — see TDD
5. Sanity-check against real faulty sample
6. Gradio demo wired to trained model
7. Documentation: PRD (this doc), TDD, methodology/limitations summary for investors

## 11. Open Questions

- Final target F1/accuracy threshold to consider the MVP "demo-ready" — to be set after first training run's actual numbers are seen (not being set arbitrarily in advance).
- Whether to include a small set of non-fan background noise clips as a third "irrelevant/no-fan" class in a later iteration (currently out of scope).
