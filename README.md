# EchoPrime Transfer Learning for Echocardiographic Function Prediction

A focused, reproducible research engineering project demonstrating transfer learning with a pretrained echocardiography encoder on public cardiac ultrasound data.

## Goal

Build a clean benchmark around **EchoPrime + EchoNet-Dynamic** without introducing or exposing any unpublished methodological ideas.

Initial experiments:

1. reproduce a pretrained EchoPrime encoder smoke test;
2. extract fixed video embeddings;
3. train an EF regression head on frozen embeddings;
4. partially fine-tune only the last encoder stage;
5. compare models with MAE, RMSE, and R²;
6. optionally concatenate simple structured metadata as a standard multimodal baseline.

This repository is a **skills demonstration**, not a claim of a new clinical method.

## Why this project

It demonstrates practical ability in echocardiography video processing, pretrained medical foundation models, PyTorch transfer learning, reproducible experiments, regression evaluation, testing, and CI.

## Data

Use **EchoNet-Dynamic** from Stanford AIMI:
https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound

The dataset contains >10,000 apical four-chamber echocardiography videos with labels including ejection fraction (EF), end-systolic volume (ESV), and end-diastolic volume (EDV).

**Do not commit the dataset to GitHub.** Access is subject to Stanford AIMI's applicable data-use terms.

Expected local layout:

```text
data/
└── EchoNet-Dynamic/
    ├── FileList.csv
    ├── VolumeTracings.csv
    └── Videos/
```

## EchoPrime

Official repository: https://github.com/echonet/EchoPrime

Clone it separately:

```bash
git clone https://github.com/echonet/EchoPrime external/EchoPrime
```

Then download the official model data following the EchoPrime README.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Validate the EchoNet-Dynamic manifest:

```bash
python scripts/check_dataset.py --root data/EchoNet-Dynamic
```

## First milestone

- [ ] dataset access confirmed
- [ ] manifest parsed and train/val/test counts checked
- [ ] EchoPrime official checkpoint downloaded
- [ ] encoder smoke test passes
- [ ] one real video can be transformed into a 512-d embedding
- [ ] unit tests pass

Only after this milestone do we implement the frozen-feature benchmark.

## Evaluation

Primary EF regression metrics: MAE, RMSE, and R².

## Scope boundary

This public repository intentionally does **not** include unpublished work on novel trustworthy-AI methods, new Bayesian uncertainty methodology, new longitudinal clinical-risk modelling, unpublished multimodal fusion architectures, or future postdoctoral research hypotheses.

## Attribution

EchoPrime: M. Vukadinovic et al., *EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation.*

EchoNet-Dynamic: D. Ouyang et al., *Video-based AI for beat-to-beat assessment of cardiac function.*

Please follow the official model and dataset licenses/terms.
