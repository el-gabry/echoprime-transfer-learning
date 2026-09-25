# Project plan

## Objective
Demonstrate readiness to work on modern echocardiography ML by building a reproducible transfer-learning benchmark using public data and a public pretrained model.

## Milestone 0 — Repository + environment
**Done when:** repository is clean; CI runs tests; dependencies install; no data or weights are committed.

## Milestone 1 — Data + encoder smoke test
- obtain EchoNet-Dynamic under official terms
- validate `FileList.csv`
- clone official EchoPrime
- download official model data
- verify pretrained video encoder
- process one real video end-to-end
- confirm embedding dimensionality

**Done when:** one real EchoNet-Dynamic video produces a valid EchoPrime embedding and preprocessing is documented.

## Milestone 2 — Frozen-feature EF baseline
- precompute embeddings for train/validation/test
- train a small regression head
- evaluate EF with MAE, RMSE, R²
- create predicted-vs-actual and residual plots
- record runtime and hardware

## Milestone 3 — Partial fine-tuning
- unfreeze a controlled final encoder stage
- compare with frozen features
- track trainable parameter count and GPU memory
- use early stopping and preserve official test split

## Optional Milestone 4 — Standard metadata fusion
Concatenate imaging representations with simple structured variables available under the dataset terms/manifest.

## Non-goals
- no novel trustworthy-AI method
- no unpublished Bayesian methodology
- no claim of clinical deployment
- no reproduction of the full MOSAIC program
- no private or restricted data
