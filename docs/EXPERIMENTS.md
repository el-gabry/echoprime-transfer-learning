# Experiment protocol

## E1 — Manifest audit
Record train/validation/test counts, missing videos, EF distribution, and duplicate identifiers.

## E2 — EchoPrime smoke test
Verify that the official pretrained video encoder loads and returns the expected embedding dimension.

## E3 — Frozen encoder
Freeze EchoPrime, precompute embeddings, fit a regression head on train, select hyperparameters on validation, and report final test MAE/RMSE/R² once.

## E4 — Partial fine-tuning
Unfreeze a small final portion of the encoder, use a lower encoder learning rate, select by validation MAE, and evaluate once on test.

## Reproducibility
For every experiment record random seed, Git commit, Python/PyTorch versions, GPU, config, checkpoint, runtime, and trainable parameter count.

## Reporting rule
Negative or neutral results are still results. Do not selectively report only improving runs.
