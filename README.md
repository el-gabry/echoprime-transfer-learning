cat > README.md <<'EOF'
# EchoPrime Transfer Learning for Echocardiographic Function Prediction

A reproducible proof-of-concept for adapting pretrained echocardiography representations to cardiac function estimation from ultrasound video.

## Overview

This project evaluates whether frozen representations from **EchoPrime**, a pretrained echocardiography vision-language model, can support prediction of left ventricular ejection fraction (EF) on **EchoNet-Dynamic**.

The goal is to demonstrate an end-to-end medical imaging ML workflow:

```text
EchoNet-Dynamic AVI
        ↓
EchoPrime-compatible preprocessing
        ↓
Pretrained EchoPrime video encoder
        ↓
512-dimensional representation
        ↓
Regularized regression
        ↓
Ejection Fraction