# Inverse PINN

Inverse PINN for identifying building thermal dynamics with a dimensionless 2R2C model.

## Install

```bash
uv sync
```

## Run

```bash
uv run main.py
```

## Features

- PINN-based thermal modeling
- Learnable 2R2C parameters
- Fourier feature time encoding
- Random Weight Factorization (RWF)
- ADAM + LBFGS optimization

## Structure

```text
main.py              # entrypoint
PINN.py              # PINN implementation
MLP.py               # neural network
dimlessParams.py     # RC parameterization
fourierFeatures.py   # Fourier embeddings
RWF.py               # factorized linear layers
train.py             # training pipeline
evaluate.py          # evaluation + plots
utils.py             # data + simulation utilities
```