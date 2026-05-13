# Differentiable State-Space Model

Differentiable 2R2C state-space model for building thermal system identification and simulation.

## Install

```bash
uv sync
```

## Run

```bash
uv run main.py
```

## Features

- Differentiable continuous-time 2R2C model
- Exact ZOH discretization
- Learnable physical RC parameters
- ADAM + LBFGS optimization

## Structure

```text
main.py                 # entrypoint
DifferentiableSSM.py    # differentiable SSM
solar_features.py       # solar gain weighting
train.py                # training pipeline
utils.py                # data + simulation utilities
```