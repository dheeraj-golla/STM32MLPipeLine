# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Experimental lab repo for building Gridware's ML ODA (on-device algorithm) deployment pipeline: train models in Python, convert them to C via **ST Edge AI Studio** (ST's desktop tool wrapping ST Edge AI Core / X-CUBE-AI), and validate on the **B-L4S5I-IOT01A** discovery board (STM32L4S5VI, Cortex-M4F). It is a proving ground, not a product — the production integration target is the separate `gridware-bytes` firmware repo (locally at `~/gridware-bytes`).

The documentation in `docs/` is the primary deliverable. Files are numbered chronologically (`00_`–`07_`). **`docs/07_ml_oda_pipeline_summary.md` is the living status doc — read it first for current state, the eight framework requirements, and next steps; update it when work advances.**

## Commands

Python side (Poetry-managed, Python pinned to 3.12 for TensorFlow 2.21 wheel compatibility — do not bump either without reading the comment in `pyproject.toml`):

```bash
poetry install
poetry run python example_keras_classifier/train_model.py   # retrain reference model + test vectors
poetry run python example_keras_classifier/resave_model.py  # re-export as .h5/.tflite for AI Studio
```

There are no tests or linters configured.

STM32 side: `projects/*` are STM32CubeIDE projects (`.ioc` files are CubeMX configs). Building and flashing happens in the STM32CubeIDE GUI — there is no CLI build in this repo.

## Conventions that matter

- **Determinism for bit-exactness:** every script sets `TF_ENABLE_ONEDNN_OPTS=0` *before* importing TensorFlow, and fixes all RNG seeds. Python-side outputs are compared bit-for-bit against C output on the MCU; oneDNN float reordering silently breaks this. Preserve this in any new script.
- **Model handoff format:** ST AI Studio's bundled Keras is older than the Keras that trains the models — native `.keras` files from Keras 3.3+ fail to load (unrecognized `quantization_config`). Hand models to AI Studio as `.tflite` (preferred) or legacy `.h5`, produced by `resave_model.py`.

## Layout

- `example_keras_classifier/` — reference model: Dense(1, sigmoid), 6 inputs, 7 parameters, deliberately trivial (validates the pipeline, not the ML). `outputs/` holds the trained model in all formats plus `test_vectors.json` (254 input/output pairs in three categories for downstream bit-exactness checks).
- `projects/keras_classifier_simple_test/` — the AI Studio–generated CubeIDE project for that model. `AI/App/network*.{c,h}` are the generated inference graph + weights (regenerated per model); `Middlewares/ST/AI/` holds ST's precompiled ARM-only NetworkRuntime static library + headers (shared across models). `docs/07` §"Which files migrate" categorizes exactly which of these belong in a real firmware repo.
- `projects/blink_led/` — board bring-up sanity project.
- `ml_models/zdlb/` — a real Gridware model (ZDLB XGBoost, hand-written C++ header) kept for reference/analysis.
