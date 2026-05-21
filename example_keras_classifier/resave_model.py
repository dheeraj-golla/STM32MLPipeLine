"""
Workaround for AI Studio Keras version skew.

The .keras file saved by newer Keras (3.3+) includes a `quantization_config`
field on Dense layers that ST Edge AI Studio's bundled (older) Keras doesn't
recognize. This script reloads the model and re-saves it in formats that
AI Studio's loader can handle.

Outputs:
  - model.h5      : legacy Keras H5 format (no quantization_config)
  - model.tflite  : TFLite flatbuffer (most stable handoff format)

Run from the same Python environment that trained the model.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

SRC = "outputs/model.keras"
H5_DST = "outputs/model.h5"
TFLITE_DST = "outputs/model.tflite"

print(f"Loading {SRC} ...")
model = keras.models.load_model(SRC)
model.summary()

# 1) Legacy H5. Older format, no quantization_config field.
print(f"\nSaving H5 to {H5_DST} ...")
model.save(H5_DST)
print(f"  size: {os.path.getsize(H5_DST)} bytes")

# 2) TFLite flatbuffer. Stable, versioned schema. The right long-term handoff.
print(f"\nSaving TFLite to {TFLITE_DST} ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No optimizations -> float32 weights, no quantization. Matches our pass-1 intent.
tflite_model = converter.convert()
with open(TFLITE_DST, "wb") as f:
    f.write(tflite_model)
print(f"  size: {os.path.getsize(TFLITE_DST)} bytes")

print("\nDone. Try AI Studio with model.tflite first (preferred), fall back to model.h5.")
