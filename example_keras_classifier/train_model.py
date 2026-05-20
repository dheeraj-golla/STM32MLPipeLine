"""
Reference Keras model for the STM32 ML deployment pipeline.

Purpose: produce a small, fully-specified model and a set of test vectors
that can flow through the Python -> C/C++ conversion pipeline. The point
is to validate the *pipeline*, not to do interesting ML.

Architecture:
    Dense(1, activation='sigmoid'), 6 float32 inputs.
    7 trainable parameters total (6 weights + 1 bias).

Synthetic task:
    - x ~ Uniform[1, 10]^6
    - true label = 1 if sum(x) > 33 else 0   (33 is the midpoint of [6, 60])
    - 10% of labels flipped to add label noise (irreducible error ~10%)

Determinism:
    - All RNG seeds fixed.
    - oneDNN reordering disabled so we get bit-stable Python-side outputs
      we can compare against C output later.
"""

import os
# Must be set before TF import. Prevents oneDNN from reordering float ops,
# which is the kind of nondeterminism that makes Python<->C bit-exactness
# checks fail mysteriously.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet info logs

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---- Reproducibility -------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# ---- Synthetic dataset -----------------------------------------------------
N_FEATURES = 6
INPUT_MIN, INPUT_MAX = 1.0, 10.0
THRESHOLD = 33.0           # midpoint of sum range [6, 60]
NOISE_RATE = 0.10
N_TRAIN = 4000
N_VAL = 1000
N_TEST_RANDOM = 200        # random test vectors for the firmware team

def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(INPUT_MIN, INPUT_MAX, size=(n, N_FEATURES)).astype(np.float32)
    true_labels = (X.sum(axis=1) > THRESHOLD).astype(np.int32)
    # Flip NOISE_RATE of labels
    flip_mask = rng.uniform(size=n) < NOISE_RATE
    y = np.where(flip_mask, 1 - true_labels, true_labels).astype(np.float32)
    return X, y

X_train, y_train = make_dataset(N_TRAIN, seed=SEED)
X_val, y_val = make_dataset(N_VAL, seed=SEED + 1)

# ---- Model -----------------------------------------------------------------
# Explicit dtype on the input avoids surprises later. We want float32
# everywhere on the Python side; the quantization step happens downstream
# in the pipeline, not here.
model = keras.Sequential([
    keras.layers.Input(shape=(N_FEATURES,), dtype="float32", name="features"),
    keras.layers.Dense(1, activation="sigmoid", name="classifier"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---- Train -----------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    verbose=2,
)

# ---- Evaluate on a clean (noise-free) test set so we can see what the
# ---- model actually learned vs. what the noise prevents it from learning.
rng_clean = np.random.default_rng(SEED + 99)
X_clean = rng_clean.uniform(INPUT_MIN, INPUT_MAX, size=(2000, N_FEATURES)).astype(np.float32)
y_clean = (X_clean.sum(axis=1) > THRESHOLD).astype(np.float32)
clean_loss, clean_acc = model.evaluate(X_clean, y_clean, verbose=0)
print(f"\nNoise-free test accuracy: {clean_acc:.4f}  (ceiling, what the model truly learned)")
print(f"Noisy val accuracy:       {history.history['val_accuracy'][-1]:.4f}  (ceiling ~0.90 due to label flips)")

# ---- Extract weights for human inspection ---------------------------------
# A Dense(1) layer has:
#   W: shape (6, 1)
#   b: shape (1,)
# Output for a single sample x is: sigmoid(x @ W + b)
W, b = model.get_layer("classifier").get_weights()
print(f"\nWeights W (shape {W.shape}):")
print(W.flatten())
print(f"Bias b   (shape {b.shape}): {b}")

# ---- Save artifacts --------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) The model itself, in Keras v3 native format.
model_path = os.path.join(OUT_DIR, "model.keras")
model.save(model_path)
print(f"\nSaved model to {model_path}")

# 2) Human-readable weights (so you can hand-check the C code against this).
weights_json = {
    "architecture": "Dense(1, activation=sigmoid)",
    "input_shape": [N_FEATURES],
    "input_dtype": "float32",
    "weights": W.flatten().tolist(),       # length 6
    "bias": b.tolist(),                    # length 1
    "activation": "sigmoid",
    "formula": "y_prob = sigmoid(sum(x[i] * W[i] for i in 0..5) + b[0])",
    "classification_threshold": 0.5,
    "training": {
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "epochs": 30,
        "batch_size": 64,
        "optimizer": "adam(lr=0.01)",
        "loss": "binary_crossentropy",
    },
    "data": {
        "input_range": [INPUT_MIN, INPUT_MAX],
        "label_rule": f"1 if sum(x) > {THRESHOLD} else 0",
        "label_noise_rate": NOISE_RATE,
    },
    "metrics": {
        "noisy_val_accuracy": float(history.history["val_accuracy"][-1]),
        "noise_free_test_accuracy": float(clean_acc),
    },
}
weights_path = os.path.join(OUT_DIR, "model_weights.json")
with open(weights_path, "w") as f:
    json.dump(weights_json, f, indent=2)
print(f"Saved weights to {weights_path}")

# 3) Test vectors for the firmware team.
# Three categories:
#   (a) Random samples from the same distribution as training.
#   (b) Edge cases: inputs whose sum sits right around the threshold.
#       These are the inputs where quantization is most likely to flip
#       the predicted class. Critical for verifying #3 (verifiability).
#   (c) Boundary inputs: all-min, all-max, midpoint. Useful as a sanity
#       check during C porting.

def predict(x_batch):
    """Run model and return (logit_pre_activation, probability)."""
    # logit = x @ W + b
    logits = x_batch @ W + b
    probs = model.predict(x_batch, verbose=0).flatten()
    return logits.flatten(), probs

# (a) Random
rng_test = np.random.default_rng(SEED + 7)
X_random = rng_test.uniform(INPUT_MIN, INPUT_MAX, size=(N_TEST_RANDOM, N_FEATURES)).astype(np.float32)

# (b) Edge cases near the decision boundary. We construct inputs whose
# sum is in [32, 34] so the sigmoid output is close to 0.5.
n_edge = 50
edge_sums = rng_test.uniform(THRESHOLD - 1.0, THRESHOLD + 1.0, size=n_edge)
X_edge = []
for s in edge_sums:
    # Distribute the target sum across 6 features uniformly with jitter.
    base = s / N_FEATURES
    jitter = rng_test.uniform(-0.5, 0.5, size=N_FEATURES)
    jitter -= jitter.mean()  # keep sum exactly at s
    x = np.clip(base + jitter, INPUT_MIN, INPUT_MAX)
    X_edge.append(x)
X_edge = np.array(X_edge, dtype=np.float32)

# (c) Boundary inputs.
X_boundary = np.array([
    [INPUT_MIN] * N_FEATURES,                          # min input -> sum=6, expect 0
    [INPUT_MAX] * N_FEATURES,                          # max input -> sum=60, expect 1
    [(INPUT_MIN + INPUT_MAX) / 2] * N_FEATURES,        # midpoint  -> sum=33, expect ~0.5
    [INPUT_MIN, INPUT_MAX, INPUT_MIN, INPUT_MAX, INPUT_MIN, INPUT_MAX],  # alternating
], dtype=np.float32)

def vectors_for(X, category):
    logits, probs = predict(X)
    out = []
    for i in range(len(X)):
        out.append({
            "category": category,
            "input": X[i].tolist(),
            "input_sum": float(X[i].sum()),
            "logit_pre_sigmoid": float(logits[i]),
            "probability": float(probs[i]),
            "predicted_class": int(probs[i] >= 0.5),
        })
    return out

test_vectors = (
    vectors_for(X_random, "random")
    + vectors_for(X_edge, "edge_near_boundary")
    + vectors_for(X_boundary, "boundary_extreme")
)
tv_path = os.path.join(OUT_DIR, "test_vectors.json")
with open(tv_path, "w") as f:
    json.dump({
        "model_info": {
            "input_features": N_FEATURES,
            "input_dtype": "float32",
            "output": "probability (float32 in [0,1])",
            "classification_threshold": 0.5,
        },
        "tolerance": {
            "float32_reference": "exact (bit-equal to Python when using same float32 math)",
            "quantized_int8": "to be defined by quantization step",
        },
        "vectors": test_vectors,
    }, f, indent=2)
print(f"Saved {len(test_vectors)} test vectors to {tv_path}")

# 4) Print a couple of edge-case vectors so the user sees what's interesting.
print("\n--- Sample edge-case vectors (probability near 0.5) ---")
edge_only = [v for v in test_vectors if v["category"] == "edge_near_boundary"]
edge_sorted = sorted(edge_only, key=lambda v: abs(v["probability"] - 0.5))[:5]
for v in edge_sorted:
    print(f"  sum={v['input_sum']:.3f}  logit={v['logit_pre_sigmoid']:+.4f}  "
          f"prob={v['probability']:.4f}  class={v['predicted_class']}")

print("\nDone.")
