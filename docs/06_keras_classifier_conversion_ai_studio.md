# Keras Classifier Conversion via ST Edge AI Studio

Living document for the experiment of converting a Python-trained Keras
classifier to C for the B-L4S5I-IOT01A (STM32L4S5VI) using ST Edge AI Studio v4.0.

**Scope of this pass**: float32 only, end-to-end pipeline.
**Out of scope for this pass**: int8 quantization, replay-system integration,
power profiling, multi-layer model verification.

---

## 1. Status

| | |
|---|---|
| Pipeline end-to-end | ✅ working (Keras → TFLite → AI Studio → CubeIDE → board) |
| Host conversion accuracy | ✅ validated (1.4e-9 max abs error, 10 random samples) |
| On-target execution | ✅ inference runs, deterministic timing |
| On-target conversion accuracy | ✅ validated by AI Studio (10 random samples; MAE ≈ 0, SNR 130 dB) |
| **Next: test with real data** | ❌ on-target accuracy vs 254-vector reference set not yet measured |
| **Then: quantize** (int8) | ❌ deferred |
| **Then: test with actual gridware model** | ❌ deferred (realistic multi-layer model) |

---

## 2. The eight framework requirements

The broader framework being designed must address eight requirements. This
experiment exercises some of them and is informative about others.

| # | Requirement | Exercised in pass 1? |
|---|---|---|
| 1 | Quantization (reduce model size via lower precision) | Partially — quantization paths characterized but not run |
| 2 | Maintainability (handoff stable across tool/library versions) | Yes — Bug #1 forced an early decision point |
| 3 | Verifiability (outputs incl. intermediates checkable vs Python) | Partially — host + AI-Studio target validation done; intermediate-tensor inspection not exercised |
| 4 | Replay-system compatibility | No |
| 5 | Power consumption | No |
| 6 | Compute resources (FPU, CMSIS-NN, etc.) | Partially — float32 cycle counts measured |
| 7 | Ease of use for Applied Scientists (no C/C++) | Partially — AI Studio UI walked through; gotchas documented |
| 8 | Configurability via config file (no FW rev needed) | No, but constraint clarified (means app hyperparameters, not swappable weights) |

---

## 3. The model

A toy logistic-regression classifier — single Dense layer, 6 float32 inputs,
sigmoid activation. 7 parameters total (6 weights + 1 bias). Deliberately
simple so the conversion tool's behavior is easy to inspect.

```python
model = keras.Sequential([
    keras.Input(shape=(6,), name="features"),
    keras.layers.Dense(1, activation="sigmoid", name="classifier"),
])
```

---

## 4. Tool stack and configuration

| | |
|---|---|
| Training | Python 3.x with Keras 3.3+ (whatever the training env has) |
| Conversion | ST Edge AI Studio v4.0 (replaces X-CUBE-AI — older tutorials use the old name) |
| Build | STM32CubeIDE |
| Flash | STM32CubeProgrammer (via Debug-as-application in CubeIDE) |
| Target board | B-L4S5I-IOT01A |
| Target MCU | STM32L4S5VI (Cortex-M4F, FPU, no FMA) |
| HCLK | 120 MHz (MSI 4 MHz × PLL N=60 / M=1 / R=2) |

**AI Studio build options recorded** (for reproducibility):
- `optimization: balanced`
- `target/series: stm32l4`
- `c_api: st-ai`
- `options: allocate-inputs, allocate-outputs, multi-heaps, use-lite-runtime, use-st-ai`
- `memory_pool: default` (activations are 28 B — placement doesn't matter for this model)
- Runtime version: `v12.0.0-2ca1f59f` compiled with GCC 13.3.1

These all belong in the artifact contract (Req #2). Memory pool, optimization
preference, and runtime version can all change generated code structure.

---

## 5. Workflow

1. Train in Python → save model.
2. Re-save as `.tflite` (workaround for Bug #1).
3. Load into ST Edge AI Studio → Analyze → Generate Project (full CubeIDE project).
4. Open `CubeIDE/` subfolder in STM32CubeIDE (Bug #2).
5. Build → flash via Debug → open serial terminal at 115200 8N1 on ST-Link VCP.
6. Observe `LC_PRINT` output every 5 seconds.

---

## 6. Handoff format: three options

AI Studio accepts several input formats. Each has trade-offs that matter
for the framework. **The framework-level choice is still open** and will be
revisited after pass 2.

| Format | Pros | Cons | Quantization path |
|---|---|---|---|
| **`.keras`** (Keras native) | Single tool from training to C; AI Studio offers in-tool Quantize button (Pattern A) | Serializes Python objects keyed to specific Keras class layouts. Loading fails when AI Studio's bundled Keras doesn't match the training-env version (Bug #1). Quantization scheme lives in ST's tool — harder to reproduce in Python | AI Studio does PTQ; you provide calibration dataset |
| **`.h5`** (Keras legacy) | More stable than `.keras` across versions; AI Studio still offers in-tool quantize | Older format, no longer Keras default; same loss-of-quantization-control as `.keras` | Same as `.keras` |
| **`.tflite`** | Versioned FlatBuffer schema, much more stable across versions. Failure mode under skew is "unsupported op" (debuggable) instead of "can't deserialize" (a wall). Used in this pass | Quantization must be done in Python *before* handoff (AI Studio does NOT offer in-tool quantize for TFLite input). More steps in pipeline | TFLiteConverter in Python (Pattern B); quantized `.tflite` runs in Python interpreter as ground-truth reference |

### Three quantization patterns (Req #1)

| Pattern | Flow | Trade-offs |
|---|---|---|
| **A** | Keras → AI Studio does PTQ → C | Simplest. Tool controls scheme. Verification harder (no Python ground truth). |
| **B** | Keras → quantized `.tflite` (Python) → AI Studio → C | Reproducible quantization. Python ground truth for verification. More work. |
| **Hybrid** | `.h5` → AI Studio does PTQ → C | Sidesteps Bug #1. Still cedes quantization control to ST. |

**Open question for pass 2**: which to use? Pattern B is most rigorous but
adds friction for Applied Scientists (Req #7). Pattern A is simpler but
ties verification to ST's internal quantization choices. Decide based on
pass-2 experimentation, not a priori.

---

## 7. Conversion output

### Files generated

| File | Compile into FW? | Source/Binary | Per-model or shared | Role |
|---|---|---|---|---|
| `network.c` / `.h` | Yes | Source | Per-model | Inference graph + public API |
| `network_data.c` / `.h` | Yes | Source | Per-model | Weight blob as packed `uint64_t` |
| `network_weights.c` / `.h` | Yes | Source | Per-model | Weight-copy helper (empty stub for internal Flash) |
| `network_details.c` / `.h` (if present) | Yes | Source | Per-model | Layer metadata, optional debug info |
| `stai.h`, `stai_debug.h`, `ai_*.h` | Yes (include) | Source headers | Shared | Runtime API |
| `NetworkRuntime1200_CM4_GCC.a` | Yes (link) | **Binary** | Shared | Precompiled runtime |
| Generation report, validation CSVs | No (bookkeeping) | — | Per-run | Version-control alongside model |

### The `.a` is ABI- and target-locked

- `GCC` in the name = GCC ABI. Won't link with Keil/ARMCC or IAR. AI Studio
  typically generates per-toolchain variants.
- Generated for Cortex-M4F hard-float ABI. Won't work on M0+/M7 without
  regenerating.
- Runtime version (`v12.0.0-2ca1f59f`) is part of the artifact contract — pin
  it alongside the model and AI Studio version.

### Weight storage (decoded)

```c
STAI_ALIGNED(8)
const uint64_t g_network_weights_array[4] = {
  0x3e514cc83e50d92aU, 0x3e46f0ba3e3fa7a5U, 0x3e5c14b63e701286U, 0xc0d4e416U,
};
```

- 7 float32 weights = 28 B. Padded to 32 B (`uint64_t[4]`) for 8-byte alignment.
- `STAI_ALIGNED(8)` lets FPU `VLDM` load 2 floats/cycle.
- Opaque `uint64_t` (not `float[7]`) so the same generator handles fp32/fp16/int8 without changing declared type.
- Little-endian: first u64 contains float32 `0x3e50d92a` ≈ 0.2040 (low addr) and `0x3e514cc8` ≈ 0.2043.
- Last entry `0xc0d4e416` ≈ −6.65 is the bias.
- `const` → `.rodata` → Flash.

### Memory footprint

ML-only (from AI Studio's report):

| | Size |
|---|---|
| Flash (text + rodata + lib) | 2,322 B |
| RAM (activations, bss) | 28 B |
| MACC | 17 |

Full firmware image (built CubeIDE project, includes HAL + board peripherals):

| Section | Bytes |
|---|---|
| `text` (Flash) | 27,472 |
| `data` (Flash → RAM at boot) | 96 |
| `bss` (RAM only) | 10,852 |
| **Total Flash** | **27,568** |
| **Total RAM** | **10,948** |

The ~25 KB Flash overhead is HAL drivers, peripheral init, `printf` machinery,
and C runtime — none of it ML-related. The ~10.9 KB RAM is mostly main stack,
heap, HAL handles, UART buffers.

**Framework takeaway**: adding ML to STM32L4S5VI firmware costs ~2.3 KB Flash
+ 28 B RAM for this model. Small models are dominated by runtime overhead;
big models will invert.

---

## 8. Integration into firmware

AI Studio's generated `main.c` wraps the runtime into two entry points:

```c
int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  STM32CubeAI_Studio_AI_Init();      // initializes UART + ST AI runtime
  while (1) {
    STM32CubeAI_Studio_AI_Process(); // runs inference in a 5 s loop
  }
}
```

The most relevant underlying API calls:

| Function | What it does |
|---|---|
| `stai_runtime_init()` | Initialize the runtime library globally. |
| `stai_network_init()` | Initialize a model's runtime context (called once per model). |
| `stai_network_set_activations()` | Tell runtime where the scratch buffer lives. |
| `stai_network_get_inputs()` / `set_inputs()` | Buffer pointers for input tensors. |
| `stai_network_get_outputs()` / `set_outputs()` | Buffer pointers for output tensors. |
| `stai_network_run()` | Execute one inference (sync or async). |
| `stai_network_set_callback()` | Register callback to inspect intermediate activations during inference. The planned hook for Reqs #3 and #4. |
| `stai_network_get_error()` | Read the runtime's error state. |

### Default harness runs on ZERO inputs

`acquire_and_process_data()` and `post_process()` in `app_x-cube-ai.c` are
empty stubs. The activation buffer is zero-initialized (`bss`), so the model
runs inference on `[0, 0, 0, 0, 0, 0]` every loop iteration — output is
constant `sigmoid(bias) ≈ 0.0013`. To exercise real inputs, fill these stubs.

### Useful capabilities already wired

| Feature | Notes |
|---|---|
| UART (`LC_PRINT`) | USART1 → ST-Link VCP, 115200 8N1. Working out of the box. |
| DWT cycle counter | Microsecond-precision inference timing in `aiRun()`. |
| Linker section `.AI_RAM` | Activation buffer placed in its own section — hook for placing activations in specific SRAM region (Reqs #5, #6). |
| `STAI_MODE_SYNC` | Default mode. `STAI_MODE_ASYNC` exists; not yet investigated. |

---

## 9. Validation

### Host (PC): TFLite reference vs generated C

AI Studio's `validate --mode host` runs both on x86 with 10 random inputs.

| Metric | Value |
|---|---|
| Max absolute error | 1.4 × 10⁻⁹ |
| Mean absolute error | 5.4 × 10⁻¹⁰ |
| Samples bit-exact | 2 of 10 |

About 6 ULPs at the output magnitude. Consistent with float32 rounding only.
No structural divergence.

> **ULP (Unit in the Last Place)**: the gap between two consecutive
> representable float values at a given magnitude. Float32 has a 23-bit
> mantissa → relative epsilon ≈ 1.19e-7. At output magnitude ~2e-3, one ULP
> ≈ 2.4e-10 absolute. Max measured error 1.4e-9 ÷ 2.4e-10 ≈ 5.8 ULPs.
> Saying "6 ULPs" instead of "1.4e-9" is more useful: it conveys that the
> error is just float-rounding noise (no structural divergence), regardless
> of the output's absolute scale. Bit-exact = 0 ULP.

### Target (board): same setup, but generated C runs on the MCU

AI Studio's `validate --mode target` flashes a validation firmware that
exchanges inputs/outputs with the PC over UART (protobuf protocol), runs the
model on-device, and compares against the host reference.

| Metric | Value |
|---|---|
| MAE | ≈ 0 |
| RMSE | 1 × 10⁻⁹ |
| L2 relative error | 3 × 10⁻⁷ |
| SNR | 130 dB |
| Cosine similarity | 1.000000 |

**On-target error is essentially equivalent to host error.** The FMA/`expf`
differences between x86 and Cortex-M4F that could in principle cause drift
did NOT materialize at a meaningful level for this model. Likely because the
model is small (no error accumulation in a 6-MAC dot product) and the sigmoid
is operating in a saturated region (small input deltas don't move the output).

### Important caveat

Both validations use **10 random inputs picked by AI Studio**, not our
254-vector reference set (`test_vectors.json`). Random-input validation is a
smoke test on the conversion pipeline. It is **not** sign-off-grade validation
of the model. For sign-off (Req #3) we need:

- Inputs from the reference test set
- Output comparison against the Python reference's saved probabilities
- Sign-off criterion = class agreement, not numerical tolerance

This is the immediate next step.

### Why "max error < X" is not a sign-off criterion

- Max error depends on the input set. If validation never saw the failure
  case, max error tells you nothing about it.
- For a classifier, what matters is whether errors cross the decision
  threshold (e.g. 0.5 for binary sigmoid). A 0.1 error at output 0.9 is
  fine; the same error at output 0.49 is a wrong prediction.
- Right framing: "0 disagreements on N reference vectors" or "≤ K
  disagreements where K is justified," with max/mean error as diagnostic.

---

## 10. On-target performance

### Whole-network timing

Two numbers, both correct but measuring different things:

| Source | Cycles | µs @ 120 MHz | What it measures |
|---|---|---|---|
| AI Studio Validate-on-target | **1,026** | 8.6 | Pure `stai_network_run()` call |
| Flashed Cube project (our build, UART-printed) | 1,234 | 10.3 | Same call + harness overhead (UART, DWT setup, error checks) |

Use 1,026 as the "inference compute" baseline. Use 1,234 as "what it
costs in a realistic app loop." Both are deterministic across 70+ runs
(zero variance).

### Per-layer breakdown (from Validate-on-target)

| Layer | Cycles | % of total |
|---|---|---|
| Dense (matmul + bias) | 446 | 43% |
| Sigmoid (nonlinearity) | 580 | 57% |
| **Total** | **1,026** | |

**Key finding: sigmoid dominates inference cost.** A single sigmoid takes
longer than the entire 6-input Dense layer. Cause: `expf()` from libm,
software-emulated on Cortex-M4F (no hardware transcendentals).

Per-MAC cost in the Dense layer = 446 / 7 ≈ 64 cycles/MAC. Way above the
theoretical ~1 cycle/MAC for VFMA-pipelined. The overhead is in kernel
dispatch and bounds-checking, not arithmetic — fixed-cost-dominated regime
for a model this tiny. Real models with 1000+ MACs will amortize this.

### Implications for pass 2 (int8)

- TFLite's int8 sigmoid is LUT-based, not `expf`-based. Expected: ~50–100
  cycles instead of 580. Net inference cycle reduction proportional to that.
- Int8 Dense layer can use SIMD instructions (`SMLAD`, etc.) on M4F's DSP
  extension. Expected: meaningfully lower than 446 cycles for the same MACs.
- Combined effect: a fair-but-rough expectation is 3–5× speedup for this
  model, more for bigger models.

### Runtime capabilities (from Validate-on-target metadata)

- Callback granularity: `IO_ONLY`, `PER_LAYER`, `PER_LAYER_WITH_DATA`.
  `PER_LAYER_WITH_DATA` is the option that gives us the intermediate tensors
  needed for Req #3.
- Device attrs: `fpu, art_lat=5, art_icache, art_dcache`. ART cache is on
  for both instructions and data. **Reported cycle counts are warm-cache.**
  Cold-cache numbers would be higher; relevant if the framework ever has to
  wake from a stop mode and run one inference before the cache warms.

---

## 11. Bugs and gotchas

### Bug #1: Keras format version skew

- **Symptom**: AI Studio v4.0 fails to load `model.keras`:
  ```
  E010(InvalidModelError): Unrecognized keyword arguments passed to Dense:
  {'quantization_config': None}
  ```
- **Root cause**: Training env (Keras 3.3+) writes a `quantization_config`
  field on Dense layers; AI Studio bundles older Keras that can't deserialize it.
- **Workaround**: re-save model as `.tflite` (or `.h5`) before AI Studio. See
  `resave_model.py`.
- **Framework implication**: drives Section 6 — the choice of handoff format
  has real maintainability consequences. `.keras` is fragile across Keras
  versions; `.tflite` and `.h5` are more stable.

### Bug #2: "Build All" greyed out in CubeIDE

- **Symptom**: After importing AI Studio's "Generate Project" output into
  CubeIDE, the Build All action is disabled and the project doesn't show as
  a C project.
- **Root cause**: AI Studio puts the buildable project in a `CubeIDE/`
  subfolder of its output. The top-level output folder is not a project.
- **Fix**: Import the `CubeIDE/` subfolder, not the top-level.
- **Framework implication**: document this in onboarding docs (Req #7) —
  cheap way to save AS users an hour of confusion.

### Gotchas (not bugs, worth knowing)

- `MX_USART1_UART_Init()` is called from the AI runtime init path, not from
  `main()` directly. So UART works "for free" if you use the generated
  scaffolding, but a custom `main()` needs to call it explicitly.
- `allocate-inputs` / `allocate-outputs` build options place I/O inside the
  activation buffer. Application MUST consume outputs before the next
  inference, or they get clobbered.
- `STAI_ALIGNED(32)` on the activation buffer — bigger than M4F strictly
  needs. Defensive for future caches or wider loads on other parts.
- `aiInit()` silently overwrites `ret_code` without checking — the framework
  should tighten this; don't rely on ST's template error handling.

---

## 12. Open questions for later passes

1. Will Pattern A (AI Studio quantization) and Pattern B (Python quantization)
   produce numerically different int8 outputs? If so, which to trust?
2. What's the representative dataset for the calibration step?
3. Does AI Studio honor all TFLite quantization choices (per-tensor vs
   per-axis, symmetric vs asymmetric)?
4. What does the full runtime variant offer over `use-lite-runtime` (which
   we used)?
5. How does `STAI_MODE_ASYNC` work? Can inference overlap with other work?
6. Memory-pool partitioning behavior for multi-region models — not relevant
   here, but will be for real models with activations > 64 KB.
7. Cold-cache vs warm-cache cycle counts — relevant for power-aware
   wake-from-stop scenarios.
8. Callback overhead with `PER_LAYER_WITH_DATA` — what's the perf cost when
   inspecting every intermediate tensor?

---

## 13. To-do

### Done

- [x] Install AI Studio v4.0
- [x] Re-save model as `.tflite` (workaround for Bug #1)
- [x] Generate C code + CubeIDE project
- [x] Inspect generated source (weight storage, API, harness)
- [x] Read generation report (Flash 2,322 B / RAM 28 B / MACC 17)
- [x] Host Validate (1.4e-9 max abs error)
- [x] Target Validate via AI Studio (MAE ≈ 0, 1,026 cycles, per-layer breakdown)
- [x] Build CubeIDE project (text=27,472 / data=96 / bss=10,852)
- [x] Flash to B-L4S5I-IOT01A
- [x] Observe UART output (`LC_PRINT` cycle counts every 5 s)

### Next: test with real data (current toy model, float32)

- [ ] Replace `acquire_and_process_data()` stub with real test vectors from `test_vectors.json` (start with one hardcoded; later UART loop for all 254)
- [ ] Add output print in `post_process()` (may need `-u _printf_float` linker flag)
- [ ] Compute class-agreement rate (sign-off criterion for Req #3)
- [ ] Update tolerance field in `test_vectors.json` with measured value

### Then: quantize (int8)

- [ ] Define calibration dataset
- [ ] Quantize in Python via `tf.lite.TFLiteConverter` (Pattern B sketch in Appendix B)
- [ ] Validate quantized `.tflite` in TFLite Python interpreter
- [ ] Feed quantized `.tflite` to AI Studio; verify int8 C generation
- [ ] On-target validation of int8 outputs
- [ ] Compare cycle counts vs float32 baseline (1,026 cycles)
- [ ] Decide: Pattern A or Pattern B for the framework?

### Then: test with actual gridware model (realistic multi-layer)

- [ ] Run the full pipeline on a realistic gridware model
- [ ] Exercise `stai_network_set_callback` with `PER_LAYER_WITH_DATA` (relevant for Reqs #3, #4)
- [ ] Re-evaluate footprint, cycle counts, accuracy on a real workload
- [ ] Revisit handoff-format decision in light of real-model behavior

### Other framework architecture work

- [ ] Investigate `STAI_MODE_ASYNC`
- [ ] Replay-system integration (Req #4)
- [ ] Config-file loader design (Req #8)
- [ ] Power profiling (Req #5)
- [ ] Linker script: map `.AI_RAM` to specific SRAM region for Reqs #5, #6
- [ ] Compare `use-lite-runtime` vs full runtime
- [ ] Vary AI Studio optimization preference ("Time" vs "RAM" vs "Balanced")

---

## Appendix A: TFLite resave script

```python
# resave_model.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("outputs/model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("outputs/model.tflite", "wb") as f:
    f.write(tflite_model)
```

## Appendix B: Pass-2 quantization sketch (not yet run)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = my_calibration_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
```

