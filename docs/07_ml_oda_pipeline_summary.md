# ML ODA Pipeline — Work Summary

> **Created:** 2026-08-05
> **Last edited:** 2026-08-05
> **Status:** Living document

Status document. Consolidates what's been done, where we are, and what's
next. For the deep technical details on any single item, see the companion
document `keras_classifier_conversion_ai_studio.md`.

---

## 1. Objective

Build a framework to take ML algorithms developed and trained in Python
and deploy them as C/C++ code on STM32 Cortex-M microcontrollers. Current
target hardware is STM32L4S5VI (Cortex-M4F), exercised on the
B-L4S5I-IOT01A discovery board; framework should generalize to other
STM32 Cortex-M parts.

The framework must satisfy eight requirements:

1. **Quantization** — reduce model size via lower numeric precision.
2. **Maintainability** — handoff/build stable across tool and library upgrades.
3. **Verifiability** — outputs checkable against Python reference, on host and target.
4. **Replay-system compatibility** — deployed model works inside our regression infrastructure.
5. **Power consumption** — fits energy budget.
6. **Compute resources** — uses available acceleration (FPU, CMSIS-NN, etc.).
7. **Ease of use for Applied Scientists** — no C/C++ knowledge required.
8. **Configurability** — app-level params changeable via config file, no FW rev.

Working assumption: **ST Edge AI Studio** is the right conversion core.
Under active evaluation.

---

## 2. What we did (chronological)

### Toy model chosen for the first experiment

To exercise the pipeline before touching a real Gridware model, we used a
deliberately trivial model: a single Dense layer with 6 float32 inputs and
sigmoid activation. 7 parameters total. Small enough that any tool
behavior is easy to inspect.

### Ran Keras → C conversion using ST Edge AI Studio

Loaded the trained model into AI Studio v4.0. The intent was Keras
directly, but we hit a bug immediately.

### Bug #1: Keras format version skew (important, and worth understanding)

Loading `model.keras` into AI Studio failed with:

```
E010(InvalidModelError): Unrecognized keyword arguments passed to Dense:
{'quantization_config': None}
```

**Root cause:** starting in Keras 3.3, the `Dense` layer's constructor
accepts a `quantization_config` parameter (added for native QAT support).
When Keras 3.3+ serializes a model to `.keras`, it writes out the value of
every constructor argument, including ones that default to `None`. So every
`.keras` file saved by Keras 3.3+ has `"quantization_config": null` on every
Dense layer, whether the user quantized or not.

AI Studio bundles an older Keras. Its `Dense.__init__` doesn't accept
`quantization_config`. Loading fails.

**This is the general fragility of `.keras` as a format**, not something
specific to our model or our config. Any Keras-3.3+-saved model will hit
this against tools bundling older Keras.

**Workaround:** re-save the model via `TFLiteConverter` in the training
environment's Python (`resave_model.py`). The `.tflite` file uses a
versioned FlatBuffer schema whose failure mode under version drift is
"unsupported op" (debuggable), not "can't deserialize" (a wall). We used
this workaround for the rest of the experiment.

**Framework-level implication:** the choice of handoff format (`.keras`
vs `.h5` vs `.tflite`) is a real architectural decision. We have
characterized three options but not committed to one; see companion doc
for the trade-off table.

### Generated C code, inspected the output

AI Studio produced:
- Per-model C source files (`network.c`, `network_data.c`, etc.)
- A precompiled runtime library (`.a`, ~976 KB archive, GCC ABI, Cortex-M4F)
- A complete CubeIDE project
- A generation report with predicted Flash/RAM/MACC numbers

Inspected the generated code to understand what the framework is
committing to. Notable findings:
- Weights are packed as `uint64_t` arrays with 8-byte alignment for FPU
  load efficiency.
- The runtime is a black-box archive — we can call it, we can't read it.
- There's a callback hook (`stai_network_set_callback`) that gives access
  to intermediate tensors during inference. This is the framework's
  planned lever for verifiability (Req #3) and replay (Req #4).

### Host-side validation

AI Studio's host validation ran the TFLite reference and the generated C
side-by-side on x86, over 10 random inputs. Max absolute error 1.4×10⁻⁹
(~6 ULPs at output magnitude). No structural divergence.

### Built and flashed to the discovery board

Imported the CubeIDE project (subtle gotcha: the buildable project is in
a `CubeIDE/` subfolder, not the output root — Bug #2). Built cleanly.
Total firmware size: 27,568 B Flash / 10,948 B RAM. ML-specific overhead
alone is 2,322 B Flash / 28 B RAM; the rest is HAL, board peripherals,
`printf`.

Flashed via Debug (F11). Opened a serial terminal at 115200 8N1 on the
ST-Link VCP. Observed cyclic UART output confirming inference runs and
the DWT cycle counter is measuring it.

### On-target validation via AI Studio's Validate-on-target mode

Same 10 random inputs, this time with inference running on the MCU and
outputs compared against the host TFLite reference. Results:
MAE ≈ 0, RMSE 1×10⁻⁹, SNR 130 dB, cosine similarity 1.0.

The FMA and `expf()` differences between x86 and Cortex-M4F that could
in theory introduce drift did not materialize at meaningful levels for
this small model.

### Per-layer timing breakdown

Pure `stai_network_run` = 1,026 cycles at HCLK 120 MHz = 8.6 µs. Split:
- Dense (matmul + bias): 446 cycles (43%)
- Sigmoid: 580 cycles (57%)

**Interesting finding:** a single sigmoid takes longer than the entire
6-input Dense layer because `expf()` is software-emulated on M4F. For
models with nonlinearities, activation cost is not negligible. Pass-2
int8 with LUT-based sigmoid is expected to reduce this substantially.

### Documentation and versioning
- Two Confluence pages published (project scope, experiment log).
- Repository cleaned up: `.gitignore` rewritten to track build
  prerequisites (runtime `.a`, HAL drivers, project descriptors) and
  exclude regenerable workspace files. Previous commit amended and
  pushed.

---

## 3. Where we are now

**Working end-to-end pipeline for the toy model, float32.**

- ✅ Keras → TFLite → AI Studio → CubeIDE → flashed board.
- ✅ Conversion accuracy validated (host and target, on AI Studio's random inputs).
- ✅ Inference runs on target, timing measured.
- ✅ Framework requirements #2, #3, #7 partially exercised. Learned real
     things about #6 (compute) from the per-layer breakdown.

**Not yet done:**

- ❌ Accuracy validated against our own reference test set (currently uses
     AI Studio's random inputs, not `test_vectors.json`).
- ❌ Quantization tested.
- ❌ Multi-layer model exercised.
- ❌ Real Gridware model exercised.
- ❌ Integration with the actual firmware repo (`gridware-bytes`).
- ❌ Any of Reqs #4 (replay), #5 (power), #8 (config).

---

## 4. Next steps

In the order we intend to tackle them:

### Step 1: Convert a real Gridware Keras model

- Identify which Gridware Keras model to use as the first realistic subject.
- Attempt handoff via `.keras` first, only to characterize the failure
  mode against the current Keras version of that model. May or may not hit
  Bug #1 depending on Keras version used to save it.
- If `.keras` fails: try `.h5` (the older Keras format may not have the
  `quantization_config` field and may load into AI Studio directly).
- If `.h5` also fails: fall back to `.tflite` via `TFLiteConverter`.
- **Document which path worked.** This is a real data point for the
  framework-level handoff-format decision.

### Step 2: Convert that model to network C files via AI Studio

- Load into AI Studio, generate C.
- Capture the generation report: memory footprint, MACC count, per-layer
  breakdown.
- Run host validation (AI Studio's random inputs). Confirm no structural
  divergence.
- Run target validation (if possible on the current dev board, or defer
  to Step 3).

### Step 3: Integrate the generated C into `gridware-bytes` and verify

This is the big step and has many substeps. Rough shape:

1. **Repo integration**: figure out where the generated files live in
   `gridware-bytes`. Probably a new subdirectory. Coordinate with the
   firmware team on the location and naming convention.
2. **Build system integration**: get the generated files, the ST AI
   runtime `.a`, and the appropriate headers into the `gridware-bytes`
   build. May need to add include paths, link flags, and possibly a new
   library target.
3. **Toolchain compatibility**: `gridware-bytes` may use a different
   compiler or ABI than the CubeIDE-generated project. If GCC ABI
   mismatches, we need a different runtime `.a` from AI Studio. Confirm.
4. **Application integration**: wire the model's I/O buffers into
   whatever produces the input features on the real device. This replaces
   the empty `acquire_and_process_data()` stub with real feature-gathering
   code. Coordinate with the algorithm and firmware teams.
5. **Output consumption**: whatever consumes the model output
   (thresholding, decision logic) needs to actually use `stai_output[]`.
6. **On-target accuracy verification**: run the reference test set
   through the integrated system and compare against the Python reference.
   This is the real Req #3 sign-off. Depending on the model, sign-off
   criterion might be class agreement, regression error in physical units,
   or FPR/FNR shift.
7. **Performance measurement**: cycle counts, timing, memory usage in the
   integrated environment. Compare against the standalone-project numbers
   to see if integration added overhead.
8. **Handling for the callback hook**: if intermediate-tensor verification
   is needed for this model, wire up `stai_network_set_callback` and
   validate against Python reference at each layer.

#### Which files migrate from the AI Studio project to `gridware-bytes`

The AI Studio-generated project has ~300 files. Only a subset is
appropriate to bring into a real firmware repo. Categorized:

**Category 1 — must migrate — model-specific source (7 files, regenerated per model)**

| File | Role |
|---|---|
| `AI/App/network.c` | Inference graph (which layers, in what order) |
| `AI/App/network.h` | Public API: buffer sizes, model signature, layer count |
| `AI/App/network_data.c` | Weight blob (packed `uint64_t` array) |
| `AI/App/network_data.h` | Declares the weight array |
| `AI/App/network_weights.c` | `network_load_weights()` stub |
| `AI/App/network_weights.h` | Its header |
| `AI/App/network_details.h` | Layer metadata |

**Category 2 — must migrate — shared runtime (not model-specific, reused across all models)**

| File(s) | Role |
|---|---|
| `Middlewares/ST/AI/Lib/NetworkRuntime1200_CM4_GCC.a/NetworkRuntime1200_CM4_GCC.a` | Precompiled inference runtime (the actual static library) |
| `Middlewares/ST/AI/Inc/*.h` (~100 headers) | Runtime API declarations |

**Category 3 — adapt, don't copy**

| File | Migration question |
|---|---|
| `AI/App/app_x-cube-ai.c` / `.h` | The harness with `aiInit()`, `aiRun()`, `acquire_and_process_data()`, etc. Rewrite as a proper `gridware-bytes` module fitting the existing task/scheduling model. Don't paste wholesale. |
| `AI/App/bsp_ai.h`, `user_init.c` / `.h` | Board-support glue; may duplicate what `gridware-bytes` already provides. |
| `Middlewares/ST/AI/Misc/*` (`lc_print`, cycle counter, test utility) | Debug helpers. `gridware-bytes` likely has its own logging/timing infrastructure. |

**Do NOT migrate**

- `Core/Src/main.c`, `stm32l4xx_hal_msp.c`, `stm32l4xx_it.c`,
  `system_stm32l4xx.c` — board bring-up; `gridware-bytes` has its own.
- `Drivers/CMSIS/`, `Drivers/STM32L4xx_HAL_Driver/` — HAL and CMSIS;
  `gridware-bytes` has its own pinned versions.
- `STM32CubeIDE/*.ld` (linker scripts) — `gridware-bytes` has its own;
  may need to *edit* it to add a section for AI activations, not replace.
- `test_05_14_2026.ioc` — CubeMX config; irrelevant to `gridware-bytes`.

#### Open questions about `gridware-bytes` (before finalizing the migration plan)

Answering these turns the generic guidance above into a concrete plan:

1. **Toolchain**: does `gridware-bytes` build with GCC? If ARMCC/Keil or
   IAR, we need to regenerate the runtime `.a` from AI Studio with the
   matching toolchain flag.
2. **Cortex-M variant**: `gridware-bytes` targets STM32L4A6 (Hamilton)
   and STM32L4R9 (Diablo), both Cortex-M4F. The `_CM4_GCC.a` should work
   for both. Confirm.
3. **Conventions**: where do third-party libraries live in `gridware-bytes`?
   Is there an existing `middlewares/` or `third_party/` directory?
4. **Build system**: does `gridware-bytes` use a Makefile, CMake, or
   something custom? Are new `.c` files picked up automatically or via
   a manifest?
5. **Scheduling**: the ST harness runs inference from a `while(1)` with
   a 5-second sleep. `gridware-bytes` likely has a real scheduler (RTOS,
   cooperative, interrupt-driven). Inference must be adapted to fit that
   model, not just called from `main()`.
6. **libm**: is `expf()` (needed by sigmoid) already linked in
   `gridware-bytes`? If not, add `-lm` or equivalent.
7. **Existing ML/DSP area**: does `gridware-bytes` already ship DSP or ML
   code (e.g. the ZDLB XGBoost model)? If so, the new ST AI runtime should
   coexist rather than replace.

**Note**: this file-categorization is generic guidance based on inspection
of the AI Studio project alone. Actual migration should be planned with
someone who knows the `gridware-bytes` structure firsthand — the answers
to the seven questions above turn generic advice into a concrete plan.

### Step 4: Integrate quantization

Also large. Two main sub-decisions inside this step:

- **PTQ vs QAT.** Post-Training Quantization is simpler; you quantize a
  trained model using a small calibration set. Quantization-Aware Training
  bakes quantization into the training loop, giving better accuracy on
  aggressive quantization at the cost of retraining. PTQ first, likely.
- **Pattern A vs Pattern B.** Pattern A: hand `.keras`/`.h5` to AI Studio
  and let it do PTQ. Pattern B: quantize in Python via `TFLiteConverter`
  first, hand quantized `.tflite` to AI Studio. Both are viable; Pattern B
  gives better verifiability (TFLite Python interpreter as ground truth).

Substeps within this step:

1. Define the calibration dataset for PTQ. Should be representative of the
   input distribution the deployed model will see.
2. Quantize the model (via chosen pattern), produce an int8 artifact.
3. Verify the quantized model in Python against float32 reference: what's
   the accuracy degradation? Is it acceptable?
4. Convert the quantized artifact to C via AI Studio. Verify the tool
   supports the specific quantization scheme used (per-tensor vs per-axis,
   symmetric vs asymmetric).
5. Rebuild the firmware, on-target validation against Python reference.
6. Compare cycle counts and memory footprint vs float32 baseline. Expected
   direction: substantially fewer cycles (LUT sigmoid replaces `expf`),
   smaller weights.
7. If Pattern A was used: figure out how to reproduce ST's quantization
   choices in Python for verification, or document that we're accepting
   ST's PTQ as-is.
8. If accuracy degradation is unacceptable, consider QAT: modify the
   training loop to include fake-quantization ops, retrain, re-export,
   repeat.

---

## 5. Open framework-level questions

- **Handoff format**: `.keras` vs `.h5` vs `.tflite`. Data being gathered
  as we work through real models.
- **Quantization pattern**: A vs B, per above.
- **Runtime lite vs full**: we used `use-lite-runtime` (~750 B). The full
  runtime is larger; unclear what it offers.
- **Config-file mechanism (Req #8)**: not addressed. Design TBD.
- **Replay-system integration (Req #4)**: not addressed. Depends on what
  the replay system consumes (inputs only? intermediate state?).
- **Power measurement methodology (Req #5)**: not addressed.

---

## 6. Companion documents

- `keras_classifier_conversion_ai_studio.md` — detailed notes from the
  first experiment. Deep reference material.
- Confluence page: *ML ODA Pipeline — Project Scope and Requirements*
  (stakeholder-facing charter).
- Confluence page: *Converting a Keras Model to C for STM32: First Pass
  with ST Edge AI Studio* (stakeholder-facing experiment log).
- Repository: `github.com/dheeraj-golla/STM32MLPipeLine`.
