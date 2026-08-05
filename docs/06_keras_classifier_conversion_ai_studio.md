# Keras Classifier Conversion via ST Edge AI Studio — First Pass

> **Created:** 2026-05-20 (approximate — first experiment session)
> **Last edited:** 2026-08-05
> **Status:** Living document — detailed reference for the first experiment

Living document. Notes from the first attempt at running the
example Keras classifier through ST Edge AI Studio for deployment on
B-L4S5I-IOT01A (STM32L4S5VI). Pass 1 is float32; pass 2 will be int8.

---

## Session decisions (chronological)

- Float32 first; int8 quantization is pass 2.
- Tool: ST Edge AI Studio v4.0 (replaces X-CUBE-AI).
- Target: B-L4S5I-IOT01A (STM32L4S5VI).
- Optimization preference: "Balanced". Compression off.
- Memory pool: default / single pool (activations are 4 bytes for this model, doesn't matter functionally).
- Validate input data: random/synthetic for this pass. `test_vectors.json` schema not directly consumable by AI Studio.
- Validate location: host first, target later.
- **Handoff format: TFLite, not Keras.** See Req #2 and Bug #1.
- **Quantization workflow (pass 2): Pattern B** (Keras → quantized TFLite in Python → AI Studio). See Req #1.

---

## Notes organized by requirement

### Req #1 — Reduce model size (Quantization)

#### Three options for getting a quantized model into AI Studio

| Approach | Workflow | Trade-offs |
|---|---|---|
| **Pattern A** | Keras (`.keras`/`.h5`) → AI Studio does PTQ → C | Single tool. Tool has a Quantize button when input is Keras. But `.keras` is fragile (Bug #1). Quantization scheme lives in ST's tool; verification is harder (no Python ground truth for int8). |
| **Pattern B** | Keras → quantized `.tflite` in Python → AI Studio → C | **Chosen.** More steps. Stable handoff. Reproducible quantization in Python. Can run quantized `.tflite` in TFLite Python interpreter to produce ground-truth int8 outputs for verification. |
| **Hybrid** | Keras → `.h5` (legacy, no `quantization_config` field) → AI Studio does PTQ → C | Sidesteps Bug #1. Still cedes quantization control to ST. Worth knowing as a fallback. |

#### What AI Studio's UI offers (observed)

- **Input = Keras**: Quantize button present, calibration dataset prompt available.
- **Input = TFLite (float)**: No Quantize button. Tool treats the TFLite as source of truth — if float in, float out; if int8 in, int8 out. To get int8 you must quantize in TFLite first.
- **Input = TFLite (int8)**: Tool generates int8 C using `STAI_FORMAT_S8` data path.

So "do both quantization and conversion in AI Studio" requires staying on the Keras input path = Pattern A or Hybrid. Pattern B requires quantizing in Python before the tool sees the model.

#### Why Pattern B for this framework

- **Verification**: TFLite Python interpreter runs the quantized model in Python → ground-truth int8 outputs. Verification reduces to "does TFLite-in-Python agree with C-on-target?" — tractable.
- **Reproducibility**: Quantization choices are explicit in Python code, version-controlled, diffable. AI Studio's internal PTQ choices are opaque.
- **CI-ability**: Quantization is reproducible across machines; ST's PTQ may depend on tool version.
- Cost: extra step in the pipeline; AS needs to understand TFLite quantization API.

#### Pass-2 conversion sketch (for reference, not yet run)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = my_calibration_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
```

#### Open questions for pass 2

- What's the representative dataset for this model? Needs to cover the input distribution at deployment time.
- Does AI Studio honor all TFLite quantization choices (per-tensor vs per-axis, symmetric vs asymmetric)? Per-axis Conv weights are typical in TFLite; some kernels in ST runtime may only support per-tensor.
- What's the runtime's int8 op coverage? `stai.h` declares `STAI_FORMAT_S8` and the runtime supports it — but does it have every TFLite int8 op, or a subset?

#### Runtime format support (from `stai.h`)

The runtime supports: `FLOAT32`, `FLOAT64`, signed/unsigned ints `S/U 1/4/8/16/32/64`, and Q-format `Q7/Q15`. Standard int8 quantization targets `STAI_FORMAT_S8`. Confirms the runtime can handle pass-2 output; doesn't confirm every TFLite op is implemented.


### Req #2 — Maintainability

- **`.keras` is NOT a stable handoff format.** Serializes Python objects keyed to specific Keras class layouts. New Keras versions add fields older Keras can't deserialize. Confirmed by Bug #1 below.
- **`.tflite` IS a stable handoff format.** FlatBuffer with explicit, versioned schema (`schema.fbs`). Forward/backward compatibility is a design goal. Failure mode is "unsupported op" (debuggable), not "can't deserialize" (a wall).
- **`.onnx` is similar to `.tflite`** in stability. Versioned opsets, defined semantics. Backup option if TFLite hits walls in pass 2.
- **Decision**: framework handoff format = `.tflite`.
- Memory pool configuration is part of the artifact contract — must be version-controlled with the model, not a UI choice that drifts between runs.
- AI Studio version must be pinned and documented for reproducible builds.

### Req #3 — Verifiability (intermediate calculations)

**Bit-exactness is well-defined but rarely achievable across heterogeneous HW.**

IEEE-754 float32 is per-op deterministic. Non-determinism between Python (x86) and Cortex-M4F comes from:

- Op ordering in dot products (associativity not guaranteed)
- FMA presence (x86 yes, Cortex-M4F's VFPv4-SP no) — single biggest factor
- `expf()` implementation differences (libm varies across platforms)

User confirmed small differences between Claude-server Python and Mac Python — consistent with FMA + libm.

**Tolerance is application-defined, not converter-defined.**

- "Max error < X" is a smoke test, not a sign-off criterion.
- Classifier sign-off = agreement on predicted class across reference set.
- Regression = absolute or relative error tied to physical units.
- Anomaly detection = FPR/FNR shift on a labeled reference set.

**For THIS model** (Dense(1, sigmoid) on 6 features), starting absolute tolerance `1e-5`:

- Dot product of 6 mul-adds → ~1e-6 from accumulated rounding.
- Sigmoid approximation → up to ~1e-5 from libm differences.
- 1e-5 is a round number above both. Tighter than this is suspicious (likely not running what you think).

Smoke-test thresholds for AI Studio's reported max error:

- `< 1e-5`: clean, move on
- `1e-4` to `1e-3`: worth investigating, not blocking
- `> 1e-2`: stop, investigate

**AI Studio's built-in Validate step is one input to verifiability, not the whole story.** It compares Keras (Python) vs generated C (x86). It does NOT compare C-on-target vs Python-reference, which is the comparison the framework needs.

Update `test_vectors.json` tolerance field AFTER measuring actual error, not before.

### Req #4 — Compatible with Replay System

Not addressed this session.

Open: what does the Replay System consume? Inputs only? Inputs + intermediate state? This shapes whether the generated C API needs to expose intermediate-tensor hooks.

### Req #5 — Reduce power consumption

Not addressed directly this session.

Note for later: memory pool placement matters here. SRAM2 on L4S5VI can be retained in stop modes; SRAM1/3 may not be. If sleeping between inferences, weight/activation placement affects wake cost. Memory pool config has implications across Req #5, Req #6, and Req #2 (artifact contract).

### Req #6 — Reduce compute resources (optimization, acceleration)

- Open question (open #3): does AI Studio dispatch int8 ops to CMSIS-NN? Won't matter for this 7-param model, but the answer is foundational for real models.
- Memory pool config affects whether activations fit at all.
- AI Studio's optimization preference ("Balanced" / "RAM" / "Time") is the user-facing axis here. Varying it is a pass-3 experiment.

### Req #7 — Ease of use by Applied Scientist (no C/C++)

- AI Studio GUI is a partial answer but not full picture. AS still needs to:
  - Run a script to re-save model in stable format (`resave_model.py`).
  - Define calibration data for pass 2.
  - Define application-level sign-off criterion for verification.
- These are not C/C++ but they're not zero-effort. The framework should make them turnkey: templates, scripts, docs, sane defaults.

### Req #8 — Configurability (change parameters without FW revision)

- **User clarification**: this means hyperparameters / app-level config loaded from a config file at boot. NOT swap of model weights.
- Weights baked into `.rodata` is acceptable.
- AI Studio default output is fine for #8 as defined.
- Config-file loader is a separate component, not part of the AI Studio output. Design TBD.

---

## Findings from generated source (after successful TFLite load)

### Files generated by AI Studio

Categorized by what goes into firmware vs bookkeeping:

| File | Compile into FW? | Source/Binary | Per-model or shared | Role |
|---|---|---|---|---|
| `network.c` | Yes | Source | Per-model | Inference graph, node sequencing |
| `network.h` | Yes (include) | Source | Per-model | Public API, macros, model metadata |
| `network_data.c` | Yes | Source | Per-model | Embedded weight blob (packed `uint64_t`) |
| `network_data.h` | Yes (include) | Source | Per-model | Weight array declaration |
| `network_weights.c` | Yes | Source | Per-model | Weight-copy helper (`network_load_weights()`); empty stub for internal-Flash deployment |
| `network_weights.h` | Yes (include) | Source | Per-model | Header for the above |
| `network_details.c/h` (if generated) | Yes | Source | Per-model | Layer metadata, optional debug info |
| `stai.h`, `stai_debug.h`, `ai_*.h` | Yes (include) | Source headers | Shared | Runtime API surface |
| `NetworkRuntime1200_xxx_GCC.a` | Yes (link) | **Binary** | Shared | Precompiled inference runtime |
| Generation report `.txt`/`.json` | No | — | Per-model | Bookkeeping; version-control |
| Validation `.csv` | No | — | Per-model | Bookkeeping; version-control |

#### About the `.a` file

- Name pattern likely encodes version + ABI: `NetworkRuntime1200` ≈ runtime v12.0.0; `GCC` = GCC ABI.
- **ABI-locked**: a GCC `.a` won't link with Keil/ARMCC or IAR. AI Studio typically produces variants for each toolchain.
- **Target-locked**: this `.a` was generated for `stm32l4` (Cortex-M4F, hard-float ABI). Don't reuse on M0+/M7 without regenerating.
- Inspect with `arm-none-eabi-size NetworkRuntime1200_xxx_GCC.a` to see section sizes — that's the floor of the ML-enabled firmware overhead.
- **Implication for Req #2**: the `.a` is part of the artifact contract. Pin its version alongside the model and AI Studio version.

#### About `network_weights.c`

Body is currently empty (just comments). It's a stub for the **external-storage deployment scenario**: when weights live in external Flash (e.g., the MX25R6435F QSPI Flash on the B-L4S5I-IOT01A), this function copies them into SRAM at boot.

For internal-Flash deployment (what you're doing), weights live in `.rodata` and there's nothing to copy → empty body.

**Always call `network_load_weights()` from your firmware boot path anyway.** It costs nothing today (empty function call) and makes the firmware portable to external-Flash deployments without code changes. This is also the natural hook for a future "swappable weights" path (Req #8, if scope ever expands).

### Key facts from `network.h`

- Model signature: `0xd5597816a49de03468df38b9de4d58e1` — model hash. Use as a CI artifact-version identifier.
- `STAI_NETWORK_NODES_NUM = 2` (input + dense).
- `STAI_NETWORK_MACC_NUM = 17`.
- Input: float32, shape `{1, 6}`, 24 bytes.
- Output: float32, shape `{1, 1}`, 4 bytes.
- Activations: 28 bytes.
- Weights: 28 bytes useful / 32 bytes in file (with alignment padding).

### Weight storage — decoded (Req #8, Req #3)

```c
STAI_ALIGNED(8)
const uint64_t g_network_weights_array[4] = {
  0x3e514cc83e50d92aU, 0x3e46f0ba3e3fa7a5U, 0x3e5c14b63e701286U, 0xc0d4e416U,
};
```

**Brief on the `uint64_t[4]` storage**:
- 7 float32 weights = 28 B. Rounded up to multiple of 8 → 32 B → 4 × `uint64_t`.
- `STAI_ALIGNED(8)` ensures 8-byte alignment; lets FPU `VLDM` load 2 floats/cycle. (Not CMSIS-NN — that's integer-only.)
- Opaque `uint64_t` instead of `float[7]` so the same generator handles float32/float16/int8 without changing the declaration type.
- Little-endian: `0x3e514cc83e50d92aU` stores bytes `2a d9 50 3e 8c 4c 51 3e` at increasing addresses → two float32s `0x3e50d92a` (≈0.2040) and `0x3e514cc8` (≈0.2043).
- Last element `0xc0d4e416` is the bias (≈ −6.65); upper 4 bytes zero-padded.

**Verifiability**: cannot eyeball-verify against the source model. Need a Python script that reads `model.tflite` weights and compares against the hex here. CI-able.

### C API surface (Req #4, Req #7)

| Function | Role |
|---|---|
| `stai_network_init()` | Initialize opaque context (once at boot) |
| `network_load_weights()` | Copy weights from storage to RAM (empty stub for internal-Flash) |
| `stai_network_set_inputs/outputs/activations/weights()` | Point runtime at buffers |
| `stai_network_run()` | Run inference |
| `stai_network_set_callback()` | **Hook into intermediate activations during run** |
| `stai_network_get_error()` | Error reporting |

- API is buffer-pointer-based: application owns memory, runtime is told where it is.
- Integration ≈ 10 lines of glue (allocate, init, set pointers, run, read).
- `STAI_NETWORK_FLAGS = (INPUTS|OUTPUTS|WEIGHTS)` declares what buffers the runtime expects.

### The callback hook (`stai_network_set_callback`) is important

ST documents this as: *"an API to retrieve the content on intermediate activations buffers while executing run."*

Implication: **this is ST's primary mechanism for Req #3 (verifiability of intermediates) and Req #4 (replay).** Framework will register a callback that dumps every intermediate tensor.

For this 2-node model the hook is degenerate. **Exercise it on a real multi-layer model in a future pass** and verify:
- Granularity: per-node or per-tensor?
- Performance overhead with callback enabled.
- Buffer ownership: live runtime buffers, or copies?

### Library-size considerations (Req #6)

Per-model overhead:

| Component | This model | Scaling |
|---|---|---|
| Weights | 32 B | Linear with model size |
| Activations | 28 B | Scales with largest intermediate tensor |
| `network.c` | ~1 KB | Grows with #layers |
| Runtime (`.a`) | ~10–50 KB | **Fixed per build, not per model** |

Implication: per-model cost is small; per-framework cost is fixed. Adding a second model on top of an existing AI-enabled firmware costs only the model-specific files.

### Runtime library = part of the artifact contract (Req #2)

The `.a` is compiled with certain features. If a future model uses ops not present in the linked runtime, **link will fail**. Pin the runtime version alongside the model, AI Studio version, and memory pool config.

### Bug #2: "Build All" greyed out — wrong folder imported

- **Symptom**: After importing the AI Studio "Generate project" output into CubeIDE, the **Build All** action is greyed out and the project doesn't appear as a C project.
- **Root cause**: AI Studio puts the buildable project in a `CubeIDE/` subfolder. The top-level output folder contains generation reports and other artifacts, not a project. Importing the top-level folder gives you a non-C-project that can't be built.
- **Fix**: Import the `CubeIDE/` subfolder, not the top-level. File → Import → General → Existing Projects into Workspace → point at `<output>/CubeIDE/`.
- **Framework lesson**: document the import path explicitly in the Applied Scientist onboarding docs (Req #7). This is the kind of friction that wastes hours for new users.

## Bugs and gotchas encountered

### Bug #1: Keras version skew (E010 InvalidModelError)

- **Symptom**: AI Studio v4.0 failed to load `model.keras` with:
  ```
  E010(InvalidModelError): ...
  Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
  ```
- **Root cause**: Model was saved by Keras 3.3+, which writes a `quantization_config` field on Dense layers. AI Studio v4.0 bundles older Keras that doesn't recognize this field. Fails at deserialization time.
- **Workaround**: `resave_model.py` reloads the model and saves it as `.tflite` (preferred).
- **Framework lesson**: drove the Req #2 decision to standardize on `.tflite`.
- **Status**: workaround successful — TFLite loaded and generated code cleanly.

---

## Measured baseline (pass 1, float32)

### Generation report numbers

From `network_generate_report.txt`:

| Metric | Value | Notes |
|---|---|---|
| Model format | float32 | Confirmed `model_fmt: float` |
| Params | 7 (28 B) | 6 weights + 1 bias |
| MACC | 17 | 7 for Dense, 10 for sigmoid |
| Input | f32(1,6), 24 B | Allocated inside activations buffer |
| Output | f32(1,1), 4 B | Allocated inside activations buffer |
| **Flash (RO)** | **2,322 B** | Code + weights + lib |
| **RAM (RW)** | **28 B** | Activations only; in `.bss` (static) |

Flash breakdown:

| Segment | Bytes | Source |
|---|---|---|
| `network.o` text | 754 | Per-model inference graph |
| `NetworkRuntime1200_CM4_GCC.a` | 752 | **Lite runtime** (`use-lite-runtime` option) |
| `libm` / `libgcc` | 712 | `expf()` for sigmoid + helpers |
| `network.o` rodata | 48 | Layer metadata |
| weights rodata | 32 | The 4 × `uint64_t` array (28 B + 4 B padding) |
| toolchain rodata | 24 | Constants |
| **Total Flash** | **2,322** | |

RAM breakdown: 28 B `bss` (activations). Zero `data`. No stack growth from inference itself.

**Implication**: ST's "lite runtime" is ~750 B, far smaller than the ~10-50 KB figure I quoted earlier. The mode used was `use-lite-runtime` — there's a "full" runtime variant that's bigger; worth investigating what features the full variant offers (more ops? finer-grained callbacks?) before committing the framework to lite.

### Build options used (recorded for reproducibility)

```
options: allocate-inputs, allocate-outputs, multi-heaps, use-lite-runtime, use-st-ai
optimization: balanced
target/series: stm32l4
c_api: st-ai
```

- `allocate-inputs`/`allocate-outputs`: I/O buffers placed inside the activations buffer. Saves RAM. **Caveat**: app must consume outputs before the next inference, since the buffer is reused.
- `use-lite-runtime`: smaller runtime. Open question: feature trade-off vs full runtime.
- `multi-heaps`: irrelevant for this model; relevant for larger ones spanning multiple SRAM regions.

### Host validation results (`network_val_*_outputs_1.csv`)

10 random samples, TFLite reference (`m_outputs`) vs generated C (`c_outputs`), both compiled for x86:

| Metric | Value |
|---|---|
| Max absolute error | **1.4 × 10⁻⁹** |
| Mean absolute error | 5.4 × 10⁻¹⁰ |
| Max relative error | 5.0 × 10⁻⁷ |
| Mean relative error | 2.3 × 10⁻⁷ |
| Samples bit-exact | **2 of 10** |

Float32 epsilon at output magnitude (~2e-3) is `~2.4 × 10⁻¹⁰`, so worst error is ~6 ULPs. Consistent with rounding-only difference in the dot product and sigmoid path. No structural divergence.

Bit-exact agreement on 2 of 10 samples is a good sign — it means TFLite-on-x86 and generated-C-on-x86 take identical arithmetic paths for some inputs.

**This is the baseline.** On-target error (Cortex-M4F vs x86) will add to this; expect on-target error in the range 1e-7 to 1e-5 due to FMA absence and `expf()` differences.

### Tolerance decision (updates Req #3 section)

- Set `test_vectors.json` float32 absolute tolerance to **`1e-5`**: well above the 1.4e-9 host-side error, with headroom for FMA + `expf` differences on M4F.
- Sign-off criterion for the classifier is still **class agreement** (not tolerance) — to be checked on-target with the 254 test vectors.
- Update tolerance after on-target measurement if measured error is substantially lower.

## On-target verification: staged plan

AI Studio's "Generate project" creates a CubeIDE project with `STM32CubeAI_Studio_AI_Init()` and `STM32CubeAI_Studio_AI_Process()` wired into `main()`. Reading `app_x-cube-ai.c` shows the full picture:

- `acquire_and_process_data()` and `post_process()` are **empty stubs** (`return 0;`). Real code is inside comment blocks as a template.
- Activation buffer is in `.bss` → zero-initialized. With `allocate-inputs` build option, **inputs are also in this buffer = all zeros**.
- Inference runs on `[0,0,0,0,0,0]` every iteration → output is constant `sigmoid(bias) ≈ 0.0013`.
- Output goes nowhere (empty `post_process`).
- **UART IS initialized** (called from `STM32CubeAI_Studio_AI_Init`). The earlier note that "UART is not called" was wrong — `MX_USART1_UART_Init` is called via `MX_UARTx_Init()` inside the AI init path.
- `LC_PRINT(...)` macros print inference timing and cycle counts over UART every 5s.
- DWT cycle counter is wired up — real microsecond-precision inference timing available out of the box.

**Staged plan** (this session = Stage 1):

### Stage 1 — Build, flash, observe UART output (this session)

- Open project in CubeIDE.
- Build all. Record total binary size.
- Flash via Debug (F11) or Run.
- Open serial terminal at **115200 8N1** on the ST-Link VCP COM port.
- Expect lines like:
  ```
  ---- Inference number N ----
  Results for network "network"
  Running...
   duration DWT    : 0.XXX ms
   CPU cycles      : XXXX
   Sleep for 5s...
  ```
  every 5 seconds.

**Proves**: toolchain, runtime, inference execution, UART path, DWT cycle counter — all working. You also get **real inference timing** for free.
**Does NOT prove**: output correctness (inputs are zeros).

### Stage 2 — Inspect the harness (already done)

✅ Done by reading `app_x-cube-ai.c`. Key entry points identified:
- `acquire_and_process_data()`: where to fill inputs.
- `post_process()`: where to read outputs.
- Both are clearly marked `USER CODE BEGIN/END` regions — safe to edit.
- `stai_input[]` / `stai_output[]` arrays already hold the right pointers after `aiInit()`.

### Stage 3 — Real verification (next session)

To replace zero inputs with real test vectors, edit `acquire_and_process_data()`:

```c
int acquire_and_process_data()
{
  /* USER CODE BEGIN acquire_and_process_data */
  static const float test_input[6] = {4.27f, 6.34f, ...};  // from test_vectors.json
  memcpy(stai_input[0], test_input, sizeof(test_input));
  return 0;
  /* USER CODE END acquire_and_process_data */
}
```

And edit `post_process()` to print the output:

```c
int post_process()
{
  /* USER CODE BEGIN post_process */
  float prob = *(float*)stai_output[0];
  LC_PRINT(" prob = %f\r\n", prob);  // may need printf-float flag in linker
  return 0;
  /* USER CODE END post_process */
}
```

For full automation, replace the hardcoded vector with a UART-receive loop.

### Other findings from `app_x-cube-ai.c` (for future framework design)

| Observation | Implication |
|---|---|
| `STAI_ALIGNED(32)` on activation `RAM[]` | 32-byte alignment, more than strictly needed; defensive for future caches/wider loads |
| `__attribute__((section(".AI_RAM")))` on activation buffer | **This is how you control which SRAM region holds activations.** Edit linker script to map `.AI_RAM` to SRAM1/SRAM2/SRAM3 as needed for Req #5/Req #6 |
| `states_1[4]` allocated despite report saying states_size=0 | Dead allocation; harmless; just be aware generator can emit unused buffers |
| `stai_network_run(..., STAI_MODE_SYNC)` | Sync mode parameter; ASYNC mode presumably exists. Investigate for Req #6 (can inference overlap with other work?) |
| Loose error handling: `ret_code` overwritten without checking | Framework should gate each step on the previous return code. Don't rely on ST's template error path. |



## Stage 1 on-target results (float32, B-L4S5I-IOT01A)

### Build size (full binary, including HAL and board peripherals)

| Section | Bytes | Where |
|---|---|---|
| `text` | 27,472 | Flash (code + rodata) |
| `data` | 96 | Flash → RAM at boot |
| `bss` | 10,852 | RAM only |
| **Total Flash** | **27,568** | |
| **Total RAM** | **10,948** | |

ML overhead from the AI Studio report was 2,322 B Flash / 28 B RAM. The remaining ~25 KB Flash and ~10.9 KB RAM is HAL drivers, board peripheral init, `printf` machinery, C runtime, and stack/heap — non-ML.

**Framework number to remember**: adding ML to an STM32L4S5VI firmware costs **~2.3 KB Flash + ~28 B RAM** for this model. Small models are dominated by runtime overhead, not weights; big models will invert.

### On-target inference timing (UART output, inference #69 of >70 observed)

```
duration DWT    : 0.010 ms
duration SysTick: 0 ms
CPU cycles      : 1234
CPU cycles (avg): 1234
```

- **CPU cycles: 1,234 per inference, zero variance across 70 runs.** Deterministic execution as expected (no input-dependent branches).
- HCLK = 120 MHz (derived from `SystemClock_Config`: MSI 4 MHz × PLL N=60 / M=1 / R=2 = 120 MHz). FLASH_LATENCY_5 confirms.
- **Inference time = 1,234 / 120e6 = 10.28 µs per inference**.
- At 5 s between inferences (`HAL_Delay(5000)`), inference is ~0.0002% of CPU time. Trivial duty cycle.

### Where the cycles are spent (analysis)

- 7 MACs (Dense layer) ≈ 7 cycles on M4F with pipelined VFMA. **Negligible.**
- 1 sigmoid → `expf()` from libm. **Typically 200–500 cycles on Cortex-M4F.** Dominant cost.
- Remaining ~700–1,000 cycles: ST runtime dispatcher, kernel call overhead, buffer/error-code plumbing, callback no-op checks.

**Implication for framework**: nonlinearities (sigmoid, tanh, softmax) carry hidden cost on cores without hardware transcendentals. Pass 2 (int8) should drop this — TFLite typically uses LUT-based sigmoid, eliminating the `expf` call. Expect significant cycle reduction in int8, more than the Dense layer alone would predict.

For large models, per-MAC cost will dominate and activation overhead becomes proportionally small. But for shallow models with nonlinearities, the activation can be the bottleneck.

### Stage 1 outcomes — all proved

- ✅ Toolchain (CubeIDE + GCC) builds the AI Studio output cleanly
- ✅ ST AI runtime links with `NetworkRuntime1200_CM4_GCC.a`
- ✅ Inference executes on target without faulting
- ✅ UART path functional (`LC_PRINT` via USART1 → ST-Link VCP at 115200 8N1)
- ✅ DWT cycle counter functional; deterministic timing measurements
- ✅ First on-target performance baseline established: **1,234 cycles / 10.28 µs / inference on L4S5VI @ 120 MHz, float32**

### Stage 1 outcomes — NOT proved (Stage 3 work)

- Output correctness — inputs are zeros, output is constant `sigmoid(bias)`
- Output observability with real values — `post_process()` is empty
- Class-agreement rate vs `test_vectors.json` reference

## Open questions (answer during / after this pass)

1. ~~Where do weights live in generated C?~~ ✅ `const uint64_t[]` in `.rodata`; Flash.
2. ~~Runtime library — source or blob?~~ ✅ Precompiled `.a` (`NetworkRuntime1200_CM4_GCC.a`) in `Middlewares/.../Lib/`. Headers (source) in `Inc/`.
3. ~~Are int8 ops dispatching to CMSIS-NN?~~ Not relevant for float32 pass; check in pass 2.
4. ~~What's the exact C API the firmware team will call?~~ ✅ Mapped from `network.h` and `stai.h`.
5. ~~What does AI Studio's Validate report contain?~~ ✅ Random-input m/c CSV pair; we measured 1.4e-9 max abs error.
6. **NEW**: What does the "full" runtime variant offer vs `use-lite-runtime`? Worth a side-by-side generate to compare. Likely a pass-2 or pass-3 question.
7. **NEW**: The report says `multi-heaps` is enabled. For a model with activations larger than one SRAM region, how does the tool partition? Relevant for Req #5, #6 on real models.

## To do (this session — Stage 1)

- [x] Install / open AI Studio (v4.0)
- [x] Load `model.keras` — FAILED, see Bug #1
- [x] Re-save model as TFLite via `resave_model.py`
- [x] Load `model.tflite` in AI Studio — succeeded
- [x] Generate code — succeeded
- [x] Inspect `network_data.c` — weight array decoded
- [x] Inspect `network.h` — API surface mapped
- [x] Locate runtime — `.a` confirmed in `Middlewares/.../Lib/`
- [x] Read generation report — Flash 2,322 B / RAM 28 B / MACC 17
- [x] Run host Validate — max abs error 1.4e-9 (10 samples)
- [x] Generate full CubeIDE project via "Generate project"
- [x] Inspect generated `main.c` — confirms AI init+process wired into main loop
- [x] Inspect `app_x-cube-ai.c` — confirms zero inputs, UART working, DWT timing wired
- [x] Build full project in CubeIDE — text=27,472 / data=96 / bss=10,852 (38,420 total)
- [x] Flash to B-L4S5I-IOT01A — succeeded via Debug (F11)
- [x] Open serial terminal at 115200 8N1 on ST-Link VCP — output observed
- [x] Confirm `LC_PRINT` output every 5s with cycle counts — 1,234 cycles deterministic
- [x] **Stage 1 baseline: 10.28 µs/inference @ HCLK=120 MHz**

## To do (next sessions)

### Stage 3 — on-target verification with real data
- [ ] Add input data into `acquire_and_process_data()` — start with one hardcoded vector from `test_vectors.json`
- [ ] Add output print to `post_process()` — print the float probability
- [ ] May need to enable printf-float in linker flags (`-u _printf_float`)
- [ ] Compare against the expected probability in `test_vectors.json`
- [ ] Compute class agreement across all 254 vectors (sign-off criterion)
- [ ] Update tolerance field in `test_vectors.json` with measured value

### Pass 2 — int8 quantization
- [ ] Define calibration dataset for `tf.lite.TFLiteConverter`
- [ ] Generate quantized `.tflite` in Python (Pattern B)
- [ ] Compare TFLite-Python int8 outputs against float32 reference
- [ ] Feed quantized `.tflite` to AI Studio; verify it generates int8 C
- [ ] Compare on-target int8 outputs against TFLite-Python int8 outputs

### Framework architecture
- [ ] Exercise `stai_network_set_callback` with a real multi-layer model
- [ ] Investigate `STAI_MODE_ASYNC` — can inference overlap with other work?
- [ ] Replay-system integration (Req #4)
- [ ] Config-file loader design (Req #8)
- [ ] Power profiling (Req #5)
- [ ] Linker script: map `.AI_RAM` to specific SRAM region for Req #5 retention / Req #6 perf
- [ ] Compare `use-lite-runtime` vs full runtime (Req #6 trade-off study)
- [ ] Vary AI Studio optimization preference ("Time" vs "RAM" vs "Balanced")

## To do (later passes)

- Int8 quantization (pass 2) — Pattern B
- Calibration dataset definition for pass 2
- Replay-system integration (Req #4) — exercise `stai_network_set_callback`
- Per-layer intermediate dumps via callback hook (Req #3, beyond end-to-end)
- Config-file loader design (Req #8)
- Power profiling (Req #5)
- Vary AI Studio optimization preference (Req #6)
- Multi-layer model test — verify callback granularity, perf overhead
