# ST Edge AI Integration into gridware-bytes — Tracker

> **Created:** 2026-08-06
> **Last edited:** 2026-08-06
> **Status:** Living document — tracks the design discussion and decisions for
> integrating ST Edge AI Studio–generated model code into the `gridware-bytes`
> firmware repo. Companion to `07_ml_oda_pipeline_summary.md` (overall pipeline
> status); this doc owns the firmware-integration thread specifically.

---

## 1. Objective and scope

Integrate the ST Edge AI Studio conversion output (validated in this repo on
the toy Keras classifier — see `06_keras_classifier_conversion_ai_studio.md`)
into the production firmware repo `gridware-bytes` (locally `~/gridware-bytes`).

Scope decisions made so far:

- **The Python conversion pipeline stays out of `gridware-bytes`.** Model
  regeneration happens outside the firmware repo (here, or wherever the ML
  tooling ends up living). Only C artifacts and their wrapper move over.
- Work will be done jointly with the FW team (Dheeraj is not a C++ firmware
  expert; the FW engineer owns the firmware-side conventions).

Terminology used throughout:

- **ST Edge AI Studio / ST Edge AI Core / X-CUBE-AI** — ST's conversion tool
  (desktop UI / underlying engine / legacy Cube plugin name). Output: generated
  `network*.c/h` files per model + a precompiled ARM-only static runtime
  library ("NetworkRuntime").
- **TFLM** — TensorFlow Lite for Microcontrollers (now under the LiteRT
  umbrella, "LiteRT for Microcontrollers"). Open-source C++ interpreter; the
  model ships as a `.tflite` flatbuffer data blob.
- **CMSIS-NN** — ARM's optimized int8 kernel library, usable as a TFLM backend.
- **ODA** — on-device algorithm (Gridware term).

---

## 2. What we learned about gridware-bytes (survey, 2026-08-05)

The relevant tree is `firmware/` (the STM32L4 target; `fg28_firmware/` is the
Silicon Labs Wi-SUN side and is out of scope).

- **Build system:** Meson, cross-compiled with GCC (`arm.gcc.ini`,
  `platform/cortex-m4.ini`). Target: STM32L4 (Cortex-M4F).
- **Vendored dependencies** live as Meson subprojects —
  `firmware/subprojects/cubel4/` wraps STM32CubeL4 (HAL + CMSIS-DSP compiled
  from source with FFT-table size trims), plus `nanopb`, `zlib-deflate-nostdlib`.
  This is the pattern to copy for the ST runtime.
- **Application structure:** C++ with FreeRTOS. Services live in
  `firmware/src/services/` (apollo, athena, demeter, helios, hermes, hyperion…);
  shared infrastructure in `firmware/src/gridware/` (config, app_storage, blob,
  ota, crypto, hal, rtos wrappers).
- **Host-native build exists** (`native.gcc.ini`, `*.nativetest.cc`,
  `firmware/src/native/` analysis tools). Regression tests run on the host.
  This is load-bearing for the runtime decision (see §3).
- **Config system is protobuf (nanopb), not JSON.** JSON exists only cloud-side.
  Device config = statically-sized nanopb structs (`proto/config.proto`,
  `firmware/src/gridware/config.cc`).
- **Precedent:** model parameters already ship via config — decision-tree
  (forest) `bytes` fields, 160 B and 440 B (`config.proto` ~lines 544, 603).
  Pole health does weights-via-config with 16 parameters.
- An existing real-model reference: ZDLB XGBoost is integrated as a hand-written
  C++ header (copy in this repo at `ml_models/zdlb/xgbmodel_v0_7_4.h`).

---

## 3. Architecture options (runtime choice)

The ST-generated output has two parts: per-model generated code
(`network.c`, `network_data.c`, …) and the shared **NetworkRuntime** inference
library — a **closed-source, ARM-only, precompiled `.a`**. That binary is what
forces the decision, because it cannot run in the host-native test build.

### Option A — ST runtime, target-only inference

Vendor NetworkRuntime + its ~100 headers as a Meson subproject (like `cubel4`);
generated `network*.c` + a thin C++ init/run wrapper in e.g.
`firmware/src/gridware/ml/`. Host builds link a stub; bit-exactness is verified
on target (test vectors over UART or a test image).

- Pros: smallest flash/RAM, fastest inference (ST's optimized kernels), least
  code to own, direct continuation of the validated AI Studio flow.
- Cons: **no on-host model execution** (native regression tests can't run
  inference); the `.a` is toolchain-version-sensitive (must match firmware
  GCC); vendor lock-in; binary blob in repo.

### Option B — TFLM, runs everywhere

Vendor TFLM (source) as a subproject, optionally with CMSIS-NN kernels. Model =
`.tflite` byte array (data, not code). AI Studio remains useful as a
benchmark/validation tool but stops being the code generator.

- Pros: same inference code runs in the native build → host regression tests
  execute the model (framework Reqs #3 verifiability, #4 replay); model is
  data (see §4); no vendor lock-in.
- Cons: larger footprint (tens of KB interpreter + op kernels vs ST's ~KB);
  a C++ dependency the FW team must own; interpreter overhead per inference.

### Option C — Swappable backend behind one interface

Small `MlModel` C++ interface in `src/gridware/ml/`; ARM build links ST
runtime, native build links a reference backend (TFLM or plain CMSIS-DSP math).
Golden test vectors pin the two together.

- Pros: ST efficiency on target + host testability; the interface is what
  Applied Scientists see (Req #7), so a later runtime switch is invisible.
- Cons: two backends to keep in agreement; float rounding differences mean
  host↔target checks become tolerance-based, not bit-exact — weakens Req #3.

### Decision (2026-08-06): proceed with Option A first

**Option A (ST Edge AI Core generated code + NetworkRuntime) is the first
integration.** Rationale: it is the direct continuation of the flow already
validated end-to-end on the toy model (`06_...ai_studio.md`), has the smallest
footprint, and §5 (OTA reality) removed what looked like TFLM's biggest
near-term differentiator — under either runtime, weights ship baked into the
FW image today.

What this defers, not discards:

- **Host-testability gap is accepted for now** — see "Host (native x86)
  testing plan" below for what native coverage we keep and the strategies
  for the remaining gap. If it proves too painful in practice, Option B/C
  reopens.
- The A-vs-B flash/RAM/cycles measurement becomes optional background work
  rather than a decision gate.
- Keep the wrapper interface thin and runtime-agnostic (per the §5 design
  rule), so a later move to B or C changes the backend, not the callers.

FW engineer input still needed on: vendoring a closed-source `.a` (toolchain
pinning to the firmware GCC version), and where the wrapper lives (§7 Q3).

### Host (native x86) testing plan under Option A

The ARM-only piece is exactly one function's internals — the `Predict()` call
into NetworkRuntime. Plan:

1. **Design requirement: keep the un-testable surface minimal.** `Predict()`
   goes behind a clean seam (virtual method or link-time swap). Everything
   around it — feature assembly, circular buffers, gating/latching, config —
   stays portable C++ so the existing nativetest pattern
   (cf. `ml_plc.nativetest.cc`) covers it on x86 unchanged. This is the
   non-negotiable part; do it from the first commit.
2. **Host strategy for `Predict()` itself** — investigate in this order:
   - **ST's x86 host runtime (CONFIRMED, 2026-08-07, from ST Edge AI docs).**
     The pack ships static x86 libraries + headers at
     `$STEDGEAI_CORE_DIR/Utilities/<os>/targets/common/EmbedNets/tools/inspector/workspace/{include,lib}`
     (link: `-lruntime -lst_cmsis_nn -lcmsis_nn -lx86_cmsis -lm`), with the
     same embedded API as on target. ST's own docs describe this flow as
     intended for "a CI/CD flow w/o STM32 board" (article: "How to run
     locally a c-model"). `stedgeai validate --mode host` is the default
     validation mode, and `generate --dll` produces a shared lib for
     Python-driven checks (AiRunner). Caveat: host kernels are *reference*
     implementations — close to but **not bit-exact** with the ARM library
     (rounding/accumulation differ); host tests use tolerances
     (ST thresholds for float32: L2r < 0.01, COS >= 0.9999), while
     bit-exactness stays an on-target check. Libraries ship per-OS
     (windows/linux/mac) — CI OS must match an installed pack.
   - **Playback stub.** Native `Predict()` looks up the input in the golden
     test vectors (`test_vectors.json`) and returns the recorded Python
     reference output. No host inference, but end-to-end native pipeline
     tests run against ground-truth model behavior. Cheap, deterministic,
     no new dependencies.
   - **Reference implementation.** Hand-written portable C++ math as the
     native backend (trivial for Dense-sigmoid). Real host computation, but
     a second implementation to keep aligned (tolerance-based, not
     bit-exact) — this is Option C creeping back; only if the above fail.
3. **Golden vectors are shared test data** — host tests and on-target
   validation must check against the identical reference outputs, wired into
   the native test build (Meson) rather than copied ad hoc.

---

## 4. "Model as data": weights vs graph, ST vs TFLM

Precise statement of who can update what without recompiling firmware:

| Update type | ST Edge AI Core | TFLM |
| --- | --- | --- |
| Same architecture, new weights | ✅ via external-weights option | ✅ blob update |
| Different architecture (new layer sizes/count) | ❌ new `network.c` → FW build | ✅ blob update *if* ops already registered and arena fits |
| New operator type | ❌ FW build | ❌ FW build (kernel must be compiled in) |

Details:

- **ST, default flow:** weights are generated into `network_data.c` and
  compiled in. Any retrain → regenerate → recompile → new FW image.
- **ST, external-weights flow:** ST Edge AI Core can emit weights as a raw
  binary blob (placed in a chosen flash region / linker section); the generated
  graph code references them by address. Same-architecture retrains regenerate
  an **identical** `network.c` (same topology + same tool version), so only the
  blob changes — no recompile. Caveats: blob layout is tied to the ST tool
  version (pin it; regenerate blob and graph with the same version); "same
  architecture" is strict — any shape change is a FW update. A useful CI check:
  diff regenerated `network.c` against the committed one before accepting a
  weights-only update.
- **TFLM:** the `.tflite` flatbuffer encodes **graph + weights** together. The
  interpreter takes a pointer to it at runtime; there is no codegen step. If
  firmware loads the blob from mutable storage (flash partition / SD / OTA
  channel) instead of a compiled-in `const` array, both weights *and*
  architecture changes ship as data — bounded by the ops-registered and
  arena-size caveats above. Someone must build the loader once: partition,
  CRC/version check, fallback to known-good model, arena-fit check.
  `gridware-bytes` already has the right primitives (`app_storage`, `blob`,
  `ota`).
- Both frameworks' runtime cost is **once per image**, not per model. Marginal
  cost per additional model ≈ weights (+ small graph code for ST) + activation
  arena (shareable across models that never run concurrently). Kernel/op code
  grows only when a model first uses a new layer/op type.

---

## 5. OTA delivery reality in gridware-bytes (the hard constraint)

Investigated 2026-08-05/06, including Slack thread
"Pushing ODA model weights through configs"
(<https://gridwarehq.slack.com/archives/C09K95MGYKH/p1779403245643599>, 2026-05-21).

There are exactly **two** cloud→device data mechanisms; nothing in between:

1. **Config overlay update** (`ConfigOverlayUpdateRequest`,
   `proto/cloud_message.proto:213`):
   - `config_data` capped at **2040 bytes** (zlib-compressed overlay);
     decompressed encoded config also bounded (~one flash page).
   - **Full-replacement semantics** — the device zero-initializes and decodes
     the complete overlay; there is no merge/patch path in firmware.
   - **No chunking** for configs.
   - The 2040 B budget is shared by *all* config fields (the existing forest
     fields already use 600 B).
   - On "partial configs" (Maggie, Slack thread): the new cloud-side CMS
     relaxes partial *authoring*, but the wire protocol still sends the whole
     overlay — no size relief on the device side. (Status as of May 2026;
     re-check with Maggie Lonergan if this becomes decision-relevant.)
2. **Full firmware image OTA** (`proto/firmware_update.proto`): chunked
   4096-byte pages, signed, written to the inactive A/B flash bank, validated,
   rebooted into. Arbitrary size, but the delivery unit is a whole signed
   FW image. Supports **delta compression** (`DeltaCompression`, line ~110).

Consequences:

- **Weights-via-config is viable only for tiny models** (realistically ≤~1 KB
  of weights after sharing the overlay). The 28-byte toy model: fine. A real
  NN (e.g. ml_energization ≈ 3344 weights ≈ 13.4 KB) is ~6× over the cap —
  blocked. This matches the Slack thread's conclusion (Alex Berrian, Keyu
  Chen: "impossible because of the size limit").
- **This kills the near-term no-FW-update story for both ST and TFLM equally.**
  The §4 distinction (weights-only vs graph-as-data) only matters once a
  transport exists for KB–tens-of-KB blobs. Today, weights get baked into the
  FW image under either runtime.
- **Long-term fix:** a chunked "model blob" channel — essentially a model
  analog of the firmware page-transfer path, plus a mounted flash region.
  This is the "fundamental redesign" Keyu referenced. If it gets built, Ben
  Blasdell's expectation (ODA iterations change structure, not just weights)
  argues for a graph+weights blob format (TFLM-style) over weights-only
  (ST-style).
- **Pragmatic middle path — delta FW OTA:** if weights sit in their own
  contiguous linker section, a weights-only retrain changes only that section,
  and the existing delta-OTA mechanism transmits roughly weights-sized diffs
  **with no new protocol**. Still formally a FW release (signing, validation,
  release train) — solves bandwidth, not process decoupling. Combining this
  with a lighter-weight release process for weight-only deltas (Ben Blasdell's
  "ODA patch release" idea) may be the realistic near-term compromise.

**Design rule adopted:** do not couple the inference integration to the config
system. The C++ wrapper's init API should treat the model source (compiled-in
array today, mounted blob someday) as an implementation detail, so a future
transport redesign doesn't touch inference code.

---

## 6. What migrates into gridware-bytes

See `07_ml_oda_pipeline_summary.md` §"Which files migrate" for the full
categorized table (per-model generated files vs shared runtime vs
adapt-don't-copy vs do-not-migrate). Summary:

- Per model: `network.c/h`, `network_data.c/h`, `network_weights.c/h`,
  `network_details.h` (regenerated each model rev).
- Shared (Option A only): `NetworkRuntime1200_CM4_GCC.a` + `Middlewares/ST/AI/Inc`
  headers → Meson subproject.
- Rewrite, don't copy: `app_x-cube-ai.c` harness → a proper gridware-bytes
  module in the existing service/task model.
- Never: Core/ board bring-up, Drivers/ HAL+CMSIS, linker scripts, .ioc.

---

## 7. Open questions

1. Flash/RAM/cycles comparison, toy model, Option A vs B (the decision gate).
2. FW engineer's position: own a TFLM source dependency vs an ST binary-blob
   dependency (incl. toolchain pinning for the `.a`).
3. ~~Where does the wrapper live — new service vs library module?~~
   **Mostly answered (2026-08-06):** existing ML detectors (ml_plc,
   ml_energization, zdlb, major_haz) all live as plain C++ classes under
   `firmware/src/services/athena/ml_detectors/`, called synchronously from
   athena's task (e.g. `MlPlcDetector::Process()` → private `Predict()`,
   weights compiled in as versioned files under `ml_detectors/models/`).
   Follow that pattern: new detector class in `ml_detectors/`, with
   `Predict()` backed by the ST runtime instead of hand-written math.
   Remaining check with FW engineer: athena's per-cycle time budget — a
   dedicated inference task only becomes worth discussing if NN inference
   time grows long enough to disrupt athena's processing loop.
4. Will ODA iterations ever need *different-architecture* updates without a FW
   release? (Ben Blasdell expects structure changes; Keyu confirmed XGBoost
   weight-only is feasible today but NN needs more work.)
5. Practical ceiling of the config overlay channel and status of the new CMS
   (ask Maggie Lonergan / FW team).
6. If/when a model-blob OTA channel is designed: format (graph+weights vs
   weights-only), signing, fallback semantics.
7. ~~Does ST ship an x86 NetworkRuntime usable outside the validation
   tool?~~ **Answered (2026-08-07): yes** — static x86 libs + headers ship
   in the pack and linking them into a host C application is a documented,
   supported flow (see §3 host-testing plan). Remaining sub-question for
   the FW engineer: pinning the ST Edge AI Core pack version in CI so the
   x86 library version matches the ARM library/generated code version.

---

## 8. Log

- **2026-08-05** — Surveyed gridware-bytes firmware tree; identified Meson
  subproject pattern, native build, nanopb config system, forest-weights
  precedent. Framed Options A/B/C.
- **2026-08-05** — Clarified weights-as-data parity (ST external-weights ≈
  TFLM blob for same-architecture updates); graph-as-data remains TFLM-only.
- **2026-08-06** — Verified config channel hard limits in proto (2040 B,
  full-replacement, no chunking); read Slack thread; enumerated both OTA
  mechanisms; identified delta-FW-OTA middle path. Adopted design rule:
  decouple inference wrapper from model delivery.
- **2026-08-06** — Decided Option A first. Found the placement precedent:
  ML detectors run library-style inside the athena service
  (`athena/ml_detectors/`); new ST-backed detector should follow that
  pattern.
- **2026-08-06** — Added host (native x86) testing plan: minimal
  ARM-only surface behind a `Predict()` seam; host backend strategy order =
  ST x86 runtime (verify) → playback stub → reference implementation;
  golden vectors as shared test data.
- **2026-08-07** — Confirmed from official ST Edge AI documentation that
  the x86 host runtime exists and is supported for standalone C builds /
  CI (static libs in the pack, `validate --mode host` default flow,
  `--dll` + AiRunner for Python). Host-vs-target is tolerance-based, not
  bit-exact. Q7 closed.
