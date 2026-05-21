# Keras → C Conversion: First Pass with ST AI Studio

Notes and orientation for the next phase of the STM32 ML pipeline work:
converting the reference Keras classifier (`example_keras_classifier/`) to
C/C++ using ST Edge AI Studio on a Windows machine.

## Context

- **Input artifact**: `example_keras_classifier/outputs/model.keras`
  (trained Dense(1, sigmoid) model with 7 parameters)
- **Target hardware**: STM32L4S5VI, Cortex-M4F (no Helium, no NPU,
  CMSIS-NN supported for int8 ops)
- **Tool**: ST Edge AI Studio (Windows). Uses ST Edge AI Core under the
  hood — same engine as the browser-based ST Edge AI Developer Cloud
  and the legacy X-CUBE-AI desktop plugin.
- **Reference test vectors**: `example_keras_classifier/outputs/test_vectors.json`
  contains 254 input/output pairs across three categories (random,
  edge-near-boundary, boundary-extreme) for downstream bit-exactness
  checks on the MCU.

## What to pay attention to during the first pass

These five questions inform framework design decisions later. Don't
just generate C and move on — answer each one.

### 1. How are the weights stored in the generated C?

Look at the generated header/source files. Specifically:

- `const float weights[] = {...}` — weights baked into the firmware
  binary as constants in Flash. Default for AI Studio.
- `const int8_t weights[] = {...}` — same, but quantized.
- Are they placed in a specific linker section (e.g., `__attribute__((section(".nn_weights")))`)?
  This matters for the configurability requirement (#8): if weights
  are in their own section, you can swap them without rebuilding the
  whole firmware. If they're in `.rodata` along with everything else,
  swapping = rebuilding.

**Why this matters**: requirement #8 ("change parameters without
needing FW revision") cannot be satisfied if weights are inline
`const` arrays in the default `.rodata` section. Architecting around
this is a separate engineering task.

### 2. What runtime library does it require?

AI Studio doesn't generate self-contained C. The generated code links
against a precompiled ST runtime library, typically named something
like `NetworkRuntime.lib` or `libai_runtime_*.a`.

Find out:

- Where does the library live? (Inside the AI Studio install, or
  copied into the generated package?)
- What's its size in bytes? (Adds to your Flash budget.)
- Is the source available, or is it a binary blob?
- Does it have its own dependencies (CMSIS-DSP, libm)?

**Why this matters**: requirement #2 (maintainability). A binary-blob
runtime is a dependency you don't own. If ST deprecates or changes
the API in a future release, your firmware breaks and you can't patch
it directly.

### 3. Does the generated code call CMSIS-NN?

Look for `arm_*` function calls in the generated source:

- `arm_fully_connected_s8(...)` — int8 Dense layer via CMSIS-NN. Good.
- `arm_nn_mat_mult_*` — int8 matmul via CMSIS-NN. Good.
- Plain `for` loops doing the matmul in C — unaccelerated. Bad on M4.

For our 7-parameter toy model, this won't matter for performance —
the model is too small for CMSIS-NN's overhead to pay off. But it
indicates what the *pipeline* will do for larger models.

**Why this matters**: requirement #6 (acceleration). On Cortex-M4
without Helium, CMSIS-NN is the only meaningful acceleration option.
If AI Studio is emitting plain C for some op types, those ops are
slow.

### 4. What's the C API for invoking the model?

The firmware team needs to call something like:

```c
ai_error err = aiInit();
ai_run(input_buffer, output_buffer);
```

Find out the actual function names, the exact signatures, what status
codes / errors look like, and how input/output buffers are typed
(float? int8? quantized with separate scale/zero_point?).

**Why this matters**: this is the *artifact contract* between your
pipeline and the firmware team. Every downstream framework decision
flows from it:

- Replay system input format (#4) depends on what the model accepts.
- "Ease of use by Applied Scientist" (#7) means hiding this C API
  behind a Python-friendly interface.
- Verifiability (#3) means comparing C-API output against Python output
  bit-for-bit (or within tolerance) for every test vector.

### 5. Does AI Studio provide on-target validation?

ST has a feature called "Validate on target" — runs the converted
model on the actual MCU and compares output against a reference run
on host. This partially solves requirement #3 (verifiability) for
free, if it works for our model.

Find out:

- Does it work for our Dense(1, sigmoid) model? (Should be trivial,
  but check.)
- Does it report end-to-end output diffs only, or per-layer
  intermediates?
- What tolerance does it use? (Bit-exact? Relative error? Absolute
  error?)
- Can you feed it our `test_vectors.json` directly, or does it use
  its own random inputs?

**Why this matters**: if Validate on Target works well, a chunk of
your verifiability tooling is solved by ST. If it doesn't (e.g., only
reports end-to-end, doesn't allow custom inputs, uses tolerance
incompatible with your needs), you'll need to build your own
verification layer.

## Practical suggestions

- **Take notes as you go.** Create a new doc when you start —
  something like `docs/03_ai_studio_first_pass.md`. Capture:
    - What you clicked/configured
    - The generated C code (or a representative slice)
    - File sizes for float vs int8 builds
    - Any error messages or surprises
- **Don't optimize early.** First pass is "does the default workflow
  produce something that runs?" Worry about quantization tuning,
  custom layer support, and code-size optimization only after the
  default flow works.
- **Compare float and int8 builds.** Generate both. Note the size
  difference, the API difference, and any accuracy difference vs the
  Keras reference.
- **Keep the test vectors handy.** Once you have generated C, the
  first real test is: feed the inputs from `test_vectors.json` to
  the generated model and check the outputs match. This is your
  bit-exactness baseline.

## What to bring back for the next design conversation

When the first pass is done, bring:

1. The generated C source (or a sample) so we can read it together.
2. Answers to the five questions above.
3. Any surprises or sticking points — those usually expose the most
  important framework decisions.

The next conversation is: **does ST's pipeline output satisfy our
eight requirements, and where does our framework need to wrap or
extend it?**

## Reminder: the eight requirements

1. Reduce model size (quantization)
2. Maintainability
3. Verifiability (intermediate calculations)
4. Compatible with replay system
5. Reduce power consumption
6. Reduce compute resources (optimization and acceleration)
7. Ease of use by Applied Scientist (no need to know C/C++)
8. Configurability (change parameters without needing FW revision)
