# Embedded ML Dev — Session Summary (Day 1 & 2)

## What we built

### 1. Blink LED on B-L4S5I-IOT01A
- Created a CubeIDE project correctly from CubeMX standalone (board selector → B-L4S5I-IOT01A → initialize all peripherals)
- Configured PB14 (LED2) as GPIO output
- Built with 0 errors, 0 warnings
- Flashed over SWD via ST-LINK using CubeIDE Run

### 2. Accelerometer Pipeline
- Read raw XYZ data from LSM6DSL over I2C2 (hi2c2, address 0x6A)
- Computed L2 norm of XYZ in float32 (hardware FPU accelerated)
- Printed values over UART1 (huart1, PB6/PB7 → ST-LINK VCP) at 115200 baud
- Threshold gate: LED2 blinks when norm > 1200 mg (tap/shake detection)
- Viewed output in PuTTY on COM4

### 3. Repo Setup
- Created `STM32MLPipeLine` on GitHub (private, under dheeraj-golla)
- Structure:
  ```
  STM32MLPipeLine/
  ├── projects/
  │   ├── blink_led/Core/Src|Inc + .ioc
  │   └── accel_pipeline/Core/Src|Inc + .ioc
  ├── ml_models/
  │   ├── zdlb/xgbmodel_v0_7_4.h
  │   └── accel_gesture/model_generated/
  ├── docs/
  │   ├── 01_blink_led_B-L4S5I-IOT01A.md
  │   └── 02_accelerometer_pipeline_B-L4S5I-IOT01A.md
  └── tools/
  ```
- `.gitignore` excludes `Debug/`, `Drivers/`, `.elf`, `.o`, `.map` etc.

---

## Concepts covered

### Toolchain & compilation
- `arm-none-eabi-gcc` is a cross-compiler: same GCC, targets ARM Thumb-2 instead of x86
- Full pipeline: `.c` → [compiler] → `.s` → [assembler] → `.o` → [linker] → `.elf`
- CubeIDE splits compile and link into separate phases for incremental builds (Make)
- Compile flags: `-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard`
- Build commands live in `Debug/Core/Src/subdir.mk`, orchestrated by `Debug/makefile`

### Memory layout
- `.text` — machine code, lives and executes from Flash at `0x08000000`
- `.data` — initialized globals, stored in Flash, copied to RAM at startup
- `.bss` — uninitialized globals, zeroed in RAM at startup, costs zero Flash
- `.rodata` — constants, Flash only (prefer `const` over mutable globals)
- Linker script (`STM32L4S5VITX_FLASH.ld`) defines exact memory regions and addresses
- Stack grows downward from top of RAM; heap grows upward from above `.bss`
- `startup_stm32l4s5vitx.s` runs before `main()`: sets stack pointer, copies `.data`, zeroes `.bss`

### HAL and CubeMX
- CubeMX generates HAL initialization code (`MX_GPIO_Init`, `MX_I2C2_Init` etc.)
- Always write user code inside `/* USER CODE BEGIN */` ... `/* USER CODE END */` blocks — CubeMX preserves these on regeneration
- HAL handles are named by peripheral: `hi2c2`, `huart1`, `hspi1` etc.

### Printf and newlib-nano
- `stdio.h` is declarations only — no code
- Actual `printf` implementation lives in `libc.a` (newlib-nano by default)
- newlib-nano strips float formatting to save ~15 KB Flash
- `-u _printf_float` forces linker to pull in float-capable printf
- `__io_putchar` hooks printf output to UART transmit

### FPU and math
- Cortex-M4 FPU (FPv4-SP) natively accelerates: `+`, `-`, `*`, `/`, `sqrtf`, `fabsf`, `fmaf`
- `sqrtf(x)` compiles to a single `VSQRT.F32` instruction — no libm needed
- `expf`, `logf`, `sinf` — no hardware instruction, computed via software polynomial in libm (slow, costs Flash)
- ZDLB inference uses `expf` for softmax (2 calls per inference) — optimization candidate

### Peripheral addressing
- I2C2 is the second I2C hardware controller on the STM32L4S5 — a separate silicon block, not just the protocol name
- All onboard sensors (LSM6DSL, LIS3MDL, LPS22HB, HTS221, VL53L0X) share the I2C2 bus, each with a unique 7-bit address
- USART1 is the ST-LINK VCP — confirmed from UM2708 Figure 2 and MSP init pin labels

### Gridscope hardware context
- **Hamilton:** STM32L4A6ZGT6, 1 MB Flash (nearly full), 320 KB RAM, 2 MB external QSPI SRAM
- **Diablo (next gen):** STM32L4R9ZIY6P, 2 MB Flash, 640 KB RAM, external QSPI SRAM
- Dev board (STM32L4S5VIT6) matches Diablo internal memory spec but has no external SRAM
- Flash capacity on Hamilton is the primary constraint driving quantization work

### ZDLB XGBoost model analysis
- `xgbmodel_v0_7_4.h`: 118 trees, 1564 nodes, 1682 leaves
- Index arrays (`root_idx_`, `left_idx_`, `right_idx_`, `threshold_idx_`) already `int16_t` — smart
- Float arrays: `xgb_thresholds_` (1564 × 4B = 6.25 KB) + `xgb_weights_` (1682 × 4B = 6.73 KB) = ~13 KB
- Quantizing both to `uint16_t` saves ~6.5 KB Flash
- Weights range: ~-0.075 to +0.075; thresholds have different range — need separate scale/offset per array
- Dequantization code will be `.cpp` to match existing C++ codebase (`namespace athena`)

### Git workflow
- Rebase vs merge: rebase replays your commits on top of latest main (linear history); merge creates a merge commit (preserves exact history)
- `git pull --rebase` set as default in this repo to match Gridware's likely convention
- gridware-bytes workflow: branch → commits → rebase on main → PR → review → merge
- Never commit directly to main on shared repos; your personal repo is fine for now

---

## Next steps (in order)

1. **Verify accelerometer UART output on Mac** — confirm serial terminal works cross-platform
2. **Cube.AI hello-world** — convert a simple TFLite/PyTorch model to C using AI Studio, integrate into CubeIDE project, flash and run inference on dev board
3. **ZDLB quantization** — design Python script (`quantize.py`) to convert float32 → uint16 with per-array scale/offset, and C++ dequantization code to integrate into gridware-bytes
4. **Check `gridware-data` repo access** — may contain relevant data pipeline tooling

---

## Key files

| File | Location | Purpose |
|---|---|---|
| `main.c` | `projects/blink_led/Core/Src/` | Blink + accelerometer pipeline |
| `main.h` | `projects/blink_led/Core/Inc/` | Pin definitions (LED2_Pin etc.) |
| `blink_led.ioc` | `projects/blink_led/` | CubeMX config — regenerates entire project |
| `xgbmodel_v0_7_4.h` | `ml_models/zdlb/` | ZDLB XGBoost inference engine |
| `01_blink_led_B-L4S5I-IOT01A.md` | `docs/` | Step-by-step blink guide |
| `02_accelerometer_pipeline_B-L4S5I-IOT01A.md` | `docs/` | Accelerometer pipeline guide |
