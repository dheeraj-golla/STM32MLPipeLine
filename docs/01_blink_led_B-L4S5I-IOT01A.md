# Blink an LED on the B-L4S5I-IOT01A — Pipeline Walkthrough

**Goal:** Compile a simple C program, flash it to the dev board over USB, and see an LED blink. This teaches you the full build → flash pipeline you'll use for all future ML firmware work.

**Board:** B-L4S5I-IOT01A (MCU: STM32L4S5VIT6, Cortex-M4)  
**Tools needed:** STM32CubeIDE (already installed)

---

## What you're actually learning here

| Step | Concept |
|---|---|
| CubeMX pin config | Telling the MCU which physical pin does what |
| Build | C source → `.elf` binary (ARM cross-compiler) |
| Flash | Copying `.elf` into the MCU's flash memory over SWD via ST-LINK |
| Run | MCU executes your code |

The LED we'll blink is **LED2**, connected to pin **PB14** on the STM32L4S5VIT6 (from UM2708 Table 6).

---

## Step 1 — Connect the board

Plug in the **Micro-B USB cable** to the port labeled **"USB STLINK"** — this is the connector in the **top-right corner** of the board (CN7, next to the "USB ST-LINK" silkscreen label).

> ⚠️ There are two USB ports on this board. The other one (CN9, labeled "USB OTG") is for data — don't use that one for flashing.

When connected, your computer should detect it as an ST-LINK device. On Windows you may hear the device-connected sound.

---

## Step 2 — Create a new project in STM32CubeIDE

1. Open **STM32CubeIDE**
2. Go to **File → New → STM32 Project**
3. The "Target Selection" window opens. In the search bar at the top, type:
   ```
   STM32L4S5VIT6
   ```
4. Select **STM32L4S5VIT6** from the results list (it should be the only one)
5. Click **Next**
6. Name your project: `blink_led`
7. Leave all other settings as defaults (C language, Executable, STM32Cube)
8. Click **Finish**
9. If prompted to open the CubeMX perspective, click **Yes**

CubeMX (the pin configurator) will open automatically.

---

## Step 3 — Configure the GPIO pin in CubeMX

CubeMX shows a diagram of the STM32 chip with all its pins. You need to configure pin **PB14** as an output so you can drive the LED.

1. In the chip diagram, find pin **PB14**
   - Tip: Use **Ctrl+F** (or the search icon) and type `PB14` to highlight it
2. **Left-click** on PB14 → a dropdown appears → select **GPIO_Output**
   - The pin will turn green
3. **Right-click** on PB14 → select **Enter User Label** → type `LED2` → press Enter
   - This creates a named constant `LED2_Pin` and `LED2_GPIO_Port` in the generated code, so your code is readable

That's all the hardware configuration needed. No clock changes, no other peripherals.

---

## Step 4 — Generate the C code

1. Click **Project → Generate Code** (or press `Alt+K`)
   - If prompted to download the STM32L4 firmware package, click **Yes** and wait for it to download (~200 MB, one-time)
2. When generation is complete, a dialog asks "Do you want to open the C/C++ perspective?" — click **Yes**

CubeIDE now shows your project with generated source files.

---

## Step 5 — Write the blink code

In the **Project Explorer** panel (left side), navigate to:
```
blink_led → Core → Src → main.c
```
Double-click `main.c` to open it.

Find this comment block inside the `main()` function (around line 90):
```c
/* USER CODE BEGIN WHILE */
  while (1)
  {

    /* USER CODE END WHILE */
```

Add your code **between** `/* USER CODE BEGIN WHILE */` and `/* USER CODE END WHILE */` so it looks like this:

```c
/* USER CODE BEGIN WHILE */
  while (1)
  {
    HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
    HAL_Delay(500);
    /* USER CODE END WHILE */
  }
```

> **What this does:**
> - `HAL_GPIO_TogglePin` flips PB14 between high (3.3V, LED on) and low (0V, LED off)
> - `HAL_Delay(500)` waits 500 milliseconds
> - Result: LED blinks at 1 Hz (on 500ms, off 500ms)

> **Why write inside the USER CODE comments?**  
> CubeMX regenerates `main.c` if you change pin config later. It only preserves code inside `/* USER CODE BEGIN */` ... `/* USER CODE END */` blocks. Code outside those blocks gets deleted on regeneration.

Save the file: **Ctrl+S**

---

## Step 6 — Build the project

Go to **Project → Build Project** (or press **Ctrl+B**).

Watch the **Console** panel at the bottom. A successful build ends with something like:
```
Finished building: blink_led.elf
Build Finished. 0 errors, 0 warnings.
```

The output file `blink_led.elf` is in the `Debug/` folder of your project. This is the compiled binary in ELF format — it contains your machine code plus debug symbols and memory layout information.

**If you see errors:** The most common cause is a missing firmware package. Go to **Help → Manage Embedded Software Packages**, find `STM32L4`, install the latest version, then rebuild.

---

## Step 7 — Flash and run

Make sure your board is still connected via USB ST-LINK.

Go to **Run → Run** (or press **F11**).

CubeIDE will:
1. Build the project (if not already built)
2. Connect to the board over SWD (the 2-wire debug interface running through the ST-LINK USB connection)
3. Write `blink_led.elf` into the STM32's internal flash memory
4. Reset the MCU and start running your code

**LED2 (the green LED on the board) should now blink once per second.**

If a "Edit launch configuration" dialog pops up, just click **OK** — the defaults are correct.

---

## Step 8 — Verify with STM32CubeProgrammer (optional but useful)

This step shows you the flashing tool directly — relevant because it's what you'll use in more advanced workflows.

1. Open **STM32CubeProgrammer**
2. In the top-right, select **ST-LINK** as the connection type
3. Click **Connect**
4. You should see the MCU recognized: `STM32L4S5VITx` with its flash memory map
5. Go to the **Memory & File edition** tab — you can see the contents of flash at address `0x08000000` (this is where your code was written)
6. Click **Disconnect** when done

The LED keeps blinking — the MCU runs independently once flashed. Power the board from any USB source and it will run your code automatically.

---

## What just happened — the full pipeline

```
main.c  (your C source)
    ↓  arm-none-eabi-gcc (ARM cross-compiler, bundled in CubeIDE)
blink_led.elf  (compiled binary with debug info)
    ↓  STM32CubeProgrammer / ST-LINK / SWD
STM32L4S5 flash memory at 0x08000000
    ↓  MCU reset handler → main()
LED blinks
```

This same pipeline — write code, build `.elf`, flash via SWD — is what you'll use when deploying ML inference code generated by STM32Cube.AI. The ML step just adds a code-generation stage before the build.

---

## Next steps

Once this is working, the natural progression is:

1. **Add UART printf** — so you can print values from the MCU to your computer's serial terminal. Essential for verifying ML inference outputs ("verifiability" requirement).
2. **Run a Cube.AI hello-world** — generate C code from a simple Python model using AI Studio, drop it into a CubeIDE project, build and flash it.
3. **Read sensor data** — the board has a temperature/humidity sensor (HTS221) and accelerometer (LSM6DSL) on I2C2. These are the kinds of inputs your ML models will process.
