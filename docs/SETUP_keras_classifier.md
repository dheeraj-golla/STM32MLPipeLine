# Setup: STM32 ML Pipeline (macOS Apple Silicon)

Step-by-step setup for the Python training environment. Assumes you already
have `pyenv` and `poetry` installed.

## Background: what these tools do

- **pyenv** manages Python *interpreters*. Lets you have multiple Python
  versions installed side-by-side (e.g., 3.10, 3.12, 3.14) and pick
  which one a given project uses.
- **Poetry** manages Python *packages and virtual environments*. Reads
  `pyproject.toml`, resolves all dependencies, writes the exact versions
  to `poetry.lock`, and installs everything into an isolated `.venv/`
  folder per project.

The division of labor: **pyenv picks which Python; Poetry picks which
libraries, in a sandbox bound to that Python.**

## Version constraints

- Python: **3.12.x** (TensorFlow 2.21 supports 3.10–3.13; 3.14 is not yet
  supported by TF — `poetry install` will fail on 3.14).
- TensorFlow: **2.21** (specified in `pyproject.toml`).
- Platform: macOS 12 (Monterey) or newer on Apple Silicon. Check with
  `sw_vers`.

---

## Step 1: Install Python 3.12 via pyenv

**Where to run this:** anywhere. `pyenv install` is a global operation
that stores Python at `~/.pyenv/versions/3.12.7/` regardless of your
current directory.

```bash
# See what's installed
pyenv versions

# Install 3.12.7 if not already there
pyenv install 3.12.7

# Confirm
pyenv versions
```

The install compiles Python from source — takes a few minutes. If it
fails on Apple Silicon, the usual fix is:

```bash
brew install openssl readline sqlite3 xz zlib tcl-tk
```

Then retry `pyenv install 3.12.7`.

## Step 2: Clone the repo

Pick a workspace directory:

```bash
cd ~/code   # or wherever you keep projects

git clone https://github.com/dheeraj-golla/STM32MLPipeLine.git
cd STM32MLPipeLine
```

(SSH variant: `git clone git@github.com:dheeraj-golla/STM32MLPipeLine.git`)

## Step 3: Pin Python 3.12 for this repo

```bash
pyenv local 3.12.7
python --version    # should print Python 3.12.7
```

This creates `.python-version` in the repo root. Commit it — it tells
anyone cloning the repo which Python to use.

## Step 4: Configure Poetry (one-time, global)

```bash
poetry config virtualenvs.in-project true
```

Tells Poetry to put `.venv/` inside each project folder rather than
buried in `~/Library/Caches/...`. Easier to find, easier to delete,
IDEs auto-detect them.

## Step 5: Create project structure

```bash
mkdir example_keras_classifier
```

## Step 6: Place the project files

After downloading from chat, the layout should be:

```
STM32MLPipeLine/
├── .gitignore                          # repo root
├── .python-version                     # created by `pyenv local`
├── README.md                           # repo root
├── pyproject.toml                      # repo root
└── example_keras_classifier/
    └── train_model.py
```

## Step 7: Install dependencies

From the repo root:

```bash
poetry install
```

What happens:
1. Poetry creates `.venv/` in the repo root.
2. Resolves all dependencies (TF pulls in ~50 transitive packages).
   First-time resolution takes 2–5 minutes.
3. Generates `poetry.lock` with exact versions of everything.
4. Installs everything into `.venv/`.

**Verify Poetry picked up the right Python:**

```bash
poetry env info
```

Look for `Python: 3.12.x`. If it shows anything else:

```bash
poetry env use 3.12.7
poetry install
```

## Step 8: Run the training

```bash
poetry run python example_keras_classifier/train_model.py
```

Expected output: 30 training epochs, then a summary, then:

```
Weights W (shape (6, 1)):
[0.20395342 0.20439447 0.18716294 0.19427767 0.23444569 0.21492276]
Bias b   (shape (1,)): [-6.6528425]

Noise-free test accuracy: 0.9590
Noisy val accuracy:       0.8760
```

If weights match exactly, your environment reproduces the reference setup.
If they differ in the last few digits, it's likely a different BLAS or
oneDNN setting — flag it but don't panic.

Outputs land in `example_keras_classifier/outputs/`:
- `model.keras`
- `model_weights.json`
- `test_vectors.json`

## Step 9: Commit and push

```bash
# Sanity check before committing
git status
```

You should see roughly:
```
new file:   .gitignore
new file:   .python-version
new file:   README.md
new file:   poetry.lock
new file:   pyproject.toml
new file:   example_keras_classifier/train_model.py
```

You should NOT see:
- `.venv/` anywhere
- `example_keras_classifier/outputs/...` (model artifacts)
- `__pycache__/` anywhere

If any of those appear, the `.gitignore` isn't being applied — usually
because it's in the wrong directory. It must be at the repo root.

```bash
git add .
git commit -m "Initial Keras reference classifier and Poetry setup"
git push
```

HTTPS pushes need a Personal Access Token (not your GitHub password —
password auth was disabled years ago). Generate one at GitHub →
Settings → Developer settings → Personal access tokens → Fine-grained
tokens.

---

## Common failure modes

**`poetry install` uses the wrong Python.**
Run `poetry env info`. If Python isn't 3.12.x:
```bash
poetry env use 3.12.7
poetry install
```

**You committed `.venv/`.**
Check `git status` before `git add .`. If `.venv/` shows up, the
`.gitignore` isn't being picked up. Verify it's in the repo root.

**You committed `outputs/` artifacts.**
Same check. The `outputs/` exclusion is in `.gitignore`.

**`CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)` when running
the script.**
Harmless. TF looks for an NVIDIA GPU, doesn't find one on Apple Silicon,
falls back to CPU. Ignore.

**Different weights than the reference.**
Save the output. Two cases:
- Different each run on your machine → seeding issue.
- Same across runs but different from reference → TF or BLAS version
  difference.

Both are worth investigating before trusting bit-exactness later.

---

## Daily workflow reminders

### When do I run `poetry install`?

Almost never. Specifically:

- **Once when you first clone the repo** — creates `.venv/` and installs
  everything from the lockfile.
- **After `git pull` if `poetry.lock` changed** — syncs your local
  `.venv/` to whatever versions a teammate added/updated.
- **If you ever delete `.venv/`** — to rebuild it.

That's it. You do NOT run `poetry install` every time you work in the
folder. Once `.venv/` exists, it persists.

### Running things day-to-day

```bash
cd ~/code/STM32MLPipeLine                                       # pyenv picks 3.12.7 automatically
poetry run python example_keras_classifier/train_model.py       # runs inside the venv
```

`poetry run <command>` executes `<command>` inside the venv without
activating it in your shell. No `source` needed.

### What about `source .venv/bin/activate`?

That's the old manual way of using Python's built-in `venv` module.
**You don't need it with Poetry.** `poetry run` handles activation
transparently.

If you really want a shell with the venv active (so you can type
`python` without the `poetry run` prefix):

- **Poetry 1.x**: `poetry shell`
- **Poetry 2.x**: `eval $(poetry env activate)`, OR install the shell
  plugin once with `poetry self add poetry-plugin-shell` and then use
  `poetry shell` like before.

Check your version: `poetry --version`.

### Adding a new dependency

Don't hand-edit `pyproject.toml`. Use:

```bash
poetry add scikit-learn
```

This updates `pyproject.toml`, regenerates `poetry.lock`, and installs
the package atomically. Hand-editing `pyproject.toml` without re-locking
causes drift between the two files.

### After someone else pushes a dependency change

```bash
git pull
poetry install   # syncs .venv to match the new poetry.lock
```

