# chess-detector

Detect chess moves from camera images of a physical board.

`chess-detector` trains a small CNN on the *diff* between two top-down board
photos (before and after a move) and predicts the played move in UCI notation.
A complete pipeline is included: a synthetic dataset is generated from Lichess
PGNs by rendering Blender scenes, the model is trained with PyTorch + MLflow,
and predictions are exposed through a CLI, a Flask HTTP API and a Tkinter demo
that pulls frames from an IP camera.

> A longer write-up is available in [`docs/2025-04-27-chess-tracker.md`](docs/2025-04-27-chess-tracker.md).

---

## Repository layout

```
chess-detector/
├── src/chess_detector/        # the importable Python package
│   ├── data/                  # dataset generation, loading, image preprocessing
│   ├── models/                # PyTorch architectures (diff + image-pair baseline)
│   ├── training/              # training loops and metric helpers
│   ├── inference/             # CLI prediction
│   └── api/                   # Flask HTTP API
├── demos/                     # runnable end-user demos (e.g. phone camera UI)
├── scripts/                   # one-off / experimental scripts
├── assets/chess_board/        # Blender scene + textures used by the renderer
├── docs/                      # design notes, blog article
├── tests/                     # placeholder for tests
├── pyproject.toml             # Poetry project + ruff + ty configuration
└── .github/workflows/ci.yml   # lint, format and type-check gate
```

The package follows the standard `src/` layout, so it can be installed with
Poetry and imported as `chess_detector`.

---

## Requirements

- **Python ≥ 3.11** (Blender's `bpy` module requires *exactly* 3.11).
- **[Poetry](https://python-poetry.org/)** for dependency management.
- **[Blender ≥ 4.3](https://www.blender.org/download/)** only if you want to
  regenerate the synthetic dataset. Set `CHESS_DETECTOR_BLENDER` to the
  blender executable path if it is not at the default location.
- A CUDA-capable GPU is recommended for training.

## Installation

```bash
git clone https://github.com/lorenzonotaro/chess-detector
cd chess-detector
poetry config virtualenvs.in-project false
poetry config virtualenvs.path "$HOME/venvs"
poetry install                 # core dependencies + dev tools
poetry install --extras dataset # add `bpy` for dataset generation (Python 3.11 only)
```

With the Poetry config above, environments are created under `$HOME/venvs`
instead of inside this repository. Poetry names the environment automatically
from the project name and path, so machine-specific paths stay out of
`pyproject.toml`.

---

## Quickstart

### 1. Generate the synthetic dataset (optional, only if training from scratch)

```bash
poetry run chess-detector-gen-dataset
```

By default this writes everything under `./dataset/` (configurable via
`CHESS_DETECTOR_DATA_DIR`). It downloads a PGN file from Lichess, samples a
balanced set of positions and renders each one twice (before / after move) via
Blender, then produces the diff images used by the diff model.

### 2. Train

```bash
# Diff-based model (recommended)
poetry run chess-detector-train-diff

# Image-pair baseline (legacy)
poetry run chess-detector-train-pair
```

Both commands log to MLflow under the experiment name `ChessMovePrediction`
and save checkpoints to `models/`.

### 3. Predict

```bash
poetry run chess-detector-predict \
    --checkpoint models/best.pth \
    --before path/to/before.png \
    --after  path/to/after.png \
    --preprocess
```

`--preprocess` warps the perspective and converts to grayscale. Omit it if
your inputs are already 224×224 normalized board crops.

### 4. HTTP API

```bash
poetry run chess-detector-api
```

Starts a Flask server on `0.0.0.0:5000`. See
[`src/chess_detector/api/http.py`](src/chess_detector/api/http.py) for the
endpoints (`/session/begin`, `/predict`, …).

### 5. Phone-camera demo

```bash
export CHESS_DETECTOR_CAMERA_URL="http://<phone-ip>:8080/photo.jpg"
export CHESS_DETECTOR_CHECKPOINT="models/your-checkpoint.pth"
poetry run chess-detector-demo-phone
```

Streams snapshots from an IP camera (e.g. Android *IP Webcam*), detects the
board once at start-up, then captures move-by-move and overlays the predicted
move on a Tkinter UI.

---

## Development workflow

The repository ships an opinionated quality gate built on
[Astral](https://astral.sh)'s tooling.

```bash
poetry install              # install dev dependencies (ruff, ty)
poetry run ruff check .     # lint
poetry run ruff format .    # format
poetry run ty check         # static type check
```

The same three commands run on every push and pull request via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Conventions

- **Line length 100**, double-quoted strings, `ruff format` style.
- **Type-annotate public functions.** `ty` is configured with default
  strictness; warnings are tolerated, errors block merging.
- **One concern per module.** Cross-package imports go through the public
  sub-package surface (`chess_detector.data`, `chess_detector.models`, …).
- **No top-level side-effects** in modules used as console-script targets:
  wrap everything in `main()`.

---

## Models at a glance

| Model | Module | Input | Output |
|---|---|---|---|
| `ChessMoveModel` (diff) | [`models/diff.py`](src/chess_detector/models/diff.py) | 64 patches of a single diff image | `[64×64]` from→to logits |
| `ChessMovePredictor` (pair, legacy) | [`models/pair.py`](src/chess_detector/models/pair.py) | Before + after images | from / to / promotion logits |

The diff model is the recommended approach; the pair-based one is kept as a
baseline reference.

---

## License

MIT.
