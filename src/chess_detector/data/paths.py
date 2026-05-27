"""Path resolution helpers for the chess-detector data layout.

All dataset-related paths are resolved relative to ``CHESS_DETECTOR_DATA_DIR``
(default: ``./dataset``). Model checkpoints live under
``CHESS_DETECTOR_MODELS_DIR`` (default: ``./models``).
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Return the active dataset root, honoring ``CHESS_DETECTOR_DATA_DIR``."""
    return Path(os.environ.get("CHESS_DETECTOR_DATA_DIR", "dataset")).resolve()


def models_dir() -> Path:
    """Return the active checkpoint root, honoring ``CHESS_DETECTOR_MODELS_DIR``."""
    return Path(os.environ.get("CHESS_DETECTOR_MODELS_DIR", "models")).resolve()


def preprocessed_dir() -> Path:
    return data_dir() / "preprocessed"


def diff_dir() -> Path:
    return data_dir() / "diff"


def diff_real_dir() -> Path:
    return data_dir() / "diff_real"


def images_dir() -> Path:
    return data_dir() / "images"


def last_index_file() -> Path:
    return data_dir() / "last_index.txt"


def entries_file() -> Path:
    return data_dir() / "entries.csv"


def diff_entries_train_file() -> Path:
    return data_dir() / "diff_entries_train.csv"


def diff_entries_eval_file() -> Path:
    return data_dir() / "diff_entries_eval.csv"


def entries_train_file() -> Path:
    return data_dir() / "entries_train.csv"


def entries_eval_file() -> Path:
    return data_dir() / "entries_eval.csv"


def diff_real_metadata_file() -> Path:
    return data_dir() / "diff_real.csv"


def empty_boards_glob() -> str:
    """Glob pattern for empty calibration board images."""
    return str(images_dir() / "empty_board_*.png")
