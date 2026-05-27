"""Tests for paths module."""

import os
from pathlib import Path
from unittest.mock import patch

from chess_detector.data.paths import data_dir, models_dir


def test_data_dir_default():
    """Test data_dir function with default environment."""
    with patch.dict(os.environ, {}, clear=True):
        assert data_dir() == Path("dataset").resolve()


def test_data_dir_env_var():
    """Test data_dir function with environment variable."""
    with patch.dict(os.environ, {"CHESS_DETECTOR_DATA_DIR": "/tmp/custom_data"}):
        assert data_dir() == Path("/tmp/custom_data").resolve()


def test_models_dir_default():
    """Test models_dir function with default environment."""
    with patch.dict(os.environ, {}, clear=True):
        assert models_dir() == Path("models").resolve()
