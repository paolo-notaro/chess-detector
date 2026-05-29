"""Tests for dataset preprocessing helpers."""

import cv2
import numpy as np
import pytest
import torch

from chess_detector.data.chess_utils import from_uci
from chess_detector.data.dataset import ChessMoveFromDiffDataset


def test_patch_image_returns_64_single_channel_patches():
    """Test patch extraction shape and deterministic board-square order."""
    img = np.arange(64, dtype=np.float32).reshape(8, 8)

    patches = ChessMoveFromDiffDataset.patch_image(img)

    assert patches.shape == (64, 1, 1, 1)
    assert patches.dtype == torch.float32
    assert patches[0, 0, 0, 0].item() == 63
    assert patches[-1, 0, 0, 0].item() == 0


def test_preprocess_image_normalizes_and_resizes_square_image():
    """Test preprocessing produces normalized float data with the requested size."""
    img = np.full((4, 4), 255, dtype=np.uint8)

    processed = ChessMoveFromDiffDataset.preprocess_image(img, preprocess_resize=8)

    assert processed.shape == (8, 8)
    assert processed.dtype == np.float32
    assert np.allclose(processed, 1.0)


def test_preprocess_image_preserves_shape_without_resize():
    """Test preprocessing normalizes in-place-sized square images."""
    img = np.array([[0, 255], [128, 64]], dtype=np.uint8)

    processed = ChessMoveFromDiffDataset.preprocess_image(img)

    assert processed.shape == (2, 2)
    assert processed.dtype == np.float32
    assert processed.min() == 0.0
    assert processed.max() == 1.0


def test_patch_image_rejects_non_square_images_and_bad_resize():
    """Test patch extraction validates its square-image and resize contracts."""
    with pytest.raises(AssertionError, match="Expected square image"):
        ChessMoveFromDiffDataset.patch_image(np.zeros((8, 16), dtype=np.float32))

    with pytest.raises(AssertionError, match="greater than 0"):
        ChessMoveFromDiffDataset.patch_image(np.zeros((8, 8), dtype=np.float32), resize_size=0)


def test_load_image_rejects_missing_file():
    """Test image loading fails clearly for missing diff images."""
    with pytest.raises(AssertionError, match="not found"):
        ChessMoveFromDiffDataset._load_image("/tmp/definitely-not-a-chess-diff.png")


def test_diff_dataset_getitem_returns_tensor_and_label_schema(tmp_path):
    """Test a CSV row and grayscale diff image become model-ready tensors and labels."""
    csv_path = tmp_path / "diff_entries.csv"
    image_dir = tmp_path / "diff"
    image_dir.mkdir()

    csv_path.write_text("id,move_uci\n0,e2e4\n")
    image = np.full((224, 224), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "0.png"), image)

    dataset = ChessMoveFromDiffDataset(str(csv_path), str(image_dir))
    diff_tensor, label = dataset[0]

    assert diff_tensor.shape == (64, 1, 32, 32)
    assert set(label) == {"from", "to", "promotion"}
    assert all(value.dtype == torch.long for value in label.values())
    assert (label["from"].item(), label["to"].item(), label["promotion"].item()) == from_uci("e2e4")
