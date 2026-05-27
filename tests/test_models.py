"""Tests for models module."""

import torch

from chess_detector.models.diff import ChessMoveModel, ConvPatchEncoder
from chess_detector.models.pair import ChessMovePredictor, SmallCNNEncoder


def test_diff_model_forward():
    # Use ConvPatchEncoder to avoid downloading ResNet weights on CI
    """Test ChessMoveModel forward pass."""
    model = ChessMoveModel(encoder_class=ConvPatchEncoder)
    model.eval()

    # Batch of 2, 64 patches, 1 channel, 32x32 size
    dummy_input = torch.randn(2, 64, 1, 32, 32)

    with torch.no_grad():
        scores = model(dummy_input)

    # Output should be pairwise move scores [Batch, From_Square, To_Square]
    assert scores.shape == (2, 64, 64)


def test_pair_model_forward():
    # Use SmallCNNEncoder for a fast, lightweight forward pass
    """Test ChessMovePredictor forward pass."""
    model = ChessMovePredictor(encoder_class=SmallCNNEncoder)
    model.eval()

    # Batch of 2, 1 channel, 224x224 size
    dummy_before = torch.randn(2, 1, 224, 224)
    dummy_after = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        from_logits, to_logits, promo_logits = model(dummy_before, dummy_after)

    assert from_logits.shape == (2, 64)
    assert to_logits.shape == (2, 64)
    assert promo_logits.shape == (2, 5)
