"""Tests for models module."""

import torch

from chess_detector.models.diff import ChessMoveModel, ConvPatchEncoder, MoveScorer
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


def test_conv_patch_encoder_respects_embedding_dim():
    """Test ConvPatchEncoder output preserves board squares and embedding dimension."""
    model = ConvPatchEncoder(embed_dim=37)
    model.eval()

    patches = torch.randn(3, 64, 1, 32, 32)

    with torch.no_grad():
        embeddings = model(patches)

    assert embeddings.shape == (3, 64, 37)


def test_move_scorer_outputs_pairwise_square_logits():
    """Test MoveScorer projects every from-square against every to-square."""
    scorer = MoveScorer(embed_dim=16, proj_size=8)
    scorer.eval()

    embeddings = torch.randn(4, 64, 16)

    with torch.no_grad():
        scores = scorer(embeddings)

    assert scores.shape == (4, 64, 64)
    assert scorer.temperature.shape == ()


def test_diff_model_positional_encoding_matches_board_and_embedding_dim():
    """Test ChessMoveModel positional encoding is one vector per board square."""
    model = ChessMoveModel(embed_dim=24, encoder_class=ConvPatchEncoder)

    assert model.positional_encoding.shape == (64, 24)


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
