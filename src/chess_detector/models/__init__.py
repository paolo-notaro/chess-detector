"""Neural network architectures for chess move prediction."""

from chess_detector.models.diff import (
    ChessMoveModel,
    ConvPatchEncoder,
    MoveScorer,
    ResnetPatchEncoder,
)

__all__ = [
    "ChessMoveModel",
    "ConvPatchEncoder",
    "MoveScorer",
    "ResnetPatchEncoder",
]
