"""Tests for chess_utils module."""

import pytest
import torch

from chess_detector.data.chess_utils import (
    SQUARE_TO_IDX,
    argmax_2d_indices_batch,
    best_valid_move_from_logits,
    from_uci,
    get_board_id,
    get_legal_moves,
    is_move_valid,
    is_path_fenlike,
    to_uci,
    topk_valid_moves_from_logits,
)


def test_to_uci():
    # a1 is 0, e2 is 12, e4 is 28, h8 is 63
    """Test to_uci function."""
    assert to_uci(12, 28) == "e2e4"
    assert to_uci(0, 63, 1) == "a1h8q"  # 1 is Queen promotion


def test_from_uci():
    """Test from_uci function."""
    assert from_uci("e2e4") == (12, 28, 0)
    assert from_uci("a1h8q") == (0, 63, 1)


def test_is_move_valid_no_fen():
    # Without FEN, it just checks that from_idx != to_idx
    """Test is_move_valid function without FEN."""
    assert is_move_valid(0, 1) is True
    assert is_move_valid(0, 0) is False


def test_is_move_valid_with_fen():
    """Test is_move_valid function with FEN."""
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # e2e4 is a valid opening move for white
    assert is_move_valid(12, 28, board_fen=start_fen, turn="w") is True
    # a1a8 is an invalid move (rook blocked/can't jump)
    assert is_move_valid(0, 56, board_fen=start_fen, turn="w") is False


def test_argmax_2d_indices_batch_returns_coordinates_and_confidence():
    """Test batch argmax conversion from flattened logits to square pairs."""
    logits = torch.zeros(2, 64, 64)
    logits[0, SQUARE_TO_IDX["e2"], SQUARE_TO_IDX["e4"]] = 3.5
    logits[1, SQUARE_TO_IDX["g1"], SQUARE_TO_IDX["f3"]] = 2.25

    result = argmax_2d_indices_batch(logits)

    assert result == [
        (SQUARE_TO_IDX["e2"], SQUARE_TO_IDX["e4"], 3.5),
        (SQUARE_TO_IDX["g1"], SQUARE_TO_IDX["f3"], 2.25),
    ]


def test_topk_valid_moves_skips_invalid_candidates():
    """Test top-k move selection filters through the supplied validator."""

    def validator(from_idx, to_idx, board_fen=None, turn="wb"):
        return from_idx != to_idx

    logits = torch.zeros(64, 64)
    logits[0, 0] = 10.0
    logits[SQUARE_TO_IDX["e2"], SQUARE_TO_IDX["e4"]] = 9.0
    logits[SQUARE_TO_IDX["g1"], SQUARE_TO_IDX["f3"]] = 8.0

    result = topk_valid_moves_from_logits(logits, move_validator=validator, topk=2)

    assert [move[:2] for move in result] == [
        (SQUARE_TO_IDX["e2"], SQUARE_TO_IDX["e4"]),
        (SQUARE_TO_IDX["g1"], SQUARE_TO_IDX["f3"]),
    ]
    assert result[0][2] > result[1][2]


def test_best_valid_move_raises_when_no_valid_moves():
    """Test best-move selection reports an empty valid-candidate set."""
    logits = torch.zeros(64, 64)

    with pytest.raises(ValueError, match="No valid moves found"):
        best_valid_move_from_logits(logits, move_validator=lambda *args, **kwargs: False)


def test_get_legal_moves_honors_turn_filter():
    """Test legal-move generation can be restricted by side to move."""
    fen = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"

    assert "e2e4" in get_legal_moves(fen, turn="w")
    assert "e2e4" not in get_legal_moves(fen, turn="b")


def test_fenlike_path_and_board_id_round_trip():
    """Test board IDs can be recovered from FEN-like file names."""
    fen = "8/8/8/8/8/8/4P3/4K3"
    board_id = get_board_id(fen)

    assert board_id == "8_8_8_8_8_8_4P3_4K3"
    assert is_path_fenlike(f"/tmp/{board_id}.png") == fen
    assert is_path_fenlike("not-a-board.png") is False
