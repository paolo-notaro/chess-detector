"""Tests for chess_utils module."""

from chess_detector.data.chess_utils import from_uci, is_move_valid, to_uci


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
