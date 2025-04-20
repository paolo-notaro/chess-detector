"""chess_utils.py: Utility functions for chess move prediction and validation."""

from os.path import basename
from typing import Callable, List, Optional, Tuple

import chess
import torch

SQUARES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]
PROMOTIONS = ["", "q", "r", "b", "n"]
SQUARE_TO_IDX = {sq: i for i, sq in enumerate(SQUARES)}
PROMOTION_TO_IDX = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}


def to_uci(from_idx: int, to_idx: int, promo_index: int = 0) -> str:
    """
    Convert from_idx and to_idx to UCI format.

    Arguments:
        from_idx (int): Starting square index (0-63).
        to_idx (int): Target square index (0-63).
        promo_index (int): Promotion piece index (0-4).

                0: No promotion, 1: Queen, 2: Rook, 3: Bishop, 4: Knight.

    Returns:
        str: UCI formatted move string.
    """
    assert 0 <= from_idx < 64 and 0 <= to_idx < 64, "Indices must be in range [0, 64)"
    from_square = SQUARES[from_idx]
    to_square = SQUARES[to_idx]
    promo = PROMOTIONS[promo_index]

    return from_square + to_square + promo


def from_uci(move: str) -> tuple[int, int, int]:
    """
    Convert UCI move string to from_idx, to_idx, and promo_index.

    Arguments:
        move (str): UCI formatted move string.

    Returns:
        tuple[int, int, int]: from_idx, to_idx, promo_index.
    """
    assert len(move) >= 4 and len(move) <= 6, "Invalid UCI move format"

    from_square = move[:2]
    to_square = move[2:4]
    promo = move[4:] if len(move) > 4 else ""

    from_idx = SQUARE_TO_IDX[from_square]
    to_idx = SQUARE_TO_IDX[to_square]
    promo_index = PROMOTION_TO_IDX.get(promo, 0)

    return from_idx, to_idx, promo_index


def argmax_2d_indices_batch(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """
    Return a list of (from_idx, to_idx, confidence) for each item in a [B, 64, 64] tensor.

    Args:
        tensor (torch.Tensor): shape [B, 64, 64]

    Returns:
        List of tuples: [(from_idx, to_idx, confidence), ...]
    """
    assert tensor.dim() == 3 and tensor.shape[1:] == (
        64,
        64,
    ), f"Expected shape [B, 64, 64], got {tensor.shape}"
    flattened_tensor = tensor.view(tensor.size(0), -1)  # [B, 4096]
    flat_idx = flattened_tensor.argmax(dim=1)
    from_idx = flat_idx // 64
    to_idx = flat_idx % 64
    confidence = flattened_tensor.max(dim=1)
    return list(zip(from_idx.tolist(), to_idx.tolist(), confidence.tolist()))


def is_move_valid(
    from_idx: int, to_idx: int, board_fen: str = None, turn: str = "wb"
) -> bool:
    """
    Check if the move from from_idx to to_idx is valid.
    If board_fen is provided, validate the move against the current board state.

    Args:
        from_idx (int): Starting square index (0-63).
        to_idx (int): Target square index (0-63).
        board_fen (str): FEN string for the current board position.
        turn (str): 'w', 'b', or 'wb' indicating the turn.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    if board_fen:
        valid_moves = get_legal_moves(board_fen, turn=turn)
        # Convert from_idx and to_idx to UCI format
        move_uci = to_uci(from_idx, to_idx)
        return move_uci in valid_moves

    return from_idx != to_idx


def topk_valid_moves_from_logits(
    logits: torch.Tensor,
    move_validator: Callable[[int, int, Optional[str]], bool] = is_move_valid,
    board_fen: Optional[str] = None,
    turn: str = "wb",
    topk: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Returns the top-k valid (from_idx, to_idx, confidence) tuples from logits[64, 64],
    skipping invalid moves.

    Args:
        logits (torch.Tensor): shape [64, 64]
        move_validator (function): function to validate moves
        board_fen (str, optional): FEN string for the current board position
        turn (str): 'w', 'b', or 'wb' indicating the turn
        topk (int): number of valid moves to return

    Returns:
        List[Tuple[int, int, float]]: list of top-k valid moves
    """
    assert logits.shape == (64, 64), "Expected logits of shape [64, 64]"

    probs = torch.nn.functional.softmax(logits.view(-1), dim=0)  # [4096]
    sorted_indices = torch.argsort(probs, descending=True)

    valid_moves = []
    for flat_idx in sorted_indices:
        from_idx = flat_idx // 64
        to_idx = flat_idx % 64
        if move_validator(
            from_idx.item(), to_idx.item(), board_fen=board_fen, turn=turn
        ):
            confidence = probs[flat_idx].item()
            valid_moves.append((from_idx.item(), to_idx.item(), confidence))
            if len(valid_moves) == topk:
                break

    if not valid_moves:
        raise ValueError("No valid moves found in logits")

    return valid_moves


def best_valid_move_from_logits(
    logits: torch.Tensor,
    move_validator=is_move_valid,
    board_fen: str = None,
    turn: str = "wb",
) -> tuple[int, int, float]:
    """
    Returns the best (from_idx, to_idx) pair from logits[64,64],
    skipping invalid moves (e.g. from == to).

    Args:
        logits (torch.Tensor): shape [64, 64]
        move_validator (function): function to validate moves
        board_fen (str): FEN string for the current board position
        turn (str): 'w', 'b', or 'wb' indicating the turn

    Returns:
        tuple[int, int, float]: (from_idx, to_idx) of the most likely valid move
    """
    assert logits.shape == (64, 64)
    probs = torch.nn.functional.softmax(logits.view(-1), dim=0)  # [4096]

    sorted_indices = torch.argsort(probs, descending=True)

    for flat_idx in sorted_indices:
        from_idx = flat_idx // 64
        to_idx = flat_idx % 64
        if move_validator(
            from_idx.item(), to_idx.item(), board_fen=board_fen, turn=turn
        ):
            confidence = probs[flat_idx].item()
            return from_idx.item(), to_idx.item(), confidence

    raise ValueError("No valid moves found in logits")


def get_legal_moves(fen: str, turn="wb", castling_rights=chess.BB_CORNERS) -> set[str]:
    """
    Get legal moves for the given FEN string.
    Parameters:
        fen (str): FEN string representing the board position.
        turn (None): 'w' or 'b' or 'wb'.
        castling_rights (chess.BB_CORNERS): Castling rights for the position.
    """
    board = chess.Board(fen)
    board.castling_rights = castling_rights

    moves = set()

    if "w" in turn:
        board.turn = chess.WHITE
        legal_moves = board.legal_moves
        for move in legal_moves:
            moves.add(move.uci())

    if "b" in turn:
        board.turn = chess.BLACK
        legal_moves = board.legal_moves
        for move in legal_moves:
            moves.add(move.uci())

    return moves


def is_path_fenlike(file_path: str) -> bool | str:
    """
    Check if the given path represents a FEN-like string.

    Keep basename, convert _ to / and remove .png

    If it is fenlike, return FEN string. Else, return false.
    """
    fen_string = basename(file_path)
    fen_string = fen_string.replace("_", "/").replace(".png", "")

    if fen_string.count("/") == 7 and all(
        char in "rnbqkpRNBQKP12345678/" for char in fen_string
    ):
        # Check if the FEN string is valid
        try:
            chess.Board(fen_string)
        except ValueError:
            return False
        return fen_string
    else:
        return False
