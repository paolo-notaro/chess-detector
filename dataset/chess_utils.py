import torch
import chess
from os.path import basename

SQUARES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]
PROMOTIONS = ["", "q", "r", "b", "n"]
SQUARE_TO_IDX = {sq: i for i, sq in enumerate(SQUARES)}
PROMOTION_TO_IDX = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}

def to_uci(from_idx: int, to_idx: int, promo_index: int = 0) -> str:
    """
    Convert from_idx and to_idx to UCI format.
    """
    assert 0 <= from_idx < 64 and 0 <= to_idx < 64, "Indices must be in range [0, 64)"
    from_square = SQUARES[from_idx]
    to_square = SQUARES[to_idx]
    promo = PROMOTIONS[promo_index]
    
    return from_square + to_square + promo

def from_uci(move: str) -> tuple[int, int, int]:
    """
    Convert UCI move string to from_idx, to_idx, and promo_index.
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
    Return a list of (from_idx, to_idx) for each item in a [B, 64, 64] tensor.

    Args:
        tensor (torch.Tensor): shape [B, 64, 64]

    Returns:
        List of tuples: [(from_idx, to_idx), ...]
    """
    assert tensor.dim() == 3 and tensor.shape[1:] == (64, 64), \
        f"Expected shape [B, 64, 64], got {tensor.shape}"
    flat_idx = tensor.view(tensor.size(0), -1).argmax(dim=1)
    from_idx = flat_idx // 64
    to_idx = flat_idx % 64
    return list(zip(from_idx.tolist(), to_idx.tolist()))


def is_move_valid(from_idx: int, to_idx: int, board_fen: str = None) -> bool:
    """
    Basic move validation:
    - must not move to the same square
    """
    if board_fen:
        valid_moves = get_legal_moves(board_fen)
        # Convert from_idx and to_idx to UCI format
        move_uci = to_uci(from_idx, to_idx)
        return move_uci in valid_moves
    
    return from_idx != to_idx


def best_valid_move_from_logits(
    logits: torch.Tensor,
    move_validator=is_move_valid,
    board_fen: str = None
) -> tuple[int, int]:
    """
    Returns the best (from_idx, to_idx) pair from logits[64,64],
    skipping invalid moves (e.g. from == to).

    Args:
        logits (torch.Tensor): shape [64, 64]
        move_validator (function): function to validate moves
        board_fen (str): FEN string for the current board position
    
    Returns:
        tuple[int, int]: (from_idx, to_idx) of the most likely valid move
    """
    assert logits.shape == (64, 64)
    probs = torch.nn.functional.softmax(logits.view(-1), dim=0)  # [4096]

    sorted_indices = torch.argsort(probs, descending=True)

    for flat_idx in sorted_indices:
        from_idx = flat_idx // 64
        to_idx = flat_idx % 64
        if move_validator(from_idx.item(), to_idx.item(), board_fen=board_fen):
            return from_idx.item(), to_idx.item()

    raise ValueError("No valid moves found in logits")

def get_legal_moves(fen: str, turn = 'wb', castling_rights = chess.BB_CORNERS) -> set[str]:
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

    if 'w' in turn:
        board.turn = chess.WHITE
        legal_moves = board.legal_moves
        for move in legal_moves:
            moves.add(move.uci())
    
    if 'b' in turn:
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

    if fen_string.count("/") == 7 and all(char in "rnbqkpRNBQKP12345678/" for char in fen_string):
        # Check if the FEN string is valid
        try:
            chess.Board(fen_string)
        except ValueError:
            return False
        return fen_string
    else:
        return False
