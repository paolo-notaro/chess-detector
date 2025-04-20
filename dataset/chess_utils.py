import torch


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


def is_move_valid(from_idx: int, to_idx: int) -> bool:
    """
    Basic move validation:
    - must not move to the same square
    """
    return from_idx != to_idx


def best_valid_move_from_logits(
    logits: torch.Tensor,
    move_validator=is_move_valid
) -> tuple[int, int]:
    """
    Returns the best (from_idx, to_idx) pair from logits[64,64],
    skipping invalid moves (e.g. from == to).
    """
    assert logits.shape == (64, 64)
    probs = torch.nn.functional.softmax(logits.view(-1), dim=0)  # [4096]

    sorted_indices = torch.argsort(probs, descending=True)

    for flat_idx in sorted_indices:
        from_idx = flat_idx // 64
        to_idx = flat_idx % 64
        if move_validator(from_idx.item(), to_idx.item()):
            return from_idx.item(), to_idx.item()

    raise ValueError("No valid moves found in logits")
