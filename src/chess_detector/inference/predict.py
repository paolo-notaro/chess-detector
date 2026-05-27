"""Predict a chess move from a pair of board images.

This module is invoked as the ``chess-detector-predict`` console script.
"""

import argparse
import os

import cv2 as cv
import torch

from chess_detector.data import chess_utils
from chess_detector.data.chess_utils import is_path_fenlike, topk_valid_moves_from_logits
from chess_detector.data.dataset import ChessMoveFromDiffDataset
from chess_detector.data.postprocessing import (
    calibrate_camera_from_path_match,
    gen_diff,
    rectify_board,
)
from chess_detector.models.diff import ChessMoveModel, ConvPatchEncoder

PREPROCESSING_OUT_SIZE = 224  # Size of the preprocessed images
PATCH_SIZE = PREPROCESSING_OUT_SIZE // 8  # As used in the dataset loading script
FINAL_PATCH_SIZE = 32  # Final patch size for the model input

ENCODER_CLASS = ConvPatchEncoder  # Change this to ResnetPatchEncoder if needed


def predict_move(
    model: torch.nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
    board_fen: str | None = None,
    turn: str = "wb",
    topk: int = 1,
) -> list[tuple[str, float]]:
    """
    Predict the move from the patch tensor using the model.

    Args:
        model (torch.nn.Module): The trained model.
        patch_tensor (torch.tensor): The input tensor of shape [64, 1, 32, 32].
        device (torch.device): The device to run the model on.
        board_fen (str): The FEN string of the board state before the move.
         If None, no validation is performed.
        turn (str): The turn ('w' for white, 'b' for black, 'wb' for either).

    Returns:

        list[tuple[str, float]]: A list of moves as tuples, containing the predicted move in UCI
          format and their confidence score.
    """
    topk_moves = []
    model.eval()
    patch_tensor = patch_tensor.to(device)

    with torch.no_grad():
        # Get the logits from the model
        patch_tensor = patch_tensor.unsqueeze(0)  # Add batch dimension
        scores = model(patch_tensor)
        scores = scores.squeeze(0)  # Remove batch dimension

        # Get the top-k valid moves from the logits
        topk_from_to = topk_valid_moves_from_logits(
            logits=scores, board_fen=board_fen, turn=turn, topk=topk
        )

    # Convert from_idx and to_idx to UCI format
    for from_idx, to_idx, score in topk_from_to:
        move_uci = chess_utils.SQUARES[from_idx] + chess_utils.SQUARES[to_idx]
        topk_moves.append((move_uci, score))

    return topk_moves


def main(args: argparse.Namespace | None = None) -> None:
    """Run chess move prediction from a pair of images.

    Args:
        args: Optional pre-parsed argparse namespace. When ``None`` (the default,
            used by the console-script entry point), arguments are read from
            ``sys.argv`` via :func:`parse_args`.
    """
    if args is None:
        args = parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found.")
        return

    if not os.path.exists(args.before) or not os.path.exists(args.after):
        print("One or both input images not found.")
        return

    fen_candidate = is_path_fenlike(args.before)
    board_fen = fen_candidate if isinstance(fen_candidate, str) else None

    if args.preprocess:
        img_before = cv.imread(args.before)
        img_after = cv.imread(args.after)

        # Find chessboard corners using base board images
        chessboard_corners = calibrate_camera_from_path_match("dataset/images/empty_board_*.png")

        # Warp the images to get a top-down view of the chessboard, will also convert to grayscale
        img_before = rectify_board(img_before, chessboard_corners, size=PREPROCESSING_OUT_SIZE)
        img_after = rectify_board(img_after, chessboard_corners, size=PREPROCESSING_OUT_SIZE)

    else:
        img_before = cv.imread(args.before, cv.IMREAD_GRAYSCALE)
        img_after = cv.imread(args.after, cv.IMREAD_GRAYSCALE)

    diff_image = gen_diff(img_before, img_after)

    # Resize and normalize the diff image
    preprocessed_diff_image = ChessMoveFromDiffDataset.preprocess_image(
        diff_image, preprocess_resize=PREPROCESSING_OUT_SIZE
    )

    # Split the image into patches
    patch_tensor = ChessMoveFromDiffDataset.patch_image(
        preprocessed_diff_image, resize_size=FINAL_PATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMoveModel(embed_dim=256, encoder_class=ENCODER_CLASS)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    model.to(device)

    move, confidence = predict_move(model, patch_tensor, device, board_fen=board_fen, topk=1)[
        0
    ]  # Get the top move
    print(f"Predicted move: {move} (confidence: {confidence:.4f})")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for :func:`main`."""
    parser = argparse.ArgumentParser(description="Chess Move Diff Prediction Test")
    parser.add_argument(
        "--before",
        type=str,
        default="dataset/preprocessed/r4rk1_pp1qbpp1_3p1n1p_2pNp3_4P3_1P1P2PB_PBP2P1P_R2Q1RK1.png",
        help="Path to before-move image",
    )
    parser.add_argument(
        "--after",
        type=str,
        default="dataset/preprocessed/r4rk1_pp2bpp1_3p1n1p_2pNp3_4P3_1P1P2Pq_PBP2P1P_R2Q1RK1.png",
        help="Path to after-move image",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Preprocess the images before prediction (warp + grayscale + resize)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best.pth",
        help="Path to model checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
