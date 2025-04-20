# infer_move.py
import argparse
import torch
import cv2 as cv
from dataset.postprocessing import gen_diff, rectify_board, calibrate_camera
from dataset.dataset import ChessMoveFromDiffDataset
from dataset.chess_utils import argmax_2d_indices_batch, best_valid_move_from_logits, is_path_fenlike
from diff_models import ChessMoveModel
from dataset import chess_utils  # for SQUARES and SQUARE_TO_IDX
import os

PREPROCESSING_OUT_SIZE = 224  # Size of the preprocessed images
PATCH_SIZE = PREPROCESSING_OUT_SIZE // 8  # As used in the dataset loading script
FINAL_PATCH_SIZE = 32  # Final patch size for the model input



def predict_move(model: torch.nn.Module, patch_tensor: torch.tensor, device: torch.device, board_fen: str = None) -> str:
    """
    Predict the move from the patch tensor using the model.
    
    Args:
        model (torch.nn.Module): The trained model.
        patch_tensor (torch.tensor): The input tensor of shape [64, 1, 32, 32].
        device (torch.device): The device to run the model on.
        board_fen (str): The FEN string of the board state before the move. If None, no validation is performed.

    Returns:
        str: The predicted move in UCI format.    
    """
    # assert patch_tensor.shape == (64, 1, 32, 32), f"Expected shape [64, 1, 32, 32], got {patch_tensor.shape}"
    model.eval()
    patch_tensor = patch_tensor.to(device)
    with torch.no_grad():
        patch_tensor = patch_tensor.unsqueeze(0)  # Add batch dimension
        scores = model(patch_tensor) 
        from_idx, to_idx = best_valid_move_from_logits(scores.squeeze(0), board_fen=board_fen)  # Remove batch dimension
        # from_idx, to_idx = argmax_2d_indices_batch(scores)[0]
        move_uci = chess_utils.SQUARES[from_idx] + chess_utils.SQUARES[to_idx]
        return move_uci


def main(args):
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found.")
        return

    if not os.path.exists(args.before) or not os.path.exists(args.after):
        print("One or both input images not found.")
        return

    if not (board_fen := is_path_fenlike(args.before)):
        board_fen = None
    
    if args.preprocess:
        img_before = cv.imread(args.before)
        img_after = cv.imread(args.after)

        # Find chessboard corners using base board images
        chessboard_corners = calibrate_camera("dataset/images/empty_board_*.png")
        
        # Warp the images to get a top-down view of the chessboard, will also convert to grayscale
        img_before = rectify_board(img_before, chessboard_corners, size=PREPROCESSING_OUT_SIZE)
        img_after = rectify_board(img_after, chessboard_corners, size=PREPROCESSING_OUT_SIZE)

    else:

        img_before = cv.imread(args.before, cv.IMREAD_GRAYSCALE)
        img_after = cv.imread(args.after, cv.IMREAD_GRAYSCALE)


    diff_image = gen_diff(img_before, img_after)

    # Resize and normalize the diff image
    preprocessed_diff_image = ChessMoveFromDiffDataset.preprocess_image(diff_image, preprocess_resize=PREPROCESSING_OUT_SIZE)

    # Split the image into patches
    patch_tensor = ChessMoveFromDiffDataset.patch_image(preprocessed_diff_image, resize_size=FINAL_PATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMoveModel(embed_dim=256)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    model.to(device)

    move = predict_move(model, patch_tensor, device, board_fen=board_fen)
    print(f"Predicted move: {move}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=str, default="dataset/preprocessed/r4rk1_pp1qbpp1_3p1n1p_2pNp3_4P3_1P1P2PB_PBP2P1P_R2Q1RK1.png", help="Path to before-move image")
    parser.add_argument("--after", type=str, default="dataset/preprocessed/r4rk1_pp2bpp1_3p1n1p_2pNp3_4P3_1P1P2Pq_PBP2P1P_R2Q1RK1.png", help="Path to after-move image")
    parser.add_argument("--preprocess", action="store_true", default=False, help="Preprocess the images before prediction (warp + grayscale + resize)")
    parser.add_argument("--checkpoint", type=str, default="models/best.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    main(args)
