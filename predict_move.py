# infer_move.py
import argparse
import torch
import cv2 as cv
import numpy as np
from dataset.postprocessing import gen_diff
from diff_models import ChessMoveModel
from dataset import dataset  # for SQUARES and SQUARE_TO_IDX
import os

PATCH_SIZE = 28  # As used in the dataset loading script
FINAL_PATCH_SIZE = 32

def process_diff_to_tensor(diff_img):
    """
    Process 224x224 grayscale diff image to a (64, 1, 32, 32) tensor
    """
    patches = []
    for rank in range(1, 9):
        for file in "abcdefgh":
            file_index = ord(file) - ord('a')
            patch_index = dataset.SQUARE_TO_IDX[f"{file}{rank}"]
            row = 7 - file_index
            col = 8 - rank
            x = col * PATCH_SIZE
            y = row * PATCH_SIZE
            patch = diff_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch = cv.resize(patch, (FINAL_PATCH_SIZE, FINAL_PATCH_SIZE))
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)
            patches.append(patch_tensor)
    return torch.stack(patches).unsqueeze(0)  # (1, 64, 1, 32, 32)

def predict_move(model, patch_tensor, device):
    model.eval()
    patch_tensor = patch_tensor.to(device)
    with torch.no_grad():
        print(patch_tensor.shape)
        scores = model(patch_tensor)
        print(scores.shape)
        pred_flat_idx = scores.view(-1, 64 * 64).argmax(dim=1).item()
        from_sq_idx = pred_flat_idx // 64
        to_sq_idx = pred_flat_idx % 64
        SQUARES = dataset.SQUARES
        move_uci = SQUARES[from_sq_idx] + SQUARES[to_sq_idx]
        return move_uci

def main(args):
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found.")
        return

    if not os.path.exists(args.before) or not os.path.exists(args.after):
        print("One or both input images not found.")
        return

    img_before = cv.imread(args.before, cv.IMREAD_GRAYSCALE)
    img_after = cv.imread(args.after, cv.IMREAD_GRAYSCALE)

    diff = gen_diff(img_before, img_after)

    patch_tensor = process_diff_to_tensor(diff)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMoveModel(embed_dim=256)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    model.to(device)

    move = predict_move(model, patch_tensor, device)
    print(f"Predicted move: {move}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=str, default="dataset/preprocessed/r4rk1_pp1qbpp1_3p1n1p_2pNp3_4P3_1P1P2PB_PBP2P1P_R2Q1RK1.png", help="Path to before-move image")
    parser.add_argument("--after", type=str, default="dataset/preprocessed/r4rk1_pp2bpp1_3p1n1p_2pNp3_4P3_1P1P2Pq_PBP2P1P_R2Q1RK1.png", help="Path to after-move image")
    parser.add_argument("--checkpoint", type=str, default="models/best.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    main(args)
