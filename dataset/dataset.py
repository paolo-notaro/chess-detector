import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from dataset import rendering
from matplotlib import pyplot as plt

SQUARES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]
SQUARE_TO_IDX = {sq: i for i, sq in enumerate(SQUARES)}
PROMOTION_TO_IDX = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}

class ChessMoveDatasetFromCSV(Dataset):
    def __init__(self, csv_path, image_dir):
        """
        csv_path: path to CSV with columns: before_fen, move_uci, after_fen
        image_dir: path to directory with preprocessed .png files (224x224 grayscale)
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_id):
        path = os.path.join(self.image_dir, f"{image_id}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224)) if img.shape != (224, 224) else img
        img = img.astype('float32') / 255.0
        return torch.from_numpy(img).unsqueeze(0)  # shape: (1, 224, 224)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        before_id = rendering.get_board_id(row["before_fen"])
        move_uci = row["move_uci"]
        after_id = rendering.get_board_id(row["after_fen"])

        before_tensor = self._load_image(before_id)
        after_tensor = self._load_image(after_id)

        from_sq = move_uci[0:2]
        to_sq = move_uci[2:4]
        promo = move_uci[4:] if len(move_uci) > 4 else ""

        label = {
            "from": torch.tensor(SQUARE_TO_IDX[from_sq], dtype=torch.long),
            "to": torch.tensor(SQUARE_TO_IDX[to_sq], dtype=torch.long),
            "promotion": torch.tensor(PROMOTION_TO_IDX.get(promo, 0), dtype=torch.long)
        }

        return before_tensor, after_tensor, label


class ChessMoveFromDiffDataset(Dataset):
    """
    Dataset for moves from diff images.
    Each image is a 224x224 grayscale diff image, representing a move.
    The image is split into 64 patches (8x8 grid), and each patch is encoded as a 32x32 tensor.
    The output is a tensor of shape (64, 1, 32, 32) representing the board as a sequence of patches.
    """

    def __init__(self, csv_path, diff_images_dir, limit=None):
        """
        csv_path: path to CSV with columns: image_id, move_uci
        image_id: ID of the diff image (0.png, 1.png, ...)
        diff_images_dir: path to directory with diff images (224x224 grayscale)
        """
        self.df = pd.read_csv(csv_path, nrows=limit)
        self.diff_images_dir = diff_images_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, id):
        path = os.path.join(self.diff_images_dir, f"{id}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image {path} not found.")

        img = cv2.resize(img, (224, 224)) if img.shape != (224, 224) else img
        # cv2.imwrite(f"test/debug_output.png", img)
        img = img.astype('float32') / 255.0

        # Split into 64 (32x32) patches
        patches = []
        PATCH_SIZE = 28
        for rank in range(1, 9):
            for file in "abcdefgh":
                file_index = ord(file) - ord('a')
                patch_index = SQUARE_TO_IDX[f"{file}{rank}"]
                # consider that a1 is in lower right corner
                row = 7 - file_index
                col = 8 - rank
                x = col * PATCH_SIZE
                y = row * PATCH_SIZE
                patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                patch = cv2.resize(patch, (32, 32))  # Standardize to 32x32
                # cv2.imwrite(f"test/debug_output_{patch_index}.png", patch * 255)
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)
                # print(f"Patch {patch_index} shape: {patch.shape}, row: {row}, col: {col}, x: {x}, y: {y}")
                patches.append(patch)

        patches_tensor = torch.stack(patches)  # (64, 1, 32, 32)
        return patches_tensor

    def __getitem__(self, index):
        row = self.df.iloc[index]
        id = row["id"]
        move_uci = row["move_uci"]

        diff_tensor = self._load_image(id)

        from_sq = move_uci[0:2]
        to_sq = move_uci[2:4]
        promo = move_uci[4:] if len(move_uci) > 4 else ""

        label = {
            "from": torch.tensor(SQUARE_TO_IDX[from_sq], dtype=torch.long),
            "to": torch.tensor(SQUARE_TO_IDX[to_sq], dtype=torch.long),
            "promotion": torch.tensor(PROMOTION_TO_IDX.get(promo, 0), dtype=torch.long)
        }

        return diff_tensor, label
