import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from dataset import rendering
from dataset.chess_utils import PROMOTION_TO_IDX, SQUARE_TO_IDX, from_uci


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
    
    @staticmethod
    def patch_image(img: np.ndarray, resize_size: int = None) -> torch.Tensor:
        """
        Splits a square image into 64 patches of size resize_size and returns them as a tensor.
        The patches are ordered from a1 to h8, with a1 being the bottom left corner.
        The patches are resized to resize_size x resize_size if resize_size is provided.

        Args:
            img (np.ndarray): Input image of shape (H, W) where H == W.
            resize_size (int, optional): Size to resize each patch to. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (64, 1, resize_size, resize_size) containing the patches.
        """

        assert img.shape[0] == img.shape[1], f"Expected square image, got {img.shape}"
        assert img.shape[0] % 8 == 0, f"Expected image size divisible by 8, got {img.shape}"
        assert resize_size is None or resize_size > 0, f"Expected resize size greater than 0, got {resize_size}"

        # Split into 64 (32x32) patches
        patches = []
        PATCH_SIZE = img.shape[0] // 8  # e.g. input size 224 // 8 = 28
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
                if resize_size and resize_size != PATCH_SIZE:
                    patch = cv2.resize(patch, (resize_size, resize_size))  # Standardize size
                # cv2.imwrite(f"test/debug_output_{patch_index}.png", patch * 255)
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)
                # print(f"Patch {patch_index} shape: {patch.shape}, row: {row}, col: {col}, x: {x}, y: {y}")
                patches.append(patch)

        patches = torch.stack(patches)  # (64, 1, 32, 32)
        return patches
    
    @staticmethod
    def preprocess_image(img: np.ndarray, preprocess_resize: int = None) -> np.ndarray:
        """
        Preprocess the image by resizing it to preprocess_resize x preprocess_resize and normalizing it.

        Args:
            img (np.ndarray): Input image of shape (H, W).
            preprocess_resize (int, optional): Size to resize the image to. Defaults to None.
        
        Returns:
            np.ndarray: Preprocessed image.
        """
        assert img.shape[0] == img.shape[1], f"Expected square image, got {img.shape}"
        assert preprocess_resize is None or preprocess_resize > 0, f"Expected resize size greater than 0, got {preprocess_resize}"

        if preprocess_resize:
            img = cv2.resize(img, (preprocess_resize, preprocess_resize)) if img.shape != (preprocess_resize, preprocess_resize) else img
        # cv2.imwrite(f"test/debug_output.png", img)

        # normalize
        img = img.astype('float32') / 255.0

        return img

    @staticmethod
    def _load_image(img_path: str, preprocess_resize: int = None, out_resize: int = None) -> torch.Tensor:
        """
        Load an image from the given path, resize it to 224x224, normalize it, and split it into patches.

        Args:
            img_path (str): Path to the image file.
            preprocess_resize (int, optional): Resize the image to this size. Defaults to None.
            out_resize (int, optional): Resize the patches to this size. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (64, 1, 32, 32) containing the patches.
        """
        assert os.path.exists(img_path), f"Image {img_path} not found."
        assert preprocess_resize is None or preprocess_resize > 0, f"Expected resize size greater than 0, got {preprocess_resize}"
        
        # Load the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image (resize and normalize)
        img = ChessMoveFromDiffDataset.preprocess_image(img, preprocess_resize=preprocess_resize)

        # Split into 64 patches and stack them
        patches_tensor = ChessMoveFromDiffDataset.patch_image(img, resize_size=out_resize)

        return patches_tensor

    def __getitem__(self, index):
        row = self.df.iloc[index]
        id = row["id"]
        img_path = os.path.join(self.diff_images_dir, f"{id}.png")
        move_uci = row["move_uci"]

        diff_tensor = ChessMoveFromDiffDataset._load_image(img_path, preprocess_resize=224, out_resize=32)

        from_sq, to_sq, promo = from_uci(move_uci)

        label = {
            "from": torch.tensor(from_sq, dtype=torch.long),
            "to": torch.tensor(to_sq, dtype=torch.long),
            "promotion": torch.tensor(promo, dtype=torch.long)
        }

        return diff_tensor, label
