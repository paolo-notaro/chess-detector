import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from dataset import rendering

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
