"""Train the image-pair baseline chess move model.

This module is invoked as the ``chess-detector-train-pair`` console script.
"""

import csv
import os
import random

import mlflow
import torch
from tqdm import tqdm

from chess_detector.data import dataset
from chess_detector.models.pair import (
    ChessMovePredictor,
    SmallCNNEncoder,
    count_params,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100000
checkpoint_to_load = None  # "models/checkpoint_hilarious-doe-40_epoch1.pth" or None

train_eval_ratio = 0.8

if device.type == "cpu":
    print("Warning: CUDA not available, using CPU...")
else:
    print(f"Using {device}")


criterion = torch.nn.CrossEntropyLoss()


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct_from = correct_to = correct_promo = 0
    total = 0

    for before, after, label in tqdm(dataloader):
        before = before.to(device)
        after = after.to(device)
        y_from = label["from"].to(device)
        y_to = label["to"].to(device)
        y_promo = label["promotion"].to(device)

        optimizer.zero_grad()
        out_from, out_to, out_promo = model(before, after)

        loss_from = criterion(out_from, y_from)
        loss_to = criterion(out_to, y_to)
        loss_promo = criterion(out_promo, y_promo)
        loss = loss_from + loss_to + loss_promo

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * before.size(0)
        total += before.size(0)

        # Accuracy metrics
        pred_from = out_from.argmax(dim=1)
        pred_to = out_to.argmax(dim=1)
        pred_promo = out_promo.argmax(dim=1)

        correct_from += (pred_from == y_from).sum().item()
        correct_to += (pred_to == y_to).sum().item()
        correct_promo += (pred_promo == y_promo).sum().item()

    avg_loss = total_loss / total
    acc_from = correct_from / total
    acc_to = correct_to / total
    acc_promo = correct_promo / total

    return avg_loss, acc_from, acc_to, acc_promo


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_from = correct_to = correct_promo = 0
    total = 0

    with torch.no_grad():
        for before, after, label in dataloader:
            before = before.to(device)
            after = after.to(device)
            y_from = label["from"].to(device)
            y_to = label["to"].to(device)
            y_promo = label["promotion"].to(device)

            out_from, out_to, out_promo = model(before, after)

            loss_from = criterion(out_from, y_from)
            loss_to = criterion(out_to, y_to)
            loss_promo = criterion(out_promo, y_promo)
            loss = loss_from + loss_to + loss_promo

            total_loss += loss.item() * before.size(0)
            total += before.size(0)

            pred_from = out_from.argmax(dim=1)
            pred_to = out_to.argmax(dim=1)
            pred_promo = out_promo.argmax(dim=1)

            correct_from += (pred_from == y_from).sum().item()
            correct_to += (pred_to == y_to).sum().item()
            correct_promo += (pred_promo == y_promo).sum().item()

    avg_loss = total_loss / total
    acc_from = correct_from / total
    acc_to = correct_to / total
    acc_promo = correct_promo / total

    return avg_loss, acc_from, acc_to, acc_promo


def save_checkpoint(model, optimizer, epoch, best_val_loss: float, path="models/checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )
    print(f"Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # continue from next epoch
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"📦 Loaded checkpoint from {path}, starting at epoch {start_epoch}")
    return start_epoch, best_val_loss


def _prepare_splits() -> None:
    if os.path.exists("dataset/entries_train.csv") and os.path.exists("dataset/entries_eval.csv"):
        return

    with open("dataset/last_index.txt") as f:
        last_index = int(f.read())

    with open("dataset/entries.csv") as f:
        entries = list(csv.reader(f))[1 : last_index + 1]

    print(
        f"Total of {len(entries)} moves found. "
        f"Splitting dataset with a ratio of {train_eval_ratio}..."
    )
    random.shuffle(entries)
    split_index = int(len(entries) * train_eval_ratio)

    with open("dataset/entries_train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["before_fen", "move_uci", "after_fen"])
        writer.writerows(entries[:split_index])

    with open("dataset/entries_eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["before_fen", "move_uci", "after_fen"])
        writer.writerows(entries[split_index:])


def _run_training() -> None:
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("ChessMovePrediction")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", run_id)

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("train_eval_ratio", train_eval_ratio)
        mlflow.log_param("learning_rate", 1e-4)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("checkpoint_to_reload", checkpoint_to_load)
        mlflow.log_param("device", device.type)

        model = ChessMovePredictor(encoder_class=SmallCNNEncoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print(f"Total parameters: {count_params(model, trainable_only=False)}")
        print(f"Trainable parameters: {count_params(model, trainable_only=True)}")

        train_dataset = dataset.ChessMoveDatasetFromCSV(
            "dataset/entries_train.csv", "dataset/preprocessed"
        )
        val_dataset = dataset.ChessMoveDatasetFromCSV(
            "dataset/entries_eval.csv", "dataset/preprocessed"
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Val batches per epoch: {len(val_loader)}")

        if checkpoint_to_load:
            _start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, path=checkpoint_to_load, device=device
            )
        else:
            _start_epoch, best_val_loss = 0, float("inf")

        for epoch in range(num_epochs):
            train_loss, train_acc_from, train_acc_to, train_acc_promo = train(
                model, train_loader, optimizer, device
            )
            val_loss, val_acc_from, val_acc_to, val_acc_promo = evaluate(model, val_loader, device)

            print(f"[Epoch {epoch + 1}]")
            print(
                f" Train Loss: {train_loss:.4f} | "
                f"From: {train_acc_from:.2%}, To: {train_acc_to:.2%}, Promo: {train_acc_promo:.2%}"
            )
            print(
                f" Val   Loss: {val_loss:.4f} | "
                f"From: {val_acc_from:.2%}, To: {val_acc_to:.2%}, Promo: {val_acc_promo:.2%}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc_from": train_acc_from,
                    "train_acc_to": train_acc_to,
                    "train_acc_promo": train_acc_promo,
                    "val_loss": val_loss,
                    "val_acc_from": val_acc_from,
                    "val_acc_to": val_acc_to,
                    "val_acc_promo": val_acc_promo,
                },
                step=epoch + 1,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_name = f"models/checkpoint_{run_name}_epoch{epoch + 1}.pth"
                save_checkpoint(
                    model, optimizer, epoch, path=ckpt_name, best_val_loss=best_val_loss
                )
                mlflow.log_artifact(ckpt_name)


def main() -> None:
    """Console-script entry point: train the image-pair baseline model."""
    if not os.path.exists("dataset/preprocessed") or not os.path.exists("dataset/last_index.txt"):
        print("Preprocessed dataset not found, please run chess-detector-gen-dataset first")
        return

    _prepare_splits()
    _run_training()


if __name__ == "__main__":
    main()
