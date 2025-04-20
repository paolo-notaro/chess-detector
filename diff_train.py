from diff_models import ChessMoveModel
from image_pairs_models import count_params
import torch
from dataset import dataset
from tqdm import tqdm
import csv
import os
import random
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100000
checkpoint_to_load = None  # "models/checkpoint_hilarious-doe-40_epoch1.pth" or None

LEARNING_RATE = 1e-4
EMBED_DIM = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 128
LIMIT_DATASET = None  # None for no limit, otherwise set to the number of samples to use. To test if overfitting works
SEED = 42

train_eval_ratio = 0.8

if device.type == "cpu":
    print("Warning: CUDA not available, using CPU...")
else:
    print(f"Using {device}")


criterion = torch.nn.CrossEntropyLoss(reduction="mean")

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    correct_inverse = 0
    total = 0

    for patch_tensor, label in (progress_bar := tqdm(dataloader)):
        patch_tensor = patch_tensor.to(device)  # shape: (B, 64, 1, 32, 32)
        B = patch_tensor.size(0)  # batch size
        y_from = label["from"].to(device)
        y_to = label["to"].to(device)

        optimizer.zero_grad()
        scores = model(patch_tensor)  # [B, 64, 64], batch size, from square, to square

        scores_flat = scores.view(B, -1)  # [B, 4096]
        
        
        # Flatten (from, to) into single label
        move_idx = y_from * 64 + y_to  # [B]

        # print(scores[0][y_from[0]][y_to[0]], scores_flat[0][move_idx[0]])

        # Compute loss
        loss = criterion(scores_flat, move_idx)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total += patch_tensor.size(0)

        # Accuracy metrics
        pred_move_idx = scores.view(B, -1).argmax(dim=1)  # predicted flat index
        true_move_idx = y_from * 64 + y_to
        inverse_true_move_idx = y_to * 64 + y_from

        correct += (pred_move_idx == true_move_idx).sum().item()
        correct_inverse += (pred_move_idx == inverse_true_move_idx).sum().item()
        correct_total = correct + correct_inverse
        acc = correct / total
        acc_incl_inverse = correct_total / total

        progress_bar.set_description(
            f"Train Loss: {total_loss / total:.4f} | Acc: {acc:.2%}, Acc (incl. inverse): {acc_incl_inverse:.2%}")

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, acc_incl_inverse


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    correct_inverse = 0
    total = 0

    with torch.no_grad():
        for patch_tensor, label in dataloader:
            B = patch_tensor.size(0)  # batch size
            patch_tensor = patch_tensor.to(device)
            y_from = label["from"].to(device)
            y_to = label["to"].to(device)

            scores = model(patch_tensor)

            scores_flat = scores.view(B, -1)  # [B, 4096]

            # Flatten (from, to) into single label
            move_idx = y_from * 64 + y_to  # [B]
            inverse_move_idx = y_to * 64 + y_from

            loss = criterion(scores_flat, move_idx)

            # Compute loss
            total_loss += loss.item() * B
            total += B
                    
            # Accuracy metrics
            pred_move_idx = scores.view(B, -1).argmax(dim=1)  # predicted flat index
            true_move_idx = y_from * 64 + y_to

            correct += (pred_move_idx == true_move_idx).sum().item()
            correct_inverse += (pred_move_idx == inverse_move_idx).sum().item()
            correct_total = correct + correct_inverse

    avg_loss = total_loss / total
    acc = correct / total
    acc_incl_inverse = correct_total / total

    return avg_loss, acc, acc_incl_inverse


def save_checkpoint(
    model, optimizer, epoch, best_val_loss: float, path="models/checkpoint.pth"
):
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


if not os.path.exists("dataset/preprocessed") or not os.path.exists(
    "dataset/last_index.txt"
):
    print("Preprocessed dataset not found, please run dataset/gen_dataset.py first")
    exit()


if not os.path.exists("dataset/diff_entries_train.csv") or not os.path.exists(
    "dataset/diff_entries_eval.csv"
):
    with open("dataset/last_index.txt", "r") as f:
        last_index = int(f.read())

    with open("dataset/entries.csv", "r") as f:
        entries = [
            (i, row[1])
            for i, row in enumerate(list(csv.reader(f))[1 : last_index + 1])
            if os.path.exists(os.path.join("dataset/diff", str(i) + ".png"))
        ]  # skip header
    print(
        f"Total of {len(entries)} moves found. Splitting dataset with a ratio of {train_eval_ratio}..."
    )

    random.shuffle(entries)

    split_index = int(len(entries) * train_eval_ratio)

    with open("dataset/diff_entries_train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "move_uci"])
        writer.writerows(entries[:split_index])

    with open("dataset/diff_entries_eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "move_uci"])
        writer.writerows(entries[split_index:])


# Start MLflow run
mlflow.set_experiment("ChessMovePrediction")

with mlflow.start_run() as run:

    run_id = run.info.run_id
    run_name = run.data.tags.get(
        "mlflow.runName", run_id
    )  # fallback to run_id if name not set

    print(f"Run name: {run_name}")

    # Log hyperparameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("train_eval_ratio", train_eval_ratio)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", TRAIN_BATCH_SIZE)
    mlflow.log_param("checkpoint_to_reload", checkpoint_to_load)
    mlflow.log_param("device", device.type)
    mlflow.log_param("starting_checkpoint", checkpoint_to_load)
    mlflow.log_param("model", "ChessMoveModel")
    mlflow.log_param("embed_dim", EMBED_DIM)

    model = ChessMoveModel(embed_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    tot_params = count_params(model, trainable_only=False)
    trainable_params = count_params(model, trainable_only=True)
    encoder_params = count_params(
        model.encoder, trainable_only=True
    )
    scorer_params = count_params(
        model.scorer, trainable_only=True
    )
    print(f"Total params: {tot_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Encoder params: {encoder_params}")
    print(f"Scorer params: {scorer_params}")

    mlflow.log_param("total_params", tot_params)
    mlflow.log_param("trainable_params", trainable_params)
    mlflow.log_param("encoder_params", encoder_params)
    mlflow.log_param("scorer_params", scorer_params)
    

    train_dataset = dataset.ChessMoveFromDiffDataset(
        "dataset/diff_entries_train.csv", "dataset/diff", limit=LIMIT_DATASET
    )
    val_dataset = dataset.ChessMoveFromDiffDataset(
        "dataset/diff_entries_eval.csv", "dataset/diff", limit=LIMIT_DATASET
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")

    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    if checkpoint_to_load:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, path=checkpoint_to_load, device=device
        )
    else:
        start_epoch, best_val_loss = 0, float("inf")

    torch.manual_seed(SEED)
    for epoch in range(num_epochs):
        train_loss, train_acc, train_acc_incl_inverse = train(model, train_loader, optimizer, device)

        val_loss, val_acc, val_acc_incl_inverse = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}]")
        print(f" Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Acc (incl. inverse): {train_acc_incl_inverse:.2%}")
        print(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Acc (incl. inverse): {val_acc_incl_inverse:.2%}")

        # Log metrics
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_acc_incl_inverse": train_acc_incl_inverse,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_acc_incl_inverse": val_acc_incl_inverse,
            },
            step=epoch + 1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = f"models/checkpoint_{run_name}_epoch{epoch+1}.pth"
            save_checkpoint(
                model, optimizer, epoch, path=ckpt_name, best_val_loss=best_val_loss
            )
            mlflow.log_artifact(ckpt_name)
