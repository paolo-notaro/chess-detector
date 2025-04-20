from diff_models import ChessMoveModel
from image_pairs_models import count_params
import torch
from dataset import dataset
from tqdm import tqdm
import csv
import os
import random
import mlflow
from metrics import compute_metrics, aggregate_metrics


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

def prepare_batch(patch_tensor, label, device):
    patch_tensor = patch_tensor.to(device) # shape: (B, 64, 1, 32, 32)
    y_from = label["from"].to(device) # shape: (B,)
    y_to = label["to"].to(device) # shape: (B,)

    B = patch_tensor.size(0) # batch size
    gt_idx = y_from * 64 + y_to
    inverse_idx = y_to * 64 + y_from

    return patch_tensor, gt_idx, inverse_idx, B


def train(model, dataloader, optimizer, device) -> dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on.

    Returns:
        dict[str, float]: Dictionary containing several metrics, including:
            - loss: Average loss for the epoch.
            - acc: Accuracy of the model on the training data.
            - acc_incl_inverse: Accuracy including inverse moves.
            - avg_gt_move_score: Average score for the ground truth move.
            - avg_inverse_gt_move_score: Average score for the inverse ground truth move.
            - avg_pred_move_score: Average score for the predicted move.
    """
    model.train()
    all_metrics = []

    for patch_tensor, label in (progress_bar := tqdm(dataloader)):
        patch_tensor, gt_move_idx, inverse_gt_move_idx, B = prepare_batch(patch_tensor, label, device)

        # Forward pass
        scores = model(patch_tensor)  # [B, 64, 64], batch size, from square, to square
        scores_flat = scores.view(B, -1)  # [B, 4096]

        # Compute loss
        loss = criterion(scores_flat, gt_move_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        metrics = compute_metrics(scores_flat, gt_move_idx, inverse_gt_move_idx, loss_value=loss.item())
        all_metrics.append(metrics)

        # Update progress bar
        progress_bar.set_description(f"Batch Loss: {metrics['loss']:.4f}")


    return aggregate_metrics(all_metrics)


def evaluate(model, dataloader, device) -> dict[str, float]:
    """
    Evaluate the model on the dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device to run the model on.
    
    Returns:
        dict[str, float]: Dictionary containing several metrics, including:
            - loss: Average loss for the evaluation.
            - acc: Accuracy of the model on the evaluation data.
            - acc_incl_inverse: Accuracy including inverse moves.
    """

    model.eval()
    all_metrics = []

    with torch.no_grad():
        for patch_tensor, label in tqdm(dataloader, desc="Evaluating"):
            patch_tensor, gt_move_idx, inverse_move_idx, B = prepare_batch(patch_tensor, label, device)

            scores = model(patch_tensor)
            scores_flat = scores.view(B, -1)

            loss = criterion(scores_flat, gt_move_idx)

            metrics = compute_metrics(scores_flat, gt_move_idx, inverse_move_idx, loss_value=loss.item())
            all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)


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
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    print(f"📦 Loaded checkpoint from {path}, starting at epoch {start_epoch}")
    return start_epoch, best_val_loss, best_val_acc


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
        start_epoch, best_val_loss, best_val_acc = load_checkpoint(
            model, optimizer, path=checkpoint_to_load, device=device
        )
    else:
        start_epoch, best_val_loss, best_val_acc = 0, float("inf"), 0

    torch.manual_seed(SEED)
    for epoch in range(num_epochs):
        train_results = train(model, train_loader, optimizer, device)

        val_results = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}]")
        print(f"Train results: {train_results}")
        print(f"Val results: {val_results}")

        # Log metrics
        mlflow.log_metrics(
            {
                **{f"{k}.train": v
                for k, v in train_results.items()},
                **{f"{k}.val": v
                for k, v in val_results.items()},
            },
            step=epoch + 1,
        )

        if val_results["acc"] >  best_val_acc:
            best_val_acc = val_results["acc"]
            ckpt_name = f"models/checkpoint_{run_name}_epoch{epoch+1}.pth"
            save_checkpoint(
                model, optimizer, epoch, path=ckpt_name, best_val_loss=best_val_loss
            )
            mlflow.log_artifact(ckpt_name)
