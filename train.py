import models
import torch
from dataset import dataset
from tqdm import tqdm
import csv
import os 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100000
resume = True

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

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth", device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # continue from next epoch
    print(f"📦 Loaded checkpoint from {path}, starting at epoch {start_epoch}")
    return start_epoch


if not os.path.exists("dataset/preprocessed") or not os.path.exists("dataset/last_index.txt"):
    print("Preprocessed dataset not found, please run dataset/gen_dataset.py first")
    exit()


if not os.path.exists("dataset/entries_train.csv") or not os.path.exists("dataset/entries_eval.csv"):
    with open("dataset/last_index.txt", "r") as f:
        last_index = int(f.read())

    with open("dataset/entries.csv", "r") as f:
        entries = list(csv.reader(f))[1:last_index + 1]
    
    print(f"Total of {len(entries)} moves found. Splitting dataset with a ratio of {train_eval_ratio}...")

    random.shuffle(entries)

    split_index = int(len(entries) * train_eval_ratio)


    with open("dataset/entries_train.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["before_fen", "move_uci", "after_fen"])
        writer.writerows(entries[:split_index])
    
    with open("dataset/entries_eval.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["before_fen", "move_uci", "after_fen"])
        writer.writerows(entries[split_index:])



model = models.ChessMovePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_dataset = dataset.ChessMoveDatasetFromCSV("dataset/entries_train.csv", "dataset/preprocessed")
val_dataset = dataset.ChessMoveDatasetFromCSV("dataset/entries_eval.csv", "dataset/preprocessed")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Train batches per epoch: {len(train_loader)}")

print(f"Val dataset size: {len(val_dataset)}")
print(f"Val batches per epoch: {len(val_loader)}")

if resume:
    start_epoch = load_checkpoint(model, optimizer, path="checkpoint.pth", device=device)
else:
    start_epoch = 0

for epoch in range(num_epochs):
    train_loss, train_acc_from, train_acc_to, train_acc_promo = train(
        model, train_loader, optimizer, device)

    val_loss, val_acc_from, val_acc_to, val_acc_promo = evaluate(
        model, val_loader, device)

    print(f"[Epoch {epoch+1}]")
    print(f" Train Loss: {train_loss:.4f} | From: {train_acc_from:.2%}, To: {train_acc_to:.2%}, Promo: {train_acc_promo:.2%}")
    print(f" Val   Loss: {val_loss:.4f} | From: {val_acc_from:.2%}, To: {val_acc_to:.2%}, Promo: {val_acc_promo:.2%}")

    save_checkpoint(model, optimizer, epoch, path="checkpoint.pth")