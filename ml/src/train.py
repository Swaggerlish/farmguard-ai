import os
import copy
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import build_dataloaders
from src.model import build_model
from src.transforms import get_train_transforms, get_eval_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, train=True):
    model.train(train)
    all_preds, all_labels = [], []
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        with autocast(enabled=(DEVICE == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1

def train_model(train_dir, val_dir, test_dir, out_dir="outputs/checkpoints", batch_size=32, epochs_head=4, epochs_ft=6):
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        train_dir, val_dir, test_dir, batch_size=batch_size
    )

    with open(os.path.join(out_dir, "class_to_idx.json"), "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    model = build_model(num_classes=len(train_ds.classes), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_model = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    print("Stage 1: training head")
    for epoch in range(epochs_head):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, train=False)
        print(f"[Head][{epoch+1}/{epochs_head}] train_f1={tr_f1:.4f} val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_model = copy.deepcopy(model.state_dict())

    print("Stage 2: fine-tuning top layers")
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)

    for epoch in range(epochs_ft):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, train=False)
        print(f"[FT][{epoch+1}/{epochs_ft}] train_f1={tr_f1:.4f} val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
    print(f"Saved best model with val macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_model(
        train_dir="data/processed/train",
        val_dir="data/processed/val",
        test_dir="data/processed/test",
    )