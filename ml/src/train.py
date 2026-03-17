import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from src.dataset import build_dataloaders
from src.model import build_model

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


def resolve_pretrained_checkpoint(
    pretrained_checkpoint: str | None,
    hf_repo_id: str | None,
    hf_filename: str,
    hf_revision: str,
) -> str | None:
    if pretrained_checkpoint:
        checkpoint_path = Path(pretrained_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    if not hf_repo_id:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --hf-repo-id. Install with `pip install huggingface_hub`."
        ) from exc

    downloaded = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, revision=hf_revision)
    return downloaded


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    if missing_keys:
        print(f"Warning: missing keys while loading checkpoint ({len(missing_keys)}).")
    if unexpected_keys:
        print(f"Warning: unexpected keys while loading checkpoint ({len(unexpected_keys)}).")


def train_model(
    train_dir,
    val_dir,
    test_dir,
    out_dir="outputs/checkpoints",
    batch_size=32,
    epochs_head=4,
    epochs_ft=6,
    img_size=224,
    num_workers=2,
    pin_memory=None,
    label_smoothing=0.0,
    lr_head=3e-4,
    lr_ft=1e-4,
    weight_decay=1e-4,
    pretrained_checkpoint=None,
    hf_repo_id=None,
    hf_filename="best_model.pth",
    hf_revision="main",
):
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        train_dir,
        val_dir,
        test_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    with open(os.path.join(out_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    model = build_model(num_classes=len(train_ds.classes), freeze_backbone=True).to(DEVICE)

    resolved_checkpoint = resolve_pretrained_checkpoint(
        pretrained_checkpoint=pretrained_checkpoint,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
    )
    if resolved_checkpoint:
        load_pretrained_weights(model, resolved_checkpoint)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
        weight_decay=weight_decay,
    )
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_model = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    print(f"Device: {DEVICE}")
    print(f"Classes: {len(train_ds.classes)}")
    print(f"Image size: {img_size}")
    print(f"Learning rates -> head: {lr_head}, fine-tune: {lr_ft}")
    print("Stage 1: training head")

    for epoch in range(epochs_head):
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scaler, train=True
        )
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, train=False)
        print(
            f"[Head][{epoch+1}/{epochs_head}] "
            f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
            f"val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_model = copy.deepcopy(model.state_dict())

    print("Stage 2: fine-tuning top layers")
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_ft,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs_ft):
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scaler, train=True
        )
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, train=False)
        print(
            f"[FT][{epoch+1}/{epochs_ft}] "
            f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
            f"val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    checkpoint_path = os.path.join(out_dir, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved best model with val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FarmGuard model.")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--val-dir", default="data/processed/val")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--out-dir", default="outputs/checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-ft", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Force DataLoader pin_memory=True (defaults to CUDA-aware auto mode).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-ft", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Local checkpoint path to warm-start fine-tuning.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face model repo id for warm-start (e.g. swaggerlish/farmguard-ai-multi-crops-disease).",
    )
    parser.add_argument(
        "--hf-filename",
        default="best_model.pth",
        help="Checkpoint filename inside Hugging Face repo.",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Hugging Face revision/branch/tag to download from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
        img_size=args.img_size,
        num_workers=args.num_workers,
        pin_memory=True if args.pin_memory else None,
        label_smoothing=args.label_smoothing,
        lr_head=args.lr_head,
        lr_ft=args.lr_ft,
        weight_decay=args.weight_decay,
        pretrained_checkpoint=args.pretrained_checkpoint,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        hf_revision=args.hf_revision,
    )
