import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.model import build_model
from src.train import DEVICE, load_pretrained_weights, resolve_pretrained_checkpoint, run_epoch
from src.transforms import get_eval_transforms, get_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with stratified K-fold cross validation.")
    parser.add_argument(
        "--data-dir",
        default="data/processed/train",
        help="Root directory containing class subfolders (ImageFolder layout).",
    )
    parser.add_argument("--out-dir", default="outputs/checkpoints_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Force DataLoader pin_memory=True (defaults to CUDA-aware auto mode).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Local checkpoint path to warm-start fine-tuning.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face model repo id for warm-start.",
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


def build_fold_loaders(dataset_dir, train_idx, val_idx, img_size, batch_size, num_workers, pin_memory):
    train_ds = ImageFolder(dataset_dir, transform=get_train_transforms(img_size))
    val_ds = ImageFolder(dataset_dir, transform=get_eval_transforms(img_size))

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_ds, train_loader, val_loader


def train_kfold(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    pin_memory = True if args.pin_memory else torch.cuda.is_available()

    base_ds = ImageFolder(args.data_dir)
    targets = base_ds.targets

    with open(Path(args.out_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(base_ds.class_to_idx, f, indent=2)

    resolved_checkpoint = resolve_pretrained_checkpoint(
        pretrained_checkpoint=args.pretrained_checkpoint,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        hf_revision=args.hf_revision,
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_scores = []

    print(f"Device: {DEVICE}")
    print(f"Classes: {len(base_ds.classes)}")
    print(f"Samples: {len(base_ds)}")
    print(f"Running {args.n_splits}-fold stratified CV")

    for fold, (train_idx, val_idx) in enumerate(skf.split(base_ds.samples, targets), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        train_ds, train_loader, val_loader = build_fold_loaders(
            args.data_dir,
            train_idx,
            val_idx,
            args.img_size,
            args.batch_size,
            args.num_workers,
            pin_memory,
        )

        model = build_model(num_classes=len(train_ds.classes), freeze_backbone=False).to(DEVICE)
        if resolved_checkpoint:
            load_pretrained_weights(model, resolved_checkpoint)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=(DEVICE == "cuda"))

        best_f1 = 0.0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(args.epochs):
            tr_loss, tr_acc, tr_f1 = run_epoch(
                model, train_loader, criterion, optimizer, scaler, train=True
            )
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, train=False)
            print(
                f"[Fold {fold}][{epoch+1}/{args.epochs}] "
                f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
                f"val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
            )

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        fold_ckpt = Path(args.out_dir) / f"best_model_fold{fold}.pth"
        torch.save(model.state_dict(), fold_ckpt)
        fold_scores.append(best_f1)
        print(f"Fold {fold} best val macro-F1: {best_f1:.4f}")
        print(f"Saved fold checkpoint to: {fold_ckpt}")

    mean_f1 = float(sum(fold_scores) / len(fold_scores))
    std_f1 = float(torch.tensor(fold_scores).std(unbiased=False).item())
    results = {
        "n_splits": args.n_splits,
        "fold_macro_f1": fold_scores,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
    }

    with open(Path(args.out_dir) / "kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== K-fold summary =====")
    for idx, score in enumerate(fold_scores, start=1):
        print(f"Fold {idx}: {score:.4f}")
    print(f"Mean macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")


def main() -> None:
    args = parse_args()
    train_kfold(args)


if __name__ == "__main__":
    main()
