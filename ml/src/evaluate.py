import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import build_dataloaders
from src.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FarmGuard model.")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--val-dir", default="data/processed/val")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--class-map", default="outputs/checkpoints/class_to_idx.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--save-report",
        default="outputs/checkpoints/test_classification_report.txt",
        help="Where to save classification report text.",
    )
    parser.add_argument(
        "--save-confusion",
        default="outputs/checkpoints/test_confusion_matrix.json",
        help="Where to save confusion matrix JSON.",
    )
    return parser.parse_args()


def load_target_names(class_map_path: Path, fallback_names: list[str]) -> list[str]:
    if not class_map_path.exists():
        return fallback_names

    with class_map_path.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def evaluate() -> None:
    args = parse_args()

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        args.train_dir,
        args.val_dir,
        args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    model = build_model(num_classes=len(train_ds.classes), freeze_backbone=False).to(DEVICE)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")

    target_names = load_target_names(Path(args.class_map), train_ds.classes)
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Device: {DEVICE}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    save_report_path = Path(args.save_report)
    save_report_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_path.write_text(report, encoding="utf-8")

    save_confusion_path = Path(args.save_confusion)
    save_confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with save_confusion_path.open("w", encoding="utf-8") as f:
        json.dump({"labels": target_names, "matrix": cm}, f, indent=2)

    print(f"Saved classification report to: {save_report_path}")
    print(f"Saved confusion matrix to: {save_confusion_path}")


if __name__ == "__main__":
    evaluate()
