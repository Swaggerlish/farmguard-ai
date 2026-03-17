import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model import build_model
from src.transforms import get_eval_transforms
from src.utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CLASS_MAP,
    get_device,
    load_class_names,
    softmax_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", default=DEFAULT_CLASS_MAP)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def infer() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = load_class_names(args.class_map)
    device = get_device()

    model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = get_eval_transforms(args.img_size)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = softmax_probs(logits)[0]

    top_k = max(1, min(args.top_k, len(class_names)))
    top_probs, top_idx = torch.topk(probs, k=top_k)

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print("\nTop predictions:")

    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_idx.tolist()), start=1):
        label = class_names[idx]
        print(f"{rank}. {label}: {prob:.4f}")

    best_prob = top_probs[0].item()
    best_idx = top_idx[0].item()
    best_label = class_names[best_idx]

    if best_prob < args.confidence_threshold:
        print(
            f"\nLow-confidence prediction (< {args.confidence_threshold:.2f}). "
            f"Best guess: {best_label} ({best_prob:.4f})"
        )
    else:
        print(f"\nPredicted class: {best_label} ({best_prob:.4f})")


if __name__ == "__main__":
    infer()
