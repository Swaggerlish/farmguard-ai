import json
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = "outputs/checkpoints/best_model.pth"
DEFAULT_CLASS_MAP = "outputs/checkpoints/class_to_idx.json"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_exists(path: str | Path, name: str = "path") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    return p


def load_class_names(class_map_path: str | Path) -> list[str]:
    p = ensure_exists(class_map_path, "class map")
    with p.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)
