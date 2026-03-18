import json
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.config import settings


# Mapping from raw class labels to friendly disease names by language
DISEASE_NAME_MAPPING = {
    "english": {
        "cassava_bacterial_blight": "Bacterial Blight",
        "cassava_brown_streak_disease": "Brown Streak Disease",
        "cassava_green_mite": "Green Mite",
        "cassava_healthy": "Healthy",
        "cassava_mosaic_disease": "Mosaic Disease",
        "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
        "maize_common_rust": "Common Rust",
        "maize_healthy": "Healthy",
        "maize_northern_leaf_blight": "Northern Leaf Blight",
        "pepper_bacterial_spot": "Bacterial Spot",
        "pepper_healthy": "Healthy",
        "tomato_early_blight": "Early Blight",
        "tomato_healthy": "Healthy",
        "tomato_late_blight": "Late Blight",
        "tomato_mosaic_virus": "Mosaic Virus",
        "tomato_septoria_leaf_spot": "Septoria Leaf Spot",
        "tomato_target_spot": "Target Spot",
        "tomato_yellow_leaf_curl_virus": "Yellow Leaf Curl Virus",
    },
    "pidgin": {
        "cassava_bacterial_blight": "Bacterial Blight",
        "cassava_brown_streak_disease": "Brown Streak Disease",
        "cassava_green_mite": "Green Mite",
        "cassava_healthy": "Healthy",
        "cassava_mosaic_disease": "Mosaic Disease",
        "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
        "maize_common_rust": "Common Rust",
        "maize_healthy": "Healthy",
        "maize_northern_leaf_blight": "Northern Leaf Blight",
        "pepper_bacterial_spot": "Bacterial Spot",
        "pepper_healthy": "Healthy",
        "tomato_early_blight": "Early Blight",
        "tomato_healthy": "Healthy",
        "tomato_late_blight": "Late Blight",
        "tomato_mosaic_virus": "Mosaic Virus",
        "tomato_septoria_leaf_spot": "Septoria Leaf Spot",
        "tomato_target_spot": "Target Spot",
        "tomato_yellow_leaf_curl_virus": "Yellow Leaf Curl Virus",
    },
    "yoruba": {
        "cassava_bacterial_blight": "Bacterial Blight",
        "cassava_brown_streak_disease": "Brown Streak Disease",
        "cassava_green_mite": "Green Mite",
        "cassava_healthy": "Ni Ilera",
        "cassava_mosaic_disease": "Mosaic Disease",
        "maize_cercospora_leaf_spot": "Cercospora Leaf Spot",
        "maize_common_rust": "Common Rust",
        "maize_healthy": "Ni Ilera",
        "maize_northern_leaf_blight": "Northern Leaf Blight",
        "pepper_bacterial_spot": "Bacterial Spot",
        "pepper_healthy": "Ni Ilera",
        "tomato_early_blight": "Early Blight",
        "tomato_healthy": "Ni Ilera",
        "tomato_late_blight": "Late Blight",
        "tomato_mosaic_virus": "Mosaic Virus",
        "tomato_septoria_leaf_spot": "Septoria Leaf Spot",
        "tomato_target_spot": "Target Spot",
        "tomato_yellow_leaf_curl_virus": "Yellow Leaf Curl Virus",
    }
}


class ModelService:
    def __init__(self):
        self.model = None
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self, num_classes: int):
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        return model

    def load(self):
        os.makedirs(settings.MODEL_DIR, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=settings.HF_REPO_ID,
            filename=settings.HF_MODEL_FILENAME,
            local_dir=settings.MODEL_DIR,
        )

        classmap_path = hf_hub_download(
            repo_id=settings.HF_REPO_ID,
            filename=settings.HF_CLASSMAP_FILENAME,
            local_dir=settings.MODEL_DIR,
        )

        with open(classmap_path, "r") as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = self._build_model(num_classes=len(self.class_to_idx))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def get_friendly_disease_name(self, raw_label: str, language: str = "english") -> str:
        """Convert raw class label to friendly disease name."""
        lang_mapping = DISEASE_NAME_MAPPING.get(language.lower(), DISEASE_NAME_MAPPING["english"])
        return lang_mapping.get(raw_label, raw_label.replace("_", " ").title())

    def predict(self, image_tensor) -> Tuple[str, float]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)

        predictions = []
        for i in range(3):
            label = self.idx_to_class[int(top3_indices[0][i].item())]
            confidence = float(top3_probs[0][i].item())
            predictions.append((label, confidence))

        return predictions


model_service = ModelService()