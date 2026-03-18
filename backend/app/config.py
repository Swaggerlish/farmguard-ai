import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "swaggerlish/farmgaurd-ai-crop-disease")
    HF_MODEL_FILENAME: str = os.getenv("HF_MODEL_FILENAME", "best_model.pth")
    HF_CLASSMAP_FILENAME: str = os.getenv("HF_CLASSMAP_FILENAME", "class_to_idx.json")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))


settings = Settings()