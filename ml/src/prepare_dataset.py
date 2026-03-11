import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Full PlantVillage mirror that includes maize/corn
PLANTVILLAGE_DATASET = os.getenv(
    "PLANTVILLAGE_DATASET",
    "abdallahalidev/plantvillage-dataset",
)

# Kaggle competition name for cassava
CASSAVA_COMPETITION = os.getenv(
    "CASSAVA_COMPETITION",
    "cassava-leaf-disease-classification",
)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_IMAGES_PER_CLASS = 600
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)


def normalize_name(name: str) -> str:
    normalized = name.lower()
    for ch in [" ", "-", ",", "(", ")", "__", "___"]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


PLANTVILLAGE_CLASS_MAP = {
    # maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_": "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "maize_northern_leaf_blight",
    "Corn_(maize)___healthy": "maize_healthy",

    # tomato
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___healthy": "tomato_healthy",

    # tomato naming variants
    "Tomato_Early_blight": "tomato_early_blight",
    "Tomato_Late_blight": "tomato_late_blight",
    "Tomato_Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato__Target_Spot": "tomato_target_spot",
    "Tomato__Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato_healthy": "tomato_healthy",

    # pepper
    "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper,_bell___healthy": "pepper_healthy",
    "Pepper__bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper__bell___healthy": "pepper_healthy",
}

PLANTVILLAGE_NORMALIZED_MAP = {
    normalize_name(source): target
    for source, target in PLANTVILLAGE_CLASS_MAP.items()
}

REQUIRED_MAIZE_CLASSES = {
    "maize_cercospora_leaf_spot",
    "maize_common_rust",
    "maize_northern_leaf_blight",
    "maize_healthy",
}

CASSAVA_LABEL_MAP = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mite",
    3: "cassava_mosaic_disease",
    4: "cassava_healthy",
}

REQUIRED_CASSAVA_CLASSES = set(CASSAVA_LABEL_MAP.values())


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clear_processed_dir() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clear_previous_raw_downloads() -> None:
    """
    Optional cleanup to reduce the chance of mixing old and new raw assets.
    """
    for path in [
        RAW_DIR / "train_images",
        RAW_DIR / "test_images",
        RAW_DIR / "sample_submission.csv",
        RAW_DIR / "train.csv",
        RAW_DIR / f"{CASSAVA_COMPETITION}.zip",
    ]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def download_kaggle_dataset(dataset_name: str) -> bool:
    print(f"Downloading Kaggle dataset: {dataset_name}")
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "-p",
                str(RAW_DIR),
                "--unzip",
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Warning: failed to download dataset '{dataset_name}'.")
        print(f"Kaggle CLI error: {exc}")
        return False


def download_kaggle_competition_dataset(competition_name: str) -> bool:
    print(f"Downloading Kaggle competition dataset: {competition_name}")

    zip_path = RAW_DIR / f"{competition_name}.zip"

    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition_name,
                "-p",
                str(RAW_DIR),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Warning: failed to download competition '{competition_name}'.")
        print(f"Kaggle CLI error: {exc}")
        return False

    if not zip_path.exists():
        print(f"Warning: expected archive not found: {zip_path}")
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
    except zipfile.BadZipFile as exc:
        print(f"Warning: invalid zip archive for competition dataset: {exc}")
        return False

    print(f"Extracted cassava competition archive to: {RAW_DIR}")
    return True


def get_image_files(folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def sample_images(image_paths: list[Path], max_images: int) -> list[Path]:
    if len(image_paths) <= max_images:
        return image_paths
    return random.sample(image_paths, max_images)


def split_image_paths(image_paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    if len(image_paths) < 3:
        raise ValueError(f"Need at least 3 images to split, found {len(image_paths)}")

    train_paths, temp_paths = train_test_split(
        image_paths,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=relative_test_size,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return train_paths, val_paths, test_paths


def copy_split(paths: list[Path], split_name: str, class_name: str) -> None:
    target_dir = PROCESSED_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for img_path in paths:
        destination = target_dir / img_path.name

        # Avoid overwriting if duplicate filenames exist.
        if destination.exists():
            destination = target_dir / f"{img_path.stem}_{abs(hash(str(img_path)))}{img_path.suffix}"

        shutil.copy2(img_path, destination)


def process_class_images(image_paths: list[Path], class_name: str) -> None:
    image_paths = sample_images(image_paths, MAX_IMAGES_PER_CLASS)
    train_paths, val_paths, test_paths = split_image_paths(image_paths)

    copy_split(train_paths, "train", class_name)
    copy_split(val_paths, "val", class_name)
    copy_split(test_paths, "test", class_name)

    print(
        f"Prepared {class_name}: "
        f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
    )


def find_plantvillage_root() -> Path:
    expected = {normalize_name(name) for name in PLANTVILLAGE_CLASS_MAP.keys()}
    maize_expected = {
        normalize_name("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
        normalize_name("Corn_(maize)___Common_rust_"),
        normalize_name("Corn_(maize)___Northern_Leaf_Blight"),
        normalize_name("Corn_(maize)___healthy"),
    }

    best_match: tuple[int, int, Path] | None = None

    for folder in RAW_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        child_dirs = {normalize_name(child.name) for child in folder.iterdir() if child.is_dir()}
        total_overlap = len(child_dirs & expected)
        maize_overlap = len(child_dirs & maize_expected)

        if maize_overlap == 0:
            continue

        score = (maize_overlap, total_overlap)
        if best_match is None or score > (best_match[0], best_match[1]):
            best_match = (maize_overlap, total_overlap, folder)

    if best_match:
        return best_match[2]

    raise FileNotFoundError(
        "Could not find a PlantVillage root containing maize/corn folders."
    )


def prepare_plantvillage() -> None:
    plantvillage_root = find_plantvillage_root()
    print(f"Using PlantVillage root: {plantvillage_root}")

    class_to_images: dict[str, list[Path]] = {}
    detected_source_folders: list[str] = []

    for class_dir in sorted(plantvillage_root.iterdir()):
        if not class_dir.is_dir():
            continue

        detected_source_folders.append(class_dir.name)

        normalized = normalize_name(class_dir.name)
        target_class = PLANTVILLAGE_NORMALIZED_MAP.get(normalized)
        if not target_class:
            continue

        image_paths = get_image_files(class_dir)
        if not image_paths:
            print(f"Warning: no images found in {class_dir}")
            continue

        class_to_images.setdefault(target_class, []).extend(image_paths)

    print("\nDetected PlantVillage source folders:")
    for folder_name in detected_source_folders:
        print(f"  - {folder_name}")

    if not class_to_images:
        raise FileNotFoundError(
            "No PlantVillage class folders matched expected mappings under detected root."
        )

    found_classes = set(class_to_images.keys())
    missing_maize = REQUIRED_MAIZE_CLASSES - found_classes

    print("\nMatched PlantVillage output classes:")
    for class_name in sorted(found_classes):
        print(f"  - {class_name}")

    if missing_maize:
        raise RuntimeError(
            "PlantVillage dataset is missing required maize classes: "
            f"{sorted(missing_maize)}"
        )

    for target_class, image_paths in sorted(class_to_images.items()):
        unique_paths = sorted(set(image_paths))
        process_class_images(unique_paths, target_class)


def find_cassava_competition_assets() -> tuple[Path, Path]:
    train_csv_path = RAW_DIR / "train.csv"
    train_images_dir = RAW_DIR / "train_images"

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Cassava train.csv not found at: {train_csv_path}")

    if not train_images_dir.exists() or not train_images_dir.is_dir():
        raise FileNotFoundError(f"Cassava train_images folder not found at: {train_images_dir}")

    return train_csv_path, train_images_dir


def prepare_cassava_from_competition(train_csv_path: Path, train_images_dir: Path) -> None:
    print(f"Using cassava CSV: {train_csv_path}")
    print(f"Using cassava images dir: {train_images_dir}")

    df = pd.read_csv(train_csv_path)

    required_columns = {"image_id", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Cassava CSV must contain columns: {required_columns}")

    found_classes: set[str] = set()

    for label_id, class_name in CASSAVA_LABEL_MAP.items():
        class_df = df[df["label"] == label_id]
        image_paths: list[Path] = []

        for image_id in class_df["image_id"].tolist():
            img_path = train_images_dir / image_id
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(img_path)

        if not image_paths:
            print(f"Warning: no cassava images found for class {class_name}")
            continue

        found_classes.add(class_name)
        process_class_images(image_paths, class_name)

    missing_classes = REQUIRED_CASSAVA_CLASSES - found_classes
    if missing_classes:
        raise RuntimeError(
            "Cassava competition dataset is missing required classes: "
            f"{sorted(missing_classes)}"
        )


def print_dataset_summary() -> None:
    print("\nFinal dataset summary:")
    for split_name in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split_name
        if not split_dir.exists():
            print(f"{split_name}: missing")
            continue

        total_images = 0
        class_counts: dict[str, int] = {}

        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(get_image_files(class_dir))
                class_counts[class_dir.name] = count
                total_images += count

        print(f"\n{split_name.upper()} ({total_images} images)")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")


def main() -> None:
    print(f"PLANTVILLAGE_DATASET in use: {PLANTVILLAGE_DATASET}")
    print(f"CASSAVA_COMPETITION in use: {CASSAVA_COMPETITION}")

    ensure_dirs()
    clear_processed_dir()
    clear_previous_raw_downloads()

    plantvillage_ok = download_kaggle_dataset(PLANTVILLAGE_DATASET)
    cassava_ok = download_kaggle_competition_dataset(CASSAVA_COMPETITION)

    if not plantvillage_ok:
        raise RuntimeError("PlantVillage download failed; cannot prepare training data.")

    print("\nPreparing PlantVillage classes...")
    prepare_plantvillage()

    if not cassava_ok:
        raise RuntimeError("Cassava competition download failed; cannot prepare cassava data.")

    print("\nPreparing cassava classes...")
    train_csv_path, train_images_dir = find_cassava_competition_assets()
    prepare_cassava_from_competition(train_csv_path, train_images_dir)

    print_dataset_summary()
    print("\nDataset preparation complete.")
    print(f"Processed dataset saved to: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
