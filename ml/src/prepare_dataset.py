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

PLANTVILLAGE_DATASET = "abdallahalidev/plantvillage-dataset"
CASSAVA_KAGGLEHUB_DATASET = os.getenv("CASSAVA_KAGGLEHUB_DATASET", "visalakshiiyer/cassava-image-dataset")

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
    # tomato (original naming)
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___healthy": "tomato_healthy",
    # tomato (observed variant naming)
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
CASSAVA_LABEL_MAP = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mite",
    3: "cassava_mosaic_disease",
    4: "cassava_healthy",
}

CASSAVA_CLASS_ALIASES = {
    "cassava_bacterial_blight": "cassava_bacterial_blight",
    "cassava_brown_streak_disease": "cassava_brown_streak_disease",
    "cassava_green_mite": "cassava_green_mite",
    "cassava_mosaic_disease": "cassava_mosaic_disease",
    "cassava_healthy": "cassava_healthy",
    # common short aliases in external datasets
    "cbb": "cassava_bacterial_blight",
    "cbsd": "cassava_brown_streak_disease",
    "cgm": "cassava_green_mite",
    "cmd": "cassava_mosaic_disease",
    "healthy": "cassava_healthy",
    # verbose folder names seen in manually downloaded datasets
    "cassava_cb_cassava_blight": "cassava_bacterial_blight",
    "cassava_cm_cassava_mosaic": "cassava_mosaic_disease",
    "cassava_healthy_leaf": "cassava_healthy",
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clear_processed_dir() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_name: str) -> bool:
    print(f"Downloading dataset {dataset_name}...")
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


def download_kagglehub_cassava() -> bool:
    """
    Download cassava data via kagglehub dataset API (non-competition flow).
    If this fails, we continue and rely on already present files in data/raw.
    """
    print(f"Downloading cassava dataset via kagglehub: {CASSAVA_KAGGLEHUB_DATASET}")

    try:
        import kagglehub
    except ImportError:
        print("Warning: kagglehub is not installed; skipping cassava auto-download.")
        print("Run 'pip install kagglehub' or rely on manually uploaded cassava files in data/raw.")
        return True

    try:
        source_path = Path(kagglehub.dataset_download(CASSAVA_KAGGLEHUB_DATASET))
    except Exception as exc:
        print("Warning: failed to download cassava dataset via kagglehub.")
        print(f"kagglehub error: {exc}")
        return True

    if not source_path.exists():
        print(f"Warning: kagglehub returned missing path: {source_path}")
        return True

    destination = RAW_DIR / "kagglehub_cassava"
    if destination.exists():
        shutil.rmtree(destination)

    if source_path.is_dir():
        shutil.copytree(source_path, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination / source_path.name)

    # Some kagglehub datasets may include zipped assets; extract them for discovery.
    for archive_path in destination.rglob("*.zip"):
        extract_dir = archive_path.parent / archive_path.stem
        if extract_dir.exists():
            continue
        try:
            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted {archive_path} -> {extract_dir}")
        except Exception as exc:
            print(f"Warning: failed to extract {archive_path}: {exc}")

    print(f"Cassava dataset downloaded to: {destination}")
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
    best_match: tuple[int, Path] | None = None

    for folder in RAW_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        child_dirs = {normalize_name(child.name) for child in folder.iterdir() if child.is_dir()}
        overlap = len(child_dirs & expected)

        if overlap >= 2 and (best_match is None or overlap > best_match[0]):
            best_match = (overlap, folder)

    if best_match:
        return best_match[1]

    if best_match:
        return best_match[2]

    raise FileNotFoundError(
        "Could not find a PlantVillage root containing maize/corn folders."
    )


def prepare_plantvillage() -> None:
    plantvillage_root = find_plantvillage_root()
    print(f"Using PlantVillage root: {plantvillage_root}")

    class_to_images: dict[str, list[Path]] = {}

    for class_dir in sorted(plantvillage_root.iterdir()):
        if not class_dir.is_dir():
            continue

        normalized = normalize_name(class_dir.name)
        target_class = PLANTVILLAGE_NORMALIZED_MAP.get(normalized)
        if not target_class:
            continue

        image_paths = get_image_files(class_dir)
        if not image_paths:
            print(f"Warning: no images found in {class_dir}")
            continue

        class_to_images.setdefault(target_class, []).extend(image_paths)

    if not class_to_images:
        raise FileNotFoundError(
            "No PlantVillage class folders matched expected mappings under detected root."
        )

    for target_class, image_paths in sorted(class_to_images.items()):
        unique_paths = sorted(set(image_paths))
        process_class_images(unique_paths, target_class)


def find_cassava_assets_csv() -> tuple[Path, Path] | None:
    csv_candidates = list(RAW_DIR.rglob("train.csv"))
    image_dir_candidates = [path for path in RAW_DIR.rglob("train_images") if path.is_dir()]

    if not csv_candidates or not image_dir_candidates:
        return None

    for target_class, image_paths in sorted(class_to_images.items()):
        unique_paths = sorted(set(image_paths))
        process_class_images(unique_paths, target_class)


def find_cassava_assets_folder() -> Path | None:
    aliases = {normalize_name(k) for k in CASSAVA_CLASS_ALIASES.keys()}

    best_match: tuple[int, Path] | None = None
    for folder in RAW_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        child_dirs = {normalize_name(child.name) for child in folder.iterdir() if child.is_dir()}
        overlap = len(child_dirs & aliases)

        if overlap >= 2 and (best_match is None or overlap > best_match[0]):
            best_match = (overlap, folder)

    return best_match[1] if best_match else None


def prepare_cassava_from_csv(train_csv_path: Path, train_images_dir: Path) -> None:
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


def prepare_cassava_from_folders(cassava_root: Path) -> None:
    print(f"Using cassava folder root: {cassava_root}")

    found_any = False
    for class_dir in sorted(cassava_root.iterdir()):
        if not class_dir.is_dir():
            continue

        alias = normalize_name(class_dir.name)
        target = CASSAVA_CLASS_ALIASES.get(alias)
        if not target:
            continue

        image_paths = get_image_files(class_dir)
        if not image_paths:
            continue

        found_any = True
        process_class_images(image_paths, target)

    if not found_any:
        raise FileNotFoundError(
            "No cassava class folders matched expected aliases under cassava root."
        )


def prepare_cassava() -> None:
    csv_assets = find_cassava_assets_csv()
    if csv_assets:
        train_csv_path, train_images_dir = csv_assets
        prepare_cassava_from_csv(train_csv_path, train_images_dir)
        return

    folder_root = find_cassava_assets_folder()
    if folder_root:
        prepare_cassava_from_folders(folder_root)
        return

    raise FileNotFoundError(
        "Could not find cassava assets. Expected either train.csv + train_images "
        "or class-folder formatted cassava dataset in data/raw. "
        "Tip: rename class folders to aliases like cbb/cmd/healthy when using custom datasets."
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
    cassava_download_ok = download_kagglehub_cassava()

    if plantvillage_ok:
        print("\nPreparing PlantVillage classes...")
        prepare_plantvillage()
    else:
        print("\nSkipping PlantVillage preparation due to download failure.")

    if cassava_download_ok:
        print("\nPreparing cassava classes...")
        try:
            prepare_cassava()
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            print("Skipping cassava preparation.")
    else:
        print("\nSkipping cassava preparation due to download failure.")

    print_dataset_summary()
    print("\nDataset preparation complete.")
    print(f"Processed dataset saved to: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
