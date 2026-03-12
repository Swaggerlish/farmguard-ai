import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PLANTVILLAGE_DATASET = "abdallahalidev/plantvillage-dataset"
CASSAVA_COMPETITION = "cassava-leaf-disease-classification"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_IMAGES_PER_CLASS = 600
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)

CASSAVA_LABEL_MAP = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mite",
    3: "cassava_mosaic_disease",
    4: "cassava_healthy",
}

REQUIRED_CASSAVA_CLASSES = set(CASSAVA_LABEL_MAP.values())

CASSAVA_CLASS_ALIASES = {
    "cassava_bacterial_blight": "cassava_bacterial_blight",
    "cassava_brown_streak_disease": "cassava_brown_streak_disease",
    "cassava_green_mite": "cassava_green_mite",
    "cassava_mosaic_disease": "cassava_mosaic_disease",
    "cassava_healthy": "cassava_healthy",
    "cbb": "cassava_bacterial_blight",
    "cbsd": "cassava_brown_streak_disease",
    "cgm": "cassava_green_mite",
    "cmd": "cassava_mosaic_disease",
    "healthy": "cassava_healthy",
}


def normalize_name(name: str) -> str:
    normalized = name.lower()
    for ch in [" ", "-", ",", "(", ")", "__", "___"]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clear_processed_dir() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_name: str) -> bool:
    print(f"Downloading dataset: {dataset_name}")
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
    except Exception as exc:
        print(f"Warning: failed to download dataset '{dataset_name}': {exc}")
        return False


def download_kaggle_competition() -> bool:
    print(f"Downloading cassava competition dataset: {CASSAVA_COMPETITION}")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                CASSAVA_COMPETITION,
                "-p",
                str(RAW_DIR),
            ],
            check=True,
        )

        for archive_path in RAW_DIR.glob("*.zip"):
            try:
                shutil.unpack_archive(str(archive_path), str(RAW_DIR))
                print(f"Extracted {archive_path}")
            except Exception as exc:
                print(f"Warning: could not extract {archive_path}: {exc}")

        return True
    except Exception as exc:
        print(f"Could not download cassava competition dataset: {exc}")
        print("Will attempt to use manually uploaded data from data/raw.")
        return False


def get_image_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


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
        if destination.exists():
            destination = target_dir / f"{img_path.stem}_{abs(hash(str(img_path)))}{img_path.suffix}"
        shutil.copy2(img_path, destination)


def process_class_images(image_paths: list[Path], class_name: str) -> None:
    unique_paths = sorted(set(image_paths))

    if len(unique_paths) > MAX_IMAGES_PER_CLASS:
        unique_paths = random.sample(unique_paths, MAX_IMAGES_PER_CLASS)

    train_paths, val_paths, test_paths = split_image_paths(unique_paths)

    copy_split(train_paths, "train", class_name)
    copy_split(val_paths, "val", class_name)
    copy_split(test_paths, "test", class_name)

    print(
        f"Prepared {class_name}: "
        f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
    )


def score_csv_candidate(csv_path: Path, image_dir: Path) -> int:
    score = 0

    if csv_path.parent == image_dir.parent:
        score += 100

    try:
        df = pd.read_csv(csv_path, nrows=20)
        cols = set(df.columns)

        if {"image_id", "label"}.issubset(cols):
            score += 50

        if "image_id" in df.columns:
            existing = 0
            for image_id in df["image_id"].dropna().astype(str).tolist()[:10]:
                if (image_dir / image_id).exists():
                    existing += 1
            score += existing * 10
    except Exception:
        pass

    depth_penalty = len(csv_path.relative_to(RAW_DIR).parts)
    score -= depth_penalty

    return score


def find_cassava_assets_csv() -> tuple[Path, Path] | None:
    csv_candidates = list(RAW_DIR.rglob("train.csv"))
    image_dir_candidates = [p for p in RAW_DIR.rglob("train_images") if p.is_dir()]

    best_match: tuple[int, Path, Path] | None = None

    for csv_path in csv_candidates:
        for image_dir in image_dir_candidates:
            score = score_csv_candidate(csv_path, image_dir)
            if best_match is None or score > best_match[0]:
                best_match = (score, csv_path, image_dir)

    if best_match is None:
        return None

    _, csv_path, image_dir = best_match
    print(f"Detected cassava CSV dataset:")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {image_dir}")
    return csv_path, image_dir


def find_cassava_assets_folder() -> Path | None:
    aliases = {normalize_name(k) for k in CASSAVA_CLASS_ALIASES.keys()}
    best_match: tuple[int, Path] | None = None

    for folder in RAW_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        child_dirs = [child for child in folder.iterdir() if child.is_dir()]
        child_names = {normalize_name(child.name) for child in child_dirs}
        overlap = len(child_names & aliases)

        if overlap < 2:
            continue

        image_count = 0
        for child in child_dirs:
            if normalize_name(child.name) in aliases:
                image_count += len(get_image_files(child))

        score = overlap * 100 + image_count

        if best_match is None or score > best_match[0]:
            best_match = (score, folder)

    if best_match:
        print(f"Detected cassava folder dataset root: {best_match[1]}")
        return best_match[1]

    return None


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

        for image_id in class_df["image_id"].astype(str).tolist():
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
        print(f"Warning: missing cassava classes: {sorted(missing_classes)}")


def prepare_cassava_from_folders(cassava_root: Path) -> None:
    print(f"Using cassava folder root: {cassava_root}")

    class_to_images: dict[str, list[Path]] = {}

    for class_dir in sorted(cassava_root.iterdir()):
        if not class_dir.is_dir():
            continue

        alias = normalize_name(class_dir.name)
        target_class = CASSAVA_CLASS_ALIASES.get(alias)
        if not target_class:
            continue

        image_paths = get_image_files(class_dir)
        if not image_paths:
            print(f"Warning: no images found in {class_dir}")
            continue

        class_to_images.setdefault(target_class, []).extend(image_paths)

    if not class_to_images:
        raise FileNotFoundError(
            "No cassava class folders matched expected aliases under detected root."
        )

    for target_class, image_paths in sorted(class_to_images.items()):
        process_class_images(image_paths, target_class)


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

    raise RuntimeError(
        "Cassava dataset not found under data/raw. Expected either a nested "
        "train.csv + train_images dataset or class folders like cbb/cbsd/cgm/cmd/healthy."
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
    print(f"PlantVillage dataset: {PLANTVILLAGE_DATASET}")
    print(f"Cassava competition: {CASSAVA_COMPETITION}")

    ensure_dirs()
    clear_processed_dir()

    print("\nDownloading PlantVillage...")
    download_kaggle_dataset(PLANTVILLAGE_DATASET)

    print("\nDownloading Cassava competition...")
    download_kaggle_competition()

    print("\nPreparing cassava dataset...")
    prepare_cassava()

    print_dataset_summary()
    print(f"\nDataset ready at: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
