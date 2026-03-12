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
TRAIN_RATIO = 0.7
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

CASSAVA_CLASS_ALIASES = {
    "cbb": "cassava_bacterial_blight",
    "cbsd": "cassava_brown_streak_disease",
    "cgm": "cassava_green_mite",
    "cmd": "cassava_mosaic_disease",
    "healthy": "cassava_healthy",
}


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clear_processed_dir():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset):
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(RAW_DIR),
                "--unzip",
            ],
            check=True,
        )
        return True
    except Exception:
        return False


def download_kaggle_competition():
    print("Downloading cassava competition dataset...")

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

        for zipfile in RAW_DIR.glob("*.zip"):
            shutil.unpack_archive(zipfile, RAW_DIR)

        return True

    except Exception:
        print("Could not download cassava competition dataset.")
        print("Will attempt to use manually uploaded data.")
        return False


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_paths(paths):

    train, temp = train_test_split(
        paths,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=RANDOM_SEED,
    )

    return train, val, test


def copy_images(paths, split, classname):

    target = PROCESSED_DIR / split / classname
    target.mkdir(parents=True, exist_ok=True)

    for img in paths:
        shutil.copy2(img, target / img.name)


def process_class(images, classname):

    if len(images) > MAX_IMAGES_PER_CLASS:
        images = random.sample(images, MAX_IMAGES_PER_CLASS)

    train, val, test = split_paths(images)

    copy_images(train, "train", classname)
    copy_images(val, "val", classname)
    copy_images(test, "test", classname)

    print(f"{classname}: train={len(train)} val={len(val)} test={len(test)}")


def prepare_cassava_from_csv():

    csv_files = list(RAW_DIR.rglob("train.csv"))
    image_dirs = list(RAW_DIR.rglob("train_images"))

    if not csv_files or not image_dirs:
        return False

    csv_path = csv_files[0]
    image_dir = image_dirs[0]

    df = pd.read_csv(csv_path)

    for label, classname in CASSAVA_LABEL_MAP.items():

        subset = df[df.label == label]

        images = []

        for imgname in subset.image_id:
            p = image_dir / imgname
            if p.exists():
                images.append(p)

        if images:
            process_class(images, classname)

    return True


def prepare_cassava_from_folders():

    for folder in RAW_DIR.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if name not in CASSAVA_CLASS_ALIASES:
            continue

        classname = CASSAVA_CLASS_ALIASES[name]

        images = get_images(folder)

        if images:
            process_class(images, classname)
            return True

    return False


def prepare_cassava():

    if prepare_cassava_from_csv():
        return

    if prepare_cassava_from_folders():
        return

    raise RuntimeError(
        "Cassava dataset not found. Upload it manually to data/raw."
    )


def print_summary():

    print("\nDATASET SUMMARY")

    for split in ["train", "val", "test"]:

        splitdir = PROCESSED_DIR / split

        if not splitdir.exists():
            continue

        total = 0

        print(f"\n{split.upper()}")

        for c in splitdir.iterdir():

            count = len(list(c.iterdir()))
            total += count

            print(c.name, count)

        print("TOTAL:", total)


def main():

    print("PlantVillage dataset:", PLANTVILLAGE_DATASET)
    print("Cassava competition:", CASSAVA_COMPETITION)

    ensure_dirs()
    clear_processed_dir()

    print("\nDownloading PlantVillage...")
    download_kaggle_dataset(PLANTVILLAGE_DATASET)

    print("\nDownloading Cassava competition...")
    download_kaggle_competition()

    print("\nPreparing cassava dataset...")
    prepare_cassava()

    print_summary()

    print("\nDataset ready at:", PROCESSED_DIR.resolve())


if __name__ == "__main__":
    main()
