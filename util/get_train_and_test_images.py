# Wiederverwendbare Funktion leicht anpassen für Gleichverteilung auch im Test
import os
import random
import shutil
from pathlib import Path

split_ratio = 0.8
labels = ["real", "fake"]
target_base = "../data"

sources = {
    "celebdf": "../data/images/celeb_df",
    "dfd": "../data/images/dfd",
    "faceforensics": "../data/images/faceforensics"
}

experiments = {
    "celebdf_only": {
        "split": ["celebdf"],
        "test_only": []
    },
    "celebdf_ffpp": {
        "split": ["celebdf", "faceforensics"],
        "test_only": ["dfd"]
    }
}

# Hilfsfunktionen
def get_images(source_dir, label):
    path = Path(source_dir) / label
    if not path.exists():
        return []
    return sorted([p for p in path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

def copy_images(images, dest_dir, max_count=None):
    os.makedirs(dest_dir, exist_ok=True)
    if max_count:
        images = images[:max_count]
    for img in images:
        shutil.copy(img, os.path.join(dest_dir, img.name))
    return len(images)

# Neue Statistik mit gleichverteiltem Test
from collections import defaultdict
stats = defaultdict(lambda: defaultdict(int))

for exp_name, cfg in experiments.items():
    exp_dir = Path(target_base) / exp_name

    for source in cfg["split"]:
        for label in labels:
            all_images = get_images(sources[source], label)
            random.shuffle(all_images)

            split_idx = int(len(all_images) * split_ratio)
            train_imgs = all_images[:split_idx]
            test_imgs = all_images[split_idx:]

            if label == "real":
                real_train = len(train_imgs)
                real_test = len(test_imgs)

            # TRAIN
            dest_train = exp_dir / "train" / label
            if label == "fake":
                train_imgs = all_images[:min(len(all_images), real_train)]
            stats[(exp_name, "train")][label] += copy_images(train_imgs, dest_train)

            # TEST
            dest_test = exp_dir / "test" / label
            if label == "fake":
                test_imgs = all_images[:min(len(all_images), real_test)]
            stats[(exp_name, "test")][label] += copy_images(test_imgs, dest_test)

    for source in cfg["test_only"]:
        real_images = get_images(sources[source], "real")
        fake_images = get_images(sources[source], "fake")
        random.shuffle(real_images)
        random.shuffle(fake_images)
        limit = len(real_images)
        dest_base = Path(target_base) / exp_name / "test"
        stats[(exp_name, "test")]["real"] += copy_images(real_images, dest_base / "real")
        stats[(exp_name, "test")]["fake"] += copy_images(fake_images, dest_base / "fake", max_count=limit)
