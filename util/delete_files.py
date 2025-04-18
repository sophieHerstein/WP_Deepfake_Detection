import os
import random

# Zielanzahl pro Split
target_count = {
    "train": 1500,
    "test": 450,
    "person": 25,
    "none": 0
}

# Passe diese Liste an deine lokalen Ordner an
base_dirs = [
    "data/celeb_df/train/real",
    "data/celeb_df/train/fake",
    "data/celeb_df/test/real",
    "data/celeb_df/test/fake",
    "data/faceforensics/fake",
    "data/faceforensics/real",
    "data/custom_test/real",
    "data/custom_test/fake",
    "data/person"
]

# Funktion zur Reduktion
def reduce_images(folder, target):
    print(f"📂 Verarbeite: {folder}")
    all_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
    current_count = len(all_files)
    print(f"   Vorher: {current_count} Dateien")

    if current_count <= target:
        print("   ✔️ Keine Reduktion nötig.")
        return

    to_delete = random.sample(all_files, current_count - target)
    for file in to_delete:
        os.remove(os.path.join(folder, file))

    print(f"   🗑️ {len(to_delete)} Dateien gelöscht. Verbleibend: {target}")

# Hauptlogik
for path in base_dirs:
    split_type = "none"
    if "train" in path:
        split_type = 'train'
    elif "person" in path:
        split_type = 'person'
    else:
        split_type = 'test'
    target = target_count[split_type]
    reduce_images(path, target)

print("✅ Datensatzreduktion abgeschlossen.")