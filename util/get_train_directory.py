import os
import shutil
import csv

# 🔧 HIER deine Quellpfade eintragen:
sources = {
    "celeb_train_real": "../data/celeb_df/train/real",
    "celeb_train_fake": "../data/celeb_df/train/fake",
    "kaggle_train_fake": "../data/kaggle_synthetic/train/fake",
    "kaggle_train_real": "../data/kaggle_synthetic/train/real",
    "celeb_test_real": "../data/celeb_df/test/real",
    "celeb_test_fake": "../data/celeb_df/test/fake",
    "kaggle_test_fake": "../data/kaggle_synthetic/test/fake",
    "kaggle_test_real": "../data/kaggle_synthetic/test/real",
}

# 📁 Zielordner – hier wird alles zusammengeführt
target_train_root = "../data/combined_train/train"
target_test_root = "../data/combined_train/test"
target_train_real = os.path.join(target_train_root, "real")
target_test_real = os.path.join(target_test_root, "real")
target_train_fake = os.path.join(target_train_root, "fake")
target_test_fake = os.path.join(target_test_root, "fake")
os.makedirs(target_train_real, exist_ok=True)
os.makedirs(target_test_real, exist_ok=True)
os.makedirs(target_train_fake, exist_ok=True)
os.makedirs(target_test_fake, exist_ok=True)

# 📄 CSV-Datei zur Dokumentation der Quelle
csv_train_path = os.path.join(target_train_root, "image_train_sources.csv")
csv_test_path = os.path.join(target_test_root, "image_test_sources.csv")
csv_test_entries = []
csv_train_entries = []

# 📦 Funktion zum Kopieren + Umbenennen + Loggen
def copy_with_prefix_train(src_folder, dst_folder, prefix, label):
    if not os.path.exists(src_folder):
        print(f"⚠️ Train Ordner nicht gefunden: {src_folder}")
        return

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, file in enumerate(files):
        new_name = f"{prefix}_{i+1:04}.jpg"
        shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, new_name))
        csv_train_entries.append([new_name, label, prefix])

# 🏁 Kopiervorgang starten
copy_with_prefix_train(sources["celeb_train_real"], target_train_real, "celeb_train_real", "real")
copy_with_prefix_train(sources["celeb_train_fake"], target_train_fake, "celeb_train_fake", "fake")
copy_with_prefix_train(sources["kaggle_train_fake"], target_train_fake, "kaggle_train_fake", "fake")
copy_with_prefix_train(sources["kaggle_train_real"], target_train_fake, "kaggle_train_real", "real")

# ✍️ CSV-Datei schreiben
with open(csv_train_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label", "source"])
    writer.writerows(csv_train_entries)

# 📦 Funktion zum Kopieren + Umbenennen + Loggen
def copy_with_prefix_test(src_folder, dst_folder, prefix, label):
    if not os.path.exists(src_folder):
        print(f"⚠️ Test Ordner nicht gefunden: {src_folder}")
        return

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, file in enumerate(files):
        new_name = f"{prefix}_{i+1:04}.jpg"
        shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, new_name))
        csv_test_entries.append([new_name, label, prefix])

# 🏁 Kopiervorgang starten
copy_with_prefix_test(sources["celeb_test_real"], target_test_real, "celeb_test_real", "real")
copy_with_prefix_test(sources["celeb_test_fake"], target_test_fake, "celeb_test_fake", "fake")
copy_with_prefix_test(sources["kaggle_test_fake"], target_test_fake, "kaggle_test_fake", "fake")
copy_with_prefix_test(sources["kaggle_test_real"], target_test_fake, "kaggle_test_real", "real")

# ✍️ CSV-Datei schreiben
with open(csv_test_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label", "source"])
    writer.writerows(csv_test_entries)

print("✅ Bilder erfolgreich zusammengeführt und dokumentiert.")