import os
import shutil
import csv

# 🔧 HIER deine Quellpfade eintragen:
sources = {
    "celeb_test_real": "../data/celeb_df/test/real",
    "celeb_test_fake": "../data/celeb_df/test/fake",
    "custom_test_real": "../data/custom_test/real",
    "custom_test_fake": "../data/custom_test/fake",
    "faceforensics_test_real": "../data/faceforensics/real",
    "faceforensics_test_fake": "../data/faceforensics/fake",
}

# 📁 Zielordner – hier wird alles zusammengeführt
target_test_root = "../data/combined_test"
target_test_real = os.path.join(target_test_root, "real")
target_test_fake = os.path.join(target_test_root, "fake")
os.makedirs(target_test_real, exist_ok=True)
os.makedirs(target_test_fake, exist_ok=True)

# 📄 CSV-Datei zur Dokumentation der Quelle
csv_test_path = os.path.join(target_test_root, "image_test_sources.csv")
csv_test_entries = []

# 📦 Funktion zum Kopieren + Umbenennen + Loggen
def copy_with_prefix(src_folder, dst_folder, prefix, label):
    if not os.path.exists(src_folder):
        print(f"⚠️ Test Ordner nicht gefunden: {src_folder}")
        return

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, file in enumerate(files):
        new_name = f"{prefix}_{i+1:04}.jpg"
        shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, new_name))
        csv_test_entries.append([new_name, label, prefix])

# 🏁 Kopiervorgang starten
copy_with_prefix(sources["celeb_test_real"], target_test_real, "celeb_test_real", "real")
copy_with_prefix(sources["celeb_test_fake"], target_test_fake, "celeb_test_fake", "fake")
copy_with_prefix(sources["custom_test_real"], target_test_real, "custom_test_real", "real")
copy_with_prefix(sources["custom_test_fake"], target_test_fake, "custom_test_fake", "fake")
copy_with_prefix(sources["faceforensics_test_real"], target_test_fake, "faceforensics_test_real", "fake")
copy_with_prefix(sources["faceforensics_test_fake"], target_test_real, "faceforensics_test_fake", "real")

# ✍️ CSV-Datei schreiben
with open(csv_test_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label", "source"])
    writer.writerows(csv_test_entries)

print("✅ Bilder erfolgreich zusammengeführt und dokumentiert.")