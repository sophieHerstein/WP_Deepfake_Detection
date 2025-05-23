import os
import random

def balance_classes(folder_path, seed=42):
    random.seed(seed)

    class_dirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, d))]

    # Erfasse alle Dateien in jedem Klassenordner
    class_files = {d: [os.path.join(d, f) for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f))] for d in class_dirs}

    # Bestimme kleinste Klassenanzahl
    min_count = min(len(files) for files in class_files.values())

    for cls, files in class_files.items():
        count_to_delete = len(files) - min_count
        if count_to_delete > 0:
            to_delete = random.sample(files, count_to_delete)
            print(f"Lösche {count_to_delete} Bilder aus: {cls}")
            for f in to_delete:
                os.remove(f)

    print("✅ Klassen ausgeglichen.")

# Beispielaufruf
balance_classes("../data/celebdf_ff/test")