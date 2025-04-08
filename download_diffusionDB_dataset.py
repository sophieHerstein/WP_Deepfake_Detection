from datasets import load_dataset
from PIL import Image
import base64
import io
import os

# Dataset laden
dataset = load_dataset("poloclub/diffusiondb", split="train[:50]")

# Zielordner
output_dir = "data/diffusionDB_test"
os.makedirs(output_dir, exist_ok=True)

# Bilder extrahieren und speichern
for i, item in enumerate(dataset):
    image = item["image"]  # das ist bereits ein PIL-Bild
    image.save(os.path.join(output_dir, f"diffusiondb_{i+1:04}.jpg"))

print("✅ Bilder erfolgreich gespeichert.")