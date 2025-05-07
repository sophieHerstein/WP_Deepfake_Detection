import os
from PIL import Image
import numpy as np
from tqdm import tqdm

original_root_celebdf_only = "../data/celebdf_only/test"
output_root_jpeg_celebdf_only = "../data/celebdf_only/test_jpeg"
output_root_noisy_celebdf_only = "../data/celebdf_only/test_noisy"
output_root_scaled_celebdf_only = "../data/celebdf_only/test_scaled"
original_root_celebdf_ffpp = "../data/celebdf_ffpp/test"
output_root_jpeg_celebdf_ffpp = "../data/celebdf_ffpp/test_jpeg"
output_root_noisy_celebdf_ffpp = "../data/celebdf_ffpp/test_noisy"
output_root_scaled_celebdf_ffpp = "../data/celebdf_ffpp/test_scaled"
jpeg_quality = 50
gaussian_noise_stddev = 25
scaling_factor = 0.5

def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    print("\nApplying JPEG compression...")
    with open("temp.jpg", "wb") as f:
        image.save(f, "JPEG", quality=quality)
    return Image.open("temp.jpg")

def apply_gaussian_noise(image: Image.Image, stddev: float) -> Image.Image:
    print("\nApplying Gaussian noise...")
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, stddev, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_scaling(image: Image.Image, factor: float) -> Image.Image:
    print("\nApplying scaling...")
    original_size = image.size
    scaled_size = (int(original_size[0] * factor), int(original_size[1] * factor))
    scaled_down = image.resize(scaled_size, Image.BICUBIC)
    scaled_up = scaled_down.resize(original_size, Image.BICUBIC)
    return scaled_up

def process_images(original_root, output_root_jpeg, output_root_noisy, output_root_scaled):
    for root, _, files in os.walk(original_root):
        for file in tqdm(files, desc="Verarbeite Bilder"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            print("\nProcessing file:", file)
            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(in_path, original_root)
            out_dir_jpeg = os.path.join(output_root_jpeg, os.path.dirname(rel_path))
            out_dir_noisy = os.path.join(output_root_noisy, os.path.dirname(rel_path))
            out_dir_scaled = os.path.join(output_root_scaled, os.path.dirname(rel_path))
            os.makedirs(out_dir_jpeg, exist_ok=True)
            os.makedirs(out_dir_noisy, exist_ok=True)
            os.makedirs(out_dir_scaled, exist_ok=True)

            try:
                image = Image.open(in_path).convert("RGB")

                base_name = os.path.splitext(file)[0]

                # 1. JPEG-Kompression
                jpeg_image = apply_jpeg_compression(image, jpeg_quality)
                jpeg_image.save(os.path.join(out_dir_jpeg, f"{base_name}_jpeg.jpg"))

                # 2. Rauschen
                noisy_image = apply_gaussian_noise(image, gaussian_noise_stddev)
                noisy_image.save(os.path.join(out_dir_noisy, f"{base_name}_noisy.jpg"))

                # 3. Skalierung
                scaled_image = apply_scaling(image, scaling_factor)
                scaled_image.save(os.path.join(out_dir_scaled, f"{base_name}_scaled.jpg"))

            except Exception as e:
                print(f"Fehler bei Datei {in_path}: {e}")

if __name__ == "__main__":
    process_images(original_root_celebdf_only, output_root_jpeg_celebdf_only, output_root_noisy_celebdf_only, output_root_scaled_celebdf_only)
    process_images(original_root_celebdf_ffpp, output_root_jpeg_celebdf_ffpp, output_root_noisy_celebdf_ffpp, output_root_scaled_celebdf_ffpp)