import os
import cv2
import random
import numpy as np

# Parameter
frames_per_video = 1
max_videos_per_class = None  # z. B. 713 für ausgewogene Subsets
video_format = ".mp4"
image_format = ".jpg"

# Quelle und Zielverzeichnisse
source_paths = {
    "celeb_df": {
        "real": "../data_raw/celeb_df/real_videos",
        "fake": "../data_raw/celeb_df/fake_videos"
    },
    "faceforensics": {
        "real": "../data_raw/faceforensics/real_videos",
        "fake": "../data_raw/faceforensics/fake_videos"
    }
}

output_paths = {
    "celeb_df": {
        "train_real": "../data/celeb_df/train/real",
        "train_fake": "../data/celeb_df/train/fake",
        "test_real": "../data/celeb_df/test/real",
        "test_fake": "../data/celeb_df/test/fake"
    },
    "faceforensics": {
        "real": "../data/faceforensics/real",
        "fake": "../data/faceforensics/fake"
    }
}

# ✅ Realbilder im Training augmentieren
def apply_real_augmentation(image):
    """Wendet gezielte, moderate Augmentationen für Realbilder an."""
    # 1. JPEG-Kompression (Qualität 80–90)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(80, 90)]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(encimg, 1)

    # 2. Skalierung (±10–20 %)
    scale_factor = random.uniform(0.8, 1.2)
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    image = cv2.resize(image, (w, h))  # Zurück auf Originalgröße

    # 3. Leichtes Rauschen
    noise_std = random.uniform(2, 5)
    noise = np.random.normal(0, noise_std, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    return image

# 🎯 Frame extrahieren und ggf. augmentieren
def extract_random_frames(video_path, output_dir, video_id, label=None, num_frames=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[WARNUNG] Leeres oder fehlerhaftes Video: {video_path}")
        return

    frame_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if label == "real_train":
                frame = apply_real_augmentation(frame)
            filename = f"{video_id}_frame{i+1}{image_format}"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
    cap.release()

# Hauptlogik zur Extraktion
for dataset, classes in source_paths.items():
    for label, folder in classes.items():
        if not os.path.exists(folder):
            print(f"[WARNUNG] Ordner nicht gefunden: {folder}")
            continue

        video_files = [f for f in os.listdir(folder) if f.endswith(video_format)]
        video_files.sort()  # Reproduzierbarkeit

        # Für CelebDF mit Train/Test-Split
        if dataset == "celeb_df":
            split_idx = int(0.8 * len(video_files))
            train_files = video_files[:split_idx]
            test_files = video_files[split_idx:]

            if max_videos_per_class:
                train_files = train_files[:max_videos_per_class]

            for split_name, file_list in zip(["train", "test"], [train_files, test_files]):
                out_dir = output_paths[dataset][f"{split_name}_{label}"]
                os.makedirs(out_dir, exist_ok=True)

                for file in file_list:
                    video_id = os.path.splitext(file)[0]
                    video_path = os.path.join(folder, file)
                    combined_label = f"{label}_{split_name}"  # z. B. "real_train"
                    print(f"[{dataset.upper()}] Extrahiere Frame aus: {video_path}")
                    extract_random_frames(video_path, out_dir, video_id, label=combined_label, num_frames=frames_per_video)

        # Für FaceForensics ohne Split
        else:
            out_dir = output_paths[dataset][label]
            os.makedirs(out_dir, exist_ok=True)
            for file in video_files:
                video_id = os.path.splitext(file)[0]
                video_path = os.path.join(folder, file)
                print(f"[{dataset.upper()}] Extrahiere Frame aus: {video_path}")
                extract_random_frames(video_path, out_dir, video_id, label=None, num_frames=frames_per_video)

print("✅ Frame-Extraktion mit Augmentation abgeschlossen.")