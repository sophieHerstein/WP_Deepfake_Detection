import os
import cv2

# Parameter
frames_per_video = 1
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
    },
    "dfd": {
        "real": "../data_raw/dfd/real_videos",
        "fake": "../data_raw/dfd/fake_videos"
    }
}

output_paths = {
    "celeb_df": {
        "real": "../data/images/celeb_df/real",
        "fake": "../data/images/celeb_df/fake",
    },
    "faceforensics": {
        "real": "../data/images/faceforensics/real",
        "fake": "../data/images/faceforensics/fake"
    },
    "dfd": {
        "real": "../data/images/dfd/real",
        "fake": "../data/images/dfd/fake"
    }
}

# Funktion zur Frame-Extraktion
def extract_frames(video_path, output_dir, video_id, frames_per_video):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frames_per_video)

    count = 0
    saved = 0
    while cap.isOpened() and saved < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            filename = f"{video_id}_frame{saved+1}{image_format}"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        count += 1
    cap.release()

# Hauptlogik zur Extraktion
for dataset, classes in source_paths.items():
    for label, folder in classes.items():
        if not os.path.exists(folder):
            print(f"[WARNUNG] Ordner nicht gefunden: {folder}")
            continue

        video_files = [f for f in os.listdir(folder) if f.endswith(video_format)]
        video_files.sort()

        out_dir = output_paths[dataset][label]
        os.makedirs(out_dir, exist_ok=True)
        for file in video_files:
            video_id = os.path.splitext(file)[0]
            video_path = os.path.join(folder, file)
            print(f"[{dataset.upper()}] Extrahiere Frames aus: {video_path}")
            extract_frames(video_path, out_dir, video_id, frames_per_video)

print("✅ Frame-Extraktion abgeschlossen.")