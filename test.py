import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import csv
import logging
import time
from model_loader import get_model, MODEL_NAMES, CONFIG
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from util.plot_test_results import plot_single_run, plot_model_overview, plot_all_models


# 📁 Logger Setup
def setup_logger(name, log_dir, variante):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_dir, f"{name}_{variante}_test.log")
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_model_size(path):
    return round(os.path.getsize(path) / (1024 ** 2), 2)  # in MB

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 📊 Evaluation pro Modell
def evaluate_model(model_name, config, variante):
    logger = setup_logger(model_name, config["log_dir"], variante)
    logger.info(f"Starte Evaluation für Modell: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Gerät: {device}")

    # 📦 Daten
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.ImageFolder(config["test_dir"], transform=transform)
    if len(dataset) == 0:
        logger.warning(f"Keine Bilder gefunden in: {config['test_dir']}")
        return
    logger.info(f"Testbilder: {len(dataset)} Bilder aus {config['test_dir']}")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    if not os.path.exists(config["checkpoint_dir"]):
        logger.warning(f"Checkpoint nicht gefunden: {config['checkpoint_dir']}")
        return

    # 🧠 Modell laden
    model = get_model(model_name, config["num_classes"], pretrained=False)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{model_name}_{variante}_finetuned.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if config["variant"] == "standard":
        model_size = get_model_size(checkpoint_path)
        num_params = get_num_parameters(model)
        logger.info(f"Modellgröße: {model_size} MB")
        logger.info(f"Anzahl der Parameter: {num_params}")

        os.makedirs(os.path.dirname(config["resources_csv"]), exist_ok=True)
        write_header = not os.path.exists(config["resources_csv"])
        with open(config["resources_csv"], "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Modell", "Variante", "Size (MB)", "Params"])
            writer.writerow([model_name, variante, f"{model_size:.4f}", f"{num_params:.4f}"])
    model.to(device)
    model.eval()

    # 🧮 Vorhersagen
    cam_dir = os.path.join(config["gradcam"], variante, model_name, config["variant"])
    os.makedirs(cam_dir, exist_ok=True)

    class_to_idx = loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    collected = {f"{outcome}_{cls}": [] for outcome in ["TP", "TN", "FP", "FN"] for cls in ["fake", "real"]}
    y_true, y_pred, all_paths = [], [], []
    total_time = 0
    num_images = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            total_time += time.time() - start_time
            num_images += inputs.size(0)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_paths = [loader.dataset.samples[i][0] for i in range(len(all_paths), len(all_paths) + len(labels))]
            all_paths.extend(batch_paths)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2  # in MB
        logger.info(f"GPU Speicher belegt: {allocated:.2f} MB, reserviert: {reserved:.2f} MB")

    # 📈 Metriken
    avg_time_per_image = total_time / num_images
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"Durchschnittliche Vorhersagezeit pro Bild: {avg_time_per_image:.4f} Sekunden")

    # 📄 In CSV schreiben
    os.makedirs(os.path.dirname(config["result_csv"]), exist_ok=True)
    write_header = not os.path.exists(config["result_csv"])
    with open(config["result_csv"], "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Modell", "Variante-Training", "Variante-Test", "Accuracy", "Precision", "Recall", "F1-Score", "TP", "TN", "FP", "FN", "Avg-Time/Bild (s)"])
        writer.writerow([model_name, variante, config["variant"] ,f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", tp, tn, fp, fn, f"{avg_time_per_image:.4f}"])

    reshape = None
    if "vit" in model_name:
        target_layer = model.patch_embed.proj  # ✅ richtig für ViT
    elif "swin" in model_name:
        target_layer = model.norm
    elif hasattr(model, 'features'):
        target_layer = model.features[-1]
    elif "efficientnet" in model_name:
        target_layer = model.conv_head
    elif "mobilenet" in model_name:
        target_layer = model.features[-1]
    elif "xception" in model_name or (hasattr(model, "blocks") and isinstance(model.blocks, torch.nn.Sequential)):
        target_layer = model.blocks[-1]
    else:
        # Fallback: letzter Conv-Layer, falls vorhanden
        target_layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]

    logger.info(f"Target-Layer für Grad-CAM: {target_layer}")

    if "vit" in model_name:
        cam = ScoreCAM(model=model, target_layers=[target_layer])
        logger.info("ScoreCAM wird für ViT verwendet.")
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])

    transform_rgb = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"]))
    ])

    for path, pred, label in zip(all_paths, y_true, y_pred):
        if label == 1 and pred == 1:
            outcome = "TP"
        elif label == 0 and pred == 0:
            outcome = "TN"
        elif label == 0 and pred == 1:
            outcome = "FP"
        elif label == 1 and pred == 0:
            outcome = "FN"
        else:
            continue

        true_class = idx_to_class[label]
        key = f"{outcome}_{true_class}"
        if len(collected[key]) >= 2:
            continue

        img = Image.open(path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        input_rgb = np.array(transform_rgb(img)) / 255.0
        input_rgb = np.float32(input_rgb)

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(input_rgb, grayscale_cam, use_rgb=True)

        out_path = os.path.join(cam_dir, f"{key}_{len(collected[key]) + 1}.jpg")
        plt.imsave(out_path, visualization)
        collected[key].append(out_path)

        logger.info(f"Grad-CAM gespeichert: {out_path}")
    logger.info(f"Grad-CAM: {sum(len(v) for v in collected.values())} Visualisierungen gespeichert.")

    plot_single_run(model_name, config["variant"], variante)


# ▶️ Hauptausführung
if __name__ == "__main__":
    for name in MODEL_NAMES:

        for variant, folder in [
            ("standard", "combined_test"),
            ("jpeg", "combined_test_jpeg"),
            ("noisy", "combined_test_noisy"),
            ("scaled", "combined_test_scaled")
        ]:
            CONFIG["test_dir"] = f"data/{folder}"
            CONFIG["result_csv"] = f"results/{name}_{variant}_{...}_results.csv" #TODO
            CONFIG["variant"] = variant
            evaluate_model(name, CONFIG)

        plot_model_overview(name, variante)

    plot_all_models(variante)
