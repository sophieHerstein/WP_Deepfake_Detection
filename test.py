import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import time
import csv
import logging
import pandas as pd
from model_loader import CONFIG, get_model
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from util.generate_html_report import get_report
from util.plot_test_results import plot_test_results


# Logger

def setup_logger(model_name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"test_{model_name}")
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(log_dir, f"{model_name}_test.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # Konsole
    ch.setLevel(logging.INFO)  # Optional: nur INFO und höher im Terminal
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Modellgröße in MB

def get_model_size(path):
    size = os.path.getsize(path) / (1024 ** 2)  # MB
    return round(size, 2)

# Anzahl Parameter

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Optionales reshape_transform für ViT / Swin

def reshape_transform_vit(tensor, height=14, width=14):
    if tensor.dim() != 3:
        raise ValueError(f"reshape_transform_vit erwartet 3D Tensor, aber bekam: {tensor.shape}")
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


# Grad-CAM vorbereiten und anwenden
def generate_gradcam(config, model, device, transform, target_layer, dataset, max_images=5, reshape_transform=None):
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    output_dir = os.path.join("gradcam", config["model_name"])
    os.makedirs(output_dir, exist_ok=True)

    counter = 0
    for path, label in dataset.samples:
        if counter >= max_images:
            break
        img = Image.open(path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        input_rgb = np.array(img.resize((config["image_size"], config["image_size"]))) / 255.0
        input_rgb = np.float32(input_rgb)

        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(input_rgb, grayscale_cam, use_rgb=True)

        out_path = os.path.join(output_dir, f"gradcam_{counter + 1}.jpg")
        plt.imsave(out_path, visualization)
        counter += 1

# Testfunktion
@torch.no_grad()
def evaluate_model(config):
    mode = "_robust" if config.get("robust_test", False) else ""
    logger = setup_logger(config["model_name"] + mode, config["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starte Evaluation auf {device} {'(JPEG-komprimierte Bilder)' if mode else ''}")

    test_dir = config["test_dir"]
    if config.get("robust_test", False):
        test_dir = config.get("test_dir_robust", test_dir + "_jpeg")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = get_model(config["model_name"], config["num_classes"], pretrained=False)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{config['model_name']}_finetuned.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    total_time = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        outputs = model(inputs)
        total_time += time.time() - start
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    avg_time = total_time / len(test_dataset)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    model_size = get_model_size(checkpoint_path)
    num_params = get_num_parameters(model)

    logger.info(
        f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Zeit/Bild: {avg_time:.4f}s")
    logger.info(f"Modellgröße: {model_size} MB, Parameter: {num_params:,}")
    logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    os.makedirs(config["result_dir"], exist_ok=True)
    result_path = os.path.join(config["result_dir"], "test_results.csv")
    log_exists = os.path.isfile(result_path)
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["Modell", "Variante", "Accuracy", "Precision", "Recall", "F1-Score", "Zeit/Bild (s)",
                             "Modellgröße (MB)", "Parameter", "TP", "TN", "FP", "FN"])
        writer.writerow([
            config["model_name"], "robust" if mode else "standard", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}",
            f"{f1:.4f}", f"{avg_time:.4f}", f"{model_size}", f"{num_params}", tp, tn, fp, fn
        ])

    if not config.get("robust_test", False):
        reshape = None

        if "vit" in config["model_name"] or "swin" in config["model_name"]:
            try:
                from timm.models.vision_transformer import VisionTransformer
                if isinstance(model, VisionTransformer):
                    target_layer = model.norm  # besser als .blocks[-1].norm1
                else:
                    target_layer = model.layers[-1].blocks[-1].norm1  # für Swin
            except Exception:
                target_layer = list(model.children())[-1]
            reshape = reshape_transform_vit

            with torch.enable_grad():
                generate_gradcam(
                    config, model, device, transform, target_layer, test_dataset,
                    max_images=config.get("gradcam_images", 5),
                    reshape_transform=reshape
                )

        try:
            if os.path.exists(result_path):
                get_report()
        except Exception as e:
            logger.warning(f"HTML-Report konnte nicht erstellt werden: {e}")

    plot_test_results()

# Hauptausführung
if __name__ == "__main__":
    CONFIG["test_dir"] = "data/combined_test"
    CONFIG["test_dir_robust"] = "data/combined_test_jpeg50"
    CONFIG["result_dir"] = "results"
    CONFIG["gradcam_images"] = 5

    model_names = [
        "efficientnet_b4",
        "xception41",
        "mobilenet_v2",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224"
    ]

    for model_name in model_names:
        CONFIG["model_name"] = model_name

        # Standard-Test
        CONFIG["robust_test"] = False
        evaluate_model(CONFIG)

        # Robustheitstest (JPEG-Kompression)
        CONFIG["robust_test"] = True
        evaluate_model(CONFIG)
