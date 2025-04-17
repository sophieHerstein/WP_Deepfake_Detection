import timm
import torchvision.models as models
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights

CONFIG = {
    # Modellparameter
    "model_name": "mobilenet_v2",  # Platzhalter – wird in den Loops überschrieben
    "num_classes": 2,
    "image_size": 224,

    # Trainingsparameter
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "early_stopping_patience": 4,
    "use_pretrained": True,

    # Datenpfade
    "train_dir": "data/celeb_df/train",
    "val_dir": "data/celeb_df/test",
    "test_dir": "data/combined_test",
    "test_dir_robust": "data/combined_test_jpeg50",

    # Speicherorte
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "log_file": "results/training_results.csv",
    "result_dir": "results",

    # Grad-CAM
    "gradcam_images": 5,
    "robust_test": False  # wird zur Laufzeit angepasst
}

def get_model(name: str, num_classes: int, pretrained=True):
    if name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name in timm.list_models():
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Modell '{name}' ist nicht verfügbar.")

    return model