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
    "test_dir_robust": "data/combined_test_augmented",

    # Speicherorte
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "result_csv": "results/test_results.csv",
    "train_result": "results/train_results.csv",
    "resources_csv": "results/model_resources.csv",
    "gradcam": "gradcam",
    "results": "results",
    "plots": "plots",
    "reports": "reports",

    # Grad-CAM
    "gradcam_images": 5,
    "variant": "standard"
}

MODEL_NAMES = [
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
    "efficientnet_b4",
    "xception41",
    "mobilenet_v2"
]

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