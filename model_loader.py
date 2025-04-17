import timm
import torchvision.models as models
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights

CONFIG = {
    "model_name": "efficientnet_b4",
    "num_classes": 2,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "train_dir": "data/combined_train/train",
    "val_dir": "data/combined_train/test",
    "use_pretrained": True,
    "checkpoint_dir": "checkpoints",
    "log_file": "training_results.csv",
    "log_dir": "logs",
    "early_stopping_patience": 5
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