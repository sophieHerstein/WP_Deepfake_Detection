import timm
import torchvision.models as models
import torch.nn as nn
import os

CONFIG = {
    "model_name": "efficientnet_b1",  # Optionen: xception41, mobilenet_v2, vit_base_patch16_224, swin_tiny_patch4_window7_224
    "num_classes": 2,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "train_dir": "data/celeb_df/train",
    "val_dir": "data/celeb_df/test",
    "use_pretrained": True,
    "checkpoint_dir": "checkpoints",
    "log_file": "training_results.csv",
    "early_stopping_patience": 3
}

def get_model(name: str, num_classes: int, pretrained=True):
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name in timm.list_models():
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Modell '{name}' ist nicht verfügbar.")

    return model