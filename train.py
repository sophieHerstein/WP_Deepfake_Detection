import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from model_loader import CONFIG, get_model
import os
import time
import csv
import itertools


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(config["train_dir"], transform=transform)
    val_dataset = datasets.ImageFolder(config["val_dir"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = get_model(config["model_name"], config["num_classes"], config["use_pretrained"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_acc = 0.0
    no_improve_epochs = 0
    start_time = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # Evaluation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100

        # Zeit & Fortschritt
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (config["epochs"] - epoch - 1)
        print(f"[{epoch+1}/{config['epochs']}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | ETA: {int(eta)}s")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config["early_stopping_patience"]:
                print(f"⏹️ Early stopping ausgelöst nach {epoch+1} Epochen.")
                break

    # Modell speichern
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{config['model_name']}_finetuned.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Modell gespeichert unter: {checkpoint_path}")

    # Ergebnis loggen
    os.makedirs(os.path.dirname(config["log_file"]), exist_ok=True)
    log_exists = os.path.isfile(config["log_file"])
    with open(config["log_file"], "a", newline="") as logfile:
        writer = csv.writer(logfile)
        if not log_exists:
            writer.writerow(["Modell", "Train-Acc", "Val-Acc", "Loss", "Trainzeit (s)"])
        writer.writerow([
            config["model_name"],
            f"{train_acc:.2f}",
            f"{val_acc:.2f}",
            f"{running_loss:.4f}",
            f"{int(time.time() - start_time)}"
        ])

    return val_acc  # zur Bewertung beim Grid Test

def parameter_grid_search(config, param_grid, test_model="mobilenet_v2"):
    print("\n🔍 Starte Parameter-Test mit Grid Search")
    best_acc = 0.0
    best_config = {}

    for lr, bs in itertools.product(param_grid["learning_rate"], param_grid["batch_size"]):
        config["model_name"] = test_model
        config["learning_rate"] = lr
        config["batch_size"] = bs
        config["epochs"] = 3  # Kurztest

        print(f"\n🧪 Test: LR={lr}, Batch={bs}")
        acc = train_model(config)
        if acc > best_acc:
            best_acc = acc
            best_config = {"learning_rate": lr, "batch_size": bs}

    print(f"\n🏁 Beste Parameterkombination: LR={best_config['learning_rate']} | Batch={best_config['batch_size']} | Acc={best_acc:.2f}%")
    return best_config

# Parameter definieren
param_grid = {
    "learning_rate": [1e-4, 5e-5],
    "batch_size": [16, 32]
}

# Grid Search zuerst durchführen
optimal_params = parameter_grid_search(CONFIG, param_grid)
CONFIG["learning_rate"] = optimal_params["learning_rate"]
CONFIG["batch_size"] = optimal_params["batch_size"]
CONFIG["epochs"] = 10

# Jetzt alle Modelle mit optimalen Parametern trainieren
model_names = [
    "efficientnet_b1",
    "xception41",
    "mobilenet_v2",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224"
]

for model_name in model_names:
    print(f"\n🔁 Starte Training für Modell: {model_name}")
    CONFIG["model_name"] = model_name
    train_model(CONFIG)
