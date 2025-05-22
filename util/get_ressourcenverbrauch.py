import os
import pandas as pd
import re

# === Konfiguration ===
base_dir = "../results/test"
resource_file = "../results/model_resources.csv"

# === Mapping Testvariante -> Trainvariante ===
train_map = {
    "celebdf_only": "celebdf_only",
    "celebdf_ff": "celebdf_ff",
    "celebdf_train_ff_test": "celebdf_only"
}

# === Schritt 1: Ressourcen-Csv einlesen ===
resources_df = pd.read_csv(resource_file)
resources_df.columns = ["Model", "TrainVariante", "Size_MB", "Params"]

# === Schritt 2: CSVs mit Inferenzzeit einlesen ===
def extract_inference_times(base_dir):
    data = []
    for test_var in os.listdir(base_dir):
        test_path = os.path.join(base_dir, test_var)
        if not os.path.isdir(test_path):
            continue
        for model in os.listdir(test_path):
            model_path = os.path.join(test_path, model)
            if not os.path.isdir(model_path):
                continue
            for aug in os.listdir(model_path):
                aug_path = os.path.join(model_path, aug)
                for file in os.listdir(aug_path):
                    if file.endswith(".csv") and "results" in file:
                        csv_path = os.path.join(aug_path, file)
                        df = pd.read_csv(csv_path)
                        if "Avg-Time/Bild (s)" in df.columns:
                            try:
                                value = float(df["Avg-Time/Bild (s)"].iloc[0])
                                data.append({
                                    "Model": model,
                                    "TestVariante": test_var,
                                    "TrainVariante": train_map[test_var],
                                    "Variation": aug,
                                    "InferenceTime": value
                                })
                            except Exception:
                                pass
    return pd.DataFrame(data)

# === Schritt 3: GPU-Werte aus .log-Dateien extrahieren ===
def extract_gpu_memory(log_root="../logs/test"):
    pattern_path = re.compile(r"Testbilder: .* aus .*/test(?:_)?(\w*)")
    pattern_gpu = re.compile(r"GPU Speicher belegt: ([\d.]+) MB, reserviert: ([\d.]+) MB")

    data = []
    for root, _, files in os.walk(log_root):
        for file in files:
            if file.endswith(".log"):
                model = file.replace(".log", "")
                train_var = os.path.basename(root)
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                current_variation = None
                for line in lines:
                    # Variation aus Pfad extrahieren
                    path_match = pattern_path.search(line)
                    if path_match:
                        current_variation = path_match.group(1)
                        if current_variation == "":
                            current_variation = "standard"

                    # GPU-Werte extrahieren
                    gpu_match = pattern_gpu.search(line)
                    if gpu_match and current_variation:
                        belegt, reserviert = map(float, gpu_match.groups())
                        data.append({
                            "Model": model,
                            "TrainVariante": train_var,
                            "Variation": current_variation,
                            "GPU_Belegt_MB": belegt,
                            "GPU_Reserviert_MB": reserviert
                        })
                        current_variation = None  # zurücksetzen, um fehlerhafte Zuordnungen zu vermeiden
    return pd.DataFrame(data)

# === Schritt 4: Zusammenführen ===
inference_df = extract_inference_times(base_dir)
gpu_df = extract_gpu_memory()

print("GPU-DataFrame Vorschau:")
print(gpu_df.head())
print("Inference-DataFrame Vorschau:")
print(inference_df.head())

merged = inference_df.merge(gpu_df, on=["Model", "TrainVariante", "Variation"], how="left")
final = merged.merge(resources_df, on=["Model", "TrainVariante"], how="left")

# === Schritt 5: Speichern ===
final.to_csv("../results/combined_model_resources.csv", index=False)
print("✅ Zusammenfassung gespeichert unter: results/combined_model_resources.csv")