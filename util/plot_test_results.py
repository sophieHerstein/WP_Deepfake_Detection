import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from model_loader import CONFIG

RESULTS_DIR = CONFIG["results"]
OUTPUT_DIR = CONFIG["plots"]
RESOURCE_FILE = CONFIG["resources_csv"]
VARIANTS = ["standard", "jpeg", "noisy", "scaled"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_single_run(model_name: str, variant: str):
    path = os.path.join(RESULTS_DIR, f"{model_name}_{variant}_results.csv")
    if not os.path.exists(path):
        print(f"❌ Datei nicht gefunden: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"⚠️ Leere Datei: {path}")
        return

    # Konfusionsmatrix
    cm = [df["TN"].iloc[0], df["FP"].iloc[0], df["FN"].iloc[0], df["TP"].iloc[0]]
    matrix = [[cm[0], cm[1]], [cm[2], cm[3]]]
    labels = ["Real", "Fake"]
    plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name} ({variant})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{model_name}_{variant}.png"))
    plt.close()

    # Bar-Plot der Metriken
    metrics = [df["Accuracy"].iloc[0], df["Precision"].iloc[0], df["Recall"].iloc[0], df["F1-Score"].iloc[0]]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, metrics, color=colors)
    plt.ylim(0, 1)
    plt.title(f'{model_name} – {variant} – Metriken')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{variant}_metrics_bar.png"))
    plt.close()

    # Vergleich mit Standard-Variante
    if variant != "standard":
        standard_path = os.path.join(RESULTS_DIR, f"{model_name}_standard_results.csv")
        variant_path = os.path.join(RESULTS_DIR, f"{model_name}_{variant}_results.csv")

        if not os.path.exists(standard_path) or not os.path.exists(variant_path):
            print(f"[WARN] CSV-Dateien für {model_name} nicht vollständig.")
            return

        df_std = pd.read_csv(standard_path)
        df_var = pd.read_csv(variant_path)

        if df_std.empty or df_var.empty:
            print(f"[WARN] Leere CSVs bei {model_name}.")
            return

        # Nur erste Zeile nehmen, falls mehrere Runs existieren
        std_metrics = df_std.iloc[0][["Accuracy", "Precision", "Recall", "F1-Score"]].astype(float)
        var_metrics = df_var.iloc[0][["Accuracy", "Precision", "Recall", "F1-Score"]].astype(float)

        delta = var_metrics - std_metrics

        plt.figure(figsize=(8, 6))
        colors = ['#4CAF50' if d >= 0 else '#F44336' for d in delta]
        plt.bar(delta.index, delta.values, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"{model_name} – Δ zur Standard-Variante ({variant})")
        for i, d in enumerate(delta.values):
            plt.text(i, d + (0.005 if d >= 0 else -0.03), f'{d:+.3f}', ha='center', fontweight='bold')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{variant}_delta_vs_standard.png"))
        plt.close()


def plot_model_overview(model_name: str):
    # Lade alle vier Varianten
    data = []
    for variant in VARIANTS:
        path = os.path.join(RESULTS_DIR, f"{model_name}_{variant}_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["variant"] = variant
        data.append(df)

    if not data:
        print(f"⚠️ Keine Ergebnisse für Modell: {model_name}")
        return

    df_all = pd.concat(data)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Metrikenvergleich
    df_melted = df_all.melt(id_vars="variant", value_vars=metrics, var_name="Metric", value_name="Value")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melted, x="variant", y="Value", hue="Metric")
    plt.title(f"{model_name} – Performance across Variants")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"overview_{model_name}.png"))
    plt.close()

    # Robustheit (Δ zu Standard)
    df_pivot = df_all.set_index("variant")
    if "standard" in df_pivot.index:
        base = df_pivot.loc["standard"]
        delta = df_pivot[metrics].subtract(base[metrics], axis=1)
        delta.drop(index="standard", inplace=True)
        delta = delta.reset_index().melt(id_vars="variant", var_name="Metric", value_name="Δ")

        plt.figure(figsize=(8, 5))
        sns.barplot(data=delta, x="variant", y="Δ", hue="Metric")
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"{model_name} – Robustness compared to Standard")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"robustness_{model_name}.png"))
        plt.close()

    """Vergleicht die Accuracy-, Recall- und F1-Verluste des Modells pro Variante im Vergleich zur Standardvariante."""
    variants = ["standard", "jpeg", "noisy", "scaled"]
    metrics = ["Accuracy", "Recall", "F1-Score"]
    results = {var: {} for var in variants}

    # CSVs einlesen
    for var in variants:
        path = os.path.join(RESULTS_DIR, f"{model_name}_{var}_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for metric in metrics:
            results[var][metric] = df[metric].values[0]

    if "standard" not in results or not results["standard"]:
        print(f"⚠️ Kein Standard-Ergebnis für {model_name} gefunden.")
        return

    # Δ-Werte berechnen
    deltas = {
        metric: {
            var: results["standard"][metric] - results[var].get(metric, 0)
            for var in variants if var != "standard"
        }
        for metric in metrics
    }

    # Plot
    plt.figure(figsize=(8, 5))
    width = 0.25
    x = range(len(variants) - 1)
    offsets = [-width, 0, width]
    colors = ["tomato", "steelblue", "mediumseagreen"]

    for i, metric in enumerate(metrics):
        bars = plt.bar(
            [p + offsets[i] for p in x],
            list(deltas[metric].values()),
            width=width,
            label=metric,
            color=colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.003, f"{height:.3f}",
                     ha='center', va='bottom', fontsize=8)

    plt.xticks(x, list(deltas[metrics[0]].keys()))
    plt.ylabel("Δ Metrik (Standard - Variante)")
    plt.title(f"Robustheitsprofil für {model_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_robustness_profile.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Robustheitsprofil gespeichert: {path}")


def plot_all_models():
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_results.csv") and "standard" in f]
    rows = []
    for file in files:
        df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        if df.empty:
            continue
        model = file.replace("_standard_results.csv", "")
        df["model"] = model
        rows.append(df)

    if not rows:
        print("⚠️ Keine Standard-Ergebnisse für Vergleich gefunden.")
        return

    df = pd.concat(rows)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_melted = df.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Value")

    # Vergleichsplot aller Modelle
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="model", y="Value", hue="Metric")
    plt.title("Model Comparison (Standard Variant)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_all_models.png"))
    plt.close()

    # Größe vs Performance (optional)
    res_path = os.path.join(RESOURCE_FILE)
    if os.path.exists(res_path):
        res_df = pd.read_csv(res_path)
        merged = pd.merge(df, res_df, left_on="model", right_on="Modell")

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=merged, x="Size (MB)", y="Accuracy", hue="model")
        plt.title("Model Size vs Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "size_vs_accuracy.png"))
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=merged, x="Params", y="Accuracy", hue="model")
        plt.title("Parameter Count vs Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "params_vs_accuracy.png"))
        plt.close()

    # files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_results.csv")]
    # robust_data = []
    #
    # for file in files:
    #     variant = file.split("_")[-2]  # z. B. standard
    #     model = file.replace(f"_{variant}_results.csv", "")
    #     df = pd.read_csv(os.path.join(RESULTS_DIR, file))
    #     if df.empty:
    #         continue
    #     accuracy = df["Accuracy"].values[0]
    #     robust_data.append({"Model": model, "Variant": variant, "Accuracy": accuracy})
    #
    # robust_df = pd.DataFrame(robust_data)
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=robust_df, x="Variant", y="Accuracy", hue="Model", marker="o")
    # plt.title("Model Robustness (Accuracy Across Variants)")
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "model_robustness_accuracy.png"))
    # plt.close()
    #
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=robust_df, x="Model", y="Accuracy")
    # plt.title("Accuracy Distribution per Model Across Variants")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_distribution.png"))
    # plt.close()
    #
    # pivot = robust_df.pivot(index="Model", columns="Variant", values="Accuracy")
    # delta = pivot.sub(pivot["standard"], axis=0)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(delta, annot=True, cmap="coolwarm", center=0)
    # plt.title("Δ Accuracy (Variant vs. Standard)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_deltas_heatmap.png"))
    # plt.close()
