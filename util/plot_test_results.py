import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from model_loader import CONFIG

RESULTS_DIR = CONFIG["results"]
OUTPUT_DIR = CONFIG["plots"]
RESOURCE_FILE = CONFIG["resources_csv"]
VARIANTS = ["standard", "jpeg", "noisy", "scaled"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_single_run(model_name: str, variant: str, train_variante: str):
    path = os.path.join(RESULTS_DIR, "test", train_variante, model_name, variant, f"{model_name}_{variant}_{train_variante}_results.csv")
    if not os.path.exists(path):
        print(f"❌ Datei nicht gefunden: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"⚠️ Leere Datei: {path}")
        return

    out = os.path.join(OUTPUT_DIR, train_variante)
    os.makedirs(out, exist_ok=True)

    # Konfusionsmatrix
    cm = [df["TN"].iloc[0], df["FP"].iloc[0], df["FN"].iloc[0], df["TP"].iloc[0]]
    matrix = [[cm[0], cm[1]], [cm[2], cm[3]]]
    labels = ["Real", "Fake"]
    plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name} ({variant}, {train_variante})")
    plt.tight_layout()
    os.makedirs(os.path.join(out, 'cm', model_name), exist_ok=True)
    plt.savefig(os.path.join(out, 'cm', model_name, f"cm_{model_name}_{train_variante}_{variant}.png"))
    plt.close()

    # Bar-Plot der Metriken
    metrics = [df["Accuracy"].iloc[0], df["Precision"].iloc[0], df["Recall"].iloc[0], df["F1-Score"].iloc[0]]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, metrics, color=colors)
    plt.ylim(0, 1)
    plt.title(f'{model_name} - {train_variante} – {variant} – Metriken')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    os.makedirs(os.path.join(out, 'metrics_bar', model_name), exist_ok=True)
    plt.savefig(os.path.join(out, 'metrics_bar', model_name, f"metrics_bar_{model_name}_{train_variante}_{variant}.png"))
    plt.close()

    # Vergleich mit Standard-Variante
    if variant != "standard":
        standard_path = os.path.join(RESULTS_DIR, "test", train_variante, model_name, "standard", f"{model_name}_standard_{train_variante}_results.csv")
        variant_path = os.path.join(RESULTS_DIR, "test", train_variante, model_name, variant, f"{model_name}_{variant}_{train_variante}_results.csv")

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
        plt.title(f"{model_name} – Δ zur Standard-Variante ({variant}, {train_variante})")
        for i, d in enumerate(delta.values):
            plt.text(i, d + (0.005 if d >= 0 else -0.03), f'{d:+.3f}', ha='center', fontweight='bold')
        os.makedirs(os.path.join(out, 'delta_vs_standard', model_name), exist_ok=True)
        plt.savefig(os.path.join(out, "delta_vs_standard", model_name, f"delta_vs_standard_{model_name}_{train_variante}_{variant}.png"))
        plt.close()


def plot_model_overview(model_name: str, train_variante: str):
    # Lade alle vier Varianten
    data = []
    for variant in VARIANTS:
        path = os.path.join(RESULTS_DIR, "test", train_variante, model_name, variant, f"{model_name}_{variant}_{train_variante}_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["variant"] = variant
        data.append(df)

    if not data:
        print(f"⚠️ Keine Ergebnisse für Modell: {model_name}")
        return

    out = os.path.join(OUTPUT_DIR, train_variante)
    os.makedirs(out, exist_ok=True)

    df_all = pd.concat(data)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Metrikenvergleich
    df_melted = df_all.melt(id_vars="variant", value_vars=metrics, var_name="Metric", value_name="Value")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melted, x="variant", y="Value", hue="Metric")
    plt.title(f"{model_name} - {train_variante} – Performance across Variants")
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.join(out, 'overview', model_name), exist_ok=True)
    plt.savefig(os.path.join(out, 'overview', model_name, f"overview_{model_name}_{train_variante}.png"))
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
        plt.title(f"{model_name} {train_variante} – Robustness compared to Standard")
        plt.tight_layout()
        os.makedirs(os.path.join(out, 'robustness'), exist_ok=True)
        plt.savefig(os.path.join(out, 'robustness', f"robustness_{model_name}_{train_variante}.png"))
        plt.close()

    """Vergleicht die Accuracy-, Recall- und F1-Verluste des Modells pro Variante im Vergleich zur Standardvariante."""
    variants = ["standard", "jpeg", "noisy", "scaled"]
    metrics = ["Accuracy", "Recall", "F1-Score"]
    results = {var: {} for var in variants}

    # CSVs einlesen
    for var in variants:
        path = os.path.join(RESULTS_DIR, "test", train_variante, model_name, var, f"{model_name}_{var}_{train_variante}_results.csv")
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
    plt.title(f"Robustheitsprofil für {model_name} {train_variante}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.join(out, 'robustness_profile', model_name), exist_ok=True)
    path = os.path.join(out, 'robustness_profile', model_name, f"robustness_profile'_{model_name}_{train_variante}.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Robustheitsprofil gespeichert: {path}")


def plot_all_models(train_variante: str):
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(f"_{train_variante}_results.csv") and "standard" in f]
    rows = []
    for file in files:
        df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        if df.empty:
            continue
        model = file.replace(f"_standard_{train_variante}_results.csv", "")
        df["model"] = model
        rows.append(df)

    if not rows:
        print("⚠️ Keine Standard-Ergebnisse für Vergleich gefunden.")
        return

    out = os.path.join(OUTPUT_DIR, train_variante)
    os.makedirs(out, exist_ok=True)

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
    plt.savefig(os.path.join(out, f"{train_variante}_comparison_all_models.png"))
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
        plt.savefig(os.path.join(out, f"{train_variante}_size_vs_accuracy.png"))
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=merged, x="Params", y="Accuracy", hue="model")
        plt.title("Parameter Count vs Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{train_variante}_params_vs_accuracy.png"))
        plt.close()

def plot_metrics_comparison_between_models(csv_dir, variant, output_path):
    # === CSV-Dateien finden ===
    csv_files = glob(os.path.join(csv_dir, "*", variant, "*_results.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien gefunden unter: {csv_dir}")

    # === Daten sammeln ===
    models = []
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for file in csv_files:
        df = pd.read_csv(file)
        if df.empty:
            continue
        model_name = os.path.basename(file).split("_")[0]
        models.append(model_name)
        accuracies.append(df["Accuracy"].iloc[0])
        precisions.append(df["Precision"].iloc[0])
        recalls.append(df["Recall"].iloc[0])
        f1_scores.append(df["F1-Score"].iloc[0])

    # === Plot vorbereiten ===
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
    ax.bar(x - 0.5 * width, precisions, width, label="Precision")
    ax.bar(x + 0.5 * width, recalls, width, label="Recall")
    ax.bar(x + 1.5 * width, f1_scores, width, label="F1-Score")

    # === Achsen und Beschriftung ===
    ax.set_title(f"Modellvergleich ({variant})")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    plt.tight_layout()

    # === Speichern und Anzeigen ===
    plt.savefig(output_path)
    plt.show()
    print(f"✅ Vergleichsplot gespeichert unter: {output_path}")

def plot_train_metrics():
    # === Konfiguration ===
    LOG_DIR = "../logs/train"
    OUTPUT_DIR = "../plots/train_comparison"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Alle Logdateien laden ===
    log_files = glob(os.path.join(LOG_DIR, "*", "*.csv"))

    all_dfs = []
    for file in log_files:
        if not file.endswith("search.csv"):
            try:
                df = pd.read_csv(file)
                df["Epoche"] = df["Epoche"].astype(int)
                df["Modell"] = os.path.basename(file).replace(".csv", "")
                df["Variante"] = os.path.basename(os.path.dirname(file))
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file}: {e}")

    if not all_dfs:
        print("⚠️ Keine Trainingslogs gefunden.")
        exit()

    # === Alle Daten zusammenfassen ===
    df_all = pd.concat(all_dfs, ignore_index=True)

    # === Plotten ===
    metrics = ["Loss", "Train-Acc", "Val-Acc"]
    varianten = df_all["Variante"].unique()

    for metric in metrics:
        for variante in varianten:
            df_subset = df_all[df_all["Variante"] == variante]
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=df_subset, x="Epoche", y=metric, hue="Modell", marker="o")
            plt.title(f"{metric} über Epochen – {variante}")
            max_epoch = df_subset["Epoche"].max()
            plt.xticks(range(0, max_epoch + 1, 2))
            plt.xlabel("Epoche")
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            filename = f"{variante}-{metric.lower().replace('-', '_')}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.close()

    print(f"✅ Plots gespeichert in {OUTPUT_DIR}")

def plot_augmentation_deltas():
    # Modelle und Trainingsvarianten
    MODEL_NAMES = ["xception41", "efficientnet_b4", "mobilenet_v2", "vit_base_patch16_224",
                   "swin_tiny_patch4_window7_224"]
    TRAIN_VARIANTS = ["celebdf_only", "celebdf_ff", "celebdf_train_ff_test"]
    AUGMENT_VARIANTS = ["jpeg", "noisy", "scaled"]
    METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]

    BASE_RESULTS_DIR = "../results/test"
    OUTPUT_DIR = "../plots/delta_by_metric"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model in MODEL_NAMES:
        for train_variant in TRAIN_VARIANTS:
            try:
                # Lade Standard-CSV
                std_path = os.path.join(BASE_RESULTS_DIR, train_variant, model, "standard",
                                        f"{model}_standard_{train_variant}_results.csv")
                if not os.path.exists(std_path):
                    print(f"[WARN] Keine Standard-CSV für {model} – {train_variant}")
                    continue
                df_std = pd.read_csv(std_path)
                base_metrics = df_std.loc[0, METRICS]

                delta_data = {metric: [] for metric in METRICS}
                delta_data["Augmentierung"] = []

                for aug in AUGMENT_VARIANTS:
                    path = os.path.join(BASE_RESULTS_DIR, train_variant, model, aug,
                                        f"{model}_{aug}_{train_variant}_results.csv")
                    if not os.path.exists(path):
                        print(f"[WARN] Keine CSV für {model} – {train_variant} – {aug}")
                        continue

                    df_aug = pd.read_csv(path)
                    for metric in METRICS:
                        delta = df_aug.loc[0, metric] - base_metrics[metric]
                        delta_data[metric].append(delta)
                    delta_data["Augmentierung"].append(aug)

                if not delta_data["Augmentierung"]:
                    continue

                df_delta = pd.DataFrame(delta_data)
                df_melted = df_delta.melt(id_vars="Augmentierung", var_name="Metrik", value_name="Δ")

                # Berechne dynamisch die Grenzen für die y-Achse
                max_delta = df_melted["Δ"].abs().max()
                ylim = round(max_delta * 1.2, 2)  # etwas Puffer

                plt.figure(figsize=(8, 5))
                sns.barplot(data=df_melted, x="Metrik", y="Δ", hue="Augmentierung")
                plt.axhline(0, color="black", linestyle="--")
                plt.title(f"{model} – {train_variant}: Δ durch Augmentierung")
                plt.ylim(-ylim, ylim)
                plt.tight_layout()

                out_dir = os.path.join(OUTPUT_DIR, model)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{model}_{train_variant}_delta_plot.png")
                plt.savefig(out_path)
                plt.close()
                print(f"✅ Gespeichert: {out_path}")

            except Exception as e:
                print(f"[ERROR] {model} – {train_variant}: {e}")

#plot_metrics_comparison_between_models("../results/test/celebdf_ff" , "standard", "../plots/celebdf_ff/standard_metrics_comparison.png")

#plot_train_metrics()

#plot_augmentation_deltas()