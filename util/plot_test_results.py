import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Todo: ggf. anpassen und überarbeiten
def plot_test_results(result_dir="results", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for file in os.listdir(result_dir):
        if file.endswith(".csv") and "_results" in file:
            path = os.path.join(result_dir, file)
            df = pd.read_csv(path)
            model_name = file.replace("_results.csv", "")
            df["Modell"] = model_name.split("_")[0]
            df["Variante"] = model_name.split("_")[1]
            all_data.append(df)

    if not all_data:
        print("❌ Keine Ergebnisse gefunden.")
        return

    df_all = pd.concat(all_data)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modell", y=metric, hue="Variante", data=df_all)
        plt.title(f"{metric} pro Modell und Variante")
        plt.ylabel(metric)
        plt.xlabel("Modell")
        plt.legend(title="Variante")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_comparison.png"))
        plt.close()

    # Optional: Laufzeit
    if "Avg-Time/Bild (s)" in df_all.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modell", y="Avg-Time/Bild (s)", hue="Variante", data=df_all)
        plt.title("Durchschnittliche Laufzeit pro Bild")
        plt.ylabel("Sekunden")
        plt.xlabel("Modell")
        plt.legend(title="Variante")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
        plt.close()

    # Optional: Konfusionsmatrix-Komponenten (TP, TN, FP, FN)
    for comp in ["TP", "TN", "FP", "FN"]:
        if comp in df_all.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Modell", y=comp, hue="Variante", data=df_all)
            plt.title(f"{comp} pro Modell und Variante")
            plt.ylabel(comp)
            plt.xlabel("Modell")
            plt.legend(title="Variante")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{comp.lower()}_comparison.png"))
            plt.close()

    print(f"✅ Diagramme gespeichert in: {output_dir}")
