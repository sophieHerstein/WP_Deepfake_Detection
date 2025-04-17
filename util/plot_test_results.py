import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_test_results():
    # Einstellungen
    result_file = "results/test_results.csv"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.isfile(result_file):
        print("❌ Fehler: Testergebnisse wurden noch nicht generiert. Bitte zuerst 'evaluate.py' ausführen.")
        exit()

    # Daten laden
    df = pd.read_csv(result_file)

    # Balkendiagramm Accuracy
    plt.figure(figsize=(10, 6))
    for variant in ["standard", "robust"]:
        sub = df[df["Variante"] == variant]
        plt.bar([f"{m}\n({variant})" for m in sub["Modell"]], sub["Accuracy"], label=variant)

    plt.title("Accuracy-Vergleich: Standard vs. JPEG-komprimiert")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "accuracy_comparison.png"))
    plt.close()

    # Balkendiagramm F1-Score
    plt.figure(figsize=(10, 6))
    for variant in ["standard", "robust"]:
        sub = df[df["Variante"] == variant]
        plt.bar([f"{m}\n({variant})" for m in sub["Modell"]], sub["F1-Score"], label=variant)

    plt.title("F1-Score-Vergleich: Standard vs. JPEG-komprimiert")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "f1score_comparison.png"))
    plt.close()

    # Laufzeitvergleich
    plt.figure(figsize=(10, 6))
    for variant in ["standard", "robust"]:
        sub = df[df["Variante"] == variant]
        plt.bar([f"{m}\n({variant})" for m in sub["Modell"]], sub["Zeit/Bild (s)"], label=variant)

    plt.title("⏱️ Laufzeitvergleich pro Bild: Standard vs. JPEG-komprimiert")
    plt.ylabel("Zeit pro Bild (Sekunden)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "runtime_comparison.png"))
    plt.close()

    # Delta Accuracy
    pivot = df.pivot(index="Modell", columns="Variante", values="Accuracy")
    if "standard" in pivot.columns and "robust" in pivot.columns:
        pivot["Delta_Accuracy"] = pivot["standard"] - pivot["robust"]
        plt.figure(figsize=(10, 6))
        pivot["Delta_Accuracy"].plot(kind="bar", color="orange")
        plt.title("🔁 Δ Accuracy (Standard – Robust)")
        plt.ylabel("Differenz in Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "delta_accuracy.png"))
        plt.close()

    print("✅ Diagramme wurden erzeugt (sofern Daten vorhanden sind).")

    # Nur Standardwerte verwenden
    df = pd.read_csv(result_file)
    df_std = df[df["Variante"] == "standard"].copy()

    # Konvertiere Werte
    df_std["Modellgröße (MB)"] = df_std["Modellgröße (MB)"].astype(float)
    df_std["Parameter (Mio)"] = df_std["Parameter"].astype(int) / 1e6

    # Plot: Modellgröße
    plt.figure(figsize=(10, 6))
    plt.bar(df_std["Modell"], df_std["Modellgröße (MB)"])
    plt.title("📦 Modellgröße (MB) pro Architektur")
    plt.ylabel("Größe in MB")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "model_size.png"))
    plt.close()

    # Plot: Parameteranzahl
    plt.figure(figsize=(10, 6))
    plt.bar(df_std["Modell"], df_std["Parameter (Mio)"])
    plt.title("🔢 Anzahl Parameter (in Mio) pro Architektur")
    plt.ylabel("Parameteranzahl (Millionen)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "parameter_count.png"))
    plt.close()

    # Plot: Accuracy vs. Modellgröße
    plt.figure(figsize=(8, 6))
    plt.scatter(df_std["Modellgröße (MB)"], df_std["Accuracy"])
    for i, row in df_std.iterrows():
        plt.text(row["Modellgröße (MB)"], row["Accuracy"], row["Modell"], fontsize=9, ha='right')
    plt.title("🎯 Accuracy vs. Modellgröße")
    plt.xlabel("Modellgröße (MB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "accuracy_vs_size.png"))
    plt.close()

    # Plot: Accuracy vs. Parameteranzahl
    plt.figure(figsize=(8, 6))
    plt.scatter(df_std["Parameter (Mio)"], df_std["Accuracy"])
    for i, row in df_std.iterrows():
        plt.text(row["Parameter (Mio)"], row["Accuracy"], row["Modell"], fontsize=9, ha='right')
    plt.title("🎯 Accuracy vs. Parameteranzahl")
    plt.xlabel("Parameteranzahl (Mio)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "accuracy_vs_params.png"))
    plt.close()

    print("✅ Modell-Komplexitätsdiagramme wurden unter 'plots/' gespeichert.")
