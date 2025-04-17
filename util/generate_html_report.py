import os
import pandas as pd

# Pfade
result_csv = "results/test_results.csv"
gradcam_dir = "gradcam"
report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)

# HTML-Bausteine
def generate_html(model_name, metrics_std, metrics_robust, image_paths):
    delta_acc = float(metrics_std['Accuracy']) - float(metrics_robust['Accuracy'])

    html = f"""
    <html>
    <head>
        <meta charset='UTF-8'>
        <title>Report – {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; margin-bottom: 20px; }}
            td, th {{ border: 1px solid #ccc; padding: 8px 12px; }}
            .gradcam-container {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .gradcam-container img {{ max-width: 300px; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Modell: {model_name}</h1>

        <h2>Metriken im Vergleich</h2>
        <table>
            <tr><th></th><th>Standard</th><th>JPEG-komprimiert</th><th>Δ</th></tr>
            <tr><td>Accuracy</td><td>{metrics_std['Accuracy']}</td><td>{metrics_robust['Accuracy']}</td><td>{delta_acc:.4f}</td></tr>
            <tr><td>Precision</td><td>{metrics_std['Precision']}</td><td>{metrics_robust['Precision']}</td><td>-</td></tr>
            <tr><td>Recall</td><td>{metrics_std['Recall']}</td><td>{metrics_robust['Recall']}</td><td>-</td></tr>
            <tr><td>F1-Score</td><td>{metrics_std['F1-Score']}</td><td>{metrics_robust['F1-Score']}</td><td>-</td></tr>
            <tr><td>Zeit/Bild (s)</td><td>{metrics_std['Zeit/Bild (s)']}</td><td>{metrics_robust['Zeit/Bild (s)']}</td><td>-</td></tr>
        </table>

        <h2>Grad-CAM Beispiele</h2>
        <div class='gradcam-container'>
    """
    for path in image_paths:
        rel_path = os.path.relpath(path, report_dir).replace("\\", "/")
        html += f"<img src='{rel_path}' alt='Grad-CAM'>"

    html += """
        </div>
    </body>
    </html>
    """
    return html

# CSV einlesen
df = pd.read_csv(result_csv)

# Report generieren
for model in df['Modell'].unique():
    rows = df[df['Modell'] == model]
    if len(rows) != 2:
        continue

    metrics_std = rows[rows['Variante'] == 'standard'].iloc[0]
    metrics_robust = rows[rows['Variante'] == 'robust'].iloc[0]

    model_dir = os.path.join(gradcam_dir, model)
    if not os.path.isdir(model_dir):
        continue
    images = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".jpg")])

    html = generate_html(model, metrics_std, metrics_robust, images)
    with open(os.path.join(report_dir, f"{model}.html"), "w", encoding="utf-8") as f:
        f.write(html)

print("✅ HTML-Reports mit Robustheitsvergleich wurden erfolgreich erstellt.")
