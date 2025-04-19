import os
import pandas as pd

def get_report():
    result_csv = "results/test_results.csv"
    gradcam_dir = "gradcam"
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    def generate_html(model_name, metrics_std, metrics_robust, image_paths):
        delta_acc = float(metrics_std['Accuracy']) - float(metrics_robust['Accuracy'])
        delta_f1 = float(metrics_std['F1-Score']) - float(metrics_robust['F1-Score'])
        delta_prec = float(metrics_std['Precision']) - float(metrics_robust['Precision'])
        delta_rec = float(metrics_std['Recall']) - float(metrics_robust['Recall'])

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
                <tr><td>Precision</td><td>{metrics_std['Precision']}</td><td>{metrics_robust['Precision']}</td><td>{delta_prec:.4f}</td></tr>
                <tr><td>Recall</td><td>{metrics_std['Recall']}</td><td>{metrics_robust['Recall']}</td><td>{delta_rec:.4f}</td></tr>
                <tr><td>F1-Score</td><td>{metrics_std['F1-Score']}</td><td>{metrics_robust['F1-Score']}</td><td>{delta_f1:.4f}</td></tr>
                <tr><td>Zeit/Bild (s)</td><td>{metrics_std['Zeit/Bild (s)']}</td><td>{metrics_robust['Zeit/Bild (s)']}</td><td>-</td></tr>
            </table>

            <h2>Grad-CAM Beispiele</h2>
        """

        if image_paths:
            html += "<div class='gradcam-container'>"
            for path in image_paths:
                rel_path = os.path.relpath(path, report_dir).replace("\\", "/")
                html += f"<img src='{rel_path}' alt='Grad-CAM'>"
            html += "</div>"
        else:
            html += "<p><i>Keine Grad-CAM-Bilder verfügbar.</i></p>"

        html += "</body></html>"
        return html

    if not os.path.isfile(result_csv):
        print(f"⚠️ Report wurde übersprungen – Datei '{result_csv}' existiert noch nicht.")
        return

    try:
        df = pd.read_csv(result_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(result_csv, encoding="ISO-8859-1")

    for model in df['Modell'].unique():
        rows = df[df['Modell'] == model]
        if len(rows) != 2:
            continue

        if "standard" not in rows["Variante"].values or "robust" not in rows["Variante"].values:
            print(f"⏳ Report für '{model}' übersprungen – beide Varianten (standard/robust) noch nicht vorhanden.")
            continue

        metrics_std = rows[rows['Variante'] == 'standard'].iloc[0]
        metrics_robust = rows[rows['Variante'] == 'robust'].iloc[0]

        model_dir = os.path.join(gradcam_dir, model)
        images = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".jpg")]) if os.path.isdir(model_dir) else []

        html = generate_html(model, metrics_std, metrics_robust, images)
        with open(os.path.join(report_dir, f"{model}.html"), "w", encoding="utf-8") as f:
            f.write(html)

    print("✅ HTML-Reports mit Δ-Werten und Bild-Fallback wurden erfolgreich erstellt.")

get_report()
