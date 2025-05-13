import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image

st.set_page_config(page_title="Deepfake Detection Dashboard", layout="wide")

# === Helper Functions ===

def load_summary():
    try:
        return pd.read_csv("data/summary.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Model", "TrainingSet", "FinalTrainAcc", "FinalValAcc", "FinalLoss", "Epochs", "TotalTime"])

def load_training_log(model, training_set):
    path = f"data/training_logs/{model}_{training_set}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_test_results(model, training_set, test_set, variation):
    path = f"data/test_results/{model}_{training_set}_{test_set}_{variation}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_line_chart(df, x, y, title):
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

def plot_bar_chart(data, title):
    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def show_gradcam_gallery(model, training_set):
    folder = f"images/gradcam/{model}_{training_set}"
    categories = ["TP", "TN", "FP", "FN"]
    cols = st.columns(len(categories))
    for i, cat in enumerate(categories):
        image_paths = sorted(glob(f"{folder}/{cat}_*.jpg"))
        with cols[i]:
            st.markdown(f"**{cat}**")
            for path in image_paths:
                try:
                    img = Image.open(path)
                    st.image(img, caption=os.path.basename(path), use_column_width=True)
                except:
                    pass

# === Sidebar Selection ===

st.sidebar.title("Modellauswahl")
summary_df = load_summary()

models = summary_df["Model"].unique().tolist()
model = st.sidebar.selectbox("Modell", models) if models else None

train_sets = summary_df[summary_df["Model"] == model]["TrainingSet"].unique().tolist() if model else []
training_set = st.sidebar.selectbox("Trainingsdaten", train_sets) if train_sets else None

# === Tabs ===

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Übersicht", "📈 Training", "🧪 Testmetriken", "🧪 Robustheit", "🖼️ Grad-CAM"])

# === Übersicht ===
with tab1:
    st.header("📋 Modellübersicht")
    if model and training_set:
        row = summary_df[(summary_df["Model"] == model) & (summary_df["TrainingSet"] == training_set)]
        st.dataframe(row)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus der Seitenleiste.")

# === Training ===
with tab2:
    st.header("📈 Trainingsverlauf")
    if model and training_set:
        log_df = load_training_log(model, training_set)
        if log_df is not None:
            plot_line_chart(log_df, "epoch", "train_acc", "Train Accuracy")
            plot_line_chart(log_df, "epoch", "val_acc", "Validation Accuracy")
            plot_line_chart(log_df, "epoch", "loss", "Loss")
        else:
            st.warning("Keine Trainingsdaten gefunden.")
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")

# === Testmetriken ===
with tab3:
    st.header("🧪 Testmetriken")
    if model and training_set:
        test_set = st.selectbox("Testset", ["celebdff", "faceforensics", "own"])
        variation = st.selectbox("Datenvariation", ["original", "jpg", "noise", "scale"])
        test_df = load_test_results(model, training_set, test_set, variation)
        if test_df is not None:
            st.dataframe(test_df)
            metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
            if all(col in test_df.columns for col in metric_cols):
                plot_bar_chart(test_df[metric_cols].T, f"{model} – {test_set} ({variation})")
        else:
            st.warning("Keine Testdaten gefunden.")
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")

# === Robustheit ===
with tab4:
    st.header("🧪 Robustheitsvergleich")
    st.write("Hier könnten Δ-Werte zu Originaldaten geplottet werden – z. B. Accuracy-Differenz.")
    st.info("Logik muss noch angepasst werden, sobald Original+veränderte Werte vorliegen.")

# === Grad-CAM ===
with tab5:
    st.header("🖼️ Grad-CAM Bilder")
    if model and training_set:
        show_gradcam_gallery(model, training_set)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")