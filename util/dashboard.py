import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Deepfake Detection Dashboard", layout="wide")

# === Helper Functions ===

@st.cache_data
def load_summary():
    try:
        return pd.read_csv("results/train_results.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Modell", "Variante", "Train-Acc", "Val-Acc", "Loss", "Trainzeit (s)"])

@st.cache_data
def load_resources():
    try:
        return pd.read_csv("results/model_resources.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Modell", "Variante", "Size (MB)", "Params"])

def load_training_log(model, training_set):
    path = f"logs/train/{training_set}/{model}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_test_results(model, test_set, variation):
    path = f"results/{model}_{variation}_{test_set}_results.csv"
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
    data.plot(kind="bar", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel("Wert")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_confusionmatrix(df):
    cm = np.array(df)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def show_gradcam_gallery(model, test_set, variante):
    folder = f"gradcam/{test_set}/{model}/{variante}"
    categories = ["TP", "TN", "FP", "FN"]
    cols = st.columns(len(categories))
    for i, cat in enumerate(categories):
        image_paths = sorted(glob(f"{folder}/{cat}_*.jpg"))
        with cols[i]:
            st.markdown(f"**{cat}**")
            for path in image_paths:
                try:
                    img = Image.open(path)
                    st.image(img, caption=os.path.basename(path), use_container_width=True)
                except:
                    pass

def compute_deltas(model, test_set, base="standard"):
    variations = ["jpeg", "noisy", "scaled"]
    base_df = load_test_results(model, test_set, base)
    if base_df is None:
        return None

    deltas = {}
    for var in variations:
        comp_df = load_test_results(model, test_set, var)
        if comp_df is not None:
            try:
                delta = comp_df["Accuracy"].iloc[0] - base_df["Accuracy"].iloc[0]
                deltas[var] = delta
            except:
                pass
    return deltas

# === Load Data ===
summary_df = load_summary()
resources_df = load_resources()

# === Sidebar Selection ===
st.sidebar.title("Modellauswahl")

models = summary_df["Modell"].unique().tolist()
model = st.sidebar.selectbox("Modell", models) if models else None

train_sets = summary_df[summary_df["Modell"] == model]["Variante"].unique().tolist() if model else []
training_set = st.sidebar.selectbox("Trainingsdaten", train_sets) if train_sets else None

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Übersicht", "📈 Training", "🧪 Testmetriken", "🧪 Robustheit", "🖼️ Grad-CAM"])

# === Übersicht ===
with tab1:
    st.header("📋 Modellübersicht")
    if model and training_set:
        row = summary_df[(summary_df["Modell"] == model) & (summary_df["Variante"] == training_set)]
        res = resources_df[(resources_df["Modell"] == model) & (resources_df["Variante"] == training_set)]
        st.subheader("Trainingszusammenfassung")
        st.dataframe(row, hide_index=True, use_container_width=True)
        if not res.empty:
            st.subheader("Modellressourcen")
            st.dataframe(res, hide_index=True, use_container_width=True)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus der Seitenleiste.")

# === Training ===
with tab2:
    st.header("📈 Trainingsverlauf")
    if model and training_set:
        log_df = load_training_log(model, training_set)
        if log_df is not None:
            plot_line_chart(log_df, "Epoche", "Train-Acc", "Train Accuracy")
            plot_line_chart(log_df, "Epoche", "Val-Acc", "Validation Accuracy")
            plot_line_chart(log_df, "Epoche", "Loss", "Loss")
        else:
            st.warning("Keine Trainingsdaten gefunden.")
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")

# === Testmetriken ===
with tab3:
    st.header("🧪 Testmetriken")
    if model:
        test_set = st.selectbox("Testset", ["celebdf_only", "celebdf_ff", "celebdf_train_ff_test"])
        variation = st.selectbox("Datenvariation", ["standard", "jpeg", "noisy", "scaled"])
        test_df = load_test_results(model, test_set, variation)
        if test_df is not None:
            st.dataframe(test_df, hide_index=True)
            metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
            if all(col in test_df.columns for col in metric_cols):
                plot_confusionmatrix([[test_df["TP"].iloc[0], test_df["FP"].iloc[0]],
                                      [test_df["FN"].iloc[0], test_df["TN"].iloc[0]]])
                plot_bar_chart(test_df[metric_cols].T, f"{model} – {test_set} ({variation})")
        else:
            st.warning("Keine Testdaten gefunden.")
    else:
        st.info("Bitte wähle ein Modell aus.")

# === Robustheit ===
with tab4:
    st.header("🧪 Robustheitsvergleich (Δ Accuracy vs. Standard)")
    if model:
        test_set = st.selectbox("Testset für Robustheit", ["celebdf_only", "celebdf_ff", "celebdf_train_ff_test"], key="robust_testset")
        deltas = compute_deltas(model, test_set)
        if deltas:
            delta_df = pd.DataFrame.from_dict(deltas, orient="index", columns=["Δ Accuracy"])
            st.dataframe(delta_df, use_container_width=True)
            plot_bar_chart(delta_df, f"Δ Accuracy zu 'standard' – {model}")
        else:
            st.warning("Nicht genügend Daten für Vergleich gefunden.")
    else:
        st.info("Bitte wähle ein Modell aus.")

# === Grad-CAM ===
# === Grad-CAM ===
with tab5:
    st.header("🖼️ Grad-CAM Bilder")
    if model and training_set:
        variante = st.selectbox("Testset für Grad-CAM", ["standard", "jpeg", "noisy", "scaled"], key="gradcam_testset")
        show_gradcam_gallery(model, training_set, variante)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")