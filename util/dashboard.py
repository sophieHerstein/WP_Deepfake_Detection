import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
import seaborn as sns
import numpy as np
import plotly.express as px

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
    path = f"results/test/{test_set}/{model}/{variation}/{model}_{variation}_{test_set}_results.csv"
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
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
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

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Übersicht", "📈 Training", "🧪 Testmetriken", "🧪 Robustheit",
    "🖼️ Grad-CAM", "🧮 Testvergleich", "📉 Trainingsvergleich"
])
# === Übersicht ===
with tab1:
    st.header("📋 Modellübersicht")
    training_set = st.selectbox("Trainingsdaten", train_sets, key="trainingsdaten_tab1") if train_sets else None
    if model and training_set:
        row = summary_df[(summary_df["Modell"] == model) & (summary_df["Variante"] == training_set)]
        res = resources_df[(resources_df["Modell"] == model)]
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
    training_set = st.selectbox("Trainingsdaten", train_sets, key="trainingsdaten_tab2") if train_sets else None
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
with tab5:
    st.header("🖼️ Grad-CAM Bilder")
    if model:
        test_set = st.selectbox("Testsett", ["celebdf_only", "celebdf_ff", "celebdf_train_ff_test"], key="gradcam_testset")
        variante = st.selectbox("Variante", ["standard", "jpeg", "noisy", "scaled"], key="gradcam_variante")
        show_gradcam_gallery(model, test_set, variante)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")

# === Eigene Analyse ===
with tab6:
    st.header("🧮 Vergleich der Testergebnisse")
    csv_files = sorted(glob("results/test/*/*/*/*.csv"))
    selected_files = st.multiselect("Wähle Test-CSV-Dateien für den Vergleich", csv_files)

    if selected_files:
        dfs = []
        for file in selected_files:
            try:
                df = pd.read_csv(file)
                df["Quelle"] = os.path.basename(file).replace("_results.csv", "")
                dfs.append(df)
            except Exception as e:
                st.warning(f"Fehler beim Laden von {file}: {e}")

        if dfs:
            df_merged = pd.concat(dfs, ignore_index=True)
            st.dataframe(df_merged)

            group_by = st.selectbox("Gruppieren nach", ["Modell", "Variante-Training", "Variante-Test", "Quelle"])
            metrics = st.multiselect("Metriken", ["Accuracy", "Precision", "Recall", "F1-Score"],
                                     default=["Accuracy"])
            chart_type = st.radio("Diagrammtyp", ["Bar", "Line", "Scatter"])

            if st.button("Plot anzeigen"):

                for metric in metrics:
                    if chart_type == "Bar":
                        fig = px.bar(df_merged, x=group_by, y=metric, color="Modell", barmode="group",
                                     title=f"{metric} nach {group_by}")
                    elif chart_type == "Line":
                        fig = px.line(df_merged, x=group_by, y=metric, color="Modell", markers=True,
                                      title=f"{metric} nach {group_by}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df_merged, x=group_by, y=metric, color="Modell",
                                         title=f"{metric} nach {group_by}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine gültigen Daten geladen.")
    else:
        st.info("Bitte wähle mindestens eine CSV-Datei aus.")

# === Trainingsvergleich ===
with tab7:
    st.header("📉 Vergleich von Trainingsverläufen")

    train_logs = sorted(glob("logs/train/*/*.csv"))
    selected_logs = st.multiselect("Trainings-Logs auswählen", train_logs)

    if selected_logs:
        all_dfs = []
        for path in selected_logs:
            try:
                df = pd.read_csv(path)
                df["Epoche"] = df["Epoche"].astype(int)
                df["Quelle"] = os.path.basename(path).replace(".csv", "")
                all_dfs.append(df)
            except Exception as e:
                st.warning(f"Fehler beim Laden von {path}: {e}")

        if all_dfs:
            df_train = pd.concat(all_dfs, ignore_index=True)

            metric = st.selectbox("Metrik", ["Train-Acc", "Val-Acc", "Loss"])
            fig = px.line(df_train, x="Epoche", y=metric, color="Quelle",
                          title=f"{metric} über Epochen", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine Daten geladen.")
    else:
        st.info("Bitte wähle mindestens eine Datei.")