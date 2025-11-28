import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------- STREAMLIT PAGE SETTINGS ---------------------- #
st.set_page_config(
    page_title="🐾 Animal Sound Detection",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
    <h1 style="text-align:center;">🐾 Animal Sound Detection System 🎧</h1>
    <p style="text-align:center;">Detect animal types from audio using Google's YAMNet model.</p>
    <hr>
""", unsafe_allow_html=True)


# ---------------------- LOAD YAMNET MODEL ---------------------- #
@st.cache_resource
def load_yamnet():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    return model

yamnet_model = load_yamnet()

# Load YAMNet label map
CLASS_MAP = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)
CLASS_NAMES = [line.strip().split(",")[2] for line in open(CLASS_MAP).readlines()[1:]]


# ---------------------- IMPROVED PREDICT FUNCTION ---------------------- #
def predict_sound(file_path):
    wav, sr = librosa.load(file_path, sr=16000, mono=True)

    scores, embeddings, spectrogram = yamnet_model(wav)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    top5_idx = np.argsort(mean_scores)[::-1][:5]
    top5_labels = [CLASS_NAMES[i].lower() for i in top5_idx]
    top5_scores = [mean_scores[i] * 100 for i in top5_idx]

    animal_keywords = {
        "Dog": ["dog", "bark", "barking", "yip", "howl", "woof", "growl"],
        "Cat": ["cat", "meow", "meowing", "purr"],
        "Cow": ["cow", "moo"],
        "Bird": ["bird", "chirp", "tweet", "singing", "crow", "caw"],
        "Duck": ["duck", "quack"],
        "Horse": ["horse", "neigh", "whinny"],
        "Sheep": ["sheep", "bleat"],
        "Rooster": ["rooster", "crow"],
        "Lion": ["lion", "roar"],
        "Monkey": ["monkey", "chimp", "howl"]
    }

    for lbl, sc in zip(top5_labels, top5_scores):
        for animal, keys in animal_keywords.items():
            if any(k in lbl for k in keys):
                return animal, sc, top5_labels, top5_scores

    if top5_scores[0] >= 10:
        return top5_labels[0].title(), top5_scores[0], top5_labels, top5_scores

    return "Unknown", top5_scores[0], top5_labels, top5_scores


# ---------------------- DEBUGGING FUNCTION ---------------------- #
def debug_visualization(wav):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    t = np.arange(len(wav)) / 16000
    ax1.plot(t, wav)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")

    S = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = ax2.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
    ax2.set_title("Mel Spectrogram")
    plt.colorbar(img, ax=ax2)

    st.pyplot(fig)


# ---------------------- SIDEBAR HISTORY ---------------------- #
st.sidebar.header("📜 Detection History")

if "history" not in st.session_state:
    st.session_state["history"] = []

for h in st.session_state["history"]:
    st.sidebar.write(f"🐾 {h['animal']} — {h['confidence']:.1f}% at {h['time']}")

st.sidebar.markdown("---")
st.sidebar.write("🧠 Model: Google YAMNet")


# ---------------------- TABS ---------------------- #
tab1, tab2 = st.tabs(["🎵 Upload Sound", "🎙 Record Sound"])


# ---------------------- UPLOAD TAB ---------------------- #
with tab1:
    uploaded = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

    if uploaded:
        with open("input.wav", "wb") as f:
            f.write(uploaded.read())

        st.audio("input.wav")

        with st.spinner("Analyzing sound..."):
            animal, conf, labels, scores = predict_sound("input.wav")

        st.success(f"Detected: {animal} ({conf:.2f}%)")

        st.write("### Raw Top-5 Predictions:")
        for lbl, sc in zip(labels, scores):
            st.write(f"- {lbl} — {sc:.2f}%")

        # ---------------------- BAR GRAPH ADDED HERE ---------------------- #
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels, scores)
        ax.set_title("Top-5 Prediction Confidence")
        ax.set_ylabel("Confidence (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        # ------------------------------------------------------------------- #

        st.session_state["history"].append({
            "animal": animal,
            "confidence": conf,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        wav, _ = librosa.load("input.wav", sr=16000)
        debug_visualization(wav)


# ---------------------- RECORD TAB ---------------------- #
with tab2:
    st.write("Use your browser mic to record audio.")
    audio_input = st.audio_input("Click to record...", sample_rate=16000)

    if audio_input:
        with open("recorded.wav", "wb") as f:
            f.write(audio_input.read())

        st.audio("recorded.wav")

        with st.spinner("Analyzing recorded audio..."):
            animal, conf, labels, scores = predict_sound("recorded.wav")

        st.success(f"Detected: {animal} ({conf:.2f}%)")

        st.write("### Raw Top-5 Predictions:")
        for lbl, sc in zip(labels, scores):
            st.write(f"- {lbl} — {sc:.2f}%")

        # ---------------------- BAR GRAPH ADDED HERE ---------------------- #
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels, scores)
        ax.set_title("Top-5 Prediction Confidence")
        ax.set_ylabel("Confidence (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        # ------------------------------------------------------------------- #

        st.session_state["history"].append({
            "animal": animal,
            "confidence": conf,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        wav, _ = librosa.load("recorded.wav", sr=16000)
        debug_visualization(wav)


st.markdown("<hr><p style='text-align:center;'>© Animal Sound Detection App</p>", unsafe_allow_html=True)