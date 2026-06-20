import streamlit as st
import tempfile, os, sys, librosa, librosa.display, matplotlib.pyplot as plt, numpy as np

# ✅ Ensure src folder is discoverable by Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict  # Import predict() from src/predict.py

# ---------------- Streamlit App UI ----------------

st.set_page_config(page_title="🎧 Music Genre & Mood Classifier", page_icon="🎵", layout="centered")
st.title("🎧AI Music Genre & Mood Classifier")

st.markdown("""
Upload a music file below (🎵 `.wav`, `.mp3`, or `.m4a`)  
and let the AI predict its **Genre** and **Mood** instantly!
""")

uploaded_file = st.file_uploader("📤 Upload an audio file", type=['wav', 'mp3', 'm4a'])

if uploaded_file:
    # 🔹 Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tfile.write(uploaded_file.read())
    tfile.flush()

    # 🎵 Play the uploaded file
    st.audio(tfile.name)

    # 🧠 Prediction
    with st.spinner("Extracting features & predicting... ⏳"):
        result = predict(tfile.name)  # result = {'genre': ..., 'mood': ..., 'proba': ...}

    st.success("✅ Prediction Complete!")

    # 🎯 Display results
    st.markdown(f"""
    ### 🎼 **Predicted Genre:** `{result['genre']}`
    ### 💫 **Predicted Mood:** `{result['mood']}`
    """)
    if 'proba' in result:
        st.progress(float(result['proba']))
        st.caption(f"Confidence: {result['proba']*100:.2f}%")

    # 🎶 Visualizations
    y, sr = librosa.load(tfile.name, sr=22050)

    # Waveform plot
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1DB954')
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)

    # Spectrogram plot
    D = np.abs(librosa.stft(y))
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax2, cmap='viridis')
    ax2.set(title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    st.pyplot(fig2)

    # Cleanup temp file
    os.unlink(tfile.name)
else:
    st.info("👆 Upload an audio file to get started.")
