import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
  .stApp, section[data-testid="stAppViewContainer"], .main { background-color: #050810 !important; background-image: linear-gradient(rgba(0,200,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,200,255,0.04) 1px, transparent 1px); background-size: 40px 40px; font-family: 'Syne', sans-serif; }
  html, body, p, span, div, label, h1, h2, h3 { color: #e8eaf0 !important; font-family: 'Syne', sans-serif; }
  [data-testid="stMarkdownContainer"] * { color: #e8eaf0 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 4rem !important; max-width: 1200px !important; }
  .hero-title { font-size: 4rem; font-weight: 800; letter-spacing: -2px; line-height: 1; background: linear-gradient(135deg, #00c8ff 0%, #b06fff 60%, #ff4d8d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }
  .hero-sub { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #4a5568 !important; letter-spacing: 3px; text-transform: uppercase; margin-top: 0.6rem; }
  .hero-desc { color: #718096 !important; font-size: 1rem; max-width: 480px; line-height: 1.7; margin-top: 1rem; }
  .neon-divider { height: 1px; background: linear-gradient(90deg, transparent, #00c8ff44, #b06fff44, transparent); margin: 2rem 0; }
  [data-testid="stFileUploader"] { background: rgba(0,200,255,0.03) !important; border: 1px dashed rgba(0,200,255,0.25) !important; border-radius: 16px !important; padding: 1rem !important; }
  [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p { color: #718096 !important; }
  .metric-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 1.4rem 1.8rem; text-align: center; }
  .metric-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: #4a5568 !important; margin-bottom: 0.5rem; }
  .metric-value { font-size: 2rem; font-weight: 800; line-height: 1; }
  .metric-danger { color: #ff4d8d !important; } .metric-safe { color: #00e5a0 !important; } .metric-neutral { color: #00c8ff !important; }
  .stButton > button { background: linear-gradient(135deg, #00c8ff, #b06fff) !important; color: #050810 !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.9rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; border: none !important; border-radius: 12px !important; padding: 0.9rem 2rem !important; width: 100% !important; }
  [data-testid="stDownloadButton"] > button { background: rgba(0,200,255,0.1) !important; color: #00c8ff !important; border: 1px solid rgba(0,200,255,0.3) !important; border-radius: 12px !important; font-family: 'Space Mono', monospace !important; width: 100% !important; }
  [data-testid="stImage"] img { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
  [data-testid="stImage"] p { color: #4a5568 !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; text-align: center !important; }
  .section-label { font-family: 'Space Mono', monospace; font-size: 0.72rem; letter-spacing: 3px; text-transform: uppercase; color: #00c8ff !important; margin-bottom: 0.8rem; }
  .disclaimer { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #2d3748 !important; text-align: center; margin-top: 3rem; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")

    # Fix batch_shape compatibility
    with h5py.File(model_path, "r+") as f:
        model_config = f.attrs.get("model_config")
        if model_config is not None:
            config_str = model_config.decode("utf-8") if isinstance(model_config, bytes) else str(model_config)
            if "batch_shape" in config_str:
                config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
                f.attrs["model_config"] = config_str.encode("utf-8")

    # Handle Lambda layer and function_type issues
    custom_objects = {
        "tf": tf,
    }
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception:
        # Fallback: load weights only by rebuilding the architecture
        model = build_unet()
        model.load_weights(model_path)
    return model


def build_unet():
    """Rebuild U-Net architecture matching the trained model."""
    inputs = tf.keras.layers.Input((256, 256, 3))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation="sigmoid")(c9)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def predict(img_pil):
    model = load_model()
    img_resized = img_pil.resize((256, 256)).convert("RGB")
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
    pred = model.predict(arr, verbose=0)
    return np.squeeze(pred), img_resized


def build_result_figure(original_pil, mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#050810")
    for ax, (data, title, cmap) in zip(axes, [
        (np.array(original_pil), "Original MRI", None),
        (mask, "Tumor Mask", "magma"),
        (np.array(original_pil), "Overlay", None),
    ]):
        ax.set_facecolor("#050810")
        ax.imshow(data, cmap=cmap)
        if title == "Overlay":
            ax.imshow(mask, alpha=0.55, cmap="magma")
        ax.set_title(title, color="#718096", fontsize=11, pad=10)
        ax.axis("off")
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# Hero
col_hero, col_badge = st.columns([3, 1])
with col_hero:
    st.markdown('<p class="hero-title">Brain<br>Tumor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Brain Tumor Segmentation · U-Net Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-desc">Upload an MRI brain scan and the model will automatically detect and segment tumor regions using a trained U-Net architecture.</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.spinner("Loading model..."):
        try:
            load_model()
            st.markdown('<div class="metric-card" style="margin-top:1rem"><div class="metric-label">Model Status</div><div class="metric-value metric-safe" style="font-size:1.1rem">READY</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="metric-card" style="margin-top:1rem"><div class="metric-label">Model Status</div><div class="metric-value metric-danger" style="font-size:1.1rem">ERROR</div></div>', unsafe_allow_html=True)
            st.error(f"Model error: {e}")

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-label">// Upload Scan</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop or browse", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, use_column_width=True, caption="Uploaded MRI")
        w, h = img_pil.size
        st.markdown(f'<div class="metric-card" style="margin-top:1rem;text-align:left;padding:1rem 1.4rem"><span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#4a5568;letter-spacing:2px">FILE: {uploaded.name}<br>SIZE: {w}x{h} px | {round(uploaded.size/1024,1)} KB</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("RUN SEGMENTATION")
    else:
        st.markdown('<div style="height:300px;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(255,255,255,0.06);border-radius:16px;color:#2d3748;font-family:Space Mono,monospace;font-size:0.8rem;letter-spacing:2px">NO IMAGE LOADED</div>', unsafe_allow_html=True)
        run = False

with right:
    st.markdown('<p class="section-label">// Analysis Results</p>', unsafe_allow_html=True)
    if uploaded and run:
        with st.spinner("Scanning neural tissue..."):
            try:
                mask, img_resized = predict(img_pil)
                binary = (mask > 0.5).astype(np.uint8)
                tumor_pct = round(float(binary.sum()) / binary.size * 100, 2)
                tumor_detected = binary.sum() > 0
                result_bytes = build_result_figure(img_resized, mask).read()

                c1, c2, c3 = st.columns(3)
                with c1:
                    cls = "metric-danger" if tumor_detected else "metric-safe"
                    val = "POSITIVE" if tumor_detected else "CLEAR"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Detection</div><div class="metric-value {cls}" style="font-size:1.3rem">{val}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Tumor Area</div><div class="metric-value metric-neutral">{tumor_pct}%</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value metric-neutral">{round(float(mask.max())*100,1)}%</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.image(result_bytes, use_column_width=True, caption="Original | Mask | Overlay")
                st.download_button("DOWNLOAD RESULT", data=result_bytes, file_name="brain_tumor_result.png", mime="image/png")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown('<div style="height:340px;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(255,255,255,0.06);border-radius:16px;color:#2d3748;font-family:Space Mono,monospace;font-size:0.8rem;letter-spacing:2px">AWAITING INPUT</div>', unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="disclaimer">FOR RESEARCH & EDUCATIONAL PURPOSES ONLY - NOT A MEDICAL DEVICE</p>', unsafe_allow_html=True)


