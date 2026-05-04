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
from datetime import datetime

# Production optimizations 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Page config
st.set_page_config(
    page_title="Brain Tumor AI Detector", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
/* نفس الـ CSS بتاعك - ممتاز */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
.stApp { background: linear-gradient(135deg, #050810 0%, #0a0f20 50%, #050810 100%); }
.hero-title { font-size: 4.5rem; font-weight: 800; background: linear-gradient(135deg, #00c8ff 0%, #b06fff 50%, #ff4d8d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,200,255,0.2); border-radius: 20px; padding: 1.5rem 2rem; }
.stButton > button { background: linear-gradient(135deg, #00c8ff, #b06fff); color: #000; font-family: 'Space Mono', monospace; font-weight: 700; border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# Model loader Fixed: better error handling
@st.cache_resource
def load_model():
    model_path = "brain_tumor_model.h5"
    if not os.path.exists(model_path):
        st.error(" **Model file missing!** Upload `brain_tumor_model.h5` to repo")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # Quick warmup
        dummy = np.ones((1, 256, 256, 3))
        _ = model.predict(dummy, verbose=0)
        return model
    except Exception as e:
        st.error(f" **Model error:** {str(e)[:100]}...")
        st.stop()

# Load model
try:
    model = load_model()
    st.success(" **AI Model LIVE**")
except:
    st.error(" **Model failed to load**")
    st.stop()

# UI Layout
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="hero-title">Brain Tumor<br>AI Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Medical AI · U-Net · Production Ready</p>', unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.markdown("###  **Upload MRI**")
    uploaded_file = st.file_uploader("Choose MRI...", type=['png','jpg','jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded", use_column_width=True)
        
        col_a, col_b = st.columns(2)
        col_a.metric("Width", image.width)
        col_b.metric("Height", image.height)
        
        if st.button(" **ANALYZE TUMOR**", type="primary", use_container_width=True):
            with st.spinner(" Scanning..."):
                # Preprocess
                img_resized = image.resize((256, 256), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                pred = model.predict(img_array, verbose=0)[0]
                
                # Results
                tumor_mask = (pred > 0.5).astype(np.uint8)
                tumor_area = np.sum(tumor_mask) / tumor_mask.size * 100
                confidence = float(np.max(pred)) * 100
                
                st.session_state.results = {
                    'mask': pred,
                    'image': img_resized,
                    'tumor_area': tumor_area,
                    'confidence': confidence,
                    'has_tumor': tumor_area > 0.1
                }
                st.success(" Analysis complete!")

with right:
    st.markdown("###  **Results**")
    if 'results' in st.session_state:
        r = st.session_state.results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            status = " DETECTED" if r['has_tumor'] else " CLEAR"
            color = "#ff4d8d" if r['has_tumor'] else "#00ff88"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#aaa;letter-spacing:1px">STATUS</div>
                <div style="font-size:2rem;font-weight:800;color:{color}">{status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#aaa">TUMOR AREA</div>
                <div style="font-size:2rem;font-weight:800;color:#00c8ff">{r['tumor_area']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#aaa">CONFIDENCE</div>
                <div style="font-size:2rem;font-weight:800;color:#b06fff">{r['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("###  **Visualization**")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#0a0f20')
        
        axes[0].imshow(r['image'])
        axes[0].set_title('Original', color='white', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(r['mask'], cmap='hot')
        axes[1].set_title('Tumor Heatmap', color='white', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(r['image'])
        axes[2].imshow(r['mask'], cmap='hot', alpha=0.6)
        axes[2].set_title('Overlay', color='white', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0a0f20', bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            " Download Report", 
            buf.getvalue(),
            f"tumor_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            "image/png"
        )
    else:
        st.info(" **Upload MRI & click ANALYZE**")

st.markdown("---")
st.markdown("###  **Disclaimer**: Research only - not medical advice")
