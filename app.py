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
import time
import psutil
from datetime import datetime

# Production optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.optimizer.set_jit(False)

# Page config
st.set_page_config(
    page_title=" Brain Tumor AI Detector", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - التصميم الجميل
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
.stApp { 
    background: linear-gradient(135deg, #050810 0%, #0a0f20 50%, #050810 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-title { 
    font-size: 4.5rem; 
    font-weight: 800; 
    background: linear-gradient(135deg, #00c8ff 0%, #b06fff 50%, #ff4d8d 100%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    background-clip: text; 
}
.hero-sub { 
    font-family: 'Space Mono', monospace; 
    font-size: 0.9rem; 
    color: #00c8ff; 
    letter-spacing: 3px; 
    text-transform: uppercase; 
}
.metric-card { 
    background: rgba(255,255,255,0.05); 
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0,200,255,0.2); 
    border-radius: 20px; 
    padding: 1.5rem 2rem; 
    text-align: center; 
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0,200,255,0.5);
    box-shadow: 0 20px 40px rgba(0,200,255,0.1);
}
.stButton > button { 
    background: linear-gradient(135deg, #00c8ff, #b06fff); 
    color: #000; 
    font-family: 'Space Mono', monospace; 
    font-weight: 700; 
    border-radius: 16px; 
    padding: 1rem 2.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title & Status
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="hero-title">Brain Tumor<br>AI Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Medical AI · U-Net Segmentation · Production Ready</p>', unsafe_allow_html=True)
with col2:
    # Model status
    @st.cache_resource(show_spinner=False)
    def load_model():
        model_path = "brain_tumor_model.h5"
        if not os.path.exists(model_path):
            st.error("Model file not found!")
            st.stop()
        
        model = tf.keras.models.load_model(model_path, compile=False)
        # Warmup
        dummy = np.random.random((1, 256, 256, 3))
        _ = model.predict(dummy, verbose=0)
        return model
    
    try:
        model = load_model()
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:#00c8ff;letter-spacing:2px;margin-bottom:0.5rem">AI STATUS</div>
            <div style="font-size:2rem;font-weight:800;color:#00ff88"> LIVE</div>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.7rem;color:#00c8ff;letter-spacing:2px;margin-bottom:0.5rem">AI STATUS</div>
            <div style="font-size:2rem;font-weight:800;color:#ff4d8d"> ERROR</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

st.markdown("---")

# Main interface
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("###  Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose MRI image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Supports PNG, JPG, JPEG up to 50MB"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        
        col1, col2 = st.columns(2)
        col1.metric("Width", image.width)
        col2.metric("Height", image.height)
        
        if st.button(" ANALYZE TUMOR", type="primary", use_container_width=True):
            with st.spinner(" Scanning neural tissue..."):
                # Prediction
                img_resized = image.resize((256, 256)).convert("RGB")
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred = model.predict(img_array, verbose=0)[0]
                
                # Metrics
                tumor_mask = (pred > 0.5).astype(np.uint8)
                tumor_area = np.sum(tumor_mask) / tumor_mask.size * 100
                confidence = np.max(pred) * 100
                
                st.session_state.results = {
                    'mask': pred,
                    'resized': img_resized,
                    'tumor_area': tumor_area,
                    'confidence': confidence,
                    'tumor_detected': tumor_area > 0.1
                }

with right:
    st.markdown("###  Analysis Results")
    if 'results' in st.session_state:
        results = st.session_state.results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = " TUMOR DETECTED" if results['tumor_detected'] else " CLEAR SCAN"
            color = "#ff4d8d" if results['tumor_detected'] else "#00ff88"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.7rem;color:#aaa">DETECTION</div>
                <div style="font-size:1.8rem;font-weight:800;color:{color}">{status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.7rem;color:#aaa">TUMOR AREA</div>
                <div style="font-size:1.8rem;font-weight:800;color:#00c8ff">
                    {results['tumor_area']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.7rem;color:#aaa">CONFIDENCE</div>
                <div style="font-size:1.8rem;font-weight:800;color:#b06fff">
                    {results['confidence']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 🖼️ Results")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#0a0f20')
        
        axes[0].imshow(results['resized'])
        axes[0].set_title('Original MRI', color='white', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(results['mask'], cmap='magma')
        axes[1].set_title('Tumor Probability', color='white', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(results['resized'])
        axes[2].imshow(results['mask'], cmap='magma', alpha=0.6)
        axes[2].set_title('Tumor Overlay', color='white', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0f20')
        st.download_button(
            "💾 Download Results", 
            buf.getvalue(), 
            f"brain_tumor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "image/png"
        )
    else:
        st.info(" Upload an MRI scan and click ANALYZE")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:2rem;color:#666;font-size:0.8rem'>
    <strong> Brain Tumor AI Detector</strong> | 
    For Research & Educational Use Only | 
    Not a Medical Device | 
    Powered by U-Net & TensorFlow
</div>
""", unsafe_allow_html=True)


