# 🧠 Brain Tumor Segmentation

<div align="center">

![Brain Animation](brain_animation.html)

**U-Net Deep Learning Model · Streamlit Web Application**

![Python](https://img.shields.io/badge/Python-3.8--3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-Research%20Only-purple?style=flat-square)

</div>

---

## 📌 Overview

This project implements a **Brain Tumor Segmentation** system using a **U-Net** convolutional neural network. The model automatically detects and segments tumor regions in brain MRI scans, and is served through a clean Streamlit web interface.

> ⚠️ **For Research & Educational Purposes Only — NOT a medical device.**

---

## 📁 Project Structure

```
brain-tumor-segmentation/
├── app.py                    # Streamlit web application
├── brain_tumor_model.h5      # Trained U-Net model weights
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🏗️ Model Architecture — U-Net

```
Input (256×256×3)
      │
      ▼
┌─────────────────────────────────────────────┐
│              ENCODER (Contraction)          │
│  Block 1 → Conv2D(16)  → Dropout → MaxPool │
│  Block 2 → Conv2D(32)  → Dropout → MaxPool │
│  Block 3 → Conv2D(64)  → Dropout → MaxPool │
│  Block 4 → Conv2D(128) → Dropout → MaxPool │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│     BOTTLENECK Conv2D(256)  │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│              DECODER (Expansion)            │
│  Block 6 → UpConv(128) + Skip → Conv2D(128)│
│  Block 7 → UpConv(64)  + Skip → Conv2D(64) │
│  Block 8 → UpConv(32)  + Skip → Conv2D(32) │
│  Block 9 → UpConv(16)  + Skip → Conv2D(16) │
└─────────────────────────────────────────────┘
      │
      ▼
Output Conv2D(1, sigmoid) → Segmentation Mask (256×256×1)
```

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Epochs | 30 |
| Batch Size | 16 |
| Validation Split | 10% |
| Input Size | 256 × 256 × 3 |
| Normalization | ÷ 255 (Lambda layer) |
| Dataset | Brain Tumor Segmentation (Kaggle) |

---

## 🚀 Installation & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

### 2. Create virtual environment (Python 3.11)
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
tensorflow-cpu>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
```

---

## 🖥️ How to Use

1. Launch the app and wait for **● READY** status
2. Upload a brain MRI image (JPG / PNG)
3. Click **RUN SEGMENTATION**
4. View results:
   - ✅ / ⚠️ Detection status (CLEAR / POSITIVE)
   - 📊 Tumor area percentage
   - 🎯 Max confidence score
   - 🖼️ 3-panel visualization: Original · Mask · Overlay
5. Click **DOWNLOAD RESULT** to save the output PNG

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New App**
4. Select repo → branch: `main` → file: `app.py`
5. Click **Deploy** ✅

> Your app will be live at:
> `https://YOUR_USERNAME-brain-tumor-segmentation-app-XXXXX.streamlit.app`

---

## 📊 Output Example

| Metric | Description |
|--------|-------------|
| **Detection** | POSITIVE or CLEAR |
| **Tumor Area** | % of pixels classified as tumor |
| **Confidence** | Max prediction probability |
| **Visualization** | Original · Mask · Overlay figure |

---

## ⚠️ Disclaimer

This application is developed **for research and educational purposes only**.
It is **NOT a medical device** and must **NOT** be used for clinical diagnosis,
medical decision-making, or patient care.
Always consult a qualified medical professional for any health concerns.

---

<div align="center">
Made with ❤️ using TensorFlow & Streamlit
</div>
