# 🩺 Chest X-Ray Classification & Object Detection System

This repository contains two deep-learning applications for analyzing chest X-ray images:

1. **Image Classification** — Predicts one of 5 disease classes.  
2. **Object Detection** — Detects and localizes abnormalities using bounding boxes.

Both applications run through **Gradio web interfaces** for fast, simple, real-time inference.

This project is for **research and academic use only.**

---

# 🚀 Live Demos

### 🔵 Classification Demo  
https://huggingface.co/spaces/Mohab-Hossam/ChestXray-Classifier

### 🟠 Detection Demo  
https://huggingface.co/spaces/Mohab-Hossam/ChestXray-Detector

---

# 📥 Download Trained Models (Google Drive)

### **Classification Model — `best_classification_model.keras`**  
👉 *https://drive.google.com/file/d/1spNYR_BZ1v1yAGI_iqHfNVdwnMGjy88w/view?usp=sharing*

### **Detection Model — `fasterrcnn_best.pt`**  
👉 *https://drive.google.com/file/d/1Ll-1E2Fjs5dhw_YTTRzPLI78oB0z5vR9/view?usp=sharing*

Place each downloaded model **next to its corresponding `app.py` file** exactly as structured below.

---

# 📦 Included Files

### **Classification**
- `app.py`
- `README.md`
- `requirements.txt`

### **Detection**
- `app.py`
- `README.md`
- `requirements.txt`

---

# 🧠 Features

### **Classification**
- Predicts 5 classes: **Cardiomegaly – Pneumonia – Sick – Healthy – Tuberculosis**
- High-accuracy predictions  
- Confidence scores  
- Probability distribution visualization  
- Supports JPG / PNG images  

### **Detection**
- **Faster R-CNN (ResNet50-FPN)**
- Bounding boxes with class labels  
- Adjustable confidence threshold  
- Per-class detection summary  

---

# 🛠 Tech Stack
- **TensorFlow / Keras** (classification)  
- **PyTorch + Torchvision** (detection)  
- **Gradio** (web UI)  
- **MLflow** (tracking)  
- **Albumentations** (preprocessing)

---

# ⚠️ Medical Disclaimer
This tool is **NOT** a medical device.  
It is intended **only for research and educational purposes.**

---

GitHub Repository:  
https://github.com/Mohab-Hoss/Image-Classification-and-Object-Detection-System


