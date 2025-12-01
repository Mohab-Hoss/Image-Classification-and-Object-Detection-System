# 🩺 Chest X-Ray Object Detection (Faster R-CNN)

This folder contains the chest X-ray object detection system using a fine-tuned Faster R-CNN (ResNet50-FPN) model. The interface is implemented with Gradio for real-time visualization.

## 📂 Files Included
- app.py  
- fasterrcnn_best.pt  
- requirements.txt  

## 🔗 Download Model
Place the trained model here:

fasterrcnn_best.pt  
👉 https://drive.google.com/file/d/1Ll-1E2Fjs5dhw_YTTRzPLI78oB0z5vR9/view?usp=drive_link

## ▶️ Run Locally
pip install -r requirements.txt  
python app.py

## 🧠 Model Output
- Bounding boxes  
- Confidence scores  
- Per-class detection summary  
- Adjustable confidence threshold  

**Detection Classes:**  
Cardiomegaly, Pneumonia, Sick, Healthy, Tuberculosis

## ⚠️ Disclaimer
For academic and research use only. Not intended for medical diagnosis.
