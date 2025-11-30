"""
Chest X-Ray Object Detection System
Gradio Interface - Hugging Face Compatible
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from collections import defaultdict

# ============================================
# Configuration
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5

# Class mapping
IDX2LABEL = {
    1: "Cardiomegaly",
    2: "Pneumonia",
    3: "Sick",
    4: "Healthy",
    5: "Tuberculosis"
}

# Colors (RGB)
CLASS_COLORS = {
    'Cardiomegaly': (255, 69, 58),
    'Pneumonia': (255, 159, 10),
    'Sick': (255, 214, 10),
    'Healthy': (48, 209, 88),
    'Tuberculosis': (191, 90, 242)
}

# ============================================
# Load Model
# ============================================
print("📦 Loading detection model...")

model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

checkpoint = torch.load("fasterrcnn_best.pt", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully!")

# ============================================
# Transform
# ============================================
transform = A.Compose([ToTensorV2()])

# ============================================
# Drawing Function
# ============================================
def draw_boxes(img_rgb, boxes, scores, labels):
    """Draw bounding boxes on image"""
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    for box, score, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        class_name = IDX2LABEL[int(lab)]
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        text = f"{class_name} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.rectangle([x1, y1 - text_h - 8, x1 + text_w + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - text_h - 5), text, fill=(255, 255, 255), font=font)
    
    return np.array(img_pil)

# ============================================
# Report Generation
# ============================================
def generate_report(boxes, scores, labels, inference_time, threshold):
    """Generate detailed detection report"""
    num_detections = len(boxes)
    
    if num_detections == 0:
        return f"""# 🎯 Detection Results

🔴 NO DETECTIONS

No objects detected above {int(threshold*100)}% confidence threshold.

Processing Time: {inference_time:.1f}ms

---

### Possible Reasons:
- Image may be healthy/normal
- Pathologies below confidence threshold
- Try lowering the threshold

💡 Tip: Adjust the confidence slider
"""
    
    avg_conf = float(np.mean(scores))
    max_conf = float(np.max(scores))
    
    badge = "🟢 HIGH CONFIDENCE" if avg_conf >= 0.7 else "🟡 MODERATE CONFIDENCE" if avg_conf >= 0.5 else "🟠 LOW CONFIDENCE"
    
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[IDX2LABEL[int(label)]] += 1
    
    report = f"""# 🎯 Detection Results

{badge}

Detected {num_detections} object(s) | Processing: {inference_time:.1f}ms

---

## 📋 Detected Objects

"""
    
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        class_name = IDX2LABEL[int(label)]
        conf_bar = "█" * int(score * 25)
        report += f"{i}. {class_name} - {score*100:.1f}% {conf_bar}\n\n"
    
    report += f"""
---

## 📊 Summary

| Metric | Value |
|--------|-------|
| Total Detections | {num_detections} |
| Avg Confidence | {avg_conf*100:.1f}% |
| Max Confidence | {max_conf*100:.1f}% |
| Unique Classes | {len(class_counts)} |

### Class Distribution:
"""
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"- {class_name}: {count}\n"
    
    report += f"""

---

⚕️ Disclaimer: For research use only. Not for clinical diagnosis.
🕐 Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

# ============================================
# Detection Function
# ============================================
def detect_objects(image, score_threshold=0.25):
    """Detect objects in chest X-ray image"""
    if image is None:
        return None, "⚠️ Please upload an image first!"
    
    try:
        start_time = datetime.now()
        
        # Convert to RGB
        img_rgb = np.array(image) if isinstance(image, Image.Image) else image
        
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
        
        # Transform
        transformed = transform(image=img_rgb)
        tensor_img = transformed["image"].float().to(DEVICE)
        
        # Predict
        with torch.no_grad():
            predictions = model([tensor_img])[0]
        
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Filter
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        selected = scores >= score_threshold
        boxes = boxes[selected]
        scores = scores[selected]
        labels = labels[selected]
        
        # Draw
        result_img = draw_boxes(img_rgb.copy(), boxes, scores, labels)
        report = generate_report(boxes, scores, labels, inference_time, score_threshold)
        
        return result_img, report
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}\n\nPlease try a different image."

# ============================================
# Gradio Interface
# ============================================

with gr.Blocks(title="🎯 X-Ray Detection") as demo:
    
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;">
            <h1 style="color: white; margin: 0;">🎯 Chest X-Ray Object Detection System</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Faster R-CNN Object Detection | Accurate & Fast</p>
        </div>
    """)
    
    gr.Markdown("""
    ### 🔍 Upload a chest X-ray image to detect and localize pathologies
    
    This system can detect:
    - 🔴 Cardiomegaly - Enlarged heart
    - 🟠 Pneumonia - Lung infection
    - 🟡 Sick - General pathologies
    - 🟢 Healthy - Normal regions
    - 🟣 Tuberculosis - TB infection
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-Ray Image")
            
            input_image = gr.Image(type="pil", label="Input Image")
            
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
            
            gr.Markdown("""
            ---
            
            ### 🎨 Color Legend
            - 🔴 Cardiomegaly
            - 🟠 Pneumonia
            - 🟡 Sick
            - 🟢 Healthy
            - 🟣 Tuberculosis
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            
            output_image = gr.Image(label="Detected Objects")
            output_report = gr.Markdown("Upload an image to start")
    
    with gr.Accordion("ℹ️ About This System", open=False):
        gr.Markdown(f"""
        ### 🤖 Model Details
        
        - Architecture: Faster R-CNN ResNet50-FPN-V2
        - Framework: PyTorch + Torchvision
        - Device: {DEVICE}
        - Classes: {NUM_CLASSES}
        
        ### ⚕️ Medical Disclaimer
        
        IMPORTANT: This system is for research and educational purposes only.
        
        - ❌ Not FDA approved
        - ❌ Not for clinical diagnosis
        - ✅ Results should be verified by qualified medical professionals
        """)
    
    gr.Markdown("---\n\nMade with ❤️ using Gradio & PyTorch")
    
    detect_btn.click(
        fn=detect_objects,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_report]
    )

if __name__ == "__main__":
    print("🚀 Launching detection interface...")
    demo.launch()