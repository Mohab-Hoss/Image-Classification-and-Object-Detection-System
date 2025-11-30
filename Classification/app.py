"""
Chest X-Ray Classification System
Gradio Interface - Hugging Face Compatible
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from datetime import datetime

# ============================================
# Configuration
# ============================================
IMG_SIZE = 224
class_names = ['Cardiomegaly', 'Pneumonia', 'Sick', 'Healthy', 'Tuberculosis']

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# ============================================
# Load Model
# ============================================
print("🤖 Loading model...")
model = tf.keras.models.load_model("best_classification_model.keras")
print("✅ Model loaded successfully!")

# ============================================
# Preprocessing
# ============================================
def preprocess_image(image):
    """Preprocess image for prediction"""
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - imagenet_mean) / imagenet_std
    
    return np.expand_dims(img_normalized, axis=0)

# ============================================
# Prediction Function
# ============================================
def predict_xray(image):
    """Predict chest X-ray classification"""
    if image is None:
        return None, "⚠️ Please upload an image first!"
    
    try:
        start_time = datetime.now()
        
        # Preprocess
        img_processed = preprocess_image(image)
        
        # Predict
        predictions = model.predict(img_processed, verbose=0)[0]
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Create results dictionary
        results = {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }
        
        # Confidence badge
        if confidence >= 0.95:
            conf_badge = "🟢 VERY HIGH CONFIDENCE"
        elif confidence >= 0.85:
            conf_badge = "🟡 HIGH CONFIDENCE"
        elif confidence >= 0.70:
            conf_badge = "🟠 MODERATE CONFIDENCE"
        else:
            conf_badge = "🔴 LOW CONFIDENCE"
        
        # Create report
        report = f"""# 🏥 Analysis Report

## 🎯 Diagnosis: {predicted_class.upper()}

{conf_badge}
Confidence: {confidence*100:.2f}%
Processing Time: {inference_time:.2f}ms

---

## 📊 Probability Breakdown

"""
        
        for cls, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            emoji = "🎯" if cls == predicted_class else "○"
            report += f"{emoji} {cls}: {prob*100:.2f}% {bar}\n\n"
        
        if confidence < 0.75:
            report += """
---

## ⚠️ Attention Required

Low confidence detected. Please:
- Verify image quality
- Consult medical professional
- Consider additional tests
"""
        
        report += f"""
---

⚕️ Disclaimer: Research tool only. Not for clinical use.
🕐 Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return results, report
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease try uploading a different image."
        return {}, error_msg

# ============================================
# Gradio Interface
# ============================================

with gr.Blocks(title="🏥 X-Ray Classifier") as demo:
    
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;">
            <h1 style="color: white; margin: 0;">🏥 Chest X-Ray Classification System</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">AI-Powered Medical Image Analysis | DenseNet121</p>
        </div>
    """)
    
    gr.Markdown("""
    ### 🎯 Upload a chest X-ray image for instant AI analysis
    
    This system can detect:
    - ✅ Cardiomegaly - Enlarged heart
    - ✅ Pneumonia - Lung infection  
    - ✅ Tuberculosis - TB infection
    - ✅ General Pathologies - Other abnormalities
    - ✅ Healthy - Normal findings
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-Ray Image")
            
            input_image = gr.Image(
                type="pil",
                label="Chest X-Ray"
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image",
                variant="primary"
            )
            
            gr.Markdown("""
            ---
            
            #### 📋 Instructions
            
            1. Upload chest X-ray (JPG/PNG)
            2. Click "Analyze Image"
            3. View detailed results
            
            #### 💡 Tips
            
            - Use clear X-ray images
            - PA or AP view preferred
            - Good quality photos
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            
            output_probs = gr.Label(
                label="Confidence Distribution",
                num_top_classes=5
            )
            
            output_report = gr.Markdown("Upload an image to see analysis")
    
    with gr.Accordion("ℹ️ About This System", open=False):
        gr.Markdown("""
        ### 🤖 Model Information
        
        - Architecture: DenseNet121 (Transfer Learning)
        - Framework: TensorFlow/Keras
        - Training: Custom chest X-ray dataset
        - Accuracy: 98.46% on test set
        - Classes: 5 pathologies
        
        ### ⚕️ Medical Disclaimer
        
        IMPORTANT: This system is for research and educational purposes only.
        
        - ❌ Not FDA approved
        - ❌ Not for clinical diagnosis
        - ✅ Results should be verified by qualified medical professionals
        - ✅ Always consult healthcare providers for medical decisions
        """)
    
    gr.Markdown("""
    ---
    
    Made with ❤️ using Gradio & TensorFlow
    """)
    
    # Connect button
    analyze_btn.click(
        fn=predict_xray,
        inputs=input_image,
        outputs=[output_probs, output_report]
    )
    
# Launch
if __name__ == "__main__":
    print("🚀 Launching Gradio interface...")
    demo.launch()