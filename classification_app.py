import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("best_classification_model.keras")

IMG_SIZE = 224
class_names = ['Cardiomegaly', 'Pneumonia', 'Sick', 'Healthy', 'Tuberculosis']

def preprocess(image):
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def predict(image):
    if image is None:
        return {}, "Please upload an image."

    img = preprocess(image)
    preds = model.predict(img, verbose=0)[0]

    probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    predicted = class_names[np.argmax(preds)]
    conf = float(np.max(preds))

    return probs, f"Prediction: {predicted}\nConfidence: {conf*100:.2f}%"

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Chest X-Ray Classification")
    img = gr.Image(type="pil", label="Upload X-Ray")
    out_prob = gr.Label(label="Probabilities")
    out_text = gr.Textbox(label="Result")
    btn = gr.Button("Analyze")
    btn.click(predict, inputs=img, outputs=[out_prob, out_text])

demo.launch()
