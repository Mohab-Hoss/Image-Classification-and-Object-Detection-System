import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5
IDX2LABEL = {
    1: "Cardiomegaly",
    2: "Pneumonia",
    3: "Sick",
    4: "Healthy",
    5: "Tuberculosis"
}

CLASS_COLORS = {
    'Cardiomegaly': (255, 69, 58),
    'Pneumonia': (255, 159, 10),
    'Sick': (255, 214, 10),
    'Healthy': (48, 209, 88),
    'Tuberculosis': (191, 90, 242)
}

transform = A.Compose([ToTensorV2()])

model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_f = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES + 1)
model.load_state_dict(torch.load("m3_best.pt", map_location=DEVICE))

model.to(DEVICE)
model.eval()

def draw_boxes(img, boxes, scores, labels):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cls = IDX2LABEL[int(label)]
        color = CLASS_COLORS[cls]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1), f"{cls} {score:.2f}", fill=color)

    return np.array(img_pil)

def predict(image, threshold=0.25):
    if image is None:
        return None, "Please upload an image"

    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    transformed = transform(image=img)
    tensor = transformed["image"].float().to(DEVICE)

    with torch.no_grad():
        pred = model([tensor])[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    mask = scores >= threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    result = draw_boxes(img, boxes, scores, labels)

    txt = f"Detections: {len(boxes)}\n"
    for cls, sc in zip(labels, scores):
        txt += f"- {IDX2LABEL[int(cls)]}: {sc*100:.1f}%\n"

    return result, txt

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Chest X-Ray Object Detection")
    img = gr.Image(type="pil")
    slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence")
    out_img = gr.Image(label="Detections")
    out_txt = gr.Textbox(label="Details", lines=8)
    btn = gr.Button("Detect")
    btn.click(predict, inputs=[img, slider], outputs=[out_img, out_txt])

demo.launch()
