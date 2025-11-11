# app.py - Streamlit demo for Chest X-ray Classification
# -------------------------------------------------------
# Local usage:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py

import io, time, json, numpy as np
from typing import List, Tuple, Optional
import streamlit as st
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

st.set_page_config(page_title="Chest X-ray Classifier Demo", layout="centered")
st.title("Chest X-ray Classification - Demo App")
st.markdown("Upload a chest X-ray, load DenseNet121 weights, and view predicted probabilities.\n\n"
            "Disclaimer: Educational demo only; not for medical use.")

with st.sidebar:
    st.header("Settings")
    default_labels = "Cardiomegaly,Pneumonia,Sick,healthy,tuberculosis"
    label_text = st.text_input("Class labels (comma-separated)", value=default_labels)
    class_names = [c.strip() for c in label_text.split(",") if c.strip()]
    num_classes = len(class_names) if class_names else 2
    img_size = st.number_input("Input image size", min_value=64, max_value=1024, value=224, step=32)
    normalize_choice = st.selectbox("Normalization", ["ImageNet (0.485/0.456/0.406, 0.229/0.224/0.225)", "Zero-Mean/Unit-Std (per-image)"])
    saliency_on = st.checkbox("Show saliency map (simple)", value=True)
    use_tta = st.checkbox("Use TTA (horizontal flip)", value=False)
    st.divider()
    st.caption("Model weights")
    weights_file = st.file_uploader("Upload .pth/.pt (optional)", type=["pth","pt"], accept_multiple_files=False)
    weights_url = st.text_input("Weights URL (optional)")
    device_choice = st.selectbox("Device", ["cpu","cuda"] if torch.cuda.is_available() else ["cpu"])
    topk = st.slider("Top-k to display", 1, 10, min(5, num_classes) if num_classes>=5 else num_classes)

@st.cache_resource(show_spinner=False)
def build_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

@st.cache_resource(show_spinner=False)
def load_weights_to_model(model: nn.Module, weights_bytes: Optional[bytes]) -> Tuple[nn.Module, bool, str]:
    if weights_bytes is None:
        return model, False, "No weights uploaded - demo mode with random weights."
    try:
        buffer = io.BytesIO(weights_bytes)
        state = torch.load(buffer, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            elif k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        msg = model.load_state_dict(new_state, strict=False)
        return model, True, f"Weights loaded (strict=False). Missing/unexpected: {msg}"
    except Exception as e:
        return model, False, f"Failed to load weights: {e}"

def make_transforms(img_size: int, mode: str):
    tfms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if mode.startswith("ImageNet"):
        tfms.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(tfms)

def per_image_normalize(t: torch.Tensor) -> torch.Tensor:
    mean = t.mean(dim=(1,2), keepdim=True)
    std = t.std(dim=(1,2), keepdim=True).clamp_min(1e-6)
    return (t - mean) / std

def pil_from_dicom(file) -> Image.Image:
    ds = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    arr = (arr * 255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def predict(model: nn.Module, pil_img: Image.Image, class_names, img_size: int, normalize_choice: str, device: str, tta: bool=False):
    device = torch.device(device)
    model = model.to(device).eval()
    tfm = make_transforms(img_size, normalize_choice)
    img = ensure_rgb(pil_img)
    x = tfm(img)
    if normalize_choice.startswith("Zero-Mean/Unit-Std"):
        x = per_image_normalize(x)
    x = x.unsqueeze(0).to(device)
    start = time.perf_counter()
    if tta:
        x_flip = torch.flip(x, dims=[3])
        with torch.no_grad():
            logits = model(x)
            logits_flip = model(x_flip)
            logits = (logits + logits_flip) / 2.0
    else:
        with torch.no_grad():
            logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    saliency_img = None
    try:
        x.requires_grad_(True)
        logits2 = model(x)
        top_idx = int(torch.argmax(logits2, dim=1).item())
        score = logits2[0, top_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
        grad = x.grad.detach().cpu().abs().squeeze(0)
        grad = grad / (grad.max() + 1e-6)
        x_cpu = x.detach().cpu().squeeze(0)
        if normalize_choice.startswith("ImageNet"):
            mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            vis = (x_cpu * std + mean).clamp(0,1)
        else:
            vis = (x_cpu - x_cpu.min())/(x_cpu.max()-x_cpu.min()+1e-6)
        vis_gray = vis.mean(dim=0, keepdim=True).repeat(3,1,1)
        heat = grad.mean(dim=0, keepdim=True).repeat(3,1,1)
        alpha = 0.6
        blended = (1-alpha)*vis_gray + alpha*heat
        blended = (blended.clamp(0,1)*255).byte().permute(1,2,0).numpy()
        saliency_img = Image.fromarray(blended)
    except Exception:
        saliency_img = None
    return probs, elapsed, saliency_img

uploaded = st.file_uploader("Upload Chest X-ray image", type=["png","jpg","jpeg","dcm"])
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".dcm"):
            if not _HAS_PYDICOM:
                st.error("pydicom not installed. Upload PNG/JPG instead.")
                st.stop()
            pil_img = pil_from_dicom(uploaded)
        else:
            pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    st.image(pil_img, caption="Input image", use_container_width=True)
    model = build_model(num_classes)
    loaded_ok, load_msg = False, "Demo mode: random weights."
    if weights_file is not None:
        model, loaded_ok, load_msg = load_weights_to_model(model, weights_file.read())
    elif weights_url:
        try:
            import requests
            r = requests.get(weights_url, timeout=60); r.raise_for_status()
            model, loaded_ok, load_msg = load_weights_to_model(model, r.content)
        except Exception as e:
            load_msg = f"Failed to fetch weights from URL: {e}"
    with st.expander("Model load status", expanded=False):
        st.write(load_msg)
    probs, elapsed, saliency_img = predict(model, pil_img, class_names, img_size, normalize_choice, device_choice, tta=use_tta)
    st.subheader("Prediction")
    top_indices = np.argsort(probs)[::-1][:topk]
    for rank, idx in enumerate(top_indices, start=1):
        p = float(probs[idx])
        st.write(f"Top {rank}: {class_names[idx]} - {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Inference time: {elapsed*1000:.1f} ms on {device_choice}. " + ("Weights loaded." if loaded_ok else "Demo mode (random weights)."))
    if saliency_on:
        st.subheader("Saliency (simple)")
        if saliency_img is not None:
            st.image(saliency_img, caption="Simple saliency (not a medical heatmap)", use_container_width=True)
        else:
            st.info("Saliency unavailable for this configuration.")
else:
    st.info("Upload a PNG/JPG (or DICOM if pydicom is installed) to begin.")
