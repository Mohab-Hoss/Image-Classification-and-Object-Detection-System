# Chest X-ray Classification - Streamlit Demo

Educational demo for a DenseNet121-based chest X-ray classifier.

## Local run
- Create venv, install requirements, run:
    python -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

Open http://localhost:8501

## Deploy online
- Streamlit Community Cloud: push repo, create new app, point to app.py
- Hugging Face Spaces: create Streamlit Space, upload app.py + requirements.txt
- Docker hosts (Render/Railway): use Dockerfile

## Weights
Upload .pth/.pt in the sidebar or paste a direct Weights URL. Ensure class labels match training order.
