import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import urllib.request

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Lensify", layout="wide")

# -------------------------------
# LOAD LABELS
# -------------------------------
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        classes = [line.decode("utf-8").strip() for line in f.readlines()]
    return classes

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# -------------------------------
# PREDICT
# -------------------------------
def predict(image, model, classes):
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_prob, top3_catid = torch.topk(probs, 3)

    results = []
    for i in range(3):
        results.append({
            "label": classes[top3_catid[i]],
            "confidence": float(top3_prob[i])
        })

    return results

# -------------------------------
# CSS (FINAL POSITION FIX)
# -------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* HEADER */
.header {
    position: relative;
    text-align: center;
    margin-top: 30px;
}

/* LOGO LEFT */
.logo-left {
    position: absolute;
    left: -90px;
    top: 10px;
}

/* TITLE CENTER */
.title-center {
    font-size: 150px;
    font-weight: 800;
    margin: 0;
    line-height: 1;

    background: linear-gradient(270deg, #6366f1, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* UPLOAD BOX CENTERED */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139, 92, 246, 0.5);
    padding: 60px;
    border-radius: 18px;
    background: rgba(255,255,255,0.02);
    transition: all 0.3s ease;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

/* HOVER */
[data-testid="stFileUploader"]:hover {
    border-color: #8b5cf6;
    box-shadow: 0 0 35px rgba(139, 92, 246, 0.5);
}

/* REMOVE GRAY BAR */
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
}

/* CENTER BUTTON */
[data-testid="stFileUploader"] button {
    margin: 10px auto;
    display: block;
}

/* FEATURE CARDS */
.feature-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER (ACTUAL CENTER FIX)
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;800&display=swap');

/* TRUE CENTER */
.brand-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 40px;
}

/* LOGO + TITLE */
.brand {
    display: flex;
    align-items: center;
    gap: 20px;
}

/* TITLE */
.brand-title {
    font-family: 'Sora', sans-serif;
    font-size: 90px;   /* 🔥 BIGGER */
    font-weight: 800;
    margin: 0;
    line-height: 1;

    background: linear-gradient(270deg, #6366f1, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# TRUE CENTER BLOCK (NO OUTER COLUMNS)
col = st.columns([1,3,1])[1]

with col:
    c1, c2 = st.columns([1,4])

    with c1:
       st.image("logo.jpg", width=100)

    with c2:
        st.markdown('<h1 class="brand-title">LENSIFY</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown(
    '<p style="text-align:center; color:#94a3b8; margin-top:10px;">Turn images into insights — instantly</p>',
    unsafe_allow_html=True
)
# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image\n\nClick or drag & drop your image here",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# PROCESS
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    model = load_model()
    classes = load_labels()

    results = predict(image, model, classes)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🔎 Prediction")
        st.success(f"{results[0]['label']} ({results[0]['confidence']*100:.2f}%)")

        for res in results:
            st.write(f"{res['label']} — {res['confidence']*100:.2f}%")
            st.progress(res['confidence'])

# -------------------------------
# FEATURES
# -------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## 🚀 Features")

cols = st.columns(3)

features = [
    ("⚡ Fast Predictions", "Instant results using optimized ResNet18."),
    ("🧠 AI Recognition", "Trained on ImageNet with 1000+ classes."),
    ("📊 Confidence Scores", "See probability for each prediction.")
]

for col, (title, desc) in zip(cols, features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
