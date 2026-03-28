import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚑 Ambulance Detection AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #ff4b4b;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 20px;
}
.card {
    padding: 25px;
    border-radius: 20px;
    background: white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
}
.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚑 Ambulance Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Emergency Vehicle Detection using Deep Learning</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2967/2967350.png", width=120)
    st.title("Project Dashboard")
    st.markdown("""
    ### 📌 Model Details
    - **Model:** ResNet18
    - **Framework:** PyTorch
    - **Task:** Binary Classification
    
    ### 🧠 Classes
    - 🚑 Ambulance
    - ❌ Non-Ambulance
    """)

    st.markdown("---")
    st.info("Upload an image and let AI detect emergency vehicles instantly.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load("ambulance_model_v2.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🚀 Detection", "📊 About Model", "💬 AI Insights"])

# ---------------- TAB 1: DETECTION ----------------
with tab1:
    st.subheader("Upload Image for Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("### 🔍 AI Prediction")

            with st.spinner("Analyzing Image..."):
                time.sleep(1)
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)

            label_map = {0: "Non-Ambulance", 1: "Ambulance"}
            result = label_map[predicted.item()]
            conf = confidence.item() * 100

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if result == "Ambulance":
                st.success(f"🚑 Ambulance Detected")
            else:
                st.error(f"❌ Not an Ambulance")

            st.write(f"### Confidence: {conf:.2f}%")
            st.progress(int(conf))

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 📈 Probability Distribution")
            st.bar_chart({
                "Ambulance": float(probs[0][1]),
                "Non-Ambulance": float(probs[0][0])
            })

# ---------------- TAB 2: ABOUT MODEL ----------------
with tab2:
    st.subheader("About This AI Model")

    st.markdown("""
    This project uses a **ResNet18 Deep Learning model** trained to classify images into:
    - Ambulance
    - Non-Ambulance

    ### ⚙️ Key Features:
    - Real-time image classification
    - High accuracy with CNN
    - Lightweight deployment using Streamlit

    ### 🧠 Use Cases:
    - Smart Traffic Systems
    - Emergency Vehicle Detection
    - AI Surveillance Systems
    """)

# ---------------- TAB 3: AI CHAT INSIGHTS ----------------
# ---------------- TAB 3: AI CHAT INSIGHTS ----------------
with tab3:
    st.subheader("💬 AI Assistant (Project Explanation)")

    user_query = st.text_input("Ask something about this project:")

    if user_query:
        query = user_query.lower()

        if "model" in query:
            st.write("This project uses ResNet18, a Convolutional Neural Network (CNN) architecture designed for image classification tasks.")

        elif "accuracy" in query:
            st.write("The accuracy depends on training data quality. ResNet18 generally provides good performance for image classification problems.")

        elif "dataset" in query:
            st.write("The model is trained on a dataset containing ambulance and non-ambulance images for binary classification.")

        elif "how it works" in query or "working" in query:
            st.write("The system processes an image, converts it into a tensor, and passes it through a trained ResNet18 model to classify it.")

        elif "use" in query or "application" in query:
            st.write("This system can be used in smart traffic systems, emergency vehicle detection, and AI surveillance.")

        elif "technology" in query:
            st.write("Technologies used include PyTorch for deep learning and Streamlit for web deployment.")

        elif "resnet" in query:
            st.write("ResNet18 is a deep learning model that uses skip connections to avoid vanishing gradient problems.")

        else:
            st.write("❗ Sorry, I don't have an answer for that. Try asking about model, accuracy, dataset, or usage.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<div class="footer">🚀 Built with Streamlit & PyTorch | AI Project Deployment</div>', unsafe_allow_html=True)
