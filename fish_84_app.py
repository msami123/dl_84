import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Load the trained model
MODEL_PATH = "/Users/anody/Downloads/fish_0.84model_eyes.pth"  # Update this path to the correct model location
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Define class names
CLASS_NAMES = ["Fresh", "Highly Fresh", "Not Fresh"]

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app title and description
st.title("Fish Freshness Classification")
st.write("Upload an image or use your camera to classify the fish's freshness.")

# Select input method
option = st.selectbox("Choose input method:", ["Upload an image", "Use camera"])

if option == "Upload an image":
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100

        # Display prediction
        st.write(f"### Prediction: **{CLASS_NAMES[predicted_class]}**")
        st.write(f"### Confidence: **{confidence:.2f}%**")

        # Display probability distribution
        st.write("### Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASS_NAMES[i]}: {prob * 100:.2f}%")

elif option == "Use camera":
    # Camera input
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        # Display captured image
        st.image(camera_image, caption="Captured Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the captured image
        image = Image.open(camera_image).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100

        # Display prediction
        st.write(f"### Prediction: **{CLASS_NAMES[predicted_class]}**")
        st.write(f"### Confidence: **{confidence:.2f}%**")

        # Display probability distribution
        st.write("### Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASS_NAMES[i]}: {prob * 100:.2f}%")
