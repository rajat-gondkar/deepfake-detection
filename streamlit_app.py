import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from model import get_model

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_trained_model(model_path, device, num_classes=2):
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image, transform, device, class_names):
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    class_name = class_names[predicted_class]
    return class_name, confidence, probabilities.cpu().numpy()[0]

def main():
    st.title("Deepfake Detection Demo")
    st.write("Upload an image to check if it's Real or Deepfake.")

    model_path = st.sidebar.text_input("Path to model checkpoint", "./checkpoints/best_model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['deepfake', 'real']

    # Load model only once
    @st.cache_resource()
    def load_model():
        return load_trained_model(model_path, device, num_classes=2)

    model = load_model()
    transform = get_inference_transform()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        class_name, confidence, probs = predict_image(model, image, transform, device, class_names)
        st.write(f"**Prediction:** {class_name.upper()} ({confidence*100:.2f}% confidence)")
        st.progress(float(confidence))
        st.write(f"Probabilities: Deepfake={probs[0]:.4f}, Real={probs[1]:.4f}")

if __name__ == "__main__":
    main()
