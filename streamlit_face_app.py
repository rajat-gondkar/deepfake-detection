import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN

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

def detect_faces(image):
    """Detect faces using MTCNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Convert PIL to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Detect faces
    boxes, probs = mtcnn.detect(image)
    
    faces = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob > 0.9:  # Confidence threshold
                # Convert box to integers
                box = [int(coord) for coord in box]
                x1, y1, x2, y2 = box
                
                # Extract face region
                face = image.crop((x1, y1, x2, y2))
                faces.append({
                    'face': face,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': prob
                })
    
    return faces

def predict_face(model, face_image, transform, device, class_names):
    """Predict if a face is real or deepfake"""
    face_image = face_image.convert('RGB')
    input_tensor = transform(face_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    class_name = class_names[predicted_class]
    
    return class_name, confidence, probabilities.cpu().numpy()[0]

def draw_face_boxes(image, faces, predictions):
    """Draw bounding boxes and predictions on the image"""
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a better font
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    colors = {'real': 'green', 'deepfake': 'red'}
    
    for i, (face_data, prediction) in enumerate(zip(faces, predictions)):
        x1, y1, x2, y2 = face_data['bbox']
        class_name, confidence, _ = prediction
        
        # Draw bounding box
        color = colors.get(class_name, 'yellow')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"Face {i+1}: {class_name.upper()} ({confidence*100:.1f}%)"
        
        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1-30), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1-30), label, fill='white', font=font)
    
    return image

def main():
    st.title("Multi-Face Deepfake Detection")
    st.write("Upload an image to detect all faces and classify each as Real or Deepfake.")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input("Path to model checkpoint", "./checkpoints/best_model.pth")
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found: {model_path}")
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"Using device: {device}")
    
    class_names = ['deepfake', 'real']
    
    # Load model
    @st.cache_resource()
    def load_model():
        return load_trained_model(model_path, device, num_classes=2)
    
    try:
        model = load_model()
        transform = get_inference_transform()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Detect faces
        with st.spinner("Detecting faces..."):
            faces = detect_faces(image)
        
        if len(faces) == 0:
            st.warning("No faces detected in the image.")
            return
        
        st.success(f"Detected {len(faces)} face(s)")
        
        # Predict each face
        predictions = []
        with st.spinner("Classifying faces..."):
            for face_data in faces:
                prediction = predict_face(model, face_data['face'], transform, device, class_names)
                predictions.append(prediction)
        
        # Display results
        st.subheader("Face Classification Results")
        
        # Create annotated image
        annotated_image = image.copy()
        annotated_image = draw_face_boxes(annotated_image, faces, predictions)
        st.image(annotated_image, caption='Detected Faces with Classifications', use_column_width=True)
        
        # Display individual face results
        st.subheader("Individual Face Results")
        
        # Create columns for face display
        cols = st.columns(min(len(faces), 3))  # Max 3 columns
        
        for i, (face_data, prediction) in enumerate(zip(faces, predictions)):
            col_idx = i % len(cols)
            with cols[col_idx]:
                class_name, confidence, probs = prediction
                
                st.write(f"**Face {i+1}**")
                st.image(face_data['face'], caption=f"Face {i+1}", width=150)
                
                # Color-coded prediction
                if class_name == 'real':
                    st.success(f"✅ REAL ({confidence*100:.1f}%)")
                else:
                    st.error(f"❌ DEEPFAKE ({confidence*100:.1f}%)")
                
                # Progress bar for confidence
                st.progress(float(confidence))
                
                # Detailed probabilities
                with st.expander(f"Details for Face {i+1}"):
                    st.write(f"Detection confidence: {face_data['confidence']:.3f}")
                    st.write(f"Deepfake probability: {probs[0]:.3f}")
                    st.write(f"Real probability: {probs[1]:.3f}")
        
        # Summary statistics
        st.subheader("Summary")
        real_count = sum(1 for pred in predictions if pred[0] == 'real')
        deepfake_count = len(predictions) - real_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Faces", len(faces))
        with col2:
            st.metric("Real Faces", real_count)
        with col3:
            st.metric("Deepfake Faces", deepfake_count)

if __name__ == "__main__":
    main()
