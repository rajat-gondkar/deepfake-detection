#!/usr/bin/env python3
"""
Inference script for deepfake detection
Load trained model and make predictions on single images or directories
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import time
import numpy as np

from model import get_model


def get_inference_transform():
    """Get image preprocessing transforms for inference"""
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_trained_model(model_path, device, num_classes=2):
    """Load trained model from checkpoint"""
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_single_image(model, image_path, transform, device, class_names=None):
    """Make prediction on a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # Get class name
        if class_names:
            class_name = class_names[predicted_class]
        else:
            class_name = "Real" if predicted_class == 1 else "Deepfake"
            
        return {
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0]
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def predict_directory(model, directory_path, transform, device, class_names=None):
    """Make predictions on all images in a directory"""
    results = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory_path, filename))
    
    if not image_files:
        print("No image files found in the directory")
        return results
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    for image_path in image_files:
        result = predict_single_image(model, image_path, transform, device, class_names)
        if result:
            result['image_path'] = image_path
            result['filename'] = os.path.basename(image_path)
            results.append(result)
            print(f"{result['filename']}: {result['class_name']} (confidence: {result['confidence']:.4f})")
    
    return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--directory_path', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--output_file', type=str,
                        help='Path to save results (optional)')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.directory_path:
        print("Error: Either --image_path or --directory_path must be specified")
        return
    
    if args.image_path and args.directory_path:
        print("Error: Specify either --image_path or --directory_path, not both")
        return
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path, device)
    print("Model loaded successfully!")
    
    # Get preprocessing transform
    transform = get_inference_transform()
    
    # Class names
    class_names = ['Deepfake', 'Real']  # Assuming this order based on typical dataset structure
    
    # Make predictions
    start_time = time.time()
    
    if args.image_path:
        # Single image prediction
        print(f"\nPredicting on single image: {args.image_path}")
        result = predict_single_image(model, args.image_path, transform, device, class_names)
        
        if result:
            print(f"\nPrediction Results:")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: Deepfake={result['probabilities'][0]:.4f}, Real={result['probabilities'][1]:.4f}")
            
            # Save results if output file specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(f"Image: {args.image_path}\n")
                    f.write(f"Prediction: {result['class_name']}\n")
                    f.write(f"Confidence: {result['confidence']:.4f}\n")
                    f.write(f"Probabilities: Deepfake={result['probabilities'][0]:.4f}, Real={result['probabilities'][1]:.4f}\n")
                print(f"Results saved to {args.output_file}")
    
    elif args.directory_path:
        # Directory prediction
        print(f"\nPredicting on directory: {args.directory_path}")
        results = predict_directory(model, args.directory_path, transform, device, class_names)
        
        if results:
            # Summary statistics
            total_images = len(results)
            deepfake_count = sum(1 for r in results if r['class_name'] == 'Deepfake')
            real_count = total_images - deepfake_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\nSummary:")
            print(f"Total images processed: {total_images}")
            print(f"Predicted as Deepfake: {deepfake_count} ({deepfake_count/total_images*100:.1f}%)")
            print(f"Predicted as Real: {real_count} ({real_count/total_images*100:.1f}%)")
            print(f"Average confidence: {avg_confidence:.4f}")
            
            # Save results if output file specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write("Filename,Prediction,Confidence,Deepfake_Prob,Real_Prob\n")
                    for result in results:
                        f.write(f"{result['filename']},{result['class_name']},{result['confidence']:.4f},")
                        f.write(f"{result['probabilities'][0]:.4f},{result['probabilities'][1]:.4f}\n")
                print(f"Results saved to {args.output_file}")
    
    inference_time = time.time() - start_time
    print(f"\nInference completed in {inference_time:.2f} seconds")


if __name__ == '__main__':
    main()
