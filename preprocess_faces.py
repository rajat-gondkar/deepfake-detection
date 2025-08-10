import os
import torch
from PIL import Image
import shutil
from facenet_pytorch import MTCNN
from tqdm import tqdm
import argparse

def detect_and_crop_face(image_path, mtcnn, output_path, min_confidence=0.9):
    """
    Detect and crop the largest/most confident face from an image
    Returns True if face was successfully extracted and saved
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect faces
        boxes, probs = mtcnn.detect(image)
        
        if boxes is not None and len(boxes) > 0:
            # Filter faces by confidence
            valid_faces = [(box, prob) for box, prob in zip(boxes, probs) if prob > min_confidence]
            
            if valid_faces:
                # Take the face with highest confidence
                best_box, best_prob = max(valid_faces, key=lambda x: x[1])
                
                # Convert to integers and ensure valid coordinates
                x1, y1, x2, y2 = [max(0, int(coord)) for coord in best_box]
                
                # Add some padding around the face (10% on each side)
                width, height = image.size
                padding_x = int((x2 - x1) * 0.1)
                padding_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                # Crop face
                face = image.crop((x1, y1, x2, y2))
                
                # Save cropped face
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                face.save(output_path, quality=95)
                return True
                
        return False
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def preprocess_dataset(input_dir, output_dir, min_confidence=0.9):
    """
    Preprocess entire dataset to extract faces
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Dataset splits
    splits = ['train', 'val', 'test']
    classes = ['deepfake', 'real']
    
    stats = {
        'total_processed': 0,
        'faces_extracted': 0,
        'no_face_detected': 0,
        'processing_errors': 0
    }
    
    failed_files = []
    
    for split in splits:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        
        if not os.path.exists(split_input_dir):
            print(f"Warning: {split_input_dir} does not exist, skipping...")
            continue
            
        print(f"\nProcessing {split} split...")
        
        for class_name in classes:
            class_input_dir = os.path.join(split_input_dir, class_name)
            class_output_dir = os.path.join(split_output_dir, class_name)
            
            if not os.path.exists(class_input_dir):
                print(f"Warning: {class_input_dir} does not exist, skipping...")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing {len(image_files)} {class_name} images...")
            
            # Process each image
            for image_file in tqdm(image_files, desc=f"{split}/{class_name}"):
                input_path = os.path.join(class_input_dir, image_file)
                output_path = os.path.join(class_output_dir, image_file)
                
                stats['total_processed'] += 1
                
                # Extract face
                success = detect_and_crop_face(input_path, mtcnn, output_path, min_confidence)
                
                if success:
                    stats['faces_extracted'] += 1
                else:
                    stats['no_face_detected'] += 1
                    failed_files.append(input_path)
    
    # Print statistics
    print(f"\n{'='*50}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Faces successfully extracted: {stats['faces_extracted']}")
    print(f"No face detected: {stats['no_face_detected']}")
    print(f"Processing errors: {stats['processing_errors']}")
    print(f"Success rate: {stats['faces_extracted']/stats['total_processed']*100:.1f}%")
    
    # Save failed files list
    if failed_files:
        failed_log_path = os.path.join(output_dir, 'failed_face_detection.txt')
        with open(failed_log_path, 'w') as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"\nFailed files logged to: {failed_log_path}")
    
    print(f"\nFace-cropped dataset saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess deepfake dataset to extract faces')
    parser.add_argument('--input_dir', type=str, default='./dataset', 
                       help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, default='./dataset_faces', 
                       help='Output directory for face-cropped dataset')
    parser.add_argument('--min_confidence', type=float, default=0.9, 
                       help='Minimum confidence threshold for face detection')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist!")
        return
    
    print("Starting face preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum confidence: {args.min_confidence}")
    
    preprocess_dataset(args.input_dir, args.output_dir, args.min_confidence)
    print("\nFace preprocessing completed!")

if __name__ == "__main__":
    main()
