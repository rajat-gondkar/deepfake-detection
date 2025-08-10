#!/bin/bash

echo "Face Preprocessing for Deepfake Detection Dataset"
echo "================================================"

# Check if original dataset exists
if [ ! -d "./dataset" ]; then
    echo "Error: ./dataset directory not found!"
    echo "Please make sure your original dataset is in the ./dataset folder"
    exit 1
fi

echo "Original dataset structure:"
find ./dataset -type d -maxdepth 2

echo ""
echo "Starting face preprocessing..."
echo "This will create a new dataset with cropped faces in ./dataset_faces/"

# Run the preprocessing script
python preprocess_faces.py --input_dir ./dataset --output_dir ./dataset_faces --min_confidence 0.9

if [ $? -eq 0 ]; then
    echo ""
    echo "Face preprocessing completed successfully!"
    echo ""
    echo "New face-cropped dataset structure:"
    find ./dataset_faces -type d -maxdepth 2
    
    echo ""
    echo "To train with the face-cropped dataset, update your train.py:"
    echo "Change: data_dir = './dataset'"
    echo "To:     data_dir = './dataset_faces'"
    
else
    echo "Error: Face preprocessing failed!"
    exit 1
fi
