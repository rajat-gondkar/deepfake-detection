python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install timm tqdm numpy pandas scikit-learn Pillow matplotlib seaborn
sudo apt install screen -y
pip install streamlit torch torchvision pillow numpy opencv-python facenet-pytorch timm
chmod +x run_face_preprocessing.sh
python3 train.py --data_dir ./dataset --epochs 40
streamlit run streamlit_face_app.py