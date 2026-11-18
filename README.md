Image Caption Generator
A Streamlit-based application that generates captions for uploaded images using Deep Learning.
Features
- Upload images in JPG, JPEG, or PNG format.
- Extract image features using a trained feature extractor.
- Generate captions using an LSTM-based model.
- Display the uploaded image and generated caption.
Technologies Used
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pickle
Project Structure
project/
│
├── app.py
└── models/
    ├── model.keras
    ├── feature_extractor.keras
    └── tokenizer.pkl


Installation
pip install streamlit tensorflow numpy pillow


